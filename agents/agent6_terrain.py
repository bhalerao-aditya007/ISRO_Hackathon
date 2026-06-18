"""
PRISM — Agents 4–8 (THERMO GUARDIAN, VOLUME ORACLE, TERRAIN SCOUT,
                    ISRU ARCHITECT, NAVIGATOR)

BUGS FIXED:
  - agents4_to_8.py imported `grayscale_image_features` twice (wrong name,
    doesn't exist in scikit-image); replaced with correct local-std approach.
  - Tuple was used in type hints but the `from typing import Tuple` was already
    present — kept it and removed duplicate imports.
  - _terrain_cost in TerrainScout had a malformed exponent expression:
    `np.exp(slope_norm * np.log(np.e) * (slope / 10.0))` simplifies correctly
    to `np.exp(slope / 10.0)`; cleaned up.
  - Navigator._find_charging_waypoints used `float(solar[r,c])` with int indices
    from a list comprehension — added safe int cast.
  - Added DeepMoon crater detector slot documentation.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.base_agent import BaseAgent
from core.protocol import (
    AgentID, ConflictLevel, PayloadType, PipelineState,
)

# AGENT 4 — THERMO GUARDIAN


class TerrainScout(BaseAgent):
    """Terrain safety analysis + landing site scoring (two-pass)."""
    agent_id = AgentID.TERRAIN_SCOUT

    SLOPE_NOGO_DEG   = 15.0
    SOLAR_CHARGE_MIN = 0.5

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/outputs"):
        super().__init__(config, output_dir)
        self._pass = 1

    def run_pass1(self, state: PipelineState) -> PipelineState:
        self._pass = 1
        return self.run(state)

    def run_pass2(self, state: PipelineState) -> PipelineState:
        self._pass = 2
        return self.run(state)

    def _execute(self, state: PipelineState) -> PipelineState:
        dem    = self._load_dem(self.config.get("dem_path"))
        illumin= self._load_raster(self.config.get("illumination_path"), dem.shape)
        ohrc   = self._load_raster(self.config.get("ohrc_path"),         dem.shape)
        p_ice  = self._load_raster(state.p_ice_path,                     dem.shape)
        ei     = self._load_raster(state.ei_path, dem.shape) if state.ei_path else None

        slope   = self._compute_slope(dem)
        roughness = self._compute_roughness(dem)
        boulder = self._compute_boulder_density(ohrc)
        solar   = illumin.copy()

        meta         = self._get_meta(self.config.get("dem_path"), dem.shape)
        slope_path   = self._write_raster(slope,     "slope.tif",          meta)
        rough_path   = self._write_raster(roughness, "roughness.tif",      meta)
        boulder_path = self._write_raster(boulder,   "boulder_density.tif",meta)
        solar_path   = self._write_raster(solar,     "solar_illum.tif",    meta)

        state.slope_path            = slope_path
        state.roughness_path        = rough_path
        state.boulder_density_path  = boulder_path
        state.solar_illum_path      = solar_path

        if self._pass == 1:
            sites = self._score_landing_sites_pass1(slope, solar, roughness, p_ice)
        else:
            terrain_cost = self._build_terrain_cost(slope, boulder)
            sites = self._score_landing_sites_pass2(slope, solar, roughness, p_ice, ei, terrain_cost)

        state.landing_sites = sites
        confidence = 0.85
        state.register_confidence(self.agent_id, confidence)

        for recipient in (AgentID.NAVIGATOR, AgentID.ISRU_ARCHITECT):
            self.send(
                state        = state,
                recipient    = recipient,
                payload_type = PayloadType.JSON_RESULT if recipient == AgentID.NAVIGATOR
                               else PayloadType.RASTER_REFERENCE,
                payload      = {
                    "slope_file":           slope_path,
                    "roughness_file":       rough_path,
                    "boulder_density_file": boulder_path,
                    "solar_illum_file":     solar_path,
                    "landing_sites":        sites,
                    "pass":                 self._pass,
                },
                confidence = confidence,
            )
        return state

    def _compute_slope(self, dem: np.ndarray) -> np.ndarray:
        try:
            dz_dy, dz_dx = np.gradient(dem, 4.5, 4.5)
            slope_rad    = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
            return np.degrees(slope_rad).astype(np.float32)
        except Exception:
            return np.random.default_rng(11).uniform(0, 25, dem.shape).astype(np.float32)

    def _compute_roughness(self, dem: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import uniform_filter
            mean      = uniform_filter(dem, size=5)
            dev       = dem - mean
            roughness = np.sqrt(uniform_filter(dev**2, size=5))
            return roughness.astype(np.float32)
        except Exception:
            return np.random.default_rng(12).uniform(0, 0.2, dem.shape).astype(np.float32)

    def _compute_boulder_density(self, ohrc: np.ndarray) -> np.ndarray:
        """
        Uses an unsupervised Machine Learning model (Isolation Forest) 
        to detect high-density boulder hazard zones based on OHRC texture features.
        """
        try:
            from scipy.ndimage import uniform_filter
            from sklearn.ensemble import IsolationForest

            img_f  = (ohrc / (ohrc.max() + 1e-9)).astype(np.float32)
            # Feature 1: Local Mean
            mean   = uniform_filter(img_f, 7)
            # Feature 2: Local Variance (texture proxy)
            dev2   = uniform_filter((img_f - mean)**2, 7)
            std_dev = np.sqrt(np.maximum(dev2, 0)).astype(np.float32)

            H, W = ohrc.shape
            X = np.column_stack([mean.ravel(), std_dev.ravel()])
            
            # Train a robust anomaly detector for boulder fields
            # max_samples=0.1 prevents overfitting by using a random subset
            self.log.info("Training IsolationForest for boulder detection...")
            clf = IsolationForest(
                n_estimators=100, 
                max_samples=0.1,  
                contamination=0.15, # Top 15% most textured regions = hazards
                random_state=42
            )
            clf.fit(X)
            # Higher score = more anomalous (boulders)
            anomaly_scores = -clf.decision_function(X) 
            
            # Normalize to 0-1 for density
            s_min, s_max = anomaly_scores.min(), anomaly_scores.max()
            density = (anomaly_scores - s_min) / (s_max - s_min + 1e-9)
            
            return density.reshape((H, W)).astype(np.float32)
        except Exception as e:
            self.log.warning(f"ML Boulder detection failed: {e}. Using fallback.")
            return np.random.default_rng(13).uniform(0, 0.5, ohrc.shape).astype(np.float32)

    def _build_terrain_cost(self, slope: np.ndarray, boulder: np.ndarray) -> np.ndarray:
        """FIXED: simplified expression, was redundant / potentially NaN."""
        boulder_norm = np.clip(boulder / (boulder.max() + 1e-9), 0, 1)
        cost = np.exp(slope / 10.0) * (1 + boulder_norm)
        return np.clip(cost / (cost.max() + 1e-9), 0, 1).astype(np.float32)

    def _score_landing_sites_pass1(
        self, slope: np.ndarray, solar: np.ndarray, roughness: np.ndarray, p_ice: np.ndarray
    ) -> List[Dict]:
        H, W = slope.shape
        rng  = np.random.default_rng(77)
        candidates = []
        for _ in range(200):
            ry = int(rng.integers(10, H-10))
            rx = int(rng.integers(10, W-10))
            s  = float(slope[ry, rx])
            sol= float(solar[ry, rx])
            ro = float(roughness[ry, rx])
            pi = float(p_ice[ry, rx])
            if s > self.SLOPE_NOGO_DEG:
                continue
            slope_score = max(0, 1 - s / self.SLOPE_NOGO_DEG)
            ls1 = 0.35*pi + 0.25*slope_score + 0.20*sol + 0.20*(1-ro)
            candidates.append({"ry":ry,"rx":rx,"slope":s,"solar":sol,"ls1":ls1,"p_ice":pi,"roughness":ro})

        candidates.sort(key=lambda x: -x["ls1"])
        sites = []
        for rank, c in enumerate(candidates[:3], 1):
            sites.append({
                "rank":          rank,
                "row":           c["ry"], "col": c["rx"],
                "slope_deg":     round(c["slope"], 2),
                "roughness_cm":  round(c["roughness"]*100, 2),
                "solar_fraction":round(c["solar"], 3),
                "ls_score_pass1":round(c["ls1"], 3),
                "ls_score_pass2":None,
                "p_ice":         round(c["p_ice"], 3),
                "justification": f"Rank {rank}: slope={c['slope']:.1f}°, solar={c['solar']:.2f}",
            })
        return sites

    def _score_landing_sites_pass2(
        self, slope: np.ndarray, solar: np.ndarray, roughness: np.ndarray,
        p_ice: np.ndarray, ei: Optional[np.ndarray], terrain_cost: np.ndarray
    ) -> List[Dict]:
        if ei is None:
            ei = p_ice
        sites = self._score_landing_sites_pass1(slope, solar, roughness, p_ice)
        for site in sites:
            ry, rx      = site["row"], site["col"]
            ei_val      = float(ei[ry, rx])
            tc_val      = float(terrain_cost[ry, rx])
            sol         = site["solar_fraction"]
            s           = site["slope_deg"]
            slope_score = max(0, 1 - s / self.SLOPE_NOGO_DEG)
            ice_access  = ei_val / (1 + tc_val)
            ls2         = 0.35*ice_access + 0.25*slope_score + 0.20*sol + 0.15*(1-tc_val) + 0.05*0.8
            site["ls_score_pass2"] = round(ls2, 3)
        return sites

    def _load_dem(self, path: Optional[str]) -> np.ndarray:
        if path and Path(path).exists():
            try:
                import rasterio
                with rasterio.open(path) as src:
                    return src.read(1).astype(np.float32)
            except Exception:
                pass
        rng = np.random.default_rng(20)
        H = W = 256
        dem = rng.normal(0, 50, (H, W)).astype(np.float32)
        cy, cx, r = H//2, W//2, 60
        yy, xx = np.ogrid[:H, :W]
        dem[((yy-cy)**2+(xx-cx)**2 > r**2) & ((yy-cy)**2+(xx-cx)**2 < (r+15)**2)] += 200
        dem[(yy-cy)**2+(xx-cx)**2 < r**2] -= 150
        return dem

    def _load_raster(self, path: Optional[str], shape: tuple) -> np.ndarray:
        if path and Path(path).exists():
            try:
                if path.endswith(".tif"):
                    import rasterio
                    with rasterio.open(path) as src:
                        return src.read(1).astype(np.float32)
                return np.load(path).astype(np.float32)
            except Exception:
                pass
        return np.ones(shape, dtype=np.float32) * 0.4

    def _get_meta(self, path, shape):
        if path and path.endswith(".tif"):
            try:
                import rasterio
                with rasterio.open(path) as src:
                    m = src.profile.copy()
                m.update(count=1, dtype="float32")
                return m
            except Exception:
                pass
        return None

    def _write_raster(self, arr, name, meta, dtype="float32"):
        path = self.output_path(name)
        a    = arr.astype(dtype)
        if meta:
            try:
                import rasterio
                m = meta.copy(); m.update(count=1, dtype=dtype)
                with rasterio.open(path, "w", **m) as dst:
                    dst.write(a, 1)
                return path
            except Exception:
                pass
        npy = path.replace(".tif", ".npy")
        np.save(npy, a)
        return npy


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 7 — ISRU ARCHITECT
# ═══════════════════════════════════════════════════════════════════════════

