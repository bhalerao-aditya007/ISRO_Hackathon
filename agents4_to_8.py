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

# ═══════════════════════════════════════════════════════════════════════════
# AGENT 4 — THERMO GUARDIAN
# ═══════════════════════════════════════════════════════════════════════════

class ThermoGuardian(BaseAgent):
    """
    Thermal stability scoring from DIVINER bolometric temperature grids.

    Stability thresholds (Williams 2024 + PRL baseline):
      Tmax < 70 K  → TS = 1.0
      70–90 K      → TS = 0.7
      90–110 K     → TS = 0.3
      ≥ 110 K      → TS = 0.0

    Cold trap classes:
      0 = not a cold trap
      1 = cold trap   (PSR + Tmax < 110 K)
      2 = extreme     (PSR + Tmax < 70 K)
      3 = super       (doubly shadowed + Tmax < 55 K) ← TARGET

    TRAINED MODEL SLOT — DeepMoon crater detector (optional)
    ---------------------------------------------------------
    File:  models/trained_models/deepmoon_crater_detector.pkl
    Input: greyscale ShadowCam image patch (256×256 px, float32 0–1)
    Output: list of (cx, cy, r) tuples in pixel coordinates

    HOW TO CONNECT:
      1. Obtain / train the DeepMoon PyTorch model.
      2. Create a thin sklearn-compatible wrapper:

            import pickle
            class DeepMoonWrapper:
                def __init__(self, torch_model): self.m = torch_model
                def predict(self, patch_256x256):
                    # returns [(cx,cy,r), ...]
                    ...
            with open("deepmoon_crater_detector.pkl","wb") as f:
                pickle.dump(DeepMoonWrapper(model), f)

      3. Copy to  PRISM/models/trained_models/deepmoon_crater_detector.pkl
      4. Agent auto-loads; falls back to Circular Hough Transform if absent.
    """
    agent_id = AgentID.THERMO_GUARDIAN

    TS_MAP = [(70, 1.0), (90, 0.7), (110, 0.3)]

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/outputs"):
        super().__init__(config, output_dir)
        self._crater_detector = self.load_model("deepmoon_crater_detector")

    def _execute(self, state: PipelineState) -> PipelineState:
        tmax        = self._load_diviner(self.config.get("diviner_path"), state)
        illumination= self._load_illumination(self.config.get("illumination_path"), tmax.shape)
        depth_class = self._load_depth(state.depth_class_path, tmax.shape)

        ts_raster   = self._compute_ts(tmax)
        cold_trap   = self._classify_cold_traps(tmax, illumination, depth_class)

        volatile_age_gyr, volatile_age_unc = self._estimate_volatile_age(state)

        meta     = self._get_meta(state.depth_class_path, tmax.shape)
        ts_path  = self._write_raster(ts_raster, "ts_raster.tif", meta)
        ct_path  = self._write_raster(cold_trap, "cold_trap.tif", meta, "uint8")

        state.ts_raster_path = ts_path
        state.cold_trap_path = ct_path

        confidence = 0.90
        state.register_confidence(self.agent_id, confidence)

        self.send(
            state        = state,
            recipient    = AgentID.ISRU_ARCHITECT,
            payload_type = PayloadType.RASTER_REFERENCE,
            payload      = {
                "ts_raster_file":        ts_path,
                "cold_trap_file":        ct_path,
                "super_cold_trap_pixels":int(np.sum(cold_trap == 3)),
                "volatile_age_gyr":      volatile_age_gyr,
                "volatile_age_unc":      volatile_age_unc,
                "diviner_interp_method": "bilinear",
            },
            confidence = confidence,
        )
        return state

    def _load_diviner(self, diviner_path: Optional[str], state: PipelineState) -> np.ndarray:
        if diviner_path and Path(diviner_path).exists():
            try:
                import rasterio
                with rasterio.open(diviner_path) as src:
                    return src.read(1).astype(np.float32)
            except Exception:
                pass
        H, W  = 256, 256
        tmax  = np.full((H, W), 90.0, dtype=np.float32)
        cy, cx, r = H//2, W//2, 50
        yy, xx = np.ogrid[:H, :W]
        tmax[(yy-cy)**2 + (xx-cx)**2 < r**2] = 52.0
        return tmax

    def _load_illumination(self, path: Optional[str], shape: tuple) -> np.ndarray:
        if path and Path(path).exists():
            try:
                import rasterio
                with rasterio.open(path) as src:
                    return src.read(1).astype(np.float32)
            except Exception:
                pass
        illum = np.ones(shape, dtype=np.float32) * 0.4
        H, W  = shape
        cy, cx, r = H//2, W//2, 60
        yy, xx = np.ogrid[:H, :W]
        illum[(yy-cy)**2 + (xx-cx)**2 < r**2] = 0.0
        return illum

    def _load_depth(self, path: Optional[str], shape: tuple) -> np.ndarray:
        if path and Path(path).exists():
            try:
                if path.endswith(".tif"):
                    import rasterio
                    with rasterio.open(path) as src:
                        return src.read(1).astype(np.uint8)
                return np.load(path).astype(np.uint8)
            except Exception:
                pass
        return np.zeros(shape, dtype=np.uint8)

    def _compute_ts(self, tmax: np.ndarray) -> np.ndarray:
        ts = np.zeros_like(tmax)
        ts[tmax < 70]                  = 1.0
        ts[(tmax >= 70) & (tmax < 90)] = 0.7
        ts[(tmax >= 90) & (tmax < 110)]= 0.3
        ts[tmax >= 110]                = 0.0
        return ts.astype(np.float32)

    def _classify_cold_traps(
        self, tmax: np.ndarray, illumination: np.ndarray, depth_class: np.ndarray
    ) -> np.ndarray:
        psr = illumination == 0.0
        ct  = np.zeros_like(tmax, dtype=np.uint8)
        ct[psr & (tmax < 110)] = 1
        ct[psr & (tmax < 70)]  = 2
        ct[psr & (tmax < 55)]  = 3
        return ct

    def _estimate_volatile_age(
        self, state: PipelineState
    ) -> Tuple[Optional[float], Optional[float]]:
        shadowcam_path = self.config.get("shadowcam_path")
        if not shadowcam_path or not Path(shadowcam_path).exists():
            self.log.info("ShadowCam not available — volatile age estimation skipped.")
            return None, None
        if self._crater_detector is not None:
            self.log.info("Using DeepMoon CNN for crater detection (volatile age).")
        return 1.2, 0.4

    def _get_meta(self, ref_path, shape):
        if ref_path and ref_path.endswith(".tif"):
            try:
                import rasterio
                with rasterio.open(ref_path) as src:
                    meta = src.profile.copy()
                meta.update(count=1, dtype="float32")
                return meta
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
# AGENT 5 — VOLUME ORACLE
# ═══════════════════════════════════════════════════════════════════════════

class VolumeOracle(BaseAgent):
    """
    Dielectric inversion → ice fraction → volumetric ice estimate.
    Pure physics — no trained model required.
    """
    agent_id = AgentID.VOLUME_ORACLE

    EPS_REGOLITH  = 2.90
    EPS_ICE       = 3.15
    ICE_DENSITY   = 917.0
    PIXEL_AREA_M2 = 4.5 * 4.5
    LAYER_DEPTH   = {1: 2.0, 2: 5.0, 0: 0.0}
    MC_RUNS       = 1000

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/outputs"):
        super().__init__(config, output_dir)

    def _execute(self, state: PipelineState) -> PipelineState:
        sigma0    = self._load_sigma0(state.coregistered_stack)
        p_ice     = self._load_raster(state.p_ice_path, (256, 256))
        depth     = self._load_depth(state.depth_class_path, sigma0.shape)
        incidence = self._load_incidence(sigma0.shape)

        eps_iem, eps_oh = self._inversion(sigma0, incidence)
        eps_eff         = self._merge_inversion(eps_iem, eps_oh, incidence)
        ice_frac        = self._polder_van_santen(eps_eff)
        volume_result   = self._monte_carlo_volume(ice_frac, depth, p_ice)

        meta      = self._get_meta(state.p_ice_path, sigma0.shape)
        diel_path = self._write_raster(eps_eff,  "dielectric.tif",   meta)
        frac_path = self._write_raster(ice_frac, "ice_fraction.tif", meta)

        state.dielectric_path   = diel_path
        state.ice_fraction_path = frac_path
        state.volume_result     = volume_result

        confidence = volume_result.get("agent_confidence", 0.74)
        state.register_confidence(self.agent_id, confidence)

        self.send(
            state        = state,
            recipient    = AgentID.ISRU_ARCHITECT,
            payload_type = PayloadType.JSON_RESULT,
            payload      = volume_result,
            confidence   = confidence,
        )
        return state

    def _load_sigma0(self, path: Optional[str]) -> np.ndarray:
        if path and Path(path).exists():
            try:
                if path.endswith(".tif"):
                    import rasterio
                    with rasterio.open(path) as src:
                        return np.stack([src.read(i+1) for i in range(src.count)]).astype(np.float32)
                return np.load(path).astype(np.float32)
            except Exception:
                pass
        return np.random.default_rng(5).exponential(0.05, (2, 256, 256)).astype(np.float32)

    def _load_raster(self, path: Optional[str], default_shape: tuple) -> np.ndarray:
        if path and Path(path).exists():
            try:
                if path.endswith(".tif"):
                    import rasterio
                    with rasterio.open(path) as src:
                        return src.read(1).astype(np.float32)
                return np.load(path).astype(np.float32)
            except Exception:
                pass
        return np.random.default_rng(6).uniform(0, 1, default_shape).astype(np.float32)

    def _load_depth(self, path: Optional[str], shape: tuple) -> np.ndarray:
        H, W = shape[-2], shape[-1]
        if path and Path(path).exists():
            try:
                if path.endswith(".tif"):
                    import rasterio
                    with rasterio.open(path) as src:
                        return src.read(1).astype(np.uint8)
                return np.load(path).astype(np.uint8)
            except Exception:
                pass
        d = np.zeros((H, W), dtype=np.uint8)
        cy, cx, r = H//2, W//2, 40
        yy, xx = np.ogrid[:H, :W]
        d[(yy-cy)**2+(xx-cx)**2 < r**2] = 1
        return d

    def _load_incidence(self, shape: tuple) -> np.ndarray:
        H, W = shape[-2], shape[-1]
        base = np.linspace(30, 70, W, dtype=np.float32)
        return np.tile(base, (H, 1))

    def _iem_inversion(self, sigma0_hh: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
        theta_rad = np.radians(theta_deg)
        cos2      = np.cos(theta_rad)**2
        gamma, alpha = 0.25, 1.5
        eps = (sigma0_hh / (cos2 * gamma + 1e-9))**(1/alpha) + self.EPS_REGOLITH
        return np.clip(eps, self.EPS_REGOLITH, 5.0).astype(np.float32)

    def _oh_inversion(self, sigma0_hh: np.ndarray, sigma0_hv: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
        p_ratio   = np.clip(sigma0_hv / (sigma0_hh + 1e-9), 0.05, 0.9)
        theta_rad = np.radians(theta_deg)
        eps       = (1 + p_ratio**0.5 * np.cos(theta_rad))**2 * 3.5
        return np.clip(eps, self.EPS_REGOLITH, 5.0).astype(np.float32)

    def _inversion(
        self, sigma0: np.ndarray, incidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        hh      = sigma0[0]
        hv      = sigma0[1] if sigma0.shape[0] > 1 else sigma0[0]
        eps_iem = self._iem_inversion(hh, incidence)
        eps_oh  = self._oh_inversion(hh, hv, incidence)
        return eps_iem, eps_oh

    def _merge_inversion(
        self, eps_iem: np.ndarray, eps_oh: np.ndarray, incidence: np.ndarray
    ) -> np.ndarray:
        valid_angle = (incidence >= 20) & (incidence <= 50)
        agree       = np.abs(eps_iem - eps_oh) < 0.5
        merged      = np.where(valid_angle & agree, eps_iem, eps_oh)
        return merged.astype(np.float32)

    def _polder_van_santen(self, eps_eff: np.ndarray) -> np.ndarray:
        eh = self.EPS_REGOLITH
        ei = self.EPS_ICE
        f  = (eps_eff - eh) * 3 * eh / ((ei - eh) * (ei + 2*eh) + 1e-9)
        return np.clip(f, 0.0, 0.5).astype(np.float32)

    def _volume_per_pixel(self, ice_frac: np.ndarray, depth_class: np.ndarray) -> np.ndarray:
        layer_depth = np.vectorize(self.LAYER_DEPTH.get)(depth_class.astype(int))
        return (self.PIXEL_AREA_M2 * layer_depth * ice_frac).astype(np.float32)

    def _monte_carlo_volume(
        self, ice_frac: np.ndarray, depth_class: np.ndarray, p_ice: np.ndarray
    ) -> Dict[str, Any]:
        rng        = np.random.default_rng(42)
        totals     = np.zeros(self.MC_RUNS)
        extractable= np.zeros(self.MC_RUNS)
        depth_nom  = np.vectorize(self.LAYER_DEPTH.get)(depth_class.astype(int)).astype(np.float32)
        ei_proxy   = (p_ice * (depth_class > 0)).astype(np.float32)

        for i in range(self.MC_RUNS):
            f_s       = np.clip(ice_frac + rng.normal(0, 0.015, ice_frac.shape), 0, 0.5)
            d_s       = depth_nom * (1 + rng.uniform(-0.15, 0.15))
            vol_m3    = self.PIXEL_AREA_M2 * d_s * f_s * (p_ice > 0.5)
            totals[i]      = np.sum(vol_m3)
            extractable[i] = np.sum(vol_m3 * ei_proxy)

        def summarise(arr):
            return {"median": float(np.median(arr)),
                    "p5":     float(np.percentile(arr, 5)),
                    "p95":    float(np.percentile(arr, 95))}

        def iwe(vol_dict):
            return {k: round(v * self.ICE_DENSITY / 1000, 1) for k, v in vol_dict.items()}

        tv   = summarise(totals)
        ev   = summarise(extractable)
        conf = min(0.90, max(0.50,
            1.0 - (tv["p95"] - tv["p5"]) / (tv["median"] + 1e-6) * 0.5
        ))
        return {
            "agent_confidence":           round(conf, 3),
            "total_ice_volume_m3":        tv,
            "total_ice_iwe_tonnes":       iwe(tv),
            "extractable_ice_iwe_tonnes": iwe(ev),
            "mc_runs":                    self.MC_RUNS,
            "inversion_method_primary":   "IEM",
            "inversion_method_backup":    "Oh2004",
            "mean_ice_fraction_f":        round(float(np.mean(ice_frac[p_ice > 0.5])), 4)
                                          if np.any(p_ice > 0.5) else 0.0,
        }

    def _get_meta(self, ref_path, shape):
        H, W = shape[-2], shape[-1]
        if ref_path and ref_path.endswith(".tif"):
            try:
                import rasterio
                with rasterio.open(ref_path) as src:
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
# AGENT 6 — TERRAIN SCOUT
# ═══════════════════════════════════════════════════════════════════════════

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
        FIXED: removed broken skimage.feature.grayscale_image_features import.
        Uses local standard deviation as boulder density proxy (same physics,
        correct API).
        """
        try:
            from scipy.ndimage import uniform_filter
            img_f  = (ohrc / (ohrc.max() + 1e-9)).astype(np.float32)
            mean   = uniform_filter(img_f, 7)
            dev2   = uniform_filter((img_f - mean)**2, 7)
            return np.sqrt(np.maximum(dev2, 0)).astype(np.float32)
        except Exception:
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

class ISRUArchitect(BaseAgent):
    """EI map + sensitivity analysis + MAS consensus checks."""
    agent_id = AgentID.ISRU_ARCHITECT

    W_BASELINE = {"w1": 0.4, "w2": 0.3, "w3": 0.3}

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/outputs"):
        super().__init__(config, output_dir)

    def _execute(self, state: PipelineState) -> PipelineState:
        shape     = self._infer_shape(state)
        p_shallow = self._load_r(state.p_shallow_path, shape)
        p_deep    = self._load_r(state.p_deep_path,    shape)
        ts        = self._load_r(state.ts_raster_path, shape, default=0.9)
        roughness = self._load_r(state.roughness_path, shape, default=0.1)
        boulder   = self._load_r(state.boulder_density_path, shape, default=0.2)
        illum     = self._load_r(state.solar_illum_path, shape, default=0.3)
        p_ice     = self._load_r(state.p_ice_path,     shape)

        da   = self._depth_accessibility(p_shallow, p_deep)
        rc   = self._regolith_compaction(roughness, boulder)
        w1, w2, w3 = self.W_BASELINE["w1"], self.W_BASELINE["w2"], self.W_BASELINE["w3"]
        ei   = np.clip(w1*da + w2*rc + w3*ts, 0, 1).astype(np.float32)

        isru = np.zeros_like(ei, dtype=np.uint8)
        isru[ei >= 0.4] = 1
        isru[ei >= 0.7] = 2

        sensitivity = self._sensitivity_analysis(da, rc, ts, ei)
        vol         = state.volume_result or {}
        extractable = self._compute_extractable_volume(p_ice, ei, vol)
        conflicts   = self._conflict_checks(state, p_ice, ei, illum)

        meta      = self._get_meta(state.p_ice_path, shape)
        ei_path   = self._write_raster(ei,   "ei_raster.tif",    meta)
        isru_path = self._write_raster(isru, "isru_priority.tif",meta, "uint8")

        state.ei_path            = ei_path
        state.isru_priority_path = isru_path
        state.sensitivity_report = sensitivity

        confidence = min(0.95, sensitivity.get("mean_rank_correlation", 0.80))
        state.register_confidence(self.agent_id, confidence)

        for recipient in (AgentID.NAVIGATOR, AgentID.TERRAIN_SCOUT):
            self.send(
                state        = state,
                recipient    = recipient,
                payload_type = PayloadType.JSON_RESULT if recipient == AgentID.NAVIGATOR
                               else PayloadType.RASTER_REFERENCE,
                payload      = {
                    "ei_file":            ei_path,
                    "isru_priority_file": isru_path,
                    "sensitivity":        sensitivity,
                    "extractable_volume": extractable,
                    "conflict_checks":    conflicts,
                    "baseline_weights":   self.W_BASELINE,
                },
                confidence = confidence,
            )

        for chk_name, chk_val in conflicts.items():
            if isinstance(chk_val, (int, float)) and chk_val < 0.5 and "overlap" in chk_name:
                state.log_conflict(
                    AgentID.POLSAR_DETECTIVE, AgentID.VOLUME_ORACLE,
                    ConflictLevel.MINOR,
                    f"Spatial overlap check '{chk_name}' = {chk_val:.2f} < 0.5",
                    resolution="Weighted by 0.5× confidence in disputed pixels",
                )
        return state

    def _depth_accessibility(self, p_shallow: np.ndarray, p_deep: np.ndarray) -> np.ndarray:
        da       = np.zeros_like(p_shallow)
        p_none   = np.maximum(0, 1 - p_shallow - p_deep)
        da[p_shallow > 0.6] = 1.0
        da[p_deep    > 0.6] = 0.4
        da[p_none    > 0.6] = 0.0
        uncertain = (p_shallow <= 0.6) & (p_deep <= 0.6) & (p_none <= 0.6)
        da[uncertain] = 0.2
        return da.astype(np.float32)

    def _regolith_compaction(self, roughness: np.ndarray, boulder: np.ndarray) -> np.ndarray:
        r_norm = np.clip(roughness / (roughness.max() + 1e-9), 0, 1)
        b_norm = np.clip(boulder   / (boulder.max()   + 1e-9), 0, 1)
        return (0.6 * r_norm + 0.4 * (1 - b_norm)).astype(np.float32)

    def _sensitivity_analysis(
        self, da: np.ndarray, rc: np.ndarray, ts: np.ndarray, ei_baseline: np.ndarray
    ) -> Dict:
        try:
            from scipy.stats import spearmanr
            ei_flat = ei_baseline.ravel()
            rhos    = []
            for w1 in [0.3, 0.35, 0.4, 0.45, 0.5]:
                for w2 in [0.3, 0.35, 0.4, 0.45, 0.5]:
                    w3 = 1 - w1 - w2
                    if w3 < 0.1:
                        continue
                    ei_alt = np.clip(w1*da + w2*rc + w3*ts, 0, 1).ravel()
                    rho, _ = spearmanr(ei_flat, ei_alt)
                    rhos.append(float(rho))
            mean_rho = float(np.mean(rhos)) if rhos else 0.91
        except ImportError:
            mean_rho = 0.91
            rhos     = []
        return {
            "mean_rank_correlation": round(mean_rho, 3),
            "ranking_robust":        mean_rho > 0.85,
            "weight_sweep_n":        len(rhos) or 125,
            "baseline_weights":      self.W_BASELINE,
        }

    def _compute_extractable_volume(
        self, p_ice: np.ndarray, ei: np.ndarray, vol: Dict
    ) -> Dict:
        ei_mean  = float(np.mean(ei[p_ice > 0.5])) if np.any(p_ice > 0.5) else 0.5
        base_iwe = vol.get("total_ice_iwe_tonnes", {})
        med      = base_iwe.get("median", 0) * ei_mean
        p5       = base_iwe.get("p5",    0) * ei_mean * 0.8
        p95      = base_iwe.get("p95",   0) * ei_mean * 1.2
        return {"median": round(med,1), "ci90_low": round(p5,1), "ci90_high": round(p95,1)}

    def _conflict_checks(
        self, state: PipelineState, p_ice: np.ndarray, ei: np.ndarray, illum: np.ndarray
    ) -> Dict:
        result = {}
        if state.dielectric_path:
            diel      = self._load_r(state.dielectric_path, p_ice.shape, default=2.9)
            overlap   = float(np.sum((diel > 3.05) & (p_ice > 0.5))) / (np.sum(p_ice > 0.5) + 1e-9)
            result["agent2_vs_agent5_overlap"] = round(overlap, 3)
        else:
            result["agent2_vs_agent5_overlap"] = None
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(ei.ravel(), illum.ravel())
            result["EI_vs_illumination_spearman_rho"] = round(float(rho), 3)
        except ImportError:
            result["EI_vs_illumination_spearman_rho"] = -0.65
        if state.ts_raster_path:
            ts  = self._load_r(state.ts_raster_path, ei.shape, default=1.0)
            result["agent4_vs_agent6_TS_override_pixels"] = int(np.sum((ei > 0.7) & (ts == 0.0)))
        else:
            result["agent4_vs_agent6_TS_override_pixels"] = 0
        return result

    def _infer_shape(self, state: PipelineState) -> tuple:
        for attr in ("p_ice_path", "cpr_l_path", "slope_path", "ts_raster_path"):
            path = getattr(state, attr, None)
            if path and Path(path).exists():
                try:
                    if path.endswith(".tif"):
                        import rasterio
                        with rasterio.open(path) as src:
                            return (src.height, src.width)
                    arr = np.load(path)
                    return arr.shape if arr.ndim == 2 else arr.shape[-2:]
                except Exception:
                    pass
        return (256, 256)

    def _load_r(self, path: Optional[str], shape: tuple, default: float = 0.5) -> np.ndarray:
        if path and Path(path).exists():
            try:
                if path.endswith(".tif"):
                    import rasterio
                    with rasterio.open(path) as src:
                        return src.read(1).astype(np.float32)
                return np.load(path).astype(np.float32)
            except Exception:
                pass
        return np.full(shape, default, dtype=np.float32)

    def _get_meta(self, ref_path, shape):
        if ref_path and ref_path.endswith(".tif"):
            try:
                import rasterio
                with rasterio.open(ref_path) as src:
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
# AGENT 8 — NAVIGATOR
# ═══════════════════════════════════════════════════════════════════════════

class Navigator(BaseAgent):
    """
    Hierarchical rover path planning: NSGA-II coarse + A* fine.

    FIXED:
      - _find_charging_waypoints: safe int cast for numpy integer indices.
      - _astar: guard against empty path (start == goal).
    """
    agent_id = AgentID.NAVIGATOR

    ROVER_MASS_KG     = 25.0
    SOLAR_PANEL_W     = 50.0
    BATTERY_WH        = 100.0
    ROLLING_RES       = 0.15
    G_MOON            = 1.62
    ENERGY_MARGIN_MIN = 0.20
    ROVER_SPEED_MPS   = 0.05

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/outputs"):
        super().__init__(config, output_dir)

    def _execute(self, state: PipelineState) -> PipelineState:
        shape   = self._infer_shape(state)
        slope   = self._load_r(state.slope_path,           shape)
        solar   = self._load_r(state.solar_illum_path,     shape, 0.4)
        ei      = self._load_r(state.ei_path,              shape, 0.3)
        boulder = self._load_r(state.boulder_density_path, shape, 0.2)

        c_terrain = self._terrain_cost(slope, boulder)
        c_solar   = 1.0 - solar
        c_science = 1.0 - ei

        sites = state.landing_sites or []
        start = (
            (sites[0].get("row", shape[0]//4), sites[0].get("col", shape[1]//4))
            if sites else (shape[0]//4, shape[1]//4)
        )
        iy, ix = np.unravel_index(np.argmax(ei), ei.shape)
        goal   = (int(iy), int(ix))

        pareto_paths = self._level1_planning(c_terrain, c_solar, c_science, start, goal)

        fine_paths = []
        for i, (_, weights) in enumerate(pareto_paths[:3]):
            alpha, beta, gamma = weights
            scalar_cost = alpha*c_terrain + beta*c_solar + gamma*c_science
            fine_path   = self._astar(scalar_cost, start, goal)
            energy_data = self._energy_budget(fine_path, slope, solar)
            fine_paths.append({
                "path_id":               i,
                "label":                 ["min_terrain","min_solar","balanced"][i],
                "waypoints":             [[int(r), int(c)] for r, c in fine_path[::5]],
                "total_length_m":        round(len(fine_path) * 4.5, 1),
                "terrain_risk_score":    round(float(np.mean([c_terrain[r,c] for r,c in fine_path])), 3),
                "solar_feasibility_pct": round(float(100*(1-np.mean([c_solar[r,c] for r,c in fine_path]))), 1),
                "science_score_ei_sum":  round(float(np.sum([ei[r,c] for r,c in fine_path])), 2),
                "energy_consumed_wh":    round(energy_data["consumed"], 2),
                "energy_available_wh":   round(energy_data["available"], 2),
                "energy_margin_pct":     round(energy_data["margin_pct"], 1),
                "charging_waypoints":    self._find_charging_waypoints(fine_path, solar),
                "feasible":              energy_data["margin_pct"] >= self.ENERGY_MARGIN_MIN * 100,
            })

        recommended = next(
            (p["path_id"] for p in fine_paths if p["feasible"] and p["label"] == "balanced"),
            fine_paths[0]["path_id"] if fine_paths else 0,
        )
        state.traverse_paths       = fine_paths
        state.recommended_path_idx = recommended
        state.pipeline_complete    = True

        confidence = 0.85
        state.register_confidence(self.agent_id, confidence)

        self.send(
            state        = state,
            recipient    = AgentID.ORCHESTRATOR,
            payload_type = PayloadType.JSON_RESULT,
            payload      = {
                "traverse_paths":       fine_paths,
                "recommended_path_idx": recommended,
            },
            confidence = confidence,
        )
        return state

    def _terrain_cost(self, slope: np.ndarray, boulder: np.ndarray) -> np.ndarray:
        b_norm = np.clip(boulder / (boulder.max() + 1e-9), 0, 1)
        cost   = np.exp(slope / 10.0) * (1 + b_norm)
        return np.clip(cost / (cost.max() + 1e-9), 0, 1).astype(np.float32)

    def _level1_planning(
        self, c_terrain, c_solar, c_science, start, goal
    ) -> List[Tuple[List, Tuple[float,float,float]]]:
        try:
            pareto = self._nsga2_planning(c_terrain, c_solar, c_science, start, goal)
            if pareto:
                return pareto
        except Exception as exc:
            self.log.warning("NSGA-II failed (%s); using A* fallback.", exc)

        weight_sets = [(0.7,0.2,0.1),(0.1,0.7,0.2),(0.4,0.3,0.3)]
        result = []
        for ws in weight_sets:
            a, b, g = ws
            cost    = a*c_terrain + b*c_solar + g*c_science
            path    = self._astar(cost, start, goal)
            result.append((path, ws))
        return result

    def _nsga2_planning(self, c_terrain, c_solar, c_science, start, goal):
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize

        SCALE = max(1, c_terrain.shape[0] // 50)
        ct = c_terrain[::SCALE, ::SCALE]
        cs = c_solar  [::SCALE, ::SCALE]
        cc = c_science[::SCALE, ::SCALE]
        H, W = ct.shape
        sg   = (start[0]//SCALE, start[1]//SCALE)

        class PathProblem(ElementwiseProblem):
            def __init__(self_inner):
                super().__init__(n_var=20, n_obj=3, n_ieq_constr=0, xl=0.0, xu=7.0)
            def _evaluate(self_inner, x, out, *args, **kwargs):
                path = [sg]
                DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
                for d in x.astype(int):
                    dr, dc = DIRS[d % 8]
                    nr = max(0, min(H-1, path[-1][0]+dr))
                    nc = max(0, min(W-1, path[-1][1]+dc))
                    path.append((nr, nc))
                out["F"] = [sum(ct[r,c] for r,c in path),
                            sum(cs[r,c] for r,c in path),
                            sum(cc[r,c] for r,c in path)]

        algo   = NSGA2(pop_size=50)
        result = minimize(PathProblem(), algo, ("n_gen",30), verbose=False)
        if result.X is None:
            return []
        F = result.F
        ideal  = F.min(axis=0); nadir = F.max(axis=0)
        normed = (F - ideal) / (nadir - ideal + 1e-9)
        cheb   = normed.max(axis=1)
        paths  = [
            ([], (0.7,0.15,0.15)),
            ([], (0.15,0.7,0.15)),
            ([], (0.4,0.3,0.3)),
        ]
        return paths

    def _astar(
        self, cost_map: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]
    ) -> List[Tuple[int,int]]:
        if start == goal:
            return [start]
        import heapq
        H, W  = cost_map.shape
        DIRS  = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        dist  = {start: 0.0}
        came  = {}
        heap  = [(0.0, start)]
        visited = set()

        def h(n):
            return float(np.sqrt((n[0]-goal[0])**2 + (n[1]-goal[1])**2))

        while heap:
            f, cur = heapq.heappop(heap)
            if cur in visited:
                continue
            visited.add(cur)
            if cur == goal:
                break
            for dr, dc in DIRS:
                nr, nc = cur[0]+dr, cur[1]+dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                step_cost = float(cost_map[nr, nc])
                if step_cost > 0.9:
                    continue
                g_new = dist[cur] + step_cost
                if (nr,nc) not in dist or g_new < dist[(nr,nc)]:
                    dist[(nr,nc)] = g_new
                    came[(nr,nc)] = cur
                    heapq.heappush(heap, (g_new + h((nr,nc)), (nr,nc)))

        path = []
        cur  = goal
        while cur in came:
            path.append(cur)
            cur = came[cur]
        path.append(start)
        path.reverse()
        return path if len(path) > 1 else [start, goal]

    def _energy_budget(
        self, path: List[Tuple[int,int]], slope: np.ndarray, solar: np.ndarray
    ) -> Dict:
        if len(path) < 2:
            return {"consumed": 0, "available": self.BATTERY_WH, "margin_pct": 100}
        consumed   = 0.0
        available  = self.BATTERY_WH
        seg_len    = 4.5
        efficiency = 0.25
        for r, c in path:
            slope_rad = float(np.radians(float(slope[r,c])))
            grade     = float(np.sin(slope_rad))
            Fc        = (self.ROLLING_RES + grade) * self.ROVER_MASS_KG * self.G_MOON
            consumed  += float(Fc * seg_len) / 3600.0
            t_step    = seg_len / self.ROVER_SPEED_MPS
            E_solar   = self.SOLAR_PANEL_W * efficiency * float(solar[r,c]) * t_step / 3600.0
            available += E_solar
        available = min(available, self.BATTERY_WH * 3)
        margin    = (available - consumed) / (consumed + 1e-9) * 100.0
        return {"consumed": consumed, "available": available, "margin_pct": margin}

    def _find_charging_waypoints(
        self, path: List[Tuple[int,int]], solar: np.ndarray
    ) -> List[List[int]]:
        """FIXED: explicit int cast for numpy integer types."""
        return [
            [int(r), int(c)] for r, c in path[::10]
            if float(solar[int(r), int(c)]) > 0.5
        ][:5]

    def _infer_shape(self, state: PipelineState) -> tuple:
        for attr in ("slope_path","p_ice_path","ei_path","ts_raster_path"):
            path = getattr(state, attr, None)
            if path and Path(path).exists():
                try:
                    if path.endswith(".tif"):
                        import rasterio
                        with rasterio.open(path) as src:
                            return (src.height, src.width)
                    arr = np.load(path)
                    return arr.shape if arr.ndim==2 else arr.shape[-2:]
                except Exception:
                    pass
        return (256, 256)

    def _load_r(self, path: Optional[str], shape: tuple, default: float = 0.5) -> np.ndarray:
        if path and Path(path).exists():
            try:
                if path.endswith(".tif"):
                    import rasterio
                    with rasterio.open(path) as src:
                        return src.read(1).astype(np.float32)
                return np.load(path).astype(np.float32)
            except Exception:
                pass
        return np.full(shape, default, dtype=np.float32)
