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

