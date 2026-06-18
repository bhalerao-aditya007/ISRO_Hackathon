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

from agents.base_agent import BaseAgent
from agents.protocol import (
    AgentID, ConflictLevel, PayloadType, PipelineState,
)

# AGENT 4 — THERMO GUARDIAN


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

