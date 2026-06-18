"""
PRISM — Agent 3: DEPTH SOUNDER
================================
Physics basis:
  L-band (~1.27 GHz) penetration depth in dry regolith ≈ 3–5 m
  S-band (~3.2 GHz) penetration depth ≈ 1–2 m

Classification rules (per MC sample):
  CPR_L > 1.0 AND CPR_S > 1.0  → shallow (0–2 m)
  CPR_L > 1.0 AND CPR_S ≤ 1.0  → deep    (2–5 m)
  CPR_L ≤ 1.0                  → no detectable ice

Uncertainty:
  σ_CPR = CPR × √(2/ENL)    (from speckle statistics)
  500 Monte Carlo samples → P(shallow), P(deep), P(none)
  Final class = argmax; uncertainty = 1 − max(P vector)

TRAINED MODEL SLOT
------------------
No trained ML model required for this agent.
All classification is physics-based with MC uncertainty propagation.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.base_agent import BaseAgent
from core.protocol import AgentID, PayloadType, PipelineState

log = logging.getLogger("PRISM.DEPTH_SOUNDER")

# Depth class integer codes
NO_ICE  = 0
SHALLOW = 1   # 0–2 m
DEEP    = 2   # 2–5 m


class DepthSounder(BaseAgent):
    agent_id = AgentID.DEPTH_SOUNDER

    MC_SAMPLES        = 500
    P_ICE_MIN         = 0.5   # only process pixels where Agent2 says P_ice > this
    UNCERTAIN_THRESH  = 0.6   # max(P) < this → flag as uncertain

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/outputs"):
        super().__init__(config, output_dir)

    # -----------------------------------------------------------------------
    def _execute(self, state: PipelineState) -> PipelineState:
        enl      = state.enl or 12.0
        cpr_l    = self._load_raster(state.cpr_l_path)
        cpr_s    = self._load_raster(state.cpr_s_path)
        p_ice    = self._load_raster(state.p_ice_path)
        quality  = self._load_mask(state.quality_mask, cpr_l.shape)

        # Run MC depth stratification
        depth_class, p_shallow, p_deep, uncertainty = self._mc_depth_classify(
            cpr_l, cpr_s, p_ice, quality, enl
        )

        meta = self._get_meta(state.cpr_l_path, cpr_l.shape)

        depth_class_path   = self._write_raster(depth_class,   "depth_class.tif",    meta, "uint8")
        uncertainty_path   = self._write_raster(uncertainty,   "depth_uncertainty.tif", meta)
        p_shallow_path     = self._write_raster(p_shallow,     "p_shallow.tif",      meta)
        p_deep_path        = self._write_raster(p_deep,        "p_deep.tif",         meta)

        state.depth_class_path       = depth_class_path
        state.depth_uncertainty_path = uncertainty_path
        state.p_shallow_path         = p_shallow_path
        state.p_deep_path            = p_deep_path

        shallow_n   = int(np.sum(depth_class == SHALLOW))
        deep_n      = int(np.sum(depth_class == DEEP))
        uncertain_n = int(np.sum((uncertainty > (1 - self.UNCERTAIN_THRESH)) & (p_ice > self.P_ICE_MIN)))
        confidence  = 1.0 - float(np.mean(uncertainty[p_ice > self.P_ICE_MIN])) if p_ice.any() else 0.5

        state.register_confidence(self.agent_id, confidence)

        self.send(
            state       = state,
            recipient   = AgentID.VOLUME_ORACLE,
            payload_type= PayloadType.RASTER_REFERENCE,
            payload     = {
                "depth_class_file":    depth_class_path,
                "uncertainty_file":    uncertainty_path,
                "p_shallow_file":      p_shallow_path,
                "p_deep_file":         p_deep_path,
                "shallow_pixel_count": shallow_n,
                "deep_pixel_count":    deep_n,
                "uncertain_pixel_count": uncertain_n,
                "mc_samples_used":     self.MC_SAMPLES,
            },
            confidence  = confidence,
            notes       = f"ENL={enl:.2f}, MC={self.MC_SAMPLES} samples",
        )

        self.send(
            state       = state,
            recipient   = AgentID.ISRU_ARCHITECT,
            payload_type= PayloadType.RASTER_REFERENCE,
            payload     = {
                "depth_class_file":   depth_class_path,
                "p_shallow_file":     p_shallow_path,
                "p_deep_file":        p_deep_path,
                "uncertainty_file":   uncertainty_path,
            },
            confidence  = confidence,
        )
        return state

    # -----------------------------------------------------------------------
    # Physics
    # -----------------------------------------------------------------------

    def _mc_depth_classify(
        self,
        cpr_l:   np.ndarray,
        cpr_s:   np.ndarray,
        p_ice:   np.ndarray,
        quality: np.ndarray,
        enl:     float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        For each pixel where P_ice > 0.5 and quality == 1:
          1. σ_CPR_L = CPR_L × √(2/ENL)   (speckle noise from ENL)
          2. Draw MC_SAMPLES samples from Normal(CPR_L, σ_CPR_L)
             and Normal(CPR_S, σ_CPR_S)
          3. Vote depth class per sample
          4. P(shallow), P(deep), P(none) = vote fractions
          5. depth_class = argmax; uncertainty = 1 - max(P)
        """
        H, W = cpr_l.shape
        p_shallow   = np.zeros((H, W), dtype=np.float32)
        p_deep      = np.zeros((H, W), dtype=np.float32)
        p_none      = np.ones ((H, W), dtype=np.float32)
        depth_class = np.zeros((H, W), dtype=np.uint8)
        uncertainty = np.zeros((H, W), dtype=np.float32)

        sigma_factor = float(np.sqrt(2.0 / max(enl, 1.0)))

        # Vectorised MC for all candidate pixels at once
        candidate = (p_ice > self.P_ICE_MIN) & (quality == 1)
        idx       = np.where(candidate)
        if len(idx[0]) == 0:
            self.log.warning("No pixels passed P_ice threshold — depth map will be empty.")
            return depth_class, p_shallow, p_deep, uncertainty

        cpr_l_vec = cpr_l[idx].astype(np.float64)
        cpr_s_vec = cpr_s[idx].astype(np.float64)
        N         = len(cpr_l_vec)

        rng       = np.random.default_rng(42)
        # Samples: shape (MC_SAMPLES, N)
        sl = rng.normal(
            loc   = cpr_l_vec[None, :],
            scale = (cpr_l_vec * sigma_factor)[None, :],
            size  = (self.MC_SAMPLES, N),
        )
        ss = rng.normal(
            loc   = cpr_s_vec[None, :],
            scale = (cpr_s_vec * sigma_factor)[None, :],
            size  = (self.MC_SAMPLES, N),
        )
        sl = np.clip(sl, 0, None)
        ss = np.clip(ss, 0, None)

        # Vectorised voting
        # shallow if CPR_L > 1 AND CPR_S > 1
        # deep    if CPR_L > 1 AND CPR_S <= 1
        # none    otherwise
        vote_shallow = ((sl > 1.0) & (ss > 1.0)).astype(np.float32)  # (MC, N)
        vote_deep    = ((sl > 1.0) & (ss <= 1.0)).astype(np.float32)
        vote_none    = 1.0 - vote_shallow - vote_deep

        ps = vote_shallow.mean(axis=0)   # (N,)
        pd = vote_deep.mean(axis=0)
        pn = vote_none.mean(axis=0)

        # argmax class
        stack      = np.stack([pn, ps, pd], axis=1)   # (N, 3)
        cls        = np.argmax(stack, axis=1).astype(np.uint8)   # 0=none,1=shallow,2=deep
        uncert     = 1.0 - stack[np.arange(N), cls]

        # Write back
        p_shallow[idx]   = ps.astype(np.float32)
        p_deep[idx]      = pd.astype(np.float32)
        p_none[idx]      = pn.astype(np.float32)
        depth_class[idx] = cls
        uncertainty[idx] = uncert.astype(np.float32)

        return depth_class, p_shallow, p_deep, uncertainty

    # -----------------------------------------------------------------------
    # I/O helpers
    # -----------------------------------------------------------------------

    def _load_raster(self, path: Optional[str]) -> np.ndarray:
        if path and Path(path).exists():
            if path.endswith(".tif"):
                try:
                    import rasterio
                    with rasterio.open(path) as src:
                        return src.read(1).astype(np.float32)
                except Exception:
                    pass
            elif path.endswith(".npy"):
                return np.load(path).astype(np.float32)

        # Synthetic fallback
        rng = np.random.default_rng(99)
        arr = rng.exponential(0.8, (256, 256)).astype(np.float32)
        arr = np.clip(arr, 0, 5)
        return arr

    def _load_mask(self, path: Optional[str], shape: tuple) -> np.ndarray:
        if path and Path(path).exists():
            try:
                if path.endswith(".tif"):
                    import rasterio
                    with rasterio.open(path) as src:
                        return src.read(1).astype(np.uint8)
                else:
                    return np.load(path).astype(np.uint8)
            except Exception:
                pass
        return np.ones(shape, dtype=np.uint8)

    def _get_meta(self, ref_path: Optional[str], shape: tuple) -> Optional[dict]:
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

    def _write_raster(
        self, array: np.ndarray, name: str, meta: Optional[dict], dtype: str = "float32"
    ) -> str:
        path = self.output_path(name)
        arr  = array.astype(dtype)
        if meta is not None:
            try:
                import rasterio
                m = meta.copy()
                m.update(count=1, dtype=dtype)
                with rasterio.open(path, "w", **m) as dst:
                    dst.write(arr, 1)
                return path
            except Exception:
                pass
        npy = path.replace(".tif", ".npy")
        np.save(npy, arr)
        return npy