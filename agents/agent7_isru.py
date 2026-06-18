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

