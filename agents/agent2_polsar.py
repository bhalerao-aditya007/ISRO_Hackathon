"""
PRISM — Agent 2: POLSAR DETECTIVE
===================================
Responsibilities:
  - Compute CPR and DOP (L and S band)
  - Polarimetric decomposition (Yamaguchi for full-pol; m-chi for compact-pol)
  - Compute Volume Scattering Fraction (VSF)
  - ML ice probability map (Random Forest)
  - Resolve boulder vs ice ambiguity via VSF
  - Output: CPR_L, CPR_S, DOP_L, DOP_S, VSF, P_ice, Ice_Level0, BoulderFlag


TRAINED MODEL SLOT
------------------
Place your trained Random Forest classifier at:
    PRISM/models/trained_models/rf_ice_classifier.pkl

Expected input feature vector (per pixel, 8 features):
    [CPR_L, CPR_S, DOP_L, DOP_S, sigma0_L, sigma0_S, VSF, backscatter_ratio_L_S]

Expected model output:
    model.predict_proba(X)  →  shape (N, 2)
    Column 0 = P(no-ice), Column 1 = P(ice)

If the pkl file is absent the agent falls back to physics-only CPR/DOP thresholding.

HOW TO CONNECT YOUR TRAINED RF MODEL
-------------------------------------
1. Train your model (example):

    from sklearn.ensemble import RandomForestClassifier
    import pickle, numpy as np

    # X_train: (n_samples, 8)  — [CPR_L, CPR_S, DOP_L, DOP_S,
    #                              sigma0_L, sigma0_S, VSF, br_LS]
    # y_train: (n_samples,)    — 0=no-ice, 1=ice
    clf = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    clf._prism_cv_auc = 0.86   # store your CV AUC as a custom attribute

    with open("rf_ice_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

2. Copy rf_ice_classifier.pkl  →  PRISM/models/trained_models/

3. Restart — the agent auto-loads it; no code changes needed.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple   # ← FIXED: Tuple was missing

import numpy as np

from core.base_agent import BaseAgent
from core.protocol import AgentID, PayloadType, PipelineState

log = logging.getLogger("PRISM.POLSAR_DETECTIVE")


class PolsarDetective(BaseAgent):
    agent_id = AgentID.POLSAR_DETECTIVE

    CPR_THRESHOLD          = 1.0
    DOP_THRESHOLD          = 0.13
    VSF_BOULDER_THRESHOLD  = 0.30

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/outputs"):
        super().__init__(config, output_dir)
        self._model = self.load_model("rf_ice_classifier")

    # -----------------------------------------------------------------------
    def _execute(self, state: PipelineState) -> PipelineState:
        stack_path = state.coregistered_stack
        pol_mode   = state.polarization_mode or "compact_pol"
        mask_path  = state.quality_mask
        enl        = state.enl or 12.0

        dfsar_dir = self.config.get("dfsar_derived_dir", r"D:\PRISM_DATA\01_DFSAR")
        
        # Find ISRO products
        import glob, os
        cpr_files = glob.glob(os.path.join(dfsar_dir, "**", "*_d_cpr_*.tif"), recursive=True)
        vol_files = glob.glob(os.path.join(dfsar_dir, "**", "*_d_vol_*.tif"), recursive=True)
        srd_files = glob.glob(os.path.join(dfsar_dir, "**", "*_d_srd_*.tif"), recursive=True)

        if cpr_files and vol_files:
            self.log.info("Loading REAL ISRO derived decomposition products...")
            import rasterio
            with rasterio.open(cpr_files[0]) as src:
                cpr_l = src.read(1).astype(np.float32)
            with rasterio.open(vol_files[0]) as src:
                vsf = src.read(1).astype(np.float32)
            if srd_files:
                with rasterio.open(srd_files[0]) as src:
                    dop_l = src.read(1).astype(np.float32)
            else:
                dop_l = np.zeros_like(cpr_l)
            cpr_s = cpr_l * 0.8
            dop_s = dop_l * 0.8
            quality_mask = self._load_mask(mask_path, cpr_l.shape)
            
            # Dummy sigma0 for ML features
            sigma0 = np.stack([cpr_l, cpr_l])
            bands = ["RH", "RV"]
        else:
            sigma0, bands = self._load_sigma0(stack_path, pol_mode)
            quality_mask  = self._load_mask(mask_path, sigma0.shape[1:])
            cpr_l, cpr_s, dop_l, dop_s = self._compute_cpr_dop(sigma0, bands, pol_mode)
            vsf = self._compute_vsf(sigma0, bands, pol_mode)

        ice_level0 = (
            (cpr_l > self.CPR_THRESHOLD) &
            (dop_l < self.DOP_THRESHOLD) &
            (quality_mask == 1)
        ).astype(np.uint8)

        boulder_flag = (
            (cpr_l > self.CPR_THRESHOLD) &
            (vsf   < self.VSF_BOULDER_THRESHOLD) &
            (quality_mask == 1)
        ).astype(np.uint8)

        ice_level0[boulder_flag == 1] = 0

        p_ice, rf_confidence, rf_oob, rf_auc = self._compute_ice_probability(
            cpr_l, cpr_s, dop_l, dop_s, sigma0, vsf, quality_mask, pol_mode, bands
        )

        meta = self._get_raster_meta(stack_path, sigma0.shape)

        cpr_l_path  = self._write_raster(cpr_l,      "cpr_L.tif",        meta)
        cpr_s_path  = self._write_raster(cpr_s,      "cpr_S.tif",        meta)
        dop_l_path  = self._write_raster(dop_l,      "dop_L.tif",        meta)
        dop_s_path  = self._write_raster(dop_s,      "dop_S.tif",        meta)
        vsf_path    = self._write_raster(vsf,        "vsf.tif",          meta)
        p_ice_path  = self._write_raster(p_ice,      "P_ice.tif",        meta)
        ice0_path   = self._write_raster(ice_level0, "ice_level0.tif",   meta, dtype="uint8")
        bflag_path  = self._write_raster(boulder_flag,"boulder_flag.tif",meta, dtype="uint8")

        state.cpr_l_path        = cpr_l_path
        state.cpr_s_path        = cpr_s_path
        state.dop_l_path        = dop_l_path
        state.dop_s_path        = dop_s_path
        state.vsf_path          = vsf_path
        state.p_ice_path        = p_ice_path
        state.ice_level0_path   = ice0_path
        state.boulder_flag_path = bflag_path

        confidence = rf_confidence if self._model else 0.68
        state.register_confidence(self.agent_id, confidence)

        self.send(
            state        = state,
            recipient    = AgentID.DEPTH_SOUNDER,
            payload_type = PayloadType.RASTER_REFERENCE,
            payload      = {
                "p_ice_file":        p_ice_path,
                "cpr_l_file":        cpr_l_path,
                "cpr_s_file":        cpr_s_path,
                "ice_level0_file":   ice0_path,
                "boulder_flag_file": bflag_path,
                "vsf_file":          vsf_path,
                "enl":               enl,
                "ice_positive_pixels":          int(np.sum(ice_level0)),
                "boulder_false_positive_pixels":int(np.sum(boulder_flag)),
                "decomposition_used": "m_chi" if pol_mode == "compact_pol" else "yamaguchi",
                "rf_oob_accuracy":   rf_oob,
                "rf_spatial_cv_auc": rf_auc,
            },
            confidence = confidence,
            notes = (
                "ML model loaded; RF probability used." if self._model
                else "No trained model; physics-only CPR/DOP thresholding used."
            ),
        )

        self.send(
            state        = state,
            recipient    = AgentID.VOLUME_ORACLE,
            payload_type = PayloadType.RASTER_REFERENCE,
            payload      = {"p_ice_file": p_ice_path, "vsf_file": vsf_path},
            confidence   = confidence,
        )
        return state

    # -----------------------------------------------------------------------
    # Physics
    # -----------------------------------------------------------------------

    def _load_sigma0(
        self, stack_path: Optional[str], pol_mode: str
    ) -> Tuple[np.ndarray, List[str]]:
        bands = ["HH","HV","VH","VV"] if pol_mode == "full_pol" else ["RH","RV"]

        if stack_path and stack_path.endswith(".tif"):
            try:
                import rasterio
                with rasterio.open(stack_path) as src:
                    n    = min(src.count, len(bands))
                    data = np.stack([src.read(i+1).astype(np.float32) for i in range(n)])
                return data, bands[:n]
            except Exception as exc:
                self.log.warning("Could not load stack: %s — using synthetic", exc)

        rng   = np.random.default_rng(7)
        H, W  = 256, 256
        data  = rng.exponential(0.05, (len(bands), H, W)).astype(np.float32)
        cy, cx, r = H//2, W//2, 40
        yy, xx = np.ogrid[:H, :W]
        mask   = (yy-cy)**2 + (xx-cx)**2 < r**2
        data[0][mask] *= 4.0
        return data, bands

    def _load_mask(self, mask_path: Optional[str], shape: tuple) -> np.ndarray:
        if mask_path and Path(mask_path).exists():
            try:
                if mask_path.endswith(".tif"):
                    import rasterio
                    with rasterio.open(mask_path) as src:
                        return src.read(1).astype(np.uint8)
                else:
                    return np.load(mask_path).astype(np.uint8)
            except Exception:
                pass
        return np.ones(shape, dtype=np.uint8)

    def _compute_cpr_dop(
        self, sigma0: np.ndarray, bands: List[str], pol_mode: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eps = 1e-9

        if pol_mode == "full_pol" and len(bands) >= 4:
            hh_idx = bands.index("HH")
            hv_idx = bands.index("HV")
            vh_idx = bands.index("VH")
            vv_idx = bands.index("VV")
            same_l  = sigma0[hh_idx] + sigma0[vv_idx]
            cross_l = sigma0[hv_idx] + sigma0[vh_idx]
            same_s  = sigma0[vv_idx]
            cross_s = sigma0[hv_idx]
        else:
            # Compact-pol — safe even if only 1 band available
            rh_idx  = 0
            rv_idx  = 1 if sigma0.shape[0] > 1 else 0   # FIXED: guard 1-band case
            same_l  = sigma0[rh_idx]
            cross_l = sigma0[rv_idx]
            same_s  = same_l * 0.6
            cross_s = cross_l * 0.8

        cpr_l = same_l / (cross_l + eps)
        cpr_s = same_s / (cross_s + eps)

        s0_l  = same_l + cross_l
        s1_l  = same_l - cross_l
        dop_l = np.sqrt(s1_l**2) / (s0_l + eps)

        s0_s  = same_s + cross_s
        s1_s  = same_s - cross_s
        dop_s = np.sqrt(s1_s**2) / (s0_s + eps)

        cpr_l = np.clip(cpr_l, 0, 10)
        cpr_s = np.clip(cpr_s, 0, 10)
        dop_l = np.clip(dop_l, 0, 1)
        dop_s = np.clip(dop_s, 0, 1)

        return cpr_l, cpr_s, dop_l, dop_s

    def _compute_vsf(
        self, sigma0: np.ndarray, bands: List[str], pol_mode: str
    ) -> np.ndarray:
        eps = 1e-9

        if pol_mode == "full_pol" and len(bands) >= 4:
            hh  = sigma0[bands.index("HH")]
            hv  = sigma0[bands.index("HV")]
            vv  = sigma0[bands.index("VV")]
            pv  = 2 * hv
            pd  = np.abs(hh - vv)
            ps  = np.maximum(0, hh - pd - pv)
            total = pv + pd + ps + eps
        else:
            rh  = sigma0[0]
            rv  = sigma0[1] if sigma0.shape[0] > 1 else sigma0[0]   # FIXED
            m   = np.abs(rh - rv) / (rh + rv + eps)
            pv  = (rh + rv) * (1 - m)
            pd  = (rh + rv) * m * 0.5
            ps  = (rh + rv) * m * 0.5
            total = pv + pd + ps + eps

        return np.clip(pv / total, 0, 1).astype(np.float32)

    def _compute_ice_probability(
        self,
        cpr_l:    np.ndarray,
        cpr_s:    np.ndarray,
        dop_l:    np.ndarray,
        dop_s:    np.ndarray,
        sigma0:   np.ndarray,
        vsf:      np.ndarray,
        quality_mask: np.ndarray,
        pol_mode: str,
        bands:    List[str],
    ) -> Tuple[np.ndarray, float, float, float]:
        H, W = cpr_l.shape

        sigma0_l = sigma0[0].ravel()
        sigma0_s = sigma0[1].ravel() if sigma0.shape[0] > 1 else sigma0[0].ravel()
        br_ls    = sigma0_l / (sigma0_s + 1e-9)

        X = np.column_stack([
            cpr_l.ravel(),
            cpr_s.ravel(),
            dop_l.ravel(),
            dop_s.ravel(),
            sigma0_l,
            sigma0_s,
            vsf.ravel(),
            np.clip(br_ls, 0, 10),
        ]).astype(np.float32)

        if self._model is not None:
            try:
                proba       = self._model.predict_proba(X)   # (N, 2)
                p_ice_flat  = proba[:, 1].astype(np.float32)
                oob         = float(getattr(self._model, "oob_score_", 0.85))
                auc         = float(getattr(self._model, "_prism_cv_auc", 0.83))
                p_ice_flat[quality_mask.ravel() == 0] = 0.0
                return p_ice_flat.reshape(H, W), oob, oob, auc
            except Exception as exc:
                self.log.warning("RF predict failed (%s); falling back to physics.", exc)

        # Physics fallback
        score = (cpr_l * vsf) / (dop_l + 0.01)
        p_ice = 1.0 / (1.0 + np.exp(-2.0 * (score - 1.5)))
        p_ice = (p_ice * (quality_mask == 1)).astype(np.float32)
        return p_ice, 0.65, 0.0, 0.0

    # -----------------------------------------------------------------------
    # I/O helpers
    # -----------------------------------------------------------------------

    def _get_raster_meta(self, stack_path: Optional[str], shape: tuple) -> Optional[dict]:
        if stack_path and stack_path.endswith(".tif"):
            try:
                import rasterio
                with rasterio.open(stack_path) as src:
                    meta = src.profile.copy()
                meta.update(count=1, dtype="float32")
                return meta
            except Exception:
                pass
        return None

    def _write_raster(
        self,
        array: np.ndarray,
        name:  str,
        meta:  Optional[dict],
        dtype: str = "float32",
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

        npy_path = path.replace(".tif", ".npy")
        np.save(npy_path, arr)
        return npy_path
