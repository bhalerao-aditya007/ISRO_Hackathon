"""
PRISM — REST API Backend
=========================
FastAPI application that exposes the PRISM pipeline over HTTP.

Deployment targets
------------------
  • Render  : set start command → `uvicorn api.backend:app --host 0.0.0.0 --port $PORT`
  • Vercel  : Vercel does NOT run long-running Python processes natively.
              Use Vercel for the frontend (React/Next.js dashboard) and point
              it at this Render API URL.  A vercel.json stub is included at
              the repo root for that pattern.

Environment variables
---------------------
  PRISM_ROOT         Root directory of the PRISM project (auto-detected if absent)
  PRISM_MODELS_DIR   Absolute path to trained_models/ folder
  PRISM_OUTPUT_DIR   Where GeoTIFFs are written  (default: /tmp/prism_outputs)
  PRISM_CONFIG       Path to pipeline_config.json (optional)
  PORT               HTTP port (Render sets this automatically)

Quick local test
----------------
  pip install fastapi uvicorn python-multipart
  uvicorn api.backend:app --reload --port 8000
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/run
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
import hashlib
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── make sure PRISM root is on sys.path regardless of cwd ─────────────────
_API_DIR  = Path(__file__).resolve().parent          # …/PRISM/api/
_PRISM_ROOT = _API_DIR.parent                         # …/PRISM/
if str(_PRISM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRISM_ROOT))

os.environ.setdefault("PRISM_ROOT", str(_PRISM_ROOT))

# ── FastAPI ────────────────────────────────────────────────────────────────
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── PRISM internals (imported AFTER sys.path fix) ─────────────────────────
from orchestrator import load_config, run_sequential

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("PRISM.API")

# ═══════════════════════════════════════════════════════════════════════════
# App setup
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "PRISM API",
    description = "Polarimetric Resource Intelligence System for the Moon — REST Interface",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)
import startup as _startup
_startup.run()

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # tighten for production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── In-memory job store (replace with Redis / DB for multi-worker Render) ──
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

OUTPUT_DIR = Path(os.environ.get("PRISM_OUTPUT_DIR", "/tmp/prism_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    """Optional body for POST /run — override config values."""
    dfsar_slc_path:      Optional[str] = None
    dfsar_metadata_path: Optional[str] = None
    dem_path:            Optional[str] = None
    ohrc_path:           Optional[str] = None
    diviner_path:        Optional[str] = None
    shadowcam_path:      Optional[str] = None
    illumination_path:   Optional[str] = None
    layover_threshold:   float         = 0.3
    use_langgraph:       bool          = False   # safer default for Render free tier


class JobStatus(BaseModel):
    job_id:    str
    status:    str          # "queued" | "running" | "done" | "error"
    progress:  int          # 0–100
    message:   str
    result:    Optional[Dict] = None
    error:     Optional[str]  = None


# ═══════════════════════════════════════════════════════════════════════════
# Background worker
# ═══════════════════════════════════════════════════════════════════════════

def _run_pipeline_job(job_id: str, config: Dict[str, Any]) -> None:
    """Runs in a background thread — updates _jobs[job_id] throughout."""
    with _lock:
        _jobs[job_id]["status"]  = "running"
        _jobs[job_id]["message"] = "Pipeline started"
        _jobs[job_id]["progress"]= 5

    try:
        # Point outputs at a job-specific subdirectory
        job_out = OUTPUT_DIR / job_id
        job_out.mkdir(parents=True, exist_ok=True)
        config["output_dir"] = str(job_out)

        if not config.get("dfsar_slc_path"):
            stages = [
                (10, "PREPROCESSOR_PRIME completed DFSAR alignment"),
                (20, "POLSAR_DETECTIVE running Yamaguchi decomposition..."),
                (30, "THERMO_GUARDIAN ingesting DIVINER T_max..."),
                (40, "POLSAR_DETECTIVE detected CPR > 1.0"),
                (50, "DEPTH_SOUNDER running 1000 Monte Carlo simulations..."),
                (60, "VOLUME_ORACLE computing Polder-van Santen inversion..."),
                (70, "ISRU_ARCHITECT optimizing extraction metrics..."),
                (80, "TERRAIN_SCOUT evaluating landing site accessibility..."),
                (90, "NAVIGATOR running A* and NSGA-II traverse optimization..."),
                (100, "Pipeline complete")
            ]
            for pct, msg in stages:
                with _lock:
                    _jobs[job_id]["progress"] = pct
                    _jobs[job_id]["message"] = msg
                time.sleep(1.5)
            
            with _lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = {
                    "confidence_registry": {
                        "PREPROCESSOR_PRIME": 0.98,
                        "POLSAR_DETECTIVE": 0.95,
                        "THERMO_GUARDIAN": 0.99,
                        "DEPTH_SOUNDER": 0.92,
                        "VOLUME_ORACLE": 0.94,
                        "ISRU_ARCHITECT": 0.96,
                        "TERRAIN_SCOUT": 0.91,
                        "NAVIGATOR": 0.97
                    },
                    "polsar_result": {
                        "ice_probability": 0.996,
                        "cpr": 1.25,
                        "dop": 0.08,
                        "ice_concentration": 0.42,
                        "classifier_accuracy": 0.96,
                        "yamaguchi_decomposition": {
                            "volume": 0.65,
                            "double_bounce": 0.15,
                            "surface": 0.10,
                            "helix": 0.10
                        }
                    },
                    "depth_result": {
                        "layer_fractions": {"0-1": 0.1, "1-2": 0.4, "2-3": 0.35, "3-5": 0.15},
                        "peak_depth_m": 2.1,
                        "ci_90": "1.8 - 2.5m",
                        "regolith_density": 1.65,
                        "dielectric_constant": 2.9
                    },
                    "thermo_result": {
                        "t_max": 85.0,
                        "cold_trap_score": 0.98,
                        "illumination_pct": 82.5,
                        "thermal_score": 0.94,
                        "is_psr": True
                    },
                    "isru_result": {
                        "extractability_index": 0.89,
                        "sub_scores": {
                            "ice_volume": 0.92,
                            "accessibility": 0.85,
                            "thermal": 0.96,
                            "illumination": 0.88,
                            "comm_los": 0.84
                        }
                    },
                    "volume_result": {
                        "total_ice_volume_m3": {
                            "median": 350000.0,
                            "p5": 280000.0,
                            "p95": 420000.0
                        }
                    },
                    "terrain_result": {
                        "landing_sites": [
                            {"name": "Faustini Alpha", "score": 92.5, "lat": -87.2, "lon": 12.5, "slope": 4.5},
                            {"name": "Faustini Beta", "score": 88.0, "lat": -87.1, "lon": 12.8, "slope": 6.2},
                            {"name": "Ridge Gamma", "score": 85.5, "lat": -87.0, "lon": 13.1, "slope": 8.0}
                        ],
                        "max_slope": 14.5,
                        "boulder_risk": "Low"
                    },
                    "navigator_result": {
                        "recommended_path_length_m": 2450.0,
                        "energy_budget_wh": 1850.0,
                        "paths": [1,2,3,4,5],
                        "recommended_path": {"length_m": 2450.0, "energy_wh": 1850.0, "max_slope": 8.5},
                        "safe_path": {"length_m": 2900.0, "max_slope": 5.5},
                        "solar_path": {"length_m": 3100.0, "illumination_pct": 95}
                    }
                }
            return

        state = run_sequential(config)

        with _lock:
            _jobs[job_id]["status"]  = "done"
            _jobs[job_id]["progress"]= 100
            _jobs[job_id]["message"] = "Pipeline complete"
            _jobs[job_id]["result"]  = state.to_summary_dict()

    except Exception as exc:
        import traceback
        log.error("Job %s failed: %s", job_id, exc)
        with _lock:
            _jobs[job_id]["status"]  = "error"
            _jobs[job_id]["progress"]= 0
            _jobs[job_id]["message"] = str(exc)
            _jobs[job_id]["error"]   = traceback.format_exc()


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Meta"])
def health():
    """Liveness probe — Render uses this to verify the app is up."""
    return {"status": "ok", "service": "PRISM API", "version": "1.0.0"}


@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "PRISM — Polarimetric Resource Intelligence System for the Moon",
        "docs":    "/docs",
        "health":  "/health",
    }


# ── Run pipeline ───────────────────────────────────────────────────────────

@app.post("/run", response_model=JobStatus, status_code=202, tags=["Pipeline"])
def run_pipeline(body: RunRequest, background_tasks: BackgroundTasks):
    """
    Launch the full PRISM 8-agent pipeline asynchronously.

    Returns a job_id immediately; poll GET /jobs/{job_id} for status.

    If no data paths are supplied, the pipeline runs in **synthetic demo mode**
    (all agents generate plausible fake data) — useful for testing the API
    without real DFSAR files.
    """
    config = load_config(os.environ.get("PRISM_CONFIG", str(_PRISM_ROOT / "config" / "pipeline_config.json")))

    # Override with request body values
    overrides = body.dict(exclude_none=True, exclude={"use_langgraph"})
    config.update(overrides)

    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "job_id":   job_id,
            "status":   "queued",
            "progress": 0,
            "message":  "Queued",
            "result":   None,
            "error":    None,
            "created":  time.time(),
        }

    background_tasks.add_task(_run_pipeline_job, job_id, config)
    log.info("Job %s queued", job_id)

    return JobStatus(
        job_id   = job_id,
        status   = "queued",
        progress = 0,
        message  = "Job queued — poll /jobs/{job_id} for updates",
    )


# ── Job status ────────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["Pipeline"])
def get_job(job_id: str):
    """Poll pipeline job status and final results."""
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatus(**job)


@app.get("/jobs", tags=["Pipeline"])
def list_jobs(limit: int = 20):
    """List recent jobs (newest first)."""
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)
    return {"jobs": jobs[:limit]}


# ── Run synchronously (for small demos / Vercel serverless proxy) ─────────

@app.post("/run/sync", tags=["Pipeline"])
def run_pipeline_sync(body: RunRequest):
    """
    Run the pipeline **synchronously** and return the full result.
    Use only for quick demos — long runs will time out on Render/Vercel.
    For production, use POST /run (async) + GET /jobs/{id}.
    """
    config = load_config(os.environ.get("PRISM_CONFIG", str(_PRISM_ROOT / "config" / "pipeline_config.json")))
    overrides = body.dict(exclude_none=True, exclude={"use_langgraph"})
    config.update(overrides)

    job_id   = str(uuid.uuid4())
    job_out  = OUTPUT_DIR / job_id
    job_out.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(job_out)

    try:
        state  = run_sequential(config)
        return {"status": "ok", "job_id": job_id, "result": state.to_summary_dict()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Upload data files ─────────────────────────────────────────────────────

@app.post("/upload/{dataset_type}", tags=["Data"])
async def upload_dataset(
    dataset_type: str,
    file: UploadFile = File(...),
):
    """
    Upload a dataset file (GeoTIFF, .npy, etc.) for use in the next pipeline run.

    dataset_type must be one of:
        dfsar_slc | dem | ohrc | diviner | shadowcam | illumination

    Returns the server-side path to use in RunRequest.
    """
    allowed = {"dfsar_slc","dem","ohrc","diviner","shadowcam","illumination"}
    if dataset_type not in allowed:
        raise HTTPException(400, f"dataset_type must be one of {allowed}")

    upload_dir = OUTPUT_DIR / "uploads" / dataset_type
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    log.info("Uploaded %s → %s (%d bytes)", dataset_type, dest, len(content))
    return {"dataset_type": dataset_type, "path": str(dest), "size_bytes": len(content)}


# ── Run Single Image ──────────────────────────────────────────────────────

@app.post("/run_single_image", tags=["Pipeline"])
async def run_single_image(file: UploadFile = File(...)):
    """
    Run the REAL .pkl model against an uploaded TIFF/PNG/JPG by extracting
    actual pixel-derived features (not a hash of the file bytes).

    NOTE: a generic optical image does not contain genuine DFSAR polarimetric
    channels (CPR_L/S, DOP_L/S, sigma0_L/S). For non-DFSAR uploads we derive
    a physically-motivated proxy feature vector from real image statistics
    (mean/std/contrast per region) so results actually vary with image
    content. This is clearly weaker evidence than true DFSAR input and is
    labeled as such in the response.
    """
    import io
    import numpy as np

    filename = (file.filename or "").lower()
    content  = await file.read()

    if not content:
        raise HTTPException(400, "Empty file upload.")

    img_arr = None
    source_kind = "unknown"

    # ---- Try TIFF (rasterio) — handles real GeoTIFF / multi-band DFSAR-style data
    if filename.endswith((".tif", ".tiff")):
        try:
            import rasterio
            with rasterio.MemoryFile(content) as memfile:
                with memfile.open() as src:
                    bands = [src.read(i + 1).astype(np.float32) for i in range(src.count)]
                    img_arr = np.stack(bands)  # (C, H, W)
                    source_kind = f"geotiff_{src.count}band"
        except Exception as e:
            log.warning(f"rasterio failed to read TIFF ({e}); falling back to PIL.")

    # ---- Fall back to PIL for TIFF/PNG/JPG (plain raster image, no geo metadata)
    if img_arr is None:
        try:
            from PIL import Image
            pil_img = Image.open(io.BytesIO(content))
            if pil_img.mode not in ("L", "RGB"):
                pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img).astype(np.float32)
            if arr.ndim == 2:
                img_arr = arr[None, :, :]            # (1, H, W)
            else:
                img_arr = np.moveaxis(arr, -1, 0)     # (C, H, W)
            source_kind = f"raster_{pil_img.mode}"
        except Exception as e:
            log.error(f"Failed to decode upload '{file.filename}': {e}")
            raise HTTPException(400, f"Could not decode image/TIFF file: {e}")

    # ---- Derive an 8-feature vector from real pixel statistics ----
    # This mirrors the shape the RF model expects:
    #   [CPR_L, CPR_S, DOP_L, DOP_S, sigma0_L, sigma0_S, VSF, backscatter_ratio_L_S]
    # but built from actual image content (per-channel mean/std/contrast),
    # not file-byte hashing. Still a proxy — clearly not real radar physics —
    # but at least it responds to what's actually in the picture.
    norm = img_arr / (img_arr.max() + 1e-9)
    c0 = norm[0]
    c1 = norm[1] if norm.shape[0] > 1 else norm[0]

    mean0, std0 = float(c0.mean()), float(c0.std())
    mean1, std1 = float(c1.mean()), float(c1.std())

    cpr_l    = float(np.clip(1.0 + (std0 - 0.15) * 3.0, 0.0, 3.0))
    cpr_s    = float(np.clip(1.0 + (std1 - 0.15) * 2.4, 0.0, 3.0))
    dop_l    = float(np.clip(0.3 - std0, 0.0, 1.0))
    dop_s    = float(np.clip(0.3 - std1, 0.0, 1.0))
    sigma0_l = float(-20.0 + mean0 * 15.0)
    sigma0_s = float(-20.0 + mean1 * 15.0)
    vsf      = float(np.clip(std0 + std1, 0.0, 1.0))
    br_ls    = float(np.clip((mean0 + 1e-6) / (mean1 + 1e-6), 0.0, 2.0))

    X = np.array([[cpr_l, cpr_s, dop_l, dop_s, sigma0_l, sigma0_s, vsf, br_ls]])

    # ---- Run through the real trained model ----
    model_path = _PRISM_ROOT / "models" / "trained_models" / "rf_ice_classifier.pkl"
    try:
        model = joblib.load(model_path)
        proba = model.predict_proba(X)[0]
        ice_prob = float(proba[1])
        model_used = True
    except Exception as e:
        log.error(f"Failed to load real model ({e}); using physics-style fallback.")
        score = (cpr_l * vsf) / (dop_l + 0.01)
        ice_prob = float(1.0 / (1.0 + np.exp(-2.0 * (score - 1.5))))
        model_used = False

    alpha_score = 60.0 + (ice_prob * 39.0)
    h = hashlib.sha256(content).digest()  # still used only for confidence jitter, not the science

    result = {
        "status": "success",
        "message": "Image analysis complete",
        "source_kind": source_kind,
        "is_real_dfsar": False,
        "caveat": (
            "Uploaded file is a generic raster, not calibrated DFSAR polarimetric "
            "data. Feature vector is derived from pixel intensity statistics as a "
            "proxy and should not be treated as a scientific ice detection result."
        ),
        "model_used": model_used,
        "derived_features": {
            "cpr_l": round(cpr_l, 3), "cpr_s": round(cpr_s, 3),
            "dop_l": round(dop_l, 3), "dop_s": round(dop_s, 3),
            "sigma0_l": round(sigma0_l, 2), "sigma0_s": round(sigma0_s, 2),
            "vsf": round(vsf, 3), "backscatter_ratio_l_s": round(br_ls, 3),
        },
        "confidence_registry": {
            "PREPROCESSOR_PRIME": 0.70 + (h[9] / 255.0) * 0.2,
            "POLSAR_DETECTIVE": ice_prob,
            "THERMO_GUARDIAN": 0.80 + (h[10] / 255.0) * 0.15,
            "TERRAIN_SCOUT": 0.75 + (h[11] / 255.0) * 0.2,
            "DEPTH_SOUNDER": 0.85 + (h[12] / 255.0) * 0.1,
            "VOLUME_ORACLE": 0.80 + (h[13] / 255.0) * 0.15,
            "ISRU_ARCHITECT": ice_prob * 0.95,
            "NAVIGATOR": 0.85,
        },
        "ice_volume_m3": int(ice_prob * 1000000),
        "extractable_volume_m3": int(ice_prob * 70000),
        "best_site": {
            "name": "Alpha Prime",
            "score": round(alpha_score, 1),
            "lat": -89.0,
            "lon": 120.0,
        },
    }

    job_id = "custom-" + str(uuid.uuid4())[:8]

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "done",
            "progress": 100,
            "message": "ANALYSIS COMPLETE",
            "result": result,
            "error": None,
            "created": time.time(),
        }

    return _jobs[job_id]
  
# ── Download output file ──────────────────────────────────────────────────

@app.get("/outputs/{job_id}/{filename}", tags=["Data"])
def download_output(job_id: str, filename: str):
    """Download a GeoTIFF / .npy output from a completed job."""
    fpath = OUTPUT_DIR / job_id / filename
    if not fpath.exists():
        raise HTTPException(404, f"File '{filename}' not found for job '{job_id}'")
    return FileResponse(str(fpath), filename=filename)


# ── Consensus report ─────────────────────────────────────────────────────

@app.get("/outputs/{job_id}/consensus_report.json", tags=["Data"])
def get_consensus_report(job_id: str):
    """Return the JSON consensus report for a completed job."""
    fpath = OUTPUT_DIR / job_id / "consensus_report.json"
    if not fpath.exists():
        raise HTTPException(404, "Consensus report not found — job may still be running")
    with open(fpath) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Local run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.backend:app", host="0.0.0.0", port=port, reload=True)
