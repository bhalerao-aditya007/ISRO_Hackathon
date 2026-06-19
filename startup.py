"""
startup.py — Downloads PRISM data and models from HuggingFace Hub into /data/.
Called once at container startup. Skips files already present (fast restarts).
Set HF_TOKEN in HF Spaces secrets for private repos.
"""
import os
import logging
from pathlib import Path

log = logging.getLogger("PRISM.STARTUP")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "YOUR_HF_USERNAME/prism-lunar-data")
HF_MODEL_REPO   = os.environ.get("HF_MODEL_REPO",   "YOUR_HF_USERNAME/prism-models")
DATA_DIR        = Path(os.environ.get("DATA_CACHE_DIR", "/data/prism_data"))
MODELS_DIR      = Path(os.environ.get("PRISM_MODELS_DIR", "/data/prism_models"))


def download_if_missing(repo_id: str, filename: str, local_path: Path, repo_type: str = "dataset"):
    if local_path.exists():
        log.info("Already cached: %s", local_path)
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s/%s ...", repo_id, filename)
    try:
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(local_path.parent),
            token=os.environ.get("HF_TOKEN"),
        )
        # hf_hub_download saves with the filename; rename if needed
        tmp_path = Path(tmp)
        if tmp_path != local_path:
            tmp_path.rename(local_path)
        log.info("Downloaded → %s", local_path)
    except Exception as e:
        log.error("Failed to download %s: %s", filename, e)


def download_folder_if_missing(repo_id: str, folder: str, local_dir: Path, repo_type: str = "dataset"):
    sentinel = local_dir / ".downloaded"
    if sentinel.exists():
        log.info("Folder already cached: %s", local_dir)
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading folder %s/%s ...", repo_id, folder)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=[f"{folder}/*"],
            local_dir=str(local_dir.parent),
            token=os.environ.get("HF_TOKEN"),
        )
        sentinel.touch()
        log.info("Folder downloaded → %s", local_dir)
    except Exception as e:
        log.error("Failed to download folder %s: %s", folder, e)


def run():
    log.info("=== PRISM STARTUP: syncing data and models ===")

    # DFSAR derived TIFs (the big ones — only download once)
    download_folder_if_missing(
        HF_DATASET_REPO, "01_DFSAR",
        DATA_DIR / "01_DFSAR",
    )

    # DEM
    download_if_missing(
        HF_DATASET_REPO,
        "03_LOLA/LDEM_80S_20MPP_ADJ.TIF",
        DATA_DIR / "03_LOLA" / "LDEM_80S_20MPP_ADJ.TIF",
    )

    # DIVINER
    download_if_missing(
        HF_DATASET_REPO,
        "05_DIVINER/polar_south_80_Tmax.grd",
        DATA_DIR / "05_DIVINER" / "polar_south_80_Tmax.grd",
    )

    # Illumination
    download_if_missing(
        HF_DATASET_REPO,
        "04_ILLUMINATION/AVGVISIB_65S_240M.IMG",
        DATA_DIR / "04_ILLUMINATION" / "AVGVISIB_65S_240M.IMG",
    )

    # Models
    for model_file in ["rf_ice_classifier.pkl", "deepmoon_crater_detector.pkl"]:
        download_if_missing(
            HF_MODEL_REPO, model_file,
            MODELS_DIR / model_file,
            repo_type="model",
        )

    log.info("=== PRISM STARTUP complete ===")


if __name__ == "__main__":
    run()
