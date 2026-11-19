"""Package CPU-only Kaggle kernel to prepare CHAOST2 NIfTIs and download FoB checkpoints."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOB_CHAOST2_ZIP_URL = "https://huggingface.co/PrimeBo1/FoB_SAM/resolve/main/exps_train_on_CHAOST2_FSMIS_FoB.zip"

PREP_SOURCES = (
    "experiments/data_prep/__init__.py",
    "experiments/data_prep/chaost2_prep.py",
    "third_party/FoB_SAM/dataloaders/__init__.py",
    "third_party/FoB_SAM/dataloaders/dataset_specifics.py",
    "third_party/FoB_SAM/dataloaders/datasets.py",
    "third_party/FoB_SAM/dataloaders/image_transforms.py",
    "third_party/FoB_SAM/dataloaders/transform.py",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_chaost2_prep_bundle(run_dir: Path, kernel_id: str) -> dict:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = run_dir / "submission"
    if sub_dir.exists() and any(sub_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty submission dir: {sub_dir}")
    sub_dir.mkdir(parents=True, exist_ok=True)

    bundle_io = io.BytesIO()
    with zipfile.ZipFile(bundle_io, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("experiments/__init__.py", b"")
        zf.writestr("third_party/__init__.py", b"")
        zf.writestr("third_party/FoB_SAM/__init__.py", b"")
        for rel in PREP_SOURCES:
            src = ROOT / rel
            if rel.endswith("__init__.py") and not src.is_file():
                zf.writestr(rel, b"")
                continue
            assert src.is_file(), f"Missing source file: {src}"
            zf.writestr(rel, src.read_bytes())

    bundle_bytes = bundle_io.getvalue()
    bundle_sha = _sha256_bytes(bundle_bytes)
    payload_b64 = base64.b64encode(bundle_bytes).decode("ascii")

    entry_script = f'''import base64
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile

root = pathlib.Path("/tmp/chaost2_prep_src")
root.mkdir(exist_ok=False)
payload = base64.b64decode("{payload_b64}")
assert hashlib.sha256(payload).hexdigest() == "{bundle_sha}"
with zipfile.ZipFile(io.BytesIO(payload)) as zf:
    for member in zf.namelist():
        assert (root / member).resolve().is_relative_to(root.resolve())
    zf.extractall(root)

sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "third_party" / "FoB_SAM"))

if importlib.util.find_spec("SimpleITK") is None:
    print("Installing SimpleITK==2.4.1 (--no-deps)...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "SimpleITK==2.4.1"], check=True, timeout=180)

import torch
from experiments.data_prep.chaost2_prep import prepare_chaost2_dataset, sha256_file
from dataloaders.datasets import TestDataset

work = pathlib.Path("/kaggle/working")
chaost2_out = work / "CHAOST2"
prep_summary = prepare_chaost2_dataset(pathlib.Path("/kaggle/input"), chaost2_out)
print(
    json.dumps({{
        "patients": prep_summary["patients"],
        "consistent_align_mode": prep_summary["consistent_align_mode"],
        "min_in_body_frac": prep_summary["min_in_body_frac"],
        "min_margin": prep_summary["min_margin"],
    }}),
    flush=True,
)
shutil.copy2(chaost2_out / "alignment_audit.csv", work / "alignment_audit.csv")

# Instantiate upstream TestDataset across all 5 folds to confirm compatibility
fold_counts = {{}}
for fold in range(5):
    ds = TestDataset({{"dataset": "CHAOST2", "data_dir": str(chaost2_out), "eval_fold": fold, "supp_idx": 2}})
    fold_counts[str(fold)] = {{"support_dir": pathlib.Path(ds.support_dir).name, "query_patients": len(ds)}}
print("TestDataset folds check:", json.dumps(fold_counts), flush=True)

# Download official FoB CHAOST2 checkpoints from HuggingFace
ckpt_zip = pathlib.Path("/tmp/exps_train_on_CHAOST2_FSMIS_FoB.zip")
url = "{FOB_CHAOST2_ZIP_URL}"
print("Downloading official FoB CHAOST2 checkpoints...", flush=True)
req = urllib.request.Request(url, headers={{"User-Agent": "Mozilla/5.0"}})
with urllib.request.urlopen(req, timeout=300) as resp, ckpt_zip.open("wb") as f:
    total = 0
    while True:
        chunk = resp.read(8 * 1024 * 1024)
        if not chunk:
            break
        f.write(chunk)
        total += len(chunk)
        if total % (128 * 1024 * 1024) == 0:
            print(f"Downloaded bytes: {{total}}", flush=True)
print(f"Download complete: {{ckpt_zip.stat().st_size}} bytes", flush=True)

ckpt_root = work / "checkpoints_chaost2"
ckpt_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ckpt_zip, "r") as zf:
    for member in zf.namelist():
        assert (ckpt_root / member).resolve().is_relative_to(ckpt_root.resolve())
    zf.extractall(ckpt_root)
ckpt_zip.unlink()

pths = sorted(ckpt_root.rglob("*.pth"))
assert len(pths) >= 5, f"Expected >=5 FoB CHAOST2 checkpoints, found {{[str(p) for p in pths]}}"
ckpt_manifest = []
for pth in pths:
    obj = torch.load(pth, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else {{}}
    ckpt_manifest.append({{
        "rel_path": pth.relative_to(ckpt_root).as_posix(),
        "bytes": int(pth.stat().st_size),
        "sha256": sha256_file(pth),
        "n_keys": int(len(state)) if isinstance(state, dict) else 0,
    }})

manifest = {{
    "status": "COMPLETED",
    "dataset_prep": prep_summary,
    "test_dataset_folds": fold_counts,
    "checkpoints": ckpt_manifest,
    "source_bundle_sha256": "{bundle_sha}",
}}
(work / "prep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({{"status": "COMPLETED", "scans": len(prep_summary["scans"]), "pths": len(ckpt_manifest)}}), flush=True)
'''

    ast.parse(entry_script)
    (sub_dir / "chaost2_prep_entry.py").write_text(entry_script, encoding="utf-8", newline="\n")

    metadata = {
        "id": kernel_id,
        "title": "AdaFoB Prep CHAOST2 20261119",
        "code_file": "chaost2_prep_entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "oumaymabejaoui/chaos-ct-mri-data-downloading",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (sub_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")

    receipt = {
        "kernel_id": kernel_id,
        "submission_dir": str(sub_dir.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "source_zip_bytes": len(bundle_bytes),
        "source_bundle_sha256": bundle_sha,
        "entry_bytes": len(entry_script.encode("utf-8")),
    }
    (run_dir / "package_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--kernel-id", default="nhatphatnguyen/adafob-prep-chaost2-20261119")
    a = p.parse_args()
    receipt = build_chaost2_prep_bundle(a.run_dir, a.kernel_id)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
