"""Package pure source + Step 9 lock fingerprints into a clean Kaggle GPU submission."""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STEP9_LOCK = ROOT / "audit/step9_20261024T034851Z/protocol_lock"
STEP9_PILOT_DIR = ROOT / "results/step9_controls_20261024T034851Z/manual_verified_20261024T133636Z/pilot"

OPTION_A_SOURCES = (
    "experiments/option_a_reassessment_v1/__init__.py",
    "experiments/option_a_reassessment_v1/selectors.py",
    "experiments/option_a_reassessment_v1/metrics.py",
    "experiments/option_a_reassessment_v1/runner.py",
    "experiments/option_a_reassessment_v1/remote.py",
    "experiments/option_a_reassessment_v1/package.py",
    "experiments/option_a_reassessment_v1/verify_local.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def build_bundle(run_dir: Path, kernel_id: str) -> dict:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = run_dir / "submission"
    if sub_dir.exists() and any(sub_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty submission dir: {sub_dir}")
    sub_dir.mkdir(parents=True, exist_ok=True)

    step9_cfg = json.loads((STEP9_LOCK / "configuration.json").read_text(encoding="utf-8"))
    frozen_code = step9_cfg["code_sha256"]
    assert len(frozen_code) == 55, f"Expected 55 Step 9 frozen source hashes, got {len(frozen_code)}"

    all_code_hashes: dict[str, str] = {}
    bundle_io = io.BytesIO()
    with zipfile.ZipFile(bundle_io, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, expected_h in sorted(frozen_code.items()):
            src = ROOT / rel
            assert src.is_file(), f"Missing frozen source file: {rel}"
            actual_h = sha256_file(src)
            assert actual_h == expected_h, f"Step 9 frozen hash mismatch on {rel}: {actual_h} != {expected_h}"
            data = src.read_bytes()
            zf.writestr(rel, data)
            all_code_hashes[rel] = actual_h

        for rel in OPTION_A_SOURCES:
            src = ROOT / rel
            assert src.is_file(), f"Missing Option A file: {rel}"
            data = src.read_bytes()
            h = sha256_bytes(data)
            zf.writestr(rel, data)
            all_code_hashes[rel] = h

        lock_cfg = dict(step9_cfg)
        lock_cfg["version"] = "option_a_reassessment_v1"
        lock_cfg["locked_utc"] = datetime.now(timezone.utc).isoformat()
        lock_cfg["option_a_budget_gpu_hours"] = 2.5
        lock_cfg["option_a_pilot_repeats"] = 1
        lock_cfg["code_sha256"] = all_code_hashes
        zf.writestr("protocol_lock/configuration.json", json.dumps(lock_cfg, indent=2).encode("utf-8"))

        for fname in (
            "paired_episode_manifest.csv",
            "diagnostic_manifest.csv",
            "worker_assignment.csv",
            "step8_fingerprints.json",
            "controls_v1.md",
        ):
            p = STEP9_LOCK / fname
            assert p.is_file(), f"Missing Step 9 lock file: {p}"
            zf.writestr(f"protocol_lock/{fname}", p.read_bytes())

        diag_npzs = sorted((STEP9_LOCK / "baseline_diagnostic").glob("*.npz"))
        assert len(diag_npzs) == 8, f"Expected 8 baseline diagnostic NPZs, found {len(diag_npzs)}"
        for npz in diag_npzs:
            zf.writestr(f"protocol_lock/baseline_diagnostic/{npz.name}", npz.read_bytes())

        ref_keys = (
            "negative_source_xy",
            "positive_source_xy",
            "full10_raw",
            "full10_cc",
            "positive0_raw",
            "positive0_cc",
            "boundary_near2_raw",
            "boundary_near2_cc",
        )
        with (STEP9_LOCK / "diagnostic_manifest.csv").open("r", encoding="utf-8", newline="") as f:
            diag_rows = list(csv.DictReader(f))
        assert len(diag_rows) == 8
        for drow in diag_rows:
            eid = int(drow["episode_id"])
            nz = int(drow["native_z"])
            src_npz = STEP9_PILOT_DIR / f"ep{eid:03d}_repeat1.npz"
            assert src_npz.is_file(), f"Missing Step 9 pilot ref NPZ: {src_npz}"
            with np.load(src_npz) as ref_data:
                ref_buf = io.BytesIO()
                np.savez_compressed(ref_buf, **{k: ref_data[k] for k in ref_keys})
                zf.writestr(f"protocol_lock/step9_pilot_ref/ep{eid:03d}_z{nz}.npz", ref_buf.getvalue())

    bundle_bytes = bundle_io.getvalue()
    bundle_sha = sha256_bytes(bundle_bytes)
    payload_b64 = base64.b64encode(bundle_bytes).decode("ascii")

    # Security and cleanliness verification in a temporary extraction sandbox.
    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zf:
        assert zf.testzip() is None, "Corrupt CRC inside generated bundle"
        names = zf.namelist()
        for name in names:
            low = name.lower()
            assert not low.endswith(".nii") and not low.endswith(".nii.gz"), f"Forbidden image/label in bundle: {name}"
            assert not low.endswith(".pth"), f"Forbidden checkpoint in bundle: {name}"
            assert "kaggle.json" not in low, f"Forbidden credential in bundle: {name}"
            if low.endswith((".py", ".md", ".json", ".csv")):
                txt = zf.read(name).decode("utf-8", errors="replace")
                assert not re.search(r"KGAT_[A-Za-z0-9]{10,}", txt), f"Forbidden Kaggle token pattern in {name}"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            zf.extractall(tmp_root)
            extracted_cfg = json.loads((tmp_root / "protocol_lock/configuration.json").read_text(encoding="utf-8"))
            for rel, expected_h in extracted_cfg["code_sha256"].items():
                assert sha256_file(tmp_root / rel) == expected_h, f"Dry-run hash check failed for {rel}"

    entry_script = f'''import base64
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import zipfile

os.environ["ADAF_OB_ALLOC_START"] = str(time.time())
root = pathlib.Path("/tmp/option_a_locked_source")
root.mkdir(exist_ok=False)
payload = base64.b64decode("{payload_b64}")
assert hashlib.sha256(payload).hexdigest() == "{bundle_sha}"
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.namelist():
        assert (root / member).resolve().is_relative_to(root.resolve())
    archive.extractall(root)

os.chdir(root)
artifacts = pathlib.Path("/tmp/option_a_artifacts")
cmd = [
    sys.executable,
    "-u",
    "-m",
    "experiments.option_a_reassessment_v1.remote",
    "--lock-dir",
    str(root / "protocol_lock"),
    "--output",
    str(artifacts),
]
print("Starting Option A bounded Setting I reassessment on Kaggle T4", flush=True)
completed = subprocess.run(cmd)

for rel in (
    "measurements.csv",
    "both_estimands.csv",
    "paired_contrasts.json",
    "decision_summary.json",
    "prompt_audit.csv",
    "worst_cases.json",
    "remote_status.json",
    "remote_failure.json",
    "hardware_budget.json",
    "environment_kaggle.json",
    "projection_before_full_run.json",
    "pilot.log",
    "worker0.log",
    "worker1.log",
    "pilot/pilot_summary.json",
):
    src = artifacts / rel
    if src.is_file():
        dst = pathlib.Path("/kaggle/working") / src.name
        shutil.copy2(src, dst)

with zipfile.ZipFile("/kaggle/working/option_a_outputs.zip", "x", compression=zipfile.ZIP_DEFLATED) as out_zip:
    if artifacts.exists():
        for path in sorted(artifacts.rglob("*")):
            if path.is_file():
                out_zip.write(path, path.relative_to(artifacts))

elapsed = time.time() - float(os.environ["ADAF_OB_ALLOC_START"])
status = {{
    "entry_exit_code": completed.returncode,
    "observed_wall_seconds": elapsed,
    "source_bundle_sha256": "{bundle_sha}",
}}
pathlib.Path("/kaggle/working/option_a_entry_status.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
if completed.returncode != 0:
    raise RuntimeError(f"Remote execution failed with exit code {{completed.returncode}}")
'''

    ast.parse(entry_script)
    (sub_dir / "option_a_entry.py").write_text(entry_script, encoding="utf-8", newline="\n")

    metadata = {
        "id": kernel_id,
        "title": "AdaFoB Option A Reassess 20261119",
        "code_file": "option_a_entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["nhatphatnguyen/adafob-env"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (sub_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")

    receipt = {
        "kernel_id": kernel_id,
        "submission_dir": str(sub_dir.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "source_count": len(all_code_hashes),
        "frozen_step9_count": len(frozen_code),
        "option_a_count": len(OPTION_A_SOURCES),
        "source_zip_bytes": len(bundle_bytes),
        "source_bundle_sha256": bundle_sha,
        "entry_bytes": len(entry_script.encode("utf-8")),
        "entry_sha256": sha256_bytes(entry_script.encode("utf-8")),
        "dry_run_passed": True,
    }
    (run_dir / "package_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--kernel-id", default="nhatphatnguyen/adafob-option-a-reassess-20261119")
    a = p.parse_args()
    receipt = build_bundle(a.run_dir, a.kernel_id)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
