"""CHAOST2 dataset manifest, integrity verification, orientation transforms, and episode generator."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.gate_a.runtime import sha256

CHAOS_PIDS_SORTED = (
    "1",
    "2",
    "3",
    "5",
    "8",
    "10",
    "13",
    "15",
    "19",
    "20",
    "21",
    "22",
    "31",
    "32",
    "33",
    "34",
    "36",
    "37",
    "38",
    "39",
)
ORGANS_CHAOST2 = {"LIVER": 1, "RK": 2, "LK": 3, "SPLEEN": 4}
ORGAN_ORDER = ("LIVER", "RK", "LK", "SPLEEN")
FOLDS_CHAOST2 = {
    0: (0, 1, 2, 3, 4),
    1: (4, 5, 6, 7, 8),
    2: (8, 9, 10, 11, 12),
    3: (12, 13, 14, 15, 16),
    4: (0, 16, 17, 18, 19),
}
SUPPORT_IDX_CHAOST2 = 2
SUPPORT_PARTS_CHAOST2 = 3
SPACING_XYZ_CHAOST2 = (1.25, 1.25, 7.70)


def build_chaost2_manifest() -> pd.DataFrame:
    """Build deterministic 80-episode Setting I manifest matching TestDataset 5-fold splits."""
    rows: list[dict] = []
    eid = 0
    for fold in range(5):
        fold_pids = [CHAOS_PIDS_SORTED[idx] for idx in FOLDS_CHAOST2[fold]]
        supp_pid = f"image_{fold_pids[SUPPORT_IDX_CHAOST2]}"
        qry_pids = [f"image_{pid}" for i, pid in enumerate(fold_pids) if i != SUPPORT_IDX_CHAOST2]
        for qpid in qry_pids:
            for organ in ORGAN_ORDER:
                rows.append(
                    {
                        "episode_id": int(eid),
                        "fold": int(fold),
                        "patient_id": qpid,
                        "support_patient": supp_pid,
                        "organ": organ,
                        "label_id": int(ORGANS_CHAOST2[organ]),
                        "spacing_x": float(SPACING_XYZ_CHAOST2[0]),
                        "spacing_y": float(SPACING_XYZ_CHAOST2[1]),
                        "spacing_z": float(SPACING_XYZ_CHAOST2[2]),
                    }
                )
                eid += 1
    df = pd.DataFrame(rows)
    assert len(df) == 80 and df["episode_id"].nunique() == 80
    assert df["patient_id"].nunique() == 15
    assert df.groupby("fold").size().to_dict() == {f: 16 for f in range(5)}
    return df


def verify_chaost2_inputs(data_dir: Path, ckpt_root: Path, prep_manifest: dict) -> dict:
    """Verify SHA256 hashes of all 40 NIfTI files and 5 FoB CHAOST2 fold checkpoints."""
    scans = prep_manifest["dataset_prep"]["scans"]
    ckpts = prep_manifest["checkpoints"]
    assert len(scans) == 20, f"Expected 20 CHAOST2 scans in manifest, got {len(scans)}"
    assert len(ckpts) >= 5, f"Expected >=5 CHAOST2 checkpoints in manifest, got {len(ckpts)}"

    nifti_verified = 0
    for scan in scans:
        img_p = data_dir / scan["image_file"]
        lbl_p = data_dir / scan["label_file"]
        assert img_p.is_file(), f"Missing NIfTI image: {img_p}"
        assert lbl_p.is_file(), f"Missing NIfTI label: {lbl_p}"
        assert sha256(img_p) == scan["image_sha256"], f"SHA256 mismatch on {img_p}"
        assert sha256(lbl_p) == scan["label_sha256"], f"SHA256 mismatch on {lbl_p}"
        nifti_verified += 2

    ckpt_paths: dict[int, Path] = {}
    ckpt_sha: dict[int, str] = {}
    for fold in range(5):
        suffix = f"FSMIS_train_CHAOST2_cv{fold}/ckpt.pth"
        entry = next((c for c in ckpts if c["rel_path"].endswith(suffix)), None)
        assert entry is not None, f"Missing manifest entry for {suffix}"
        p = ckpt_root / entry["rel_path"]
        assert p.is_file(), f"Missing checkpoint file: {p}"
        h = sha256(p)
        assert h == entry["sha256"], f"Checkpoint SHA256 mismatch on {p}: {h} != {entry['sha256']}"
        ckpt_paths[fold] = p
        ckpt_sha[fold] = h

    return {
        "nifti_files_verified": int(nifti_verified),
        "checkpoints_verified": len(ckpt_paths),
        "checkpoint_rel_paths": {str(f): str(p.relative_to(ckpt_root)).replace("\\", "/") for f, p in ckpt_paths.items()},
        "checkpoint_sha256": {str(f): h for f, h in ckpt_sha.items()},
    }


@lru_cache(maxsize=64)
def _read_nii_raw(path_str: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    import SimpleITK as sitk

    itk = sitk.ReadImage(path_str)
    arr = sitk.GetArrayFromImage(itk)
    sp = tuple(float(x) for x in itk.GetSpacing())
    return arr, sp


def load_chaost2_volume(
    data_dir: Path,
    stem: str,
    is_label: bool,
    orientation: str = "identity",
) -> tuple[np.ndarray, np.ndarray]:
    """Load CHAOST2 NIfTI image or label under 'identity' or 'flip_y' orientation."""
    if orientation not in ("identity", "flip_y"):
        raise ValueError(f"Unsupported CHAOST2 orientation: {orientation}")
    path = Path(data_dir) / f"{stem}.nii.gz"
    if not path.is_file():
        path = Path(data_dir) / f"{stem}.nii"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CHAOST2 NIfTI: {stem} in {data_dir}")

    arr_raw, sp = _read_nii_raw(str(path))
    arr = arr_raw.copy()
    if arr.shape[-2:] != (256, 256):
        raise ValueError(f"Expected (..., 256, 256) CHAOST2 shape, got {arr.shape} for {path.name}")
    if orientation == "flip_y":
        arr = np.flip(arr, axis=1).copy()

    if is_label:
        return arr.astype(np.int16), np.asarray(sp, dtype=np.float64)

    arr_f = arr.astype(np.float64)
    std = float(arr_f.std())
    if std <= 0.0:
        raise ValueError(f"Zero variance in CHAOST2 scan {path.name}")
    norm = ((arr_f - float(arr_f.mean())) / std).astype(np.float32)
    return norm, np.asarray(sp, dtype=np.float64)


def episode_inputs_chaost2(
    row: dict,
    data_dir: Path,
    orientation: str = "identity",
    support_parts: int = SUPPORT_PARTS_CHAOST2,
):
    """Yield per-slice query/support dicts matching upstream FoB TestDataset for CHAOST2."""
    label_id = int(ORGANS_CHAOST2[str(row["organ"])])
    qstem = str(row["patient_id"])
    sstem = str(row["support_patient"])
    image, spacing = load_chaost2_volume(data_dir, qstem, is_label=False, orientation=orientation)
    qlabel, label_spacing = load_chaost2_volume(data_dir, qstem.replace("image_", "label_"), is_label=True, orientation=orientation)
    np.testing.assert_allclose(spacing, label_spacing, atol=1e-5, rtol=0)
    np.testing.assert_allclose(spacing, [row["spacing_x"], row["spacing_y"], row["spacing_z"]], atol=1e-4, rtol=0)

    native_z = np.flatnonzero((qlabel == label_id).any(axis=(1, 2)))
    assert len(native_z) > 0, f"Empty target label slice set for {row}"

    simage, _ = load_chaost2_volume(data_dir, sstem, is_label=False, orientation=orientation)
    slabel, _ = load_chaost2_volume(data_dir, sstem.replace("image_", "label_"), is_label=True, orientation=orientation)
    sz = np.flatnonzero((slabel == label_id).any(axis=(1, 2)))
    assert len(sz) > 0, f"Empty support label slice set for {sstem} organ={row['organ']}"

    n = int(support_parts)
    if n == 1:
        pcts = np.array([0.5])
    else:
        pcts = np.array([1.0 / (2 * n) + (1.0 - 1.0 / n) / (n - 1) * i for i in range(n)])
    support_z = sz[(pcts * len(sz)).astype(int)]
    cuts = np.linspace(0, len(native_z), n + 1).astype(int)

    for j, z in enumerate(native_z):
        part = next(k for k in range(n) if cuts[k] <= j < cuts[k + 1])
        yield {
            "episode_id": int(row["episode_id"]),
            "fold": int(row["fold"]),
            "patient_id": qstem,
            "organ": str(row["organ"]),
            "compact_z": int(j),
            "native_z": int(z),
            "active_slices": int(len(native_z)),
            "support_patient": sstem,
            "support_part": int(part),
            "support_native_z": int(support_z[part]),
            "spacing_xyz": spacing,
            "query": np.repeat(image[z][None, None], 3, axis=1),
            "supp": np.repeat(simage[support_z[part]][None, None], 3, axis=1),
            "support_mask": (slabel[support_z[part]] == label_id)[None].astype(np.float32),
            "gt": (qlabel[z] == label_id),
            "other": (qlabel[z] > 0) & (qlabel[z] != label_id),
        }
