"""Deterministic CHAOS T2SPIR (CHAOST2) DICOM/PNG preprocessing matching SSL-ALPNet / FoB."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage

CHAOS_MR_PIDS = (
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
TARGET_SPACING_XYZ = (1.25, 1.25, 7.70)
TARGET_CROP_HW = (256, 256)
ALIGN_MODES = ("identity", "flip_y", "flip_z", "flip_y_and_z")
CHAOS_LABEL_BANDS = (
    (0, 10, 0),    # BG
    (50, 80, 1),   # LIVER (nominal 63)
    (110, 145, 2), # RK (nominal 126)
    (170, 205, 3), # LK (nominal 189)
    (235, 255, 4), # SPLEEN (nominal 252)
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_chaos_mr_patients(input_root: Path, expected_pids: tuple[str, ...] = CHAOS_MR_PIDS) -> dict[str, tuple[Path, Path]]:
    """Find T2SPIR DICOM_anon and Ground folders for all expected CHAOS training patients."""
    found: dict[str, tuple[Path, Path]] = {}
    for t2_dir in sorted(input_root.rglob("T2SPIR")):
        if not t2_dir.is_dir():
            continue
        parts_low = [p.lower() for p in t2_dir.parts]
        if any("test_sets" in p for p in parts_low):
            continue
        if len(t2_dir.parts) < 3 or t2_dir.parent.parent.name.upper() != "MR":
            continue
        pid = t2_dir.parent.name
        dcm_dir = t2_dir / "DICOM_anon"
        gnd_dir = t2_dir / "Ground"
        if not dcm_dir.is_dir() or not gnd_dir.is_dir():
            continue
        has_dcm = any(dcm_dir.glob("*.dcm"))
        has_png = any(gnd_dir.glob("*.png"))
        if has_dcm and has_png and pid not in found:
            found[pid] = (dcm_dir, gnd_dir)

    missing = [p for p in expected_pids if p not in found]
    if missing:
        top_tree = sorted(str(p.relative_to(input_root)) for p in input_root.rglob("*") if len(p.relative_to(input_root).parts) <= 4)
        raise FileNotFoundError(
            f"Missing CHAOS MR T2SPIR patients {missing}; found={sorted(found)}; tree_sample={top_tree[:30]}"
        )
    return {p: found[p] for p in expected_pids}


def _trailing_int(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    if m is None:
        raise ValueError(f"Cannot extract slice index from filename: {path.name}")
    return int(m.group(1))


def map_chaos_png_labels(raw_png_vol: np.ndarray) -> np.ndarray:
    """Map CHAOS PNG label intensities {0,63,126,189,252} to class IDs {0,1,2,3,4}."""
    arr = np.asarray(raw_png_vol)
    if arr.ndim == 4:
        arr = arr[..., 0]
    arr = arr.astype(np.int32)
    out = np.full(arr.shape, -1, dtype=np.int16)
    for lo, hi, cls_id in CHAOS_LABEL_BANDS:
        mask = (arr >= lo) & (arr <= hi)
        out[mask] = cls_id
    if (out < 0).any():
        bad = sorted(int(v) for v in np.unique(arr[out < 0]))
        raise ValueError(f"Unexpected CHAOS PNG pixel intensities outside valid bands: {bad}")
    return out


def read_chaos_dicom_and_labels(dcm_dir: Path, gnd_dir: Path) -> tuple[sitk.Image, np.ndarray, bool]:
    """Load 3D DICOM series via SimpleITK GDCM and align PNG filenames to DICOM slice order."""
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dcm_dir))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dcm_dir}")
    best_files: list[str] = []
    for sid in series_ids:
        files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dcm_dir), sid))
        if len(files) > len(best_files):
            best_files = files

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(best_files)
    img_itk = reader.Execute()

    png_by_stem = {p.stem: p for p in gnd_dir.glob("*.png")}
    dcm_stems = [Path(f).stem for f in best_files]
    matched_by_stem = all(st in png_by_stem for st in dcm_stems) and len(png_by_stem) == len(dcm_stems)
    if matched_by_stem:
        ordered_pngs = [png_by_stem[st] for st in dcm_stems]
    else:
        ordered_pngs = sorted(gnd_dir.glob("*.png"), key=_trailing_int)

    if len(ordered_pngs) != img_itk.GetSize()[2]:
        raise ValueError(f"Slice count mismatch in {dcm_dir}: dcm={img_itk.GetSize()[2]} vs png={len(ordered_pngs)}")

    slices = [np.asarray(Image.open(p)) for p in ordered_pngs]
    lbl_vol = map_chaos_png_labels(np.stack(slices, axis=0))
    return img_itk, lbl_vol, matched_by_stem


def apply_flip_mode(vol: np.ndarray, mode: str) -> np.ndarray:
    """Apply candidate (Z, Y) flip alignment to a 3D volume."""
    if mode == "identity":
        return vol
    if mode == "flip_y":
        return np.flip(vol, axis=1).copy()
    if mode == "flip_z":
        return np.flip(vol, axis=0).copy()
    if mode == "flip_y_and_z":
        return np.flip(np.flip(vol, axis=0), axis=1).copy()
    raise ValueError(f"Unknown alignment mode: {mode}")


def verify_and_align_label_volume(img_zyx: np.ndarray, lbl_zyx: np.ndarray) -> tuple[np.ndarray, dict]:
    """Select (Z, Y) label alignment maximizing body containment and MRI edge overlap."""
    img = np.asarray(img_zyx, dtype=np.float64)
    lbl = np.asarray(lbl_zyx, dtype=np.int16)
    if img.shape != lbl.shape:
        raise ValueError(f"Image shape {img.shape} != label shape {lbl.shape}")

    img_pos = img - float(img.min())
    p99 = float(np.percentile(img_pos, 99.5))
    air_thr = max(1e-6, 0.05 * p99)

    z_dim = img_pos.shape[0]
    body = np.zeros_like(img_pos, dtype=bool)
    grad_norm = np.zeros_like(img_pos, dtype=np.float64)
    struct2d = np.ones((3, 3), dtype=bool)
    struct4 = ndimage.generate_binary_structure(2, 1)

    for z in range(z_dim):
        raw_fg = img_pos[z] > air_thr
        if raw_fg.any():
            body_z = ndimage.binary_fill_holes(ndimage.binary_closing(raw_fg, structure=struct2d))
        else:
            body_z = raw_fg
        body[z] = body_z
        gx = ndimage.sobel(img_pos[z], axis=1)
        gy = ndimage.sobel(img_pos[z], axis=0)
        g = np.hypot(gx, gy)
        denom = float(g[body_z].mean()) if body_z.any() else float(g.mean())
        grad_norm[z] = g / max(denom, 1e-6)

    scores: dict[str, dict] = {}
    for mode in ALIGN_MODES:
        cand = apply_flip_mode(lbl, mode)
        fg = cand > 0
        fg_vox = int(fg.sum())
        if fg_vox == 0:
            raise ValueError("Empty foreground label volume during orientation alignment")
        in_body_frac = float((fg & body).sum() / float(fg_vox))

        bnd = np.zeros_like(fg, dtype=bool)
        for z in range(z_dim):
            if not fg[z].any():
                continue
            cz = cand[z]
            for cls_id in (1, 2, 3, 4):
                m = cz == cls_id
                if m.any():
                    bnd[z] |= m & ~ndimage.binary_erosion(m, structure=struct4, border_value=0)
        edge_align = float(grad_norm[bnd].mean()) if bnd.any() else 0.0

        liv_m = cand == 1
        spl_m = cand == 4
        if liv_m.any() and spl_m.any():
            liv_x = float(np.nonzero(liv_m)[2].mean())
            spl_x = float(np.nonzero(spl_m)[2].mean())
            liver_left_of_spleen = bool(liv_x < spl_x)
        else:
            liver_left_of_spleen = True

        composite = 3.0 * in_body_frac + 1.0 * edge_align
        scores[mode] = {
            "in_body_frac": in_body_frac,
            "edge_align": edge_align,
            "composite": composite,
            "liver_left_of_spleen": liver_left_of_spleen,
        }

    ranked = sorted(scores.items(), key=lambda kv: kv[1]["composite"], reverse=True)
    best_mode, best_stats = ranked[0]
    second_mode, second_stats = ranked[1]
    margin = float(best_stats["composite"] - second_stats["composite"])
    aligned = apply_flip_mode(lbl, best_mode)
    audit = {
        "best_mode": best_mode,
        "best_composite": float(best_stats["composite"]),
        "best_in_body_frac": float(best_stats["in_body_frac"]),
        "best_edge_align": float(best_stats["edge_align"]),
        "second_mode": second_mode,
        "second_composite": float(second_stats["composite"]),
        "margin": margin,
        "liver_left_of_spleen": bool(best_stats["liver_left_of_spleen"]),
    }
    return aligned, audit


def _resample_volume_by_spacing(
    mov_obj: sitk.Image,
    new_spacing: tuple[float, float, float],
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(mov_obj.GetDirection())
    resample.SetOutputOrigin(mov_obj.GetOrigin())
    mov_spacing = np.asarray(mov_obj.GetSpacing(), dtype=np.float64)
    out_spacing = np.asarray(new_spacing, dtype=np.float64)
    resample.SetOutputSpacing(tuple(float(x) for x in out_spacing))
    new_size = np.asarray(mov_obj.GetSize(), dtype=np.float64) * (mov_spacing / out_spacing)
    resample.SetSize([int(sz + 1) for sz in new_size])
    return resample.Execute(mov_obj)


def _resample_labels_by_spacing(mov_lb_obj: sitk.Image, new_spacing: tuple[float, float, float]) -> sitk.Image:
    src_mat = sitk.GetArrayFromImage(mov_lb_obj)
    lbvs = np.unique(src_mat)
    out_vol = None
    ref_out = None
    for idx, lbv in enumerate(lbvs):
        curr = (src_mat == lbv).astype(np.float32)
        curr_obj = sitk.GetImageFromArray(curr)
        curr_obj.CopyInformation(mov_lb_obj)
        tar_obj = _resample_volume_by_spacing(curr_obj, new_spacing, sitk.sitkLinear)
        tar_mat = np.rint(sitk.GetArrayFromImage(tar_obj)) * float(lbv)
        if idx == 0:
            out_vol = tar_mat
        else:
            out_vol[tar_mat == float(lbv)] = float(lbv)
        ref_out = tar_obj
    out_obj = sitk.GetImageFromArray(out_vol.astype(np.int16))
    out_obj.SetSpacing(ref_out.GetSpacing())
    out_obj.SetOrigin(ref_out.GetOrigin())
    out_obj.SetDirection(ref_out.GetDirection())
    return out_obj


def _crop_2d_center(vol_zyx: np.ndarray, crop_hw: tuple[int, int], padval: float) -> np.ndarray:
    ori = vol_zyx.transpose(1, 2, 0)  # H, W, Z matching image_normalize.ipynb
    ch, cw = crop_hw
    ctr_h, ctr_w = ori.shape[0] // 2, ori.shape[1] // 2
    expand = (ch + 1, cw + 1, ori.shape[2])
    patch = np.ones(expand, dtype=ori.dtype) * padval
    half_h, half_w = expand[0] // 2, expand[1] // 2
    b_start_h = min(half_h, ctr_h)
    b_end_h = min(half_h, ori.shape[0] - ctr_h)
    b_start_w = min(half_w, ctr_w)
    b_end_w = min(half_w, ori.shape[1] - ctr_w)
    patch[half_h - b_start_h : half_h + b_end_h, half_w - b_start_w : half_w + b_end_w, :] = ori[
        ctr_h - b_start_h : ctr_h + b_end_h,
        ctr_w - b_start_w : ctr_w + b_end_w,
        :,
    ]
    cropped = patch[0:ch, 0:cw, :]
    return cropped.transpose(2, 0, 1)


def normalize_and_crop_scan(
    img_itk: sitk.Image,
    lbl_zyx: np.ndarray,
    new_spacing: tuple[float, float, float] = TARGET_SPACING_XYZ,
    crop_hw: tuple[int, int] = TARGET_CROP_HW,
    hist_cut_top_pct: float = 0.5,
) -> tuple[sitk.Image, sitk.Image]:
    """Apply top-0.5% intensity clip, spacing resampling, and 256x256 ROI center crop."""
    raw_arr = sitk.GetArrayFromImage(img_itk).astype(np.float64)
    hir = float(np.percentile(raw_arr, 100.0 - hist_cut_top_pct))
    clipped = np.where(raw_arr > hir, hir, raw_arr).astype(np.float32)

    his_img = sitk.GetImageFromArray(clipped)
    his_img.CopyInformation(img_itk)
    res_img = _resample_volume_by_spacing(his_img, new_spacing, sitk.sitkLinear)

    lbl_obj = sitk.GetImageFromArray(np.asarray(lbl_zyx, dtype=np.int16))
    lbl_obj.CopyInformation(img_itk)
    res_lbl = _resample_labels_by_spacing(lbl_obj, new_spacing)

    res_img_arr = sitk.GetArrayFromImage(res_img)
    crop_img_arr = _crop_2d_center(res_img_arr, crop_hw, padval=float(res_img_arr.min())).astype(np.float32)
    out_img = sitk.GetImageFromArray(crop_img_arr)
    out_img.SetSpacing(res_img.GetSpacing())
    out_img.SetOrigin(res_img.GetOrigin())
    out_img.SetDirection(res_img.GetDirection())

    res_lbl_arr = sitk.GetArrayFromImage(res_lbl)
    crop_lbl_arr = _crop_2d_center(res_lbl_arr, crop_hw, padval=0).astype(np.int16)
    out_lbl = sitk.GetImageFromArray(crop_lbl_arr)
    out_lbl.SetSpacing(res_img.GetSpacing())
    out_lbl.SetOrigin(res_img.GetOrigin())
    out_lbl.SetDirection(res_img.GetDirection())
    return out_img, out_lbl


def prepare_chaost2_dataset(
    input_root: Path,
    output_dir: Path,
    expected_pids: tuple[str, ...] = CHAOS_MR_PIDS,
) -> dict:
    """Convert raw CHAOS T2SPIR DICOM/PNG scans into normalized NIfTIs, classmap, and audit logs."""
    patients = discover_chaos_mr_patients(input_root, expected_pids=expected_pids)
    norm_dir = output_dir / "chaos_MR_T2_normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    alignment_rows: list[dict] = []
    manifest_scans: list[dict] = []
    label_names = ("BG", "LIVER", "RK", "LK", "SPLEEN")
    classmap: dict[str, dict[str, list[int]]] = {name: {} for name in label_names}

    for pid in sorted(patients.keys(), key=lambda x: int(x)):
        dcm_dir, gnd_dir = patients[pid]
        img_itk, lbl_raw, matched_by_stem = read_chaos_dicom_and_labels(dcm_dir, gnd_dir)
        img_arr = sitk.GetArrayFromImage(img_itk)
        lbl_aligned, align_audit = verify_and_align_label_volume(img_arr, lbl_raw)

        out_img, out_lbl = normalize_and_crop_scan(img_itk, lbl_aligned)
        out_img_path = norm_dir / f"image_{pid}.nii.gz"
        out_lbl_path = norm_dir / f"label_{pid}.nii.gz"
        sitk.WriteImage(out_img, str(out_img_path), True)
        sitk.WriteImage(out_lbl, str(out_lbl_path), True)

        lbl_final = sitk.GetArrayFromImage(out_lbl)
        for cls_id, name in enumerate(label_names):
            classmap[name][str(pid)] = [
                int(slc)
                for slc in range(lbl_final.shape[0])
                if int((lbl_final[slc] == cls_id).sum()) >= 1
            ]

        organ_vox = {name: int((lbl_final == cid).sum()) for cid, name in enumerate(label_names) if cid > 0}
        alignment_rows.append(
            {
                "pid": str(pid),
                "matched_by_stem": bool(matched_by_stem),
                "raw_shape_zyx": list(img_arr.shape),
                "out_shape_zyx": list(lbl_final.shape),
                **align_audit,
            }
        )
        manifest_scans.append(
            {
                "pid": str(pid),
                "image_file": out_img_path.name,
                "label_file": out_lbl_path.name,
                "image_sha256": sha256_file(out_img_path),
                "label_sha256": sha256_file(out_lbl_path),
                "shape_zyx": list(lbl_final.shape),
                "spacing_xyz": [float(x) for x in out_img.GetSpacing()],
                "organ_voxels": organ_vox,
                "best_align_mode": align_audit["best_mode"],
            }
        )

    audit_path = output_dir / "alignment_audit.csv"
    fieldnames = list(alignment_rows[0].keys())
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(alignment_rows)

    modes = {r["best_mode"] for r in alignment_rows}
    if len(modes) != 1:
        raise RuntimeError(f"Inconsistent orientation modes detected across CHAOS patients: {alignment_rows}")
    if min(r["best_in_body_frac"] for r in alignment_rows) < 0.90:
        raise RuntimeError(f"Low body-containment fraction after alignment: {alignment_rows}")
    if not all(r["liver_left_of_spleen"] for r in alignment_rows):
        raise RuntimeError(f"Anatomical L/R check failed on aligned volume: {alignment_rows}")

    (norm_dir / "classmap_1.json").write_text(json.dumps(classmap), encoding="utf-8")

    summary = {
        "patients": len(manifest_scans),
        "consistent_align_mode": next(iter(modes)),
        "min_in_body_frac": float(min(r["best_in_body_frac"] for r in alignment_rows)),
        "min_margin": float(min(r["margin"] for r in alignment_rows)),
        "scans": manifest_scans,
    }
    return summary
