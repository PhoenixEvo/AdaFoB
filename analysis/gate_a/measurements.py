"""Saved-mask metrics only: actual production AST vs independent physical KD-tree."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree
from .runtime import production_metrics

ORGANS = {"LIVER": 6, "RK": 2, "LK": 3, "SPLEEN": 1}


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(mask.ndim, 1), border_value=0)


def independent_surfaces(pred, gt, spacing_zyx):
    """Symmetric pooled 95th percentile of surface voxel-centre distances."""
    if not pred.any() and not gt.any():
        return 0.0, 0.0, 0.0
    if not pred.any() or not gt.any():
        return float("inf"), float("inf"), float("inf")
    ps, gs = surface(pred), surface(gt)
    pc = np.argwhere(ps) * np.asarray(spacing_zyx)
    gc = np.argwhere(gs) * np.asarray(spacing_zyx)
    p_to_g = cKDTree(gc).query(pc, workers=1)[0]
    g_to_p = cKDTree(pc).query(gc, workers=1)[0]
    return (float(np.percentile(np.concatenate([p_to_g, g_to_p]), 95)),
            float(np.percentile(p_to_g, 95)), float(np.percentile(g_to_p, 95)))


def disjoint_fp(pred, gt, other):
    labels, _ = ndimage.label(pred, structure=ndimage.generate_binary_structure(3, 1))
    hits = np.bincount(labels[gt].ravel(), minlength=int(labels.max())+1)
    no_overlap = hits == 0
    no_overlap[0] = False
    detached = pred & no_overlap[labels]
    fp = pred & ~gt
    other_fp = fp & ~detached & other
    residual = fp & ~detached & ~other
    assert not (detached & other_fp).any()
    assert not (detached & residual).any()
    assert not (other_fp & residual).any()
    assert np.array_equal(detached | other_fp | residual, fp)
    return {"detached_fp": int(detached.sum()), "other_annotated_organ_fp": int(other_fp.sum()),
            "residual_fp": int(residual.sum())}


def episode(row, data_dir):
    metric, _ = production_metrics()
    patient, organ, eid = row["patient_id"], row["organ"], int(row["episode_id"])
    label_path = Path(data_dir) / (patient.replace("image_", "label_") + ".nii.gz")
    itk = sitk.ReadImage(str(label_path))
    label = sitk.GetArrayFromImage(itk)
    spacing = np.asarray([row["spacing_x"], row["spacing_y"], row["spacing_z"]], dtype=float)
    np.testing.assert_allclose(itk.GetSpacing(), spacing, atol=1e-6, rtol=0)
    gt_full = label == ORGANS[organ]
    z = np.flatnonzero(gt_full.any(axis=(1, 2)))
    assert (np.diff(z) == 1).all(), "non-contiguous slab needs physical z reconstruction"
    label = label[z]
    gt = label == ORGANS[organ]
    other = (label > 0) & (label != ORGANS[organ])
    filename = f"ep_{eid:03d}_{patient}_{organ}.npz"
    records = []
    dtg = ndimage.distance_transform_edt(~gt, sampling=spacing[::-1])
    with np.load(Path("results/baselines/predictions")/filename) as b0, np.load(Path("results/step4_baselines/predictions")/filename) as other_arms:
        assert np.array_equal(other_arms["gt"] > 0, gt)
        for arm in ["b0", "b3h", "b3b", "b4h", "b4b"]:
            source = b0 if arm == "b0" else other_arms
            raw, cc = source[arm+"_raw"] > 0, source[arm+"_cc"] > 0
            assert raw.shape == cc.shape == gt.shape
            assert not (cc & ~raw).any()
            removed_tp, removed_fp = int((raw & ~cc & gt).sum()), int((raw & ~cc & ~gt).sum())
            for post, pred in [("raw", raw), ("cc", cc)]:
                dice, production_hd = metric(pred, gt, tuple(spacing))
                independent_dice = 2 * int((pred & gt).sum()) / max(1, int(pred.sum()+gt.sum()))
                independent_hd, directed_p, directed_g = independent_surfaces(pred, gt, spacing[::-1])
                np.testing.assert_allclose(dice, independent_dice, atol=1e-12, rtol=0)
                if np.isfinite(independent_hd):
                    np.testing.assert_allclose(production_hd, independent_hd, atol=1e-6, rtol=0)
                partition = disjoint_fp(pred, gt, other)
                fp, fn, tp = int((pred & ~gt).sum()), int((~pred & gt).sum()), int((pred & gt).sum())
                assert sum(partition.values()) == fp
                # Reconstruct legacy boundary-to-foreground operator for both raw/CC.
                ps, gs = surface(pred), surface(gt)
                if pred.any() and gt.any():
                    dtp = ndimage.distance_transform_edt(~pred, sampling=spacing[::-1])
                    legacy_p = float(np.percentile(dtg[ps], 95))
                    legacy_g = float(np.percentile(dtp[gs], 95))
                else:
                    legacy_p = legacy_g = float("nan")
                records.append({"episode_id": eid, "fold": int(row["fold"]), "patient_id": patient, "organ": organ,
                    "arm": arm, "post": post, "dice": dice, "production_hd95": production_hd,
                    "surface_hd95": independent_hd, "directed_surface_pred_to_gt95": directed_p,
                    "directed_surface_gt_to_pred95": directed_g, "legacy_surface_to_fg_pred95": legacy_p,
                    "legacy_surface_to_fg_gt95": legacy_g, "source_dice": row[f"dice_{arm}_{post}"],
                    "source_hd95": row[f"hd95_{arm}_{post}"], "dice_minus_source": dice-row[f"dice_{arm}_{post}"],
                    "hd95_minus_source": production_hd-row[f"hd95_{arm}_{post}"],
                    "tp": tp, "fp": fp, "fn": fn, **partition,
                    "recall": tp/max(1,int(gt.sum())), "fp_mm3": fp*float(np.prod(spacing)),
                    "fn_mm3": fn*float(np.prod(spacing)), "removed_tp": removed_tp, "removed_fp": removed_fp,
                    "empty_prediction": not pred.any(), "gt_voxels": int(gt.sum()),
                    "metric_status": "finite" if np.isfinite(independent_hd) else "undefined_one_empty"})
    return records


def summaries(df):
    measures = ["dice","surface_hd95","production_hd95","recall","fp_mm3","fn_mm3","detached_fp","other_annotated_organ_fp","residual_fp"]
    rows=[]
    for (arm, post), group in df.groupby(["arm","post"]):
        # numpy mean keeps infinities/NaNs visible, never drops failed cases.
        repeated = group.groupby(["patient_id","organ"])[measures].agg(lambda v:np.mean(v.to_numpy()))
        balanced = repeated.groupby("patient_id").agg(lambda v:np.mean(v.to_numpy()))
        for metric in measures:
            values=group[metric].to_numpy()
            rows.append({"arm":arm,"post":post,"metric":metric,"episode_weighted":np.mean(values),
                         "patient_balanced":np.mean(balanced[metric].to_numpy()),
                         "episode_median":np.median(values),"episode_p90":np.quantile(values,.9),
                         "episode_p95":np.quantile(values,.95),"episodes":len(group),
                         "patients":group.patient_id.nunique(),"nonfinite":int((~np.isfinite(values)).sum())})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--limit-episodes",type=int)
    p.add_argument("--workers",type=int,default=1)
    a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    source=pd.read_csv("results/step4_baselines/step4_baselines_results.csv")
    if a.limit_episodes:
        source=source.head(a.limit_episodes)
    rows=[]
    executor=ProcessPoolExecutor(max_workers=a.workers) if a.workers>1 else None
    inputs=source.to_dict("records")
    results=executor.map(episode,inputs,[a.data_dir]*len(inputs)) if executor else (episode(row,a.data_dir) for row in inputs)
    for batch in results:
        rows.extend(batch)
        if True:
            print(f"saved masks: {len(rows)//10}/{len(source)} episodes, {time.perf_counter()-start:.2f}s",flush=True)
        if (len(rows)//10)%10==0:
            pd.DataFrame(rows).to_csv(a.out_dir/f"checkpoint_{len(rows)//10:03d}.csv",index=False)
    if executor:executor.shutdown()
    result=pd.DataFrame(rows)
    result.to_csv(a.out_dir/"corrected_measurements.csv",index=False)
    summaries(result).to_csv(a.out_dir/"corrected_summary.csv",index=False)
    summary={"episodes":len(source),"measurements":len(result),"max_dice_csv_discrepancy":float(result.dice_minus_source.abs().max()),
             "max_hd95_csv_discrepancy_mm":float(result.hd95_minus_source.abs().max()),
             "disjoint_partition_assertions_passed":True,"empty_predictions":int(result.empty_prediction.sum()),
             "elapsed_seconds":time.perf_counter()-start,"new_inference":False,
             "independent_reference":"physical surface-voxel cKDTree; production actual AST with MedPy EDT",
             "unverifiable_saved_masks":"Step5 k2/k4/k6/GT-reference/random masks absent; only k0/k10 correspond to saved Step4 arms"}
    (a.out_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
