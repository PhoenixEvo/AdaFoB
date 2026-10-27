"""Paired fixed-prediction analysis; no inference or parameter selection.
Missing and nonfinite values fail closed. Source metric units are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ORGANS = {"LIVER", "LK", "RK", "SPLEEN"}
KEY = ["fold", "organ", "patient_id"]


def patient_id(path: str) -> str:
    return str(path).replace(chr(92), "/").rsplit("/", 1)[-1].split(".nii")[0]


def validate(frame: pd.DataFrame, manifest: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Require exact episode identities, pairing, four organs and finite metrics."""
    frame, manifest = frame.copy(), manifest.copy()
    for table in (frame, manifest):
        if "patient_id" not in table:
            table["patient_id"] = table["vol_id"].map(patient_id)
        elif "vol_id" in table and not (table.patient_id == table.vol_id.map(patient_id)).all():
            raise ValueError("patient_id disagrees with vol_id")
        if table[KEY + ["episode_id"]].isna().any().any():
            raise ValueError("missing identity")
        if table.duplicated(KEY).any() or table.episode_id.duplicated().any():
            raise ValueError("duplicate evaluation identity")
    identity = KEY + ["episode_id"]
    if set(map(tuple, frame[identity].to_numpy())) != set(map(tuple, manifest[identity].to_numpy())):
        raise ValueError("evaluation does not exactly match manifest")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("missing/nonfinite metric: investigate, do not silently drop")
    if not frame.groupby("patient_id").organ.apply(lambda x: set(x) == ORGANS).all():
        raise ValueError("every patient must have all four organs")
    for column in columns:
        if column.startswith("dice_") and not frame[column].between(0, 1).all():
            raise ValueError("Dice must be a fraction in [0,1]")
    return frame.sort_values(identity).reset_index(drop=True)


def patient_means(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Average repeats within patient-organ, then macro-average organs in patient."""
    return frame.groupby(["patient_id", "organ"])[columns].mean().groupby("patient_id").mean()


def summarize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    balanced = patient_means(frame, columns)
    return pd.DataFrame([
        {"metric": c, "episode_weighted": frame[c].mean(),
         "patient_balanced": balanced[c].mean(), "episodes": len(frame),
         "patients": len(balanced), "nonfinite": int((~np.isfinite(frame[c])).sum())}
        for c in columns
    ])


def paired_bootstrap(frame: pd.DataFrame, first: str, second: str,
                     replicates: int, seed: int) -> dict:
    """Resample the same query patients for both methods; fixed support/checkpoints."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    pt = patient_means(frame, [first, second] if first != second else [first])
    differences = pt[first].to_numpy() - pt[second].to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(pt), size=(replicates, len(pt)))
    scale = 100.0 if first.startswith("dice_") and second.startswith("dice_") else 1.0
    ci = np.quantile(differences[samples].mean(axis=1) * scale, [0.025, 0.975])
    grouped = frame.assign(delta=frame[first] - frame[second]).groupby("patient_id").delta.agg(["sum", "count"]).reindex(pt.index)
    ep_boot = grouped["sum"].to_numpy()[samples].sum(axis=1) / grouped["count"].to_numpy()[samples].sum(axis=1)
    return {"first": first, "second": second, "unit": "pp" if scale == 100 else "source units",
            "patient_balanced_effect": float(differences.mean() * scale),
            "patient_balanced_ci95": ci.tolist(),
            "episode_weighted_effect": float((frame[first] - frame[second]).mean() * scale),
            "episode_weighted_ci95": (np.quantile(ep_boot, [0.025, 0.975]) * scale).tolist(),
            "bootstrap_replicates": replicates, "seed": seed,
            "interpretation": "exploratory pointwise interval; fixed predictions/support/checkpoints; no training or support-selection uncertainty"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contrast", nargs=2, action="append", default=[])
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    columns = [c for c in frame if c.startswith(("dice_", "hd95_")) and "_std_" not in c]
    frame = validate(frame, pd.read_csv(args.manifest), columns)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    summarize(frame, columns).to_csv(args.out_dir / "summary.csv", index=False)
    patient_means(frame, columns).to_csv(args.out_dir / "patient_means.csv")
    frame.groupby(["patient_id", "organ"]).size().rename("observations").to_csv(args.out_dir / "repeat_counts.csv")
    contrasts = [paired_bootstrap(frame, a, b, args.bootstrap, args.seed) for a, b in args.contrast]
    metadata = {"source": str(args.csv), "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "contrasts": contrasts, "seed_status": "fixed-prediction reanalysis; model-run seed not inferred"}
    (args.out_dir / "analysis.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"episodes": len(frame), "patients": frame.patient_id.nunique(), "contrasts": contrasts}), flush=True)


if __name__ == "__main__":
    main()
