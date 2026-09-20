import os
import hashlib
import subprocess
import datetime
import pandas as pd
import argparse

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except:
        return "UNKNOWN"

def get_sha256(filepath):
    if not os.path.exists(filepath):
        return "MISSING"
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def update_manifest(result_file, ckpt_dir, rank, seed, folds):
    manifest_file = "results/experiment_manifest.csv"
    
    # Locate all checkpoints and compute hashes
    ckpt_paths = []
    ckpt_hashes = []
    
    if ckpt_dir and os.path.exists(ckpt_dir):
        # Scan for fold checkpoints
        pth_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
        for f in pth_files:
            p = os.path.join(ckpt_dir, f)
            ckpt_paths.append(p)
            ckpt_hashes.append(get_sha256(p))
            
    ckpt_path_str = ";".join(ckpt_paths) if ckpt_paths else (ckpt_dir if ckpt_dir else "UNKNOWN")
    ckpt_hash_str = ";".join(ckpt_hashes) if ckpt_hashes else "UNKNOWN"
                
    commit = get_git_commit()
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    
    row = {
        "result_file": result_file,
        "checkpoint_path": ckpt_path_str,
        "checkpoint_sha256": ckpt_hash_str,
        "lora_rank": rank,
        "seed": str(seed),
        "folds_covered": str(folds),
        "git_commit_hash": commit,
        "kaggle_notebook_id": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "UNKNOWN"),
        "generation_timestamp": timestamp
    }
    
    df_row = pd.DataFrame([row])
    os.makedirs(os.path.dirname(manifest_file) if os.path.dirname(manifest_file) else ".", exist_ok=True)
    if os.path.exists(manifest_file):
        df_row.to_csv(manifest_file, mode='a', header=False, index=False)
    else:
        df_row.to_csv(manifest_file, index=False)
    print(f"[PROVENANCE] Successfully logged to {manifest_file} for {result_file}")
    print(f"   Rank: {rank} | Seed: {seed} | Folds: {folds}")
    print(f"   Checkpoints hashed ({len(ckpt_hashes)} files): {ckpt_hash_str[:60]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=str, required=True)
    parser.add_argument("--folds", type=str, required=True)
    args = parser.parse_args()
    
    update_manifest(args.result_file, args.ckpt_dir, args.rank, args.seed, args.folds)
