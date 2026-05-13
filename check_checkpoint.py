"""
Quick diagnostic: inspect checkpoint normalization stats and obs keys.
Run locally (not on Snellius) — or wherever the .pth lives.

Usage:
    python check_checkpoint.py path/to/model_epoch_XXX.pth
"""

import sys
import torch
import json

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/borys/Desktop/bachalor_thesis_lfd/models/wave/bc_rnn/20260423122141/models/model_epoch_355_best_validation_114923633.6.pth"

print(f"\nLoading: {path}\n{'='*60}")
ckpt = torch.load(path, map_location="cpu")

print("\n--- Top-level keys ---")
for k in ckpt.keys():
    print(f"  {k}")

# Obs normalization stats → tells you exactly which obs keys the model used
if "obs_normalization_stats" in ckpt:
    print("\n--- obs_normalization_stats keys (what the model was trained on) ---")
    stats = ckpt["obs_normalization_stats"]
    for k, v in stats.items():
        if isinstance(v, dict):
            mean = v.get("mean", v.get("offset", None))
            std  = v.get("std",  v.get("scale",  None))
            if mean is not None:
                import numpy as np
                mean = torch.tensor(mean).numpy() if not isinstance(mean, torch.Tensor) else mean.numpy()
                print(f"  {k}: shape={mean.shape}  mean={mean.round(4)}")
        else:
            print(f"  {k}: {v}")

# Action normalization stats → tells you if large Y movements are expected
if "action_normalization_stats" in ckpt:
    print("\n--- action_normalization_stats ---")
    stats = ckpt["action_normalization_stats"]
    if isinstance(stats, dict):
        for k, v in stats.items():
            if isinstance(v, (list, torch.Tensor)):
                import numpy as np
                arr = torch.tensor(v).numpy() if not isinstance(v, torch.Tensor) else v.numpy()
                print(f"  {k}: {arr.round(4)}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"  {stats}")

# Config embedded in checkpoint
if "config" in ckpt:
    cfg = ckpt["config"]
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    obs_keys = cfg.get("observation", {}).get("modalities", {})
    print("\n--- obs keys from embedded config ---")
    print(json.dumps(obs_keys, indent=2))

    action_config = cfg.get("train", {}).get("action_config", {})
    if action_config:
        print("\n--- action_config ---")
        print(json.dumps(action_config, indent=2))

print("\nDone.")
