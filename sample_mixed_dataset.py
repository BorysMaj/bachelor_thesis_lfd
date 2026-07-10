"""
Sample mixed datasets from multiple users' HDF5 files.

Generates 10 samples (seeds 1-10) for both liftnut and push tasks.
"""

import random
from pathlib import Path

import h5py
import numpy as np


MAX_DEMOS_PER_USER = 20

def get_demo_names(f, max_demos=MAX_DEMOS_PER_USER):
    """Return up to max_demos demo names from a file, sorted by index."""
    data = f["data"]
    names = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
    return names[:max_demos]


def compute_allocation(n_users, total):
    """
    Distribute `total` demos as equally as possible across `n_users`.
    Example: 15 users, 20 demos - 5 users get 2, 10 users get 1.
    Returns a list of ints of length n_users.
    """
    base = total // n_users
    remainder = total % n_users
    return [base + 1 if i < remainder else base for i in range(n_users)]


def copy_demo(src_group, dst_data, new_index, target_keys, key_shapes):
    """Write one demo into dst_data as demo_{new_index}."""
    T = src_group["actions"].shape[0]
    grp = dst_data.create_group(f"demo_{new_index}")
    grp.attrs["num_samples"] = T

    grp.create_dataset("actions", data=src_group["actions"][:])
    grp.create_dataset("rewards", data=src_group["rewards"][:] if "rewards" in src_group else np.zeros(T))
    grp.create_dataset("dones",   data=src_group["dones"][:]   if "dones"   in src_group else np.zeros(T))
    if "states" in src_group:
        grp.create_dataset("states", data=src_group["states"][:])

    og = grp.create_group("obs")
    for key in target_keys:
        if key in src_group["obs"]:
            og.create_dataset(key, data=src_group["obs"][key][:])
        else:
            shape = (T,) + key_shapes.get(key, (1,))
            og.create_dataset(key, data=np.zeros(shape, dtype=np.float32))

    return T


def sample_mixed(data_root, task, out_path, total, seed, val_ratio=0.1):
    rng = random.Random(seed)

    # Discover user folders
    data_root = Path(data_root)
    user_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("user")])

    input_paths = []
    for user_dir in user_dirs:
        hdf5 = user_dir / task / "merged" / "merged.hdf5"
        if hdf5.exists():
            input_paths.append((user_dir.name, hdf5))
        else:
            print(f"{hdf5} not found, skipping {user_dir.name}")

    n_users = len(input_paths)
    if n_users == 0:
        print("No HDF5 files found.")
        return

    print(f"\nFound {n_users} users for task '{task}'")
    print(f"Sampling {total} demos total (seed={seed})\n")

    # Compute allocation and shuffle so extra demos aren't always from the first users
    allocation = compute_allocation(n_users, total)
    indices = list(range(n_users))
    rng.shuffle(indices)
    alloc_map = {indices[i]: allocation[i] for i in range(n_users)}

    open_files = [h5py.File(p, "r") for _, p in input_paths]

    try:
        # Get env_args from first file that has it
        env_args_str = "{}"
        for f in open_files:
            val = f["data"].attrs.get("env_args", None)
            if val:
                env_args_str = val
                break

        # Write output
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"Deleting existing file: {out_path}")
            out_path.unlink()

        with h5py.File(out_path, "w") as fout:
            dst = fout.create_group("data")
            dst.attrs["env_args"] = env_args_str

            idx = 0
            total_steps = 0
            all_names = []

            for file_idx, (username, _) in enumerate(input_paths):
                f = open_files[file_idx]
                n_take = alloc_map[file_idx]
                available = get_demo_names(f, max_demos=MAX_DEMOS_PER_USER)

                if len(available) < n_take:
                    print(f"  WARN: {username} only has {len(available)} demos, reducing allocation")
                    n_take = len(available)

                sampled = rng.sample(available, n_take)
                print(f"  {username}: taking {n_take} demo(s) - {sampled}")

                for demo_name in sampled:
                    T = copy_demo(f["data"][demo_name], dst, idx, target_keys, key_shapes)
                    total_steps += T
                    all_names.append(f"demo_{idx}")
                    idx += 1

            dst.attrs["total"] = total_steps

            # New train/val split
            names_copy = all_names.copy()
            rng.shuffle(names_copy)
            n_val = max(1, int(len(names_copy) * val_ratio))
            val_names = names_copy[:n_val]
            train_names = names_copy[n_val:]

            mg = fout.create_group("mask")
            mg.create_dataset("train", data=np.array(train_names, dtype="S"))
            mg.create_dataset("valid", data=np.array(val_names, dtype="S"))

        print(f"\n{'='*60}")
        print(f"Written: {out_path}")
        print(f"Total demos: {idx}")
        print(f"Total steps: {total_steps}")
        print(f"Train / val: {len(train_names)} / {len(val_names)}")

    finally:
        for f in open_files:
            f.close()


DATA_ROOT = "user_study_data"
TASKS     = ["nut", "push"]
SEEDS     = range(1, 11)   # seeds 1-10 - 10 samples per task
TOTAL     = 20

def main():
    for task in TASKS:
        for seed in SEEDS:
            out = f"mixed_datasets/{task}/sample{seed}.hdf5"
            print(f"Task={task}, seed={seed}, output={out}")
            sample_mixed(DATA_ROOT, task, out, TOTAL, seed)

if __name__ == "__main__":
    main()