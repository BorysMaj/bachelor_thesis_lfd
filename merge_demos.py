"""
Merge two or more robomimic HDF5 demo files into one.
Designed to combine simulation demos, real-robot demos, and existing merged files.

Usage:
    # Merge multiple files (any number)
    python merge_demos.py --inputs data/reach/obs1.hdf5 data/reach/obs2.hdf5 --out data/reach/merged/merged.hdf5

    # Legacy two-file interface still works
    python merge_demos.py --sim data/reach/obs.hdf5 --real data/reach/real_demos.hdf5 --out data/reach/merged/merged.hdf5

    # Keep all obs keys (zero-pad missing ones instead of taking intersection)
    python merge_demos.py --inputs ... --out ... --keep-all

"""

import argparse
import random
from pathlib import Path

import h5py
import numpy as np


# Helpers

def get_obs_keys(demo_group):
    return set(demo_group["obs"].keys())


def align_obs(src_group, target_keys: set):
    """
    Read obs from src_group and return a dict with exactly target_keys.
    Keys present in src but not in target are dropped.
    Keys in target but not in src are skipped (caller handles zero-padding).
    """
    obs      = src_group["obs"]
    src_keys = set(obs.keys())
    result   = {}
    warnings = []

    for key in target_keys:
        if key in src_keys:
            result[key] = obs[key][:]
        else:
            warnings.append(f"  WARN: '{key}' missing in this demo - will be zero-padded")

    extra = src_keys - target_keys
    if extra:
        warnings.append(f" INFO: dropping keys not in target set: {sorted(extra)}")

    return result, warnings


def iter_demos(f):
    """Yield (name, group) for each demo in f['data'], sorted."""
    data  = f["data"]
    names = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
    for name in names:
        yield name, data[name]


def copy_demo(src_group, dst_data, new_index: int, obs_dict: dict,
              target_keys: set, key_shapes: dict):
    """Write one demo into dst_data as demo_{new_index}."""
    T   = src_group["actions"].shape[0]
    grp = dst_data.create_group(f"demo_{new_index}")
    grp.attrs["num_samples"] = T

    grp.create_dataset("actions", data=src_group["actions"][:])
    grp.create_dataset("rewards", data=src_group["rewards"][:] if "rewards" in src_group else np.zeros(T))
    grp.create_dataset("dones",   data=src_group["dones"][:]   if "dones"   in src_group else np.zeros(T))
    if "states" in src_group:
        grp.create_dataset("states", data=src_group["states"][:])

    og = grp.create_group("obs")
    for key in target_keys:
        if key in obs_dict:
            og.create_dataset(key, data=obs_dict[key])
        else:
            # Zero-pad: use shape recorded from another file
            shape = (T,) + key_shapes.get(key, (1,))
            og.create_dataset(key, data=np.zeros(shape, dtype=np.float32))

    return T


# Main
def merge(input_paths: list[Path], out_path: Path,
          val_ratio: float, keep_all: bool, seed: int):

    print(f"  Merging {len(input_paths)} file(s) → {out_path}")
    for p in input_paths:
        print(f"    {p}")

    # Open all files and load demos
    open_files = [h5py.File(p, "r") for p in input_paths]
    try:
        all_demo_lists = [list(iter_demos(f)) for f in open_files]

        for i, (path, demos) in enumerate(zip(input_paths, all_demo_lists)):
            print(f"\n  [{i+1}] {path.name}: {len(demos)} demos")

        # Determine target obs keys
        all_key_sets = [get_obs_keys(demos[0][1]) for demos in all_demo_lists if demos]
        if not all_key_sets:
            print("  ERROR: No demos found in any file.")
            return

        if keep_all:
            target_keys = set.union(*all_key_sets)
        else:
            target_keys = set.intersection(*all_key_sets)

        print(f"\n  Obs keys kept : {sorted(target_keys)}")
        for i, (ks, path) in enumerate(zip(all_key_sets, input_paths)):
            dropped = ks - target_keys
            if dropped:
                print(f"  Dropping from {path.name}: {sorted(dropped)}")

        core = {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"}
        missing_core = core - target_keys
        if missing_core:
            print(f"\n  WARN: core robot keys missing: {missing_core}")


        # Record shapes of each key (for zero-padding when a key is absent)
        key_shapes = {}
        all_demos_flat = [grp for demos in all_demo_lists for _, grp in demos]
        for key in target_keys:
            for grp in all_demos_flat:
                if "obs" in grp and key in grp["obs"]:
                    key_shapes[key] = grp["obs"][key].shape[1:]
                    break

        # Get env_args from first file that has it
        env_args_str = "{}"
        for f in open_files:
            val = f["data"].attrs.get("env_args", None)
            if val:
                env_args_str = val
                break

        # Write output - delete existing merged file first so it's always fresh
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"\n  Deleting existing merged file: {out_path}")
            out_path.unlink()

        with h5py.File(out_path, "w") as fout:
            dst = fout.create_group("data")
            dst.attrs["env_args"] = env_args_str

            idx = 0
            total_steps = 0
            names_per_file = []

            for file_idx, (path, demos) in enumerate(zip(input_paths, all_demo_lists)):
                print(f"\n  Copying {len(demos)} demos from {path.name} …")
                file_names = []
                for _, grp in demos:
                    obs_dict, warns = align_obs(grp, target_keys)
                    for w in warns:
                        print(w)
                    T = copy_demo(grp, dst, idx, obs_dict, target_keys, key_shapes)
                    total_steps += T
                    file_names.append(f"demo_{idx}")
                    idx += 1
                names_per_file.append(file_names)

            dst.attrs["total"] = total_steps

            # Stratified train/val split across all source files
            rng = random.Random(seed)
            val_names   = []
            train_names = []
            for names in names_per_file:
                rng.shuffle(names)
                n_val = max(1, int(len(names) * val_ratio))
                val_names   += names[:n_val]
                train_names += names[n_val:]
            rng.shuffle(train_names)

            mg = fout.create_group("mask")
            mg.create_dataset("train", data=np.array(train_names, dtype="S"))
            mg.create_dataset("valid", data=np.array(val_names,   dtype="S"))

        print(f"\n{'='*60}")
        print(f"Written      : {out_path}")
        print(f"Total demos  : {idx}")
        print(f"Total steps  : {total_steps}")
        print(f"Train / val  : {len(train_names)} / {len(val_names)}")
        print(f"Obs keys     : {sorted(target_keys)}")
        print(f"{'='*60}\n")

    finally:
        for f in open_files:
            f.close()


def main():
    parser = argparse.ArgumentParser(
        description="Merge robomimic HDF5 demo files for training."
    )
    # New multi-file interface
    parser.add_argument("--inputs", nargs="+",
                        help="Two or more HDF5 files to merge")
    # Legacy two-file interface
    parser.add_argument("--sim",  help="(legacy) Path to simulation demos")
    parser.add_argument("--real", help="(legacy) Path to real-robot demos")

    parser.add_argument("--out", required=True,
                        help="Output path for merged file")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of demos for validation (default 0.1)")
    parser.add_argument("--keep-all", action="store_true",
                        help="Keep all obs keys, zero-padding missing ones")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for train/val shuffle")
    args = parser.parse_args()

    # Resolve input paths
    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
    elif args.sim and args.real:
        input_paths = [Path(args.sim), Path(args.real)]
    else:
        parser.error("Provide either --inputs (one or more files) or both --sim and --real.")

    for p in input_paths:
        if not p.exists():
            parser.error(f"File not found: {p}")

    merge(
        input_paths = input_paths,
        out_path    = Path(args.out),
        val_ratio   = args.val_ratio,
        keep_all    = args.keep_all,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
