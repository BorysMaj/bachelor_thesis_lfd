"""
Merge two robomimic HDF5 demo files into one.
Designed to combine simulation demos with real-robot demos.

Both files must already have matching observation keys.
Keys only present in one file are dropped (with a warning) unless
you pass --keep-all, which zero-pads the missing ones.

Usage:
    python src/data/merge_demos.py \
        --sim  data/reach/obs.hdf5 \
        --real data/reach/real_demos.hdf5 \
        --out  data/reach/merged.hdf5

    # Keep all obs keys (zero-pad missing ones)
    python src/data/merge_demos.py --sim ... --real ... --out ... --keep-all

    # Override train/val split ratio (default 0.1 = 10% validation)
    python src/data/merge_demos.py --sim ... --real ... --out ... --val-ratio 0.2

    # Preview what would happen without writing anything
    python src/data/merge_demos.py --sim ... --real ... --out ... --dry-run
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
            warnings.append(f" WARN: '{key}' missing in this demo - will be zero-padded")

    extra = src_keys - target_keys
    if extra:
        warnings.append(f"  INFO: dropping keys not in target set: {sorted(extra)}")

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
    grp.create_dataset("dones", data=src_group["dones"][:] if "dones"   in src_group else np.zeros(T))
    if "states" in src_group:
        grp.create_dataset("states", data=src_group["states"][:])

    og = grp.create_group("obs")
    for key in target_keys:
        if key in obs_dict:
            og.create_dataset(key, data=obs_dict[key])
        else:
            # Zero-pad: use shape recorded from the other file
            shape = (T,) + key_shapes.get(key, (1,))
            og.create_dataset(key, data=np.zeros(shape, dtype=np.float32))

    return T


# Main
def merge(sim_path: Path, real_path: Path, out_path: Path,
          val_ratio: float, keep_all: bool, dry_run: bool, seed: int):

    print(f" SIM : {sim_path}")
    print(f" REAL : {real_path}")
    print(f" OUT : {out_path}")

    with h5py.File(sim_path, "r") as fsim, \
         h5py.File(real_path, "r") as freal:

        sim_demos = list(iter_demos(fsim))
        real_demos = list(iter_demos(freal))

        print(f"Sim demos : {len(sim_demos)}")
        print(f"Real demos : {len(real_demos)}")

        sim_obs_keys = get_obs_keys(sim_demos[0][1])
        real_obs_keys = get_obs_keys(real_demos[0][1])

        print(f"\n Sim obs keys : {sorted(sim_obs_keys)}")
        print(f"Real obs keys : {sorted(real_obs_keys)}")

        if keep_all:
            target_keys = sim_obs_keys | real_obs_keys
        else:
            target_keys = sim_obs_keys & real_obs_keys

        dropped_sim  = sim_obs_keys  - target_keys
        dropped_real = real_obs_keys - target_keys

        print(f"\n  Merged obs keys : {sorted(target_keys)}")
        if dropped_sim:
            print(f"  Dropping from sim  : {sorted(dropped_sim)}")
        if dropped_real:
            print(f"  Dropping from real : {sorted(dropped_real)}")

        core = {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"}
        missing_core = core - target_keys
        if missing_core:
            print(f"\n  WARN: core robot keys missing: {missing_core}")

        if dry_run:
            print("\n  [DRY RUN] Nothing written.")
            return

        # Record shapes of each key (for zero-padding when a key is absent)
        key_shapes = {}
        for key in target_keys:
            for _, grp in sim_demos + real_demos:
                if key in grp["obs"]:
                    key_shapes[key] = grp["obs"][key].shape[1:]
                    break

        # Write
        out_path.parent.mkdir(parents=True, exist_ok=True)
        env_args_str = fsim["data"].attrs.get("env_args", "{}")

        with h5py.File(out_path, "w") as fout:
            dst = fout.create_group("data")
            dst.attrs["env_args"] = env_args_str

            idx = 0
            total_steps = 0
            sim_names = []
            real_names = []

            print(f"\n  Copying {len(sim_demos)} sim demos …")
            for _, grp in sim_demos:
                obs_dict, warns = align_obs(grp, target_keys)
                for w in warns:
                    print(w)
                T = copy_demo(grp, dst, idx, obs_dict, target_keys, key_shapes)
                total_steps += T
                sim_names.append(f"demo_{idx}")
                idx += 1

            print(f"  Copying {len(real_demos)} real demos …")
            for _, grp in real_demos:
                obs_dict, warns = align_obs(grp, target_keys)
                for w in warns:
                    print(w)
                T = copy_demo(grp, dst, idx, obs_dict, target_keys, key_shapes)
                total_steps += T
                real_names.append(f"demo_{idx}")
                idx += 1

            dst.attrs["total"] = total_steps

            # Stratified train/val split
            rng = random.Random(seed)
            sim_val_n = max(1, int(len(sim_names)  * val_ratio))
            real_val_n = max(1, int(len(real_names) * val_ratio))
            rng.shuffle(sim_names)
            rng.shuffle(real_names)
            val_names = sim_names[:sim_val_n] + real_names[:real_val_n]
            train_names = sim_names[sim_val_n:] + real_names[real_val_n:]
            rng.shuffle(train_names)

            mg = fout.create_group("mask")
            mg.create_dataset("train", data=np.array(train_names, dtype="S"))
            mg.create_dataset("valid", data=np.array(val_names,   dtype="S"))

        print(f"\n{'='*60}")
        print(f"Written : {out_path}")
        print(f"Total demos : {idx} ({len(sim_names)} sim + {len(real_names)} real)")
        print(f"Total steps : {total_steps}")
        print(f"Train / val : {len(train_names)} / {len(val_names)}")
        print(f"Obs keys : {sorted(target_keys)}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge sim and real HDF5 demo files for robomimic training."
    )
    parser.add_argument("--sim",       required=True,
                        help="Path to simulation demos")
    parser.add_argument("--real",      required=True,
                        help="Path to real-robot demos")
    parser.add_argument("--out",       required=True,
                        help="Output path for merged file")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of demos for validation (default 0.1)")
    parser.add_argument("--keep-all",  action="store_true",
                        help="Keep all obs keys, zero-padding missing ones (default: intersection only)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print what would happen without writing anything")
    parser.add_argument("--seed",      type=int, default=1,
                        help="Random seed for train/val shuffle")
    args = parser.parse_args()

    merge(
        sim_path  = Path(args.sim),
        real_path = Path(args.real),
        out_path  = Path(args.out),
        val_ratio = args.val_ratio,
        keep_all  = args.keep_all,
        dry_run   = args.dry_run,
        seed      = args.seed,
    )


if __name__ == "__main__":
    main()
