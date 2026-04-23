"""
Inspect a robomimic HDF5 demo file.
Usage: python inspect_demos.py --path /path/to/demos.hdf5
"""
import argparse
import h5py
import numpy as np

def inspect(path):
    with h5py.File(path, "r") as f:

        # ── Top-level structure ────────────────────────────────────────────
        grp  = f["data"]
        keys = list(grp.keys())
        print(f"\n{'='*60}")
        print(f"FILE: {path}")
        print(f"Total samples : {grp.attrs.get('total', '?')}")
        print(f"Num demos     : {len(keys)}")
        print(f"env_args      : {grp.attrs.get('env_args', '?')}")

        # ── Per-demo shape check ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print("DEMO SHAPES (first 5 demos):")
        for key in keys[:5]:
            dg = grp[key]
            T  = dg.attrs.get("num_samples", "?")
            a_shape = dg["actions"].shape
            s_shape = dg["states"].shape
            obs_keys = list(dg["obs"].keys())
            print(f"  {key}: T={T}  actions={a_shape}  states={s_shape}  obs={obs_keys}")

        # ── Action statistics across ALL demos ────────────────────────────
        print(f"\n{'='*60}")
        print("ACTION STATISTICS (all demos combined):")
        all_actions = np.concatenate([grp[k]["actions"][:] for k in keys], axis=0)
        labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
        print(f"  Shape : {all_actions.shape}")
        print(f"  {'dim':<10} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
        print(f"  {'-'*50}")
        for i, label in enumerate(labels):
            col = all_actions[:, i]
            print(f"  {label:<10} {col.mean():>10.5f} {col.std():>10.5f} {col.min():>10.5f} {col.max():>10.5f}")

        # ── Check for near-zero actions (movement too slow?) ───────────────
        pos_actions = all_actions[:, :3]
        mean_step_size = np.linalg.norm(pos_actions, axis=1).mean()
        print(f"\n  Mean EEF step size (metres/step): {mean_step_size:.6f}")
        if mean_step_size < 0.001:
            print("    WARNING: average step size < 1mm — actions are near zero.")
            print("    This likely means the robot was moved too slowly, or")
            print("    the action alignment is still wrong.")

        # ── Obs statistics ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("OBSERVATION STATISTICS (first demo):")
        first = grp[keys[0]]["obs"]
        for ok in first.keys():
            arr = first[ok][:]
            print(f"  {ok:<30} shape={arr.shape}  mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")

        # ── Sample one demo and print first 5 steps ───────────────────────
        print(f"\n{'='*60}")
        print("FIRST 5 STEPS OF demo_0:")
        d0_actions = grp[keys[0]]["actions"][:]
        d0_eef_pos = grp[keys[0]]["obs"]["robot0_eef_pos"][:]
        print(f"  {'step':<6} {'eef_pos':^36} {'action (dx,dy,dz)':^36}")
        print(f"  {'-'*78}")
        for i in range(min(5, len(d0_actions))):
            pos_str = np.array2string(d0_eef_pos[i], precision=4, separator=", ")
            act_str = np.array2string(d0_actions[i, :3], precision=5, separator=", ")
            print(f"  {i:<6} {pos_str:<36} {act_str:<36}")

        # ── Check action/obs length match ──────────────────────────────────
        print(f"\n{'='*60}")
        print("ACTION / OBS LENGTH CHECK:")
        all_ok = True
        for key in keys:
            dg      = grp[key]
            T_act   = dg["actions"].shape[0]
            T_obs   = dg["obs"]["robot0_eef_pos"].shape[0]
            T_state = dg["states"].shape[0]
            match   = "OK" if T_act == T_obs == T_state else "MISMATCH ⚠"
            if match != "OK":
                all_ok = False
            print(f"  {key}: actions={T_act}  obs={T_obs}  states={T_state}  [{match}]")
        if all_ok:
            print("  All demos have matching lengths.")

        # ── Train/val mask ─────────────────────────────────────────────────
        if "mask" in f:
            print(f"\n{'='*60}")
            print("TRAIN/VAL SPLIT:")
            for split in f["mask"].keys():
                names = [n.decode() if isinstance(n, bytes) else n for n in f["mask"][split][:]]
                print(f"  {split}: {len(names)} demos → {names}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to .hdf5 demo file")
    args = parser.parse_args()
    inspect(args.path)
