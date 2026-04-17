
import torch
import numpy as np
import panda_py
import panda_py.controllers
import time
import sys
import os
from scipy.spatial.transform import Rotation

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

ROBOT_IP    = "172.16.0.2"
POLICY_PATH = "/home/borys/Desktop/bachalor_thesis_lfd/models/wave/wave_bc/20260417152510/models/model_epoch_200.pth"
HORIZON     = 200
HZ          = 20
DT          = 1.0 / HZ
ACTION_SCALE = 0.15 # Scaling for safety


def get_obs(state):
    """Convert robot state to policy observation."""
    q        = np.array(state.q)
    eef_pos  = np.array(state.O_T_EE[12:15])
    T        = np.array(state.O_T_EE).reshape(4, 4, order='F')
    eef_quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x,y,z,w]

    return {
        "robot0_joint_pos":    torch.tensor(q,          dtype=torch.float32).unsqueeze(0),
        "robot0_eef_pos":      torch.tensor(eef_pos,    dtype=torch.float32).unsqueeze(0),
        "robot0_eef_quat":     torch.tensor(eef_quat,   dtype=torch.float32).unsqueeze(0),
        "robot0_gripper_qpos": torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(0),
    }


def main():
    # Load policy
    print("Loading policy.")
    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=POLICY_PATH,
        device=device,
        verbose=True
    )
    policy.start_episode()
    print("Policy loaded!")

    # Connect
    print(f"\nConnecting to Franka at {ROBOT_IP}...")
    panda = panda_py.Panda(ROBOT_IP)
    print("Connected!")

    # Go home
    print("Moving home.")
    panda.move_to_start()

    input("\nPress Enter to start execution.")

    # ── rollout ───────────────────────────────────────────
    print(f"Running policy for {HORIZON} steps at {HZ}Hz...")
    print("Press Ctrl+C to stop.\n")

    try:
        for step in range(HORIZON):
            t_start = time.time()

            # Get state
            state = panda.get_state()
            obs   = get_obs(state)

            # Get action from policy
            with torch.no_grad():
                action = policy(obs)
            action = np.array(action).flatten()  # (7,)

            delta_pos = action[:3] * ACTION_SCALE
            # action[3:6] = delta orientation (ignored for wave)
            # action[6]   = gripper (ignored for wave)

            # Current eef pose
            current_pos = np.array(state.O_T_EE[12:15])
            T = np.array(state.O_T_EE).reshape(4, 4, order='F')
            current_quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x,y,z,w]
            q_current = np.array(state.q)

            # Target position
            target_pos = current_pos + delta_pos

            # Solve IK
            target_joints = panda_py.ik(
                position=target_pos.reshape(3, 1),
                orientation=current_quat.reshape(4, 1),
                q_init=q_current.reshape(7, 1)
            )

            # Send to robot
            panda.move_to_joint_position(
                [target_joints],
                speed_factor=0.1
            )

            # Maintain frequency
            elapsed    = time.time() - t_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if step % 20 == 0:
                print(f"Step {step:3d}/{HORIZON} | delta_pos: {delta_pos.round(4)}")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("\nGoing home.")
        panda.move_to_start()
        print("Done!")


if __name__ == "__main__":
    main()