import torch
import numpy as np
import panda_py
import panda_py.controllers
from panda_py import libfranka
import time
import sys
import os
from scipy.spatial.transform import Rotation
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

ROBOT_IP = "172.16.0.2"
POLICY_PATH = "/home/borys/Desktop/bachalor_thesis_lfd/models/extend_retract/bc_rnn/20260421141429/models/model_epoch_150.pth"
#POLICY_PATH = "/home/borys/Desktop/bachalor_thesis_lfd/models/extend_retract/bc/20260421113223/models/model_epoch_500.pth"
HORIZON = 200
HZ = 20
DT = 1.0 / HZ
ACTION_SCALE = 0.03  # Scaling for safety

GRIPPER_MAX_WIDTH = 0.08   # Hand fully open
GRIPPER_SPEED     = 0.05   # m/s
GRIPPER_FORCE     = 20.0   # N
GRIPPER_THRESHOLD = 0.0 


def get_obs(state):
    """Convert robot state to policy observation."""
    T = np.array(state.O_T_EE).reshape(4, 4, order='F')
    eef_pos  = T[:3, 3]
    eef_quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    gripper_qpos = np.zeros(2)

    obs = {
        "robot0_eef_pos": eef_pos.astype(np.float32),
        "robot0_eef_quat": eef_quat.astype(np.float32),
        "robot0_joint_pos": np.array(state.q,  dtype=np.float32),
        "robot0_joint_vel": np.array(state.dq, dtype=np.float32),
        "robot0_gripper_qpos": gripper_qpos.astype(np.float32),
    }
    return obs, T

def integrate_action(current_T, raw_action, action_scale):
    """
    Integrate a 7D delta action 
    onto the current EEF pose to produce an absolute Cartesian target.
    """
    delta_pos    = raw_action[:3] * action_scale
    delta_rotvec = raw_action[3:6] * action_scale  # axis-angle: matches robomimic convention
    gripper_cmd  = float(raw_action[6])

    current_pos = current_T[:3, 3]
    current_rot = Rotation.from_matrix(current_T[:3, :3])

    target_pos  = current_pos + delta_pos
    target_rot  = Rotation.from_rotvec(delta_rotvec) * current_rot
    target_quat = target_rot.as_quat()  # xyzw
 
    return target_pos, target_quat, gripper_cmd

class GripperController:
    """Gripper commands so we don't spam open/close every step."""
    def __init__(self, gripper):
        self.gripper = gripper
        self.is_open = True
 
    def update(self, gripper_cmd: float):
        want_open = gripper_cmd > GRIPPER_THRESHOLD
        if want_open and not self.is_open:
            self.gripper.move(GRIPPER_MAX_WIDTH, GRIPPER_SPEED)
            self.is_open = True
        elif not want_open and self.is_open:
            self.gripper.grasp(0.0, GRIPPER_SPEED, GRIPPER_FORCE,
                               epsilon_inner=0.08, epsilon_outer=0.08)
            self.is_open = False

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
    print("Policy loaded")

    # Connect
    print(f"\nConnecting to Franka at {ROBOT_IP}")
    panda = panda_py.Panda(ROBOT_IP)
    gripper = libfranka.Gripper(ROBOT_IP)
    print("Connected")

    # Go home
    print("Moving home.")
    panda.move_to_start()

    input("\nPress Enter to start execution.")

    # Running policy
    print(f"Running policy for {HORIZON} steps at")