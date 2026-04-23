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
#POLICY_PATH = "/home/borys/Desktop/bachalor_thesis_lfd/models/wave/bc/20260423122021/models/model_epoch_500.pth"
POLICY_PATH = "/home/borys/Desktop/bachalor_thesis_lfd/models/wave/bc_rnn/20260423122141/models/model_epoch_355_best_validation_114923633.6.pth"
HORIZON = 200
HZ = 20
DT = 1.0 / HZ
ACTION_SCALE = 1.0  # Scaling for safety

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
    print(f"Running policy for {HORIZON} steps at {HZ}Hz")
    print("Press Ctrl+C to stop")

    # Cartesian impedance controller
    controller   = panda_py.controllers.CartesianImpedance()
    gripper_ctrl = GripperController(gripper)

    try:
        with panda.create_context(frequency=HZ, max_runtime=HORIZON / HZ) as ctx:
            panda.start_controller(controller)

            step = 0
            while ctx.ok():

                state = panda.get_state()
                obs, current_T = get_obs(state)

                # Policy inference
                raw_action = policy(obs) 

                raw_action = np.array(raw_action).flatten()

                # Integrate delta to absolute target
                target_pos, target_quat, gripper_cmd = integrate_action(
                    current_T, raw_action, ACTION_SCALE
                )

                """# forward kinematics to get target eef pose
                target_T = panda_py.fk(raw_action.reshape(7, 1))
                target_pos = target_T[:3, 3]
                target_quat = Rotation.from_matrix(target_T[:3, :3]).as_quat()"""

                controller.set_control(
                    position=target_pos.reshape(3, 1),
                    orientation=target_quat.reshape(4, 1)
                )

                gripper_cmd = 1.0  # always open for extend/retract
                gripper_ctrl.update(gripper_cmd)

                # Log
                if step % 20 == 0:
                    print(
                        f"[{step}] pos={obs['robot0_eef_pos'].round(3)},\n target_pos {target_pos}\n, raw_action {raw_action}\n"
                        f"delta pos={raw_action[:3].round(3)}  gripper={'open' if gripper_cmd > 0 else 'close'}"
                    )
                step += 1
 
    except KeyboardInterrupt:
        print("Stopped")
 
    finally:
        print("\nGoing home.")
        panda.move_to_start()
        print("Done")
 
 
if __name__ == "__main__":
    main()