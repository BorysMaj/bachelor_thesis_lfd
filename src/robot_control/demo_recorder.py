import panda_py
import panda_py.controllers
from panda_py import libfranka
import h5py
import numpy as np
import time
import json
from scipy.spatial.transform import Rotation

GRIPPER_MAX_WIDTH = 0.08  # Hand fully open

class KinestheticDemoRecorder:
    """
    Records kinesthetic demos from Franka using panda-py
    and saves in robomimic HDF5 format.
    """

    def __init__(self, robot_ip="172.16.0.2", record_hz=20):
        self.robot_ip = robot_ip
        self.record_hz = record_hz
        self.dt = 1.0 / record_hz # Time between steps
        self.panda = None
        self.gripper = None
        self.is_recording = False
        self.demos = []


    def connect(self):
        self.panda = panda_py.Panda(self.robot_ip)
        self.gripper = libfranka.Gripper(self.robot_ip)
        print(f"Connected to Franka at {self.robot_ip}")

    def enable_teaching_mode(self):
        """
        Switch to teaching mode.
        """
        self.panda.teaching_mode(True)
        print("Teaching mode ON, you can move the arm")

    def disable_teaching_mode(self):
        self.panda.teaching_mode(False)
        print("Teaching mode OFF")


    # Recording

    def start_recording(self):
        self.current_demo = {
            "obs": {
                "robot0_joint_pos":    [], # 7 joint angles
                "robot0_eef_pos":      [], # XYZ of hand
                "robot0_eef_quat":     [], # Orientation of hand
                "robot0_gripper_qpos": [],  # [left_finger, right_finger] each = width/2
            },
            "actions": [],
            "states":  [],
        }
        self.prev_eef_pos  = None
        self.prev_eef_quat = None
        self.is_recording  = True
        print("Recording started")

    def record_step(self):
        """Read one state from the robot and store it."""
        if not self.is_recording:
            return

        state = self.panda.get_state()

        # Arm observations
        q = np.array(state.q) # Joint angles
        T = np.array(state.O_T_EE).reshape(4, 4, order='F')
        eef_pos = T[:3, 3]
        eef_quat = self.mat2quat(state.O_T_EE) # Orientation

        # Gripper observation
        # GripperState with total gap
        gripper_state = self.gripper.read_once()
        width = gripper_state.width # total width in metres
        gripper_qpos = np.array([width / 2, width / 2]) # width per-finger

        # Gripper, normalise width to [-1, +1]
        gripper_cmd = np.array([(width / GRIPPER_MAX_WIDTH) * 2.0 - 1.0])

        self.current_demo["obs"]["robot0_joint_pos"].append(q)
        self.current_demo["obs"]["robot0_eef_pos"].append(eef_pos)
        self.current_demo["obs"]["robot0_eef_quat"].append(eef_quat)
        self.current_demo["obs"]["robot0_gripper_qpos"].append(gripper_qpos)
        self.current_demo["states"].append(np.concatenate([q, eef_pos, eef_quat, gripper_qpos]))

        # Action = delta eef from previous step
        if self.prev_eef_pos is not None:
            delta_pos = eef_pos - self.prev_eef_pos
            r_curr    = Rotation.from_quat(eef_quat)
            r_prev    = Rotation.from_quat(self.prev_eef_quat)
            delta_ori = (r_curr * r_prev.inv()).as_rotvec() # (3,) ,Axis-angle (rotvec): direction = rotation axis, magnitude = angle in radians
            action = np.concatenate([delta_pos, delta_ori, gripper_cmd]) # (7,)
        else:
            action = np.zeros(7)

        self.current_demo["actions"].append(action)
        self.prev_eef_pos  = eef_pos
        self.prev_eef_quat = eef_quat

        
    def stop_recording(self):
        self.is_recording = False

        demo = {
            "actions": np.array(self.current_demo["actions"]),
            "states":  np.array(self.current_demo["states"]),
            "obs": {k: np.array(v) for k, v in self.current_demo["obs"].items()}
        }

        T = demo["actions"].shape[0]
        print(f"Recorded {T} steps ({T/self.record_hz}s)")
        self.demos.append(demo)
        return demo
    

    def save(self, path, task_name="task"):
        with h5py.File(path, "w") as f:
            grp = f.create_group("data")
            grp.attrs["total"] = sum(d["actions"].shape[0] for d in self.demos)
            grp.attrs["env_args"] = json.dumps({
                "env_name": "FrankaReal",
                "type": 2,
                "env_kwargs": {"robots": ["Panda"], "task": task_name}
            })

            for i, demo in enumerate(self.demos):
                T   = demo["actions"].shape[0]
                dg  = grp.create_group(f"demo_{i}")
                dg.attrs["num_samples"] = T
                dg.create_dataset("actions", data=demo["actions"])
                dg.create_dataset("states", data=demo["states"])
                dg.create_dataset("rewards", data=np.zeros(T))
                dg.create_dataset("dones", data=np.zeros(T))
                og = dg.cr