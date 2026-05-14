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

    def get_possition(self):
        state = self.panda.get_state()
        T = np.array(state.O_T_EE).reshape(4,4, order='F')
        print("Current possition =", T[:3, 3])


    # Recording

    def start_recording(self):
        self.current_demo = {
            "obs": {
                "robot0_joint_pos": [], # 7 joint angles
                "robot0_eef_pos": [], # XYZ of hand
                "robot0_eef_quat": [], # Orientation of hand
                "robot0_gripper_qpos": [], # [left_finger, right_finger] each = width/2
            },
            "actions": [],
            "states":  [],
        }
        self.prev_eef_pos  = None
        self.prev_eef_quat = None
        self.is_recording  = True
        print("Recording started")

    def start_recording_reach(self, target_pos):
        """Like start_recording but also stores target_pos and eef_to_target obs."""
        self._reach_target = np.array(target_pos, dtype=np.float32)
        self.current_demo = {
            "obs": {
                "robot0_joint_pos": [], # 7 joint angles
                "robot0_eef_pos": [], # XYZ of hand
                "robot0_eef_quat": [], # Orientation of hand
                "robot0_gripper_qpos": [], # [left_finger, right_finger] each = width/2
                "target_pos": [], # Target to which eef has to move to
                "eef_to_target": [], # Distance between eef and target
            },
            "actions": [],
            "states":  [],
        }
        self.prev_eef_pos  = None
        self.prev_eef_quat = None
        self.is_recording  = True
        print(f"Reach recording started,  target={self._reach_target.round(3)}")

    def read_robot_state(self):
        """Read raw state from robot, return (q, eef_pos, eef_quat, gripper_qpos, gripper_cmd)."""
        state = self.panda.get_state()
        q = np.array(state.q)
        T = np.array(state.O_T_EE).reshape(4, 4, order='F')
        eef_pos  = T[:3, 3]
        eef_quat = self.mat2quat(state.O_T_EE)
        gripper_state = self.gripper.read_once()
        width = gripper_state.width
        gripper_qpos = np.array([width / 2, width / 2])
        gripper_cmd  = np.array([(width / GRIPPER_MAX_WIDTH) * 2.0 - 1.0])
        return q, eef_pos, eef_quat, gripper_qpos, gripper_cmd

    def record_step(self):
        """Read one state from the robot and store it."""
        if not self.is_recording:
            return

        q, eef_pos, eef_quat, gripper_qpos, gripper_cmd = self.read_robot_state()

        self.current_demo["obs"]["robot0_joint_pos"].append(q)
        self.current_demo["obs"]["robot0_eef_pos"].append(eef_pos)
        self.current_demo["obs"]["robot0_eef_quat"].append(eef_quat)
        self.current_demo["obs"]["robot0_gripper_qpos"].append(gripper_qpos)
        self.current_demo["states"].append(np.concatenate([q, eef_pos, eef_quat, gripper_qpos]))

        if self.prev_eef_pos is not None:
            delta_pos = eef_pos - self.prev_eef_pos
            r_curr = Rotation.from_quat(eef_quat)
            r_prev = Rotation.from_quat(self.prev_eef_quat)
            delta_ori = (r_curr * r_prev.inv()).as_rotvec()
            action = np.concatenate([delta_pos, delta_ori, gripper_cmd])
        else:
            action = np.zeros(7)

        self.current_demo["actions"].append(action)
        self.prev_eef_pos = eef_pos
        self.prev_eef_quat = eef_quat

    def record_step_reach(self):
        """
        Like record_step but also stores target_pos and eef_to_target.
        Returns current distance to target so the caller can detect success.
        """
        if not self.is_recording:
            return 999.0

        q, eef_pos, eef_quat, gripper_qpos, gripper_cmd = self._read_robot_state()

        eef_to_target = self._reach_target - eef_pos
        dist = float(np.linalg.norm(eef_to_target))

        self.current_demo["obs"]["robot0_joint_pos"].append(q)
        self.current_demo["obs"]["robot0_eef_pos"].append(eef_pos)
        self.current_demo["obs"]["robot0_eef_quat"].append(eef_quat)
        self.current_demo["obs"]["robot0_gripper_qpos"].append(gripper_qpos)
        self.current_demo["obs"]["target_pos"].append(self._reach_target.copy())
        self.current_demo["obs"]["eef_to_target"].append(eef_to_target.astype(np.float32))
        self.current_demo["states"].append(np.concatenate([q, eef_pos, eef_quat, gripper_qpos]))

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

        return dist

        
    def stop_recording(self):
        self.is_recording = False
        self.gripper.move(GRIPPER_MAX_WIDTH, 0.05)

        actions = np.array(self.current_demo["actions"])
        states  = np.array(self.current_demo["states"])
        obs     = {k: np.array(v) for k, v in self.current_demo["obs"].items()}

        actions = actions[1:]
        states = states[:-1] # Drop last state to fix allignment with actions 
        obs = {k: v[:-1] for k, v in obs.items()} # Drop last obs to fix allignment actions

        demo = {"actions": actions, "states": states, "obs": obs}

        T = demo["actions"].shape[0]
        print(f"Recorded {T} steps ({T/self.record_hz:.1f}s)")
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
                og = dg.create_group("obs")
                for k, v in demo["obs"].items():
                    og.create_dataset(k, data=v)

            # train/val split
            n = len(self.demos)
            names = [f"demo_{i}" for i in range(n)]
            split = int(n * 0.8)
            mg = f.create_group("mask")
            mg.create_dataset("train", data=np.array(names[:split], dtype="S"))
            mg.create_dataset("valid", data=np.array(names[split:], dtype="S"))

        print(f"Saved {len(self.demos)} demos to {path}")



    def mat2quat(self, pose):
        """Convert column-major 4x4 transform to xyzw quaternion."""
        T = np.array(pose).reshape(4, 4, order='F')  # column-major
        return Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]

    def move_to_home(self):
        self.panda.move_to_start()