import panda_py
import panda_py.controllers
import h5py
import numpy as np
import time
import json
from scipy.spatial.transform import Rotation

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
        self.is_recording = False
        self.demos = []


    def connect(self):
        self.panda = panda_py.Panda(self.robot_ip)
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
                "robot0_gripper_qpos": [], # Gripper open/close
            },
            "actions": [],
            "states":  [],
        }
        self.prev_eef_pos  = None
        self.prev_eef_quat = None
        self.is_recording  = True
        print("Recording started — move the arm!")

    def record_step(self):
        """Read one state from the robot and store it."""
        if not self.is_recording:
            return

        state = self.panda.get_state()

        # Observations
        q = np.array(state.q) # Joint angles
        eef_pos = np.array(state.O_T_EE[12:15]) # XYZ from transform matrix
        eef_quat = self.mat2quat(state.O_T_EE) # Orientation
        gripper = np.array([0.0, 0.0]) # Need to change -------------------------------------------

        self.current_demo["obs"]["robot0_joint_pos"].append(q)
        self.current_demo["obs"]["robot0_eef_pos"].append(eef_pos)
        self.current_demo["obs"]["robot0_eef_quat"].append(eef_quat)
        self.current_demo["obs"]["robot0_gripper_qpos"].append(gripper)
        self.current_demo["states"].append(np.concatenate([q, eef_pos, eef_quat]))

        # Action = delta eef from previous step
        if self.prev_eef_pos is not None:
            delta_pos  = eef_pos - self.prev_eef_pos
            delta_ori  = eef_quat[:3] - self.prev_eef_quat[:3]
            gripper_cmd = np.array([gripper[0]])
            action = np.concatenate([delta_pos, delta_ori, gripper_cmd])  # (7,)
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
                dg.create_dataset("states",  data=demo["states"])
                dg.create_dataset("rewards", data=np.zeros(T))
                dg.create_dataset("dones",   data=np.zeros(T))
                og = dg.create_group("obs")
                for k, v in demo["obs"].items():
                    og.create_dataset(k, data=v)

        print(f"Saved {len(self.demos)} demos to {path}")



    def mat2quat(self, pose):
    # Convert to 4x4 numpy matrix 
        T = np.array(pose).reshape(4, 4, order='F')  # column-major
        return Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]

    def move_to_home(self):
        self.panda.move_to_start()