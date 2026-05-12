"""
Custom robosuite environment: PushTask
Robot must push a box to a goal region on the table.
No camera

Usage:
    import robosuite as suite
    from src.simulation.push_env import PushTask
    import robosuite.environments  # triggers registration

    # Register and create
    env = suite.make("PushTask", robots="Panda", has_renderer=True, ...)
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
import robosuite.utils.transform_utils as T
from robosuite.environments.base import register_env


@register_env
class PushTask(ManipulationEnv):
    """
    Push a box from its spawn position to a fixed goal region.

    Observations (state-based, no camera):
        robot0_eef_pos (3,) end-effector position
        robot0_eef_quat (4,) end-effector orientation
        robot0_gripper_qpos (2,) gripper finger positions
        box_pos (3,) box centre position
        box_to_goal (3,) vector from box to goal
    """

    def __init__(
        self,
        robots,
        
        goal_pos=(0.15, 0.15, 0.82), # fixed goal in world frame
        goal_threshold=0.05, # success radius (m)
        **kwargs,
    ):
        self.goal_pos = np.array(goal_pos)
        self.goal_threshold = goal_threshold
        self.box_body_id = None # set after model load

        super().__init__(robots=robots, **kwargs)

    # Build the scene

    def _load_model(self):
        super()._load_model()

        # Flat table arena
        mujoco_arena = TableArena(table_full_size=(0.8, 0.8, 0.05), table_friction=(1.0, 5e-3, 1e-4))
        mujoco_arena.set_origin([0, 0, 0])

        # Small box to push
        self.box = BoxObject(
            name="box",
            size_min=(0.025, 0.025, 0.025),
            size_max=(0.025, 0.025, 0.025),
            rgba=[0.8, 0.2, 0.2, 1],    # red
        )

        # Spawn box randomly in a region in front of robot
        placement_sampler = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.box,
            x_range=[-0.05, 0.05],
            y_range=[-0.05, 0.05],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=np.array([0.0, 0.0, 0.82]), # table surface ~0.82m
            z_offset=0.01,
        )

        # Assemble task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.box,
        )
        self.model.place_objects_in_grid(placement_sampler)


    # Reset — randomise box spawn each episode

    def _reset_internal(self):
        super()._reset_internal()
        self.box_body_id = self.sim.model.body_name2id(self.box.root_body)


    # Reward

    def reward(self, action=None):
        box_pos   = self.sim.data.body_xpos[self.box_body_id]
        dist      = np.linalg.norm(box_pos[:2] - self.goal_pos[:2])  # XY only

        # Dense: negative distance (encourages getting closer)
        r_dist    = -dist

        # Bonus: EEF near box (encourages making contact)
        eef_pos   = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        r_contact = -np.linalg.norm(eef_pos - box_pos) * 0.3

        # Sparse success bonus
        r_success = 1.0 if self._check_success() else 0.0

        return r_dist + r_contact + r_success


    # Success check

    def _check_success(self):
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        dist_xy = np.linalg.norm(box_pos[:2] - self.goal_pos[:2])
        return bool(dist_xy < self.goal_threshold)


    # Observations

    def _setup_observables(self):
        observables = super()._setup_observables()

        # Box position
        @sensor(modality="object")
        def box_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.box_body_id])

        # Vector from box to goal (helps policy know which direction to push)
        @sensor(modality="object")
        def box_to_goal(obs_cache):
            b = np.array(self.sim.data.body_xpos[self.box_body_id])
            return self.goal_pos - b

        sensors = [box_pos, box_to_goal]
        names   = [s.__name__ for s in sensors]

        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables
