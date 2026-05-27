"""
Custom robosuite environment: Sandbox
Multi-object environment for the user study.
Users choose their own task from 4 objects.

Objects (fixed positions):
    - Red cube      (BoxObject)      -- left
    - Blue cylinder (CylinderObject) -- front-left
    - Can           (CanObject)      -- front-right
    - Green ball    (BallObject)     -- right

No reward, no success criterion.
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.objects.primitive import CylinderObject, BallObject
from robosuite.models.objects.xml_objects import CanObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.environments.base import register_env


@register_env
class Sandbox(ManipulationEnv):
    """
    Multi-object environment for user study demo collection.

    Four objects are placed in front of the robot arm,
    making them all easily reachable and visible.

    Observations (state-based, no camera):
        robot0_eef_pos       (3,)  end-effector position
        robot0_eef_quat      (4,)  end-effector orientation
        robot0_gripper_qpos  (2,)  gripper finger positions
        cube_pos             (3,)  red cube position
        cylinder_pos         (3,)  blue cylinder position
        can_pos              (3,)  can position
        ball_pos             (3,)  green ball position
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=500,
        ignore_done=True,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction  = table_friction
        self.table_offset    = np.array((0, 0, 0.8))
        self.reward_shaping  = reward_shaping
        self.use_object_obs  = use_object_obs
        self.placement_initializer = placement_initializer

        # Fixed positions relative to table_offset.
        # Half-circle layout: objects fan out in front of the robot (along +x)
        # and spread across y-axis so all are easily reachable.
        self._fixed_positions = {
            "cube":     np.array([ 0.05,  0.22, 0.02]),   # left
            "cylinder": np.array([ 0.22,  0.10, 0.02]),   # front-left
            "can":      np.array([ 0.22, -0.10, 0.02]),   # front-right
            "ball":     np.array([ 0.05, -0.22, 0.02]),   # right
        }

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )


    # Build the scene

    def _load_model(self):
        super()._load_model()

        # Position robot at table edge
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Red cube
        self.cube = BoxObject(
            name="cube",
            size_min=(0.025, 0.025, 0.025),
            size_max=(0.025, 0.025, 0.025),
            rgba=[0.8, 0.1, 0.1, 1],
            rng=self.rng,
        )

        # Blue cylinder (radius=0.02, half-height=0.05)
        self.cylinder = CylinderObject(
            name="cylinder",
            size_min=(0.02, 0.05),
            size_max=(0.02, 0.05),
            rgba=[0.1, 0.3, 0.9, 1],
            rng=self.rng,
        )

        # Can (XML object)
        self.can = CanObject(name="can")

        # Green ball (radius=0.025)
        self.ball = BallObject(
            name="ball",
            size_min=(0.025,),
            size_max=(0.025,),
            rgba=[0.1, 0.8, 0.1, 1],
            rng=self.rng,
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube, self.cylinder, self.can, self.ball],
        )


    # References

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id     = self.sim.model.body_name2id(self.cube.root_body)
        self.cylinder_body_id = self.sim.model.body_name2id(self.cylinder.root_body)
        self.can_body_id      = self.sim.model.body_name2id(self.can.root_body)
        self.ball_body_id     = self.sim.model.body_name2id(self.ball.root_body)


    # Reset - place all objects at their fixed positions

    def _reset_internal(self):
        super()._reset_internal()
        for obj, key in [
            (self.cube, "cube"),
            (self.cylinder, "cylinder"),
            (self.can, "can"),
            (self.ball, "ball"),
        ]:
            pos  = self.table_offset + self._fixed_positions[key]
            quat = np.array([1, 0, 0, 0])  # upright, no rotation
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([pos, quat])
            )


    # Reward - none

    def reward(self, action=None):
        return 0.0


    # Success - no predefined criterion

    def _check_success(self):
        return False


    # Observations

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cylinder_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cylinder_body_id])

            @sensor(modality=modality)
            def can_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.can_body_id])

            @sensor(modality=modality)
            def ball_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.ball_body_id])

            for s in [cube_pos, cylinder_pos, can_pos, ball_pos]:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables
