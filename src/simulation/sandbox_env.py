"""
Custom robosuite environment: Sandbox
Multi-object environment for the user study.
Users choose their own task from 6 objects.

Objects:
    - Red cube          (BoxObject)
    - Blue cylinder     (CylinderObject)
    - Can               (CanObject)
    - Bin               (BinObject)
    - Hollow cylinder   (HollowCylinderObject)
    - Hammer            (HammerObject)

6 predefined positions in a half-circle in front of the robot.
On each reset the objects are randomly assigned to these positions.

No reward, no success criterion.
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.objects.primitive import CylinderObject
from robosuite.models.objects.xml_objects import CanObject
from robosuite.models.objects.composite.bin import Bin as BinObject
from robosuite.models.objects.composite.hollow_cylinder import HollowCylinderObject
from robosuite.models.objects.composite.hammer import HammerObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.environments.base import register_env


# 6 positions in a half-circle, all reachable from the robot base.
_HALF_CIRCLE_POSITIONS = [
    np.array([ -0.10,  0.25, 0.02]),   # slot 0 — far left
    np.array([ 0.05,  0.20, 0.02]),   # slot 1 — left
    np.array([ 0.15,  0.08, 0.02]),   # slot 2 — centre-left
    np.array([ 0.15, -0.08, 0.02]),   # slot 3 — centre-right
    np.array([ 0.05, -0.20, 0.02]),   # slot 4 — right
    np.array([ -0.10, -0.25, 0.02]),   # slot 5 — far right
]


@register_env
class Sandbox(ManipulationEnv):
    """
    Multi-object environment for user study demo collection.

    Six objects are placed in a half-circle in front of the robot.
    On every reset the objects are randomly shuffled across the 6 slots
    so each demonstration sees a different layout.

    Observations (state-based, no camera):
        robot0_eef_pos          (3,)
        robot0_eef_quat         (4,)
        robot0_gripper_qpos     (2,)
        cube_pos                (3,)
        cylinder_pos            (3,)
        can_pos                 (3,)
        bin_pos                 (3,)
        hollow_cylinder_pos     (3,)
        hammer_pos              (3,)
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

        # Blue cylinder
        self.cylinder = CylinderObject(
            name="cylinder",
            size_min=(0.02, 0.05),
            size_max=(0.02, 0.05),
            rgba=[0.1, 0.3, 0.9, 1],
            rng=self.rng,
        )

        # Can
        self.can = CanObject(name="can")

        # Bin
        self.bin = BinObject(name="bin",
            bin_size=(0.15, 0.15, 0.075)
        )

        # Hollow cylinder
        self.hollow_cylinder = HollowCylinderObject(name="hollow_cylinder")

        # Hammer — pass rng so handle/head sizes are deterministic per seed
        self.hammer = HammerObject(name="hammer",
            handle_radius=(0.012, 0.012),
            handle_length=(0.1, 0.1),
            rng=self.rng
            )

        self._all_objects = [
            self.cube,
            self.cylinder,
            self.can,
            self.bin,
            self.hollow_cylinder,
            self.hammer,
        ]

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self._all_objects,
        )


    # References

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.cylinder_body_id = self.sim.model.body_name2id(self.cylinder.root_body)
        self.can_body_id = self.sim.model.body_name2id(self.can.root_body)
        self.bin_body_id = self.sim.model.body_name2id(self.bin.root_body)
        self.hollow_cylinder_body_id = self.sim.model.body_name2id(self.hollow_cylinder.root_body)
        self.hammer_body_id = self.sim.model.body_name2id(self.hammer.root_body)


    # Reset — shuffle objects across the 6 half-circle slots

    def _reset_internal(self):
        super()._reset_internal()

        # Randomly assign each object to one of the 6 slots
        slot_indices = self.rng.permutation(len(_HALF_CIRCLE_POSITIONS))

        for i, obj in enumerate(self._all_objects):
            pos = self.table_offset + _HALF_CIRCLE_POSITIONS[slot_indices[i]]
            # Use the object's own init_quat if defined (e.g. hammer is horizontal),
            # otherwise use upright identity quaternion (wxyz format).
            if hasattr(obj, "init_quat"):
                quat = np.array(obj.init_quat)
            else:
                quat = np.array([1, 0, 0, 0])
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([pos, quat])
            )


    # Reward - none

    def reward(self, action=None):
        return 0.0


    # Success - user confirms via spacemouse/keyboard

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
            def bin_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.bin_body_id])

            @sensor(modality=modality)
            def hollow_cylinder_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.hollow_cylinder_body_id])

            @sensor(modality=modality)
            def hammer_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.hammer_body_id])

            for s in [cube_pos, cylinder_pos, can_pos,
                      bin_pos, hollow_cylinder_pos, hammer_pos]:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables