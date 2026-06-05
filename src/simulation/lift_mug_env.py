"""
Custom robosuite environment: LiftMug
Robot must lift a mug from the table.
Based on the Lift environment but uses MugObject instead of a cube.

The mug spawns upright (z-axis up) at a random position on the table.
"""

from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.models.objects import MujocoXMLObject
from robosuite.environments.base import register_env


class MugObject(MujocoXMLObject):
    """YCB 025_mug loaded from mug.xml in robosuite assets."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/mug.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


@register_env
class LiftMug(ManipulationEnv):
    """
    Lift the mug off the table.

    The mug spawns at a random XY position but always upright (rotation
    randomised only around the Z axis so the mug never spawns on its side).

    Success: mug centre is more than 4 cm above the table surface.

    Observations (state-based):
        robot0_eef_pos              (3,)
        robot0_eef_quat             (4,)
        robot0_gripper_qpos         (2,)
        mug_pos                     (3,)
        mug_quat                    (4,)
        robot0_gripper_to_mug_pos   (3,)
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
        reward_scale=1.0,
        reward_shaping=True,
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
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
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

    # Reward

    def reward(self, action=None):
        reward = 0.0

        if self._check_success():
            reward = 2.25
        elif self.reward_shaping:
            # reaching reward
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.mug.root_body,
                target_type="body",
                return_distance=True,
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.mug):
                reward += 0.25

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    # Model

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.mug = MugObject(name="mug")

        # Rotation only around Z axis so the mug always spawns upright.
        # rotation_axis="z" + rotation=(min, max) randomises yaw only.
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.mug)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.mug,
                x_range=[-0.15, 0.15],
                y_range=[-0.15, 0.15],
                rotation=(-np.pi, np.pi), # full yaw randomisation
                rotation_axis="z", # only rotate around Z
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.mug,
        )

    # References

    def _setup_references(self):
        super()._setup_references()
        self.mug_body_id = self.sim.model.body_name2id(self.mug.root_body)

    # Observables

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def mug_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.mug_body_id])

            @sensor(modality=modality)
            def mug_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.mug_body_id]), to="xyzw"
                )

            sensors = [mug_pos, mug_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors += [
                self._get_obj_eef_sensor(full_pf, "mug_pos", f"{arm_pf}gripper_to_mug_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]

            for s in sensors:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    # Reset

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

    # Visualize

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.mug
            )

    # Success

    def _check_success(self):
        mug_height = self.sim.data.body_xpos[self.mug_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        return mug_height > table_height + 0.04
