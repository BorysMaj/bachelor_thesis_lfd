from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import SquareNutObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.base import register_env

@register_env
class Nut(ManipulationEnv):
    """
    Lifting task using a square nut object instead of a cube.
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
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
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
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
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

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:
            - a discrete reward of 2.25 is provided if the nut is lifted

        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, 1], to encourage the arm to reach the nut
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the nut
            - Lifting: in {0, 1}, non-zero if arm has lifted the nut
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.nut.root_body,
                target_type="body",
                return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.nut):
                reward += 0.25

        # scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize square nut object
        self.nut = SquareNutObject(name="SquareNut")

        # create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.nut)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.nut,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.nut,
        )

    def _setup_references(self):
        """
        Sets up references to important components.
        """
        super()._setup_references()

        # nut body reference
        self.nut_body_id = self.sim.model.body_name2id(self.nut.root_body)

        # nut handle site reference
        self.nut_handle_site_id = self.sim.model.site_name2id(
            self.nut.important_sites["handle"]
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment.
        """
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            # nut position observable
            @sensor(modality=modality)
            def nut_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.nut_body_id])

            # nut quaternion observable
            @sensor(modality=modality)
            def nut_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.nut_body_id]), to="xyzw"
                )

            # nut handle position observable
            @sensor(modality=modality)
            def nut_handle_pos(obs_cache):
                return np.array(
                    self.sim.data.site_xpos[self.nut_handle_site_id]
                )

            sensors = [nut_pos, nut_quat, nut_handle_pos]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to nut position sensor
            sensors += [
                self._get_obj_eef_sensor(full_pf, "nut_pos", f"{arm_pf}gripper_to_nut_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]

            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

    def visualize(self, vis_settings):
        """
        Visualize gripper site proportional to distance to the nut.
        """
        super().visualize(vis_settings=vis_settings)

        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.nut
            )

    def _check_success(self):
        """
        Check if nut has been lifted.

        Returns:
            bool: True if nut is lifted above table
        """
        nut_height   = self.sim.data.body_xpos[self.nut_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        return nut_height > table_height + 0.10