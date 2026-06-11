"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.

Modifications made:
    - User can confirm a demo as successful by pressing keyboard 'e'
      or pressing both SpaceMouse buttons simultaneously.
    - When confirmed, the demo is marked as successful and the episode ends.
    - Uses UserStudyEnv by default.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode,
                             demo_num=None, target=None):
    """
    Use the device (SpaceMouse) to collect a demonstration.

    The user confirms the demo is successful by pressing keyboard 'e'
    or both SpaceMouse buttons simultaneously. This marks the episode
    as successful and ends it. Pressing the right SpaceMouse button
    discards the demo (reset without saving as successful).

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arm (str): which arm to control 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
        goal_update_mode (str): mode to update goal in
        demo_num (int): current demo number (for display)
        target (int): target number of demos (for display)
    """

    env.reset()
    env.render()

    # Progress counter
    if demo_num is not None and target is not None:
        bar_len = 20
        filled = int(bar_len * (demo_num - 1) / target)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n{'='*50}")
        print(f"  Demo {demo_num} / {target}   [{bar}]")
        print(f"{'='*50}")
    else:
        print("\n New demonstration")

    print("Perform the task, then press 'e' or both SpaceMouse buttons to confirm.")
    print("Press right SpaceMouse button to discard and reset.\n")

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            print("Demo discarded (reset).")
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Check if user confirmed the demo before stepping
        # so the confirmation is recorded in this step
        if getattr(device, "_demo_confirmed", False):
            print("Demo confirmed as successful! Saving")
            # Write a marker file in the episode directory.
            if hasattr(env, "ep_directory") and env.ep_directory is not None:
                with open(os.path.join(env.ep_directory, "confirmed"), "w") as marker:
                    marker.write("1")

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        env.render()

        # Break after stepping if demo was confirmed
        # Also write marker here in case confirmation arrived during env.step()
        if getattr(device, "_demo_confirmed", False):
            if hasattr(env, "ep_directory") and env.ep_directory is not None:
                confirmed_path = os.path.join(env.ep_directory, "confirmed")
                if not os.path.exists(confirmed_path):
                    print("Demo confirmed as successful! Saving")
                    with open(confirmed_path, "w") as marker:
                        marker.write("1")
            break

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The structure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Success is determined by either:
        # 1. The env's own _check_success() — works for Lift, Push, Reach, etc.
        # 2. A "confirmed" marker file written when user pressed 'e' or both
        confirmed_path = os.path.join(directory, ep_directory, "confirmed")
        success = success or os.path.exists(confirmed_path)


        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
            print(f"Demonstration saved: {num_eps}")

        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_args"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Sandbox")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    parser.add_argument(
        "--goal_update_mode",
        type=str,
        default="target",
        choices=["target", "achieved"],
        help="Used by the device to get the arm's actions. The mode to update the goal in. Can be 'target' or 'achieved'. If 'target', the goal is updated based on the current target pose. "
        "If 'achieved', the goal is updated based on the current achieved state. "
        "We recommend using 'achieved' (and input_ref_frame='base') if collecting demonstrations with a mobile base robot.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=20,
        help="Target number of successful demonstrations to collect. Shows a progress counter.",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    # if WHOLE BODY IK; assert only one robot
    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string.
    # Must match the format robomimic expects: env_name, type (1 = robosuite), env_kwargs.
    env_args = {
        "env_name": config["env_name"],
        "type": 1,  # robomimic EnvType.ROBOSUITE_TYPE
        "env_kwargs": {
            "robots": config["robots"],
            "controller_configs": config["controller_configs"],
        },
    }
    env_info = json.dumps(env_args)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    successful_demos = 0
    while True:
        collect_human_trajectory(
            env, device, args.arm, args.max_fr, args.goal_update_mode,
            demo_num=successful_demos + 1,
            target=args.target,
        )
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)

        # count successful demos from the hdf5
        import h5py
        try:
            with h5py.File(os.path.join(new_dir, "demo.hdf5"), "r") as f:
                successful_demos = len(f["data"].keys())
        except Exception:
            pass

        if successful_demos >= args.target:
            print(f"\n{'='*50}")
            print(f"  Target reached! {successful_demos} / {args.target} demos collected.")
            print(f"  Press Ctrl+C to finish.")
            print(f"{'='*50}\n")