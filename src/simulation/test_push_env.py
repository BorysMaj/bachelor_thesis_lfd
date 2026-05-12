"""
Test the custom PushTask environment.
Run: python src/simulation/test_push_env.py
"""

import numpy as np
import robosuite as suite
from push_env import PushTask  # registers via @register_env

env = suite.make(
    "PushTask",
    robots="Panda",
    controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
    has_renderer=True,
    render_camera="frontview",
    has_offscreen_renderer=False,
    use_object_obs=True,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
)

obs = env.reset()
print("Obs keys:", list(obs.keys()))
print("box_pos:", obs["box_pos"])
print("box_to_goal:", obs["box_to_goal"])

for step in range(300):
    action = env.action_space.sample() * 0.1   # small random actions
    obs, reward, done, info = env.step(action)
    env.render()

    if step % 50 == 0:
        print(f"[{step}] reward={reward:.3f}  success={info.get('success', False)}")

    if done:
        print("Episode done")
        obs = env.reset()

env.close()
