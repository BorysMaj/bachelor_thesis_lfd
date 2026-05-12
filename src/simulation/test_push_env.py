"""
Test the custom PushTask environment.
Run: python src/simulation/test_push_env.py
"""

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from push_env import PushTask  # registers via @register_env

controller_config = load_composite_controller_config(robot="Panda")

env = suite.make(
    "PushTask",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    render_camera="frontview",
    has_offscreen_renderer=True,
    use_object_obs=True,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
)

obs = env.reset()
print("Obs keys:", list(obs.keys()))
print("box_pos:", obs.get("box_pos", "not found"))
print("box_to_goal:", obs.get("box_to_goal", "not found"))

low, high = env.action_spec

for step in range(300):
    action = np.random.uniform(low,high) * 0.05   # small random actions
    obs, reward, done, info = env.step(action)
    env.render()

    if step % 50 == 0:
        print(f"[{step}] reward={reward:.3f}  success={env._check_success()}")

    if done:
        print("Episode done")
        obs = env.reset()

env.close()
