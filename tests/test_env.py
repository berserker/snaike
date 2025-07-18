import os
import sys
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import snake_game


def test_env_observation_space_matches_reset():
    env = gym.make('Snake-v1', width=5, height=5, policy='MlpPolicy')
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    env.close()


def test_env_step_returns_correct_shapes():
    env = gym.make('Snake-v1', width=5, height=5, policy='MlpPolicy')
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    env.close()


def test_env_cnn_policy_shapes():
    env = gym.make('Snake-v1', width=5, height=5, policy='CnnPolicy')
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
    env.close()


def test_env_truncation_when_max_step_reached():
    env = gym.make('Snake-v1', width=5, height=5, max_step=1)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert truncated
    env.close()
