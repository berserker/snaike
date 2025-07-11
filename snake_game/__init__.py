from snake_game.version import VERSION as __version__
from snake_game.environment import SnakeEnv

from gymnasium.envs.registration import register

import gymnasium

__author__ = "berserker"

register(
    id="Snake-v1",
    entry_point="snake_game.environment:SnakeEnv",
)

def make(name, **kwargs):
    return gymnasium.make(name, **kwargs)
