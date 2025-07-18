import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snake_game.environment.core import Snake


def test_observation_shape_mlp():
    snake = Snake(width=5, height=5, policy="MlpPolicy")
    snake.init()
    obs = snake.observation()
    assert obs.shape == (5 * 5 * 3,)


def test_observation_shape_cnn():
    snake = Snake(width=5, height=5, policy="CnnPolicy")
    snake.init()
    obs = snake.observation()
    assert obs.shape == (3, 5, 5)


def test_move_step():
    snake = Snake(width=5, height=5)
    snake.init()
    old_x, old_y = snake.head.x, snake.head.y
    obs, reward, dead, truncated = snake.step(3)
    assert (snake.head.x, snake.head.y) == (old_x + 1, old_y)
    assert not dead


def test_no_reverse_direction():
    snake = Snake(width=5, height=5)
    snake.init()
    old_x = snake.head.x
    obs, reward, dead, truncated = snake.step(2)  # attempt to reverse direction
    assert snake.head.x == old_x + 1  # still moved right
    assert snake.direction == 3


def test_eat_food_growth():
    snake = Snake(width=5, height=5)
    snake.init()
    snake.food.block.move_to(snake.head.x + 1, snake.head.y)
    old_len = len(snake.body)
    obs, reward, dead, truncated = snake.step(3)
    assert len(snake.body) == old_len + 1
    assert snake.score == 1.0
    assert not dead


def test_dead_on_wall_collision():
    snake = Snake(width=5, height=5)
    snake.init()
    snake.direction = 2
    snake.head.x = 0
    obs, reward, dead, truncated = snake.step(None)
    assert dead


def test_truncated_flag_reaches_max_step():
    snake = Snake(width=5, height=5, max_step=1)
    snake.init()
    obs, reward, dead, truncated = snake.step(3)
    assert truncated


def test_visit_cell_recording_and_obs():
    snake = Snake(width=5, height=5, policy="MlpPolicy")
    snake.init()
    snake.food.block.move_to(0, 0)
    obs, reward, dead, truncated = snake.step(3)
    assert not dead
    assert (snake.head.x, snake.head.y) in snake.visited_cells
    grid = snake.observation().reshape(5, 5, 3)
    assert grid[snake.head.x][snake.head.y][2] == pytest.approx(0.1)


def test_observation_metadata_after_init():
    snake = Snake(width=5, height=5, policy="MlpPolicy")
    snake.init()
    snake.food.block.move_to(snake.head.x + 2, snake.head.y)
    grid = snake.observation().reshape(5, 5, 3)
    assert grid[0, 0, 1] == pytest.approx(Snake.observe_direction(3))
    assert grid[0, 1, 1] == pytest.approx(Snake.observe_direction(3))
    expected_len = len(snake.body) / (5 * 5)
    assert grid[0, 2, 1] == pytest.approx(expected_len)
    distance = ((snake.head.x - snake.food.block.x) ** 2 + (snake.head.y - snake.food.block.y) ** 2) ** 0.5
    max_dist = ((5 - 1) ** 2 + (5 - 1) ** 2) ** 0.5
    assert grid[0, 3, 1] == pytest.approx(distance / max_dist)
    assert grid[0, 4, 1] == pytest.approx(0.0)


def test_reward_increases_when_closer_to_food():
    snake = Snake(width=7, height=7)
    snake.init()
    snake.food.block.move_to(snake.head.x + 2, snake.head.y)
    _, r1, _, _ = snake.step(3)
    _, r2, _, _ = snake.step(0)
    assert r1 > r2


def test_reward_dead_penalty():
    snake = Snake(width=5, height=5, death_penalty=-1.0)
    snake.init()
    snake.direction = 2
    snake.head.x = 0
    _, reward, dead, _ = snake.step(None)
    assert dead and reward == -1.0
