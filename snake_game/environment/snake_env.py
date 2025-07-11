import gymnasium
from .core import Snake

class SnakeEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, **kwargs) -> None:
        self.render_mode = render_mode                
        self.snake = Snake(**kwargs)
        self.action_space = Snake.getActionSpace()
        self.observation_space = self.snake.getObservationSpace()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.snake.init()
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def step(self, action):
        s, r, d, t = self.snake.step(action)
        if self.render_mode == "human":
            self._render_frame()
        return s, r, d, t, self._get_info()

    def _get_obs(self):
        return self.snake.observation()         

    def _get_info(self):
        return self.snake.info()

    def _render_frame(self):
        self.snake.render()

    def close(self):
        self.snake.close()

    def play(self):
        self.snake.play()
