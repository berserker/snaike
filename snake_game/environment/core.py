import math
import numpy as np
from .utils import *
from gymnasium import spaces

class Snake:
    def __init__(
        self,
        fps=30,
        max_step=-1,
        init_length=4,
        food_reward=1.0,
        dist_scaler=1.0,
        living_bonus=0.01,
        death_penalty=-1.0,
        visited_penalty=-0.1,
        width=16,
        height=16,
        block_size=20,
        background_color=Color.black,
        food_color=Color.green,
        head_color=Color.grey,
        body_color=Color.white,
        policy="MlpPolicy"
    ) -> None:

        self.episode = 0
        self.fps = fps
        self.max_step = max_step
        self.init_length = min(init_length, width//2)
        self.food_reward = food_reward
        self.dist_scaler = dist_scaler
        self.living_bonus = living_bonus
        self.death_penalty = death_penalty
        self.visited_penalty = visited_penalty
        self.blocks_x = width
        self.blocks_y = height
        self.food_color = food_color
        self.head_color = head_color
        self.body_color = body_color
        self.background_color = background_color
        self.food = Food(self.blocks_x, self.blocks_y, food_color)
        self.policy = policy
        Block.size = block_size

        self.screen = None
        self.clock = None
        self.human_playing = False
        self.prev_distance = None
        self.max_distance = math.sqrt((self.blocks_x-1)**2 + (self.blocks_y-1)**2)
        self.max_score = float((self.blocks_x * self.blocks_y) - self.init_length)
        self.prev_score = 0.0    

    @staticmethod
    def getActionSpace():
        return spaces.Discrete(4)

    def getObservationSpace(self):
        """
        # Evaluate:
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 1, shape=(self.grid_size[0], self.grid_size[1], 3), dtype=np.float32),
            "direction": spaces.Box(0, 1, shape=(8,), dtype=np.float32)
        })"
        """

        match self.policy:
            case "CnnPolicy":
                return spaces.Box(0.0, 1.0, shape=(3, self.blocks_x, self.blocks_y), dtype=np.float32)
            case "MlpPolicy":
                return spaces.Box(0.0, 1.0, shape=(self.blocks_x * self.blocks_y * 3,), dtype=np.float32)
            
        raise ValueError(f"Unknown policy: {self.policy}")

    def init(self):
        self.episode += 1
        self.score = 0.0        
        self.prev_score = 0.0
        self.direction = 3
        self.prev_direction = self.direction
        self.current_step = 0
        self.prev_distance = None
        self.head = Block(self.blocks_x//2, self.blocks_y//2, self.head_color)
        self.body = [self.head.copy(i, 0, self.body_color)
                     for i in range(-self.init_length, 0)]
        self.blocks = [self.food.block, self.head, *self.body]
        self.food.new_food(self.blocks)
        self.visited_cells = {}

    def close(self):
        pygame.quit()
        pygame.display.quit()
        self.screen = None
        self.clock = None

    def render(self):
        if self.screen is None:
            self.screen, self.clock = game_start(
                self.blocks_x*Block.size, self.blocks_y*Block.size)
        self.clock.tick(self.fps)
        update_screen(self.screen, self)
        handle_input()

    def step(self, direction):
        if direction is None:
            direction = self.direction
        
        self.prev_direction = self.direction

        self.current_step += 1        
        truncated = True if self.current_step == self.max_step else False
        (x, y) = (self.head.x, self.head.y)
        step = Direction.step(direction)
        if (direction == 0 or direction == 1) and (self.direction == 0 or self.direction == 1):
            step = Direction.step(self.direction)
        elif (direction == 2 or direction == 3) and (self.direction == 2 or self.direction == 3):
            step = Direction.step(self.direction)
        else:
            self.direction = direction
        self.head.x += step[0]
        self.head.y += step[1]

        distance = self.distance_from_food()  # Current distance to food
        if self.prev_distance is None:  
            self.prev_distance = distance  # Initialize on the first call

        eat = self.head == self.food.block
        dead = False

        if eat:
            self.score += 1.0
        else:
            self.move(x, y)
            for block in self.body:
                if self.head == block:
                    dead = True
            if self.head.x >= self.blocks_x or self.head.x < 0 or self.head.y < 0 or self.head.y >= self.blocks_y:
                dead = True

        visited = False if dead else self.visited_cells.get((self.head.x, self.head.y), False)

        reward = self.calc_reward(eat, dead, visited, distance, self.prev_distance)

        if eat:
            self.grow(x, y)
            self.food.new_food(self.blocks)            
            self.prev_distance = self.distance_from_food()
            self.visited_cells = {}
        elif not dead:
            self.visited_cells[(self.head.x, self.head.y)] = True
            self.prev_distance = distance  # Update for the next step

        return self.observation(), reward, dead, truncated

    def distance_from_food(self):
        x = self.head.x - self.food.block.x
        y = self.head.y - self.food.block.y
        return math.sqrt(x*x + y*y)  # Current distance to food

    @staticmethod
    def observe_direction(dir):
        match dir:
            case 0:
                return 0.2
            case 1:
                return 0.4
            case 2:
                return 0.6
            case 3:
                return 0.8
            
    def observation(self):
        # Channel 0: grid representation
        # Channel 1: game state (eg. snake direction)
        # Channel 2: visited cells

        obs = np.zeros((self.blocks_x, self.blocks_y, 3), dtype=np.float32)

        # Represent the head of the snake
        if 0 <= self.head.x < self.blocks_x and 0 <= self.head.y < self.blocks_y:
            obs[self.head.x][self.head.y][0] = 0.2
        
        # Represent the body of the snake
        for block in self.body:
            if 0 <= block.x < self.blocks_x and 0 <= block.y < self.blocks_y:
                obs[block.x][block.y][0] = 0.4
        
        # Represent the food
        obs[self.food.block.x][self.food.block.y][0] = 0.6

        # Save metadata

        obs[0][0][1] = Snake.observe_direction(self.prev_direction)
        obs[0][1][1] = Snake.observe_direction(self.direction)
       
        snake_length_normalized = len(self.body) / (self.blocks_x * self.blocks_y)
        obs[0, 2, 1] = snake_length_normalized

        distance_to_food_normalized = self.distance_from_food() / self.max_distance
        obs[0, 3, 1] = distance_to_food_normalized

        score_normalized = self.score / self.max_score
        obs[0, 4, 1] = score_normalized

        # Represent the walls

        obs[:, 0, 2] = 0.5                     # Left wall
        obs[:, self.blocks_y - 1, 2] = 0.5     # Right wall
        obs[0, :, 2] = 0.5                     # Top wall
        obs[self.blocks_x - 1, :, 2] = 0.5     # Bottom wall

        # Represent the visited cells

        for (x, y), visited  in self.visited_cells.items():
            if visited:
                obs[x][y][2] += 0.1

        return self.__transposeObservation(obs)

    def __transposeObservation(self, obs):           

        match self.policy:
            case "CnnPolicy":
                # Rearrange the data from (width, height, channels) into (channels, height, width) as required by CnnPolicy
                obs = obs.transpose(2, 1, 0)
            case "MlpPolicy":
                # Flatten the observation for MlpPolicy
                obs = obs.flatten()

        return obs


    def calc_reward(self, eat, dead, visited, distance, prev_distance):    
        if dead:
            return self.death_penalty
        
        reward = self.living_bonus + (self.score / self.max_score)
        
        # (self.prev_distance - distance): positive if closer, negative if farther
        dist_reward = (prev_distance - distance) / self.max_distance        
        dist_reward *= self.dist_scaler

        reward += dist_reward

        if eat:
            reward += self.food_reward
        elif visited:
            reward += self.visited_penalty

        return reward  

    def grow(self, x, y):
        body = Block(x, y, self.body_color)
        self.blocks.append(body)
        self.body.append(body)

    def move(self, x, y):
        tail = self.body.pop(0)
        tail.move_to(x, y)
        self.body.append(tail)

    def info(self):
        return {
            'head': (self.head.x, self.head.y),
            'food': (self.food.block.x, self.food.block.y),
        }

    def play(self, fps=10, acceleration=True, step=1, frep=10):
        self.max_step = 99999
        self.fps = fps

        self.human_playing = True
        self.init()
        screen, clock = game_start(
            self.blocks_x*Block.size, self.blocks_y*Block.size)
        total_r = 0

        while pygame.get_init():
            clock.tick(self.fps)
            _, r, d, _ = self.step(handle_input())
            total_r += r
            if acceleration and total_r == frep:
                self.fps += step
                total_r = 0
            if d:
                self.init()
                total_r = 0
                self.fps = fps
            update_screen(screen, self, True)
