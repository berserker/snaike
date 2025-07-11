import os
import torch
import gymnasium as gym
import snake_game
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
import argparse
import matplotlib.pyplot as plt

def list_available_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available")

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.highest_reward = None

    def _on_step(self) -> bool:
        # Collect rewards from the environment
        reward = self.locals['rewards']
        self.episode_rewards.append(reward)
        return True

    def _on_rollout_end(self) -> None:
        # Print the total reward for the current rollout
        rewards = sum(self.episode_rewards)
                
        count = len(rewards)        
        if count > 1:
            best_reward = max(rewards)
            if self.highest_reward is None or best_reward > self.highest_reward:
                self.highest_reward = best_reward

            print(f"Rewards: best={best_reward} (highest={self.highest_reward}), worst={min(rewards)}, average={sum(rewards) / count}")
        else:
            best_reward = rewards
            print(f"Reward: {rewards}")

        '''
        plt.plot(self.episode_rewards)
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title('Reward per Step')
        plt.draw()
        plt.pause(0.001)
        plt.show(block=False)
        #plt.pause(0.001)'
        '''

        self.episode_rewards = []
    
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train PPO model for Snake game")
    parser.add_argument("--iterations", type=int, default=1, help="Number of training iterations")      # How often model's parameters myst be updated
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of training timesteps")    # How many steps (e.g.: "frames") to compute
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="Model policy to use")
    parser.add_argument("--cores", type=int, default=1, help="Number of parallel training environments")    
    parser.add_argument("--device", type=str, default="", help="Device to use ('cpu' or 'cuda', use '' for automatic selection)")    
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size")
    args = parser.parse_args()

    # Set the CUDA device to use the dedicated GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change "0" to the index of your dedicated GPU if necessary

    if not args.device:
        # List available GPUs
        list_available_gpus()
    
        # Check if GPU is available
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device

    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name()}")        
    else:
        print("Using CPU")

    options = {
        'init_length': 4,
        'width': args.grid_size,
        'height': args.grid_size,
        'block_size': 20,
        'policy': args.policy
    }

    def make_env():
        def _init():
            return gym.make("Snake-v1", **options)
        
        return _init

    # Create a vectorized environment with parallel environments
    
    env = SubprocVecEnv([make_env() for _ in range(args.cores)])

    model_path = f"ppo_snake-{args.policy}-{options['width']}x{options['height']}"

    # Default PPO hyperparameters
    """    
    ppo_params = {
        'learning_rate': get_schedule_fn(1e-3),  # The learning rate for the optimizer
        'n_steps': 2048,        # The number of steps to run for each environment per update
        'batch_size': 128,      # The number of samples in each mini-batch
        'n_epochs': 10,         # The number of epochs to optimize the loss function
        'gamma': 0.99,          # The discount factor. It determines the importance of future rewards
        'gae_lambda': 0.95,     # The lambda parameter for Generalized Advantage Estimation. It can help reduce variance in the advantage estimates.
        'ent_coef': 0.01,       # The coefficient for the entropy term. It encourages exploration by adding entropy to the loss function.
        'vf_coef': 0.5,         # The coefficient for the value function loss. It balances the importance of the value function loss and the policy loss.        
    }
    """    
    # Custom PPO hyperparameters: decrease gamma to force the agent to focus on clear local improvements (food eat).
    ppo_params = {
        'learning_rate': get_schedule_fn(1e-3),  # The learning rate for the optimizer
        'n_steps': 4096,        # The number of steps to run for each environment per update
        'batch_size': 256,      # The number of samples in each mini-batch
        'n_epochs': 20,         # The number of epochs to optimize the loss function
        'gamma': 0.25,          # The discount factor. It determines the importance of future rewards
        'gae_lambda': 0.95,     # The lambda parameter for Generalized Advantage Estimation. It can help reduce variance in the advantage estimates.
        'ent_coef': 0.01,       # The coefficient for the entropy term. It encourages exploration by adding entropy to the loss function.
        'vf_coef': 0.5,         # The coefficient for the value function loss. It balances the importance of the value function loss and the policy loss.        
    }

    # Check if the model file exists
    if os.path.exists(model_path + ".zip"):
        # Load the existing model
        model = PPO.load(model_path, env=env, device=device)
        print("Model loaded.")
        # Update model parameters
        model.learning_rate = ppo_params['learning_rate']
        model.n_steps = ppo_params['n_steps']
        model.batch_size = ppo_params['batch_size']
        model.n_epochs = ppo_params['n_epochs']
        model.gamma = ppo_params['gamma']
        model.gae_lambda = ppo_params['gae_lambda']
        model.ent_coef = ppo_params['ent_coef']
        model.vf_coef = ppo_params['vf_coef']
        print("Model parameters updated.")
    else:
        # Create a new model with custom hyperparameters
        model = PPO(args.policy, env, verbose=1, device=device, **ppo_params)
        print("New model created.")
    

    if args.iterations > 0:
        print(f"Trainining with policy={args.policy}, iterations={args.iterations}, timesteps={args.timesteps}")

        # Create the custom callback
        reward_logger = RewardLoggerCallback()

        for i in range(args.iterations):
            print(f"Train iteration #{i+1}")

            # Train the model with the custom reward callback
            model.learn(total_timesteps=args.timesteps, callback=reward_logger)

            # Save the model
            model.save(model_path)        
    
    # Create a single environment for rendering
    env = gym.make("Snake-v1", render_mode="human", **options)
    model = PPO.load(model_path, env=env, device=device)

    # Verify the device being used by the model
    print(f"Model is using device: {model.device}")

    # Enjoy the trained agent
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
