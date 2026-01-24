"""
Train an AI agent using reinforcement learning and make it independently play BattleZone video game for Atari.

Authors:
- Fabian Fetter
- Konrad Fijałkowski

Usage:
  python battlezone_agent.py --mode train --timesteps 100000
  python battlezone_agent.py --mode play --model battlezone_agent
"""

import argparse
import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Register ALE environments
gym.register_envs(ale_py)

class BattleZoneAgent:
    """
    Class used to train DQN model on BattleZone environment and play autonomously.
    """
    def __init__(self, model_path="models/battlezone_agent", verbose=1):
        self.model_path = model_path
        self.verbose = verbose
        self.model = None

    def train(self, total_timesteps=100000):
        """
        Train model on a number of steps (frames).
        """
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Create training environment
        env_train = make_atari_env("ALE/BattleZone-v5", n_envs=1, seed=42)
        env_train = VecFrameStack(env_train, n_stack=4)

        # Enhanced DQN hyperparameters
        self.model = DQN(
            "CnnPolicy",
            env_train,
            verbose=self.verbose,
            buffer_size=50000,          # Increased buffer size
            learning_starts=2500,       # Start learning a bit later
            target_update_interval=2000,# Update target network less frequently
            exploration_fraction=0.3,   # Explore for 30% of training
            exploration_final_eps=0.02, # Lower final exploration
            learning_rate=1e-4,         # Standard learning rate for DQN
            batch_size=32
        )

        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        env_train.close()

    def play(self):
        """
        Open the game window and let the trained model play.
        """
        if not os.path.exists(self.model_path + ".zip"):
            print(f"Model file {self.model_path}.zip not found. Please train first.")
            return

        print(f"Loading model from {self.model_path}")
        # Load the model - need to re-instantiate or load directly
        # We need the environment to know the action space, but load handles it usually if we pass env later or just predict
        # Ideally, we create a dummy env or load with env.
        
        # Create test environment
        env_test = make_atari_env(
            "ALE/BattleZone-v5",
            n_envs=1,
            seed=42,
            env_kwargs={"render_mode": "human"},
            wrapper_kwargs={"terminal_on_life_loss": False}
        )
        env_test = VecFrameStack(env_test, n_stack=4)

        self.model = DQN.load(self.model_path, env=env_test)

        obs = env_test.reset()
        done = False
        total_reward = 0

        print("Starting Game...")
        try:
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, dones, info = env_test.step(action)
                total_reward += reward
                done = dones[0] # VecEnv returns array of dones
                # Removed manual rendering loop as render_mode="human" handles it
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
        
        print(f"Game Over! Total Score: {total_reward}")
        env_test.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BattleZone RL Agent")
    parser.add_argument("--mode", type=str, default="play", choices=["train", "play"], help="Mode: train or play")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--model", type=str, default="models/battlezone_agent", help="Path to save/load model")
    
    args = parser.parse_args()

    agent = BattleZoneAgent(model_path=args.model)

    if args.mode == "train":
        agent.train(total_timesteps=args.timesteps)
    elif args.mode == "play":
        agent.play()