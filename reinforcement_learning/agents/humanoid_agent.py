"""
Train an AI agent to make a Humanoid robot stand up using reinforcement learning.
Environment: HumanoidStandup-v5 (Mujoco)
Algorithm: PPO (Proximal Policy Optimization)

Authors:
- Fabian Fetter
- Konrad Fijałkowski

Usage:
  python humanoid_agent.py --mode train --timesteps 1000000
  python humanoid_agent.py --mode play --model humanoid_agent
"""

import argparse
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class HumanoidAgent:
    """
    Class used to train PPO model on HumanoidStandup environment.
    """
    def __init__(self, model_path="models/humanoid_agent", verbose=1):
        self.model_path = model_path
        self.verbose = verbose
        self.model = None

    def train(self, total_timesteps=1000000):
        """
        Train model on a number of steps.
        """
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Create vectorized environment for faster training
        # Mujoco environments are continuous control, so we don't need Atari wrappers
        env = make_vec_env("HumanoidStandup-v5", n_envs=1, seed=42)

        # PPO Hyperparameters optimized for continuous control
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=self.verbose,
            n_steps=2048,
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            learning_rate=3e-4,
            clip_range=0.2,
        )

        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        env.close()

    def play(self):
        """
        Visualize the trained agent.
        """
        if not os.path.exists(self.model_path + ".zip"):
            print(f"Model file {self.model_path}.zip not found. Please train first.")
            return

        print(f"Loading model from {self.model_path}")
        
        # Create environment with render mode
        env = gym.make("HumanoidStandup-v5", render_mode="human")

        self.model = PPO.load(self.model_path, env=env)

        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        print("Starting Simulation...")
        try:
            while not (done or truncated):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        
        print(f"Episode finished! Total Score: {total_reward}")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Humanoid Standup RL Agent")
    parser.add_argument("--mode", type=str, default="play", choices=["train", "play"], help="Mode: train or play")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total timesteps for training")
    parser.add_argument("--model", type=str, default="models/humanoid_agent", help="Path to save/load model")
    
    args = parser.parse_args()

    agent = HumanoidAgent(model_path=args.model)

    if args.mode == "train":
        agent.train(total_timesteps=args.timesteps)
    elif args.mode == "play":
        agent.play()
