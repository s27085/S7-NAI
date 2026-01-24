"""
Train a creative AI agent to land a spacecraft safely on the moon.
Environment: LunarLander-v3 (Box2D)
Algorithm: PPO

Authors:
- Fabian Fetter
- Konrad Fijałkowski

Usage:
  python lunar_lander_agent.py --mode train --timesteps 100000
  python lunar_lander_agent.py --mode play --model lunar_lander_agent
"""

import argparse
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class CreativeAgent:
    """
    Class used to train PPO model on LunarLander environment.
    """
    def __init__(self, model_path="models/lunar_lander_agent", verbose=1):
        self.model_path = model_path
        self.verbose = verbose
        self.model = None

    def train(self, total_timesteps=100000):
        """
        Train the model.
        """
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Vectorized environment
        env = make_vec_env("LunarLander-v3", n_envs=4, seed=42)

        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=self.verbose,
            n_steps=1024,
            batch_size=64,
            gae_lambda=0.98,
            gamma=0.999,
            n_epochs=4,
            ent_coef=0.01,
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
        
        env = gym.make("LunarLander-v3", render_mode="human")

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
        
        print(f"Episode finished. Total Score: {total_reward}")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creative LunarLander RL Agent")
    parser.add_argument("--mode", type=str, default="play", choices=["train", "play"], help="Mode: train or play")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--model", type=str, default="models/lunar_lander_agent", help="Path to save/load model")
    
    args = parser.parse_args()

    agent = CreativeAgent(model_path=args.model)

    if args.mode == "train":
        agent.train(total_timesteps=args.timesteps)
    elif args.mode == "play":
        agent.play()
