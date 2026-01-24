"""
Train an autonomous car to drive on a highway using reinforcement learning.
Environment: highway-fast-v0
Algorithm: DQN

Authors:
- Fabian Fetter
- Konrad Fijałkowski

Usage:
  python autonomous_car.py --mode train --timesteps 20000
  python autonomous_car.py --mode play --model autonomous_car
"""

import argparse
import os
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN

class AutonomousCarAgent:
    """
    Class used to train DQN model on Highway environment.
    """
    def __init__(self, model_path="models/autonomous_car", verbose=1):
        self.model_path = model_path
        self.verbose = verbose
        self.model = None

    def train(self, total_timesteps=20000):
        """
        Train the model.
        """
        print(f"Starting training for {total_timesteps} timesteps...")
        
        env = gym.make("highway-fast-v0")

        # Hyperparameters from highway-env documentation
        self.model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=self.verbose,
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
        
        # Create environment with render mode and adjusted speed
        env = gym.make(
            "highway-fast-v0", 
            render_mode="human",
            config={
                "real_time_rendering": True,
                "simulation_frequency": 30  # Higher = faster simulation
            }
        )

        self.model = DQN.load(self.model_path, env=env)

        obs, _ = env.reset()
        done = False
        truncated = False
        
        print("Starting Simulation...")
        try:
            while not (done or truncated):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        
        print("Episode finished.")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Car RL Agent")
    parser.add_argument("--mode", type=str, default="play", choices=["train", "play"], help="Mode: train or play")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total timesteps for training")
    parser.add_argument("--model", type=str, default="models/autonomous_car", help="Path to save/load model")
    
    args = parser.parse_args()

    agent = AutonomousCarAgent(model_path=args.model)

    if args.mode == "train":
        agent.train(total_timesteps=args.timesteps)
    elif args.mode == "play":
        agent.play()
