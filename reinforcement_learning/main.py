"""
Train an AI agent using reinforcement learning and make it independently play BattleZone video game for Atari 

Authors:
- Fabian Fetter
- Konrad Fijałkowski

Usage:
Run the script from the command line in the same directory as the main.py file. 

"""

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

class Agent:
    """
    Class used to train DQN model on BattleZone environment and play autonomously the game emulating Atari.
    - total_timesteps: specifies how long should the model be trained
    """
    def __init__(self):
        self.model = None
        self.total_timesteps=20000
        self.train_DQN_model()

    def train_DQN_model(self):
        """
        Train model on a number of steps (frames) to make it better at playing
        """
        env_train = make_atari_env("ALE/BattleZone-v5", n_envs=1, seed=0)
        env_train = VecFrameStack(env_train, n_stack=4)

        self.model = DQN(
            "CnnPolicy",
            env_train, 
            verbose=1, 
            buffer_size=20000,
            learning_starts=2000, 
            target_update_interval=1000,
            exploration_fraction=0.5,
            exploration_final_eps=0.05,
        )

        print("Starting model trainig")
        self.model.learn(self.total_timesteps)
        self.model.save("battlezone_agent")
        env_train.close()

    def play(self):
        """
        Open the game in a windows and show how the model RL-trained handles the environment
        """
        env_test = make_atari_env(
            "ALE/BattleZone-v5", 
            n_envs=1, 
            seed=0, 
            env_kwargs={"render_mode": "human"},
            wrapper_kwargs={"terminal_on_life_loss": False}
        )
        env_test = VecFrameStack(env_test, n_stack=4)
        gym.register_envs(ale_py)

        obs = env_test.reset()
        done = False
        total_reward = 0

        print("Starting Game...")
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
    
            obs, reward, dones, info = env_test.step(action)
            total_reward += reward
            done = dones[0]

        print(f"Game Over! Total Score: {total_reward}")
        env_test.close()

if __name__ == "__main__":
    agent = Agent()
    agent.play()