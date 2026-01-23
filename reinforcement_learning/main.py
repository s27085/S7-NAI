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

gym.register_envs(ale_py)

class Agent:
    def __init__(self):
        self.env = gym.make("ALE/BattleZone-v5", render_mode="human")
        

        self.observation, self.info = self.env.reset()
        
        self.episode_over = False
        self.total_reward = 0

    def play(self):
        print("Starting Game...")
        while not self.episode_over:
            action = self.env.action_space.sample()

            self.observation, reward, terminated, truncated, self.info = self.env.step(action)
            
            self.total_reward += reward

            self.episode_over = terminated or truncated

        print(f"Game Over! Total Score: {self.total_reward}")
        self.env.close()

if __name__ == "__main__":
    agent = Agent()
    agent.play()