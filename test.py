import time
import os
import json
import torch
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_multiple_vacuum import MultipleVacuumEnv
from my_nn import MyCnnPolicy

# --- CONFIGURATION ---
# IMPORTANT: Update these paths to point to your trained model and VecNormalize stats.
MODEL_PATH =  "model/model_t.zip"
STATS_PATH = "model/vec_t.pkl"

# Number of episodes to test
N_TEST_EPISODES = 20
# Time to wait between steps to make the animation visible (in seconds)
FRAME_DELAY = 0.1

# --- CORRECTED WRAPPER (This is the new, required version) ---
class SharingWrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.MultiDiscrete([5,5])
        
        self.current_agent = 0
        self.stored_obs_dict = {} # We will store one obs per agent
        self.stored_actions = [[0,0]] * env.max_agents  

        self.last_team_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self.last_info = {}
 
    def reset(self,**kwargs):
        # The base env's reset now returns the obs for agent 0
        obs_agent_0 , info = self.env.reset(**kwargs)
        self.stored_obs_dict[0] = obs_agent_0
        self.current_agent = 0 

        self.last_team_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self.last_info = {}

        return self.stored_obs_dict[0] , info
        
    def step(self,action):
        # action is for self.current_agent
        self.stored_actions[self.current_agent] = action
        self.current_agent += 1
        
        # Use the actual number of agents in the current environment instance
        num_agents_in_env = self.env.n_agents

        if self.current_agent >= num_agents_in_env:
            # All agents have decided, now step the real environment
            obs_agent_0, rewards, terminated, truncated, info = self.env.step(self.stored_actions)
            
            self.stored_obs_dict.clear() # Clear old observations
            self.stored_obs_dict[0] = obs_agent_0 # Store the new obs for agent 0
            self.last_team_reward = np.mean(rewards)
            self.last_terminated = terminated
            self.last_truncated = truncated
            self.last_info = info
            self.current_agent = 0 # Reset for the next turn
            
            return self.stored_obs_dict[0], self.last_team_reward, self.last_terminated, self.last_truncated, self.last_info
        else:
            # It's not the end of the turn, get the observation for the *next* agent
            # We need to call the env's internal _get_obs method
            next_obs = self.env._get_obs(agent_id=self.current_agent)
            self.stored_obs_dict[self.current_agent] = next_obs
            
            return self.stored_obs_dict[self.current_agent], 0.0, False, False, {}

# --- HELPER FUNCTION FOR CLEARING SCREEN ---
def clear_screen():
    """Clears the console screen."""
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For macOS and Linux
    else:
        os.system('clear')

# --- MAIN EXECUTION (No changes needed below this line) ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        exit()
    if not os.path.exists(STATS_PATH):
        print(f"Error: VecNormalize stats file not found at '{STATS_PATH}'")
        exit()

    # --- 1. LOAD CONFIGURATION FOR ENVIRONMENT ---
    with open('config.json', 'r') as f:
        config = json.load(f)

    env_params = config['environment_params']
    reward_params = config['reward_params']
    warmup_params = config["warnmup_params"]
    curriculum_params = config['curriculum_params']
    
    initial_overhead = curriculum_params['overhead_factor']['start']
    initial_wall_prob_range = (curriculum_params['wall_prob']['min_prob'], curriculum_params['wall_prob']['max_prob_end'])

    # --- 2. CREATE THE EVALUATION ENVIRONMENT ---
    env_kwargs = dict(
        max_grid_shape=tuple(env_params['max_grid_shape']), 
        grid_shape_range=tuple((12,12)),
        agent_range=tuple((4,4)),
        max_agents=env_params['max_agents'],
        initial_wall_prob_range=initial_wall_prob_range,
        reward_params=reward_params,
        overhead_factor=1.15,
        warmup_stages=warmup_params["warmup_stages"],
        mode="testing"
    )

    # Note: It's better practice to access the underlying env for some attributes
    # We create a temporary env to get the true n_agents
    temp_env = MultipleVacuumEnv(**env_kwargs)
    eval_env = DummyVecEnv([lambda: SharingWrapper(temp_env)])
    
    eval_env = VecNormalize.load(STATS_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    

    # --- 3. LOAD THE TRAINED MODEL ---
    policy_kwargs = dict(
        features_extractor_class=MyCnnPolicy,
        features_extractor_kwargs=dict(features_dim=256),
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(
            weight_decay=1e-4
        )
    )
    
    
    model = PPO.load(MODEL_PATH, env=eval_env, policy_kwargs=policy_kwargs)
    print("Model and environment loaded successfully.\n")

    # --- 4. RUN THE EVALUATION LOOP ---
    for episode in range(N_TEST_EPISODES):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        step_expect = temp_env._calculate_dynamic_max_steps(4,temp_env.initial_dirt_sum)
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done_vec, info = eval_env.step(action)
            done = done_vec[0] 
            
            total_reward += reward[0]
            step_count +=1
            
            clear_screen()
            print(f"--- EPISODE {episode + 1}/{N_TEST_EPISODES} ---")
            rendered_grid = eval_env.envs[0].render()
            print(rendered_grid)
            print(f"\nStep: {step_count}")
            print(f"Reward for this step: {reward[0]:.4f}")
           
            print(f"Done: {done}")
            print(step_expect)
            print(temp_env.t)
            
            
            time.sleep(FRAME_DELAY)

        print("\nEpisode finished!")
        time.sleep(5)

    eval_env.close()
    print("\n--- Testing Complete ---")