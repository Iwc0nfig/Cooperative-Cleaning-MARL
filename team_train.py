import numpy as np  
import torch 
import os
from re import search
import json 

from stable_baselines3 import PPO


import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize


from MyCallback import CleaningStatsCallback
from env_multiple_vacuum import MultipleVacuumEnv
from my_nn import MyCnnPolicy
from cuda_optim import setup_cuda_optimizations

# --- 1. LOAD CONFIGURATION ---
with open('config.json', 'r') as f:
    config = json.load(f)


# Extract parameters from the config file

env_params = config['environment_params']
reward_params = config['reward_params']
callback_params = config['callback_params']
ppo_params = config['ppo_params']


warmup_params = config["warnmup_params"]

curriculum_params = config['curriculum_params'] # LOAD THE NEW CURRICULUM CONFIG

# --- SETUP INITIAL PARAMS FROM CURRICULUM START ---
initial_overhead = curriculum_params['overhead_factor']['start']

#We create a tuple for the initial wall probability range
initial_wall_prob_range = (curriculum_params['wall_prob']['min_prob'], 
                           curriculum_params['wall_prob']['max_prob_start'])

ppo_params['learning_rate'] = curriculum_params['learning_rate']['start']
ppo_params['ent_coef'] = curriculum_params['ent_coef']['start']
ppo_params['clip_range'] = curriculum_params['clip_range']['start']
ppo_params['n_epochs'] = curriculum_params['n_epochs']['start']


load_model_path = config.get('load_model', None)



# --- WRAPPER AND POLICY (UNCHANGED) ---
class SharingWrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        # The observation_space is now the original dictionary space from the env
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.MultiDiscrete([5,5])
        
        self.current_agent = 0
        self.stored_obs_dict = {}
        # Use n_agents from config, which is correct
        self.stored_actions = [[0,0]] * env.max_agents  

        self.last_team_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self.last_info = {}
 
    def reset(self,**kwargs):
        # The 'obs' is now the full dictionary {"map": ..., "steps": ...}
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
        
        if self.current_agent >= self.env.n_agents:
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


        
# --- 2. SETUP POLICY KWARGS FOR MULTIINPUTPOLICY ---
# We are telling MultiInputPolicy to use your custom CNN when it sees the "map" key.
policy_kwargs = dict(
    features_extractor_class=MyCnnPolicy,
    features_extractor_kwargs=dict(features_dim=256),
    optimizer_class=torch.optim.AdamW,
    optimizer_kwargs=dict(
        weight_decay=1e-4
    )
)


# --- 3. INITIALIZE ENVIRONMENT, CALLBACK, AND MODEL USING CONFIG ---

if __name__ == "__main__":

    # --- CUDA CONFIG ---
    print(torch.cuda.is_available())
    device = setup_cuda_optimizations()

    # RTX Ada 4000 optimized CUDA configuration
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Define the environment creation function
    env_kwargs = dict(
        max_grid_shape=tuple(env_params['max_grid_shape']), 
        grid_shape_range=tuple(env_params['grid_shape_range']),
        agent_range=tuple(env_params['agent_range']),
        max_agents=env_params['max_agents'],
        initial_wall_prob_range=initial_wall_prob_range,
        reward_params=reward_params,
        overhead_factor=initial_overhead,
        warmup_stages = warmup_params["warmup_stages"],
    )
    
    num_cpu_cores = 160  #os.cpu_count()
    print(f"Using {num_cpu_cores} parallel environments.")
    vec_env = VecNormalize(
        make_vec_env(
            lambda: SharingWrapper(MultipleVacuumEnv(**env_kwargs)),
            n_envs=num_cpu_cores,
            vec_env_cls=SubprocVecEnv
        ),
        norm_obs=True,
        norm_reward=True,
        norm_obs_keys=['map','steps'], #We normalize only the map and steps .We don't nomalize the messages because we want to pass the the embeddings ther real message if we normilize we are going to loose usefull information
        clip_obs=10.0 # Clip obs to avoid outliers in normalization [-10, 10]
    )


# Initialize callback with parameters from config
    callback = CleaningStatsCallback(
        check_freq=callback_params['check_freq'],
        mean_clean_rate_limit=callback_params['mean_clearn_rate_limit'],
        curriculum_params=curriculum_params,
        **warmup_params
    )

    # Initialize PPO model with parameters from config

    model = PPO( 
        policy_kwargs=policy_kwargs, # Pass our specific instructions for the feature extractor
        env=vec_env,
        verbose=0, # Set to 1 to see progress
        device=device,
        **ppo_params # This unpacks the entire ppo_params dictionary
    )

    # --- 4. START TRAINING ---
    print("--- Starting training with dynamic curriculum ---")
    print(f"Initial Overhead: {initial_overhead}, Initial LR: {ppo_params['learning_rate']}")
    goal_oriented_timesteps = 1_000_000_000 
    model.learn(
        total_timesteps=goal_oriented_timesteps, 
        callback=callback
    )
    

