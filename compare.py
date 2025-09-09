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
# IMPORTANT: Update these paths to point to your trained models and VecNormalize stats.
MODEL_1_NAME = "Original Model"
MODEL_1_PATH = "model/model.zip"
STATS_1_PATH = "model/vec.pkl"

MODEL_2_NAME = "Tuned Model"
MODEL_2_PATH = "model/model_t.zip"
STATS_2_PATH = "model/vec_t.pkl"

# Number of different, unique maps to compare the models on
N_COMPARISON_EPISODES = 20

# --- WRAPPER (Required for the environment) ---
class SharingWrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.MultiDiscrete([5,5])
        
        self.current_agent = 0
        self.stored_obs_dict = {}
        self.stored_actions = [[0,0]] * env.max_agents
        self.last_team_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self.last_info = {}
 
    def reset(self,**kwargs):
        obs_agent_0 , info = self.env.reset(**kwargs)
        self.stored_obs_dict[0] = obs_agent_0
        self.current_agent = 0
        self.last_team_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self.last_info = {}
        return self.stored_obs_dict[0] , info
        
    def step(self,action):
        self.stored_actions[self.current_agent] = action
        self.current_agent += 1
        num_agents_in_env = self.env.n_agents
        if self.current_agent >= num_agents_in_env:
            obs_agent_0, rewards, terminated, truncated, info = self.env.step(self.stored_actions)
            self.stored_obs_dict.clear()
            self.stored_obs_dict[0] = obs_agent_0
            self.last_team_reward = np.mean(rewards)
            self.last_terminated = terminated
            self.last_truncated = truncated
            self.last_info = info
            self.current_agent = 0
            return self.stored_obs_dict[0], self.last_team_reward, self.last_terminated, self.last_truncated, self.last_info
        else:
            next_obs = self.env._get_obs(agent_id=self.current_agent)
            self.stored_obs_dict[self.current_agent] = next_obs
            return self.stored_obs_dict[self.current_agent], 0.0, False, False, {}

# --- HELPER FUNCTIONS ---
def check_files():
    """Checks if all required model and stats files exist."""
    paths = {
        "Model 1": MODEL_1_PATH, "Stats 1": STATS_1_PATH,
        "Model 2": MODEL_2_PATH, "Stats 2": STATS_2_PATH
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Error: {name} file not found at '{path}'")
            return False
    return True

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not check_files():
        exit()

    # --- 1. LOAD CONFIGURATION FOR ENVIRONMENT ---
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Use a challenging, fixed scenario for a fair comparison
    env_params = config['environment_params']
    env_params['agent_range'] = (4, 4)
    env_params['grid_shape_range'] = (12, 12)
    
    env_kwargs = dict(
        max_grid_shape=tuple(env_params['max_grid_shape']), 
        grid_shape_range=tuple(env_params['grid_shape_range']),
        agent_range=tuple(env_params['agent_range']),
        max_agents=env_params['max_agents'],
        initial_wall_prob_range=(0.1, 0.35), # Use a challenging wall probability
        reward_params=config['reward_params'],
        overhead_factor=1.3,
        warmup_stages=config["warnmup_params"]["warmup_stages"],
        mode="testing"
    )

    # --- 2. CREATE THE TWO EVALUATION ENVIRONMENTS ---
    print("Setting up environments and loading models...")
    
    env1 = DummyVecEnv([lambda: SharingWrapper(MultipleVacuumEnv(**env_kwargs))])
    eval_env1 = VecNormalize.load(STATS_1_PATH, env1)
    eval_env1.training = False
    eval_env1.norm_reward = False

    env2 = DummyVecEnv([lambda: SharingWrapper(MultipleVacuumEnv(**env_kwargs))])
    eval_env2 = VecNormalize.load(STATS_2_PATH, env2)
    eval_env2.training = False
    eval_env2.norm_reward = False

    # --- 3. LOAD THE TRAINED MODELS ---
    model1 = PPO.load(MODEL_1_PATH, env=eval_env1)
    model2 = PPO.load(MODEL_2_PATH, env=eval_env2)
    print("Setup complete. Starting comparison...\n")

    # --- 4. RUN THE COMPARISON ---
    map_seeds = [np.random.randint(0, 1e7) for _ in range(N_COMPARISON_EPISODES)]
    
    results = {MODEL_1_NAME: [], MODEL_2_NAME: []}
    wins = {MODEL_1_NAME: 0, MODEL_2_NAME: 0, "Tie": 0}

    for i, seed in enumerate(map_seeds):
        # --- Run Episode for Model 1 ---
        eval_env1.seed(seed)
        obs1 = eval_env1.reset()
        done1 = False
        steps1, reward1, success1 = 0, 0, False
        
        while not done1:
            action1, _ = model1.predict(obs1, deterministic=True)
            obs1, r, done_vec, info = eval_env1.step(action1)
            done1 = done_vec[0]
            reward1 += r[0]
            steps1 += 1
        
        success1 = info[0]['episode_stats']['final_dirt'] == 0
        results[MODEL_1_NAME].append({"steps": steps1, "reward": reward1, "success": success1})

        # --- Run Episode for Model 2 ---
        eval_env2.seed(seed)
        obs2 = eval_env2.reset()
        done2 = False
        steps2, reward2, success2 = 0, 0, False

        while not done2:
            action2, _ = model2.predict(obs2, deterministic=True)
            obs2, r, done_vec, info = eval_env2.step(action2)
            done2 = done_vec[0]
            reward2 += r[0]
            steps2 += 1

        success2 = info[0]['episode_stats']['final_dirt'] == 0
        results[MODEL_2_NAME].append({"steps": steps2, "reward": reward2, "success": success2})
        
        # --- Determine the Winner for this map ---
        winner = "Tie"
        if success1 and not success2:
            winner = MODEL_1_NAME
        elif success2 and not success1:
            winner = MODEL_2_NAME
        elif success1 and success2: # If both succeeded, efficiency is the tie-breaker
            if steps1 < steps2:
                winner = MODEL_1_NAME
            elif steps2 < steps1:
                winner = MODEL_2_NAME
        
        wins[winner] += 1
        # Print a one-line summary for each map
        print(f"Map {i+1:02d}: {MODEL_1_NAME} ({'S' if success1 else 'F'}-{steps1:03d}) vs {MODEL_2_NAME} ({'S' if success2 else 'F'}-{steps2:03d}) -> Winner: {winner}")


    # --- 5. PRINT FINAL SUMMARY ---
    print("\n\n" + "="*50)
    print("---      COMPARISON COMPLETE: FINAL RESULTS      ---")
    print("="*50)
    print(f"\nOverall Score after {N_COMPARISON_EPISODES} maps:")
    print(f"  - {MODEL_1_NAME} Wins: {wins[MODEL_1_NAME]}")
    print(f"  - {MODEL_2_NAME} Wins: {wins[MODEL_2_NAME]}")
    print(f"  - Ties: {wins['Tie']}")
    print("-" * 50)

    # Calculate and print detailed stats for each model
    for name in [MODEL_1_NAME, MODEL_2_NAME]:
        model_results = results[name]
        total_successes = sum(r['success'] for r in model_results)
        success_rate = (total_successes / N_COMPARISON_EPISODES) * 100
        
        successful_runs = [r for r in model_results if r['success']]
        if successful_runs:
            avg_steps_on_success = np.mean([r['steps'] for r in successful_runs])
        else:
            avg_steps_on_success = float('nan')

        print(f"\nStatistics for {name}:")
        print(f"  - Success Rate: {success_rate:.1f}% ({total_successes}/{N_COMPARISON_EPISODES})")
        print(f"  - Avg. Steps (on successful runs): {avg_steps_on_success:.2f}")
    
    print("\n" + "="*50)

    eval_env1.close()
    eval_env2.close()