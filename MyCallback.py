import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os
from collections import deque

class CleaningStatsCallback(BaseCallback):

    def __init__(self, check_freq=100, verbose=0, mean_clean_rate_limit=0.85, warmup_clean_rate_limit=0.65, warmup_lr_optim=0.00025, warmup_mini_factor=1.6, warmup_coef=0.01, warmup_stages=dict, warmup_steps=8, warmup_max_prob=0.25 ,curriculum_params=None):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.mean_clean_rate_limit = mean_clean_rate_limit
        self.save_path = "models/"
        self.save_path_vec = "vec/"

        if curriculum_params is None:
            raise ValueError("curriculum_params must be provided")
        self.curriculum_params = curriculum_params

        self.overhead_cfg = self.curriculum_params["overhead_factor"]
        self.lr_cfg = self.curriculum_params["learning_rate"]
        self.ent_cfg = self.curriculum_params["ent_coef"]
        self.clip_cfg = self.curriculum_params["clip_range"]
        self.wall_prob_cfg = self.curriculum_params["wall_prob"]
        self.n_epochs_cfg = self.curriculum_params['n_epochs']

        self.warmup_clean_rate_limit = warmup_clean_rate_limit
        self.warmup_lr_optim = warmup_lr_optim
        self.warmup_mini_factor = warmup_mini_factor
        self.warmup_steps = warmup_steps
        self.warmup_max_prob = warmup_max_prob
        self.next_stage = 0
        self.warmup_current_stage = 1
        self.warmup_stages_count = len(warmup_stages) 
        self.warmup_coef = warmup_coef
        self.warmup = True
        self.warmup_overhead_factor_decrease_per_step = (self.overhead_cfg["start"] - self.warmup_mini_factor) / self.warmup_steps
        self.current_overhead = self.overhead_cfg["start"]

        self.wait_after_overhaerd = False
        # State variables for logging
        self.episode_count = 0
        self.successes = 0
        self.env_steps = []
        self.initial_dirt_sum_log = [] 
        self.final_dirt_sum_log = []  

        self.prefix_mean_success_rates = deque(maxlen=100)

        self.should_stop_training = False

    def _handle_warmup(self, mean_success_rate):
        """Handle warmup stage transitions"""

        if mean_success_rate >= self.warmup_clean_rate_limit:
            if self.warmup_current_stage <= self.warmup_stages_count:
                if self.next_stage < self.warmup_steps:
                    self.next_stage += 1

                    #Increase the wall prob
                    progress = self.next_stage / self.warmup_steps
                    start_max_prob = self.wall_prob_cfg['max_prob_start']
                    target_max_prob = self.warmup_max_prob
                    current_max_prob = start_max_prob + progress * (target_max_prob - start_max_prob)
                    new_wall_prob_range = (self.wall_prob_cfg['min_prob'], current_max_prob)
                    self.training_env.env_method('update_wall_prob_range', new_wall_prob_range)

                    # Log these changes
                    self.model.learning_rate = self.warmup_lr_optim
                    self.model.learn_rate = lambda _: self.warmup_lr_optim
                    self.model.ent_coef = self.warmup_coef
                    self.current_overhead -= self.warmup_overhead_factor_decrease_per_step # Corrected subtraction
                    self.training_env.env_method('update_overhead_factor', self.current_overhead)
                    self._update_optimizer(self.warmup_lr_optim)
                    print(f"  Warmup Step {self.next_stage}/{self.warmup_steps}. Overhead: {self.current_overhead:.2f}, LR: {self.warmup_lr_optim:.6f}, EntCoef: {self.warmup_coef:.4f}, WallProbMax: {current_max_prob:.3f}")

                else:
                    self.warmup_current_stage += 1
                    self.next_stage = 0
                    self.current_overhead = self.overhead_cfg["start"] # Reset overhead for new stage
                    self.model.ent_coef = self.ent_cfg["start"] # Reset ent_coef to its curriculum start value
                    self.training_env.env_method('set_warmup_current_stage', self.warmup_current_stage)
                    self.training_env.env_method('update_overhead_factor', self.current_overhead)
                    self._update_optimizer(self.lr_cfg['start']) # Use curriculum start LR for new stage

                    warmup_factor = 1.02
                    self.warmup_clean_rate_limit *= warmup_factor
                    print(f"--- WARMUP STAGE {self.warmup_current_stage} STARTING. New clean rate limit: {self.warmup_clean_rate_limit:.2f} ---")

                # Reset counters when advancing stages or steps
                self.episode_count, self.successes = 0, 0
                self.env_steps, self.initial_dirt_sum_log, self.final_dirt_sum_log = [], [], []
                self.prefix_mean_success_rates.clear() # Clear deque
            else:  # Stage 3 complete, exit warmup
                print(f"--- WARMUP COMPLETE. ADVANCING TO FULL CURRICULUM. ---")
                self.warmup = False
                #self.model.ent_coef = self.ent_cfg["start"]
                self.training_env.env_method('set_warmup', False)
                #self._update_optimizer(self.lr_cfg['start'])
                print("--- RESETTING SUCCESS RATE COUNTERS ---")
                self._update_curriculum()
                # Reset counters for curriculum
                self.episode_count, self.successes = 0, 0
                self.env_steps, self.initial_dirt_sum_log, self.final_dirt_sum_log = [], [], []
                self.prefix_mean_success_rates.clear() # Clear deque

    def _interpolate(self, current_overhead):
        max_overhead = self.overhead_cfg['start']
        mature_overhead_threshold = 1.1 
        min_overhead_for_progress = mature_overhead_threshold
        # ----------------------

        # Calculate progress over the range where annealing actually happens
        progress_denominator = max_overhead - min_overhead_for_progress
        if progress_denominator <= 0: # Avoid division by zero
             progress = 1.0
        else:
             progress = (max_overhead - current_overhead) / progress_denominator

        progress = np.clip(progress, 0.0, 1.0)

        # progress = (max_overhead - current_overhead) / (max_overhead - min_overhead)
        # progress = np.clip(progress, 0.0, 1.0) # Ensure progress is within [0, 1]

        lr = self.lr_cfg['start'] + progress * (self.lr_cfg['end'] - self.lr_cfg['start'])
       
        ent = self.ent_cfg['start'] + progress * (self.ent_cfg['end'] - self.ent_cfg['start'])
        clip = self.clip_cfg['start'] + progress * (self.clip_cfg['end'] - self.clip_cfg['start'])
        n_epochs_float = self.n_epochs_cfg['start'] + progress * (self.n_epochs_cfg['end'] - self.n_epochs_cfg['start'])
        n_epochs = int(round(n_epochs_float))

        min_prob = self.wall_prob_cfg['min_prob']
        wall_prob_start = self.wall_prob_cfg['max_prob_start']
        wall_prob_end = self.wall_prob_cfg['max_prob_end']
        current_max_prob = wall_prob_start + progress * (wall_prob_end - wall_prob_start)
        wall_prob_range = (min_prob, current_max_prob)

        return lr, ent, clip, wall_prob_range, n_epochs

    def _update_optimizer(self, new_lr):
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _update_curriculum(self):
        """Anneals the overhead and updates PPO hyperparameters accordingly."""
        current_overhead = self.training_env.get_attr("overhead_factor", 0)[0]
        new_overhead = max(self.overhead_cfg['min_factor'], current_overhead * self.overhead_cfg['annealing_rate'])

        if new_overhead >= current_overhead:
            print(f"\n--- MINIMUM OVERHEAD FACTOR ({self.overhead_cfg['min_factor']:.2f}) REACHED. ---")
            print("--- CURRICULUM COMPLETE. TRAINING WILL NOW STOP. ---")

            final_model_path = os.path.join(self.save_path, "final_model.zip")
            self.model.save(final_model_path)
            self.should_stop_training = True
            return
        
        final_squeeze_threshold = 1.1
        if new_overhead <= final_squeeze_threshold:
            # --- FINAL SQUEEZE LOGIC ---
            print(f"\n{'='*50}\n--- FINAL SQUEEZE (Overhead <= {final_squeeze_threshold}) ---\n{'='*50}")
            print(f"Annealing overhead_factor from {current_overhead:.3f} -> {new_overhead:.3f}")
            print("--- Hyperparameters are now FROZEN. ---")
            
            # ONLY update the overhead factor in the environment
            self.training_env.env_method('update_overhead_factor', new_overhead)
            
            # Log the new overhead, but nothing else changes
            self.logger.record("curriculum/overhead_factor", new_overhead)
        else:
            print(f"\n{'='*50}\n--- CURRICULUM UPDATE (Success Rate Target Met) ---\n{'='*50}")
            print(f"Annealing overhead_factor from {current_overhead:.3f} -> {new_overhead:.3f}")

            new_lr, new_ent_coef, new_clip_range, new_wall_prob_range, new_n_epochs = self._interpolate(new_overhead)
            print(f"Updating PPO: LR={new_lr:.6f}, EntCoef={new_ent_coef:.4f}, Clip={new_clip_range:.3f}, n_epochs={new_n_epochs}, WallProbRange=({new_wall_prob_range[0]:.2f}, {new_wall_prob_range[1]:.2f})")

            self.training_env.env_method('update_overhead_factor', new_overhead)
            self.training_env.env_method('update_wall_prob_range', new_wall_prob_range)

            # Apply Hyperparameter Changes to the Model
            self.model.learning_rate = new_lr
            self.model.ent_coef = new_ent_coef
            self.model.clip_range = new_clip_range
            self.model.n_epochs = new_n_epochs

            # Update schedules to return the new constant values
            self.model.learn_rate = lambda _: new_lr
            self.model.clip_range = lambda _: new_clip_range

            self._update_optimizer(new_lr)

            # Log the new values in TensorBoard
            self.logger.record("curriculum/overhead_factor", new_overhead)
            self.logger.record("curriculum/learning_rate", new_lr)
            self.logger.record("curriculum/ent_coef", new_ent_coef)
            self.logger.record("curriculum/clip_range", new_clip_range)
            self.logger.record("curriculum/wall_prob_min", new_wall_prob_range[0])
            self.logger.record("curriculum/wall_prob_max", new_wall_prob_range[1])


    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                info = self.locals["infos"][i]
                episode_stats = info.get("episode_stats", {})

                if "env_steps" in episode_stats:
                    self.env_steps.append(episode_stats["env_steps"])
                if "initial_dirt" in episode_stats:
                    self.initial_dirt_sum_log.append(episode_stats["initial_dirt"])
                if "final_dirt" in episode_stats:
                    self.final_dirt_sum_log.append(episode_stats["final_dirt"])

                if episode_stats.get("final_dirt") == 0:
                    self.successes += 1


        if self.episode_count >= self.check_freq:
            mean_success_rate = self.successes / self.episode_count if self.episode_count > 0 else 0.0
            self.prefix_mean_success_rates.append(mean_success_rate)

            current_overhead = self.training_env.get_attr("overhead_factor", 0)[0]
            current_lr = self.model.learning_rate # Get current LR from model directly
            current_ent_coef = self.model.ent_coef # Get current ent_coef from model directly


            print(f"\n--- Stats (Timestep {self.num_timesteps:_} | Overhead: {current_overhead:.2f}) ---")
            print(f"  Success Rate: {mean_success_rate:.2%} ({self.successes}/{self.episode_count})")
            print(f"  Last 100 Success Rate: {np.mean(self.prefix_mean_success_rates):.2%}")
            
            # More robust mean episode length
            mean_final_dirt = np.mean(self.final_dirt_sum_log) if self.final_dirt_sum_log else 0


            # --- TensorBoard Logging ---
            self.logger.record("custom/success_rate", mean_success_rate)
            self.logger.record("custom/mean_success_rate_last_100_episodes", np.mean(self.prefix_mean_success_rates))
            self.logger.record("custom/mean_final_dirt", mean_final_dirt)
            self.logger.record("hyperparameters/current_overhead_factor", current_overhead)
            self.logger.record("hyperparameters/current_learning_rate", current_lr)
            self.logger.record("hyperparameters/current_ent_coef", current_ent_coef)


            if self.warmup:
                self._handle_warmup(mean_success_rate)
                self.logger.record("warmup/warmup_stage", self.warmup_current_stage)
                print(f"  WARMUP STAGE : {self.warmup_current_stage}  | CLEAN RATE : {self.warmup_clean_rate_limit:.2f}")

            else:
                current_wall_prob_range = self.training_env.get_attr("wall_prob_range", 0)[0]
                self.logger.record("curriculum/current_wall_prob_min", current_wall_prob_range[0])
                self.logger.record("curriculum/current_wall_prob_max", current_wall_prob_range[1])
            
                if mean_success_rate >= self.mean_clean_rate_limit:
                    if self.wait_after_overhaerd:
                        self._update_curriculum()
                        # Save a model checkpoint each time the curriculum advances
                        if not os.path.exists(self.save_path): # Check if directory exists
                            os.makedirs(self.save_path)
                        model_file = os.path.join(self.save_path, f"model_overhead_{current_overhead:.2f}_ts_{self.num_timesteps}.zip")
                        self.model.save(model_file)
                        self.wait_after_overhaerd = False

                        if not os.path.exists(self.save_path_vec): # Check if directory exists
                            os.makedirs(self.save_path_vec)
                        stats_path = os.path.join(self.save_path_vec, f"vec_normalize_overhead_{current_overhead:.2f}_ts_{self.num_timesteps}.pkl")
                        self.model.env.save(stats_path)
                        self.prefix_mean_success_rates.clear()
                    else:
                        print("--- CURRICULUM UPDATE (1/2) ---")
                        self.wait_after_overhaerd = True
                    

                # Reset counters (moved outside the _update_curriculum call to always reset)
            self.episode_count, self.successes = 0, 0
            self.env_steps, self.initial_dirt_sum_log, self.final_dirt_sum_log = [], [], []
            

        return not self.should_stop_training