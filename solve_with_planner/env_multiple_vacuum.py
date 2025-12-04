import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import sys
from scipy.ndimage import label


MOVE_MAP = {
    0:np.array([0,0], dtype=int),
    1:np.array([-1,0], dtype=int),
    2:np.array([1,0], dtype=int),
    3:np.array([0,-1], dtype=int),
    4:np.array([0,1], dtype=int)
}

OPTIONS = {
    5: "CLEAN_NEAREST_CLUSTER", # Find the closest group of dirt and move towards it
    6: "DISPERSE",              # Move towards the quadrant with the fewest other agents
}
TOTAL_MOVE_ACTIONS = len(MOVE_MAP) + len(OPTIONS)


class MultipleVacuumEnv(gym.Env):
    metadata = {"render_modes":["ansi"],"render_fps":4}
        
    def __init__(self, max_grid_shape=(10,10),grid_shape_range=(7,12),agent_range=(2,4),max_agents=4, initial_wall_prob_range=(0.1, 0.15),
                 obs_mode="global", seed=None, render_mode=None, reward_params=None , overhead_factor=1.6,warmup_stages=None, mode="training",communication_params=None): 
        self.max_grid_shape = max_grid_shape
        self.max_H, self.max_W = max_grid_shape

        self.grid_shape_range = grid_shape_range  # New range for random grid sizes
        self.H, self.W = self.max_H, self.max_W
        self.current_size = grid_shape_range[0]  # Start with the minimum size

        #agents configs
        self.agent_range = agent_range
        self.max_agents = max_agents
        

        self.wall_prob_range = initial_wall_prob_range
        self.obs_mode = obs_mode

        self.seed_val = seed
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self.overhead_factor = overhead_factor  
        self.BASELINE_DIRT_AREA = (self.grid_shape_range[0] - 2) * (self.grid_shape_range[0] - 2)  # Rough estimate of average dirt
        #self.action_space = spaces.MultiDiscrete([5]*self.max_agents)

        self.warmup = True  if mode =="training" else False
        self.warmup_stage = 1
        self.warmup_stages = warmup_stages
        self.reward_params = reward_params 

        if communication_params is None:
            raise ValueError("communication_params must be provided")
        self.num_message_types = communication_params['num_message_types']
        
        if obs_mode == "global":
            n_channels = 4 
            self.observation_space = spaces.Dict({
                "map": spaces.Box(low=0, high=1, shape=(n_channels, self.max_H, self.max_W), dtype=np.uint8),
                "steps": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "messages": spaces.Box(low=0, high=1, shape=(self.max_agents,3), dtype=np.float32)
            })


        self.last_messages = np.zeros((self.max_agents,3),dtype=np.float32)
            
        # Rewards - MODIFIED HERE
        self.r_clean = self.reward_params.get("r_clean", 1.2)
        self.r_done = self.reward_params.get("r_done", 15.0)
        self.r_step = self.reward_params.get("r_step", -0.02)
        self.r_collision = self.reward_params.get("r_collision", -1.5)


        self.agent_action_history = [deque(maxlen=4) for _ in range(self.max_agents)]
        self.agent_position_history = [deque(maxlen=8) for _ in range(self.max_agents)] 
        self.agent_active_options = [None] * self.max_agents
        self.reset()
        
    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)


    def _calculate_structural_complexity(self):

        walkable_grid = ~self.walls
        padded_grid = np.pad(walkable_grid, 1, mode='constant', constant_values=False)

        neighbor_sum = (
            padded_grid[1:-1, :-2] +  # Left neighbors
            padded_grid[1:-1, 2:] +   # Right neighbors
            padded_grid[:-2, 1:-1] +  # Top neighbors
            padded_grid[2:, 1:-1]     # Bottom neighbors
        )
         # If there are no walkable cells, return a high penalty to be safe.
        walkable_cells_connectivity = neighbor_sum[walkable_grid]

        if walkable_cells_connectivity.size == 0:
            return 4.0 # High penalty
        
        average_connectivity = np.mean(walkable_cells_connectivity)

        # The penalty is inversely proportional to the average connectivity.
        # 4.0 is the theoretical maximum connectivity in a 2D grid.
        # If avg_connectivity is 4 (perfectly open), penalty is 1.0.
        # If avg_connectivity is 2 (all corridors), penalty is 2.0.
        complexity_penalty = 4.0 / max(average_connectivity, 1.0) # Avoid division by zero

        return complexity_penalty


    def _calculate_dynamic_max_steps(self, n_agents, initial_dirt_sum):
        if n_agents == 0 or initial_dirt_sum == 0:
            return 1

        # --- Part 1: Spatial Dispersion Calculation ---
        dirt_locations = np.argwhere(self.dirt)
        agent_locations = np.array(self.agent_pos)

        if dirt_locations.size == 0 or agent_locations.size == 0:
            return int((self.H * self.W) / n_agents) # Fallback if no dirt/agents

        # Find the "center of mass" for the dirt
        # A) Calculate the "travel to work" time.
        # We find the average distance from an agent to the center of the dirt cluster.
        dirt_centroid = np.mean(dirt_locations, axis=0)
        agent_distances_to_centroid = np.abs(agent_locations - dirt_centroid).sum(axis=1)
        avg_travel_to_task_dist = np.mean(agent_distances_to_centroid)

        # B) The core work of cleaning is, at minimum, one step per dirt patch.
        # This is a much more direct and logical measure of "internal work".
        cleaning_work = initial_dirt_sum

        # The total optimistic work is the travel time plus the cleaning time.
        # This assumes one agent does everything.
        total_optimistic_work = avg_travel_to_task_dist + cleaning_work

        # --- Part 2: Structural Complexity Penalty ---
        # Get a multiplier that accounts for how maze-like the grid is.
        complexity_penalty = self._calculate_structural_complexity()

        # --- Part 3: Final Calculation ---
        # The realistic distance is the optimistic distance scaled by the complexity
        estimated_base_steps = (total_optimistic_work * complexity_penalty) 
        
        # Distribute the work among agents and apply the curriculum's overhead factor
        # The overhead_factor now purely represents agent inefficiency, not grid layout.
        agent_adjusted_steps = (estimated_base_steps/n_agents) * self.overhead_factor

        # Replace BASE_TRAVEL_TIME with a value derived from the grid size itself,
        # representing a buffer for exploration. The grid diagonal is a good proxy.
        exploration_buffer = (self.H + self.W)

        final_steps = agent_adjusted_steps + exploration_buffer
        
        return int(max(final_steps, initial_dirt_sum))


    def _update_dynamic_rewards(self,initial_dirt_sum):
        """Scales rewards based on the initial amount of dirt, relative to a baseline."""
        base_r_done = self.reward_params.get("r_done", 10.0)
        
        # Scale rewards based on how much larger the current task is compared to the baseline
        scaling_factor = initial_dirt_sum / self.BASELINE_DIRT_AREA
        scaling_factor =  np.sqrt(max(scaling_factor, 1.0))  # Use sqrt to moderate scaling
        

        # Update the scaled reward for the current episode
        self.r_done = base_r_done * scaling_factor


    
    def set_activte_agents(self, n):
        #I use min to avoid exceeding max_agents
        self.n_agents = min(n, self.max_agents)
        print(f"\n--- SETTING ACTIVE AGENTS TO {self.n_agents} ---")

    def _flood_fill_check(self, walls, start_pos):
        """Check if all non-wall cells are reachable from start_pos using flood fill"""
        visited = np.zeros_like(walls, dtype=bool)
        queue = deque([start_pos])
        visited[start_pos] = True
        reachable_count = 1
        
        directions = [(-1,0), (1,0), (0,-1), (0,1)]  # up, down, left, right
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.H and 0 <= nc < self.W and 
                    not walls[nr, nc] and not visited[nr, nc]):
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                    reachable_count += 1
        
        # Count total free cells
        total_free = np.sum(~walls)
        return reachable_count == total_free
    
    def update_wall_prob_range(self, new_range):
        """Allows the curriculum callback to change the wall density range."""
        self.wall_prob_range = new_range
        return True
    
    def _generate_connected_walls(self):
        """Generate interior walls but ensure the free space is one connected component.

        Previous logic retried random maps up to N times and then fell back to
        border-only walls, which could produce maps with no interior obstacles.

        This version:
        - Samples a random wall mask at density p in (wall_prob_range)
        - Keeps only the largest 4-connected free component
        - Marks all other cells as walls

        This guarantees connectivity while preserving interior walls.
        """
        H, W = self.H, self.W

        # Start with borders as walls
        walls = np.zeros((H, W), dtype=bool)
        walls[0, :] = True
        walls[-1, :] = True
        walls[:, 0] = True
        walls[:, -1] = True

        # Random internal walls with probability p
        p = float(self.rng.uniform(self.wall_prob_range[0], self.wall_prob_range[1]))
        if H > 2 and W > 2:
            rnd = self.rng.random((H - 2, W - 2)) < p
            walls[1:-1, 1:-1] = rnd

        # Ensure there is at least one free cell inside the border
        free_mask = ~walls
        if not np.any(free_mask[1:-1, 1:-1]):
            # If everything became a wall inside, relax probability and reopen
            walls[1:-1, 1:-1] = False
            free_mask = ~walls

        # Keep only the largest 4-connected component of free space
        # Define 4-connectivity structure
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=int)
        labeled, num = label(free_mask, structure=structure)
        if num <= 1:
            # Already a single component (or none due to degenerate sizes)
            return walls

        # Find largest component (ignore label 0 which is background)
        counts = np.bincount(labeled.ravel())
        if counts.size <= 1:
            return walls
        counts[0] = 0
        largest_label = int(np.argmax(counts))
        keep_free = (labeled == largest_label)

        # Convert all other cells to walls; keep borders as walls
        walls = ~keep_free
        walls[0, :] = True
        walls[-1, :] = True
        walls[:, 0] = True
        walls[:, -1] = True

        return walls
    
    def set_warmup_current_stage(self, stage):
        """Update the warmup stage"""
        self.warmup_stage = stage
        print(f"Environment: Warmup stage set to {stage}")
        return True

    def set_warmup(self, warmup_enabled):
        """Enable/disable warmup mode"""
        self.warmup = warmup_enabled
        print(f"Environment: Warmup {'enabled' if warmup_enabled else 'disabled'}")
        return True

    def set_current_size(self, size):
        self.current_size = size
        print(f"Environment: Current grid size set to {size}x{size}")
        return True

    def reset_state(self):
        # Randomly determine grid size within the specified range
        

        # Randomly determine number of agents within the specified range
        if not self.warmup:  
            self.n_agents = self.rng.integers(self.agent_range[0], self.agent_range[1] + 1)
            self.current_size = self.rng.integers(self.grid_shape_range[0], self.grid_shape_range[1] + 1)
        else:
            stage = str(self.warmup_stage)
            if stage in self.warmup_stages:
                self.n_agents = self.rng.integers(self.warmup_stages[stage]["min_agent"],self.warmup_stages[stage]["max_agent"] + 1 )
                self.current_size = self.rng.integers(self.warmup_stages[stage]["min_size"], self.warmup_stages[stage]["max_size"] + 1)

        self.H, self.W = self.current_size, self.current_size    
        self.walls = self._generate_connected_walls()
        
        # Create dirt map (places we need to visit)
        self.dirt = np.zeros((self.H, self.W), dtype=bool)
        for i in range(1, self.H - 1):
            for j in range(1, self.W - 1):
                if not self.walls[i, j]:
                    self.dirt[i, j] = True
        
        self.initial_dirt_sum = np.sum(self.dirt)
        self._update_dynamic_rewards(self.initial_dirt_sum)
        
        # Place agents on free cells
        self.agent_pos = []
        free_cells = np.argwhere(~self.walls & self.dirt)

        self.initial_dirt_sum = np.sum(self.dirt) 
        
        if len(free_cells) < self.n_agents:
            # If not enough free cells, reduce number of agents or create more space
            print(f"Warning: Only {len(free_cells)} free cells for {self.n_agents} agents")
            self.n_agents = min(self.n_agents, len(free_cells))
        
        self.rng.shuffle(free_cells)
        for i in range(self.n_agents):
            self.agent_pos.append(free_cells[i])
        
        # Clean the places where agents start
        for r, c in self.agent_pos:
            self.dirt[r, c] = False
        #We divide my the number of agents becasue the counter self.t add one after all agents have make a move (eg 4 agetns -> 4 moves so 4 steps)
        self.max_steps = int(self._calculate_dynamic_max_steps(self.n_agents, self.initial_dirt_sum)/self.n_agents) + self.n_agents
        
        self.t = 0
        #Clear the last messages 
        self.last_messages = np.zeros((self.max_agents,3), dtype=np.float32)

        self.agent_active_options = [None] * self.max_agents

        
    def reset(self, *, seed=None, options=None): 
        if seed is not None:
           self.seed(seed)

        if options and 'n_agents' in options:
            self.set_activte_agents(options['n_agents'])

        self.reset_state()
        for i in range(self.max_agents):
            self.agent_action_history[i].clear()
            self.agent_position_history[i].clear()


        obs = self._get_obs(agent_id=0)
        info = {}
        return obs, info
        

    def _get_move_towards_target(self, agent_pos, target_pos):
        """Returns a single low-level move (1-4) towards a target."""
        delta = np.array(target_pos) - np.array(agent_pos)
        if abs(delta[0]) > abs(delta[1]): # Move vertically
            return 2 if delta[0] > 0 else 1
        else: # Move horizontally
            return 4 if delta[1] > 0 else 3
        
    def _execute_clean_nearest_cluster(self, agent_id):
        """Heuristic for CLEAN_NEAREST_CLUSTER option."""
        agent_pos = self.agent_pos[agent_id]
        dirt_locations = np.argwhere(self.dirt)
        if dirt_locations.size == 0:
            return 0 # No dirt left
        
        #Find dirt cluster 
        labeled_dirt, num_featrues = label(self.dirt)
        if num_featrues == 0:
            return 0
        
        cluster_centroids = []
        for i in range(1, num_featrues + 1):
            cluster_mask = (labeled_dirt == i)
            cluster_coords = np.argwhere(cluster_mask)
            cluster_centroids.append(np.mean(cluster_coords, axis=0))

        distances = [np.linalg.norm(np.array(agent_pos) - centroid) for centroid in cluster_centroids]
        closest_cluster_idx = np.argmin(distances)
        target_pos = cluster_centroids[closest_cluster_idx].astype(int)

        return self._get_move_towards_target(agent_pos, target_pos)
    

    def _execute_disperse(self, agent_id):
        """Heuristic for DISPERSE option."""
        my_pos = self.agent_pos[agent_id]
        other_agent_pos = [pos for i, pos in enumerate(self.agent_pos) if i != agent_id]

        mid_h, mid_w = self.H // 2, self.W // 2
        quadrant_centers = {
            "top_left": (mid_h / 2, mid_w / 2),
            "top_right": (mid_h / 2, mid_w + mid_w / 2),
            "bottom_left": (mid_h + mid_h / 2, mid_w / 2),
            "bottom_right": (mid_h + mid_h / 2, mid_w + mid_w / 2),
        }

        agent_counts = {name: 0 for name in quadrant_centers}
        for pos in other_agent_pos:
            r, c = pos
            if r < mid_h and c < mid_w:
                agent_counts["top_left"] += 1
            elif r < mid_h and c >= mid_w:
                agent_counts["top_right"] += 1
            elif r >= mid_h and c < mid_w:
                agent_counts["bottom_left"] += 1
            else:
                agent_counts["bottom_right"] += 1


        min_agents_quadrant = min(agent_counts, key=agent_counts.get)
        target_pos = quadrant_centers[min_agents_quadrant]
        return self._get_move_towards_target(my_pos, target_pos)

    def step(self, action):
        num_current_agents = len(self.agent_pos)
        actions_np = np.asarray(action, dtype=int)[:num_current_agents]

        raw_moves = actions_np[:,0]
        message_types = actions_np[:, 1]
        message_targets_x = actions_np[:, 2]
        message_targets_y = actions_np[:, 3]
        final_moves = []

        current_messages = np.zeros((self.max_agents, 3), dtype=np.float32)
        for i in range(num_current_agents):
            current_messages[i, 0] = message_types[i]
            current_messages[i, 1] = message_targets_x[i] / self.W
            current_messages[i, 2] = message_targets_y[i] / self.H
        self.last_messages = current_messages
        
        for i in range(num_current_agents):
            move_action = raw_moves[i]
            
            # If agent has an active option, it overrides the raw action
            if self.agent_active_options[i] is not None:
                option_id, steps_remaining = self.agent_active_options[i]
                
                # Execute one step of the option's heuristic
                if option_id == 5: # CLEAN_NEAREST_CLUSTER
                    final_move = self._execute_clean_nearest_cluster(i)
                elif option_id == 6: # DISPERSE
                    final_move = self._execute_disperse(i)
                else:
                    final_move = 0 # Fallback

                # Decrement counter and clear if done
                steps_remaining -= 1
                if steps_remaining <= 0:
                    self.agent_active_options[i] = None
                else:
                    self.agent_active_options[i] = (option_id, steps_remaining)

                final_moves.append(final_move)

            # If the agent chose a NEW option
            elif move_action in OPTIONS:
                option_id = move_action
                # Set the option to be active for a few steps (e.g., 5)
                # This duration is a hyperparameter you can tune
                self.agent_active_options[i] = (option_id, 5) 

                # Execute the FIRST step of the option immediately
                if option_id == 5: # CLEAN_NEAREST_CLUSTER
                    final_move = self._execute_clean_nearest_cluster(i)
                elif option_id == 6: # DISPERSE
                    final_move = self._execute_disperse(i)
                else:
                    final_move = 0 # Fallback
                
                final_moves.append(final_move)

            # Otherwise, it's a standard low-level move
            else:
                final_moves.append(move_action)
                # Ensure any lingering option is cleared if a low-level move is chosen
                self.agent_active_options[i] = None 
        ### --- END NEW/MODIFIED SECTION --- ###

        for i, move in enumerate(final_moves):
            self.agent_action_history[i].append(move)

        

        proposed = [pos.copy() for pos in self.agent_pos]
        
        # Apply movement from the processed final_moves
        for i, a in enumerate(final_moves):
            proposed[i] = proposed[i] + MOVE_MAP.get(a, [0,0])
        
        valid = []

        self.t += 1
        
        rewards = np.full(self.n_agents, self.r_step, dtype=float)

        inactivity_penalty_multiplier = 6 # Make staying still 6x worse than moving
        r_inactive = self.r_step * inactivity_penalty_multiplier

        for i, a in enumerate(final_moves):
            if a == 0:
                rewards[i] += r_inactive

        #We add a small penalty for leftover dirt
        dirt_penalty = np.sum(self.dirt) * 0.001
        rewards -= dirt_penalty
        
        # Check bounds and walls
        for i in range(num_current_agents):
            r, c = proposed[i]
            if self._is_bounds(r, c) and not self.walls[r, c]:
                valid.append(True)
            else:
                valid.append(False)
                proposed[i] = self.agent_pos[i].copy()
                rewards[i] += self.r_collision
        
        # Check for collisions between agents
        proposed_tuples = [tuple(x) for x in proposed]
        counts = {}
        for idx, p in enumerate(proposed_tuples):
            counts[p] = counts.get(p, 0) + 1
        
        for idx, p in enumerate(proposed_tuples):
            if counts[p] > 1:
                proposed[idx] = self.agent_pos[idx].copy()
                rewards[idx] += self.r_collision
                
        # Check for swaps
        for i in range(num_current_agents):
            for j in range(i + 1, num_current_agents):
                if (tuple(proposed[i]) == tuple(self.agent_pos[j]) and
                    tuple(proposed[j]) == tuple(self.agent_pos[i])):
                    proposed[i] = self.agent_pos[i].copy()
                    proposed[j] = self.agent_pos[j].copy()
                    rewards[i] += self.r_collision
                    rewards[j] += self.r_collision
        
        # Update positions
        self.agent_pos = proposed
        cleaned_now = 0
        
        # Clean dirt and give rewards
        for i in range(num_current_agents):
            r, c = self.agent_pos[i]
            if self.dirt[r, c]:
                self.dirt[r, c] = False
                rewards[i] += self.r_clean
                cleaned_now += 1
                self.agent_action_history[i].clear()

        stagnation_penalty = -1.4 
        for i in range(num_current_agents):
            # Check if the history is full
            if len(self.agent_position_history[i]) == self.agent_position_history[i].maxlen:
                # Calculate the unique positions visited in the recent history
                recent_positions = np.array(self.agent_position_history[i])
                unique_rows = np.unique(recent_positions, axis=0)
                # If the agent has been in 3 or fewer unique spots in the last 8 steps, it's stuck.
                if len(unique_rows) <= 3:
                    rewards[i] += stagnation_penalty
                
        
        
        # Check termination
        terminated = self._is_all_clean()
        truncated = self.t >= self.max_steps and not terminated
        done = terminated or truncated
        
        if terminated:
            rewards += self.r_done
            
        obs = self._get_obs(agent_id=0)
        
        # Shared reward for teamwork
        shared = 0.75 * cleaned_now/self.n_agents
        rewards += shared
        
        info = {}
        if terminated or truncated:
            info['episode_stats'] = {
                "initial_dirt": self.initial_dirt_sum,
                "final_dirt": np.sum(self.dirt),
                "env_steps": self.t
            }
        return obs, rewards.astype(np.float32), terminated, truncated, info 
        
    def _is_bounds(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W

    def _is_all_clean(self):
        return not self.dirt.any()
        
    def _get_obs(self, agent_id):
        # The observation space definition is now 4 channels: walls, dirt, ego, others
        # We need to change the number of channels in the Dict space definition
        # Let's adjust the __init__ for this first.
        # OLD n_channels = 2 + self.max_agents
        # NEW n_channels = 4
        
        walls_plane = self.walls.astype(np.uint8)
        dirt_plane = self.dirt.astype(np.uint8)
        ego_plane = np.zeros((self.H, self.W), dtype=np.uint8)
        other_agents_plane = np.zeros((self.H, self.W), dtype=np.uint8)

        for i in range(len(self.agent_pos)):
            r, c = self.agent_pos[i]
            if i == agent_id:
                ego_plane[r, c] = 1
            else:
                other_agents_plane[r, c] = 1
        
        ch = [walls_plane, dirt_plane, ego_plane, other_agents_plane]
        map_tensor = np.stack(ch, axis=0)

        padded_map_tensor = np.zeros((map_tensor.shape[0], self.max_H, self.max_W), dtype=np.uint8)
        padded_map_tensor[:, :self.H, :self.W] = map_tensor
        
        return {
            "map": padded_map_tensor,
            "steps": np.array([self.t / self.max_steps], dtype=np.float32),
            "messages": self.last_messages 
        }
            
    def render(self):
        grid = np.full((self.H, self.W), fill_value=' ', dtype='<U1')
        grid[self.walls] = '#'
        grid[self.dirt] = '.'
        for i, (r, c) in enumerate(self.agent_pos):
            grid[r, c] = str(i)
        lines = ["".join(row) for row in grid]
        return "\n".join(lines)
    
    def update_difficulty_params(self, max_steps, r_clean, r_step, r_collision, r_done):
        """
        Update all difficulty-related parameters atomically
        This ensures all parameters are updated together, preventing inconsistent states
        """
        self.max_steps = max_steps
        self.r_clean = r_clean
        self.r_step = r_step
        self.r_collision = r_collision
        self.r_done = r_done
        
        print(f"Environment updated: max_steps={max_steps}, rewards=({r_clean}, {r_step}, {r_collision}, {r_done})")
        
        return True
    
    def update_overhead_factor(self, new_factor):
        """Allows the curriculum callback to change the difficulty."""
        self.overhead_factor = new_factor
        return True

# Test the mixed wall generation
if __name__ == "__main__":
    env_test = MultipleVacuumEnv(max_grid_shape=(12,12), grid_shape_range=(7,12))
    for i in range(4):
        print(f"\n--- Sample {i+1} ---")
        obs, info = env_test.reset()
        print(f"Actual Grid Size: {env_test.H}x{env_test.W}")
        print(f"Observation 'map' Shape: {obs['map'].shape}")
        print(env_test.render())
