import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import sys

MOVE_MAP = {
    0:np.array([0,0], dtype=int),
    1:np.array([-1,0], dtype=int),
    2:np.array([1,0], dtype=int),
    3:np.array([0,-1], dtype=int),
    4:np.array([0,1], dtype=int)
}

class MultipleVacuumEnv(gym.Env):
    metadata = {"render_modes":["ansi"],"render_fps":4}
        
    def __init__(self, max_grid_shape=(10,10),grid_shape_range=(7,12),agent_range=(2,4),max_agents=4, initial_wall_prob_range=(0.1, 0.15),
                 obs_mode="global", seed=None, render_mode=None, reward_params=None , overhead_factor=1.6,warmup_stages=None, mode="training"): 
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
        
        if obs_mode == "global":
            n_channels = 4 
            self.observation_space = spaces.Dict({
                "map": spaces.Box(low=0, high=1, shape=(n_channels, self.max_H, self.max_W), dtype=np.uint8),
                "steps": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "messages": spaces.Box(low=0, high=4, shape=(self.max_agents,), dtype=np.int64)
            })


        self.last_messages = np.zeros(self.max_agents,dtype=np.int64)
            
        # Rewards - MODIFIED HERE
        self.r_clean = self.reward_params.get("r_clean", 1.2)
        self.r_done = self.reward_params.get("r_done", 15.0)
        self.r_step = self.reward_params.get("r_step", -0.02)
        self.r_collision = self.reward_params.get("r_collision", -1.5)


        self.agent_action_history = [deque(maxlen=4) for _ in range(self.max_agents)]
        self.agent_position_history = [deque(maxlen=8) for _ in range(self.max_agents)] 
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
        """Generate walls ensuring all free cells remain connected"""
        max_attempts = 50
        
        for attempt in range(max_attempts):
            walls = np.zeros((self.H, self.W), dtype=bool)
            
            # Add borders
            walls[0, :] = True
            walls[-1, :] = True
            walls[:, 0] = True
            walls[:, -1] = True
            

            prob_for_this_map = self.rng.uniform(self.wall_prob_range[0], self.wall_prob_range[1])

            # Add random internal walls
            for i in range(1, self.H - 1):
                for j in range(1, self.W - 1):
                    if self.rng.random() < prob_for_this_map:
                        walls[i, j] = True
            
            # Find a free cell to start connectivity check
            free_cells = np.argwhere(~walls)
            if len(free_cells) == 0:
                continue
                
            start_pos = tuple(free_cells[0])
            
            # Check if all free cells are connected
            if self._flood_fill_check(walls, start_pos):
                return walls
        
        # Fallback: return walls with just borders if we can't find a good layout
        walls = np.zeros((self.H, self.W), dtype=bool)
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
        self.last_messages = np.zeros(self.max_agents, dtype=np.int64)

        
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
        
    def step(self, action):
        num_current_agents = len(self.agent_pos)
        actions_np = np.asarray(action, dtype=int)[:num_current_agents]

        moves = actions_np[:,0]
        messages = actions_np[:,1]

        for i, move in enumerate(moves):
            self.agent_action_history[i].append(move)

        current_messages = np.zeros(self.max_agents, dtype=np.int64)
        current_messages[:num_current_agents] = messages
        self.last_messages = current_messages

        proposed = [pos.copy() for pos in self.agent_pos]  # Create proper copies
        
        # Apply movement
        for i, a in enumerate(moves):
            proposed[i] = proposed[i] + MOVE_MAP[a]
        
        valid = []

        self.t += 1
        
        rewards = np.full(self.n_agents, self.r_step, dtype=float)

        inactivity_penalty_multiplier = 6 # Make staying still 6x worse than moving
        r_inactive = self.r_step * inactivity_penalty_multiplier

        for i, a in enumerate(moves):
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

        stagnation_penalty = -0.25
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