#!/usr/bin/env python3
import argparse, json, time, os
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
from env_multiple_vacuum import MultipleVacuumEnv

# --- helpers ---
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _ensure_hw(arr, H, W):
    a = np.array(arr)
    if a.shape == (H, W):
        return a
    # try to reshape flat -> (H, W)
    return a.reshape((H, W))

def build_env_from_config(H:int, W:int, n_agents:int):
    # Load config.json if available for reward/comm params
    reward_params = {"r_clean":1.2, "r_step":-0.02, "r_collision":-1.5, "r_done":15.0}
    communication_params = {"num_message_types":3}
    if Path("config.json").exists():
        try:
            cfg = json.loads(Path("config.json").read_text())
            reward_params = cfg.get("reward_params", reward_params)
            communication_params = cfg.get("communication_params", communication_params)
        except Exception as e:
            print("[warn] Could not read config.json, using defaults:", e)

    env = MultipleVacuumEnv(
        max_grid_shape=(H, W),
        grid_shape_range=(H, W),
        agent_range=(n_agents, n_agents),
        max_agents=n_agents,
        initial_wall_prob_range=(0.15, 0.15),  # irrelevant if we force walls
        obs_mode="global",
        reward_params=reward_params,
        overhead_factor=1.3,
        warmup_stages={"1":{"min_agent":n_agents,"max_agent":n_agents,"min_size":H,"max_size":H}},
        mode="testing",
        communication_params=communication_params
    )
    return env

def force_map_state(env: MultipleVacuumEnv, walls, dirt, starts: List[List[int]]):
    """Inject exact map + agents into env, recompute derived fields."""
    H, W = env.H, env.W
    env.walls = _ensure_hw(walls, H, W).astype(bool)
    env.dirt  = _ensure_hw(dirt,  H, W).astype(bool)
    env.agent_pos = [np.array([int(r), int(c)], dtype=int) for (r,c) in starts]
    # Clean starting cells
    for r, c in env.agent_pos:
        env.dirt[r, c] = False
    # Recompute scalars the env expects
    env.initial_dirt_sum = int(env.dirt.sum())
    # rewards may scale with dirt; update them
    if hasattr(env, "_update_dynamic_rewards"):
        env._update_dynamic_rewards(env.initial_dirt_sum)
    # recompute max_steps the same way the env does
    if hasattr(env, "_calculate_dynamic_max_steps"):
        env.max_steps = int(env._calculate_dynamic_max_steps(len(env.agent_pos), env.initial_dirt_sum) / len(env.agent_pos)) + len(env.agent_pos)
    env.t = 0
    # clear messages
    env.last_messages = np.zeros((env.max_agents, 3), dtype=np.float32)

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def replay(plan_path: str, delay: float, stop_on_done: bool):
    plan = load_json(plan_path)
    H = int(plan["H"]); W = int(plan["W"]); n_agents = int(plan["n_agents"])
    per_agent_actions: List[List[int]] = plan["per_agent_actions"]
    starts = plan.get("starts", [])
    makespan = int(plan.get("makespan", max(len(a) for a in per_agent_actions)))

    # Build env and reset (optionally with seed)
    env = build_env_from_config(H, W, n_agents)
    seed = plan.get("seed", None)
    obs, info = env.reset(seed=seed)

    # If walls/dirt are provided in the plan, force-map for perfect reproduction
    if "walls" in plan and "dirt" in plan and starts:
        try:
            force_map_state(env, plan["walls"], plan["dirt"], starts)
            print("[replay] Using walls/dirt from plan for exact reproduction.")
        except Exception as e:
            print("[replay][warn] Failed to force map from plan; continuing with random map:", e)
    else:
        print("[replay][note] Plan JSON has no 'walls'/'dirt'. Replaying on a fresh map; layout may differ.")

    # Ensure the environment will not truncate before the plan finishes
    try:
        env.max_steps = max(int(env.max_steps), int(makespan) + 1)
        print(f"[replay] Set env.max_steps={env.max_steps} to cover makespan={makespan}.")
    except Exception as e:
        print("[replay][warn] Could not adjust env.max_steps:", e)

    # Main loop
    t = 0
    done_mask = np.array([False]*n_agents)
    final_info = {}
    # Run for the full makespan unless --stop-on-done is set
    while t < makespan:
        # Build (n_agents, 4) MultiDiscrete action: [move, msg_type, target_x, target_y]
        joint = np.zeros((n_agents, 4), dtype=int)
        for i in range(n_agents):
            move = int(per_agent_actions[i][t]) if t < len(per_agent_actions[i]) else 0
            joint[i, 0] = move            # low-level move from plan
            joint[i, 1] = 0               # message type: IDLE
            # use current position as message target placeholders (x=col, y=row)
            r, c = env.agent_pos[i]
            joint[i, 2] = int(c)
            joint[i, 3] = int(r)

        # step
        step_out = env.step(joint)
        obs, reward, terminated, truncated, info = step_out
        done_mask = terminated | truncated if isinstance(terminated, np.ndarray) else np.array([terminated or truncated]*n_agents)
        final_info = info

        # render
        clear_screen()
        print(f"Step {t+1}/{makespan}  |  reward(mean)={np.mean(reward):.3f}  |  done={bool(np.all(done_mask))}")
        print(env.render())
        time.sleep(delay)

        if stop_on_done and np.all(done_mask):
            break
        t += 1

    # final stats
    if isinstance(final_info, dict) and "episode_stats" in final_info:
        es = final_info["episode_stats"]
        print(f"\nEpisode stats: steps={es.get('env_steps')}  initial_dirt={es.get('initial_dirt')}  final_dirt={es.get('final_dirt')}")
    else:
        print("\nNo episode_stats found in info.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=str, default="vacuum_plan.json", help="Path to plan JSON produced by solve_with_planner.py")
    ap.add_argument("--delay", type=float, default=0.03, help="Seconds between frames")
    ap.add_argument("--stop-on-done", action="store_true", help="Stop replay when env reports done")
    args = ap.parse_args()
    replay(args.plan, args.delay, args.stop_on_done)
