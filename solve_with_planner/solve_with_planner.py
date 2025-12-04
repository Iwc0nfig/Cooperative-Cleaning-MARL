#!/usr/bin/env python3
"""
solve_with_planner.py

Builds a 128x128 grid with MultipleVacuumEnv (random connected walls; if not a wall => dirt),
then solves it **without training** using a deterministic multi-robot coverage planner:

  1) Multi-source BFS Voronoi partition (assigns each free cell to nearest agent).
  2) DFS-style coverage per region with shortest-path "teleports" (real paths via BFS).
  3) Reservation-table deconfliction (prevents collisions and edge swaps).

Optionally exports the instance to the LS-MCPP repo format if you clone it, but runs standalone.
Requires: numpy

Usage:
  python solve_with_planner.py --H 128 --W 128 --agents 6 --wall-prob 0.15 --seed 42

To export for LS-MCPP after cloning https://github.com/reso1/LS-MCPP:
  python solve_with_planner.py --export-lsmcpp --lsmcpp-dir /path/to/LS-MCPP

References (planning, no learning):
  - Tang et al., "Large-Scale Multi-Robot Coverage Path Planning on Grids with Path Deconfliction" (T-RO). arXiv:2411.01707
  - Tang & Ma, "Large-Scale Multi-Robot Coverage Path Planning via Local Search" (AAAI 2024). arXiv:2312.10797
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import deque
import numpy as np

# Import lazily to avoid requiring gymnasium at import time of this script
def _import_env():
    from env_multiple_vacuum import MultipleVacuumEnv
    return MultipleVacuumEnv

Coord = Tuple[int, int]
FOUR_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
MOVE_TO_DELTA: Dict[int, Tuple[int,int]] = {
    0:(0,0), 1:(-1,0), 2:(1,0), 3:(0,-1), 4:(0,1)
}
DELTA_TO_MOVE: Dict[Tuple[int,int], int] = {v:k for k,v in MOVE_TO_DELTA.items()}

@dataclass
class PlanResult:
    per_agent_coords: List[List[Coord]]
    per_agent_actions: List[List[int]]
    makespan: int
    covered_fraction: float

def in_bounds(r:int,c:int,H:int,W:int)->bool:
    return 0<=r<H and 0<=c<W

def neighbors4(r:int,c:int,H:int,W:int,walls:np.ndarray)->List[Coord]:
    res=[]
    for dr,dc in FOUR_DIRS:
        rr,cc=r+dr,c+dc
        if in_bounds(rr,cc,H,W) and not walls[rr,cc]:
            res.append((rr,cc))
    return res

def multisource_bfs_voronoi(H:int,W:int,walls:np.ndarray,starts:List[Coord])->np.ndarray:
    owner=-np.ones((H,W),dtype=np.int32)
    dist=np.full((H,W),np.iinfo(np.int32).max,dtype=np.int32)
    
    q=deque()
    for i,(r,c) in enumerate(starts):
        if walls[r,c]: raise ValueError(f"Start {i} on wall at {(r,c)}")
        owner[r,c]=i; dist[r,c]=0; q.append((r,c))
    while q:
        r,c=q.popleft()
        for rr,cc in neighbors4(r,c,H,W,walls):
            nd=dist[r,c]+1
            if dist[rr,cc]>nd:
                dist[rr,cc]=nd; owner[rr,cc]=owner[r,c]; q.append((rr,cc))
            elif dist[rr,cc]==nd:
                owner[rr,cc]=min(owner[rr,cc], owner[r,c])
    owner[walls]=-1
    return owner

def bfs_path(start:Coord, goal:Coord, H:int, W:int, walls:np.ndarray)->List[Coord]:
    if start==goal: 
        return [start]
    
    prev={}
    q=deque([start]); seen={start}
    while q:
        u=q.popleft()
        for v in neighbors4(u[0],u[1],H,W,walls):
            if v in seen: continue
            seen.add(v); prev[v]=u
            if v==goal:
                path=[v]
                while path[-1]!=start:
                    path.append(prev[path[-1]])
                path.reverse(); return path
            q.append(v)
    return []

def dfs_coverage_sequence(region:set[Coord], start:Coord, H:int,W:int,walls:np.ndarray)->List[Coord]:
    visited=set([start]); path=[start]; cur=start
    region=set(region)
    def nearest_unvisited(from_cell:Coord):
        
        q=deque([from_cell]); prev={}; seen={from_cell}
        while q:
            u=q.popleft()
            if u in region and u not in visited:
                walk=[u]
                while walk[-1]!=from_cell:
                    walk.append(prev[walk[-1]])
                walk.reverse(); return walk,u
            for v in neighbors4(u[0],u[1],H,W,walls):
                if v not in seen:
                    seen.add(v); prev[v]=u; q.append(v)
        return None
    while len(visited & region)<len(region):
        stepped=False
        for v in neighbors4(cur[0],cur[1],H,W,walls):
            if v in region and v not in visited:
                path.append(v); visited.add(v); cur=v; stepped=True; break
        if stepped: continue
        hop=nearest_unvisited(cur)
        if hop is None: break
        walk,nxt=hop
        if walk and walk[0]==cur: walk=walk[1:]
        path.extend(walk); cur=nxt; visited.add(cur)
    return path

def coords_to_actions(coords:List[Coord])->List[int]:
    actions=[]
    for (r1,c1),(r2,c2) in zip(coords, coords[1:]):
        dr,dc=(r2-r1,c2-c1)
        actions.append(DELTA_TO_MOVE.get((dr,dc),0))
    return actions

def deconflict_schedules(per_agent_paths:List[List[Coord]])->List[List[Coord]]:
    K=len(per_agent_paths)
    idx=[0]*K
    cur=[p[0] for p in per_agent_paths]
    done=[False]*K
    schedules=[[cur[i]] for i in range(K)]
    t=0
    while True:
        t+=1
        proposals=[]
        for i in range(K):
            if done[i]: proposals.append(cur[i]); continue
            nxt=cur[i] if idx[i]+1>=len(per_agent_paths[i]) else per_agent_paths[i][idx[i]+1]
            proposals.append(nxt)
        cell2ags={}
        for i,p in enumerate(proposals):
            cell2ags.setdefault(p,[]).append(i)
        winners=set(min(ags) for ags in cell2ags.values())
        # prevent swaps
        for i in range(K):
            for j in range(i+1,K):
                if proposals[i]==cur[j] and proposals[j]==cur[i]:
                    if j in winners: winners.remove(j)
        moved=False
        new_cur=list(cur)
        for i in range(K):
            if done[i]: continue
            if i in winners and proposals[i]!=cur[i]:
                new_cur[i]=proposals[i]; idx[i]+=1; moved=True
        cur=new_cur
        for i in range(K):
            schedules[i].append(cur[i])
            if idx[i]+1>=len(per_agent_paths[i]): done[i]=True
        if all(done): break
        if t>sum(len(p) for p in per_agent_paths)*3: break
    return schedules

def plan_no_learning(walls:np.ndarray, dirt:np.ndarray, starts:List[Coord])->PlanResult:
    H,W=walls.shape
    free_mask=(~walls)&dirt
    owner=multisource_bfs_voronoi(H,W,walls,starts)
    cells=[set() for _ in starts]
    for r in range(H):
        for c in range(W):
            if free_mask[r,c]:
                k=int(owner[r,c])
                if k>=0: cells[k].add((r,c))
    per_paths=[]
    for k,start in enumerate(starts):
        if not cells[k]:
            per_paths.append([start]); continue
        path_k=dfs_coverage_sequence(cells[k], start, H,W,walls)
        if not path_k or path_k[0]!=start: path_k=[start]+path_k
        per_paths.append(path_k)
    schedules=deconflict_schedules(per_paths)
    per_actions=[coords_to_actions(coords) for coords in schedules]
    covered=np.zeros_like(free_mask, dtype=bool)
    for coords in schedules:
        for (r,c) in coords:
            if free_mask[r,c]: covered[r,c]=True
    covered_fraction=float(covered.sum())/float(free_mask.sum() if free_mask.sum()>0 else 1)
    makespan=max((len(a) for a in per_actions), default=0)
    return PlanResult(schedules, per_actions, makespan, covered_fraction)

def export_to_lsmcpp(out_dir:str, walls:np.ndarray, starts:List[Coord], name:str)->Tuple[Path,Path]:
    out=Path(out_dir)
    grid_dir=out/"benchmark"/"gridmaps"; grid_dir.mkdir(parents=True, exist_ok=True)
    inst_dir=out/"benchmark"/"instances"; inst_dir.mkdir(parents=True, exist_ok=True)
    H,W=walls.shape
    ascii_lines=["".join("@" if walls[r,c] else "." for c in range(W)) for r in range(H)]
    map_text="\n".join(["type octile", f"height {H}", f"width {W}", "map", *ascii_lines])
    map_path=grid_dir/f"{name}.map"; map_path.write_text(map_text)
    instance={
        "gridmap": f"{name}.map",
        "roots": [[int(r),int(c)] for (r,c) in starts],
        "weights":[1]*len(starts),
        "note":"Keys may need alignment with LS-MCPP/benchmark/instance.py"
    }
    istc_path=inst_dir/f"{name}.json"; istc_path.write_text(json.dumps(instance, indent=2))
    return map_path, istc_path

def run(H:int,W:int,n_agents:int,wall_prob:float,seed:int, export_lsmcpp:bool, lsmcpp_dir:str|None):
    # Build env and map
    MultipleVacuumEnv=_import_env()

    # Load config.json if present to pick up reward/communication params; else defaults
    reward_params = {"r_clean":1.2, "r_step":-0.02, "r_collision":-1.5, "r_done":15.0}
    communication_params = {"num_message_types":3}
    cfg_path = Path("config.json")
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            if "reward_params" in cfg: reward_params = cfg["reward_params"]
            if "communication_params" in cfg: communication_params = cfg["communication_params"]
        except Exception as e:
            print("[warn] Failed to read config.json; using built-in defaults:", e)

    env=MultipleVacuumEnv(
        max_grid_shape=(H,W),
        grid_shape_range=(H,W),
        agent_range=(n_agents,n_agents),
        max_agents=n_agents,
        initial_wall_prob_range=(wall_prob,wall_prob),
        obs_mode="global",
        seed=seed,
        reward_params=reward_params,
        mode="eval",
        communication_params=communication_params,
    )
    env.reset()
    walls=env.walls.astype(bool)
    dirt=env.dirt.astype(bool)
    starts=[tuple(map(int, rc)) for rc in env.agent_pos]

    

    result=plan_no_learning(walls, dirt, starts)
    print(f"[Planner] makespan={result.makespan} steps | covered≈{result.covered_fraction*100:.1f}% | agents={n_agents} | map={H}x{W}")

    # Save actions
    out_json = {
        "H": H,
        "W": W,
        "n_agents": n_agents,
        "starts": [list(x) for x in starts],  # ensure lists, not NumPy arrays
        "seed": seed,
        "makespan": result.makespan,
        "covered_fraction": result.covered_fraction,
        "per_agent_actions": result.per_agent_actions,  # already lists of ints
        "walls": env.walls.astype(int).tolist(),
        "dirt":  env.dirt.astype(int).tolist(),
    }
    out_path=Path("vacuum_plan.json")
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"[Output] wrote per-agent actions: {out_path.resolve()}")

    if export_lsmcpp:
        assert lsmcpp_dir is not None, "--lsmcpp-dir is required when --export-lsmcpp is set"
        name=f"vacuum_{H}x{W}_rnd"
        map_path, istc_path = export_to_lsmcpp(lsmcpp_dir, walls, starts, name=name)
        print(f"[LS-MCPP] wrote map: {map_path}")
        print(f"[LS-MCPP] wrote instance: {istc_path}")
        print("Run LS-MCPP CLI (from its repo):  python main.py", name)

    # (Optional) You can now step env with these actions if you want to verify execution.

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--wall-prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export-lsmcpp", action="store_true", help="Also export map+instance for LS-MCPP repo")
    ap.add_argument("--lsmcpp-dir", type=str, default=None, help="Path to cloned LS-MCPP repo")
    args=ap.parse_args()

    run(
        H=args.H,
        W=args.W,
        n_agents=args.agents,
        wall_prob=args.wall_prob,
        seed=args.seed,
        export_lsmcpp=args.export_lsmcpp,
        lsmcpp_dir=args.lsmcpp_dir,
    )
