# Multi‑Robot Grid Coverage Planner — How It Works

This document explains the **no‑training** planner we use to solve multi‑agent vacuum coverage on grids, and how it relates to the recent **LS‑MCPP / ESTC** approach with a **MAPF** (multi‑agent path finding) deconfliction pass.

> TL;DR: We partition the grid among robots, generate a coverage walk per partition, then **deconflict** the robots so they don’t collide. It’s fast and scales without PPO training.

---

## 1) Problem setup

- **Input:** a 4‑connected grid with **walls** and **free cells**, and the **start cell** of each robot.
- **Goal:** visit (cover) **all free cells** while minimizing **makespan** (time until the last robot finishes).
- **Assumption:** static, known map (good for cleaning/inspection). Dynamic/unknown maps need replanning or exploration.

---

## 2) High‑level pipeline

We use a standard three‑stage pipeline that mirrors the literature and the LS‑MCPP framework:

1. **Region Assignment (Partitioning).** Assign each free cell to *one* robot.
2. **Within‑Region Coverage.** Produce a walk that touches every cell in the region.
3. **Multi‑Robot Deconfliction.** Edit the per‑robot walks so they execute **safely and simultaneously** (no vertex/edge conflicts; optional turning cost).

Our lightweight implementation (in `solve_with_planner.py`) uses:

- **Voronoi assignment on the grid** (multi‑source BFS by robot starts).  
- **DFS‑style coverage** inside each region with shortest‑path hops to the nearest unvisited cell.  
- **Reservation‑table deconfliction** (prevents two robots from entering the same cell at the same time or swapping positions).

This is intentionally simple and dependency‑free; it demonstrates the core ideas and works well on large maps.

---

## 3) Stage A — Region assignment (Voronoi on the grid)

We compute **distance from all starts at once** (multi‑source BFS). Each free cell is owned by the **closest** start (ties break by smaller robot index). This creates a **Voronoi partition** that is fast to compute and roughly balances travel. In cluttered maps Voronoi regions can be **disconnected**; the coverage stage handles that by shortest‑path “hops.”

**Why Voronoi?** It is a classic way to split workspace among robots with minimal coordination overhead and good empirical balance. (Many exploration/coverage systems use Voronoi‑like partitions.)

**Implementation sketch:**
```python
owner = multisource_bfs_voronoi(H, W, walls, starts)  # HxW int array in [-1..K-1]
cells[k] = { (r,c) | owner[r,c] == k and not walls[r,c] }
```

---

## 4) Stage B — Within‑region coverage (DFS walk + shortest‑path hops)

We build a **walk** that touches every cell in the region, allowing revisits. If there is no unvisited neighbor, we **BFS** to the **nearest unvisited** cell and continue. The output is a list of **coordinates** `[(r0,c0), (r1,c1), ...]` per robot.

**Properties.** This simple method is fast, guarantees coverage, and tolerates disconnected partitions. It is not mathematically optimal, but it’s a strong baseline and a good initializer for local‑search refinements (see ESTC/LS‑MCPP, below).

**Implementation sketch:**
```python
while unvisited_in_region:
    if exists unvisited neighbor: step there (DFS)
    else: BFS to nearest unvisited, append the shortest path
```

---

## 5) Stage C — Multi‑robot deconfliction (reservation table)

Robots try to follow their individual paths **in lockstep**. At each step we gather **proposals** for the next cell of every robot and resolve conflicts:

- **Vertex conflict:** 2+ robots propose the same cell → lowest index wins; others **wait**.
- **Edge swap conflict:** robots proposing to swap cells → only one moves; the other **waits**.

We then append the committed positions as the next step. Repeat until all robots have reached the ends of their paths. The result is a set of **feasible schedules** and an overall **makespan**.

**Implementation sketch:**
```python
while not all_finished:
    proposals = [next(path_i) or wait] for each robot i
    resolve vertex + swap conflicts
    advance winners; others wait
```

---

## 6) Relation to ESTC / LS‑MCPP (Paper Algorithms)

Our minimal planner follows the **same structure** as the current state‑of‑the‑art, but replaces the heavy parts with simple primitives so it runs without extra repos:

- **ESTC (Extended STC).** Extends the classic **Spanning Tree Coverage (STC)** paradigm so it works **directly on the grid**, even when 2×2 blocks are partially obstructed. ESTC guarantees complete coverage with bounded suboptimality. In LS‑MCPP, ESTC is used both as a **standalone** method and as a component inside a **local‑search** framework that improves makespan by exploring smart neighborhood moves.
- **MAPF Deconfliction with Turning Costs.** After generating per‑robot coverage paths, LS‑MCPP applies a **MAPF** post‑processor to remove inter‑robot conflicts and to handle **turning costs** (turns as separate actions in the cost model). This unifies coverage planning with mature MAPF techniques and yields high‑quality, executable plans at scale.
- **Scaling.** Reported results: **up to 100 robots on 256×256 grids within minutes** of runtime, with substantial makespan improvements over prior methods.

> You can export maps/instances from this repo and run the **official LS‑MCPP** implementation to get these additional optimizations, while still using our adapter to replay the solution inside your environment.

---

## 7) How this differs from PPO training

- **No data, no training time.** Planning returns a good solution **immediately** for a known map; PPO needs many rollouts to converge and becomes harder to stabilize on large grids.
- **Instance‑optimality vs. policy generality.** Planning targets the **specific** map instance; PPO learns a **general** policy for a distribution (useful when maps are unknown or change online).
- **Hybrid is great.** Use the planner to generate **expert trajectories** for behavior cloning or reward shaping; deploy PPO for **reactivity** under uncertainty, with **replanning** when the map changes.

---

## 8) Practical notes for this repo

- `solve_with_planner.py` builds a random map, computes the plan, and writes:
  - `per_agent_actions` — the low‑level moves (0:wait, 1:up, 2:down, 3:left, 4:right)
  - `walls`, `dirt` — exact map snapshot for deterministic replay
  - `starts`, `H`, `W`, `n_agents`, `makespan`
- `replay_plan.py` replays on the **exact** saved layout and prints the **ANSI** render from the environment.
- Actions are sent as **MultiDiscrete** 4‑tuples per agent: `[move, message_type, target_x, target_y]`. We keep `message_type=0` and use the agent’s current cell as target placeholders.

---

## 9) Pseudocode (end‑to‑end)

```text
INPUT: grid walls/free, robot starts
OUTPUT: per‑agent schedules (conflict‑free), makespan

# A) Partition
owner = multi_source_BFS(starts, walls)        # Voronoi regions on grid
for each robot k: R_k = {cells owned by k and not walls}

# B) Per‑region coverage
for each robot k:
    path_k = DFS_coverage_with_BFS_hops(R_k, start_k)

# C) Deconfliction
schedules = reservation_table(paths=[path_k])
makespan  = max(len(act_seq_k) for act_seq_k in schedules)
```

---

## 10) Extending to LS‑MCPP (paper code)

1. Export to the repo’s format (`.map` + instance JSON).  
2. Run LS‑MCPP to get an **ESTC + local‑search** solution with **MAPF deconfliction** and **turning costs**.  
3. Replay inside this environment using our adapter.

---

## References & Further Reading

- Tang, Mao, Ma (2024–2025). *Large‑Scale Multi‑Robot Coverage Path Planning on Grids with Path Deconfliction (ESTC + LS‑MCPP with MAPF deconfliction and turning costs; 100 robots on 256×256 grids in minutes).* arXiv:2411.01707 / T‑RO version.
- Tang, Ma (AAAI‑24). *Large‑Scale Multi‑Robot Coverage Path Planning via Local Search* (introduces LS‑MCPP and ESTC).
- Gabriely & Rimon (STC). Classic spanning‑tree coverage paradigm (basis for ESTC).
- MAPF with Turn Actions. Optimal MAPF variant modeling turns as actions (for realistic motion costs).
- Surveys / background. Multi‑robot coverage and partitioning (Voronoi, decomposition, etc.).


## How to run it 
 - python solve_with_planner.py --H 32 --W 32 --agents 20 --wall-prob 0.20 --seed 1
 - replay_plan.py --plan vacuum_plan.json --delay 0.03 --stop-on-done