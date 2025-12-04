# **Multi-Agent Cooperative Cleaning: MARL vs. Classical Planning**

## **Overview**

This project implements and compares two distinct approaches for solving the multi-agent cooperative cleaning problem on grid worlds:

1. **Deep Reinforcement Learning (MARL):** Agents train using **PPO (Stable-Baselines3)** with a custom CNN feature extractor, structured observations, and a curriculum scheduler. They learn to cooperate purely through trial-and-error interaction with the environment.  
2. **Algorithmic Planner (Benchmark):** A deterministic, no-training baseline located in solve\_with\_planner/. This uses **Voronoi Partitioning** for task allocation and **DFS** for coverage, serving as a "perfect" baseline to measure the RL agent's efficiency against.

## **Benchmarks: RL vs. Classical Planner**

To evaluate the quality of the PPO agents, I implemented a classical planner inspired by the **LS-MCPP** (Large-Scale Multi-Robot Coverage Path Planning) framework. This allows for a direct comparison between "Learned Behavior" and "Algorithmic Optimality."

| Feature | MARL Agent (PPO) | Classical Planner (Voronoi \+ DFS) |
| :---- | :---- | :---- |
| **Method** | Neural Network (Policy Gradient) | Graph Search (BFS/DFS) |
| **Setup Time** | Hours of training (Curriculum) | Instant (0 seconds) |
| **Environment** | Adapts to dynamic/unknown maps | Requires static/known map |
| **Coordination** | Emergent (learned via reward) | Explicit (Reservation Table) |
| **Use Case** | Robustness & Generalization | Efficiency & Predictability |

The Planner Implementation:  
The planner solves the grid in three steps:

1. **Partitioning:** Uses Multi-Source BFS (Voronoi) to assign every tile to the nearest agent.  
2. **Coverage:** Generates a DFS path for each agent to cover their assigned region.  
3. **Deconfliction:** Uses a reservation table to prevent collisions and edge swaps, ensuring a valid multi-agent path.

## **Repository Layout**

* solve\_with\_planner/: **(NEW)** Contains the deterministic solver and replay tools. Includes its own documentation (PLANNER\_EXPLAINED.md) and scripts to generate optimal plans.  
* env\_multiple\_vacuum.py: Multi-agent environment with connected-wall generation, dynamic max-steps heuristic, shared reward shaping, MultiDiscrete actions, and multi-channel global observations including messages.  
* my\_nn.py: Custom residual CNN feature extractor with dilations, semi-global pooling, and message embeddings integrated for MultiInputPolicy features.  
* team\_train.py: Main training script using SubprocVecEnv \+ VecNormalize, curriculum/warmup callback, and policy kwargs for the custom extractor and AdamW optimizer.  
* MyCallback.py: Warmup stage progression, curriculum annealing of overhead and PPO hyperparameters, TensorBoard logging, and controlled training stop when the minimum overhead is reached.  
* compare.py: Head-to-head evaluator that loads two models and their matching VecNormalize stats, runs multiple seeded maps, and prints per-map winners and a final summary.  
* test.py: Terminal visualizer that loads a trained model and VecNormalize stats, fixes a test scenario (e.g., 12×12 grid, 4 agents), and prints the rendered grid each step.  
* config.json: Centralized hyperparameters for environment, PPO, curriculum/warmup, and TensorBoard logging path.  
* cuda\_optim.py: Device selection and CUDA/cuDNN settings (e.g., TF32, benchmark) for faster training on RTX-class GPUs.  
* logs/: TensorBoard summaries written during training as configured in ppo\_params.tensorboard\_log.  
* model/: Trained model checkpoints and VecNormalize stats accessed by compare.py and test.py for evaluation.

## **Requirements**

Install the Python packages imported across the scripts: PyTorch, Gymnasium, Stable-Baselines3, and NumPy (plus TensorBoard for monitoring). CUDA is optional but used when available by the training script’s optimization routine.

pip install torch gymnasium stable-baselines3 numpy tensorboard

## **Running the Classical Planner (No Training)**

You can generate and watch a deterministic solution immediately without training.

1\. Generate a Plan:  
Run the solver to create a vacuum\_plan.json file.  
python solve\_with\_planner/solve\_with\_planner.py \--H 32 \--W 32 \--agents 6 \--wall-prob 0.15

2\. Replay the Plan:  
Visualize the generated plan in the terminal.  
python solve\_with\_planner/replay\_plan.py \--plan vacuum\_plan.json \--delay 0.05

*Note: For a deep dive into the algorithmic logic, see [solve\_with\_planner/PLANNER\_EXPLAINED.md](https://www.google.com/search?q=solve_with_planner/PLANNER_EXPLAINED.md).*

## **Training the RL Agent**

Training is driven by team\_train.py, which builds a multi-process vectorized environment wrapped with VecNormalize, configures TensorBoard logging, and applies warmup stages followed by a curriculum that anneals difficulty and PPO hyperparameters. The script reads all environment and PPO parameters from config.json.

Run:
```python
python team\_train.py
```
**Key details:**

* Curriculum/warmup thresholds, overhead factor schedule, wall probabilities, and PPO schedules are taken from config.json.  
* The vectorized setup normalizes observations and rewards, clipping observations and excluding message tokens from normalization via norm\_obs\_keys.

## **Evaluation & Visualization**

### **TensorBoard**

TensorBoard logs are written to the directory specified by ppo\_params.tensorboard\_log in config.json.
```python
tensorboard \--logdir logs/ppo\_tensorboard
```
### **Evaluate RL in terminal**

Use test.py to render the policy’s behavior step-by-step in the terminal. The script loads a trained PPO model and the corresponding VecNormalize statistics.
```python
python test.py
```
*Note: Update MODEL\_PATH and STATS\_PATH in test.py if using different filenames.*

### **Compare models**

Use compare.py to run head-to-head evaluations across multiple randomly seeded maps.

python compare.py

## **How the RL Agent Works**

The environment exposes a Dict observation with multi-channel global maps (walls, dirt, ego, others), a normalized step fraction, and per-agent messages; actions are MultiDiscrete \[move, message\] mapped to movement and a discrete communication token. Reward shaping includes per-step penalties, collision penalties, a shared teamwork component, a stagnation penalty, and a success bonus, while dynamic max-steps scales with map structure, agent count, and an overhead inefficiency factor. The custom CNN uses residual blocks with dilation, a semi-global pooling pathway for strategic context, and message embeddings concatenated with map and step features to form the policy/value backbone.

## **Reproducibility notes**

Always load the corresponding VecNormalize stats file alongside a trained model during evaluation or comparison, disabling training mode on the normalizer to avoid skewed inputs. The training script centralizes hyperparameters and logging paths in config.json so a single file controls environment ranges, curriculum schedules, and PPO settings.

## **Acknowledgments**

* PPO training loop, vectorized environments, and normalization are powered by Stable-Baselines3 and Gymnasium.  
* Classical planning logic inspired by the LS-MCPP framework (Tang et al.).  
* GPU acceleration and deterministic trade-offs are set via cuda\_optim.py (TF32, cuDNN benchmark) to speed up training on RTX hardware.
