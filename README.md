## Overview

This project trains and evaluates a multi-agent grid-world vacuum environment using PPO (Stable-Baselines3) with a custom CNN feature extractor, structured observations, and a curriculum + warmup scheduler for stable learning across increasing difficulty. Logs are written to a TensorBoard directory while training, and pretrained models with their VecNormalize statistics are used for evaluation and head-to-head comparisons across identical maps. CUDA optimizations are enabled when available to accelerate training on compatible GPUs.[^2][^8][^3][^4][^5][^6][^7][^1]

## Repository layout

- env_multiple_vacuum.py: Multi-agent environment with connected-wall generation, dynamic max-steps heuristic, shared reward shaping, MultiDiscrete actions , and multi-channel global observations including messages.[^3][^4]
- my_nn.py: Custom residual CNN feature extractor with dilations, semi-global pooling, and message embeddings integrated for MultiInputPolicy features.[^4]
- team_train.py: Main training script using SubprocVecEnv + VecNormalize, curriculum/warmup callback, and policy kwargs for the custom extractor and AdamW optimizer.[^6]
- MyCallback.py: Warmup stage progression, curriculum annealing of overhead and PPO hyperparameters, TensorBoard logging, and controlled training stop when the minimum overhead is reached.[^5]
- compare.py: Head-to-head evaluator that loads two models and their matching VecNormalize stats, runs multiple seeded maps, and prints per-map winners and a final summary.[^1]
- test.py: Terminal visualizer that loads a trained model and VecNormalize stats, fixes a test scenario (e.g., 12×12 grid, 4 agents), and prints the rendered grid each step.[^7]
- config.json: Centralized hyperparameters for environment, PPO, curriculum/warmup, and TensorBoard logging path.[^2]
- cuda_optim.py: Device selection and CUDA/cuDNN settings (e.g., TF32, benchmark) for faster training on RTX-class GPUs.[^8]
- logs/: TensorBoard summaries written during training as configured in ppo_params.tensorboard_log.[^2]
- model/: Trained model checkpoints and VecNormalize stats accessed by compare.py and test.py for evaluation.[^7][^1]


## Requirements

Install the Python packages imported across the scripts: PyTorch, Gymnasium, Stable-Baselines3, and NumPy (plus TensorBoard for monitoring). CUDA is optional but used when available by the training script’s optimization routine.[^8][^6][^1][^7]

Example:

```bash
pip install torch gymnasium stable-baselines3 numpy tensorboard
```


## Training

Training is driven by team_train.py, which builds a multi-process vectorized environment wrapped with VecNormalize, configures TensorBoard logging, and applies warmup stages followed by a curriculum that anneals difficulty and PPO hyperparameters. The script reads all environment and PPO parameters from config.json and enables CUDA optimizations when a GPU is available.[^5][^6][^8][^2]

Run:

```bash
python team_train.py
```

Key details:

- Curriculum/warmup thresholds, overhead factor schedule, wall probabilities, and PPO schedules are taken from config.json and applied by the callback during training.[^5][^2]
- The vectorized setup normalizes observations and rewards, clipping observations and excluding message tokens from normalization via norm_obs_keys.[^6]


## TensorBoard

TensorBoard logs are written to the directory specified by ppo_params.tensorboard_log in config.json (default: ./logs/ppo_tensorboard/). Launch TensorBoard pointing to the logs directory to monitor success rates, hyperparameters, and custom metrics recorded by the callback.[^2][^5]

Run:

```bash
tensorboard --logdir logs/ppo_tensorboard
```


## Evaluate in terminal

Use test.py to render the policy’s behavior step-by-step in the terminal on a fixed, challenging scenario (e.g., 12×12 map with 4 agents). The script loads a trained PPO model and the corresponding VecNormalize statistics, turns off normalization updates, and prints the grid each step with basic episode info.[^7]

Run:

```bash
python test.py
```

Notes:

- Update MODEL_PATH and STATS_PATH in test.py if using different filenames (e.g., model/model_t.zip and model/vec_t.pkl).[^7]
- The script uses DummyVecEnv + VecNormalize in evaluation mode (training=False, norm_reward=False) to ensure deterministic and correctly scaled observations.[^7]


## Compare models

Use compare.py to run head-to-head evaluations across multiple randomly seeded maps, tallying success, steps, and overall winners. The script expects each model to be paired with its matching VecNormalize stats and prints a per-map summary plus aggregate win/tie counts and success/efficiency statistics.[^1]

Run:

```bash
python compare.py
```

Tips:

- Set MODEL_1_PATH/MODEL_2_PATH and STATS_1_PATH/STATS_2_PATH to the desired artifacts under model/ before running.[^1]
- The comparison uses a consistent evaluation environment and seeded maps so both models face identical scenarios with the same wall-density range and grid size constraints.[^1]


## How it works

The environment exposes a Dict observation with multi-channel global maps (walls, dirt, ego, others), a normalized step fraction, and per-agent messages; actions are MultiDiscrete [move, message] mapped to movement and a discrete communication token. Reward shaping includes per-step penalties, collision penalties, a shared teamwork component, a stagnation penalty, and a success bonus, while dynamic max-steps scales with map structure, agent count, and an overhead inefficiency factor. The custom CNN uses residual blocks with dilation, a semi-global pooling pathway for strategic context, and message embeddings concatenated with map and step features to form the policy/value backbone.[^3][^4]

## Reproducibility notes

Always load the corresponding VecNormalize stats file alongside a trained model during evaluation or comparison, disabling training mode on the normalizer to avoid skewed inputs. The training script centralizes hyperparameters and logging paths in config.json so a single file controls environment ranges, curriculum schedules, and PPO settings.[^6][^2][^1][^7]

## Acknowledgments

- PPO training loop, vectorized environments, and normalization are powered by Stable-Baselines3 and Gymnasium as imported in the scripts.[^6]
- GPU acceleration and deterministic trade-offs are set via cuda_optim.py (TF32, cuDNN benchmark) to speed up training on RTX hardware when available.[^8]


