
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env_multiple_vacuum import MultipleVacuumEnv
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from MyCallback import CleaningStatsCallback



def setup_cuda_optimizations():
    if torch.cuda.is_available():
        # Enable optimizations for RTX Ada series
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster training
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy for 20GB VRAM
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of 20GB
        
        
        return "cuda"
    else:
        print("CUDA not available, using CPU")
        return "mps"

