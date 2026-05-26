"""
Programming Assignment 3: Comparing Q-learning, SARSA, and REINFORCE on FrozenLake 8x8
"""
 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from copy import deepcopy

# ──────────────────────────────────────────────
# General Parameters
# ──────────────────────────────────────────────
GAMMA               = 0.99
MAX_STEPS           = 200
TOTAL_TRAIN_STEPS   = 500_000
EVAL_INTERVAL       = 10_000
N_EVAL_EPISODES     = 500
SEEDS               = [0, 1, 2, 3, 4]

# Q-learning / SARSA hyper-params
ALPHA           = 0.1
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY_STEPS = 300_000

# REINFORCE hyper-params
POLICY_LR  = 0.005
THETA_CLIP = 20.0

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def make_env(seed=None):
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    return env

def get_epsilon(step):
    return max(EPS_END, EPS_START - step / EPS_DECAY_STEPS * (EPS_START-EPS_END))