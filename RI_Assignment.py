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

def softmax(x):
    x = x-np.max(x)
    exp_x = np.exp(x)
    return

def evaluate(get_action_fn, eval_env, n_episodes=N_EVAL_EPISODES):
    """
    Run n_episodes episodes from the initial state using get_action_fn.
    Returns the mean discounted return (empirical V(s0)).
    """
    returns = []
    for ep in range(n_episodes):
        obs, _ =eval_env.reset()
        G = 0.0
        discount = 1.0
        for _ in range(MAX_STEPS):
            action = get_action_fn(obs)
            obs, reward, terminated, trundacted, _ = eval_env.step(action)
            G += discount*reward
            discount *= GAMMA
            if terminated or trundacted:
                break
        returns.append(G)
    return np.mean(returns)

# ──────────────────────────────────────────────
# Q-learning
# ──────────────────────────────────────────────
def run_q_learning(seed):
    env = make_env()
    eval_env = make_env()

    #apply seed
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    eval_env.reset(seed=seed + 100)
    eval_env.action_space.seed(seed + 100)
    np.random.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states,n_actions))

    steps_log = []
    values_log = []

    step = 0
    while step < TOTAL_TRAIN_STEPS:
        # ── evaluation checkpoint ──
        if step % EVAL_INTERVAL == 0:
            greedy = lambda s: int(np.argmax(q[s]))
            v = evaluate(greedy, eval_env)
            steps_log.append(step)
            values_log.append(v)
        
        eps = get_epsilon(step)
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            action = int(np,argmax(Q[obs]))

        next_obs, reward, terminated, truncated,_ = env.step(action)
        step+=1
        done = terminated or truncated

        target = reward if done else reward + GAMMA* np.max(Q[next_obs])
        Q[obs, action] += ALPHA * (target-Q[obs, action])

        obs = next_obs
        if done:
            obs, _ =env.reset
    greedy = lambda s: int(np.argmax(Q[s]))
    v= evaluate(greedy, eval_env)
    steps_log.append(step)
    values_log.append(v)

    env.close()
    eval_env.close()
    return np.array(steps_log),np.array(values_log)