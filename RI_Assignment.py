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
    return max(EPS_END,
               EPS_START - step / EPS_DECAY_STEPS * (EPS_START - EPS_END))
 
 
def softmax(x):
    x = x - np.max(x)          # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
 
 
def policy_probs(theta, state):
    return softmax(theta[state])
 
 
# ──────────────────────────────────────────────
# Evaluation (shared by all algorithms)
# ──────────────────────────────────────────────
def evaluate(get_action_fn, eval_env, n_episodes=N_EVAL_EPISODES):
    """
    Run n_episodes episodes from the initial state using get_action_fn.
    Returns the mean discounted return (empirical V(s0)).
    """
    returns = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        G = 0.0
        discount = 1.0
        for _ in range(MAX_STEPS):
            action = get_action_fn(obs)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            G += discount * reward
            discount *= GAMMA
            if terminated or truncated:
                break
        returns.append(G)
    return np.mean(returns)
 
 
# ──────────────────────────────────────────────
# Q-learning
# ──────────────────────────────────────────────
def run_q_learning(seed):
    env      = make_env()
    eval_env = make_env()
 
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    eval_env.reset(seed=seed + 100)
    eval_env.action_space.seed(seed + 100)
    np.random.seed(seed)
 
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
 
    steps_log  = []
    values_log = []
 
    step = 0
    while step < TOTAL_TRAIN_STEPS:
        # ── evaluation checkpoint ──
        if step % EVAL_INTERVAL == 0:
            greedy = lambda s: int(np.argmax(Q[s]))
            v = evaluate(greedy, eval_env)
            steps_log.append(step)
            values_log.append(v)
 
        eps = get_epsilon(step)
 
        # ε-greedy action
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[obs]))
 
        next_obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        done = terminated or truncated
 
        # Q-learning update
        target = reward if done else reward + GAMMA * np.max(Q[next_obs])
        Q[obs, action] += ALPHA * (target - Q[obs, action])
 
        obs = next_obs
        if done:
            obs, _ = env.reset()
 
    # final evaluation
    greedy = lambda s: int(np.argmax(Q[s]))
    v = evaluate(greedy, eval_env)
    steps_log.append(step)
    values_log.append(v)
 
    env.close()
    eval_env.close()
    return np.array(steps_log), np.array(values_log)
 
 
# ──────────────────────────────────────────────
# SARSA
# ──────────────────────────────────────────────
def run_sarsa(seed):
    env      = make_env()
    eval_env = make_env()
 
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    eval_env.reset(seed=seed + 100)
    eval_env.action_space.seed(seed + 100)
    np.random.seed(seed)
 
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
 
    steps_log  = []
    values_log = []
 
    # choose initial action
    eps = get_epsilon(0)
    if np.random.random() < eps:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(Q[obs]))
 
    step = 0
    while step < TOTAL_TRAIN_STEPS:
        # ── evaluation checkpoint ──
        if step % EVAL_INTERVAL == 0:
            greedy = lambda s: int(np.argmax(Q[s]))
            v = evaluate(greedy, eval_env)
            steps_log.append(step)
            values_log.append(v)
 
        next_obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        done = terminated or truncated
 
        # choose next action (on-policy)
        eps = get_epsilon(step)
        if np.random.random() < eps:
            next_action = env.action_space.sample()
        else:
            next_action = int(np.argmax(Q[next_obs]))
 
        # SARSA update
        target = reward if done else reward + GAMMA * Q[next_obs, next_action]
        Q[obs, action] += ALPHA * (target - Q[obs, action])
 
        obs    = next_obs
        action = next_action
        if done:
            obs, _ = env.reset()
            eps = get_epsilon(step)
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[obs]))
 
    # final evaluation
    greedy = lambda s: int(np.argmax(Q[s]))
    v = evaluate(greedy, eval_env)
    steps_log.append(step)
    values_log.append(v)
 
    env.close()
    eval_env.close()
    return np.array(steps_log), np.array(values_log)
 
 
# ──────────────────────────────────────────────
# REINFORCE (tabular stochastic policy)
# ──────────────────────────────────────────────
def run_reinforce(seed):
    env      = make_env()
    eval_env = make_env()
 
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    eval_env.reset(seed=seed + 100)
    eval_env.action_space.seed(seed + 100)
    np.random.seed(seed)
 
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    theta = np.zeros((n_states, n_actions))
 
    eval_grid  = list(range(0, TOTAL_TRAIN_STEPS + 1, EVAL_INTERVAL))
    steps_log  = []
    values_log = []
    next_eval_idx = 0

    step = 0
    while step < TOTAL_TRAIN_STEPS:
        # ── evaluation checkpoints ──
        while next_eval_idx < len(eval_grid) and eval_grid[next_eval_idx] <= step:
            greedy = lambda s: int(np.argmax(policy_probs(theta, s)))
            v = evaluate(greedy, eval_env)
            steps_log.append(eval_grid[next_eval_idx])
            values_log.append(v)
            next_eval_idx += 1
 
        # ── collect one episode ──
        states, actions, rewards = [], [], []
        obs, _ = env.reset()
        for _ in range(MAX_STEPS):
            probs  = policy_probs(theta, obs)
            action = np.random.choice(n_actions, p=probs)
 
            next_obs, reward, terminated, truncated, _ = env.step(action)
            step += 1
 
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
 
            obs  = next_obs
            done = terminated or truncated
            if done or step >= TOTAL_TRAIN_STEPS:
                break
 
        T = len(states)
 
        # ── compute discounted returns ──
        G_t = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + GAMMA * G
            G_t[t] = G
 
        # ── policy gradient update ──
        for t in range(T):
            s = states[t]
            a = actions[t]
            probs = policy_probs(theta, s)
 
            # gradient of log π(a|s) w.r.t. theta[s]
            grad = -probs.copy()
            grad[a] += 1.0          # one-hot minus probs
 
            # update with clipping
            theta[s] += POLICY_LR * G_t[t] * grad
            theta[s]  = np.clip(theta[s], -THETA_CLIP, THETA_CLIP)
 
    # final evaluation
    # flush any remaining eval points
    while next_eval_idx < len(eval_grid):
        greedy = lambda s: int(np.argmax(policy_probs(theta, s)))
        v = evaluate(greedy, eval_env)
        steps_log.append(eval_grid[next_eval_idx])
        values_log.append(v)
        next_eval_idx += 1
 
    env.close()
    eval_env.close()
    return np.array(steps_log), np.array(values_log)
 
 
# ──────────────────────────────────────────────
# Run all seeds
# ──────────────────────────────────────────────
def run_all(run_fn, name):
    all_steps  = []
    all_values = []
    for seed in SEEDS:
        print(f"  [{name}] seed={seed} ...", flush=True)
        s, v = run_fn(seed)
        all_steps.append(s)
        all_values.append(v)
    # align on common step axis (all seeds share same eval points)
    steps  = all_steps[0]
    values = np.array(all_values)   # shape (n_seeds, n_checkpoints)
    return steps, values
 
 

if __name__ == "__main__":
    print("Running Q-learning ...")
    ql_steps, ql_vals = run_all(run_q_learning, "Q-learning")
 
    print("Running SARSA ...")
    sarsa_steps, sarsa_vals = run_all(run_sarsa, "SARSA")
 
    print("Running REINFORCE ...")
    rf_steps, rf_vals = run_all(run_reinforce, "REINFORCE")

    # ──────────────────────────────────────────
    # Plot
    # ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
 
    def plot_curve(ax, steps, vals, label, color):
        mean = vals.mean(axis=0)
        std  = vals.std(axis=0)
        ax.plot(steps, mean, label=label, color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)
 
    plot_curve(ax, ql_steps,    ql_vals,    "Q-learning", "steelblue")
    plot_curve(ax, sarsa_steps, sarsa_vals, "SARSA",      "darkorange")
    plot_curve(ax, rf_steps,    rf_vals,    "REINFORCE",  "forestgreen")
 
    ax.set_xlabel("Number of environment steps", fontsize=13)
    ax.set_ylabel("Estimated value of initial state V(s₀)", fontsize=13)
    ax.set_title("FrozenLake 8×8 – Comparison of RL Algorithms\n"
                 f"(mean ± std over {len(SEEDS)} seeds)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = "Results/rl_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
 
    # also save numpy results for inspection
    np.savez("Results/rl_results.npz",
             ql_steps=ql_steps, ql_vals=ql_vals,
             sarsa_steps=sarsa_steps, sarsa_vals=sarsa_vals,
             rf_steps=rf_steps, rf_vals=rf_vals)
    print("Results saved.")

