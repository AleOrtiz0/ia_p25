# train_dqn.py
import itertools
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn_agent import DQNAgent

ENV_NAME = "CartPole-v1"
NUM_EPISODES = 600
MAX_STEPS = 10_000  # safety cap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_rewards(rewards, window=20):
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label="Reward por episodio")
    plt.plot(range(window - 1, len(rewards)), ma, label=f"Media móvil ({window})")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rewards.png")
    plt.close()


def main():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim, action_dim, device)

    rewards_history = []
    scores_window = deque(maxlen=100)

    for i_ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0

        for t in itertools.count():
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(
                state,
                action,
                reward,
                next_state,
                float(done),
            )
            agent.learn()
            agent.maybe_update_target()

            state = next_state
            ep_reward += reward

            if done or t >= MAX_STEPS:
                break

        rewards_history.append(ep_reward)
        scores_window.append(ep_reward)

        if i_ep % 10 == 0:
            avg_reward = np.mean(scores_window)
            print(
                f"[{time.strftime('%H:%M:%S')}] Episodio {i_ep:4d} | "
                f"Recompensa ∑={ep_reward:.0f} | Promedio(últ.100)={avg_reward:.1f}"
            )

        # parar si cumple criterio de éxito oficial
        if np.mean(scores_window) >= 475.0:
            print(f"✅ Objetivo alcanzado en el episodio {i_ep}")
            agent.save()
            break

    env.close()
    agent.save()
    plot_rewards(rewards_history)


if __name__ == "__main__":
    main()
