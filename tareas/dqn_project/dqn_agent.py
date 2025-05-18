# dqn_agent.py
import math
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from models import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        *,
        buffer_size: int = 10_000,
        batch_size: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-4,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: int = 1_000,
        target_update: int = 500,
    ):
        self.device = device
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.target_update = target_update

        # Redes
        self.policy_net = QNetwork(obs_dim, action_dim).to(device)
        self.target_net = QNetwork(obs_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizador y memoria
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

        # Contadores
        self.steps_done = 0

    # ---------- Selección de acción ----------
    def choose_action(self, state, evaluate: bool = False) -> int:
        eps_threshold = (
            self.eps_end
            + (self.eps_start - self.eps_end)
            * math.exp(-1.0 * self.steps_done / self.eps_decay)
        )
        self.steps_done += 1

        if evaluate or random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device).unsqueeze(0)
                return int(self.policy_net(state).argmax(dim=1).item())
        return random.randrange(self.action_dim)

    # ---------- Aprendizaje ----------
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(self.batch_size, self.device)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    # ---------- Sincronización de redes ----------
    def maybe_update_target(self):
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ---------- Almacenamiento ----------
    def remember(self, *transition: Tuple):
        self.memory.push(*transition)

    # ---------- Guardar / cargar ----------
    def save(self, path="dqn_cartpole.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="dqn_cartpole.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()
