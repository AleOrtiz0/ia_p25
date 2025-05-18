# replay_buffer.py
import random
from collections import deque, namedtuple

import torch

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """Experience Replay genérico."""

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Guarda una transición."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int, device):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.tensor(batch.state, dtype=torch.float32, device=device)
        action = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(
            1
        )
        next_state = torch.tensor(
            batch.next_state, dtype=torch.float32, device=device
        )
        done = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)
