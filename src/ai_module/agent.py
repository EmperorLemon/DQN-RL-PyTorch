from tensorboardX import SummaryWriter

from game_environment.env import IEnv

from .model import DQN
from .memory import ReplayMemory, Transition

from typing import Any

from torch import optim


class DQNAgent:
    def __init__(
        self,
        device: str,
        writer: SummaryWriter | None,
        env: IEnv,
        hp: dict[str, Any],  # Hyperparameters
    ):
        self.device = device
        self.writer = writer

        self.env = env

        self.memory = ReplayMemory(hp.get("BUFFER_SIZE"))

        self.policy_net = DQN(env.state_size, env.action_size, [64, 64, 64]).to(device)
        self.target_net = DQN(env.state_size, env.action_size, [64, 64, 64]).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hp.get("LR"))

        self.steps_done = 0

    def optimize_model(self, batch_size: int):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        print(batch)
