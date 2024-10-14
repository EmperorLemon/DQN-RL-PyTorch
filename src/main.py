from game_app_module.app import App

from game_environment.game import Game
from game_environment.env import GameEnv

from ai_module.agent import DQNAgent

from tensorboardX import SummaryWriter

from utils.globals import *
from utils.utils import *
from config import HYPERPARAMETERS

import torch
import time


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(
        join_path(join_path(LOG_DIR, "runs"), f"2048_DQN_{time.time()}")
    )

    env = GameEnv(size=4)
    agent = DQNAgent(
        env,
    )

    game = Game(env)

    app = App(800, 600, game)
    app.run()

    writer.close()

    return 0


if __name__ == "__main__":
    main()
