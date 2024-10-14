from game_app_module.app import App

from game_environment.game import Game
from game_environment.env import GameEnv

from ai_module.agent import DQNAgent
from ai_module.train import Trainer

from tensorboardX import SummaryWriter

from utils.globals import *
from utils.utils import *
from config import HYPERPARAMETERS

import torch


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(get_log_path(LOG_DIR, "2048_DQN"))

    agent = DQNAgent(device, writer, HYPERPARAMETERS)
    env = GameEnv(size=4)
    agent.create_agent_models(env.state_size, env.action_size)

    game = Game(env, ai_agent=agent)

    app = App(800, 600, game)
    app.run()

    writer.close()

    return 0


if __name__ == "__main__":
    main()
