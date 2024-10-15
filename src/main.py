# from game_app_module.app import App

# from game_environment.game import Game
from game_environment.env import GameEnv

from ai_module.agent import DQNAgent
from ai_module.train import Trainer
from ai_module.utils import check_cuda

from tensorboardX import SummaryWriter

from utils.globals import *
from utils.utils import *
from config import HYPERPARAMETERS

def main() -> int:
    check_cuda()
    
    writer = SummaryWriter(get_log_path(LOG_DIR, "2048_DQN"))

    env = GameEnv(size=4)
    agent = DQNAgent(env.state_size, env.action_size, HYPERPARAMETERS)
    
    trainer = Trainer(env=env, agent=agent, writer=writer, config=HYPERPARAMETERS)
    trainer.train()

    # game = Game(env)

    # app = App(800, 600, game)
    # app.run()

    writer.close()

    return 0


if __name__ == "__main__":
    main()
