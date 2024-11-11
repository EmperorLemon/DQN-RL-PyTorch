from game_environment.env import GameEnv

from ai_module.agent import DQNAgent
from ai_module.train import Trainer
from ai_module.test import play_2048
from ai_module.utils import check_cuda

from tensorboardX import SummaryWriter

from utils.globals import *
from utils.utils import *
from config import HYPERPARAMETERS

BOARD_SIZE = 4
IS_TRAIN = True

def main() -> int:
    check_cuda()
    
    env = GameEnv(BOARD_SIZE)
    agent = DQNAgent(BOARD_SIZE, HYPERPARAMETERS)
    
    writer = SummaryWriter(get_log_path(LOGS_DIR, "2048_DQN"))
    
    trainer = Trainer(env=env, agent=agent, writer=writer, config=HYPERPARAMETERS)
    
    if IS_TRAIN:
        trainer.train()
        play_2048(env, agent, 10)
    else:
        play_2048(env, agent, 20)

    writer.close()

    return 0


if __name__ == "__main__":
    main()
