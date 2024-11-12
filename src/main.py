from game_module.app import App
from game_module.game import Game

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
IS_TRAIN = False # Change this to False to test model

def main() -> int:
    check_cuda()
    
    env = GameEnv(BOARD_SIZE)
    agent = DQNAgent(BOARD_SIZE, HYPERPARAMETERS)
    
    if IS_TRAIN:
        writer = SummaryWriter(get_log_path(LOGS_DIR, "2048_DQN"))
    
        trainer = Trainer(env=env, agent=agent, writer=writer, config=HYPERPARAMETERS)
        trainer.train()
        
        play_2048(env, agent, 10)
        writer.close()
    else:
        game = Game(env, agent)
        
        app = App(800, 600, game)
        app.run()
        
    # Uncomment this to add model diagram to tensorboard
    # dummy_input = torch.zeros(1, env.board_size * env.board_size).to(agent.device)
    # writer.add_graph(agent.policy_net, dummy_input)

    return 0


if __name__ == "__main__":
    main()
