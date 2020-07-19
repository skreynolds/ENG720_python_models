import argparse
from engine.engine import *
from utils.utils import read_config

parser = argparse.ArgumentParser(description='Run training')
parser.add_argument("--config", type=str, help="Path to the config file.")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    config = read_config(args['config'])
    engine = load_engine(config)
    print(engine.agent.actor_local)
    print(engine.agent.critic_local)
    engine.train()