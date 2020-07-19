import random

# import the logger
from utils.logger import Logger

# import the environment
from env.powersystem import TwoAreaPowerSystemEnv

# import the power system demand signal
from demand.powerdemand import StepSignal

# import ddpg
from agent.ddpg import DdpgAgent

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_engine(config):
    print("Loading engine...")
    return Engine(config)



class Engine():
    def __init__(self, config):

        # Store config parameters
        self.config = config
        self.n_episodes = config['n_episodes']
        self.log_every = config['log_every']
        self.episode_duration = config['episode_duration']
        self.first_area = config['first_area']
        self.first_area_step_size = config['first_area_step_size']
        self.second_area = config['second_area']
        self.second_area_step_size = config['second_area_step_size']
        self.noise_decay_value = config['noise_decay_value']
        self.random_seed = config['random_seed']

        # Create logging directory


        # create the logger
        self.logger = Logger()

        # create the environment
        self.env = TwoAreaPowerSystemEnv()
        self.env.seed(self.random_seed)

        # create the signal
        self.signal = StepSignal()

        # create the ddpg agent and add models to tensorboard
        self.agent = DdpgAgent(config)
        self.logger.graph_model(self.agent.actor_local, (torch.zeros(1,7)).to(device))
        

    def train(self):

        # create directory for the experiment
        experiment_dir = ''

        scores = []

        # train agent (i.e. run through all episodes)
        for i_episode in range(1, self.n_episodes+1):

            # initialise the environment, agent, and demand signal
            state = self.env.reset()
            self.agent.reset()
            self.signal.reset(self.episode_duration*random.random(),                   # trip time
                              self.first_area,                                         # frist area on/off
                              [-1,1][random.randrange(2)]*self.first_area_step_size,   # first area
                              self.second_area,                                        # second area on/off
                              [-1,1][random.randrange(2)]*self.second_area_step_size)  # second area

            # Re-initialise the score
            score = 0
            time_list = [0]

            # Run through a singla episode
            while True:

                # take control action
                action = self.agent.act(state)

                # determine demand
                demand = (self.signal.del_p_L_1_func(self.env.t),
                          self.signal.del_p_L_2_func(self.env.t))

                time_list.append(self.env.t)

                # step the environment forward in time
                next_state, reward, done, _ = self.env.step(action, demand)

                if len(time_list) % 10 == 0:
                    do_train = True
                else:
                    do_train = False

                # step the agent forward in time
                self.agent.step(state, action, reward, next_state, done, do_train)

                # save the future state as the current state
                state = next_state

                # save the reward from the current time step
                score += reward

                # if episode is completed finish
                if done:
                    break

            # decay the noise on the action signal
            self.agent.noise.noise_decay(self.noise_decay_value)

            # Record the score in tensorboard
            self.logger.scalar_summary('Reward/raw',score, i_episode)

        print('Training completed.')