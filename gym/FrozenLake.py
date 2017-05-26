import gym
from gym import wrappers

import numpy as np

from chainerrl import q_functions
from chainerrl import explorers
from chainerrl.agents.dqn import DQN
from chainer import optimizers

import logging
logging.basicConfig(level=logging.DEBUG)

#Configuration variables
class Config:
    def __init__(self,episode):
        self.num_episode = episode
        self.hidden_layers = 2
        self.hidden_channels = 100
        self.start_exploration_epsilon = 1.0
        self.end_exploration_epsilon = 0.1
        self.decay_exploration_steps = 10**4
        self.replay_buffer = 5 * 10**5
        self.gamma = 0.99
        self.gpu = 0;
        self.start_epsilon = 0.99


#Setting up of Gym Environment
def setupEnv(envName):
    env = gym.make(envName)
    print(env)
    env = wrappers.Monitor(env,'/tmp/'+envName,force=True)
    print("Action space: {}".format(env.action_space))
    print("Observation space: {}".format(env.observation_space))
    print("Reward space: {}".format(env.reward_range))
    return env

def setupAgent(env,config):

    q_function = q_functions.FCStateQFunctionWithDiscreteAction(
        ndim_obs=env.observation_space.n,
        n_actions=env.env.action_space.n,
        n_hidden_channels=config.hidden_channels,
        n_hidden_layers=config.hidden_layers
    )

    optimizer = optimizers.Adam()
    optimizer.setup(q_function)

    replay_buffer = config.replay_buffer

    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=config.start_exploration_epsilon,
        end_epsilon=config.end_exploration_epsilon,
        decay_steps=config.decay_exploration_steps,
        random_action_func=env.action_space.sample())

    def phi(obs):
        return obs.astype(np.float32)

    #To setup this configuration
    agent = DQN(
        q_function,
        optimizer,
        replay_buffer,
        config.gamma,
        explorer,
        gpu=config.gpu,
        replay_start_size=50000,
        minibatch_size=32,
        update_frequency=1,
        target_update_frequency=10000,
        clip_delta=True,
        phi=phi,
        target_update_method=u'hard',
        soft_update_tau=0.01,
        n_times_update=1,
        average_q_decay=0.999,
        average_loss_decay=0.99,
        batch_accumulator=u'mean',
        episodic_update=4,
        episodic_update_len=16)


def main():
    env = setupEnv('FrozenLake-v0')
    config = Config(episode=20)
    setupAgent(env,config)

    for i_episode in range(config.num_episode):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, rewards, done,info = env.step(action)

            if done:
                print("Episode {} finished after {} timesteps".format(i_episode+1,t+1))
                break


if __name__ == "__main__":
    main()