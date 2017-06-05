import gym
from gym import wrappers

import numpy as np

from chainerrl import q_functions
from chainerrl import explorers
from chainerrl.agents.dqn import DQN
from chainerrl import experiments

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
        self.gpu = None
        self.start_epsilon = 0.99
        self.replay_start_size = 50
        self.mini_batch_size = 4
        self.update_frequency = 1
        self.target_update_frequency = 100
        self.episodic_update_len = 16
        self.episodic_update = False
        self.average_loss_decay = 0.99
        self.average_q_decay = 0.999
        self.soft_update_tau = 0.01
        self.clip_delta = True
        self.isTraining = True




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

    replay_buffer = config.replay_buffer #why?

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
        replay_start_size=config.replay_start_size,
        minibatch_size=config.mini_batch_size,
        update_frequency=config.update_frequency,
        target_update_frequency=config.target_update_frequency,
        clip_delta=config.clip_delta,
        phi=phi,
        target_update_method=u'hard',
        soft_update_tau=config.soft_update_tau,
        n_times_update=1,
        average_q_decay=config.average_q_decay,
        average_loss_decay=config.average_loss_decay,
        batch_accumulator=u'mean',
        episodic_update=config.episodic_update,
        episodic_update_len=config.episodic_update_len)

    return agent


def main():
    env = setupEnv('FrozenLake-v0')
    config = Config(episode=100)
    agent = setupAgent(env,config)

    steps = 10000
    eval_n_runs = 100
    eval_frequency = 1000
    max_episode_len = 100
    step_offset = 100
    outdir = '/tmp/FrozenLake-v0' #??

    eval_stats = experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        steps=steps,
        eval_n_runs=eval_n_runs,
        eval_frequency=eval_frequency,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        outdir=outdir)

    print('n_runs: {} mean: {} median: {} stdev {}'.format(
        eval_n_runs, eval_stats['mean'], eval_stats['median'],
        eval_stats['stdev']))


if __name__ == "__main__":
    main()