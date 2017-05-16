import gym


env = gym.make('FrozenLake-v0')
print("Action space: {}".format(env.action_space))
print("Observation space: {}".format(env.observation_space))

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, rewards, done,info = env.step(action)

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode+1,t+1))
            break

