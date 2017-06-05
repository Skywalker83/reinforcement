import gym
from gym_gomoku.envs import GomokuEnv
env = gym.make('Gomoku9x9-v0') # default 'beginner' level opponent policy


env = GomokuEnv('black','random',9)
env.reset()
env.render()
env.step(0) # place a single stone, black color first
env.render()
print("end")

# play a game
for a in range(10):
    env.reset()
    for _ in range(40):
        action = env.action_space.sample() # sample without replacement
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            print ("Game is Over")
            break
