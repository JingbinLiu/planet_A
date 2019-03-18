
import gym
import numpy as np

env = gym.make('Breakout-v0')

cnt = 0

s0 = env.reset()
while cnt<1000:
    env.render()
    env.step(0)
    env.step(0)
    env.step(0)
    obs, reward, done, info = env.step(env.action_space.sample())
    print(cnt,done,reward)
    #print(s)
    cnt += 1
    # if cnt % 100 == 0:
    #     env.reset()

# def discrete_action(a):
#     condition_list = [a>0.5, a>0.0, a>-0.5, True]
#     choice_list = [0 ,1, 2 ,3]
#     return np.select(condition_list, choice_list)
#
# for n in range(20):
#     print((n-10)/10,discrete_action((n-10)/10))