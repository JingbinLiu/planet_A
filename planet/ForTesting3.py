
from planet.envs.carla.env import CarlaEnv

env = CarlaEnv()

cnt = 0

s0 = env.reset()
while True:
    #img = env.render()
    s, _, done, _ = env.step(env.action_space.sample()+(0.5,0.0))
    cnt += 1
    if cnt % 100 == 0:
        env.reset()