import gym
game_name = "Gopher-v0"
env = gym.make(game_name)
env.reset()

action_space = env.action_space

states = set()

for _ in range(1000):
    env.render()
    s,r,done,info = env.step(action_space.sample())



