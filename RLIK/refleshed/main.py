import pygame as py
import sys
from environment import *
from params import TrainConfig as config
from agent import *


load = False
save = True

env = Environment()
agent = Agent()

if load:
    agent.load_model()

for i in range(config.max_episods):
    s = env.reset()

    for j in range(config.steps_each_ep):
        a = agent.predict_action()
        s_, r, done = env.step(a)
        s = s_
        env.render()

        if done:
            print("epsiode terminated")
            break

    if save:
        agent.save_model()
