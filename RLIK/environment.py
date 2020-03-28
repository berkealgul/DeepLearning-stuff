import pygame as py
from agent import Agent
import sys
from params import TrainConfig as tc


class Environment:
    def __init__(self):
        self.w = 640
        self.h = 480
        self.dt = 1 / 30
        self.screen = py.display.set_mode((self.w, self.h))
        self.clock = py.time.Clock()
        self.agent = Agent((int(self.w/2), int(self.h/2)))

    def begin(self):
        self.agent.brain.load()
        for i in range(tc.max_episodes):
            self.update_log(str(i+1))
            for j in range(tc.steps_each_ep):
                self.clock.tick(30)

                for e in py.event.get():
                    if e.type == py.QUIT:
                        sys.exit()

                self.render()
                done = self.agent.step(self.dt)
                if done is True:
                    print("--------------DONE------------------")
                    break
            self.agent.brain.train()
            self.agent.reset()
            self.agent.brain.save()

    def update_log(self, ep):
        print("----------------------------------------")
        print("")
        print("        EPISODE: " + ep + "           ")
        print("        BERKE ALGÜL                    ")
        print("")
        print("----------------------------------------")

    def render(self):
        self.screen.fill((0,0,0))
        self.agent.render(self.screen)
        py.display.flip()
