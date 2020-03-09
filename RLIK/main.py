import pygame as py
import sys
from agent import Agent


w = 640
h = 480

screen = py.display.set_mode((w, h))


def main():
    py.init()
    agent = Agent((int(w/2), int(h/2)))
    a = 5
    while 1:
        screen.fill((0,0,0))
        agent.render(screen)
        agent.step()
        s = agent.get_state()
        py.display.flip()
        py.event.get()


if __name__ == "__main__":
    main()
