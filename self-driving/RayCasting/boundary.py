import pygame as py


class Boundary:
    def __init__(self, _loc, _loc2):
        self.loc = _loc
        self.loc2 = _loc2

    def render(self, screen):
        py.draw.line(screen, (255, 0, 255), self.loc, self.loc2, 3)
