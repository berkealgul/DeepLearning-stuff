import math
import pygame as py


class Ray:

    maxDistance = 100

    def __init__(self, _loc, _dir):
        # tuple (x, y)
        self.loc = _loc
        self.dir = _dir
        self.intersection = None

    def update(self, boundaries):
        closestIntersection = None

        for boundary in boundaries:
            intersection = self.cast(boundary)

            if intersection is None:
                continue

            if closestIntersection is not None:
                if self.distance(closestIntersection) > self.distance(intersection):
                    closestIntersection = intersection
            else:
                closestIntersection = intersection

        self.intersection = closestIntersection

    def cast(self, boundary):
        x1 = boundary.loc[0]
        y1 = boundary.loc[1]
        x2 = boundary.loc2[0]
        y2 = boundary.loc2[1]

        x3 = self.loc[0]
        y3 = self.loc[1]
        x4 = self.loc[0] + self.dir[0]
        y4 = self.loc[1] + self.dir[1]

        den = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

        if den == 0:
            return None
        # den 0 ise doğrular parereldir keşişemezler

        t = (((x1 - x3) * (y3 - y4)) - ((y1 - y3) * (x3 - x4))) / den
        u = -(((x1 - x2) * (y1 - y3)) - ((y1 - y2) * (x1 - x3))) / den

        if not (0 < t < 1 and u > 0):
            return None

        # ışın ile engel kesişirse bu noktanın kordinatları hesaplanıp verilir
        Ix = x1 + t * (x2 - x1)
        Iy = y1 + t * (y2 - y1)

        return Ix, Iy

    def render(self, screen):
        if self.intersection is None:
            return
        py.draw.line(screen, (255, 255, 255), self.loc, self.intersection)

    # yardımcı fonksiyonlar
    def distance(self, point):
        return math.sqrt(pow(self.loc[0] - point[0], 2) + pow(self.loc[1] - point[1], 2))

