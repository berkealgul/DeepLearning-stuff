from RayCasting.ray import Ray
import math


class RaySource:
    maxRange = 100

    def __init__(self, _loc, FOV, angle, rayCount):
        self.loc = _loc  # loc = (x, y)
        self.rays = self.initialize_rays(FOV, rayCount)
        # başlangıç açısı normalde 0 ama istediğimiz bir açı varsa döndürecez
        self.rotate(angle)

    def initialize_rays(self, FOV, rayCount):
        rays = []

        startA = -FOV / 2
        stopA = -startA
        dA = FOV / rayCount

        while startA < stopA:
            r = math.radians(startA)
            dir = (math.cos(r), math.sin(r))
            rays.append(Ray(self.loc, dir))

            startA += dA

        return rays

    def scan(self, boundaries):
        for ray in self.rays:
            ray.update(boundaries)

    def render(self, screen):
        for ray in self.rays:
            ray.render(screen)

    def set_loc(self, targetLoc):
        self.loc = targetLoc
        for ray in self.rays:
            ray.loc = self.loc

    def rotate(self, angle):
        for ray in self.rays:
            cosA = ray.dir[0]
            sinA = ray.dir[1]
            cosB = math.cos(math.radians(angle))
            sinB = math.sin(math.radians(angle))

            # cos(A + B) = cosA * cosB - sinA * sinB
            newDirX = cosA * cosB - sinA * sinB
            # sin(A + B) = sinA * cosB + sinB * cosA
            newDirY = sinA * cosB + sinB * cosA

            ray.dir = newDirX, newDirY
