import pygame
import random
import math


class Map:
    def __init__(self, width, height, spawnpoint=None):
        self.width = width
        self.height = height
        self.walls = []

        if spawnpoint is None:
            self.spawnpoint = random.randint(0, width), random.randint(0, height)
        else:
            self.spawnpoint = spawnpoint

    def convert_to_json(self):
        data = {}
        data['width'] = self.width
        data['height'] = self.height
        data['spawnpoint'] = self.spawnpoint
        data['walls'] = []
        for w in self.walls:
            loc = w.loc
            loc2 = w.loc2
            data['walls'].append([loc, loc2])
        return data

    def add_wall(self, loc, loc2):
        self.walls.append(Wall(loc, loc2))

    def remove_wall(self, wall):
        self.walls.remove(wall)

    def set_spawnpoint(self, loc):
        self.spawnpoint = loc

    # yardımcı fonksiyonlar
    def detect_closest_wall(self, pos):
        closestDis = None
        closestWall = None
        # mesafe sınırı olmalı ki pozisyona yakın duvarlar seçilmeli
        disTreshold = 10
        for w in self.walls:
            x0 = pos[0]
            y0 = pos[1]
            x1 = w.loc[0]
            y1 = w.loc[1]
            x2 = w.loc2[0]
            y2 = w.loc2[1]

            # 'pos' çizginin köşegen olduğu dikdörtgenin içinde olmalı
            if (x0-x1) * (x0-x2) > 0 or (y0-y1) * (y0-y2) > 0:
                continue

            dis = math.fabs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / math.sqrt(pow(y2-y1, 2)+pow(x2-x1, 2))

            # 'pos' duvarlara belli bir miktarda yakın olmali
            if dis > disTreshold:
                continue

            if closestDis is None:
                closestDis = dis
                closestWall = w
            elif dis < closestDis:
                closestDis = dis
                closestWall = w

        return closestWall

    # görüntüleme işlemleri
    def render(self, screen):
        self.render_walls(screen)
        self.render_spawnpoint(screen)

    def render_walls(self, screen):
        for w in self.walls:
            pygame.draw.line(screen, (255, 0, 255), w.loc, w.loc2, 3)

    def render_spawnpoint(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), self.spawnpoint, 6)


class Wall:
    def __init__(self, loc, loc2):
        self.loc = loc
        self.loc2 = loc2
