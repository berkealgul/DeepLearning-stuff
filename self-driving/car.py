from RayCasting.raySource import RaySource
from NeuroEvolution.matrix import Matrix
from NeuroEvolution.neuralNetwork import NeuralNetwork
import math
import pygame as py
import random


class Car:
    FOV = 360  # eğer bunu değiştirir isen tekrardan lidarı döndürmen gerekir
    maxSpeed = 100
    dimensions = (20, 15)
    maxLidarRange = 100
    maxSteeringAngle = 40
    maxAccelerationPower = 30
    spawnpoint = 0, 0

    def __init__(self, angle=90, brain=None):
        self.loc = Car.spawnpoint
        self.dir = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        self.velocity = 0
        #self.color = random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)
        self.color = random.randint(0,255),random.randint(0,255),random.randint(0,255)
        lidarLoc = self.loc
        self.lidar = RaySource(lidarLoc, Car.FOV, angle, 8)

        self.geometry = None
        self.construct_and_alling_geometry(angle)

        # input = 5lidarIşını, hız, 2yönDeğeri
        # output = sol dönüş, sağ dönüş, gaz, fren, gaz/fren gücü
        if brain is None:
            self.brain = NeuralNetwork(9, 4, 5, 1)
        else:
            self.brain = brain

        # doğal seçilim yaparken aracın skoru('score') lazım olacak
        self.checkpointPassed = 0
        self.score = 0
        self.fitness = 0
        self.totalWent = 0

    # sürüş mekaniklerini sağlayan fonksiyonlar
    def move(self, acceleration, dt):
        locX = self.loc[0]
        locY = self.loc[1]

        self.update_velocity(acceleration, dt)

        dx = self.velocity * dt * self.dir[0]
        dy = self.velocity * dt * self.dir[1]

        locX += dx
        locY += dy

        self.totalWent += math.fabs(dx) + math.fabs(dy)

        self.loc = locX, locY
        self.lidar.set_loc(self.loc)

        self.move_geometry(dt)

    def rotate(self, angle):
        self.lidar.rotate(angle)

        cosA = self.dir[0]
        sinA = self.dir[1]
        cosB = math.cos(math.radians(angle))
        sinB = math.sin(math.radians(angle))

        # cos(A + B) = cosA * cosB - sinA * sinB
        newDirX = cosA * cosB - sinA * sinB
        # sin(A + B) = sinA * cosB + sinB * cosA
        newDirY = sinA * cosB + sinB * cosA
        self.dir = newDirX, newDirY

        self.rotate_geometry(angle)

    def update_velocity(self, acceleration, dt):
        if -Car.maxSpeed / 3 < self.velocity < Car.maxSpeed:
            self.velocity += acceleration * dt
            # aracın azami hızı aşması engellenir
            if self.velocity > Car.maxSpeed:
                self.velocity = Car.maxSpeed

    # diğer işlevsellikler
    def check_collusion(self, boundary):
        # eğer "boundary" çizgizi bizim geometrimizin herhangi bi kenarı ile kesişirse olay bitmiştir
        for i in range(len(self.geometry) - 1, -1, -1):
            x1 = boundary.loc[0]
            y1 = boundary.loc[1]
            x2 = boundary.loc2[0]
            y2 = boundary.loc2[1]

            # geometrinin kenarı berirlenir
            x3 = self.geometry[i][0]
            y3 = self.geometry[i][1]
            x4 = self.geometry[i - 1][0]
            y4 = self.geometry[i - 1][1]

            den = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

            if den == 0:
                continue
            # den 0 ise doğrular parereldir keşişemezler

            t = (((x1 - x3) * (y3 - y4)) - ((y1 - y3) * (x3 - x4))) / den
            u = -(((x1 - x2) * (y1 - y3)) - ((y1 - y2) * (x1 - x3))) / den

            if not (0 <= t <= 1 and 1 >= u >= 0):
                continue

            return True

        return False

    def render(self, screen, color=None):
        if color is not None:
            c = color
        else:
            c = self.color
        py.draw.polygon(screen, c, self.geometry)
        #self.lidar.render(screen)

    def update_score(self, totalCheckpoints):
        self.score += self.checkpointPassed
        # son 3 checkointe geldi ise 2 katı puan alır
        if self.checkpointPassed == totalCheckpoints - 2:
            self.score += self.checkpointPassed

    # otonom sürüş fonksiyonları
    def drive(self, boundaries, dt):
        output = self.think(boundaries)
        # output =
        # [ sağa kır                    ] --\ hangisi büyükse
        # [ sola kır                    ] --/ uygulanır
        # [ gaz        ]
        # [fren]

        des = output.values

        sa = Car.maxSteeringAngle
        # dönme uygulanır
        if des[0][0] < des[1][0]:
           sa *= -1  # sola kırılması gerekirse
        self.rotate(sa * dt)

        a = Car.maxAccelerationPower
        if des[2][0] < des[3][0]:
            a *= -1
        self.move(a, dt)

    def think(self, boundaries):
        input = Matrix(9, 1)
        lidarData = self.get_lidar_data(boundaries)
        # input =
        # [ lidar 1                      ]
        # [ lidar 2                      ]
        # [ lidar 3                      ]
        #   ...
        # [ lidar 8                      ]
        # [ self.velocity / max velocity ]
        for i in range(len(lidarData)):
            input.values[i][0] = lidarData[i]

        input.values[8][0] = self.velocity / Car.maxSpeed

        output = self.brain.feedforward(input)
        return output

    def get_lidar_data(self, boundaries):
        data = []
        self.lidar.scan(boundaries)
        R = Car.maxLidarRange
        for r in self.lidar.rays:
            try:
                x = r.distance(r.intersection)  # intersection None ise hata verir
            except:
                x = R
            if x > R:
                x = R
            # veriler 0-1 arasında olması tercih edilir
            data.append(x / R)
        return data


    # geometrisini güncelleyen fonksiyonlar
    def construct_and_alling_geometry(self, angle):
        self.geometry = self.construct_geometry()
        self.rotate_geometry(angle)

    def construct_geometry(self):
        # ağırlık merkezi
        cx = self.loc[0]
        cy = self.loc[1]
        w = Car.dimensions[0] / 2
        h = Car.dimensions[1] / 2

        geometry = []
        # köşeler ayarlanır
        geometry.append((cx + w, cy + h))  # sağ alt
        geometry.append((cx - w, cy + h))  # sol alt
        geometry.append((cx - w, cy - h))  # sol üst
        geometry.append((cx + w, cy - h))  # sağ üst

        return geometry

    def move_geometry(self, dt):
        for i in range(len(self.geometry)):
            vx = self.geometry[i][0]
            vy = self.geometry[i][1]

            vx += self.velocity * dt * self.dir[0]
            vy += self.velocity * dt * self.dir[1]

            self.geometry[i] = vx, vy

    def rotate_geometry(self, angle):
        #  rotation_matrix = Matrix(2, 2)
        a = math.radians(angle)
        rotation_matrix = Matrix(2, 2)
        # rotasyon matrisi hazırlanır
        rotation_matrix.values[0][0] = math.cos(a)
        rotation_matrix.values[0][1] = -math.sin(a)
        rotation_matrix.values[1][0] = math.sin(a)
        rotation_matrix.values[1][1] = math.cos(a)

        for i in range(len(self.geometry)):
            # ağırlık merkezi
            cx = self.loc[0]
            cy = self.loc[1]
            input = Matrix(2, 1)

            x = self.geometry[i][0] - cx
            y = self.geometry[i][1] - cy

            input.values[0][0] = x
            input.values[1][0] = y

            output = Matrix.matrix_product(rotation_matrix, input)

            newX = output.values[0][0] + cx
            newY = output.values[1][0] + cy

            self.geometry[i] = newX, newY
