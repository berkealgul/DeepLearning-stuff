import pygame
from NeuroEvolution.neuralNetwork import NeuralNetwork as neuralNet
import random


class Bird:
    maxSpeed = 35
    startLoc = None
    dimensions = None

    def __init__(self, brain):
        self.rect = pygame.Rect(Bird.startLoc, Bird.dimensions)  # loc ve dimension tuple türünden veriler
        self.velocityY = 15
        self.accelerationY = 10
        self.color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))

        # eğer kuş belli bir nöron ağı vermeden yaratmaya çalışıyorsak
        # yeni rastgele bir tane nöron ağı oluştururuz
        if brain is None:
            self.brain = neuralNet(5, 2, 3, 1)
        else:
            self.brain = brain

        self.score = 0
        self.fitness = 0

    def render(self, targetScreen):
        pygame.draw.rect(targetScreen, self.color, self.rect)

    def jump(self):
        self.velocityY = -18

    def move(self, deltaTime):
        # kuşun düşüş hızını limitliyoruz ki hızla aşağıya düşmesin
        if self.velocityY < Bird.maxSpeed:
            self.velocityY += self.accelerationY * deltaTime
        self.rect.y += self.velocityY * deltaTime

    def think(self, input):
        decision = self.brain.feedforward(input)
        if decision.values[0][0] > decision.values[1][0]:
            self.jump()
