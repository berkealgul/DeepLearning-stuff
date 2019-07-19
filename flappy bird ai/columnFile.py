import pygame
import random


class Column:
    blankSpaceSize = 100
    width = 50
    velocityX = 15

    def __init__(self, locX, screenH):
        # dikdörtgenlerini oluşturduk
        self.lowerLimitOfUpRect = random.randint(150, 350)
        self.upRect = pygame.Rect(locX, 0, self.width, self.lowerLimitOfUpRect)
        self.locYOfDownRect = self.lowerLimitOfUpRect + self.blankSpaceSize
        self.downRect = pygame.Rect(locX, self.locYOfDownRect, self.width, screenH - self.locYOfDownRect)

        self.color = (255, 0, 255)
        self.score = 0

    def is_hit_to_bird(self, bird):
        if self.upRect.colliderect(bird.rect) or self.downRect.colliderect(bird.rect):
            return True
        else:
            return False

    def move(self, deltaTime):
        self.upRect.x -= Column.velocityX * deltaTime
        self.downRect.x -= Column.velocityX * deltaTime

    def render(self, targetScreen):
        pygame.draw.rect(targetScreen, self.color, self.upRect)
        pygame.draw.rect(targetScreen, self.color, self.downRect)
