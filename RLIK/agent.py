import pygame as py
import random
import torch
import math
import copy
from arm import Arm
from params import *
from brain import Brain


class Agent:
    def __init__(self, _pivot):
        self.pivot = _pivot
        self.arm = Arm(_pivot)
        self.starting_angles = copy.copy(self.arm.joint_angles)
        self.brain = Brain()
        self.goal = py.Rect(0, 0, Dimensions.goal_w, Dimensions.goal_w)
        self.set_goal()

    def set_goal(self):
        x = self.pivot[0] + random.randint(-120, 120)
        y = self.pivot[1] + random.randint(-120, 120)
        self.goal.center = (x, y)

    def step(self, dt):
        s = self.get_state()
        a = self.brain.predict_action(s) * dt
        self.arm.update(a)
        r, done = self.get_reward()
        sn = self.get_state()
        self.brain.replayBuffer.add_sample(s, a ,r, sn)
        return done

    def render(self, canvas):
        self.arm.render(canvas, 0)
        py.draw.rect(canvas, Colors.Goal, self.goal)

    def get_state(self):
        end_eff_p = self.arm.axis_pivots[-1]
        goal_p = self.goal.center
        angles = self.arm.joint_angles

        state = list()
        for i in range(len(angles)):
            state.append(angles[i])

        px = self.pivot[0]
        py = self.pivot[1]

        state.append(end_eff_p[0] - px)
        state.append(end_eff_p[1] - py)
        state.append(goal_p[0] - px)
        state.append(goal_p[1] - py)

        state = torch.FloatTensor(state)

        return state

    def get_reward(self):
        a = TrainConfig.a
        b = TrainConfig.b
        k = TrainConfig.k
        end_eff = self.arm.axis_pivots[-1]
        goal = self.goal.center

        done = self.goal.collidepoint(end_eff)

        if self.arm.is_collusion_free():
            dx = goal[0] - end_eff[0]
            dy = goal[1] - end_eff[1]
            dist = math.sqrt(dx*dx+dy*dy)

            dA = 0
            for i in range(len(self.starting_angles)):
                a = math.radians(self.starting_angles[i] - self.arm.joint_angles[i])
                dA += (a * a)
            dA = math.sqrt(dA)

            r = (-a * dist) - (b * dA)

            if done is True:
                r += k
        else:
            r = -k

        return r, done

    def reset(self):
        self.set_goal()
        for i in range(len(self.arm.joint_angles)):
            self.arm.joint_angles[i] = random.randint(0,360)
