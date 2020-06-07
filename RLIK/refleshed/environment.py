import pygame as py
import copy
import sys
import math
import random
import numpy as np
from params import *


class Environment:
    def __init__(self):
        py.init()
        self.w = 640
        self.h = 480
        self.fps = 30
        self.dt = 1 / self.fps
        self.screen = py.display.set_mode((self.w, self.h))
        self.clock = py.time.Clock()
        self.arm = Arm((int(self.w/2), int(self.h/2)))
        self.starting_angles = copy.copy(self.arm.joint_angles)
        self.goal = py.Rect(0, 0, Dimensions.goal_w, Dimensions.goal_w)
        self.set_goal()

    def reset(self):
        self.set_goal()
        for i in range(len(self.arm.joint_angles)):
            self.arm.joint_angles[i] = random.randint(0,360)
        state = self.get_state()
        return state

    def set_goal(self):
        x = self.arm.pivot[0] + random.randint(-120, 120)
        y = self.arm.pivot[1] + random.randint(-120, 120)
        self.goal.center = (x, y)

    """
     Peforms similation step on environment.

     Arguments:
     - action: Decided action taken by Agent. The enviroment
     will be affected by the action

     Returns:
     - state: new state of agent after action is done (np matrix)
     - reward: the reward of the Agent's action (float)
     - done: indicates is similation done. if it is done the
     the value is 1, otherwise 0
    """
    def step(self, a):
        self.arm.update(a)

        reward, done = self.get_reward()
        state = self.get_state()

        for e in py.event.get():
            if e.type == py.QUIT:
                sys.exit()

        self.clock.tick(self.fps)

        return state, reward, done

    def get_observation(self):
        end_eff_p = self.arm.axis_pivots[-1]
        goal_p = self.goal.center
        angles = self.arm.joint_angles

        obs = list()
        for i in range(len(angles)):
            state.append(angles[i])

        px = self.arn.pivot[0]
        py = self.arm.pivot[1]

        state.append(end_eff_p[0] - px)
        state.append(end_eff_p[1] - py)
        state.append(goal_p[0] - px)
        state.append(goal_p[1] - py)

        state = np.array(state)

        return state

    def get_reward(self):
        a = 1 / 70
        k = 20
        b = 10 / (2 * math.pi)
        end_eff = self.arm.axis_pivots[-1]
        goal = self.goal.center
        done = False

        if self.arm.is_collusion_free():
            dx = goal[0] - end_eff[0]
            dy = goal[1] - end_eff[1]
            dist = math.sqrt(dx*dx+dy*dy)

            if dist < 10:
                done = True

            dA = 0
            for i in range(len(self.starting_angles)):
                an = math.radians(self.starting_angles[i] -self.arm.joint_angles[i])
                dA += (an * an)
            dA = math.sqrt(dA)

            r = (-a * dist) - (b * dA)

            if done is True:
                r += k
        else:
            r = -k

        return r/20, int(done)

    def render(self):
        self.screen.fill((0,0,0))
        self.arm.render(canvas, 0)
        py.draw.rect(canvas, Colors.Goal, self.goal)
        py.display.flip()


class Arm:
    def __init__(self, _pivot):
        self.pivot = _pivot                        # Tuple (x,y)
        self.axis_pivots = list()                  # Including end effector
        self.joint_angles = [30, 70, 100]          # in Degrees and Excluding end effector
        self.connection_lengths = [60, 60, 60]     # length of connection between axis
        self.joint_count = 3                       # Excluding end effector
        self.__init_axis()

    def __init_axis(self):
        for i in range(self.joint_count+1):
            #self.joint_angles.append(0)
            self.axis_pivots.append(self.pivot)
        self.__update_axis_pivots()

    # RENDER FUNCTIONS BELOW
    def render(self, canvas, offsetVec):
        # TODO: Render alırken offsetleri ayarla (Hesaplama ile render konumları farklı)
        self.__render_connections(canvas, offsetVec)
        self.__render_joints(canvas, offsetVec)
        self.__render_end_effector(canvas, offsetVec)

    def __render_joints(self, canvas, offsetVec):
        for i in range(self.joint_count):
            c = self.axis_pivots[i]
            py.draw.circle(canvas, Colors.Joint, c, Dimensions.joint_r)

    def __render_end_effector(self, canvas, offsetVec):
        c = self.axis_pivots[-1]
        w = Dimensions.end_effector_w
        rect = py.Rect(0, 0, w, w)
        rect.center = c
        py.draw.rect(canvas, Colors.End_Effector, rect)

    def __render_connections(self, canvas, offsetVec):
        for i in range(self.joint_count):
            p1 = self.axis_pivots[i]
            p2 = self.axis_pivots[i+1]
            py.draw.line(canvas, Colors.Connector, p1, p2, Dimensions.connector_w)

    # UPDATE FUNCTIONS BELOW
    def update(self, action_vector):
        self.__update_axis_angles(action_vector)
        self.__update_axis_pivots()

    def __update_axis_angles(self, action_vector):
        for i in range(self.joint_count):
            self.joint_angles[i] += action_vector[0][i]
            self.joint_angles[i] = self.joint_angles[i] % 360

    def __update_axis_pivots(self):
        for i in range(self.joint_count):
            c_len = self.connection_lengths[i]
            a = math.radians(self.joint_angles[i])

            x = int(self.axis_pivots[i][0] + (math.cos(a) * c_len))
            y = int(self.axis_pivots[i][1] + (math.sin(a) * c_len))

            self.axis_pivots[i + 1] = (x, y)

    # SPECIAL FUNCTIONS BELOW
    def is_collusion_free(self):
        for p_axis in self.axis_pivots:
            for axis in self.axis_pivots:
                if p_axis == axis:
                    continue
                dx = p_axis[0] - axis[0]
                dy = p_axis[1] - axis[1]
                dist = math.sqrt(dx*dx+dy*dy)
                if dist <= 20:
                    return False
        return True
