import pygame as py
import math
from params import Colors, Dimensions


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

    def __update_axis_pivots(self):
        for i in range(self.joint_count):
            c_len = self.connection_lengths[i]
            a = math.radians(self.joint_angles[i])

            x = int(self.axis_pivots[i][0] + (math.cos(a) * c_len))
            y = int(self.axis_pivots[i][1] + (math.sin(a) * c_len))

            self.axis_pivots[i + 1] = (x, y)

    # SPECIAL FUNCTIONS BELOW
    def is_collusion_free(self):
        for j in range(-2, 0):
            ef = self.axis_pivots[j]

            x0 = ef[0]
            y0 = ef[1]

            for i in range(3):
                p1 = self.axis_pivots[i]
                p2 = self.axis_pivots[i+1]

                x1 = p1[0]
                x2 = p2[0]
                y1 = p1[1]
                y2 = p2[1]

                if min((x1, x2)) < x0 < max((x1, x2)) and min((y1, y2)) < y0 < max((y1, y2)):
                    dx = x2-x1
                    dy = y2-y1
                    dist = abs(dy*x0-dx*y0 + x2*y1-y2*x1) / math.sqrt(dy*dy + dx*dx)

                    # TODO: Azami uzaklık parametre haline getirilebilir
                    if dist <= 20:
                        return False
        return True
