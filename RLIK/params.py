import math


# Object Colors during rendering
class Colors:
    Joint        = (227, 42, 0)
    Connector    = (44, 153, 5)
    Goal         = (29, 102, 219)
    End_Effector = (252, 232, 3)


# Object dimensions during rendering
class Dimensions:
    connector_w    = 8
    joint_r        = 10
    end_effector_w = 15
    goal_w         = 15


# Hyperparameters for simulation
class TrainConfig:
    steps_each_ep = 150
    max_episodes  = 50000
    lr            = 0.01
    gama          = 0.99
    inputs        = 7
    outputs       = 3
    hidden1       = 300
    hidden2       = 400
    a             = 1 / 70
    k             = 20
    b             = 10 / (2 * math.pi)
