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
    load          = True
    steps_each_ep = 150
    max_episodes  = 10000
    lr            = 0.01
    gamma         = 0.99
    inputs        = 7
    outputs       = 3
    hidden1       = 400
    hidden2       = 300
    tau           = 0.001
