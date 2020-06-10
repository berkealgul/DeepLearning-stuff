import numpy as np
import time as t
from dql import Agent
from utils import *


# Train Mode parameters
train = True
load_checkpoint = True
render = True
n_games = 500
######################

plot = 'plots/atari.png'
plot_loss = 'plots/atari_loss.png'
env_name = 'PongNoFrameskip-v4'

env = make_env(env_name, no_ops=30, clip_rewards=True)
best_score = -np.inf

agent = Agent(lr=0.001, gamma=0.99, epsilon=1.0, env_name=env_name,
        n_actions=env.action_space.n, in_dims=env.observation_space.shape,
        batch_size=32, mem_size=2500)

device = agent.get_device()

print("Device is ", device)

if load_checkpoint:
    agent.load_model()

scores, eps_history, steps_history, game, loss = [], [], [], [], []
n_steps = 0
time = 0

for i in range(n_games):
    done = False
    score = 0
    obs = env.reset()

    while not done:
        action = agent.choice_action(obs)

        obs_, reward, done, info = env.step(action)
        score += reward

        # Train while accoring time
        if train:
            agent.store_translition(obs, obs_, action, reward, int(done))
            time = t.time()
            agent.train()
            time = (t.time() - time)

        if render:
            env.render()

        n_steps += 1
        obs = obs_

    avg_score = np.mean(scores[-100:])
    avg_loss = agent.get_avg_loss()

    if avg_score > best_score:
        best_score = avg_score
        if train:
            agent.save_model()

    scores.append(score)
    eps_history.append(agent.epsilon)
    steps_history.append(n_steps)

    game.append(i)
    loss.append(avg_loss)

    plot_learning_curve(steps_history, scores, eps_history, plot)
    plot_loss_curve(game, loss, plot_loss)

    print("-----------------")
    print("Step: ", n_steps),
    print("Game: ", i)
    print("\nEpsilon: ", agent.epsilon)
    print("Avarage score: ", avg_score)
    print("Avarage loss: ", avg_loss)
    print("\nTraining device: ", device)
    print("Train took: ", time, " secs")
    print("-----------------")

print("Similation Done")
