import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve

env_name = "BipedalWalker-v3"
filename = "plots/"+env_name+".png"
load = True

env = gym.make(env_name)
agent = Agent(alpha=0.001, beta=0.001, input_shape=env.observation_space.shape,
				n_actions=env.action_space.shape[0], tau=0.005,  env=env,
				name=env_name)

best_score = env.reward_range[0]
score_history = []
n_games = 10

if load:
	try:
		agent.load()
	except:
		print("Error while loading...")

device = agent.get_device()
print("Device is : ", device)

for i in range(n_games):
	obs = env.reset()
	done = False
	score = 0

	while not done:
		action = agent.choose_action(obs)
		obs_, reward, done, info = env.step(action)
		agent.store_memory(obs, obs_, action, reward, done)
		agent.learn()
		obs = obs_
		score += reward

		env.render()

	score_history.append(score)
	avg_score = np.mean(score_history[-100:])

	if avg_score > best_score:
		best_score = avg_score
		agent.save()

	print("Game: ", i+1, " avg_score: %.1f " % avg_score)

x = [i+1 for i in range(n_games)]
plot_learning_curve(x, score_history, filename)
