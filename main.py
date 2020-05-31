import numpy as np
from dql import Agent
from utils import plot_learning_curve, make_env

def main():
    env_name = 'PongNoFrameskip-v4'
    env = make_env(env_name)
    best_score = -np.inf

    load_checkpoint = False
    n_games = 500
    agent = Agent(lr=0.001, gamma=0.99, epsilon=1.0, env_name=env_name,
            n_actions=env.action_space.n, in_dims=env.observation_space.shape,
            batch_size=32, mem_size=2500)

    plot = 'plots/atari.png'

    if load_checkpoint:
        agent.load_model()

    scores, eps_history, steps_history = [], [], []
    n_steps = 0

    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()

        while not done:
            action = agent.choice_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_translition(obs, obs_, action, reward, int(done))
                agent.train()

            env.render()
            n_steps += 1
            obs = obs_

        avg_score = np.mean(scores[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_model()

        print("Step: ", n_steps, "Game: ", i, "avg score: ", avg_score,
        "epsilon: ", agent.epsilon)

        scores.append(score)
        eps_history.append(agent.epsilon)
        steps_history.append(n_steps)

    plot_learning_curve(steps_history, scores, eps_history, plot)


if __name__ == "__main__":
    main()
