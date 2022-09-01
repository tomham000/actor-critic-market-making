import numpy as np
import runstats
import pickle

from simulated_environment.environments import RLEnvironment
from simulated_environment.agents.q_learning.q_learning import QLearningAgent
from hyperparameters import parameters


def train_q_learning_agent(displacement_actions=15, spread_actions=30,
                           episodes=parameters['episodes']) -> QLearningAgent:
    env = RLEnvironment(spread_actions=spread_actions, displacement_actions=displacement_actions)
    agent = QLearningAgent(actions=[(i, j) for i in range(displacement_actions) for j in range(spread_actions)],
                           alpha=parameters['alpha'], gamma=parameters['gamma'])

    w_stats = runstats.ExponentialStatistics(decay=0.99)
    reward_stats = runstats.ExponentialStatistics(decay=0.99)

    for episode in range(episodes):
        if episode % 100 == 0 and episode > 1:
            print("Episode: {}, std: {:.2f}, w: {:.2f}, reward: {:.2f}".format(episode, np.sqrt(w_stats.variance()),
                                                                               w_stats.mean(), reward_stats.mean()))

        env.reset()
        state = env.state()

        total_reward = 0
        while env.t < env.T:
            action = agent.choose_action(state)
            observation, reward, done, info = env.step(action)

            agent.learn(state, action, reward, observation)
            total_reward += reward
            state = observation
        w_stats.push(env.W)
        reward_stats.push(total_reward)

    return agent


if __name__ == '__main__':
    q_agent = train_q_learning_agent(episodes=parameters['episodes'])
    with open(f'../../agents/q_learning/tabular_q_agent.pkl', 'wb') as file:
        pickle.dump(q_agent, file)
