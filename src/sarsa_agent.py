import numpy as np
from src.base_agent import Agent

class SarsaAgent(Agent):
    def __init__(self, n_actions, n_states, epsilon, alpha, gamma=1.0):
        super().__init__(n_actions, n_states)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))

    def decay_epsilon(self, decay_rate, min_epsilon):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            values = self.Q[state, :]
            max_value = np.max(values)
            actions = np.where(values == max_value)[0]
            return np.random.choice(actions)

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA Update: Uses the next action that will actually be taken.
        """
        current_q = self.Q[state, action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]
            
        self.Q[state, action] = current_q + self.alpha * (target - current_q)