# src/mc_agent.py
import numpy as np
from src.base_agent import Agent

class MonteCarloAgent(Agent):
    """
    Every-Visit Monte Carlo Control agent.
    Based on the logic required for the project.
    """
    def __init__(self, n_actions, n_states, epsilon, alpha, gamma=1.0):
        super().__init__(n_actions, n_states)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))

    def decay_epsilon(self, decay_rate, min_epsilon):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_action(self, state):
        """Select action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Random tie-breaking for max values
            values = self.Q[state, :]
            max_value = np.max(values)
            actions = np.where(values == max_value)[0]
            return np.random.choice(actions)

    def update(self, episode_data):
        """
        Update Q-table based on the full episode (Monte Carlo).
        episode_data: list of (state, action, reward)
        """
        G = 0
        for state, action, reward in reversed(episode_data):
            G = self.gamma * G + reward
            old_value = self.Q[state, action]
            # Update rule: Q(s,a) = Q(s,a) + alpha * (G - Q(s,a)) 
            self.Q[state, action] = old_value + self.alpha * (G - old_value)

    def print_q_table(self):
        """Helper to visualize the learned values in the console."""
        print("\n--- Current Q-Table (Values) ---")
        print(np.round(self.Q, 3))