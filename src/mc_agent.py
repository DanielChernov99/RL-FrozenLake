# src/mc_agent.py
import numpy as np
from src.base_agent import Agent

class MonteCarloAgent(Agent):
    """
    Implements the Every-Visit Monte Carlo Control algorithm.
    
    This agent learns from complete episodes. It calculates the return (G) 
    for each state-action pair visited during an episode and updates the Q-values 
    using a constant alpha (learning rate) update rule.
    """
    def __init__(self, n_actions, n_states, epsilon, alpha, gamma=1.0):
        super().__init__(n_actions, n_states)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))

    def decay_epsilon(self, decay_rate, min_epsilon):
        """
        Decays the exploration rate (epsilon) by multiplying it by the decay rate.
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_action(self, state):
        """
        Selects an action using the Epsilon-Greedy policy.

        With probability epsilon, a random action is chosen (exploration).
        Otherwise, the action with the highest Q-value is chosen (exploitation).
        Ties in Q-values are broken randomly to prevent bias.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            values = self.Q[state, :]
            max_value = np.max(values)
            actions = np.where(values == max_value)[0]
            return np.random.choice(actions)

    def update(self, episode_data):
        """
        Update Q-table based on the full episode (Monte Carlo update).

        Iterates through the episode in reverse order to calculate the cumulative discounted 
        return (G) and updates the Q-values using the incremental mean formula:
        Q(s,a) = Q(s,a) + alpha * (G - Q(s,a))
        """
        G = 0
        for state, action, reward in reversed(episode_data):
            G = self.gamma * G + reward
            old_value = self.Q[state, action]
            self.Q[state, action] = old_value + self.alpha * (G - old_value)

    def print_q_table(self):
        """Helper to visualize the learned values in the console."""
        print("\n--- Current Q-Table (Values) ---")
        print(np.round(self.Q, 3))