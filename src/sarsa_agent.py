import numpy as np
from src.base_agent import Agent

class SarsaAgent(Agent):
    """
    Implements the SARSA (State-Action-Reward-State-Action) algorithm.

    SARSA is an On-Policy Temporal Difference (TD) learning algorithm. 
    It updates the Q-values based on the action actually taken by the current policy 
    in the next state, rather than maximizing over possible next actions (like Q-Learning).
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

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Performs the SARSA update step.

        Update Rule: Q(s,a) = Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        Unlike Q-learning, this uses 'next_action' (a'), which is the specific action 
        chosen by the policy for the next step, maintaining the on-policy property.
        """
        current_q = self.Q[state, action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]
            
        self.Q[state, action] = current_q + self.alpha * (target - current_q)