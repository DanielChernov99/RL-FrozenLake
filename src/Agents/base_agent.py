import numpy as np
from collections import defaultdict

class BaseAgent:
    def __init__(self, action_space_n, gamma=1.0, epsilon=0.1, alpha=0.1):
        self.action_space_n = action_space_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        
        # q-table is initialized as zero's
        self.q_table = defaultdict(lambda: np.zeros(action_space_n))

    def get_action(self, state):
        """
        Epsilon-greedy action selection
        """
        #Exploration 
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_n)
        
        #Exploitation 
        return np.argmax(self.q_table[state])