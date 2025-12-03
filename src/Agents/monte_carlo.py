from src.agents.base_agent import BaseAgent

class MonteCarloAgent(BaseAgent):
    """"
    every-visit Monte Carlo Agent,the agent learns only at the end of an episode so 
    we must store everything

    """
    def __init__(self, action_space_n, gamma=1.0, epsilon=0.1, alpha=0.1):
        # Init parent
        super().__init__(action_space_n, gamma, epsilon, alpha)
        
        self.episode_memory = []

    def store_step(self, state, action, reward):
        """
        Records a single steps(state,action,reward)
        do not update the Q-table yet
        """
        self.episode_memory.append((state, action, reward))

    def update(self):
        """
        at the end of the episode, updates the Q-table based on the memory we got
        """
        G = 0  # the total return

        # iterate in reverse order over the episode memory
        # and we calc the return G and update the Q-values
        for state, action, reward in reversed(self.episode_memory):
            
            # G = reward_now + (discount_gamma * future_reward)
            G = self.gamma * G + reward
            
            # Retrieve old Q-value
            old_value = self.q_table[state][action]
            
            # Update Q-values
            #Q(state,action) = old_value + alpha * (return - old_value)
            self.q_table[state][action] = old_value + self.alpha * (G - old_value)
            
        # Reset for next episode
        self.episode_memory = []