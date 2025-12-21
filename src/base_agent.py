# src/base_agent.py
from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract Base Class for Reinforcement Learning Agents.
    Defines the interface that all agents (MC, SARSA) must implement.
    """
    def __init__(self, n_actions, n_states):
        self.n_actions = n_actions
        self.n_states = n_states

    @abstractmethod
    def get_action(self, state):
        """Select an action based on the current state (e.g., epsilon-greedy)."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update the agent's knowledge (Q-table) based on experience."""
        pass