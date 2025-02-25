from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
    @abstractmethod
    def choose_action(self, state):
        pass
        
    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass
        
    @abstractmethod
    def get_policy(self):
        """Return the current policy for visualization"""
        pass
        
    @abstractmethod
    def reset(self):
        """Reset the agent's policy/weights to initial values"""
        pass 