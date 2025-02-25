import numpy as np
from .base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(state_size, action_size)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        # Q-Learning update
        best_next_value = np.max(self.q_table[next_state])
        current_value = self.q_table[state, action]
        
        # Q(s,a) = Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
        self.q_table[state, action] = current_value + \
            self.alpha * (reward + self.gamma * best_next_value - current_value)
    
    def get_policy(self):
        # Return the current greedy policy
        return np.argmax(self.q_table, axis=1)
    
    def reset(self):
        # Reset Q-table to zeros
        self.q_table = np.zeros((self.state_size, self.action_size)) 