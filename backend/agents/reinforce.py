import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from .base_agent import BaseAgent

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

class ReinforceAgent(BaseAgent):
    def __init__(self, state_size, action_size, alpha=0.001, gamma=0.99):
        super().__init__(state_size, action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        
        # Policy network
        self.policy = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
        
        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        # For state conversion
        self.state_size = state_size
        self.state_shape = (1, state_size)
    
    def _to_tensor(self, state):
        if isinstance(state, (int, np.integer)):
            state_tensor = torch.zeros(1, self.state_size)
            state_tensor[0, state] = 1
        else:
            state_tensor = torch.FloatTensor(state).view(self.state_shape)
        return state_tensor.to(self.device)
    
    def choose_action(self, state):
        state_tensor = self._to_tensor(state)
        with torch.no_grad():
            probs = self.policy(state_tensor)
            m = Categorical(probs)
            action = m.sample()
            
        # Store state and action for training
        self.states.append(state)
        self.actions.append(action.item())
        return action.item()
    
    def learn(self, state, action, reward, next_state, done):
        # Store reward
        self.rewards.append(reward)
        
        if done:
            # Calculate discounted rewards
            G = 0
            returns = []
            for r in reversed(self.rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns).to(self.device)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Calculate loss and update policy
            loss = 0
            for state, action, G in zip(self.states, self.actions, returns):
                state_tensor = self._to_tensor(state)
                probs = self.policy(state_tensor)
                m = Categorical(probs)
                loss -= m.log_prob(torch.tensor(action).to(self.device)) * G
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Clear episode memory
            self.states = []
            self.actions = []
            self.rewards = []
    
    def get_policy(self):
        with torch.no_grad():
            policy = []
            for state in range(self.state_size):
                state_tensor = self._to_tensor(state)
                probs = self.policy(state_tensor)
                action = torch.argmax(probs).item()
                policy.append(action)
            return np.array(policy)
    
    def reset(self):
        # Reinitialize policy network
        self.policy = PolicyNetwork(self.state_size, self.action_size).to(self.device)
        
        # Reset optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.alpha)
        
        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = [] 