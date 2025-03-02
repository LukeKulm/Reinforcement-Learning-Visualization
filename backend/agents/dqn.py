import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from .base_agent import BaseAgent

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, alpha=0.001, gamma=0.99, epsilon=0.1):
        super().__init__(state_size, action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
        # DQN network
        self.model = DQNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.memory = deque(maxlen=2000)
        
        # For converting state to tensor
        self.state_shape = (1, state_size)
    
    def _to_tensor(self, state):
        # Convert state to tensor with proper shape
        if isinstance(state, (int, np.integer)):
            # Convert scalar state to one-hot encoding
            state_tensor = torch.zeros(1, self.state_size)
            state_tensor[0, state] = 1
        else:
            state_tensor = torch.FloatTensor(state).view(self.state_shape)
        return state_tensor.to(self.device)
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = self._to_tensor(state)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
    
    def learn(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([self._process_state(s[0]) for s in batch], 
                            dtype=torch.float32).to(self.device)
        actions = torch.tensor([s[1] for s in batch], 
                             dtype=torch.long).to(self.device)
        rewards = torch.tensor([s[2] for s in batch], 
                             dtype=torch.float32).to(self.device)
        next_states = torch.tensor([self._process_state(s[3]) for s in batch], 
                                 dtype=torch.float32).to(self.device)
        dones = torch.tensor([s[4] for s in batch], 
                           dtype=torch.float32).to(self.device)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if np.random.random() < 0.001:  # Update target network occasionally
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _process_state(self, state):
        # Convert scalar state to one-hot encoding
        if isinstance(state, (int, np.integer)):
            processed_state = np.zeros(self.state_size)
            processed_state[state] = 1
            return processed_state
        return state
    
    def get_policy(self):
        # Get the current greedy policy for all states
        with torch.no_grad():
            policy = []
            for state in range(self.state_size):
                state_tensor = self._to_tensor(state)
                q_values = self.model(state_tensor)
                policy.append(torch.argmax(q_values).item())
            return np.array(policy)
    
    def reset(self):
        # Reinitialize both networks
        self.model = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Reset optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        
        # Clear memory
        self.memory.clear()
        
        # Reset epsilon
        self.epsilon = 0.1  # Reset to initial value 