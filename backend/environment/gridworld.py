import numpy as np

class GridWorld:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.reset()
        
        # Define penalty square position (somewhere in the middle of the grid)
        self.penalty_pos = (2, 2)  # You can adjust this position
        
    def reset(self):
        self.state = (0, 0)  # Start at top-left corner
        self.goal = (self.width-1, self.height-1)  # Goal at bottom-right
        return self._get_state(), self.state  # Return both state and position
    
    def step(self, action):
        # Actions: 0: up, 1: right, 2: down, 3: left
        x, y = self.state
        print(f"GridWorld step - Current position: ({x}, {y})")
        
        if action == 0: y = max(0, y - 1)
        elif action == 1: x = min(self.width - 1, x + 1)
        elif action == 2: y = min(self.height - 1, y + 1)
        elif action == 3: x = max(0, x - 1)
        
        self.state = (x, y)
        print(f"GridWorld step - New position: ({x}, {y})")
        
        done = self.state == self.goal
        
        # Calculate reward
        if done:
            reward = 1.0
        elif self.state == self.penalty_pos:
            reward = -1.0  # Penalty for stepping on the penalty square
        else:
            reward = -0.01  # Small negative reward for each step
        
        print(f"GridWorld step - Reward: {reward}, Done: {done}")
        return self._get_state(), reward, done, self.state  # Return position along with other info
    
    def _get_state(self):
        return self.state[0] + self.state[1] * self.width
    
    def get_grid_info(self):
        """Return information about special grid positions for visualization"""
        return {
            'width': self.width,
            'height': self.height,
            'start': (0, 0),
            'goal': self.goal,
            'penalty': self.penalty_pos
        } 