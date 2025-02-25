from flask import Blueprint, jsonify, request
from backend.environment.gridworld import GridWorld
from backend.agents.q_learning import QLearningAgent
from backend.agents.dqn import DQNAgent
from backend.agents.reinforce import ReinforceAgent
import numpy as np
import torch

api_bp = Blueprint('api', __name__)

# Global variables to maintain state
env = GridWorld(5, 5)
agents = {
    'q-learning': QLearningAgent(25, 4),
    'dqn': DQNAgent(25, 4),
    'reinforce': ReinforceAgent(25, 4)
}
current_agent = None

@api_bp.route('/train_step', methods=['POST'])
def train_step():
    data = request.json
    algorithm = data['algorithm']
    
    global current_agent
    if current_agent is None or current_agent != agents[algorithm]:
        current_agent = agents[algorithm]
    
    state = env.reset()
    total_reward = 0
    policy = []
    
    while True:
        action = current_agent.choose_action(state)
        next_state, reward, done = env.step(action)
        current_agent.learn(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    # Get current policy for visualization
    policy = current_agent.get_policy()
    
    return jsonify({
        'reward': total_reward,
        'agent_pos': env.state,
        'policy': policy.tolist() if isinstance(policy, np.ndarray) else policy
    })

@api_bp.route('/reset', methods=['POST'])
def reset():
    global current_agent
    if current_agent is not None:
        current_agent.reset()  # Reset the current agent
    env.reset()  # Reset the environment
    return jsonify({'status': 'success'})

@api_bp.route('/update_params', methods=['POST'])
def update_params():
    data = request.json
    if current_agent:
        if data['parameter'] == 'Learning Rate (α)':
            current_agent.alpha = data['value']
        elif data['parameter'] == 'Discount Factor (γ)':
            current_agent.gamma = data['value']
        elif data['parameter'] == 'Exploration Rate (ε)':
            current_agent.epsilon = data['value']
    return jsonify({'status': 'success'})

@api_bp.route('/get_policy', methods=['POST'])
def get_policy():
    data = request.json
    algorithm = data['algorithm']
    
    if current_agent is None:
        current_agent = agents[algorithm]
    
    policy = current_agent.get_policy()
    
    return jsonify({
        'policy': policy.tolist() if isinstance(policy, np.ndarray) else policy
    })

@api_bp.route('/play_policy', methods=['POST'])
def play_policy():
    data = request.json
    algorithm = data['algorithm']
    start_state = env.reset()  # Reset to start state
    
    if current_agent is None:
        current_agent = agents[algorithm]
    
    trajectory = []
    state = start_state
    done = False
    total_reward = 0
    
    while not done and len(trajectory) < 100:  # Add max steps to prevent infinite loops
        # Get action from current policy (no exploration)
        if isinstance(current_agent, DQNAgent) or isinstance(current_agent, ReinforceAgent):
            with torch.no_grad():
                state_tensor = current_agent._to_tensor(state)
                if isinstance(current_agent, DQNAgent):
                    q_values = current_agent.model(state_tensor)
                    action = torch.argmax(q_values).item()
                else:  # ReinforceAgent
                    probs = current_agent.policy(state_tensor)
                    action = torch.argmax(probs).item()
        else:  # QLearningAgent
            action = np.argmax(current_agent.q_table[state])
        
        next_state, reward, done = env.step(action)
        trajectory.append({
            'state': env.state,
            'action': action,
            'reward': reward
        })
        
        total_reward += reward
        state = next_state
    
    return jsonify({
        'trajectory': trajectory,
        'total_reward': total_reward
    })

@api_bp.route('/get_grid_info', methods=['GET'])
def get_grid_info():
    return jsonify(env.get_grid_info()) 