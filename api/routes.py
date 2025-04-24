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
    
    state, pos = env.reset()  # Get both state and position
    total_reward = 0
    policy = []
    
    while True:
        action = current_agent.choose_action(state)
        next_state, reward, done, next_pos = env.step(action)  # Get position from step
        current_agent.learn(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    # Get current policy for visualization
    policy = current_agent.get_policy()
    
    response_data = {
        'reward': total_reward,
        'agent_pos': next_pos,
        'policy': policy.tolist() if isinstance(policy, np.ndarray) else policy,
    }
    
    # Add Q-table for Q-learning visualization
    if isinstance(current_agent, QLearningAgent):
        response_data['q_table'] = current_agent.q_table.tolist()
    
    return jsonify(response_data)

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
    print("\n=== Starting play_policy endpoint ===")
    global current_agent
    
    data = request.json
    algorithm = data['algorithm']
    print(f"Requested algorithm: {algorithm}")
    
    # Reset environment and get initial state
    state, pos = env.reset()
    print(f"Initial state: {state}, position: {pos}")
    
    if current_agent is None:
        print("Creating new agent")
        current_agent = agents[algorithm]
    else:
        print("Using existing agent")
    
    trajectory = []
    done = False
    total_reward = 0
    steps = 0
    max_steps = 100
    last_pos = None  # Add this to track if agent is stuck
    
    # Add initial state to trajectory
    trajectory.append({
        'state': tuple(map(int, pos)),
        'action': None,
        'reward': float(0)
    })
    print(f"Added initial position to trajectory: {pos}")
    
    try:
        while not done and steps < max_steps:
            # Check if agent is stuck (same position twice)
            if last_pos == pos:
                print("Agent appears to be stuck")
                break
            last_pos = pos
            
            # Get action from current policy
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
                action = int(np.argmax(current_agent.q_table[state]))
            
            print(f"Step {steps}: Chosen action: {action}")
            next_state, reward, done, next_pos = env.step(action)
            print(f"Step {steps}: New state: {next_state}, new position: {next_pos}, reward: {reward}, done: {done}")
            
            trajectory.append({
                'state': tuple(map(int, next_pos)),
                'action': int(action),
                'reward': float(reward)
            })
            
            total_reward += reward
            state = next_state
            pos = next_pos
            steps += 1
        
        print(f"Final trajectory length: {len(trajectory)}")
        print(f"Final trajectory: {trajectory}")
        return jsonify({
            'trajectory': trajectory,
            'total_reward': float(total_reward),
            'stuck': last_pos == pos  # Add this flag to indicate if agent got stuck
        })
    except Exception as e:
        print(f"Error during policy execution: {e}")
        raise

@api_bp.route('/get_grid_info', methods=['GET'])
def get_grid_info():
    return jsonify(env.get_grid_info()) 