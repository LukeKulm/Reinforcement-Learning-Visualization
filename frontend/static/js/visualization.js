class RLVisualizer {
  constructor() {
    this.grid = new GridRenderer('gridworld', 5, 5);
    this.rewardChart = this.initRewardChart();
    this.isTraining = false;
    this.currentEpisode = 0;
    this.totalReward = 0;
    this.cumulativeReward = 0;
    this.rewardHistory = [];
    this.isPlayingPolicy = false;
    this.policyPlaybackCancelled = false;
    this.qValueGrid = new QValueRenderer('q-value-grid', 5, 5);

    this.setupControls();
    this.setupAlgorithmSpecificParams();

    // Fetch grid information
    this.fetchGridInfo();
  }

  initRewardChart() {
    const ctx = document.getElementById('reward-chart').getContext('2d');
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Cumulative Reward',
          data: [],
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              font: {
                family: "Helvetica Neue"
              }
            }
          },
          x: {
            min: 0,
            max: 500,  // Show 500 episodes worth of space initially
            ticks: {
              font: {
                family: "Helvetica Neue"
              }
            }
          }
        },
        plugins: {
          legend: {
            labels: {
              font: {
                family: "Helvetica Neue"
              }
            }
          }
        }
      }
    });
  }

  setupControls() {
    const startButton = document.getElementById('start-training');
    const resetButton = document.getElementById('reset');
    const algorithmSelect = document.getElementById('algorithm');
    const playPolicyButton = document.getElementById('play-policy');

    startButton.addEventListener('click', () => {
      if (this.isTraining) {
        this.pauseTraining();
        startButton.textContent = 'Resume Training';
      } else {
        // Cancel any ongoing policy playback when resuming training
        this.policyPlaybackCancelled = true;
        this.isPlayingPolicy = false;
        playPolicyButton.disabled = false;

        this.startTraining();
        startButton.textContent = 'Pause Training';
      }
    });

    resetButton.addEventListener('click', () => this.resetEnvironment());

    algorithmSelect.addEventListener('change', () => {
      if (this.isTraining) {
        this.pauseTraining();
        startButton.textContent = 'Start Training';
      }
      this.resetEnvironment();
      this.setupAlgorithmSpecificParams();
    });

    playPolicyButton.addEventListener('click', async () => {
      if (!this.isPlayingPolicy && !this.isTraining) {
        this.isPlayingPolicy = true;
        playPolicyButton.disabled = true;
        await this.playPolicy();
        playPolicyButton.disabled = false;
        this.isPlayingPolicy = false;
      }
    });
  }

  setupAlgorithmSpecificParams() {
    const algorithm = document.getElementById('algorithm').value;
    const paramContainer = document.getElementById('parameters');
    paramContainer.innerHTML = ''; // Clear existing parameters

    const commonParams = {
      'Learning Rate (α)': { min: 0.001, max: 1, step: 0.001, default: algorithm === 'dqn' ? 0.001 : 0.1 },
      'Discount Factor (γ)': { min: 0.1, max: 0.99, step: 0.01, default: 0.99 },
      'Exploration Rate (ε)': { min: 0.01, max: 1, step: 0.01, default: 0.1 }
    };

    const dqnParams = {
      'Epsilon Decay': { min: 0.9, max: 0.999, step: 0.001, default: 0.995 },
      'Batch Size': { min: 16, max: 128, step: 16, default: 32 }
    };

    const params = algorithm === 'dqn' ? { ...commonParams, ...dqnParams } : commonParams;

    Object.entries(params).forEach(([name, config]) => {
      const div = document.createElement('div');
      div.className = 'param-control';

      const label = document.createElement('label');
      label.textContent = name;

      const paramRow = document.createElement('div');
      paramRow.className = 'param-row';

      const input = document.createElement('input');
      input.type = 'range';
      input.min = config.min;
      input.max = config.max;
      input.step = config.step;
      input.value = config.default;

      const value = document.createElement('span');
      value.className = 'param-value';
      value.textContent = config.default;

      input.addEventListener('input', () => {
        value.textContent = input.value;
        this.updateParameter(name, parseFloat(input.value));
      });

      paramRow.appendChild(input);
      paramRow.appendChild(value);
      div.appendChild(label);
      div.appendChild(paramRow);
      paramContainer.appendChild(div);
    });
  }

  async startTraining() {
    this.isTraining = true;

    while (this.isTraining) {
      const response = await fetch('/api/train_step', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          algorithm: document.getElementById('algorithm').value,
          episode: this.currentEpisode
        })
      });

      const data = await response.json();
      this.updateVisualization(data);

      await new Promise(resolve => requestAnimationFrame(resolve));
    }
  }

  async requestAndShowPolicy() {
    const response = await fetch('/api/get_policy', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        algorithm: document.getElementById('algorithm').value
      })
    });

    const data = await response.json();
    this.grid.drawPolicy(data.policy);
  }

  async playPolicy() {
    console.log("Play Policy button clicked");
    const algorithm = document.getElementById('algorithm').value;
    console.log("Current algorithm:", algorithm);

    try {
      console.log("Making API call to /play_policy");
      const response = await fetch('/api/play_policy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          algorithm: algorithm
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Received policy data:", data);

      if (!data.trajectory || data.trajectory.length === 0) {
        console.error("Received empty trajectory");
        return;
      }

      // Disable controls during playback
      const playPolicyButton = document.getElementById('play-policy');
      const startButton = document.getElementById('start-training');
      playPolicyButton.disabled = true;
      startButton.disabled = true;

      try {
        await this.visualizeTrajectory(data.trajectory);
      } finally {
        // Always re-enable controls and reset playback state
        this.isPlayingPolicy = false;
        playPolicyButton.disabled = false;
        startButton.disabled = false;

        // If agent got stuck, show a message
        if (data.stuck) {
          console.log("Agent got stuck - policy may need more training");
          // Optionally show a message to the user
          alert("Agent got stuck - the policy may need more training");
        }
      }
    } catch (error) {
      console.error("Error during policy playback:", error);
      // Make sure controls are re-enabled even if there's an error
      this.isPlayingPolicy = false;
      document.getElementById('play-policy').disabled = false;
      document.getElementById('start-training').disabled = false;
    }
  }

  async visualizeTrajectory(trajectory) {
    if (!trajectory || trajectory.length === 0) {
      console.log("Empty trajectory received");
      return;
    }

    // Reset the cancellation flag
    this.policyPlaybackCancelled = false;

    console.log("Starting trajectory visualization");
    console.log("Full trajectory:", trajectory);

    // Start from the beginning
    console.log("Setting initial position:", trajectory[0].state);
    this.grid.updateAgentPosition(trajectory[0].state[0], trajectory[0].state[1]);

    // Add initial pause to see starting position
    await new Promise(resolve => setTimeout(resolve, 500));

    // Animate through each state in the trajectory
    for (let i = 1; i < trajectory.length; i++) {
      // Check if playback was cancelled
      if (this.policyPlaybackCancelled) {
        console.log("Policy playback cancelled");
        return;
      }

      const step = trajectory[i];
      console.log(`Step ${i}:`, step);
      console.log(`Moving to position:`, step.state);
      this.grid.updateAgentPosition(step.state[0], step.state[1]);
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    console.log("Trajectory visualization complete");
  }

  pauseTraining() {
    this.isTraining = false;
  }

  updateVisualization(data) {
    // Update episode counter
    document.getElementById('episode-counter').textContent = this.currentEpisode;

    // Update cumulative reward
    this.cumulativeReward += data.reward;
    document.getElementById('total-reward').textContent = this.cumulativeReward.toFixed(2);

    // Update chart with cumulative reward
    this.rewardChart.data.labels.push(this.currentEpisode);
    this.rewardChart.data.datasets[0].data.push(this.cumulativeReward);

    // Only keep last 500 points in memory
    if (this.rewardChart.data.labels.length > 500) {
      this.rewardChart.data.labels.shift();
      this.rewardChart.data.datasets[0].data.shift();
    }

    this.rewardChart.update('none');
    this.currentEpisode++;

    // Update Q-value visualization if using Q-learning
    if (document.getElementById('algorithm').value === 'q-learning' && data.q_table) {
      this.qValueGrid.drawQValues(data.q_table);
    }
  }

  resetEnvironment() {
    // Cancel any ongoing policy playback
    this.policyPlaybackCancelled = true;

    this.currentEpisode = 0;
    this.totalReward = 0;
    this.cumulativeReward = 0;
    this.rewardHistory = [];

    document.getElementById('episode-counter').textContent = '0';
    document.getElementById('total-reward').textContent = '0';

    this.rewardChart.data.labels = [];
    this.rewardChart.data.datasets[0].data = [];

    this.rewardChart.update();

    this.grid.updateAgentPosition(0, 0);

    this.fetchGridInfo();
    fetch('/api/reset', { method: 'POST' });

    // Re-enable controls
    const playPolicyButton = document.getElementById('play-policy');
    const startButton = document.getElementById('start-training');
    playPolicyButton.disabled = false;
    startButton.disabled = false;
  }

  updateParameter(name, value) {
    fetch('/api/update_params', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        parameter: name,
        value: value
      })
    });
  }

  async fetchGridInfo() {
    const response = await fetch('/api/get_grid_info');
    const gridInfo = await response.json();
    this.grid.updateGridInfo(gridInfo);
  }
}

// Initialize visualization when document is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.rlViz = new RLVisualizer();
}); 