class GridRenderer {
  constructor(containerId, width, height, cellSize = 50) {
    this.container = document.getElementById(containerId);
    this.width = width;
    this.height = height;
    this.cellSize = cellSize;

    this.canvas = document.createElement('canvas');
    this.canvas.width = width * cellSize;
    this.canvas.height = height * cellSize;
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');

    this.agentPos = [0, 0];
    this.goalPos = [width - 1, height - 1];
    this.penaltyPos = [2, 2];  // Default position

    // Initialize the grid
    this.drawGrid();
  }

  drawGrid() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw grid lines
    for (let x = 0; x <= this.width; x++) {
      this.ctx.beginPath();
      this.ctx.moveTo(x * this.cellSize, 0);
      this.ctx.lineTo(x * this.cellSize, this.height * this.cellSize);
      this.ctx.stroke();
    }
    for (let y = 0; y <= this.height; y++) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y * this.cellSize);
      this.ctx.lineTo(this.width * this.cellSize, y * this.cellSize);
      this.ctx.stroke();
    }

    // Draw penalty square
    this.ctx.fillStyle = '#FF6B6B';  // Red color for penalty
    this.ctx.fillRect(
      this.penaltyPos[0] * this.cellSize,
      this.penaltyPos[1] * this.cellSize,
      this.cellSize, this.cellSize
    );

    // Draw goal
    this.ctx.fillStyle = '#4CAF50';
    this.ctx.fillRect(
      this.goalPos[0] * this.cellSize,
      this.goalPos[1] * this.cellSize,
      this.cellSize, this.cellSize
    );

    // Draw agent
    this.ctx.fillStyle = '#2196F3';
    this.ctx.fillRect(
      this.agentPos[0] * this.cellSize + 5,
      this.agentPos[1] * this.cellSize + 5,
      this.cellSize - 10, this.cellSize - 10
    );
  }

  updateGridInfo(gridInfo) {
    this.width = gridInfo.width;
    this.height = gridInfo.height;
    this.goalPos = gridInfo.goal;
    this.penaltyPos = gridInfo.penalty;
    this.drawGrid();
  }

  updateAgentPosition(x, y) {
    this.agentPos = [x, y];
    this.drawGrid();
  }

  // Draw policy arrows
  drawPolicy(policy) {
    const arrowSize = this.cellSize * 0.3;
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const action = policy[y * this.width + x];
        const centerX = x * this.cellSize + this.cellSize / 2;
        const centerY = y * this.cellSize + this.cellSize / 2;

        this.ctx.beginPath();
        this.ctx.moveTo(centerX, centerY);

        // Draw arrow based on action (0: up, 1: right, 2: down, 3: left)
        switch (action) {
          case 0: this.ctx.lineTo(centerX, centerY - arrowSize); break;
          case 1: this.ctx.lineTo(centerX + arrowSize, centerY); break;
          case 2: this.ctx.lineTo(centerX, centerY + arrowSize); break;
          case 3: this.ctx.lineTo(centerX - arrowSize, centerY); break;
        }

        this.ctx.stroke();
      }
    }
  }
} 