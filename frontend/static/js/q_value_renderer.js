class QValueRenderer {
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

    // Colors for different value ranges
    this.valueColorScale = (value) => {
      if (value > 0) return `rgba(0, 255, 0, ${Math.min(Math.abs(value), 1)})`;
      return `rgba(255, 0, 0, ${Math.min(Math.abs(value), 1)})`;
    };
  }

  drawQValues(qTable) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw grid and Q-values for each cell
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const cellX = x * this.cellSize;
        const cellY = y * this.cellSize;
        const stateIdx = x + y * this.width;

        // Draw cell border
        this.ctx.strokeStyle = '#000';
        this.ctx.strokeRect(cellX, cellY, this.cellSize, this.cellSize);

        // Draw triangles for each action
        this.drawActionTriangles(cellX, cellY, qTable[stateIdx]);
      }
    }
  }

  drawActionTriangles(x, y, qValues) {
    const centerX = x + this.cellSize / 2;
    const centerY = y + this.cellSize / 2;

    // Up triangle
    this.drawTriangle(
      centerX, centerY,
      [x, y],
      [x + this.cellSize, y],
      qValues[0]
    );

    // Right triangle
    this.drawTriangle(
      centerX, centerY,
      [x + this.cellSize, y],
      [x + this.cellSize, y + this.cellSize],
      qValues[1]
    );

    // Down triangle
    this.drawTriangle(
      centerX, centerY,
      [x + this.cellSize, y + this.cellSize],
      [x, y + this.cellSize],
      qValues[2]
    );

    // Left triangle
    this.drawTriangle(
      centerX, centerY,
      [x, y + this.cellSize],
      [x, y],
      qValues[3]
    );
  }

  drawTriangle(centerX, centerY, point1, point2, value) {
    this.ctx.beginPath();
    this.ctx.moveTo(centerX, centerY);
    this.ctx.lineTo(point1[0], point1[1]);
    this.ctx.lineTo(point2[0], point2[1]);
    this.ctx.closePath();

    // Fill with color based on value
    this.ctx.fillStyle = this.valueColorScale(value);
    this.ctx.fill();

    // Draw value text
    this.ctx.save();
    this.ctx.fillStyle = 'black';
    this.ctx.font = '10px Helvetica';
    const textX = (centerX + point1[0] + point2[0]) / 3;
    const textY = (centerY + point1[1] + point2[1]) / 3;
    this.ctx.fillText(value.toFixed(2), textX, textY);
    this.ctx.restore();
  }
} 