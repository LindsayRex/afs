# AFS Flow Dynamics HUD

Real-time monitoring dashboard for AFS optimization flows with sparsity visualization.

## Features

- **Sparsity Monitoring**: Real-time visualization of solution sparsity (0.0 = highly compressed, 1.0 = dense)
- **Color-coded Feedback**: Blue (sparse/good) → Green (balanced) → Red (dense/needs regularization)
- **Compression Animations**: Visual feedback when solutions become significantly more sparse
- **Health Indicators**: System health based on sparsity levels and optimization progress
- **Interactive Controls**: Test different sparsity scenarios

## Quick Start

1. **Install dependencies** (already done):
   ```bash
   cd src/ux
   npm install
   ```

2. **Start the dashboard**:
   ```bash
   cd src/ux
   npm run dev
   # or
   python -m http.server 8000
   ```

3. **Open in browser**:
   Navigate to `http://localhost:8000/sparsity_hud_demo.html`

## Usage

- **Sparsity Bar**: Shows current solution sparsity with color coding
- **Interactive Slider**: Adjust sparsity values to see visual changes
- **Compression Events**: Click "Trigger Compression Event" to see sparsity animations
- **Event Log**: Real-time updates on sparsity changes

## Integration

The dashboard reads telemetry data in this format:

```json
{
  "sparsity_wx": 0.15,
  "phase": "GREEN",
  "E": 12.345,
  "eta_dd": 0.72,
  "gamma": 1.2e-6
}
```

## Dependencies

- **anime.js**: For smooth animations and transitions
- **Python HTTP Server**: For local development (or any web server)

## Development

- Edit `sparsity_hud_demo.html` for UI changes
- Modify `package.json` for dependency updates
- Run `npm install` after changing dependencies