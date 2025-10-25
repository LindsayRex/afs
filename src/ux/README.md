# AFS Flow Dynamics HUD

Real-time monitoring dashboard for AFS optimization flows with sparsity visualization.

## Features

- **Sparsity Monitoring**: Real-time visualization of solution compression (0.0 = highly compressed, 1.0 = dense)
- **Color-coded Feedback**: Blue (sparse/good) → Green (balanced) → Red (dense/needs regularization)
- **Interactive Animations**: Compression waves, elastic effects, and smooth transitions
- **Particle Effects**: Energy particles appear for very sparse solutions (< 0.2)
- **Timeline Progress**: Visual progress bar for optimization runs
- **Event System**: Real-time notifications of sparsity changes and optimization events
- **Flow Selection**: Switch between different optimization flows by name
- **Theme Toggle**: Switch between dark and light modes
- **Sound Effects**: Optional audio feedback for compression events
- **Simulation Mode**: Full optimization run simulation with realistic sparsity evolution

## Quick Start

1. **Install dependencies** (already done):
   ```bash
   cd src/ux
   npm install
   ```

2. **Start the dashboard server**:
   ```bash
   cd src/ux
   npm run dev
   # or
   python -m http.server 8000
   ```

3. **Open in browser**:
   Navigate to `http://localhost:8000/sparsity_hud_demo.html`

## Interactive Features

### Controls Panel
- **Sparsity Slider**: Adjust sparsity values to see real-time color changes
- **Compression Event**: Trigger dramatic sparsity reduction with wave animations
- **Add Event**: Manually add sparsity change notifications
- **Simulate Optimization**: Run a complete optimization simulation

### UI Controls (Top-Right)
- **Theme Toggle**: 🌙 Dark Mode / ☀️ Light Mode
- **Sound Toggle**: 🔊 Sound On / 🔇 Sound Off

### Flow Selector (Top-Left)
- **Flow Dropdown**: Select from available optimization flows by name
- **Dynamic Updates**: Dashboard automatically updates to show selected flow's data
- **Flow Events**: Flow changes are logged in the event ticker
- **State Reset**: Timeline resets when switching flows

### Visual Elements

#### Sparsity Bar
- **Blue (0.0-0.3)**: Highly compressed solutions - good sparsity
- **Green (0.3-0.7)**: Balanced energy distribution
- **Red (0.7-1.0)**: Dense solutions - may need regularization

#### Particle System
When sparsity drops below 0.2, floating energy particles appear representing compressed solution energy.

#### Timeline Progress
Shows optimization completion percentage with smooth animated progress bar.

#### Event Ticker
Real-time log of sparsity changes and optimization milestones.

## Technical Details

### Sparsity Metric
```
sparsity_wx = ||x||₁ / (||x||₂ * √n)
```
- **Range**: 0.0 (perfect compression) to ~1.0 (fully dense)
- **Physical Meaning**: Measures energy concentration in multiscale representations
- **FDA Compliant**: Aligns with Flow-Dynamic Analysis sparsity principles

### Animation Library
- **anime.js v3.2.1**: High-performance CSS/SVG/DOM animations
- **Easing Functions**: Smooth transitions with elastic and quadratic curves
- **Timeline Control**: Coordinated multi-element animations

### Audio System
- **Web Audio API**: Procedural sound generation for compression events
- **Frequency Sweep**: 800Hz → 400Hz exponential decay
- **Optional**: Can be disabled via sound toggle

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

## Demo Scenarios

1. **Normal Operation**: Sparsity stabilizes around 0.15-0.25
2. **Compression Event**: Sudden drop from 0.8 to 0.2 with wave animation
3. **Dense Solution**: Sparsity > 0.7 triggers red warning state
4. **Full Optimization**: 20-step simulation showing complete flow dynamics

## Dependencies

- **anime.js ^3.2.1** - Animation library for smooth transitions
- **Node.js/npm** - Package management for web dependencies

## Development

- Edit `sparsity_hud_demo.html` for UI changes
- Modify `package.json` for dependency updates
- Run `npm install` after changing dependencies

## Future Enhancements

- Real-time telemetry integration
- Multiple visualization modes
- Historical trend analysis
- Customizable alert thresholds
- Export capabilities for reports
- Flow comparison and analysis tools
- Performance metrics dashboard
