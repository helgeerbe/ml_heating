# ML Heating Control (Stable)

Physics-based machine learning heating control system with online learning.

## Overview

This stable channel provides production-ready ML heating control with automatic model optimization and real-time learning capabilities.

## Features

- **ThermalEquilibriumModel** - Physics-based machine learning with continuous parameter adaptation
- **Delta Forecast Calibration** - 🆕 **Local weather forecast calibration** for enhanced thermal prediction accuracy
- **Enhanced Model Wrapper** - Intelligent outlet temperature prediction with smart rounding
- **Adaptive Learning** - Real-time parameter optimization based on prediction feedback
- **Multi-Heat Source Integration** - PV solar, fireplace, and electronics heat coordination
- **Active & Shadow Modes** - Safe testing mode alongside production heat curve operation
- **Live Performance Tracking** - Real-time confidence and accuracy monitoring
- **Comprehensive Safety** - Blocking detection, gradual temperature control, and health monitoring
- **External heat source detection** - Accounts for PV, fireplace, and other heat sources

## Installation

1. Add this repository to Home Assistant:
   ```
   https://github.com/helgeerbe/ml_heating
   ```
2. Install "ML Heating Control" from the Add-on Store
3. Configure with your Home Assistant entities
4. Start the add-on

## Configuration

The add-on requires configuration of various Home Assistant entities including:

- Temperature sensors (indoor, outdoor, outlet)
- Climate control entity
- InfluxDB connection (for historical data)
- Optional: PV forecast, fireplace detection, etc.

See the add-on configuration tab for detailed setup instructions.

## Dashboard

Access the ML Heating dashboard at: `http://[HOST]:3001`

The dashboard provides:
- Real-time performance metrics
- Model prediction accuracy
- System learning status
- Physics validation results

## Support

- **Documentation**: See configuration tab for detailed setup
- **Issues**: Report bugs via GitHub repository
- **Development**: For testing latest features, install the Development channel separately

## Production Features

- **Auto-updates**: Enabled for stable releases
- **Optimized logging**: INFO level for production use
- **Validated models**: Thoroughly tested configurations
- **Physics validation**: Real-world constraint checking
