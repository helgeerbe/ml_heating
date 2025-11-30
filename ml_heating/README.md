# ML Heating Control (Stable)

Physics-based machine learning heating control system with online learning.

## Overview

This stable channel provides production-ready ML heating control with automatic model optimization and real-time learning capabilities.

## Features

- **Heat Balance Controller** - 🆕 **Intelligent 3-phase temperature control** (CHARGING/BALANCING/MAINTENANCE modes)
- **Trajectory Prediction** - 4-hour thermal forecasting with oscillation prevention
- **Physics-based ML optimization** - Intelligent heating control using real-world physics models
- **Real-time dashboard** - Advanced analytics and performance monitoring
- **Multi-model comparison** - Compare different heating strategies
- **Automated backup system** - Protect your ML models and configurations
- **Seasonal learning** - Automatically adapts to seasonal patterns
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
