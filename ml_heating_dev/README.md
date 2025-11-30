# ML Heating Control (Development)

Development channel with latest features and debug capabilities.

⚠️ **Development Release** - For testing purposes only!

## Overview

This development channel provides access to the latest experimental features, debugging capabilities, and development tools for ML heating control.

## Development Features

- **Heat Balance Controller** - 🆕 **Latest intelligent 3-phase temperature control** (CHARGING/BALANCING/MAINTENANCE modes)
- **Trajectory Prediction** - 4-hour thermal forecasting with oscillation prevention
- **Latest experimental features** - Access cutting-edge ML improvements
- **Debug logging enabled** - Detailed DEBUG level logging for troubleshooting
- **Development API access** - Enable external tools and Jupyter notebooks
- **Advanced diagnostics** - Enhanced monitoring and analysis tools
- **Jupyter integration** - Direct notebook access for model analysis

## Installation

1. Add this repository to Home Assistant:
   ```
   https://github.com/helgeerbe/ml_heating
   ```
2. Install "ML Heating Control (Development)" from the Add-on Store
3. Configure with your Home Assistant entities
4. Start the add-on

## Development Configuration

Additional development settings available:

- `enable_dev_api: true` - Development API enabled by default
- `log_level: DEBUG` - Verbose logging for troubleshooting
- `dev_api_key` - Set API key for external tool access

## Dashboard

Access the development dashboard at: `http://[HOST]:3001`

Enhanced development features:
- Real-time debug information
- Model training metrics
- Advanced physics diagnostics
- Development API status

## Development API

When enabled, the development API provides:
- Model state access for Jupyter notebooks
- Real-time data streaming
- Advanced diagnostics endpoints
- Custom experiment running

## ⚠️ Development Notice

This is a **development release** intended for testing and experimentation:

- **Experimental features** - May contain unfinished functionality
- **Potential bugs** - Development code may have stability issues
- **Breaking changes** - Updates may introduce incompatible changes
- **Manual updates** - Auto-updates disabled for safety
- **Advanced users** - Requires understanding of ML heating concepts

## Production Alternative

**For production use**, install the stable version: **"ML Heating Control"**

The stable channel provides:
- Production-tested features
- Automatic updates
- Optimized performance
- Validated configurations

## Support

- **Development Issues**: Report via GitHub with DEBUG logs
- **Feature Requests**: Suggest improvements for testing
- **Documentation**: See configuration tab for setup details
- **Community**: Share development experiences and findings
