# ML Heating Control Add-on

Physics-based machine learning heating control system with online learning for Home Assistant.

## About

This add-on transforms your heat pump heating system into an intelligent, self-learning control system that continuously adapts to your home's unique thermal characteristics. Unlike traditional static heat curves, this ML system learns from every heating cycle to optimize both energy efficiency and comfort.

## Features

### Core ML Capabilities
- **Physics-based online learning** - Combines thermodynamic principles with data-driven adaptation
- **Real-time performance tracking** - Dynamic confidence and accuracy monitoring
- **Multi-lag learning** - Captures thermal mass effects for solar, fireplace, and other heat sources
- **Seasonal adaptation** - Automatically adjusts to changing conditions without manual recalibration
- **5-layer safety system** - Comprehensive protection against dangerous operation

### User Interface
- **Professional web dashboard** - Integrated with Home Assistant sidebar
- **Real-time monitoring** - Live performance metrics and system status
- **Easy configuration** - All settings through Home Assistant UI
- **Development API** - Access for Jupyter notebook analysis
- **Automatic backups** - Model preservation during updates

### Advanced Features
- **Shadow mode** - Safe testing without affecting your heating
- **External heat source learning** - Solar PV, fireplace, TV/electronics integration
- **Weather forecast integration** - Proactive heating adjustments
- **Comprehensive diagnostics** - Understanding system decisions and performance

## Installation

### Prerequisites
- Home Assistant OS or Supervised
- Heat pump with controllable outlet temperature
- Indoor and outdoor temperature sensors
- InfluxDB add-on (recommended for historical data)

### Quick Install

1. **Add Repository**
   ```
   Add this repository URL in Home Assistant:
   https://github.com/helgeerbe/ml_heating
   ```

2. **Install Add-on**
   - Navigate to Add-on Store
   - Find "ML Heating Control"
   - Click Install

3. **Configure Settings**
   - Set your entity IDs for temperature sensors
   - Configure heating control entity
   - Adjust safety limits and learning parameters
   - Enable optional features (PV, fireplace, etc.)

4. **Start Add-on**
   - Enable "Start on boot" (recommended)
   - Click Start

### Configuration

#### Required Settings
```yaml
target_indoor_temp_entity: "climate.thermostat"
indoor_temp_entity: "sensor.living_room_temperature" 
outdoor_temp_entity: "sensor.outdoor_temperature"
heating_control_entity: "switch.heating_system"
outlet_temp_entity: "sensor.heat_pump_outlet_temp"
```

#### Optional Features
```yaml
# External heat sources
pv_power_entity: "sensor.solar_power"
fireplace_status_entity: "binary_sensor.fireplace"

# InfluxDB integration
influxdb_host: "a0d7b954-influxdb"
influxdb_database: "homeassistant"

# Development access
enable_dev_api: true
dev_api_key: "your-secret-key"
```

## Dashboard Access

Once running, access the dashboard through:
- **Home Assistant Sidebar** - "ML Heating Control" panel
- **Direct URL** - `http://homeassistant:3001`

### Dashboard Features
- **Overview** - Real-time system status and performance
- **Control** - Start/stop, mode switching, manual controls  
- **Performance** - Live MAE/RMSE, confidence tracking
- **Analysis** - Feature importance and learning progress
- **Configuration** - Live settings management
- **Backup** - Model backup and restore

## Development Access

For advanced users wanting to analyze the system with Jupyter notebooks:

```python
# Install in your notebook environment
pip install requests pandas numpy

# Connect to add-on
from notebooks.addon_client import AddonDevelopmentClient
addon = AddonDevelopmentClient("http://homeassistant:3003", "your-dev-key")

# Download live model and data
live_model = addon.download_live_model()
live_state = addon.get_live_state()
recent_logs = addon.get_recent_logs()
```

## Operation

### Getting Started
1. **Start in Shadow Mode** (recommended)
   - System observes your current heat curve
   - Learns house characteristics safely
   - Provides performance comparison

2. **Monitor Learning Progress**
   - Check dashboard for confidence levels
   - Review learning metrics and milestones
   - Analyze feature importance

3. **Switch to Active Mode**
   - When confidence > 0.9 and MAE < 0.2°C
   - ML takes control of heating
   - Continue monitoring performance

### Performance Expectations

**Learning Milestones:**
- **Cycles 0-200**: Basic learning, confidence building
- **Cycles 200-1000**: Advanced features activate
- **Cycles 1000+**: Mature operation with seasonal adaptation

**Typical Performance:**
- **MAE**: < 0.2°C (excellent), < 0.3°C (good)
- **Confidence**: > 0.9 (optimal), > 0.7 (acceptable)  
- **Energy Savings**: 10-25% vs static heat curves
- **Temperature Stability**: 50-70% variance reduction

## Safety

### Multiple Protection Layers
1. **Absolute temperature limits** (14-65°C configurable)
2. **Rate limiting** (max 2°C change per 30-min cycle)
3. **Blocking detection** (DHW, defrost, disinfection)
4. **Grace periods** (intelligent recovery after blocking)
5. **Physics validation** (thermodynamic compliance)

### Monitoring
- **ML State Sensor** - Real-time system status
- **Error Detection** - Network, sensor, and model errors
- **Comprehensive Logging** - Detailed operation history
- **Alert Integration** - Home Assistant notifications

## Troubleshooting

### Common Issues

**Add-on won't start:**
- Check entity IDs are correct
- Verify InfluxDB connectivity (if used)
- Review add-on logs for configuration errors

**Low confidence/high errors:**
- Ensure stable sensor readings
- Check for missing historical data
- Verify heating system is responsive

**Dashboard not accessible:**
- Confirm port 3001 is not blocked
- Check Home Assistant network configuration
- Review dashboard logs for errors

### Getting Help

1. **Check logs** - Add-on logs show detailed operation
2. **Review configuration** - Validate all entity IDs
3. **Monitor sensors** - Ensure stable, accurate readings
4. **Use shadow mode** - Safe testing and comparison

## Migration from Standalone

If you're currently using the standalone systemd version:

1. **Stop systemd service**
   ```bash
   sudo systemctl stop ml_heating
   sudo systemctl disable ml_heating
   ```

2. **Backup your model**
   ```bash
   cp /opt/ml_heating/ml_model.pkl /backup/location/
   ```

3. **Install add-on** and configure entities

4. **Import existing model**
   - Enable "import_existing_model" in add-on config
   - Set "existing_model_path" to your backup location
   - Restart add-on

Your learning progress will be preserved!

## Support

- **Documentation**: [GitHub Repository](https://github.com/helgeerbe/ml_heating)
- **Issues**: [Report Bugs](https://github.com/helgeerbe/ml_heating/issues)
- **Discussions**: [Community Forum](https://github.com/helgeerbe/ml_heating/discussions)

## Developer Notes

### Build Requirements

This add-on uses symbolic links to reference code from the parent repository:
- `ml_heating_addon/src/` → `../src/`
- `ml_heating_addon/notebooks/` → `../notebooks/`
- `ml_heating_addon/requirements.txt` (copied from root)

These are created automatically during setup and allow the Home Assistant Builder to access necessary files. The symlinks are ignored by git but should be recreated if you clone the repository:

```bash
cd ml_heating_addon
ln -s ../src src
ln -s ../notebooks notebooks
cp ../requirements.txt requirements.txt
```

### Docker Build Context

The Dockerfile is designed to work with two different build contexts:
- **Repository root** - Used by GitHub Actions for multi-arch builds
- **Add-on directory** - Used by Home Assistant Builder

Both contexts work because of the symlinks and relative COPY paths in the Dockerfile.

## License

MIT License - see [LICENSE](https://github.com/helgeerbe/ml_heating/blob/main/LICENSE) for details.

---

*This add-on provides sophisticated ML heating control while maintaining the safety and reliability expected in production home automation systems.*
