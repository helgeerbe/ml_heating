# ML Heating Add-on Quick Start

Get your ML Heating Control system up and running in 15 minutes.

## 🚀 Quick Setup

### Prerequisites Check
✅ Home Assistant OS/Supervised  
✅ Heat pump with controllable outlet temperature  
✅ Indoor, outdoor, and outlet temperature sensors  

### 1. Install Add-on (5 minutes)

1. **Add Repository**: Settings → Add-ons → ⋮ → Repositories
   ```
   https://github.com/helgeerbe/ml_heating
   ```

2. **Install**: Find "ML Heating Control" → Install

3. **Basic Config**: Replace with your entity IDs:
   ```yaml
   target_indoor_temp_entity: "climate.thermostat"
   indoor_temp_entity: "sensor.living_room_temperature"
   outdoor_temp_entity: "sensor.outdoor_temperature"
   heating_control_entity: "climate.heating_system"
   outlet_temp_entity: "sensor.heat_pump_outlet_temp"
   ```

4. **Start**: Info tab → Start → Enable "Start on boot"

### 2. Access Dashboard (1 minute)

- **Sidebar**: Look for "ML Heating Control" panel
- **Direct**: `http://homeassistant:3001`

### 3. Monitor Learning (Ongoing)

**Week 1:** Shadow mode - Confidence 0.3→0.7  
**Week 2-4:** Active learning - Confidence 0.7→0.9  
**Month 2+:** Mature operation - Confidence 0.9+  

### 4. Go Active (When Ready)

When confidence > 0.9 and MAE < 0.2°C:
- Dashboard → Control → Toggle "Active Control Mode"

## 🎯 Success Indicators

- ✅ **Confidence**: > 0.9
- ✅ **MAE**: < 0.2°C  
- ✅ **State**: "OK"
- ✅ **Temperature stability**: Improved vs. heat curve

## 📋 Common Entity Examples

### Climate Controls
```yaml
heating_control_entity: "climate.heating_system"
target_indoor_temp_entity: "climate.thermostat"
```

### Temperature Sensors
```yaml
indoor_temp_entity: "sensor.living_room_temperature"
outdoor_temp_entity: "sensor.outdoor_temperature"  
outlet_temp_entity: "sensor.heat_pump_outlet_temp"
```

### Blocking Detection (Optional)
```yaml
dhw_status_entity: "binary_sensor.dhw_active"
defrost_status_entity: "binary_sensor.defrost_active"
```

### External Heat Sources (Optional)
```yaml
pv_power_entity: "sensor.solar_power"
fireplace_status_entity: "binary_sensor.fireplace_active"
```

## 🔧 Troubleshooting

**Add-on won't start:**
- Check all entity IDs exist in HA
- Use Developer Tools → States to verify
- Review add-on logs for errors

**Poor performance:**
- Allow 2-4 weeks for learning
- Verify stable sensor readings
- Check cycle timing (30min recommended)

**Dashboard not loading:**
- Check port 3001 is accessible
- Verify add-on is running
- Review network configuration

## 📚 Next Steps

- **Full Guide**: [Installation Guide](INSTALLATION_GUIDE.md)
- **Advanced Config**: See repository `.env_sample`
- **Analysis**: Enable dev API for Jupyter notebooks
- **Support**: [GitHub Issues](https://github.com/helgeerbe/ml_heating/issues)

---

**⚡ Quick Tip**: Start in shadow mode for safe learning, then switch to active when confidence > 0.9!
