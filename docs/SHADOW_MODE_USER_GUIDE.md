# Shadow Mode User Guide

## Quick Start

Shadow mode enables ML heating to learn your building's physics without disrupting your current heating system. The heat curve continues controlling your heating while ML learns in the background and provides efficiency insights.

### Enable Shadow Mode

1. **Set configuration**:
   ```yaml
   # In config.yaml
   SHADOW_MODE: true
   ```

2. **Restart ML Heating service**:
   ```bash
   docker restart ml_heating
   ```

3. **Monitor learning progress** via dashboard or Home Assistant sensors

## What Shadow Mode Does

### For Your Heating System
- ✅ **Heat curve continues operating normally** - No disruption to heating
- ✅ **Maintains target temperatures** - Comfort remains unchanged
- ✅ **Uses existing heat curve settings** - No configuration changes needed

### For ML Learning
- 🎯 **Learns pure physics** - Outlet temperature → Indoor temperature relationships
- 📊 **Benchmarks efficiency** - Compares ML vs heat curve performance
- 📈 **Builds confidence** - Gradually improves prediction accuracy
- 🔍 **Provides insights** - Shows potential energy savings

## Startup Scenarios

### Scenario 1: First Time Setup (Fresh Start)

**What happens:**
1. System loads default physics parameters
2. Starts learning immediately with `learning_confidence = 3.0`
3. Begins collecting heat curve → indoor temperature data
4. Starts benchmarking ML efficiency vs heat curve

**Timeline:**
- **Day 1**: Initial learning begins, basic benchmarks available
- **Week 1**: Physics parameters start converging to your building
- **Month 1**: Accurate efficiency comparison and transition readiness

### Scenario 2: Existing System (With thermal_state.json)

**What happens:**
1. Loads your existing calibrated physics parameters
2. Continues refining parameters with shadow mode learning  
3. Enhanced ML model provides better efficiency benchmarks
4. Seamless integration with existing calibration

**Timeline:**
- **Day 1**: Enhanced learning with existing baseline
- **Week 1**: Improved parameter accuracy and efficiency insights
- **Month 1**: Optimized model ready for high-confidence active mode

## Monitoring Your System

### Dashboard Overview

Access the Streamlit dashboard to monitor:

1. **Learning Progress**
   - Confidence level progression
   - Parameter convergence trends
   - Physics error reduction

2. **Efficiency Benchmarks**
   - ML vs Heat Curve outlet temperature comparison
   - Energy savings potential
   - Target achievement accuracy

3. **Smart Recommendations**
   - When to switch to active mode
   - Configuration optimization suggestions
   - Performance improvement insights

### Home Assistant Sensors

Monitor key metrics via Home Assistant:

```yaml
# Available sensors
sensor.ml_heating_learning_confidence    # Current learning confidence (0-10)
sensor.ml_heating_efficiency_advantage   # ML vs heat curve efficiency (°C)
sensor.ml_heating_energy_savings_pct     # Potential energy savings (%)
sensor.ml_heating_learning_cycles        # Number of learning cycles completed
```

### Log Monitoring

Check logs for detailed insights:

```bash
# View shadow mode benchmarking logs
docker logs ml_heating | grep "Shadow Benchmark"

# Example output:
# Shadow Benchmark: ML would predict 35.2°C, Heat Curve set 38.1°C for target 21.0°C (difference: +2.9°C)
```

## Understanding the Metrics

### Learning Confidence (0-10 scale)
- **0-3**: Initial learning phase
- **3-5**: Basic competency achieved  
- **5-7**: Good accuracy, consider active mode
- **7-10**: High confidence, ready for active mode

### Efficiency Advantage (Temperature difference)
- **Positive (+2°C)**: ML more efficient, predicts lower outlet temps
- **Near Zero (±1°C)**: Similar performance between systems
- **Negative (-2°C)**: Heat curve more efficient than ML predictions

### Energy Savings Percentage
- **0-5%**: Minimal savings potential
- **5-15%**: Moderate savings, worth considering
- **15%+**: High savings potential, strong active mode candidate

## Decision Making Guide

### When to Switch to Active Mode

**Ready for Active Mode** ✅
- Learning confidence > 7.0
- Efficiency advantage > 2°C consistently
- Stable parameter convergence for 2+ weeks
- Energy savings potential > 10%

**Continue Shadow Mode** ⚠️
- Learning confidence < 5.0
- Efficiency advantage < 1°C
- Parameter drift or instability
- Insufficient learning cycles (< 100)

**Review Configuration** ❌
- Efficiency advantage < -2°C (heat curve better)
- Learning stagnation (no confidence growth)
- Sensor calibration issues
- Unusual weather affecting learning

### Switching to Active Mode

When ready, disable shadow mode:

1. **Update configuration**:
   ```yaml
   # In config.yaml
   SHADOW_MODE: false
   ```

2. **Restart service**:
   ```bash
   docker restart ml_heating
   ```

3. **Monitor transition**:
   - Check initial active mode performance
   - Verify target temperature achievement
   - Monitor energy consumption changes

## Troubleshooting

### Common Issues

#### Slow Learning Progress
**Symptoms:** Confidence not increasing, parameters unstable
**Solutions:**
- Check sensor calibration (indoor, outdoor, outlet temperature)
- Verify heat curve stability 
- Ensure sufficient heating cycles (winter operation)
- Consider increasing learning rate in configuration

#### Poor Efficiency Benchmarks
**Symptoms:** ML predictions much higher than heat curve
**Solutions:**
- Verify ML model has sufficient training data
- Check outdoor temperature sensor accuracy
- Review heat curve settings for optimization opportunities
- Wait for more learning cycles in varying weather

#### Learning Stagnation
**Symptoms:** Confidence stuck, no parameter improvement
**Solutions:**
- Reset learning parameters (delete thermal_state.json)
- Check for sensor drift or calibration issues
- Verify heating system operating normally
- Consider manual parameter adjustment

#### Target Temperature Issues
**Symptoms:** Heat curve not maintaining target temperatures
**Solutions:**
- Review heat curve configuration
- Check indoor temperature sensor placement
- Verify heating system capacity and performance
- Consider heat curve optimization before ML transition

### Diagnostic Commands

```bash
# Check system status
docker logs ml_heating --tail 50

# Verify configuration
docker exec ml_heating cat /app/config.yaml

# Check learning data
docker exec ml_heating ls -la /data/models/

# View thermal state
docker exec ml_heating cat /data/models/thermal_state.json
```

## Advanced Features

### Multi-Heat Source Learning

Shadow mode works with complex heating systems:
- **Heat pump + fireplace combinations**
- **Solar thermal integration**
- **Multi-zone heating systems**
- **Adaptive learning for fireplace usage**

### Seasonal Adaptation

The system adapts to seasonal changes:
- **Winter**: Full learning across temperature ranges
- **Spring/Fall**: Transition period optimization
- **Summer**: Minimal heating, parameter preservation

### Weather Integration

Enhanced learning with weather data:
- **Temperature trend analysis**
- **Wind and humidity consideration** 
- **Solar irradiance integration**
- **Weather forecast preparation**

## Best Practices

### Setup Recommendations

1. **Enable during heating season** - Maximum learning opportunities
2. **Ensure stable heat curve** - Baseline performance consistency
3. **Monitor initial weeks closely** - Verify learning progress
4. **Use dashboard regularly** - Track efficiency insights

### Optimization Tips

1. **Review benchmarks weekly** - Identify efficiency trends
2. **Check weather correlation** - Understand performance patterns
3. **Monitor sensor health** - Maintain calibration accuracy
4. **Plan mode transitions** - Choose optimal switching timing

### Seasonal Planning

- **Fall Setup**: Enable before heating season for maximum learning
- **Winter Monitoring**: Regular efficiency analysis and optimization
- **Spring Transition**: Consider active mode switch before season end
- **Summer Preparation**: Parameter preservation and system maintenance

## FAQ

### Q: Will shadow mode affect my heating comfort?
**A:** No, shadow mode only learns in background. Your heat curve continues controlling heating normally.

### Q: How long should I run shadow mode?
**A:** Typically 2-4 weeks for basic competency, 1-2 months for high confidence. Depends on weather variety and heating cycles.

### Q: Can I switch back to shadow mode from active mode?
**A:** Yes, simply set `SHADOW_MODE: true` again. Previous learning is preserved.

### Q: What if ML shows worse efficiency than heat curve?
**A:** This indicates your heat curve is already well-optimized or ML needs more learning time. Continue shadow mode or optimize heat curve.

### Q: How much energy can I save?
**A:** Typical savings range from 5-20% depending on current heat curve efficiency and building characteristics.

### Q: Is shadow mode safe for my heating system?
**A:** Completely safe - shadow mode only observes and learns, never controls heating equipment.

## Support

For additional help:
- 📖 Check technical documentation in `/docs/`
- 🐛 Report issues via GitHub issues
- 💬 Community support in project discussions
- 📊 Use dashboard analytics for detailed insights

Shadow mode provides a safe, effective path to ML heating optimization with complete transparency and control over the transition process.
