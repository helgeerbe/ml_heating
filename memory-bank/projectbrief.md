# ML Heating Control System - Project Brief

## Project Overview

The **ml_heating** project is a sophisticated **physics-based machine learning heating control system** that integrates with Home Assistant to optimize heat pump operation. It uses a `RealisticPhysicsModel` to predict optimal water outlet temperatures for heat pumps, continuously learning from real-world results to efficiently maintain target indoor temperatures.

## Core Problem Statement

Traditional heating curves are static - they map outdoor temperature to a fixed outlet temperature. This results in:
- **Energy waste** from imprecise heating output matching
- **Comfort issues** with temperature overshoot/undershoot
- **Seasonal inefficiency** requiring manual recalibration
- **No adaptation** to house changes, weather patterns, or occupancy

## Solution Approach

### Physics-Based Machine Learning
- Combines thermodynamic principles with data-driven learning
- Uses `RealisticPhysicsModel` that understands heat transfer while learning house-specific characteristics
- Continuously adapts based on actual measured outcomes vs predictions

### Key Innovation: Online Learning
- Learns from every heating cycle (typically 30-minute intervals)
- Adapts to seasonal changes, house modifications, and usage patterns
- No manual recalibration required between seasons

## Goals & Success Criteria

### Primary Goals
1. **Increase Efficiency**: Minimize energy waste by precisely matching heating output to actual need
2. **Improve Comfort**: Maintain stable indoor temperature with less overshoot/undershoot
3. **Continuous Adaptation**: Learn from every cycle, adjusting to seasons, weather, house changes
4. **Anticipatory Control**: Use weather and PV forecasts for proactive adjustments
5. **Safe Operation**: Comprehensive blocking logic, gradual changes, health monitoring

### Success Metrics
- **MAE (Mean Absolute Error)**: < 0.2°C for good performance, < 0.3°C acceptable
- **RMSE (Root Mean Square Error)**: < 0.3°C for good performance, < 0.4°C acceptable  
- **Confidence**: > 0.9 for optimal operation, > 0.7 acceptable
- **Energy Efficiency**: Measurable reduction in heat pump energy consumption
- **Temperature Stability**: Reduced variance in indoor temperature

## Scope & Boundaries

### In Scope
- Heat pump outlet temperature optimization for space heating
- Integration with Home Assistant ecosystem
- InfluxDB historical data utilization
- Multiple external heat source learning (PV, fireplace, TV/electronics)
- Weather and PV forecast integration
- Comprehensive safety and blocking detection
- Production deployment with systemd service

### Out of Scope
- DHW (Domestic Hot Water) temperature control
- Direct heat pump communication (works through Home Assistant)
- HVAC system installation or configuration
- Alternative heating system types (gas, oil, electric resistance)

## Technical Approach

### Architecture Pattern
- **Physics-based modeling** rather than pure black-box ML
- **Online learning** with continuous adaptation
- **Multi-modal operation** (active control vs shadow observation)
- **Safety-first design** with multiple protection layers

### Key Technical Decisions
- Uses River framework for online machine learning
- Implements 7-stage prediction pipeline for safety and optimization
- Employs multi-lag learning for time-delayed thermal effects
- Features automatic seasonal adaptation via cos/sin modulation
- Supports both active and shadow modes for safe testing

## Deployment Context

### Production Environment
- Runs as systemd service on Linux systems
- Integrates with existing Home Assistant + InfluxDB infrastructure
- Designed for 24/7 operation with automatic restart on failure
- Comprehensive logging and monitoring capabilities

### User Profile
- Home automation enthusiasts with Home Assistant setups
- Technical users comfortable with configuration and monitoring
- Users with heat pump heating systems and temperature sensors
- Optional: Users with solar PV systems for enhanced learning

## Risk Mitigation

### Safety Mechanisms
- Absolute temperature clamping (configurable min/max limits)
- Gradual temperature changes (max change per cycle limits)
- Blocking detection for DHW, defrost, disinfection cycles
- Grace periods after blocking events for system stabilization
- Comprehensive error handling and fallback behaviors

### Operational Safety
- Shadow mode for safe testing before active deployment
- Continuous monitoring via Home Assistant sensors
- Detailed logging for troubleshooting and analysis
- Model validation tools for physics compliance checking

## Success Dependencies

### Technical Requirements
- Home Assistant with REST API access
- InfluxDB with historical sensor data
- Python 3.8+ environment
- Heat pump with controllable outlet temperature
- Indoor temperature sensor(s)
- Outdoor temperature sensor

### Data Requirements
- Minimum 7 days of historical heating data for calibration
- Continuous sensor data during operation
- Optional: Solar PV production data for enhanced learning
- Optional: Weather forecast integration

This project represents a sophisticated approach to residential heating optimization, combining advanced machine learning with practical engineering constraints and comprehensive safety mechanisms.
