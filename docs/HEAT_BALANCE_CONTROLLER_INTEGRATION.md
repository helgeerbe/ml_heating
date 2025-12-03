# Heat Balance Controller Integration Guide

## Multi-Heat-Source Physics Integration for Production Deployment

This guide demonstrates how to integrate the Week 2 Multi-Heat-Source Physics engine with your existing Heat Balance Controller for seamless production deployment.

## Overview

The multi-heat-source integration enhances your existing Heat Balance Controller by:

- **Replacing binary heat source flags** with sophisticated thermal intelligence
- **Providing 25+ enhanced physics features** for ML model learning
- **Optimizing outlet temperatures** through multi-source coordination
- **Maintaining backward compatibility** with existing architecture

## Integration Patterns

### 1. Drop-in Enhancement (Recommended)

Enhance your existing Heat Balance Controller without breaking changes:

```python
# Enhanced Heat Balance Controller with Multi-Heat-Source Integration
from src.multi_heat_source_physics import MultiHeatSourcePhysics
from src.multi_heat_source_physics import enhance_physics_features_with_heat_sources

class EnhancedHeatBalanceController:
    def __init__(self):
        # Initialize multi-heat-source physics engine
        self.multi_source_physics = MultiHeatSourcePhysics()
        
        # Keep existing Heat Balance Controller as fallback
        self.original_controller = HeatBalanceController()
        
        # Configuration
        self.multi_source_enabled = True
        self.fallback_on_error = True
        
    def determine_optimal_outlet_temperature(self, features_df, target_temp):
        """Enhanced outlet temperature determination with multi-source optimization"""
        
        try:
            if not self.multi_source_enabled:
                # Use original controller if multi-source disabled
                return self.original_controller.determine_optimal_outlet_temperature(
                    features_df, target_temp
                )
            
            # Extract current features
            features = features_df.iloc[0].to_dict()
            
            # Step 1: Calculate base physics prediction (original thermal model)
            base_outlet = self.original_controller.calculate_base_outlet_temperature(
                features['indoor_temp_lag_30m'],
                target_temp,
                features['outdoor_temp'],
                **features
            )
            
            # Step 2: Calculate multi-heat-source analysis
            heat_source_analysis = self.multi_source_physics.calculate_combined_heat_sources(
                pv_power=features.get('pv_now', 0),
                fireplace_on=bool(features.get('fireplace_on', 0)),
                tv_on=bool(features.get('tv_on', 0)),
                indoor_temp=features['indoor_temp_lag_30m'],
                outdoor_temp=features['outdoor_temp'],
                dhw_heating=bool(features.get('dhw_heating', 0)),
                dhw_disinfection=bool(features.get('dhw_disinfection', 0)),
                dhw_boost_heater=bool(features.get('dhw_boost_heater', 0)),
                defrosting=bool(features.get('defrosting', 0))
            )
            
            # Step 3: Optimize outlet temperature with multi-source intelligence
            optimization_result = self.multi_source_physics.calculate_optimized_outlet_temperature(
                base_outlet, heat_source_analysis
            )
            
            # Step 4: Return comprehensive result
            return {
                'recommended_outlet_temp': optimization_result['optimized_outlet_temp'],
                'base_outlet_temp': base_outlet,
                'multi_source_optimization': optimization_result,
                'heat_source_analysis': heat_source_analysis,
                'reasoning': optimization_result['optimization_reasoning'],
                'energy_savings_percent': optimization_result['optimization_percentage']
            }
            
        except Exception as e:
            if self.fallback_on_error:
                # Fallback to original controller on any error
                print(f"Multi-source error, falling back to original: {e}")
                return self.original_controller.determine_optimal_outlet_temperature(
                    features_df, target_temp
                )
            else:
                raise
    
    def get_enhanced_features(self, existing_features):
        """Get enhanced features with multi-heat-source analysis"""
        return enhance_physics_features_with_heat_sources(
            existing_features, self.multi_source_physics
        )
    
    def validate_multi_source_performance(self, historical_data):
        """Validate multi-source performance against historical data"""
        validation_results = []
        
        for _, row in historical_data.iterrows():
            # Original prediction
            original_result = self.original_controller.determine_optimal_outlet_temperature(
                pd.DataFrame([row]), row['target_temp']
            )
            
            # Enhanced prediction
            enhanced_result = self.determine_optimal_outlet_temperature(
                pd.DataFrame([row]), row['target_temp']
            )
            
            validation_results.append({
                'timestamp': row['timestamp'],
                'original_outlet': original_result['recommended_outlet_temp'],
                'enhanced_outlet': enhanced_result['recommended_outlet_temp'],
                'optimization_improvement': enhanced_result['energy_savings_percent'],
                'heat_contribution_kw': enhanced_result['heat_source_analysis']['total_heat_contribution_kw']
            })
        
        return pd.DataFrame(validation_results)
```

### 2. Feature Enhancement Integration

Enhance your ML model training with multi-heat-source features:

```python
# ML Model Training Enhancement
from src.multi_heat_source_physics import MultiHeatSourcePhysics
from src.multi_heat_source_physics import enhance_physics_features_with_heat_sources

class EnhancedMLTraining:
    def __init__(self):
        self.multi_source_physics = MultiHeatSourcePhysics()
        
    def enhance_training_data(self, training_df):
        """Enhance training data with multi-heat-source physics features"""
        enhanced_data = []
        
        for _, row in training_df.iterrows():
            # Convert row to base features dict
            base_features = row.to_dict()
            
            # Enhance with multi-heat-source physics
            enhanced_features = enhance_physics_features_with_heat_sources(
                base_features, self.multi_source_physics
            )
            
            enhanced_data.append(enhanced_features)
        
        # Return enhanced DataFrame
        enhanced_df = pd.DataFrame(enhanced_data)
        
        print(f"Feature enhancement complete:")
        print(f"Original features: {len(training_df.columns)}")
        print(f"Enhanced features: {len(enhanced_df.columns)}")
        print(f"Feature increase: {len(enhanced_df.columns) - len(training_df.columns)}x")
        
        return enhanced_df
    
    def compare_model_performance(self, X_original, X_enhanced, y, test_size=0.2):
        """Compare ML model performance: original vs enhanced features"""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        
        # Split data
        X_orig_train, X_orig_test, y_train, y_test = train_test_split(
            X_original, y, test_size=test_size, random_state=42
        )
        X_enh_train, X_enh_test, _, _ = train_test_split(
            X_enhanced, y, test_size=test_size, random_state=42
        )
        
        # Train original model
        model_original = RandomForestRegressor(n_estimators=100, random_state=42)
        model_original.fit(X_orig_train, y_train)
        y_pred_orig = model_original.predict(X_orig_test)
        
        # Train enhanced model
        model_enhanced = RandomForestRegressor(n_estimators=100, random_state=42)
        model_enhanced.fit(X_enh_train, y_train)
        y_pred_enh = model_enhanced.predict(X_enh_test)
        
        # Compare performance
        results = {
            'original_mae': mean_absolute_error(y_test, y_pred_orig),
            'enhanced_mae': mean_absolute_error(y_test, y_pred_enh),
            'original_r2': r2_score(y_test, y_pred_orig),
            'enhanced_r2': r2_score(y_test, y_pred_enh),
            'mae_improvement': ((mean_absolute_error(y_test, y_pred_orig) - 
                               mean_absolute_error(y_test, y_pred_enh)) / 
                               mean_absolute_error(y_test, y_pred_orig) * 100),
            'r2_improvement': ((r2_score(y_test, y_pred_enh) - 
                              r2_score(y_test, y_pred_orig)) / 
                              r2_score(y_test, y_pred_orig) * 100)
        }
        
        return results
```

### 3. Real-time Monitoring Integration

Monitor multi-heat-source performance in real-time:

```python
# Real-time Multi-Heat-Source Monitoring
class MultiHeatSourceMonitor:
    def __init__(self):
        self.multi_source_physics = MultiHeatSourcePhysics()
        self.performance_log = []
        
    def monitor_heat_source_coordination(self, current_features):
        """Monitor real-time heat source coordination effectiveness"""
        
        # Calculate current multi-source analysis
        analysis = self.multi_source_physics.calculate_combined_heat_sources(
            pv_power=current_features.get('pv_now', 0),
            fireplace_on=bool(current_features.get('fireplace_on', 0)),
            tv_on=bool(current_features.get('tv_on', 0)),
            indoor_temp=current_features['indoor_temp'],
            outdoor_temp=current_features['outdoor_temp'],
            dhw_heating=bool(current_features.get('dhw_heating', 0)),
            defrosting=bool(current_features.get('defrosting', 0))
        )
        
        # Calculate outlet optimization
        base_outlet = 45.0  # Your system's typical base outlet
        optimization = self.multi_source_physics.calculate_optimized_outlet_temperature(
            base_outlet, analysis
        )
        
        # Log performance
        performance_entry = {
            'timestamp': datetime.now(),
            'total_heat_kw': analysis['total_heat_contribution_kw'],
            'outlet_reduction': analysis['total_outlet_temp_reduction'],
            'active_sources': analysis['heat_source_diversity'],
            'energy_savings_percent': optimization['optimization_percentage'],
            'coordination_opportunities': len(analysis['coordination_analysis']['coordination_opportunities'])
        }
        
        self.performance_log.append(performance_entry)
        
        # Alert on significant opportunities
        if analysis['total_heat_contribution_kw'] > 2.0:
            print(f"🔥 High heat coordination opportunity: {analysis['total_heat_contribution_kw']:.1f}kW available")
        
        if optimization['optimization_percentage'] > 10:
            print(f"⚡ Significant energy savings available: {optimization['optimization_percentage']:.1f}%")
        
        return {
            'current_analysis': analysis,
            'optimization': optimization,
            'recommendations': self._generate_recommendations(analysis)
        }
    
    def _generate_recommendations(self, analysis):
        """Generate operational recommendations based on heat source analysis"""
        recommendations = []
        
        # High PV + other sources
        if (analysis['pv_contribution']['heat_contribution_kw'] > 1.0 and 
            analysis['heat_source_diversity'] > 1):
            recommendations.append("Consider reducing heat pump demand - high PV + auxiliary heating active")
        
        # Fireplace overheating risk
        if analysis['fireplace_contribution']['heat_contribution_kw'] > 2.0:
            recommendations.append("Monitor indoor temperature - significant fireplace heating detected")
        
        # System conflicts
        if (analysis['system_impacts']['capacity_reduction_percent'] > 20 and 
            analysis['total_heat_contribution_kw'] > 1.5):
            recommendations.append("System capacity reduced during high auxiliary heating - optimize DHW timing")
        
        return recommendations
    
    def get_performance_summary(self, hours=24):
        """Get performance summary for last N hours"""
        if not self.performance_log:
            return "No performance data available"
        
        recent_data = [entry for entry in self.performance_log 
                      if (datetime.now() - entry['timestamp']).total_seconds() < hours * 3600]
        
        if not recent_data:
            return f"No data in last {hours} hours"
        
        avg_heat = sum(entry['total_heat_kw'] for entry in recent_data) / len(recent_data)
        avg_savings = sum(entry['energy_savings_percent'] for entry in recent_data) / len(recent_data)
        max_sources = max(entry['active_sources'] for entry in recent_data)
        
        return {
            'period_hours': hours,
            'data_points': len(recent_data),
            'avg_heat_contribution_kw': avg_heat,
            'avg_energy_savings_percent': avg_savings,
            'max_concurrent_sources': max_sources,
            'coordination_events': sum(1 for entry in recent_data if entry['coordination_opportunities'] > 0)
        }
```

## Configuration Examples

### 1. Production Configuration

```yaml
# config/multi_heat_source.yaml
multi_heat_source:
  enabled: true
  fallback_on_error: true
  
  # Building-specific thermal characteristics
  building:
    thermal_mass: 3500.0  # kWh/°C
    heat_pump_cop: 3.5
    zone_heat_distribution: 0.7
  
  # Heat source coefficients (calibrate to your installation)
  heat_sources:
    pv:
      building_heating_factor: 0.25  # 25% of PV becomes building heat
      thermal_efficiency_base: 0.8
    
    fireplace:
      heat_output_kw: 8.0
      thermal_efficiency: 0.75
      heat_distribution_factor: 0.6
    
    electronics:
      tv_base_heat: 250  # Watts
      occupancy_activity_multiplier: 1.5
  
  # Monitoring and alerts
  monitoring:
    performance_logging: true
    alert_thresholds:
      high_heat_coordination: 2.0  # kW
      significant_energy_savings: 10.0  # %
```

### 2. Testing Configuration

```python
# Test configuration for validation
TEST_CONFIG = {
    'validation_scenarios': [
        {
            'name': 'Winter Evening Peak',
            'conditions': {
                'outdoor_temp': -2.0,
                'pv_power': 0,
                'fireplace_on': True,
                'tv_on': True,
                'dhw_heating': True
            },
            'expected_heat_kw': 2.5,
            'expected_savings_percent': 8.0
        },
        {
            'name': 'Sunny Day High PV',
            'conditions': {
                'outdoor_temp': 12.0,
                'pv_power': 3000,
                'fireplace_on': False,
                'tv_on': False
            },
            'expected_heat_kw': 0.8,
            'expected_savings_percent': 5.0
        }
    ]
}
```

## Deployment Checklist

### Pre-deployment Validation
- [ ] **Test multi-heat-source calculations** with your building's heat source configurations
- [ ] **Validate thermal coefficients** against actual temperature responses
- [ ] **Run integration tests** with existing Heat Balance Controller
- [ ] **Performance benchmark** original vs enhanced controller

### Production Deployment
-
