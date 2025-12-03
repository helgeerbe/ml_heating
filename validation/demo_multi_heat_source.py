#!/usr/bin/env python3
"""
Multi-Heat-Source Physics Integration Demonstration
Week 2 Implementation - Binary-to-Physics Transformation
"""

from src.multi_heat_source_physics import MultiHeatSourcePhysics

def main():
    # Initialize the multi-heat-source physics engine
    physics = MultiHeatSourcePhysics()
    
    print("🔥 Multi-Heat-Source Physics Integration Demonstration")
    print("=" * 60)
    
    # Test scenario: High PV + Fireplace + TV on cold day
    test_scenario = {
        'pv_power': 2500,  # 2.5kW solar
        'fireplace_on': True,
        'tv_on': True,
        'indoor_temp': 21.0,
        'outdoor_temp': 5.0,
        'dhw_heating': False,
        'defrosting': False
    }
    
    print(f"Test scenario: PV {test_scenario['pv_power']}W, Fireplace: {test_scenario['fireplace_on']}, TV: {test_scenario['tv_on']}")
    print(f"Indoor: {test_scenario['indoor_temp']}°C, Outdoor: {test_scenario['outdoor_temp']}°C")
    
    # Calculate combined heat sources
    analysis = physics.calculate_combined_heat_sources(**test_scenario)
    
    print(f"\n🎯 Multi-Source Analysis:")
    print(f"Total heat contribution: {analysis['total_heat_contribution_kw']:.2f}kW")
    print(f"Total outlet reduction: {analysis['total_outlet_temp_reduction']:.1f}°C")
    print(f"Active heat sources: {analysis['heat_source_diversity']}")
    print(f"Diversity factor: {analysis['diversity_factor']:.1f}x")
    print(f"Reasoning: {analysis['multi_source_reasoning']}")
    
    # Test outlet optimization
    base_outlet = 45.0
    optimization = physics.calculate_optimized_outlet_temperature(base_outlet, analysis)
    
    print(f"\n🚀 Outlet Optimization:")
    print(f"Base outlet: {base_outlet:.1f}°C")
    print(f"Optimized outlet: {optimization['optimized_outlet_temp']:.1f}°C")
    print(f"Optimization: {optimization['optimization_amount']:+.1f}°C ({optimization['optimization_percentage']:+.0f}%)")
    print(f"Reasoning: {optimization['optimization_reasoning']}")
    
    # Show individual heat source breakdown
    print(f"\n🔍 Individual Heat Source Analysis:")
    
    # PV heat contribution
    pv_analysis = physics.calculate_pv_heat_contribution(
        test_scenario['pv_power'],
        test_scenario['indoor_temp'],
        test_scenario['outdoor_temp']
    )
    print(f"PV Solar: {pv_analysis['heat_contribution_kw']:.2f}kW ({pv_analysis.get('hour_effectiveness', 0.8):.1f}x time factor)")
    
    # Fireplace heat contribution
    fireplace_analysis = physics.calculate_fireplace_heat_contribution(
        test_scenario['fireplace_on'], 
        outdoor_temp=test_scenario['outdoor_temp']
    )
    print(f"Fireplace: {fireplace_analysis['heat_contribution_kw']:.2f}kW ({fireplace_analysis['thermal_buildup_factor']:.1f}x buildup)")
    
    # Electronics/occupancy heat
    electronics_analysis = physics.calculate_electronics_occupancy_heat(test_scenario['tv_on'])
    print(f"Electronics + Occupancy: {electronics_analysis['heat_contribution_kw']:.2f}kW ({electronics_analysis['estimated_occupancy']} people)")
    
    # Enhanced features demonstration using integration function
    print(f"\n🧠 Enhanced Physics Features (for ML model):")
    from src.multi_heat_source_physics import enhance_physics_features_with_heat_sources
    
    # Create base features dict
    base_features = {
        'indoor_temp_lag_30m': test_scenario['indoor_temp'],
        'outdoor_temp': test_scenario['outdoor_temp'],
        'pv_now': test_scenario['pv_power'],
        'fireplace_on': 1 if test_scenario['fireplace_on'] else 0,
        'tv_on': 1 if test_scenario['tv_on'] else 0,
        'dhw_heating': 1 if test_scenario['dhw_heating'] else 0,
        'defrosting': 1 if test_scenario['defrosting'] else 0
    }
    
    # Enhance with multi-heat-source physics
    enhanced_features = enhance_physics_features_with_heat_sources(base_features, physics)
    
    # Show key enhanced features
    key_features = [
        'total_auxiliary_heat_kw',
        'system_capacity_reduction_percent',
        'fireplace_heat_contribution_kw',
        'electronics_heat_contribution_kw',
        'pv_heat_contribution_kw',
        'heat_source_diversity',
        'thermal_balance_score',
        'total_outlet_reduction'
    ]
    
    for feature in key_features:
        if feature in enhanced_features:
            print(f"  {feature}: {enhanced_features[feature]:.3f}")
    
    print(f"\n✅ Multi-Heat-Source Physics Integration operational!")
    print(f"Features generated: {len(enhanced_features)} total thermal intelligence features")
    print(f"Binary sensors transformed into physics-based coefficients!")

if __name__ == "__main__":
    main()
