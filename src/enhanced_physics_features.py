"""
Enhanced Physics Features Integration for Week 2 Multi-Heat-Source Integration

This module provides seamless integration between existing physics_features.py
and the new multi-heat-source physics capabilities. It maintains backward 
compatibility while adding sophisticated heat contribution analysis.

Key Features:
- Drop-in replacement for build_physics_features()
- Maintains all 34 existing thermal momentum features
- Adds 15+ new multi-heat-source features
- Intelligent heat source coordination
- Physics-based outlet temperature optimization

Total: 50+ comprehensive thermal intelligence features for ±0.1°C control precision.
"""

import logging
import pandas as pd
from typing import Optional, Tuple

# Import existing physics_features functionality
from .physics_features import build_physics_features as _build_base_physics_features

# Import new multi-heat-source capabilities
from .multi_heat_source_physics import (
    MultiHeatSourcePhysics, 
    enhance_physics_features_with_heat_sources
)

# Support both package-relative and direct import
try:
    from . import config
    from .ha_client import HAClient
    from .influx_service import InfluxService
except ImportError:
    import config
    from ha_client import HAClient
    from influx_service import InfluxService


class EnhancedPhysicsFeatureBuilder:
    """
    Enhanced feature builder that combines existing thermal momentum features
    with new multi-heat-source integration capabilities.
    """
    
    def __init__(self):
        self.multi_source_physics = MultiHeatSourcePhysics()
        self.feature_cache = {}
        self.last_build_time = None
        
    def build_enhanced_physics_features(
        self,
        ha_client: HAClient,
        influx_service: InfluxService,
        enable_multi_source: bool = True
    ) -> Tuple[Optional[pd.DataFrame], list[float]]:
        """
        Build enhanced physics features with multi-heat-source integration.
        
        This is a drop-in replacement for build_physics_features() that maintains
        all existing functionality while adding sophisticated heat source analysis.
        
        Args:
            ha_client: Home Assistant client
            influx_service: InfluxDB service
            enable_multi_source: Enable multi-heat-source analysis (default: True)
            
        Returns:
            Tuple of (enhanced_features_df, outlet_history)
            - enhanced_features_df: DataFrame with 50+ thermal intelligence features
            - outlet_history: Outlet temperature history for compatibility
        """
        # Step 1: Get base physics features (all 34 existing features)
        base_features_df, outlet_history = _build_base_physics_features(
            ha_client, influx_service
        )
        
        if base_features_df is None:
            logging.error("Failed to build base physics features")
            return None, outlet_history
        
        # Step 2: Extract feature dict for enhancement
        base_features = base_features_df.iloc[0].to_dict()
        
        if not enable_multi_source:
            # Return base features if multi-source analysis disabled
            logging.debug("Multi-source analysis disabled, returning base features")
            return base_features_df, outlet_history
        
        # Step 3: Enhance with multi-heat-source analysis
        try:
            enhanced_features = enhance_physics_features_with_heat_sources(
                base_features, self.multi_source_physics
            )
            
            # Add outlet optimization features
            outlet_optimization = self._calculate_outlet_optimization(enhanced_features)
            enhanced_features.update(outlet_optimization)
            
            # Create enhanced DataFrame
            enhanced_df = pd.DataFrame([enhanced_features])
            
            logging.debug(f"Enhanced features built successfully: "
                         f"{len(enhanced_features)} total features "
                         f"({len(base_features)} base + "
                         f"{len(enhanced_features) - len(base_features)} enhanced)")
            
            return enhanced_df, outlet_history
            
        except Exception as e:
            logging.error(f"Multi-source enhancement failed: {e}, "
                         f"falling back to base features")
            return base_features_df, outlet_history
    
    def _calculate_outlet_optimization(self, enhanced_features: dict) -> dict:
        """
        Calculate outlet temperature optimization features.
        
        Args:
            enhanced_features: Enhanced feature dictionary
            
        Returns:
            Dict with outlet optimization features
        """
        try:
            # Extract current outlet temperature
            current_outlet = enhanced_features.get('outlet_temp', 40.0)
            
            # Create mock heat source analysis for optimization calculation
            heat_source_analysis = {
                'total_heat_contribution_kw': enhanced_features.get('total_auxiliary_heat_kw', 0.0),
                'total_outlet_temp_reduction': enhanced_features.get('total_outlet_reduction', 0.0),
                'heat_source_diversity': enhanced_features.get('heat_source_diversity', 0),
                'diversity_factor': enhanced_features.get('heat_source_diversity_factor', 1.0),
                'pv_contribution': {
                    'heat_contribution_kw': enhanced_features.get('pv_heat_contribution_kw', 0.0),
                    'outlet_temp_reduction': enhanced_features.get('pv_outlet_reduction', 0.0)
                },
                'fireplace_contribution': {
                    'heat_contribution_kw': enhanced_features.get('fireplace_heat_contribution_kw', 0.0),
                    'outlet_temp_reduction': enhanced_features.get('fireplace_outlet_reduction', 0.0)
                },
                'electronics_contribution': {
                    'heat_contribution_kw': enhanced_features.get('electronics_heat_contribution_kw', 0.0),
                    'outlet_temp_reduction': enhanced_features.get('electronics_outlet_reduction', 0.0)
                },
                'system_impacts': {
                    'net_outlet_adjustment': enhanced_features.get('system_outlet_adjustment', 0.0)
                }
            }
            
            # Calculate optimized outlet temperature
            optimization_result = self.multi_source_physics.calculate_optimized_outlet_temperature(
                current_outlet, heat_source_analysis
            )
            
            return {
                'optimized_outlet_temp': optimization_result['optimized_outlet_temp'],
                'outlet_optimization_amount': optimization_result['optimization_amount'],
                'outlet_optimization_percentage': optimization_result['optimization_percentage'],
                'multi_source_outlet_reasoning': optimization_result['optimization_reasoning'][:200]  # Truncate for ML
            }
            
        except Exception as e:
            logging.warning(f"Outlet optimization calculation failed: {e}")
            return {
                'optimized_outlet_temp': enhanced_features.get('outlet_temp', 40.0),
                'outlet_optimization_amount': 0.0,
                'outlet_optimization_percentage': 0.0,
                'multi_source_outlet_reasoning': 'Optimization calculation failed'
            }
    
    def get_feature_summary(self, enhanced_features_df: pd.DataFrame) -> dict:
        """
        Get summary of enhanced features for monitoring and debugging.
        
        Args:
            enhanced_features_df: Enhanced features DataFrame
            
        Returns:
            Dict with feature summary
        """
        if enhanced_features_df is None:
            return {'error': 'No features available'}
        
        features = enhanced_features_df.iloc[0].to_dict()
        
        # Categorize features
        base_thermal_features = [k for k in features.keys() if any(x in k for x in [
            'temp', 'outdoor', 'indoor', 'outlet', 'target', 'gradient', 'delta', 'lag'
        ])]
        
        system_state_features = [k for k in features.keys() if any(x in k for x in [
            'dhw', 'defrost', 'boost', 'disinfection'
        ])]
        
        heat_source_features = [k for k in features.keys() if any(x in k for x in [
            'pv', 'fireplace', 'electronics', 'tv', 'heat_contribution', 'auxiliary'
        ])]
        
        temporal_features = [k for k in features.keys() if any(x in k for x in [
            'hour', 'month', 'sin', 'cos', 'forecast'
        ])]
        
        multi_source_features = [k for k in features.keys() if any(x in k for x in [
            'diversity', 'dominant', 'coordination', 'optimization', 'thermal_balance'
        ])]
        
        return {
            'total_features': len(features),
            'feature_categories': {
                'base_thermal': len(base_thermal_features),
                'system_state': len(system_state_features),
                'heat_source': len(heat_source_features),
                'temporal': len(temporal_features),
                'multi_source': len(multi_source_features)
            },
            'key_values': {
                'total_heat_contribution': features.get('total_auxiliary_heat_kw', 0.0),
                'outlet_optimization': features.get('outlet_optimization_amount', 0.0),
                'heat_source_diversity': features.get('heat_source_diversity', 0),
                'pv_contribution': features.get('pv_heat_contribution_kw', 0.0),
                'fireplace_contribution': features.get('fireplace_heat_contribution_kw', 0.0)
            }
        }


# Global instance for easy access
_enhanced_builder = EnhancedPhysicsFeatureBuilder()


def build_enhanced_physics_features(
    ha_client: HAClient,
    influx_service: InfluxService,
    enable_multi_source: bool = True
) -> Tuple[Optional[pd.DataFrame], list[float]]:
    """
    Enhanced physics feature builder - drop-in replacement for build_physics_features().
    
    This function maintains full backward compatibility while adding sophisticated
    multi-heat-source integration capabilities.
    
    Args:
        ha_client: Home Assistant client
        influx_service: InfluxDB service  
        enable_multi_source: Enable multi-heat-source analysis (default: True)
        
    Returns:
        Tuple of (enhanced_features_df, outlet_history)
    """
    return _enhanced_builder.build_enhanced_physics_features(
        ha_client, influx_service, enable_multi_source
    )


def get_enhanced_feature_summary(enhanced_features_df: pd.DataFrame) -> dict:
    """
    Get summary of enhanced features for monitoring.
    
    Args:
        enhanced_features_df: Enhanced features DataFrame
        
    Returns:
        Dict with feature summary
    """
    return _enhanced_builder.get_feature_summary(enhanced_features_df)


# Backward compatibility alias
build_physics_features = build_enhanced_physics_features


if __name__ == "__main__":
    # Test enhanced physics features integration
    print("🧠 Enhanced Physics Features Integration Test")
    
    # This would normally be called with real HA client and InfluxDB service
    print("Note: This test requires live Home Assistant and InfluxDB connections")
    print("Enhanced features include:")
    print("  - 34 base thermal momentum features")
    print("  - 15+ multi-heat-source features")
    print("  - 4 outlet optimization features")
    print("  - Total: 50+ comprehensive thermal intelligence features")
    
    # Test feature categorization
    mock_features = pd.DataFrame([{
        'outlet_temp': 45.0,
        'indoor_temp_lag_30m': 21.0,
        'pv_heat_contribution_kw': 0.5,
        'fireplace_heat_contribution_kw': 2.0,
        'total_auxiliary_heat_kw': 2.5,
        'heat_source_diversity': 3,
        'optimized_outlet_temp': 42.0,
        'outlet_optimization_amount': -3.0
    }])
    
    summary = get_enhanced_feature_summary(mock_features)
    print(f"\n📊 Feature Summary Test:")
    print(f"Total features: {summary['total_features']}")
    print(f"Categories: {summary['feature_categories']}")
    print(f"Key values: {summary['key_values']}")
    
    print("\n✅ Enhanced Physics Features Integration test complete!")
