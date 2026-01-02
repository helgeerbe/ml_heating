#!/usr/bin/env python3
"""
Emergency fix for corrupted thermal parameters.

This script restores the thermal_state.json file with reasonable default parameters
and resets the learning system to start fresh.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_current_state(thermal_state_path):
    """Create a backup of the current corrupted state."""
    backup_path = thermal_state_path.parent / f"thermal_state_corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(thermal_state_path, 'r') as f:
            corrupted_data = json.load(f)
        
        with open(backup_path, 'w') as f:
            json.dump(corrupted_data, f, indent=2)
        
        logger.info(f"Backed up corrupted state to {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to backup corrupted state: {e}")
        return False

def create_default_thermal_state():
    """Create a thermal state with reasonable default parameters."""
    now = datetime.now(timezone.utc).isoformat()
    
    thermal_state = {
        "metadata": {
            "version": "1.0",
            "format": "unified_thermal_state",
            "created": now,
            "last_updated": now
        },
        "baseline_parameters": {
            "thermal_time_constant": 4.0,  # 4 hour thermal time constant
            "equilibrium_ratio": 0.8,      # Restored to reasonable value (was 0.1)
            "total_conductance": 0.05,     # Restored to reasonable value (was 0.266)
            "heat_loss_coefficient": 0.1,  # Default heat loss
            "outlet_effectiveness": 0.8,   # Default effectiveness
            "pv_heat_weight": 0.0005,      # Small PV contribution
            "fireplace_heat_weight": 5.0,  # Significant fireplace heat
            "tv_heat_weight": 0.3,         # Small TV heat
            "source": "emergency_restore",
            "calibration_date": now,
            "calibration_cycles": 0
        },
        "learning_state": {
            "cycle_count": 0,               # Reset learning cycle
            "learning_confidence": 5.0,    # Reset confidence to default
            "learning_enabled": True,
            "parameter_adjustments": {
