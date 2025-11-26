#!/usr/bin/env python3
"""
ML Heating Add-on Configuration Adapter

Maps Home Assistant add-on configuration options to environment variables
for compatibility with the existing ML heating system.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import shutil

def log_info(message):
    """Log info message"""
    print(f"[INFO] {message}")

def log_warning(message):
    """Log warning message"""
    print(f"[WARNING] {message}")

def log_error(message):
    """Log error message"""
    print(f"[ERROR] {message}")

def load_addon_config():
    """Load Home Assistant add-on configuration"""
    try:
        with open('/data/options.json', 'r') as f:
            config = json.load(f)
        log_info("Add-on configuration loaded successfully")
        return config
    except FileNotFoundError:
        log_error("/data/options.json not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log_error(f"Invalid JSON in options.json: {e}")
        sys.exit(1)

def setup_data_directories():
    """Create necessary data directories with proper permissions"""
    directories = [
        '/data/models',
        '/data/backups', 
        '/data/logs',
        '/data/config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        log_info(f"Created directory: {directory}")

def convert_addon_to_env(config):
    """Convert add-on options to environment variables for existing ML system"""
    
    # Home Assistant API configuration (internal supervisor access)
    env_vars = {
        'HASS_URL': 'http://supervisor/core',
        'HASS_TOKEN': os.environ.get('SUPERVISOR_TOKEN', ''),
        
        # Core entity mappings
        'TARGET_INDOOR_TEMP_ENTITY_ID': config.get('target_indoor_temp_entity', ''),
        'INDOOR_TEMP_ENTITY_ID': config.get('indoor_temp_entity', ''),
        'OUTDOOR_TEMP_ENTITY_ID': config.get('outdoor_temp_entity', ''),
        'HEATING_CONTROL_ENTITY_ID': config.get('heating_control_entity', ''),
        'OUTLET_TEMP_ENTITY_ID': config.get('outlet_temp_entity', ''),
        
        # ML learning parameters
        'LEARNING_RATE': str(config.get('learning_rate', 0.01)),
        'PREDICTION_HORIZON_MINUTES': str(config.get('prediction_horizon_minutes', 30)),
        'CYCLE_INTERVAL_MINUTES': str(config.get('cycle_interval_minutes', 30)),
        'MAX_TEMP_CHANGE_PER_CYCLE': str(config.get('max_temp_change_per_cycle', 2.0)),
        
        # Safety configuration
        'SAFETY_MAX_TEMP': str(config.get('safety_max_temp', 25.0)),
        'SAFETY_MIN_TEMP': str(config.get('safety_min_temp', 18.0)),
        'CLAMP_MIN_ABS': str(config.get('clamp_min_abs', 14.0)),
        'CLAMP_MAX_ABS': str(config.get('clamp_max_abs', 65.0)),
        
        # External heat sources (optional)
        'PV_POWER_ENTITY_ID': config.get('pv_power_entity', ''),
        'FIREPLACE_STATUS_ENTITY_ID': config.get('fireplace_status_entity', ''),
        'TV_POWER_ENTITY_ID': config.get('tv_power_entity', ''),
        
        # InfluxDB configuration  
        'INFLUXDB_HOST': config.get('influxdb_host', 'a0d7b954-influxdb'),
        'INFLUXDB_PORT': str(config.get('influxdb_port', 8086)),
        'INFLUXDB_DATABASE': config.get('influxdb_database', 'homeassistant'),
        'INFLUXDB_USERNAME': config.get('influxdb_username', ''),
        'INFLUXDB_PASSWORD': config.get('influxdb_password', ''),
        'INFLUXDB_TOKEN': config.get('influxdb_token', ''),
        'INFLUXDB_ORG': config.get('influxdb_org', ''),
        'INFLUXDB_BUCKET': config.get('influxdb_bucket', 'homeassistant'),
        
        # Blocking detection entities
        'DHW_STATUS_ENTITY_ID': config.get('dhw_status_entity', ''),
        'DEFROST_STATUS_ENTITY_ID': config.get('defrost_status_entity', ''),
        'DISINFECTION_STATUS_ENTITY_ID': config.get('disinfection_status_entity', ''),
        'DHW_BOOST_HEATER_ENTITY_ID': config.get('dhw_boost_heater_entity', ''),
        
        # Add-on specific paths (different from standalone)
        'MODEL_FILE_PATH': '/data/models/ml_model.pkl',
        'STATE_FILE_PATH': '/data/models/ml_state.pkl',
        'BACKUP_DIR': '/data/backups',
        'LOG_FILE_PATH': '/data/logs/ml_heating.log',
        
        # Performance tuning
        'SMOOTHING_ALPHA': str(config.get('smoothing_alpha', 0.3)),
        'CONFIDENCE_THRESHOLD': str(config.get('confidence_threshold', 0.7)),
        'PHYSICS_VALIDATION_ENABLED': str(config.get('physics_validation_enabled', True)),
        'SEASONAL_LEARNING_ENABLED': str(config.get('seasonal_learning_enabled', True)),
        
        # Logging and development
        'LOG_LEVEL': config.get('log_level', 'INFO'),
        'ENABLE_DEV_API': str(config.get('enable_dev_api', False)),
        'DEV_API_KEY': config.get('dev_api_key', ''),
        
        # Dashboard settings
        'DASHBOARD_UPDATE_INTERVAL': str(config.get('dashboard_update_interval', 30)),
        'SHOW_ADVANCED_METRICS': str(config.get('show_advanced_metrics', True)),
        'DASHBOARD_THEME': config.get('dashboard_theme', 'auto'),
        
        # Model management
        'AUTO_BACKUP_ENABLED': str(config.get('auto_backup_enabled', True)),
        'BACKUP_RETENTION_DAYS': str(config.get('backup_retention_days', 30)),
    }
    
    # Set environment variables for the ML system
    for key, value in env_vars.items():
        if value is not None and value != '':
            os.environ[key] = str(value)
    
    log_info(f"Set {len([v for v in env_vars.values() if v])} environment variables")
    return env_vars

def import_existing_model(source_path):
    """Import existing model from standalone installation"""
    try:
        if not os.path.exists(source_path):
            log_warning(f"Model file not found: {source_path}")
            return False
            
        # Backup existing model if present
        target_path = '/data/models/ml_model.pkl'
        if os.path.exists(target_path):
            backup_name = f"pre_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            backup_path = f"/data/backups/{backup_name}"
            shutil.copy(target_path, backup_path)
            log_info(f"Backed up existing model to {backup_path}")
        
        # Import new model
        shutil.copy(source_path, target_path)
        log_info(f"Model imported successfully from {source_path}")
        return True
        
    except Exception as e:
        log_error(f"Model import failed: {e}")
        return False

def validate_configuration(config):
    """Validate critical configuration settings"""
    required_entities = [
        ('target_indoor_temp_entity', 'Target Indoor Temperature Entity'),
        ('indoor_temp_entity', 'Indoor Temperature Entity'),
        ('outdoor_temp_entity', 'Outdoor Temperature Entity'), 
        ('heating_control_entity', 'Heating Control Entity'),
        ('outlet_temp_entity', 'Outlet Temperature Entity')
    ]
    
    missing_entities = []
    for key, description in required_entities:
        if not config.get(key) or config.get(key).strip() == '':
            missing_entities.append(description)
    
    if missing_entities:
        log_error("Missing required entity configurations:")
        for entity in missing_entities:
            log_error(f"  - {entity}")
        log_error("Please configure all required entities in the add-on settings")
        return False
    
    # Validate numeric ranges
    learning_rate = config.get('learning_rate', 0.01)
    if not (0.001 <= learning_rate <= 0.1):
        log_warning(f"Learning rate {learning_rate} outside recommended range [0.001, 0.1]")
    
    max_temp = config.get('safety_max_temp', 25.0)
    min_temp = config.get('safety_min_temp', 18.0)
    if max_temp <= min_temp:
        log_error(f"Safety max temp ({max_temp}) must be greater than min temp ({min_temp})")
        return False
    
    log_info("Configuration validation passed")
    return True

def save_config_backup(config):
    """Save configuration backup for troubleshooting"""
    try:
        config_backup = {
            'timestamp': datetime.now().isoformat(),
            'addon_version': '1.0.0',
            'config': config
        }
        
        backup_path = '/data/config/addon_config_backup.json'
        with open(backup_path, 'w') as f:
            json.dump(config_backup, f, indent=2)
        
        log_info(f"Configuration backup saved to {backup_path}")
        
    except Exception as e:
        log_warning(f"Failed to save config backup: {e}")

def initialize_addon_environment():
    """Initialize add-on environment for ML system"""
    try:
        log_info("Initializing ML Heating Add-on environment...")
        
        # Load add-on configuration
        addon_config = load_addon_config()
        
        # Validate configuration
        if not validate_configuration(addon_config):
            log_error("Configuration validation failed")
            return False
        
        # Setup directories
        setup_data_directories()
        
        # Save config backup
        save_config_backup(addon_config)
        
        # Convert to environment variables
        env_vars = convert_addon_to_env(addon_config)
        
        # Import existing model if specified
        if addon_config.get('import_existing_model') and addon_config.get('existing_model_path'):
            import_existing_model(addon_config['existing_model_path'])
        
        log_info("Add-on environment initialized successfully")
        return env_vars
        
    except Exception as e:
        log_error(f"Add-on environment initialization failed: {e}")
        return False

def main():
    """Main configuration adapter function"""
    try:
        log_info("ML Heating Add-on Configuration Adapter Starting...")
        
        # Initialize environment
        env_vars = initialize_addon_environment()
        
        if not env_vars:
            log_error("Environment initialization failed")
            sys.exit(1)
        
        log_info("Configuration adapter completed successfully")
        
        # Print summary for debugging
        addon_config = load_addon_config()
        log_info("=== Configuration Summary ===")
        log_info(f"Target Indoor Temp Entity: {addon_config.get('target_indoor_temp_entity')}")
        log_info(f"Indoor Temp Entity: {addon_config.get('indoor_temp_entity')}")
        log_info(f"Outdoor Temp Entity: {addon_config.get('outdoor_temp_entity')}")
        log_info(f"Learning Rate: {addon_config.get('learning_rate')}")
        log_info(f"Cycle Interval: {addon_config.get('cycle_interval_minutes')} minutes")
        log_info("=============================")
        
    except Exception as e:
        log_error(f"Configuration adapter failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
