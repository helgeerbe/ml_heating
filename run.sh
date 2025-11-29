#!/usr/bin/with-contenv bashio

# ==============================================================================
# ML Heating Control Add-on Entry Point Script
# ==============================================================================

set -e

# Configure logging
bashio::log.info "Starting ML Heating Control Add-on..."

# Initialize configuration
bashio::log.info "Initializing configuration..."
python3 /app/config_adapter.py

# Check if required environment variables are set
if [[ -z "${SUPERVISOR_TOKEN}" ]]; then
    bashio::log.fatal "Home Assistant Supervisor token not available"
    exit 1
fi

# Setup data directories with proper permissions
mkdir -p /data/{models,backups,logs,config}
chown -R root:root /data

# Import existing model if specified
if bashio::config.true 'import_existing_model' && bashio::config.has_value 'existing_model_path'; then
    MODEL_PATH=$(bashio::config 'existing_model_path')
    if [[ -f "${MODEL_PATH}" ]]; then
        bashio::log.info "Importing existing model from ${MODEL_PATH}..."
        cp "${MODEL_PATH}" /data/models/ml_model.pkl
        bashio::log.info "Model imported successfully"
    else
        bashio::log.warning "Specified model path does not exist: ${MODEL_PATH}"
    fi
fi

# Create log file if it doesn't exist
touch /data/logs/ml_heating.log

# Start the ML heating system directly
bashio::log.info "Starting ML heating system..."

# Change to app directory and start the main application
cd /app

# Execute the main application directly (no supervisor needed)
exec python3 -m src.main
