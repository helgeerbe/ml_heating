"""
ML Heating Dashboard - Control Component
System control interface for ML heating management
"""

import streamlit as st
import json
import os
import subprocess
import requests
from datetime import datetime
import sys

# Add app directory to Python path
sys.path.append('/app')

def get_ml_system_status():
    """Get current ML system status"""
    try:
        # Check if ML system process is running
        result = subprocess.run(['pgrep', '-f', 'src.main'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except Exception:
        return False

def restart_ml_system():
    """Restart the ML system"""
    try:
        # Use supervisorctl to restart the ML heating service
        result = subprocess.run(
            ['supervisorctl', 'restart', 'ml_heating'],
            capture_output=True, text=True
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def stop_ml_system():
    """Stop the ML system"""
    try:
        result = subprocess.run(
            ['supervisorctl', 'stop', 'ml_heating'],
            capture_output=True, text=True
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def start_ml_system():
    """Start the ML system"""
    try:
        result = subprocess.run(
            ['supervisorctl', 'start', 'ml_heating'],
            capture_output=True, text=True
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def trigger_model_recalibration():
    """Trigger model recalibration"""
    try:
        # This would send a signal to the ML system to recalibrate
        # For now, we'll restart with a flag file
        with open('/data/config/recalibrate_flag', 'w') as f:
            f.write(datetime.now().isoformat())
        
        success, output = restart_ml_system()
        return success, "Recalibration triggered. " + output
    except Exception as e:
        return False, str(e)

def load_current_config():
    """Load current add-on configuration"""
    try:
        with open('/data/options.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return {}

def save_config_changes(config):
    """Save configuration changes (Note: requires add-on restart)"""
    try:
        # Save to a temp file for manual application
        with open('/data/config/pending_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        return True, "Configuration saved. Restart add-on to apply changes."
    except Exception as e:
        return False, str(e)

def render_system_controls():
    """Render system control buttons"""
    st.subheader("🎛️ System Controls")
    
    is_running = get_ml_system_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Restart System", type="primary"):
            with st.spinner("Restarting ML system..."):
                success, output = restart_ml_system()
                if success:
                    st.success("System restarted successfully!")
                else:
                    st.error(f"Restart failed: {output}")
    
    with col2:
        if is_running:
            if st.button("⏹️ Stop System", type="secondary"):
                with st.spinner("Stopping ML system..."):
                    success, output = stop_ml_system()
                    if success:
                        st.success("System stopped successfully!")
                    else:
                        st.error(f"Stop failed: {output}")
        else:
            if st.button("▶️ Start System", type="primary"):
                with st.spinner("Starting ML system..."):
                    success, output = start_ml_system()
                    if success:
                        st.success("System started successfully!")
                    else:
                        st.error(f"Start failed: {output}")
    
    with col3:
        if st.button("🔧 Recalibrate Model"):
            with st.spinner("Triggering model recalibration..."):
                success, output = trigger_model_recalibration()
                if success:
                    st.success("Model recalibration started!")
                    st.info("This will reset learning progress and retrain from historical data.")
                else:
                    st.error(f"Recalibration failed: {output}")
    
    with col4:
        if st.button("📋 View Logs"):
            st.session_state['show_logs'] = True

def render_mode_controls():
    """Render mode switching controls"""
    st.subheader("🔀 Operating Mode")
    
    config = load_current_config()
    
    # This would require integration with the actual ML system configuration
    current_mode = st.radio(
        "Select operating mode:",
        ["Active Mode", "Shadow Mode"],
        help="""
        - **Active Mode**: ML system controls heating directly
        - **Shadow Mode**: ML system observes but doesn't control heating
        """
    )
    
    if current_mode == "Shadow Mode":
        st.info("""
        **Shadow Mode Benefits:**
        - Safe testing without affecting heating
        - Compare ML vs current heat curve performance
        - Build confidence before switching to active control
        """)
    else:
        st.success("""
        **Active Mode:**
        - ML system optimizes heating directly
        - Continuous learning and adaptation
        - Energy efficiency improvements
        """)
    
    if st.button("Apply Mode Change"):
        st.warning("Mode changes require add-on configuration update and restart.")
        st.info("Use Home Assistant add-on configuration to change modes permanently.")

def render_manual_controls():
    """Render manual temperature override controls"""
    st.subheader("🌡️ Manual Override")
    
    st.warning("⚠️ Use manual overrides carefully - they bypass safety systems!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        override_temp = st.number_input(
            "Override Temperature (°C)",
            min_value=14.0,
            max_value=65.0,
            value=40.0,
            step=0.5,
            help="Manually set outlet temperature (bypasses ML predictions)"
        )
        
        override_duration = st.selectbox(
            "Override Duration",
            ["30 minutes", "1 hour", "2 hours", "Until next cycle"],
            help="How long to maintain the override"
        )
    
    with col2:
        st.info("**Current Status:**")
        # This would show actual override status from the system
        st.write("No active overrides")
        
        if st.button("🚨 Apply Override", type="secondary"):
            st.error("Manual override functionality requires deeper ML system integration.")
            st.info("For immediate control, use Home Assistant heating controls directly.")
    
    # Emergency controls
    st.divider()
    st.subheader("🆘 Emergency Controls")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🛑 Emergency Stop", type="secondary"):
            st.error("Emergency stop would disable all heating control.")
            st.info("Use Home Assistant controls for immediate heating shutdown.")
    
    with col4:
        if st.button("🔄 Reset to Heat Curve", type="secondary"):
            st.info("This would restore original heat curve operation.")
            st.warning("Feature requires ML system integration.")

def render_log_viewer():
    """Render log file viewer"""
    if st.session_state.get('show_logs', False):
        st.subheader("📋 System Logs")
        
        log_type = st.selectbox(
            "Select log file:",
            ["ML Heating", "Dashboard", "Health Check", "Supervisor"]
        )
        
        log_files = {
            "ML Heating": "/data/logs/ml_heating.log",
            "Dashboard": "/data/logs/dashboard.log",
            "Health Check": "/data/logs/health_server.log",
            "Supervisor": "/data/logs/supervisord.log"
        }
        
        log_file = log_files[log_type]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            lines_to_show = st.selectbox("Lines to show:", [50, 100, 200, 500])
        with col2:
            if st.button("🔄 Refresh Logs"):
                st.experimental_rerun()
        with col3:
            if st.button("❌ Close Logs"):
                st.session_state['show_logs'] = False
                st.experimental_rerun()
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-lines_to_show:]
                    log_content = ''.join(recent_lines)
                
                st.text_area(
                    f"Last {lines_to_show} lines from {log_type}:",
                    log_content,
                    height=400,
                    disabled=True
                )
            else:
                st.warning(f"Log file not found: {log_file}")
        except Exception as e:
            st.error(f"Error reading log file: {e}")

def render_configuration_editor():
    """Render live configuration editor"""
    st.subheader("⚙️ Live Configuration")
    
    config = load_current_config()
    
    if not config:
        st.error("Unable to load current configuration")
        return
    
    st.warning("Configuration changes require add-on restart to take effect.")
    
    with st.expander("Core Entity Configuration"):
        st.text_input(
            "Target Indoor Temp Entity",
            value=config.get('target_indoor_temp_entity', ''),
            disabled=True,
            help="Change in Home Assistant add-on settings"
        )
        st.text_input(
            "Indoor Temp Entity", 
            value=config.get('indoor_temp_entity', ''),
            disabled=True
        )
        st.text_input(
            "Outdoor Temp Entity",
            value=config.get('outdoor_temp_entity', ''),
            disabled=True
        )
    
    with st.expander("Learning Parameters"):
        new_config = config.copy()
        
        new_config['learning_rate'] = st.slider(
            "Learning Rate",
            min_value=0.001,
            max_value=0.1,
            value=float(config.get('learning_rate', 0.01)),
            step=0.001,
            format="%.3f"
        )
        
        new_config['max_temp_change_per_cycle'] = st.slider(
            "Max Temperature Change per Cycle (°C)",
            min_value=0.5,
            max_value=5.0,
            value=float(config.get('max_temp_change_per_cycle', 2.0)),
            step=0.1
        )
        
        if st.button("💾 Save Parameter Changes"):
            success, message = save_config_changes(new_config)
            if success:
                st.success(message)
            else:
                st.error(message)

def render_control():
    """Main control page"""
    st.header("🎛️ System Control")
    
    # System status indicator
    is_running = get_ml_system_status()
    if is_running:
        st.success("🟢 ML System Status: Running")
    else:
        st.error("🔴 ML System Status: Stopped")
    
    st.divider()
    
    # Main controls
    render_system_controls()
    
    st.divider()
    
    # Mode controls
    render_mode_controls()
    
    st.divider()
    
    # Manual controls
    render_manual_controls()
    
    st.divider()
    
    # Configuration editor
    render_configuration_editor()
    
    # Log viewer (conditional)
    if st.session_state.get('show_logs', False):
        st.divider()
        render_log_viewer()
