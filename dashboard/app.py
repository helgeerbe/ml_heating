#!/usr/bin/env python3
"""
ML Heating Add-on Dashboard - Phase 3 Full Implementation
Advanced dashboard with real-time monitoring, controls, and analytics
Supports Home Assistant ingress integration
"""

import streamlit as st
import os
import sys
from streamlit_option_menu import option_menu

# Add app directory to Python path
sys.path.append('/app')

# Ingress detection and configuration
def setup_ingress_config():
    """Configure Streamlit for Home Assistant ingress support"""
    ingress_path = os.environ.get('HASSIO_INGRESS_PATH', '')
    
    # If running under ingress, configure Streamlit appropriately
    if ingress_path:
        st.write("<!-- Home Assistant Ingress Mode -->")
        # Additional ingress-specific configuration can be added here
    
    return ingress_path

def is_ingress_mode():
    """Check if running under Home Assistant ingress"""
    return bool(os.environ.get('HASSIO_INGRESS_PATH'))

# Import dashboard components
try:
    from components.overview import render_overview
    from components.control import render_control
    from components.performance import render_performance
    from components.backup import render_backup
except ImportError:
    st.error("Dashboard components not available. Ensure all component files are present.")
    st.stop()

def main():
    """Main dashboard application"""
    
    # Setup ingress configuration if running under Home Assistant
    ingress_path = setup_ingress_config()
    
    # Page configuration
    st.set_page_config(
        page_title="ML Heating Control",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Display ingress status for debugging
    if is_ingress_mode():
        st.markdown("<!-- Running in Home Assistant Ingress Mode -->", 
                   unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🔥 ML Heating")
        st.caption("Physics-based machine learning heating optimization")
        
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Control", "Performance", "Backup"],
            icons=["speedometer2", "sliders", "bar-chart-line", "archive"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "green"},
            }
        )
        
        # System status in sidebar
        st.divider()
        st.subheader("Quick Status")
        
        # Check system status
        try:
            if os.path.exists('/data/logs/ml_heating.log'):
                st.success("🟢 ML System Active")
            else:
                st.warning("🟡 ML System Starting")
        except Exception:
            st.error("🔴 System Error")
        
        # Data directory status
        data_dirs = ['/data/models', '/data/backups', '/data/logs']
        for directory in data_dirs:
            if os.path.exists(directory):
                file_count = len(os.listdir(directory))
                st.success(f"📁 {directory.split('/')[-1]}: {file_count}")
            else:
                st.warning(f"📁 {directory.split('/')[-1]}: Missing")
    
    # Main content area
    if selected == "Overview":
        render_overview()
    elif selected == "Control":
        render_control()
    elif selected == "Performance":
        render_performance()
    elif selected == "Backup":
        render_backup()
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("ML Heating Add-on v1.0")
    with col2:
        st.caption("Phase 3 Dashboard")
    with col3:
        st.caption("🏠 Home Assistant Integration")

if __name__ == "__main__":
    main()
