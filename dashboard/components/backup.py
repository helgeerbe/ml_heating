"""
ML Heating Dashboard - Model Backup & Restore Component
Complete data preservation and model management system
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import pickle
import shutil
import zipfile
import hashlib
from pathlib import Path

# Add app directory to Python path
sys.path.append('/app')

def get_model_files():
    """Get list of all ML model and data files"""
    model_files = {
        'models': [],
        'logs': [],
        'config': [],
        'analytics': []
    }
    
    # Model files
    model_dir = Path('/data/models')
    if model_dir.exists():
        for file_path in model_dir.glob('*.pkl'):
            stat = file_path.stat()
            model_files['models'].append({
                'name': file_path.name,
                'path': str(file_path),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'type': 'model'
            })
    
    # Log files
    log_dir = Path('/data/logs')
    if log_dir.exists():
        for file_path in log_dir.glob('*.log'):
            stat = file_path.stat()
            model_files['logs'].append({
                'name': file_path.name,
                'path': str(file_path),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'type': 'log'
            })
    
    # Config files
    config_dir = Path('/data/config')
    if config_dir.exists():
        for file_path in config_dir.glob('*'):
            if file_path.is_file():
                stat = file_path.stat()
                model_files['config'].append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'type': 'config'
                })
    
    return model_files

def get_existing_backups():
    """Get list of existing backup files including uploaded models"""
    backup_dir = Path('/data/backups')
    backup_dir.mkdir(exist_ok=True)
    
    backups = []
    # Get system backups and uploaded models
    for backup_file in backup_dir.glob('*.zip'):
        stat = backup_file.stat()
        
        # Determine backup type and extract timestamp
        if backup_file.name.startswith('ml_backup_'):
            backup_type = 'system'
            timestamp_str = backup_file.stem.replace('ml_backup_', '')
        elif backup_file.name.startswith('uploaded_'):
            backup_type = 'uploaded'
            timestamp_str = backup_file.stem.replace('uploaded_', '')
        else:
            backup_type = 'unknown'
            timestamp_str = ''
        
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            timestamp = datetime.fromtimestamp(stat.st_mtime)
        
        # Calculate MD5 hash for integrity checking
        md5_hash = hashlib.md5()
        with open(backup_file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        backups.append({
            'name': backup_file.name,
            'path': str(backup_file),
            'size': stat.st_size,
            'created': timestamp,
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'md5': md5_hash.hexdigest(),
            'type': backup_type
        })
    
    return sorted(backups, key=lambda x: x['created'], reverse=True)

def create_backup_from_upload(model_file=None, state_file=None, upload_name=None):
    """Create a backup from uploaded model files"""
    try:
        if not upload_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            upload_name = f'uploaded_{timestamp}'
        
        backup_dir = Path('/data/backups')
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / f'{upload_name}.zip'
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            # Add uploaded model file
            if model_file:
                backup_zip.writestr('models/ml_model.pkl', model_file.getvalue())
            
            # Add uploaded state file or create empty one
            if state_file:
                backup_zip.writestr('models/ml_state.pkl', state_file.getvalue())
            else:
                # Create minimal state file if only model uploaded
                import pickle
                minimal_state = {
                    'confidence': 0.5,
                    'mae': 0.0,
                    'rmse': 0.0,
                    'cycle_count': 0,
                    'last_prediction': 0.0,
                    'uploaded_model': True
                }
                state_data = pickle.dumps(minimal_state)
                backup_zip.writestr('models/ml_state.pkl', state_data)
            
            # Create upload manifest
            manifest = {
                'created': datetime.now().isoformat(),
                'backup_name': upload_name,
                'backup_type': 'uploaded_model',
                'has_model': model_file is not None,
                'has_state': state_file is not None,
                'system_info': {
                    'addon_version': '1.0',
                    'upload_source': 'dashboard'
                }
            }
            
            manifest_json = json.dumps(manifest, indent=2)
            backup_zip.writestr('backup_manifest.json', manifest_json)
        
        return True, backup_path, manifest
        
    except Exception as e:
        return False, None, str(e)

def create_backup(backup_name=None, include_logs=True, include_analytics=True):
    """Create a comprehensive backup of ML system state"""
    try:
        if not backup_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f'ml_backup_{timestamp}'
        
        backup_dir = Path('/data/backups')
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / f'{backup_name}.zip'
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            # Backup models directory
            models_dir = Path('/data/models')
            if models_dir.exists():
                for file_path in models_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = f'models/{file_path.relative_to(models_dir)}'
                        backup_zip.write(file_path, arcname)
            
            # Backup configuration
            config_dir = Path('/data/config')
            if config_dir.exists():
                for file_path in config_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = f'config/{file_path.relative_to(config_dir)}'
                        backup_zip.write(file_path, arcname)
            
            # Backup add-on configuration
            addon_config = Path('/data/options.json')
            if addon_config.exists():
                backup_zip.write(addon_config, 'addon_config.json')
            
            # Include logs if requested
            if include_logs:
                logs_dir = Path('/data/logs')
                if logs_dir.exists():
                    for file_path in logs_dir.glob('*.log'):
                        arcname = f'logs/{file_path.name}'
                        backup_zip.write(file_path, arcname)
            
            # Create backup manifest
            manifest = {
                'created': datetime.now().isoformat(),
                'backup_name': backup_name,
                'include_logs': include_logs,
                'include_analytics': include_analytics,
                'files_backed_up': [],
                'system_info': {
                    'addon_version': '1.0',
                    'python_version': sys.version,
                    'platform': os.name
                }
            }
            
            # Add file list to manifest
            for file_info in backup_zip.filelist:
                manifest['files_backed_up'].append({
                    'filename': file_info.filename,
                    'file_size': file_info.file_size,
                    'compress_size': file_info.compress_size
                })
            
            # Write manifest
            manifest_json = json.dumps(manifest, indent=2)
            backup_zip.writestr('backup_manifest.json', manifest_json)
        
        return True, backup_path, manifest
        
    except Exception as e:
        return False, None, str(e)

def restore_backup(backup_path, restore_models=True, restore_config=True, restore_logs=False):
    """Restore ML system from backup"""
    try:
        backup_file = Path(backup_path)
        if not backup_file.exists():
            return False, "Backup file not found"
        
        # Create restoration timestamp
        restore_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Backup current state before restoration
        current_backup_name = f'pre_restore_{restore_time}'
        success, _, _ = create_backup(current_backup_name, include_logs=False)
        if not success:
            return False, "Failed to backup current state before restoration"
        
        with zipfile.ZipFile(backup_file, 'r') as backup_zip:
            # Read manifest
            manifest = {}
            try:
                manifest_data = backup_zip.read('backup_manifest.json')
                manifest = json.loads(manifest_data)
            except KeyError:
                st.warning("Backup manifest not found - proceeding with basic restore")
            
            # Restore models
            if restore_models:
                models_dir = Path('/data/models')
                models_dir.mkdir(exist_ok=True)
                
                for file_info in backup_zip.filelist:
                    if file_info.filename.startswith('models/'):
                        extract_path = models_dir / file_info.filename[7:]  # Remove 'models/' prefix
                        extract_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with backup_zip.open(file_info) as source:
                            with open(extract_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
            
            # Restore configuration
            if restore_config:
                config_dir = Path('/data/config')
                config_dir.mkdir(exist_ok=True)
                
                for file_info in backup_zip.filelist:
                    if file_info.filename.startswith('config/'):
                        extract_path = config_dir / file_info.filename[7:]  # Remove 'config/' prefix
                        extract_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with backup_zip.open(file_info) as source:
                            with open(extract_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                
                # Restore add-on configuration
                try:
                    addon_config_data = backup_zip.read('addon_config.json')
                    with open('/data/config/restored_addon_config.json', 'wb') as f:
                        f.write(addon_config_data)
                except KeyError:
                    pass  # No add-on config in backup
            
            # Restore logs if requested
            if restore_logs:
                logs_dir = Path('/data/logs')
                logs_dir.mkdir(exist_ok=True)
                
                for file_info in backup_zip.filelist:
                    if file_info.filename.startswith('logs/'):
                        extract_path = logs_dir / f"restored_{file_info.filename[5:]}"
                        
                        with backup_zip.open(file_info) as source:
                            with open(extract_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
        
        return True, f"Restoration completed. Current state backed up as {current_backup_name}"
        
    except Exception as e:
        return False, f"Restoration failed: {str(e)}"

def export_model_data():
    """Export model data for external use"""
    try:
        export_dir = Path('/data/exports')
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = export_dir / f'ml_export_{timestamp}.json'
        
        export_data = {
            'export_info': {
                'created': datetime.now().isoformat(),
                'addon_version': '1.0',
                'export_type': 'model_data'
            },
            'model_state': {},
            'learning_history': {},
            'configuration': {}
        }
        
        # Export thermal state if available (unified JSON format)
        thermal_state_file = Path('/opt/ml_heating/thermal_state.json')
        if thermal_state_file.exists():
            try:
                with open(thermal_state_file, 'r') as f:
                    thermal_state = json.load(f)
                # Export relevant thermal state data
                export_data['thermal_state'] = thermal_state
                export_data['model_state'] = {
                    'learning_confidence': thermal_state.get('learning_state', {}).get('learning_confidence', 0),
                    'cycle_count': thermal_state.get('learning_state', {}).get('cycle_count', 0),
                    'baseline_source': thermal_state.get('baseline_parameters', {}).get('source', 'unknown'),
                    'calibration_date': thermal_state.get('baseline_parameters', {}).get('calibration_date'),
                    'total_predictions': thermal_state.get('prediction_metrics', {}).get('total_predictions', 0)
                }
            except Exception:
                export_data['model_state'] = {'error': 'Failed to load thermal state'}
        
        # Export configuration
        config_file = Path('/data/options.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                export_data['configuration'] = config
            except Exception:
                export_data['configuration'] = {'error': 'Failed to load configuration'}
        
        # Write export file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return True, export_path
        
    except Exception as e:
        return False, str(e)

def import_model_data(import_file_path):
    """Import model data from external source"""
    try:
        import_file = Path(import_file_path)
        if not import_file.exists():
            return False, "Import file not found"
        
        with open(import_file, 'r') as f:
            import_data = json.load(f)
        
        # Validate import data structure
        required_sections = ['export_info', 'model_state', 'configuration']
        for section in required_sections:
            if section not in import_data:
                return False, f"Invalid import file: missing {section} section"
        
        # Import configuration
        if 'configuration' in import_data and import_data['configuration']:
            config_backup_path = Path('/data/config/imported_config.json')
            with open(config_backup_path, 'w') as f:
                json.dump(import_data['configuration'], f, indent=2)
        
        # Import model state
        if 'model_state' in import_data and import_data['model_state']:
            # Convert back to internal format and save
            models_dir = Path('/data/models')
            models_dir.mkdir(exist_ok=True)
            
            imported_state_path = models_dir / 'imported_ml_state.pkl'
            with open(imported_state_path, 'wb') as f:
                pickle.dump(import_data['model_state'], f)
        
        return True, "Import completed successfully"
        
    except Exception as e:
        return False, f"Import failed: {str(e)}"

def render_backup_overview():
    """Render backup system overview"""
    st.subheader("💾 Backup System Overview")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    backups = get_existing_backups()
    model_files = get_model_files()
    
    total_model_size = sum(f['size'] for f in model_files['models']) / 1024 / 1024  # MB
    total_backup_size = sum(b['size'] for b in backups) / 1024 / 1024  # MB
    
    with col1:
        st.metric("Available Backups", len(backups))
    with col2:
        st.metric("Model Data Size", f"{total_model_size:.1f} MB")
    with col3:
        st.metric("Total Backup Size", f"{total_backup_size:.1f} MB")
    with col4:
        if backups:
            days_since_backup = (datetime.now() - backups[0]['created']).days
            st.metric("Last Backup", f"{days_since_backup} days ago")
        else:
            st.metric("Last Backup", "Never")

def render_model_upload():
    """Render model upload interface"""
    st.subheader("📤 Upload Model Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        upload_type = st.radio(
            "Upload Type:",
            ["Model + State Files", "Model Only", "Complete Backup ZIP"]
        )
        
        if upload_type == "Model + State Files":
            model_file = st.file_uploader("Model File (.pkl)", type=['pkl'], key="model_upload")
            state_file = st.file_uploader("State File (.pkl)", type=['pkl'], key="state_upload")
            
            if model_file and state_file:
                upload_name = st.text_input(
                    "Upload Name (optional):",
                    placeholder="Leave empty for auto-generated name"
                )
                
                if st.button("📤 Upload as Backup", type="primary"):
                    with st.spinner("Processing upload..."):
                        success, backup_path, manifest = create_backup_from_upload(
                            model_file, state_file, upload_name if upload_name else None
                        )
                        
                        if success:
                            st.success("✅ Model uploaded successfully!")
                            st.info(f"📁 Saved as: {backup_path.name}")
                            st.caption("Model is now available in backup list for activation")
                        else:
                            st.error(f"❌ Upload failed: {manifest}")
        
        elif upload_type == "Model Only":
            model_file = st.file_uploader("Model File (.pkl)", type=['pkl'], key="model_only_upload")
            
            if model_file:
                upload_name = st.text_input(
                    "Upload Name (optional):",
                    placeholder="Leave empty for auto-generated name"
                )
                
                st.info("State file will be auto-generated with default values")
                
                if st.button("📤 Upload as Backup", type="primary"):
                    with st.spinner("Processing upload..."):
                        success, backup_path, manifest = create_backup_from_upload(
                            model_file, None, upload_name if upload_name else None
                        )
                        
                        if success:
                            st.success("✅ Model uploaded successfully!")
                            st.info(f"📁 Saved as: {backup_path.name}")
                            st.caption("Model is now available in backup list for activation")
                        else:
                            st.error(f"❌ Upload failed: {manifest}")
        
        elif upload_type == "Complete Backup ZIP":
            backup_file = st.file_uploader("Backup ZIP File", type=['zip'], key="backup_upload")
            
            if backup_file:
                st.info("Complete backup will be restored directly")
                
                if st.button("📥 Upload & Restore", type="primary"):
                    with st.spinner("Processing backup restore..."):
                        # Save uploaded backup temporarily
                        temp_backup_path = Path('/tmp') / f'temp_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
                        with open(temp_backup_path, 'wb') as f:
                            f.write(backup_file.getvalue())
                        
                        success, message = restore_backup(str(temp_backup_path))
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.info("🔄 Restart add-on to apply restored configuration")
                        else:
                            st.error(f"❌ {message}")
                        
                        # Clean up temp file
                        try:
                            temp_backup_path.unlink()
                        except:
                            pass
    
    with col2:
        st.info("**Upload Options:**")
        st.write("📤 **Model + State:** Upload both files for complete model transfer")
        st.write("📤 **Model Only:** Upload just model, state will be auto-generated")
        st.write("📦 **Complete Backup:** Upload and restore full backup ZIP")
        
        st.warning("**Important:**")
        st.write("• Uploaded models become backups that you can activate later")
        st.write("• Always backup current model before activation")
        st.write("• Restart add-on after model activation")


def render_current_model_download():
    """Render current model download interface"""
    st.subheader("📥 Download Current Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Download Options:**")
        
        download_type = st.radio(
            "Download Type:",
            ["Model + State Files", "Model Only", "Complete Backup"]
        )
        
        if download_type == "Model + State Files":
            if st.button("📥 Download Model + State", type="primary"):
                # Create temporary backup of current model
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                temp_backup_name = f'download_{timestamp}'
                
                with st.spinner("Creating download..."):
                    success, backup_path, manifest = create_backup(temp_backup_name, include_logs=False, include_analytics=False)
                    
                    if success:
                        st.success("✅ Download ready!")
                        st.info(f"📁 File: {backup_path.name}")
                        st.caption("Download link would be provided here")
                    else:
                        st.error(f"❌ Download failed: {manifest}")
        
        elif download_type == "Model Only":
            if st.button("📥 Download Model Only", type="primary"):
                model_path = Path('/data/models/ml_model.pkl')
                if model_path.exists():
                    st.success("✅ Model file ready for download!")
                    st.info("📁 File: ml_model.pkl")
                    st.caption("Download link would be provided here")
                else:
                    st.error("❌ Model file not found")
        
        elif download_type == "Complete Backup":
            if st.button("📥 Create & Download Backup", type="primary"):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f'complete_download_{timestamp}'
                
                with st.spinner("Creating complete backup..."):
                    success, backup_path, manifest = create_backup(backup_name, include_logs=True, include_analytics=True)
                    
                    if success:
                        st.success("✅ Complete backup ready!")
                        st.info(f"📁 File: {backup_path.name}")
                        st.caption("Download link would be provided here")
                    else:
                        st.error(f"❌ Backup creation failed: {manifest}")
    
    with col2:
        st.info("**Current Model Status:**")
        
        # Show current model info
        model_path = Path('/data/models/ml_model.pkl')
        state_path = Path('/data/models/ml_state.pkl')
        
        if model_path.exists():
            model_stat = model_path.stat()
            st.write(f"✅ **Model File:** {model_path.name}")
            st.write(f"📊 **Size:** {model_stat.st_size / 1024:.1f} KB")
            st.write(f"🕒 **Modified:** {datetime.fromtimestamp(model_stat.st_mtime).strftime('%Y-%m-%d %H:%M')}")
        else:
            st.write("❌ **Model File:** Not found")
        
        if state_path.exists():
            state_stat = state_path.stat()
            st.write(f"✅ **State File:** {state_path.name}")
            st.write(f"📊 **Size:** {state_stat.st_size / 1024:.1f} KB")
            st.write(f"🕒 **Modified:** {datetime.fromtimestamp(state_stat.st_mtime).strftime('%Y-%m-%d %H:%M')}")
        else:
            st.write("❌ **State File:** Not found")


def render_backup_activation():
    """Render backup activation interface"""
    st.subheader("🔄 Activate Backup as Current Model")
    
    backups = get_existing_backups()
    
    if not backups:
        st.info("No backups available for activation. Create or upload a backup first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Separate backups by type for better display
        system_backups = [b for b in backups if b['type'] == 'system']
        uploaded_backups = [b for b in backups if b['type'] == 'uploaded']
        
        backup_options = []
        for backup in system_backups:
            backup_options.append(f"📁 {backup['name']} ({backup['created'].strftime('%m-%d %H:%M')})")
        for backup in uploaded_backups:
            backup_options.append(f"📤 {backup['name']} ({backup['created'].strftime('%m-%d %H:%M')})")
        
        if backup_options:
            selected_display = st.selectbox("Select backup to activate:", backup_options)
            
            # Find the actual backup object
            selected_backup = None
            for backup in backups:
                if backup['name'] in selected_display:
                    selected_backup = backup
                    break
            
            if selected_backup:
                st.warning("⚠️ This will replace your current active model!")
                st.info("💡 Current model will be automatically backed up before activation")
                st.caption("New model will take effect at next system restart or cycle")
                
                if st.button("🔄 Activate Selected Backup", type="primary"):
                    with st.spinner("Activating backup..."):
                        success, message = activate_backup_as_current(selected_backup['path'])
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.info("🔄 Restart add-on now or wait for next cycle to apply changes")
                        else:
                            st.error(f"❌ {message}")
    
    with col2:
        if 'selected_backup' in locals() and selected_backup:
            st.info("**Selected Backup Info:**")
            st.write(f"**Name:** {selected_backup['name']}")
            st.write(f"**Type:** {'System Backup' if selected_backup['type'] == 'system' else 'Uploaded Model'}")
            st.write(f"**Created:** {selected_backup['created'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Size:** {selected_backup['size'] / 1024 / 1024:.1f} MB")
            st.write(f"**MD5:** {selected_backup['md5'][:16]}...")


def activate_backup_as_current(backup_path):
    """Activate a backup as the current model"""
    try:
        # Safety backup current state
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safety_backup_name = f'pre_activation_{timestamp}'
        success, _, _ = create_backup(safety_backup_name, include_logs=False, include_analytics=False)
        
        if not success:
            return False, "Failed to backup current state before activation"
        
        # Extract and copy files from backup
        backup_file = Path(backup_path)
        if not backup_file.exists():
            return False, "Backup file not found"
        
        with zipfile.ZipFile(backup_file, 'r') as backup_zip:
            # Extract model files to temporary location
            temp_dir = Path('/tmp/model_activation')
            temp_dir.mkdir(exist_ok=True)
            
            # Extract model and state files
            model_extracted = False
            state_extracted = False
            
            for file_info in backup_zip.filelist:
                if file_info.filename == 'models/ml_model.pkl':
                    backup_zip.extract(file_info, temp_dir)
                    model_extracted = True
                elif file_info.filename == 'models/ml_state.pkl':
                    backup_zip.extract(file_info, temp_dir)
                    state_extracted = True
            
            if not model_extracted:
                return False, "Model file not found in backup"
            
            # Copy to active locations
            models_dir = Path('/data/models')
            models_dir.mkdir(exist_ok=True)
            
            shutil.copy(temp_dir / 'models/ml_model.pkl', models_dir / 'ml_model.pkl')
            
            if state_extracted:
                shutil.copy(temp_dir / 'models/ml_state.pkl', models_dir / 'ml_state.pkl')
            else:
                # Create minimal state file if not in backup
                minimal_state = {
                    'confidence': 0.5,
                    'mae': 0.0,
                    'rmse': 0.0,
                    'cycle_count': 0,
                    'last_prediction': 0.0,
                    'activated_from_backup': True
                }
                with open(models_dir / 'ml_state.pkl', 'wb') as f:
                    pickle.dump(minimal_state, f)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True, f"Model activated successfully! Current state backed up as {safety_backup_name}"
        
    except Exception as e:
        return False, f"Activation failed: {str(e)}"


def render_create_backup():
    """Render backup creation interface"""
    st.subheader("🔄 Create New Backup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        backup_name = st.text_input(
            "Backup Name (optional)", 
            placeholder="Leave empty for auto-generated name"
        )
        
        include_logs = st.checkbox("Include Log Files", value=True)
        include_analytics = st.checkbox("Include Analytics Data", value=True)
        
        if st.button("🔄 Create Backup", type="primary"):
            with st.spinner("Creating backup..."):
                success, backup_path, manifest = create_backup(
                    backup_name if backup_name else None,
                    include_logs,
                    include_analytics
                )
                
                if success:
                    st.success("✅ Backup created successfully!")
                    st.info(f"📁 Backup saved: {backup_path.name}")
                    
                    # Show backup details
                    if manifest:
                        st.json({
                            'backup_name': manifest['backup_name'],
                            'created': manifest['created'],
                            'files_count': len(manifest['files_backed_up']),
                            'include_logs': manifest['include_logs']
                        })
                else:
                    st.error(f"❌ Backup failed: {manifest}")
    
    with col2:
        st.info("**What gets backed up:**")
        st.write("✅ ML model files (.pkl)")
        st.write("✅ Learning state and progress")
        st.write("✅ Configuration files")
        st.write("✅ Add-on settings")
        
        if include_logs:
            st.write("✅ System logs")
        else:
            st.write("⏭️ System logs (excluded)")
        
        if include_analytics:
            st.write("✅ Analytics data")
        else:
            st.write("⏭️ Analytics data (excluded)")


def render_backup_list():
    """Render list of existing backups"""
    st.subheader("📁 Existing Backups")
    
    backups = get_existing_backups()
    
    if not backups:
        st.info("No backups found. Create your first backup above.")
        return
    
    # Backup list table
    backup_data = []
    for backup in backups:
        backup_data.append({
            'Name': backup['name'],
            'Created': backup['created'].strftime('%Y-%m-%d %H:%M'),
            'Size': f"{backup['size'] / 1024 / 1024:.1f} MB",
            'MD5': backup['md5'][:8] + '...',
            'Path': backup['path']
        })
    
    df = pd.DataFrame(backup_data)
    
    # Display table
    st.dataframe(
        df[['Name', 'Created', 'Size', 'MD5']],
        use_container_width=True,
        hide_index=True
    )
    
    # Backup actions
    st.subheader("🔧 Backup Actions")
    
    selected_backup = st.selectbox(
        "Select backup for actions:",
        options=[b['name'] for b in backups],
        format_func=lambda x: f"{x} ({next(b['created'].strftime('%Y-%m-%d %H:%M') for b in backups if b['name'] == x)})"
    )
    
    if selected_backup:
        backup_info = next(b for b in backups if b['name'] == selected_backup)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Restore Backup", type="primary"):
                st.session_state['restore_backup'] = backup_info
        
        with col2:
            if st.button("📋 View Details"):
                st.session_state['view_backup'] = backup_info
        
        with col3:
            if st.button("📤 Download"):
                st.info("Download functionality would be implemented here")
        
        with col4:
            if st.button("🗑️ Delete", type="secondary"):
                st.session_state['delete_backup'] = backup_info


def render_restore_interface():
    """Render backup restoration interface"""
    if 'restore_backup' not in st.session_state:
        return
    
    backup_info = st.session_state['restore_backup']
    
    st.subheader(f"🔄 Restore: {backup_info['name']}")
    st.warning("⚠️ Restoring will replace current ML system state!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Restore Options:**")
        restore_models = st.checkbox("Restore ML Models", value=True)
        restore_config = st.checkbox("Restore Configuration", value=True)
        restore_logs = st.checkbox("Restore Logs", value=False)
        
        st.info("Current state will be automatically backed up before restoration.")
    
    with col2:
        st.write("**Backup Information:**")
        st.write(f"Created: {backup_info['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Size: {backup_info['size'] / 1024 / 1024:.1f} MB")
        st.write(f"MD5: {backup_info['md5']}")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("✅ Confirm Restore", type="primary"):
            with st.spinner("Restoring backup..."):
                success, message = restore_backup(
                    backup_info['path'],
                    restore_models,
                    restore_config, 
                    restore_logs
                )
                
                if success:
                    st.success(f"✅ {message}")
                    st.info("🔄 Restart add-on to apply restored configuration.")
                else:
                    st.error(f"❌ {message}")
            
            del st.session_state['restore_backup']
    
    with col4:
        if st.button("❌ Cancel"):
            del st.session_state['restore_backup']
            st.rerun()


def render_import_export():
    """Render import/export interface"""
    st.subheader("📦 Import / Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📤 Export Model Data**")
        st.caption("Export ML state and configuration as JSON")
        
        if st.button("📤 Export Data", type="primary"):
            with st.spinner("Exporting model data..."):
                success, export_path = export_model_data()
                
                if success:
                    st.success(f"✅ Export successful!")
                    st.info(f"📁 File: {export_path.name}")
                    st.caption(f"Location: {export_path}")
                else:
                    st.error(f"❌ Export failed: {export_path}")
    
    with col2:
        st.write("**📥 Import Model Data**")
        st.caption("Import ML state from JSON file")
        
        import_file = st.text_input(
            "Import file path:",
            placeholder="/data/exports/ml_export_20231126_120000.json"
        )
        
        if st.button("📥 Import Data") and import_file:
            with st.spinner("Importing model data..."):
                success, message = import_model_data(import_file)
                
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")


def render_automatic_backup():
    """Render automatic backup configuration"""
    st.subheader("⏰ Automatic Backup Configuration")
    
    st.info("Automatic backup scheduling will be implemented in a future update.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Planned Features:**")
        st.write("• Daily automatic backups")
        st.write("• Weekly archive backups")
        st.write("• Configurable retention policy")
        st.write("• Email notifications")
        st.write("• Cloud storage integration")
    
    with col2:
        st.write("**Current Recommendations:**")
        st.write("• Create manual backups before major changes")
        st.write("• Backup weekly during learning phase")
        st.write("• Keep 3-5 recent backups")
        st.write("• Test restore process periodically")
        st.write("• Export critical configurations")


def render_backup_analytics():
    """Render backup analytics and insights"""
    st.subheader("📊 Backup Analytics")
    
    backups = get_existing_backups()
    
    if len(backups) < 2:
        st.info("Create more backups to see analytics and trends.")
        return
    
    # Create backup timeline chart
    backup_df = pd.DataFrame(backups)
    backup_df['size_mb'] = backup_df['size'] / 1024 / 1024
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=backup_df['created'],
        y=backup_df['size_mb'],
        mode='lines+markers',
        name='Backup Size',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Backup Size Trend",
        xaxis_title="Backup Date",
        yaxis_title="Size (MB)",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Backup statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_size = backup_df['size_mb'].mean()
        st.metric("Average Size", f"{avg_size:.1f} MB")
    
    with col2:
        growth_rate = (backup_df['size_mb'].iloc[0] - backup_df['size_mb'].iloc[-1]) / len(backups)
        st.metric("Size Growth", f"{growth_rate:+.1f} MB/backup")
    
    with col3:
        backup_frequency = len(backups) / max(1, (datetime.now() - backup_df['created'].min()).days)
        st.metric("Backup Frequency", f"{backup_frequency:.1f}/day")


def render_backup():
    """Main backup page with unified model management"""
    st.header("💾 Model Management System")
    st.caption("Complete backup, upload, download, and activation system")
    
    # Auto-refresh option
    if st.button("🔄 Refresh"):
        st.rerun()
    
    # Render main sections
    render_backup_overview()
    
    st.divider()
    
    # Model Upload Section
    render_model_upload()
    
    st.divider()
    
    # Current Model Download Section  
    render_current_model_download()
    
    st.divider()
    
    # Backup Activation Section
    render_backup_activation()
    
    st.divider()
    
    # Create System Backup
    render_create_backup()
    
    st.divider()
    
    # Backup List with Actions
    render_backup_list()
    
    # Handle restore interface
    render_restore_interface()
    
    st.divider()
    
    # Import/Export for JSON data
    render_import_export()
    
    st.divider()
    
    # Future features
    render_automatic_backup()
    
    st.divider()
    
    # Analytics
    render_backup_analytics()
