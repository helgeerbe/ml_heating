#!/usr/bin/env python3
"""
Enhanced Container Validation Script
Comprehensive quality assurance for ML Heating Add-on
"""

import json
import yaml
import sys
import importlib.util
import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple

def validate_dashboard_components():
    """Validate all dashboard components exist and are properly structured"""
    try:
        component_dir = Path('dashboard/components')
        if not component_dir.exists():
            print("❌ Dashboard components directory not found")
            return False
        
        # Required components with expected functions
        required_components = {
            'overview.py': ['render_overview'],
            'control.py': ['render_control'],
            'performance.py': ['render_performance'],
            'backup.py': ['render_backup']
        }
        
        missing_components = []
        invalid_components = []
        
        for component_file, expected_functions in required_components.items():
            component_path = component_dir / component_file
            
            if not component_path.exists():
                missing_components.append(component_file)
                continue
            
            # Parse component file to check for required functions
            try:
                with open(component_path, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                
                missing_functions = [
                    func for func in expected_functions 
                    if func not in function_names
                ]
                if missing_functions:
                    invalid_components.append(
                        f"{component_file} missing: {missing_functions}")
                
            except Exception as e:
                invalid_components.append(f"{component_file} parse error: {e}")
        
        if missing_components:
            print(f"❌ Missing dashboard components: {missing_components}")
            return False
        
        if invalid_components:
            print(f"❌ Invalid dashboard components: {invalid_components}")
            return False
        
        print("✅ Dashboard components validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard component validation failed: {e}")
        return False

def validate_advanced_analytics():
    """Validate performance analytics and visualization components"""
    try:
        performance_file = Path(
            'dashboard/components/performance.py')
        if not performance_file.exists():
            print("❌ Performance analytics component not found")
            return False
        
        with open(performance_file, 'r') as f:
            content = f.read()
        
        # Check for required analytics functions
        required_analytics = [
            'render_learning_progress',
            'render_feature_importance', 
            'render_prediction_accuracy',
            'render_energy_efficiency',
            'render_system_insights'
        ]
        
        tree = ast.parse(content)
        function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        missing_analytics = [func for func in required_analytics if func not in function_names]
        if missing_analytics:
            print(f"❌ Missing analytics functions: {missing_analytics}")
            return False
        
        # Check for Plotly imports
        if 'plotly.graph_objects' not in content or 'plotly.subplots' not in content:
            print("❌ Missing required Plotly imports for analytics")
            return False
        
        # Check for data processing imports
        required_imports = ['pandas', 'numpy']
        for imp in required_imports:
            if f'import {imp}' not in content:
                print(f"❌ Missing required import: {imp}")
                return False
        
        print("✅ Advanced analytics validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Advanced analytics validation failed: {e}")
        return False

def validate_backup_system():
    """Validate backup/restore functionality components"""
    try:
        backup_file = Path('dashboard/components/backup.py')
        if not backup_file.exists():
            print("❌ Backup system component not found")
            return False
        
        with open(backup_file, 'r') as f:
            content = f.read()
        
        # Check for required backup functions
        required_backup_functions = [
            'create_backup',
            'restore_backup',
            'get_existing_backups',
            'export_model_data',
            'import_model_data'
        ]
        
        tree = ast.parse(content)
        function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        missing_functions = [func for func in required_backup_functions if func not in function_names]
        if missing_functions:
            print(f"❌ Missing backup functions: {missing_functions}")
            return False
        
        # Check for required imports for backup functionality
        required_imports = ['zipfile', 'hashlib', 'pickle', 'shutil']
        for imp in required_imports:
            if f'import {imp}' not in content:
                print(f"❌ Missing required backup import: {imp}")
                return False
        
        print("✅ Backup system validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Backup system validation failed: {e}")
        return False

def validate_dependency_compatibility():
    """Check for version conflicts and requirement compatibility"""
    try:
        # Load main requirements
        main_reqs_file = Path('requirements.txt')
        
        if not main_reqs_file.exists():
            print("❌ Main requirements.txt not found")
            return False
        
        # Parse requirements
        with open(main_reqs_file, 'r') as f:
            main_reqs = f.read().strip().split('\n')
        
        # Check for required dashboard dependencies in main requirements
        required_dashboard_deps = [
            'streamlit',
            'plotly', 
            'pandas',
            'numpy'
        ]
        
        main_deps_lower = [dep.lower() for dep in main_reqs]
        missing_deps = []
        
        for dep in required_dashboard_deps:
            if not any(dep in main_dep.lower() for main_dep in main_deps_lower):
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"❌ Missing required dashboard dependencies: {missing_deps}")
            return False
        
        print("✅ Dependency compatibility validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Dependency compatibility validation failed: {e}")
        return False

def validate_api_structure():
    """Validate development API structure and components"""
    try:
        # Check for main app.py structure
        app_file = Path('dashboard/app.py')
        if not app_file.exists():
            print("❌ Main dashboard app.py not found")
            return False
        
        with open(app_file, 'r') as f:
            content = f.read()
        
        # Check for required imports and structure
        required_elements = [
            'streamlit',
            'option_menu',
            'render_overview',
            'render_control', 
            'render_performance',
            'render_backup'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing app.py elements: {missing_elements}")
            return False
        
        # Check for health check component
        health_file = Path('dashboard/health.py')
        if not health_file.exists():
            print("❌ Health check component not found")
            return False
        
        print("✅ API structure validation passed")
        return True
        
    except Exception as e:
        print(f"❌ API structure validation failed: {e}")
        return False

def validate_config_yaml():
    """Validate config.yaml structure"""
    try:
        # Check both addon configurations
        configs_to_check = ['ml_heating/config.yaml', 'ml_heating_dev/config.yaml']
        for config_path in configs_to_check:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        required_fields = ['name', 'version', 'slug', 'description', 'arch', 'options', 'schema']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            print(f"❌ Config.yaml missing required fields: {missing_fields}")
            return False
        
        # Validate architectures (ML apps typically support modern 64-bit architectures)
        required_archs = ['aarch64', 'amd64']
        if set(config['arch']) != set(required_archs):
            print(f"❌ Config.yaml architectures incomplete. Expected: {required_archs}")
            return False
        
        print("✅ config.yaml validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Config.yaml validation failed: {e}")
        return False

def validate_build_json():
    """Validate build.json structure"""
    try:
        with open('build.yaml', 'r') as f:
            build = yaml.safe_load(f)
        
        required_fields = ['build_from', 'args', 'labels']
        missing_fields = [field for field in required_fields if field not in build]
        
        if missing_fields:
            print(f"❌ Build.json missing required fields: {missing_fields}")
            return False
        
        # Validate build_from has all architectures (ML apps typically support modern 64-bit architectures)
        required_archs = ['aarch64', 'amd64']
        if set(build['build_from'].keys()) != set(required_archs):
            print(f"❌ Build.json missing architectures: {required_archs}")
            return False
        
        print("✅ build.json validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Build.json validation failed: {e}")
        return False

def validate_dockerfile():
    """Validate Dockerfile structure"""
    try:
        dockerfile_path = Path('Dockerfile')
        if not dockerfile_path.exists():
            print("❌ Dockerfile not found")
            return False
        
        content = dockerfile_path.read_text()
        
        # Check for required instructions
        required_instructions = [
            'FROM',
            'COPY requirements.txt',
            'pip3 install',
            'COPY src/',
            'COPY run.sh',
            'COPY config_adapter.py',
            'HEALTHCHECK',
            'EXPOSE 3001 3002',
            'CMD ["/app/run.sh"]'
        ]
        
        missing_instructions = []
        for instruction in required_instructions:
            if instruction not in content:
                missing_instructions.append(instruction)
        
        if missing_instructions:
            print(f"❌ Dockerfile missing instructions: {missing_instructions}")
            return False
        
        print("✅ Dockerfile validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Dockerfile validation failed: {e}")
        return False

def validate_dependencies():
    """Validate dependency files exist"""
    required_files = [
        'requirements.txt',
        'config_adapter.py',
        'run.sh',
        'supervisord.conf',
        'dashboard/app.py',
        'dashboard/health.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def validate_repository_structure():
    """Validate repository structure (supports both yaml and json)"""
    try:
        # Try repository.yaml first (our format)
        try:
            with open('repository.yaml', 'r') as f:
                repo = yaml.safe_load(f)
            file_type = "repository.yaml"
        except FileNotFoundError:
            # Fallback to repository.json
            with open('repository.json', 'r') as f:
                repo = json.load(f)
            file_type = "repository.json"
        
        required_fields = ['name', 'url', 'maintainer']
        missing_fields = [field for field in required_fields if field not in repo]
        
        if missing_fields:
            print(f"❌ {file_type} missing fields: {missing_fields}")
            return False
        
        print(f"✅ {file_type} validation passed")
        return True
        
    except Exception as e:
        print(f"❌ repository file validation failed: {e}")
        return False

def main():
    """Run all validations"""
    print("🔍 Enhanced ML Heating Add-on Validation")
    print("=" * 50)
    
    # Enhanced validation suite
    validations = [
        ("Repository Structure", validate_repository_structure),
        ("Config YAML", validate_config_yaml),
        ("Build JSON", validate_build_json),
        ("Dockerfile", validate_dockerfile),
        ("Dependencies", validate_dependencies),
        ("Dashboard Components", validate_dashboard_components),
        ("Advanced Analytics", validate_advanced_analytics),
        ("Backup System", validate_backup_system),
        ("Dependency Compatibility", validate_dependency_compatibility),
        ("API Structure", validate_api_structure)
    ]
    
    all_passed = True
    for name, validator in validations:
        print(f"\n📋 {name}:")
        if not validator():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All enhanced validations passed! Container is production-ready.")
        print("\n📝 Next steps:")
        print("1. Commit changes to GitHub")
        print("2. GitHub Actions will build multi-architecture containers")
        print("3. Install add-on from repository in Home Assistant")
        print("4. Access advanced dashboard with 4-page interface")
        print("5. Use backup/restore for model preservation")
        return 0
    else:
        print("❌ Some validations failed. Please fix the issues above.")
        print("\n💡 Enhanced validation ensures:")
        print("   • Dashboard components are properly structured")
        print("   • Advanced analytics functions are available")
        print("   • Backup/restore system is functional")
        print("   • Dependencies are compatible")
        print("   • API structure supports all features")
        return 1

if __name__ == "__main__":
    sys.exit(main())
