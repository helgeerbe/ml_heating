#!/usr/bin/env python3
"""
Script to fix all predict_equilibrium_temperature calls in test files to include current_indoor parameter.
This fixes the function signature changes made in Phase 5 of the thermal model fix.
"""

import re
import os
import glob

def fix_predict_equilibrium_calls(file_path):
    """Fix predict_equilibrium_temperature calls to include current_indoor parameter."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match predict_equilibrium_temperature calls without current_indoor
    # This matches calls like: predict_equilibrium_temperature(outlet_temp, outdoor_temp, ...)
    pattern = r'predict_equilibrium_temperature\(\s*([^,)]+),\s*([^,)]+),\s*([^)]*)\)'
    
    def replace_call(match):
        outlet_temp = match.group(1)
        outdoor_temp = match.group(2)
        remaining_args = match.group(3)
        
        # Check if current_indoor is already present
        if 'current_indoor' in remaining_args:
            return match.group(0)  # Return unchanged
        
        # Add current_indoor=20.0 as a default reasonable indoor temperature
        if remaining_args.strip():
            # There are other arguments
            new_call = f'predict_equilibrium_temperature(outlet_temp={outlet_temp}, outdoor_temp={outdoor_temp}, current_indoor=20.0, {remaining_args})'
        else:
            # No other arguments
            new_call = f'predict_equilibrium_temperature(outlet_temp={outlet_temp}, outdoor_temp={outdoor_temp}, current_indoor=20.0)'
        
        return new_call
    
    # Apply the fix
    new_content = re.sub(pattern, replace_call, content)
    
    # Write back if changed
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Fixed {file_path}")
        return True
    
    return False

def main():
    """Fix all test files."""
    test_files = glob.glob('tests/test_*.py')
    
    # Exclude files that are already fixed
    exclude_files = ['tests/test_thermal_physics.py']
    test_files = [f for f in test_files if f not in exclude_files]
    
    fixed_count = 0
    
    for test_file in test_files:
        if fix_predict_equilibrium_calls(test_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} test files")
    print("Running tests to verify fixes...")
    
    # Run the tests to verify
    os.system('python -m pytest tests/ --tb=no -q')

if __name__ == '__main__':
    main()
