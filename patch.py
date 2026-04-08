import sys

with open('src/model_wrapper.py', 'r') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "temp_diff = target_indoor - current_indoor" in line and "if temp_diff > 0.3:" in lines[i+1]:
        start_idx = i
        for j in range(i, len(lines)):
            if "logging.debug(\"   Dynamic Horizon: 4.0h (Stability)\")" in lines[j]:
                end_idx = j
                break
        break

if start_idx != -1 and end_idx != -1:
    indent = lines[start_idx][:len(lines[start_idx]) - len(lines[start_idx].lstrip())]
    
    new_lines = [
        f"{indent}temp_diff = target_indoor - current_indoor\n",
        f"{indent}if temp_diff > 0.3:\n",
        f"{indent}    # Cold (>0.3°C gap): Focus on next 1.0 hour for aggressive\n",
        f"{indent}    # recovery. This prevents the \"morning drop\" where the\n",
        f"{indent}    # system coasts on predicted solar gain despite being cold.\n",
        f"{indent}    optimization_horizon = 1.0\n",
        f"{indent}    logging.debug(\n",
        f"{indent}        \"   Dynamic Horizon: 1.0h (Aggressive Heating)\"\n",
        f"{indent}    )\n",
        f"{indent}elif temp_diff > 0.0:\n",
        f"{indent}    # Cool (>0.0°C gap): Focus on next 2.0 hours for moderate\n",
        f"{indent}    # recovery. Any deficit should prioritize near-term target\n",
        f"{indent}    # achievement over long-term coasting.\n",
        f"{indent}    optimization_horizon = 2.0\n",
        f"{indent}    logging.debug(\n",
        f"{indent}        \"   Dynamic Horizon: 2.0h (Moderate Heating)\"\n",
        f"{indent}    )\n",
        f"{indent}elif temp_diff < -0.2:\n",
        f"{indent}    # Hot (<-0.2°C gap): Focus on next 1.0 hour for aggressive\n",
        f"{indent}    # cooling. This prevents the system from coasting on a 4h\n",
        f"{indent}    # horizon when it needs to actively cool down now.\n",
        f"{indent}    optimization_horizon = 1.0\n",
        f"{indent}    logging.debug(\n",
        f"{indent}        \"   Dynamic Horizon: 1.0h (Aggressive Cooling)\"\n",
        f"{indent}    )\n",
        f"{indent}else:\n",
        f"{indent}    # Maintenance (Within -0.2 to 0.0): Focus on 4.0h for maximum stability\n",
        f"{indent}    optimization_horizon = 4.0\n",
        f"{indent}    logging.debug(\"   Dynamic Horizon: 4.0h (Stability)\")\n"
    ]
    
    lines[start_idx:end_idx+1] = new_lines
    
    with open('src/model_wrapper.py', 'w') as f:
        f.writelines(lines)
    print("Successfully patched src/model_wrapper.py")
else:
    print("Could not find target block in src/model_wrapper.py")
    sys.exit(1)
