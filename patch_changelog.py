import sys

with open('CHANGELOG.md', 'r') as f:
    content = f.read()

search_text = """- Fixed pre-check logic in `model_wrapper.py` to prevent short-circuiting to maximum heating when the room is already above the target temperature.
- Fixed pre-check logic in `model_wrapper.py` to prevent short-circuiting to maximum heating when the room is already above the target temperature."""

replace_text = """- Fixed pre-check logic in `model_wrapper.py` to prevent short-circuiting to maximum heating when the room is already above the target temperature.
- Fixed unexpected outlet temperature jumps (e.g., 14°C to 43°C) during active cooling by introducing a symmetrical dynamic optimization horizon. The system now uses a 1.0h "Aggressive Cooling" horizon when the room is significantly above the target temperature, prioritizing immediate temperature reduction over 4-hour stability."""

if search_text in content:
    content = content.replace(search_text, replace_text)
    with open('CHANGELOG.md', 'w') as f:
        f.write(content)
    print("Successfully patched CHANGELOG.md")
else:
    print("Could not find search text in CHANGELOG.md")
    sys.exit(1)
