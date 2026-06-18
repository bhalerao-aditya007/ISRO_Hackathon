import os
import glob
import subprocess

# Move files in git
subprocess.run(["git", "mv", "core/base_agent.py", "agents/base_agent.py"])
subprocess.run(["git", "mv", "core/protocol.py", "agents/protocol.py"])

# Find all python files
py_files = glob.glob("agents/*.py") + ["orchestrator.py", "test_agent1.py", "test_agent2.py", "test_agent3.py"]

for f in py_files:
    if not os.path.exists(f): continue
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Replace the old 'core' imports with 'agents'
    content = content.replace("from core.base_agent import", "from agents.base_agent import")
    content = content.replace("from core.protocol import", "from agents.protocol import")
    
    with open(f, "w", encoding="utf-8") as file:
        file.write(content)

print("Moved base_agent.py and protocol.py to agents/ folder and updated all imports!")
