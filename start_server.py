#!/usr/bin/env python3
"""Simple script to start the FastAPI server"""

import os
import sys
import subprocess

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Change to the multi_drone_rescue directory
    os.chdir(os.path.join(os.path.dirname(__file__), "multi_drone_rescue"))
    
    # Run uvicorn with the server
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "server:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ]
    
    print("Starting FastAPI server...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)
