"""
Run script for Supply Chain AI System
Run this from the root directory: python run.py
"""

import sys
import os

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Now import and run main
from src.main_simple import main

if __name__ == "__main__":
    main()