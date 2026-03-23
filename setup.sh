#!/bin/bash

echo "Setting up Supply Chain AI System..."

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric dependencies
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-geometric

# Install other requirements
pip install -r requirements.txt

echo "Setup complete! Run: python src/main.py"