#!/bin/bash
# Setup script for NAIC Demonstrator
# Run this after cloning: ./setup.sh

set -e

echo "=== NAIC Demonstrator Setup ==="
echo ""

# Check if module system is available (NAIC VMs)
if command -v module &> /dev/null; then
    echo "Module system detected (NAIC VM)"
    echo "Loading Python module..."
    module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || echo "Using system Python"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "ERROR: Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "Python version OK"

# Check for GPU
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "GPU detected but nvidia-smi query failed"

    # Setup CUDA library symlinks if needed
    if [ ! -d "$HOME/cuda_link" ]; then
        echo "Setting up CUDA library symlinks..."
        mkdir -p $HOME/cuda_link
        ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 $HOME/cuda_link/libcuda.so.1 2>/dev/null || true
        ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 $HOME/cuda_link/libcuda.so 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH=$HOME/cuda_link:$LD_LIBRARY_PATH
else
    echo "No NVIDIA GPU detected (CPU-only mode)"
fi

# Create virtual environment
echo ""
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    echo "Reusing existing venv. To recreate: rm -rf venv && ./setup.sh"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the demonstrator:"
echo "  jupyter lab"
echo ""
