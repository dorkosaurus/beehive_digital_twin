#!/bin/bash

# GPU Machine Setup Script for Digital Twin Beehive Project
# Command-line focused setup for vast.ai, RunPod, or other GPU instances
# Usage: bash setup_gpu_machine.sh

echo "=================================================="
echo "Digital Twin Beehive - GPU Machine Setup"
echo "Setting up environment for biological intelligence research"
echo "=================================================="

# Exit on any error
set -e

# Function to print status messages
print_status() {
    echo ""
    echo "🔧 $1"
    echo "----------------------------------------"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_status "Checking system information"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"

# Check for GPU
if command_exists nvidia-smi; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected. This script is designed for GPU instances."
fi

print_status "Updating system packages"
sudo apt update && sudo apt upgrade -y

print_status "Installing system dependencies"
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    wget \
    htop \
    tree \
    unzip \
    vim \
    nano \
    tmux \
    build-essential \
    software-properties-common

# Install nvtop for GPU monitoring if not already present
if command_exists nvidia-smi && ! command_exists nvtop; then
    print_status "Installing nvtop for GPU monitoring"
    sudo apt install -y nvtop || echo "⚠️  Could not install nvtop, skipping..."
fi

print_status "Setting up Python virtual environment"
python3 -m venv ~/beehive_env
source ~/beehive_env/bin/activate

# Upgrade pip
pip install --upgrade pip

print_status "Installing PyTorch with CUDA support"
# Install PyTorch with CUDA 12.1 support (adjust version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

print_status "Installing scientific computing libraries"
pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy

print_status "Installing additional dependencies for biological intelligence demo"
pip install \
    requests \
    psutil \
    tqdm \
    pillow

# Try to install GPUtil, but don't fail if it doesn't work
pip install GPUtil || echo "⚠️  Could not install GPUtil, continuing..."

print_status "Testing PyTorch GPU installation"
python3 -c "
import torch
import torchvision

print('✓ PyTorch version:', torch.__version__)
print('✓ Torchvision version:', torchvision.__version__)
print('✓ CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('✓ CUDA version:', torch.version.cuda)
    print('✓ GPU device count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        gpu_props = torch.cuda.get_device_properties(i)
        memory_gb = gpu_props.total_memory / 1024**3
        print(f'✓ GPU {i}: {gpu_props.name}, {memory_gb:.1f}GB VRAM')
else:
    print('❌ CUDA not available - check GPU drivers')
    exit(1)
"

print_status "Creating project directory structure"
mkdir -p ~/beehive_dt/{src/validation,results,logs}

print_status "Setting up environment activation script"
cat > ~/activate_beehive.sh << 'EOF'
#!/bin/bash
# Activate beehive environment and set up paths
source ~/beehive_env/bin/activate
cd ~/beehive_dt
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
echo "🐝 Beehive environment activated!"
echo "📁 Working directory: $(pwd)"
echo "🐍 Python: $(which python3)"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "🎮 GPUs available:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        echo "   $line"
    done
fi
EOF

chmod +x ~/activate_beehive.sh

print_status "Creating useful aliases and shortcuts"
cat >> ~/.bashrc << 'EOF'

# Digital Twin Beehive aliases
alias beehive='source ~/activate_beehive.sh'
alias gpu='nvidia-smi'
alias gpuwatch='watch -n 1 nvidia-smi'
alias gpumem='nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
alias htop='htop -u $(whoami)'
alias ll='ls -la'
alias la='ls -A'
alias l='ls -CF'
EOF

print_status "Creating GPU test script"
cat > ~/beehive_dt/test_gpu_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test to verify GPU setup is working correctly
Run: python3 test_gpu_setup.py
"""
import torch
import time
import sys

def test_gpu_setup():
    print("=== GPU Setup Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_props = torch.cuda.get_device_properties(i)
        memory_gb = gpu_props.total_memory / 1024**3
        print(f"GPU {i}: {gpu_props.name}, {memory_gb:.1f}GB VRAM")
    
    # Test basic tensor operations
    device = torch.device('cuda:0')
    print(f"\n=== Testing GPU Operations ===")
    
    # Create test tensors
    print("Creating test tensors...")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Time matrix multiplication
    print("Testing matrix multiplication...")
    start_time = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"GPU matrix multiply (1000x1000): {gpu_time:.4f} seconds")
    print(f"Result shape: {z.shape}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # Test mixed precision
    print(f"\n=== Testing Mixed Precision ===")
    try:
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            result = torch.matmul(x.half(), y.half())
        print(f"✓ Mixed precision operations working")
    except Exception as e:
        print(f"⚠️  Mixed precision test failed: {e}")
    
    print(f"✓ All GPU tests passed!")
    return True

if __name__ == "__main__":
    success = test_gpu_setup()
    if not success:
        sys.exit(1)
EOF

print_status "Creating requirements.txt file"
cat > ~/beehive_dt/requirements.txt << 'EOF'
# Core ML and Scientific Computing
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# System and API
requests>=2.31.0
psutil>=5.9.0
tqdm>=4.65.0

# Image processing
pillow>=10.0.0
EOF

print_status "Creating quick start script"
cat > ~/beehive_dt/quick_start.sh << 'EOF'
#!/bin/bash
# Quick start script for running the beehive digital twin analysis

echo "🐝 Digital Twin Beehive - Quick Start"
echo "===================================="

# Activate environment
source ~/beehive_env/bin/activate

# Test GPU setup
echo "Testing GPU setup..."
python3 test_gpu_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GPU setup successful!"
    echo ""
    echo "Available commands:"
    echo "  python3 src/validation/gpu_scaling_testing.py     # Run GPU scaling analysis"
    echo "  python3 src/validation/v0_digital_twin.py         # Run biological modeling"
    echo "  python3 src/validation/validation_viz.py          # Generate visualizations"
    echo ""
    echo "Monitor GPU usage:"
    echo "  nvidia-smi              # Current GPU status"
    echo "  watch -n 1 nvidia-smi   # Real-time GPU monitoring"
    echo "  nvtop                   # Interactive GPU monitoring (if installed)"
    echo ""
    echo "Ready to run biological intelligence experiments! 🚀"
else
    echo "❌ GPU setup failed. Check the error messages above."
fi
EOF

chmod +x ~/beehive_dt/quick_start.sh

print_status "Final system check"
source ~/beehive_env/bin/activate
cd ~/beehive_dt

echo "Running final GPU test..."
python3 test_gpu_setup.py

echo ""
echo "=================================================="
echo "🎉 Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Run: source ~/activate_beehive.sh"
echo "2. Or run: ~/beehive_dt/quick_start.sh"
echo "3. Upload your project files to ~/beehive_dt/"
echo "4. Run your GPU scaling and biological intelligence analysis!"
echo ""
echo "Useful commands:"
echo "  beehive                    # Activate environment (after adding to PATH)"
echo "  source ~/activate_beehive.sh  # Activate environment"
echo "  nvidia-smi                 # Check GPU status"
echo "  cd ~/beehive_dt && python3 test_gpu_setup.py  # Test installation"
echo ""
echo "Project directory: ~/beehive_dt/"
echo "Environment: ~/beehive_env/"
echo ""
echo "Happy researching! 🐝🤖"
