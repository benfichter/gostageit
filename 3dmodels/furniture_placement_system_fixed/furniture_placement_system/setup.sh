#!/bin/bash

# Furniture Placement System - Setup Script
# ==========================================

echo "====================================="
echo "Furniture Placement System Setup"
echo "====================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing core requirements..."
pip install -r requirements.txt

# Install MoGe
echo ""
echo "Installing MoGe model..."
pip install git+https://github.com/Ruicheng/moge.git

# Optional: Install SAM for furniture detection
read -p "Do you want to install SAM for existing furniture detection? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing Segment Anything..."
    pip install 'git+https://github.com/facebookresearch/segment-anything.git'
    
    echo ""
    echo "NOTE: You'll need to download a SAM checkpoint:"
    echo "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    echo "  export SAM_CHECKPOINT_PATH=sam_vit_h_4b8939.pth"
fi

# Create output directory
mkdir -p outputs

echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the system:"
echo "  python main.py --image <your_room_image.jpg>"
echo ""
echo "Example:"
echo "  python main.py --image test_room.jpg --ceiling-height 2.7"
echo ""
