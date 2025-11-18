# Furniture Placement System

Automatically furnish empty rooms using 3D analysis, physics simulation, and interior design rules.

## Quick Start

```bash
# 1. Setup
bash setup.sh  # Or: pip install -r requirements.txt

# 2. Run
python main.py --image your_room.jpg

# 3. Check outputs/ folder for results
```

## Installation

### Option 1: Automatic Setup (Linux/Mac)
```bash
bash setup.sh
source venv/bin/activate
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install MoGe
pip install git+https://github.com/Ruicheng/moge.git

# Optional: Install SAM for furniture detection
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Usage Examples

```bash
# Basic usage
python main.py --image room.jpg

# Specify room height (default 2.4m)
python main.py --image room.jpg --ceiling-height 2.7

# Use 3D rendering
python main.py --image room.jpg --use-3d

# Custom output folder  
python main.py --image room.jpg --output my_results/

# Force CPU (if no GPU)
python main.py --image room.jpg --device cpu
```

## How It Works

1. **MoGe Analysis**: Extracts 3D room geometry from single image
2. **Room Classification**: Determines room type (living room, bedroom, etc.)
3. **Furniture Selection**: Chooses appropriate furniture based on room size
4. **Physics Placement**: Uses PyBullet to ensure valid, collision-free placement
5. **Design Rules**: Applies ergonomic standards and interior design principles
6. **Visualization**: Creates multiple views of the furnished room

## Output Files

```
outputs/placement_TIMESTAMP/
├── room_analysis.json      # Room dimensions and geometry
├── placements.json         # Furniture positions and sizes
├── placement_topdown.png   # Bird's eye view
├── placement_2d.png        # 2D visualization
├── annotated_dimensions.png # With measurements
└── summary.txt            # Human-readable report
```

## System Requirements

- Python 3.8+
- 4GB RAM minimum
- GPU recommended (but not required)
- ~2GB disk space for models

## Troubleshooting

**GPU/CUDA Issues:**
```bash
python main.py --image room.jpg --device cpu
```

**Missing MoGe Model:**
```bash
pip install --upgrade git+https://github.com/Ruicheng/moge.git
```

**Import Errors:**
```bash
pip install --upgrade -r requirements.txt
```

## License

MIT
