# ELitho: An Open-Source High-NA EUV Lithography Simulator
Python-based EUV simulation tool for lithography process development and research.

## Features
- **High-NA EUV simulation**: Support for NA 0.33 and 0.55 systems
- **Multiple illumination types**: Circular, Annular, Dipole (X/Y)
- **Multi-layer absorber modeling**: Configurable absorber stack with complex refractive indices
- **Interactive GUI**: Streamlit-based interface for parameter configuration and visualization
- **Defocus analysis**: Single-point or sweep mode with visualization
- **Parameter persistence**: Save/load simulation configurations as JSON

## Installation

### Using pip
```bash
# Clone the repository
git clone git@github.com:takahashi-edalab/elitho.git
cd elitho

# Install the package
pip install -e .
```

### Using uv (recommended)
This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone git@github.com:takahashi-edalab/elitho.git
cd elitho

# Install dependencies
uv sync

# Install the package
uv pip install -e .
```

## Usage

### Running the GUI
After installation, you can launch the GUI with:
```bash
elitho-gui
```

Or if using uv:
```bash
uv run elitho-gui
```

### Using as a Python library
```python
from elitho import config, intensity
import numpy as np

# Create simulation configuration
sc = config.SimulationConfig(
    wavelength=13.5,  # nm
    NA=0.33,
    illumination=config.CircularIllumination(outer_sigma=0.9),
    absorber_layers=config.AbsorberLayers(
        thicknesses=[60.0],  # nm
        complex_refractive_indices=[0.9567 + 0.0343j]
    ),
    mask_width=1024,  # nm
    mask_height=1024,  # nm
)

# Print configuration
print(sc)

# Generate mask pattern
mask = np.zeros((1024, 1024), dtype=np.uint8)
mask[400:600, 400:600] = 1  # 200x200 nm opening

# Run simulation
result = intensity.intensity(
    sc,
    mask,
    config.PolarizationDirection.X,
    defocus=0.0  # nm
)
```

## Project Structure
```
elitho/
├── elitho/
│   ├── absorber.py          # Absorber layer calculations
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration classes
│   ├── diffraction_order.py # Diffraction order handling
│   ├── fourier.py           # Fourier transform utilities
│   ├── gui.py               # Streamlit GUI
│   ├── intensity.py         # Main intensity calculation
│   ├── mask.py              # Mask-related functions
│   ├── source.py            # Illumination source generation
│   └── vector_potential.py  # Vector potential calculations
├── pyproject.toml           # Project metadata and dependencies
└── README.md
```

## License
See [LICENSE](LICENSE) for details.
