# elitho

Python-based EUV simulation tool for lithography process development and research.

## Features

- EUV lithography simulation with vector diffraction theory
- TCC (Transmission Cross Coefficient) matrix calculation
- Electric field and intensity computation
- Streamlit-based interactive GUI
- Support for various mask patterns and illumination sources

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd elitho

# Install dependencies
uv sync
```

## Usage

### Running the GUI

```bash
uv run streamlit run elitho/gui.py
```

### Python API

```python
from elitho import stcc, config
import numpy as np

# Create a mask pattern
mask = np.ones((config.NDIVX, config.NDIVY))

# Calculate intensity
intensity = stcc.intensity(mask, config.PolarizationDirection.X)
```

## Requirements

- Python >=3.12
- numpy >=2.4.2
- streamlit >=1.54.0
- matplotlib >=3.10.8

## License

See [LICENSE](LICENSE) for details.
