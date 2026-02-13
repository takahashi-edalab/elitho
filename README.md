# ELitho: An Open-Source High-NA EUV Lithography Simulator
Python-based EUV simulation tool for lithography process development and research.


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

### Running the GUI
After installation, you can launch the GUI with:
```bash
elitho-gui
```

Or if using uv:
```bash
uv run elitho-gui
```


## License
See [LICENSE](LICENSE) for details.
