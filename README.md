# ELitho: An Open-Source High-NA EUV Lighography Simulator
Python-based EUV simulation tool for lithography process development and research.


## Installation
This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone git@github.com:takahashi-edalab/elitho.git
cd elitho

# Install dependencies
uv sync
```

### Running the GUI
```bash
uv run streamlit run elitho/gui.py
```


## License
See [LICENSE](LICENSE) for details.
