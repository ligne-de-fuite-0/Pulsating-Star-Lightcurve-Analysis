# Pulsating Star Lightcurve Analysis

A Python toolkit for simulating, visualizing, and analyzing variable star lightcurves with machine learning-based anomaly detection.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project provides tools for:
- Simulating TESS-like lightcurves for various pulsating variable stars
- Implementing physics-based non-linear transformations
- Interactive visualization of lightcurves and frequency spectra
- Anomaly detection using GRU autoencoders

## Features

- **Physics-Based Simulation**: Generate realistic lightcurves for Cepheid, RR Lyrae, δ Scuti, and β Cephei variables
- **Interactive Visualization**: Plot lightcurves and spectra with Plotly
- **Machine Learning**: GRU autoencoder for anomaly detection
- **Astronomy Standards**: FITS file I/O compatible with astronomical tools

## Installation

```bash
git clone https://github.com/ligne-de-fuite-0/pulsating-star-analysis.git
cd pulsating-star-analysis
pip install -r requirements.txt
```

## Quick Start

```python
from pulsating_stars.simulation import LightcurveSimulator
from pulsating_stars.visualization import plot_lightcurve
from pulsating_stars.ml import GRUAutoencoder

# Generate sample lightcurves
simulator = LightcurveSimulator()
normal_lc, anomaly_lc = simulator.generate_pair()

# Visualize
plot_lightcurve(normal_lc)

# Anomaly detection
model = GRUAutoencoder()
model.fit(normal_lc)
anomaly_score = model.predict(anomaly_lc)
```

## Project Structure

```
pulsating-star-analysis/
├── pulsating_stars/           # Main package
│   ├── __init__.py
│   ├── simulation.py          # Lightcurve simulation
│   ├── visualization.py      # Plotting functions
│   ├── ml/                    # Machine learning modules
│   │   ├── __init__.py
│   │   ├── autoencoder.py     # GRU autoencoder
│   │   └── utils.py           # ML utilities
│   └── utils/                 # General utilities
│       ├── __init__.py
│       ├── io.py              # File I/O functions
│       └── physics.py         # Physics constants/functions
├── examples/                  # Example scripts
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── data/                      # Sample data
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Documentation

- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Physics Background](docs/physics.md)
- [Examples](examples/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pulsating_star_analysis,
  author = {ligne-de-fuite-0},
  title = {Pulsating Star Lightcurve Analysis},
  year = {2025},
  url = {https://github.com/ligne-de-fuite-0/pulsating-star-analysis}
}
```

## Acknowledgments

- TESS Mission for inspiration and data format standards
- Astronomical community for variable star physics knowledge
- TensorFlow/Keras teams for deep learning framework
