# Contributing to Pulsating Star Analysis

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/pulsating-star-analysis.git
   cd pulsating-star-analysis
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Use `black` for code formatting: `black .`
- Use `isort` for import sorting: `isort .`
- Run `flake8` for linting: `flake8 .`

## Testing

- Write tests for new functionality in the `tests/` directory
- Run tests with: `pytest`
- Aim for good test coverage: `pytest --cov=pulsating_stars`

## Documentation

- Update docstrings for any new or modified functions
- Add examples to the `examples/` directory
- Update README.md if adding new features

## Submitting Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature-name
   ```

4. Create a pull request on GitHub

## Areas for Contribution

- **New variable star types**: Add support for other pulsating stars
- **ML models**: Implement other anomaly detection algorithms
- **Visualization**: Enhance plotting capabilities
- **Documentation**: Improve tutorials and examples
- **Performance**: Optimize computational efficiency
- **Testing**: Increase test coverage

## Questions?

Feel free to open an issue for discussion before starting work on major features.