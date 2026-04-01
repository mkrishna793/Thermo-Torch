# Contributing to ThermoTorch

Thank you for your interest in contributing to ThermoTorch! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/mkrishna793/thermotorch/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (Python version, OS, etc.)

### Suggesting Features

1. Open an issue with the "enhancement" label
2. Describe the feature and why it would be useful
3. Include examples of how it would be used

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest thermotorch/tests/ -v`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/thermotorch.git
cd thermotorch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest thermotorch/tests/ -v
```

## Coding Standards

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for all public functions and classes

### Testing

- All new features must have tests
- Tests should be in `thermotorch/tests/`
- Run tests before submitting PR: `pytest thermotorch/tests/ -v`

### Documentation

- Update docstrings for modified functions
- Update README.md if needed
- Add examples for new features

## Project Structure

```
thermotorch/
├── core/           # Core components (PFE, loss, settling)
├── backends/       # Hardware backends (CPU, THRML, TSU)
├── examples/       # Usage examples
├── tests/          # Test files
└── docs/           # Documentation
```

## Testing

```bash
# Run all tests
pytest thermotorch/tests/ -v

# Run specific test file
pytest thermotorch/tests/test_pfe_core.py -v

# Run with coverage
pytest thermotorch/tests/ -v --cov=thermotorch
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Contact

- **Author**: N. Mohana Krishna
- **GitHub**: [@mkrishna793](https://github.com/mkrishna793)
- **Related**: [Extropic AI](https://github.com/extropic-ai)

---

Thank you for contributing! 🎉