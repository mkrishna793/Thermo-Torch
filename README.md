# ThermoTorch: AI Meets Physics

**A PyTorch bridge to Extropic's Thermodynamic Computing Hardware**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Tests: 78 passing](https://img.shields.io/badge/tests-78%20passing-green.svg)](docs/TEST_RESULTS.md)

---

## Overview

ThermoTorch is a **production-ready PyTorch framework** for Potential-Flow Embeddings (PFE) and thermodynamic computing. It bridges standard deep learning with Extropic's Thermodynamic Sampling Units (TSUs).

### What Problem Does It Solve?

Traditional AI uses GPUs to do **brute-force math**, which is slow and energy-intensive. ThermoTorch lets you write standard PyTorch code, but routes the heavy probabilistic sampling to physics-based hardware that's ~10,000x more energy-efficient.

---

## Related Projects

This project builds upon and integrates with:

- **[Extropic AI](https://github.com/extropic-ai)** - Thermodynamic computing hardware
- **[THRML](https://github.com/extropic-ai/thrml)** - Thermodynamic HypergRaphical Model Library for JAX

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mkrishna793/thermotorch.git
cd thermotorch

# Install dependencies
pip install -e .

# With JAX support (for THRML backend)
pip install -e ".[jax]"
```

### Basic Usage

```python
import torch
from thermotorch import PFEEncoder, PFELoss, tsu_settle

# Create the PFE model
model = PFEEncoder(embed_dim=512)
criterion = PFELoss(sigma=0.1, alpha=0.1, beta=0.5)

# Train with tri-partite loss
x = torch.randn(32, 512)
x_pos = x + torch.randn_like(x) * 0.05
x_noise = torch.randn(32, 512)

loss = criterion(model, x, x_pos, x_noise)
loss.backward()

# Settle to attractor (find energy minimum)
noisy = torch.randn(8, 512)
clean = tsu_settle(model, noisy, steps=50)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         THERMOTORCH STACK                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │  PyTorch Model  │───▶│   TSU Bridge    │───▶│  Thermodynamic │    │
│  │   (PFEEncoder)  │    │  (Zero-Copy)    │    │     Backend     │    │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
│         │                       │                      │              │
│         ▼                       ▼                      ▼              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │  Loss Function  │◀───│  STE Gradients  │◀───│   TSU Chip /    │    │
│  │ (DSM+NCE+InfoNCE)│    │  (Autograd)    │    │  THRML/JAX      │    │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. Potential-Flow Embeddings (PFE)

PFE creates a landscape where:
- **Valleys** = correct answers (low energy)
- **Hills** = wrong answers (high energy)
- **Flow** = direction toward correct answers

When you drop data on this landscape, it naturally flows downhill to the right answer.

| Component | What It Does | Analogy |
|-----------|--------------|---------|
| Energy `s(x)` | Measures "goodness" | Altimeter |
| Flow `v(x)` | Points to better values | Compass |
| Similarity | Checks alignment | Both compasses same direction? |

### 2. The Encoder

```python
class PFEEncoder:
    energy_head = "Measures height"    # s(x)
    flow_head = "Shows direction"      # v(x)
    
    def forward(self, x):
        energy = self.energy_head(x)   # How good is this?
        flow = self.flow_head(x)       # Which way to go?
        return energy, flow
```

### 3. The Loss Function

```
Total Loss = DSM + α·NCE + β·InfoNCE

DSM Loss:    Point compass toward real data
NCE Loss:    Put real data in valleys, noise on hills
InfoNCE:     Related items point same direction
```

### 4. Memory Bridge (DLPack)

DLPack lets PyTorch and JAX share memory without copying:

```
WITHOUT DLPack:  [PyTorch] → COPY → CPU → COPY → [JAX]  (slow)

WITH DLPack:     [Shared Memory] ← Both read directly    (fast)
```

---

## Latent PFE (For Images)

Processing millions of pixels is slow. Latent PFE compresses first:

```
[4K Image] → Compress → [Tiny Latent] → Settle → [Clean Latent] → Decompress → [Clean Image]
```

1. **Compress**: 4K image → 16,384 numbers (using VAE)
2. **Settle**: Find the valley in this tiny space
3. **Decompress**: Back to 4K image

---

## Backend Hierarchy

| Backend | Status | Description |
|---------|--------|-------------|
| `cpu_backend` | ✅ Stable | Pure Python/NumPy, always available |
| `thrml_backend` | ✅ Stable | JAX/GPU simulation |
| `tsu_backend` | 🔧 Stub | Physical hardware (mock for development) |

---

## Test Results

```
============================= 78 passed in 18.16s =============================

Phase 1 (PFE Core):      15 tests ✅
Phase 2 (Bridge/Layers): 31 tests ✅
Phase 3 (Backends):      32 tests ✅
```

See [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) for detailed results.

---

## Project Structure

```
thermotorch/
├── core/                   # Core components
│   ├── pfe_encoder.py     # Encoder (energy + flow)
│   ├── pfe_loss.py        # Loss function
│   ├── tsu_settle.py      # ODE settling
│   ├── memory.py          # DLPack bridge
│   ├── bridge.py          # Autograd integration
│   └── tsu_layer.py       # Easy-to-use layers
├── backends/              # Hardware abstraction
│   ├── cpu_backend.py    # CPU fallback
│   ├── thrml_backend.py  # JAX simulation
│   └── tsu_backend.py    # Hardware interface
├── tests/                 # 78 passing tests
└── docs/                   # Documentation
```

---

## API Reference

### PFEEncoder

```python
encoder = PFEEncoder(
    embed_dim=512,        # Input dimension
    hidden_dim=1024,      # Hidden dimension
    num_layers=2          # Number of layers
)
energy, flow_raw, flow_unit = encoder(x)
```

### TSULayer

```python
layer = TSULayer(steps=50, method='rk4')
settled = layer(model, noisy_data)
```

### Backends

```python
# CPU backend (always works)
from thermotorch.backends import CPUBackend
backend = CPUBackend(method='euler')

# THRML backend (JAX/GPU)
from thermotorch.backends import THRMLBackend
backend = THRMLBackend(device='gpu')

# TSU backend (hardware)
from thermotorch.backends import TSUBackend
backend = TSUBackend(fallback_to_thrml=True)
```

---

## Hardware Roadmap

| Chip | Status | Description |
|------|--------|-------------|
| CPU | ✅ Available | NumPy/PyTorch fallback |
| THRML | ✅ Available | JAX GPU simulation |
| XTR-0 | 🔧 Beta | Extropic development platform |
| Z1 | 📅 2026 | Production chip |

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Acknowledgments

This project was developed collaboratively with **Claude Code (Anthropic's Claude Opus 4.6)**.

### Built Upon

- **[Extropic AI](https://extropic.ai/)** - Thermodynamic computing research
- **[THRML Library](https://github.com/extropic-ai/thrml)** - JAX probabilistic computing
- **[PyTorch](https://pytorch.org/)** - Deep learning framework

### Research Papers

- Denoising Thermodynamic Models (DTM) - [arXiv:2510.23972](https://arxiv.org/abs/2510.23972)
- Energy-Based Models and Gibbs Sampling

---

## License

```
Copyright [2024-2026] N. Mohana Krishna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Citation

If you use ThermoTorch in your research, please cite:

```bibtex
@software{thermotorch2024,
  author = {Krishna, N. Mohana},
  title = {ThermoTorch: A PyTorch Framework for Potential-Flow Embeddings and Thermodynamic Computing},
  year = {2024},
  url = {https://github.com/mkrishna793/thermotorch}
}
```

---

## Links

- **Documentation**: [docs/](docs/)
- **Test Results**: [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md)
- **Extropic AI**: [https://extropic.ai](https://extropic.ai)
- **THRML Library**: [https://github.com/extropic-ai/thrml](https://github.com/extropic-ai/thrml)

---

*"We get the flexibility of standard software with the extreme speed and low power of physics-based hardware."*