# PolicyEngine Core

[![codecov](https://codecov.io/gh/PolicyEngine/policyengine-core/branch/master/graph/badge.svg?token=BLoCjCf5Qr)](https://codecov.io/gh/PolicyEngine/policyengine-core)
[![PyPI version](https://badge.fury.io/py/policyengine-core.svg)](https://badge.fury.io/py/policyengine-core)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A fork of [OpenFisca-Core](https://github.com/OpenFisca/OpenFisca-Core) that powers PolicyEngine country models (`policyengine-uk`, `policyengine-us`, `policyengine-canada`, `policyengine-il`, `policyengine-ng`) and the downstream apps and APIs. Provides the simulation engine, parameter tree, variable system, and reform machinery that each country package builds on top of.

## Install

```bash
pip install policyengine-core
```

Supports Python 3.9 – 3.14.

## Usage

Most users reach this package transitively through a country model. Import directly if you're building a new country model, writing a reform, or extending the core:

```python
from policyengine_core.simulations import Simulation
from policyengine_core.reforms import Reform
```

See the [documentation](https://policyengine.github.io/policyengine-core/) for API reference and tutorials.

## Development

```bash
# Install dev dependencies (uses uv)
make install

# Run the full test suite
make test

# Run a single test
uv run pytest tests/core/test_file.py::test_name -v

# Format before committing (CI enforces)
make format
```

**Always use `uv run`** for Python commands; bare `python` / `pytest` / `pip` may pick up the wrong environment.

Contributions: see [CONTRIBUTING.md](./CONTRIBUTING.md) for branching, PR conventions, and the towncrier `changelog.d/` workflow. Default branch is `master`.

## License

Distributed under the AGPL License. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

- Forked from [OpenFisca-Core](https://github.com/OpenFisca/OpenFisca-Core).
- README template adapted from Othneil Drew's [Best-README-Template](https://github.com/othneildrew/Best-README-Template).
