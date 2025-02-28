# PolicyEngine Core - Development Guide 📚

## Common Commands
- `make all`: Full development cycle (install, format, test, build)
- `make install`: Install package in dev mode with dependencies
- `make format`: Format code with Black (79 char line limit)
- `make test`: Run all tests with coverage
- `make mypy`: Run type checking
- `pytest tests/path/to/test_file.py::test_function_name -v`: Run single test

## Code Style Guidelines
- **Formatting**: Black with 79 character line length
- **Imports**: Standard library, third-party, local modules (standard Python grouping)
- **Types**: Use type annotations, checked with mypy
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Error Handling**: Use custom error classes from policyengine_core/errors/
- **Testing**: All new features need tests, both unit and YAML-based tests supported
- **PRs**: Include "Fixes #{issue}" in description, add changelog entry, pass all checks
- **Changelog**: Update changelog_entry.yaml with proper bump type and change description
- **Dependencies**: All versions must be capped to avoid breaking changes
- **Python**: 3.10+ required