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

## Git Workflow
- First push to a new branch: `git push -u origin feature/branch-name` to set up tracking
- Subsequent pushes: just use `git push` to update the same PR
- Always run `make format` before committing to ensure code passes style checks

## PR Reviews
- Check PR comments with: `gh pr view <PR_NUMBER> --repo PolicyEngine/policyengine-core`
- Get review comments with: `gh api repos/PolicyEngine/policyengine-core/pulls/<PR_NUMBER>/comments`
- Address all reviewer feedback before merging
- Add "Claude Code wrote this comment" to GitHub comments written by Claude
- Follow Clean Code principles when refactoring:
  - Keep functions small and focused on a single task
  - Avoid deeply nested conditionals and exceptions
  - Extract complex logic into well-named helper functions
  - Minimize duplication and optimize for readability
  - Use consistent error handling patterns
  - Avoid too many levels of indentation
  - Prefer clear, well-named functions over comments
  - Ensure 100% test coverage for error handling code
  - Separate concerns with dedicated helper functions

## Package Architecture
- **Parameter System**: Core framework for tax-benefit system parameters
  - Parameters organized in a hierarchical tree (accessible via dot notation)
  - Parameters can be scalar values or bracket-based scales
  - Supports time-varying values with date-based lookup
- **Errors**: Custom error classes in `policyengine_core/errors/` for specific failures
- **Entities**: Represents different units (person, household, etc.) in microsimulations
- **Variables**: Calculations that can be performed on entities
- **Testing**: Supports both Python tests and YAML-based test files
- **Country Template**: Reference implementation of a tax-benefit system