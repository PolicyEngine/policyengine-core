# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.10.11] - 2022-12-07 12:51:49

### Fixed

- Include scale brackets in descendants.

## [1.10.10] - 2022-12-05 16:21:04

### Fixed

- Removed warning on Enum decoding.

## [1.10.9] - 2022-12-03 15:21:15

### Fixed

- Bug with Enum decoding.

## [1.10.8] - 2022-12-02 18:11:10

### Fixed

- Singleton files have a shorter moduleName.

## [1.10.7] - 2022-11-30 20:33:21

### Fixed

- Various safety checks.

## [1.10.6] - 2022-11-30 15:19:56

### Fixed

- README.md files in parameter trees now inform metadata.

## [1.10.5] - 2022-11-28 10:37:23

### Added

- Functionality for getting all known branch-period caches.

## [1.10.4] - 2022-11-27 16:16:29

### Fixed

- Removed debug lines.

## [1.10.3] - 2022-11-27 16:08:33

### Fixed

- Fixed a bug causing Windows tests to fail.

## [1.10.2] - 2022-11-27 14:33:53

### Fixed

- Bug in defined-for causing it to be ignored.
- Branched simulations now cache variables specifically for the branch and time period, not sharing with the parent simulation.

## [1.10.1] - 2022-11-24 00:33:05

### Added

- Documentation page on reforms.

### Fixed

- Moved the uprating logic to after formulas are checked.

## [1.10.0] - 2022-11-24 00:15:28

### Added

- Variables can now specify an 'uprating' parameter.
- Variables can now specify 'adds' and 'subtracts' variable dependencies.
- Variable folders are now automatically stored in a tree.

### Fixed

- Bugs in cloned simulations.

## [1.9.0] - 2022-11-08 15:00:26

### Fixed

- Various fixes for PolicyEngine's server app.

## [1.8.1] - 2022-11-02 01:31:48

## [1.8.0] - 2022-10-23 13:41:39

### Changed

- Parameter scale components now included in `get_descendants()`.

### Fixed

- Folder paths in `Dataset`s are parsed as `Path`s.
- Empty `input:` fields are parsed as `{}`.

## [1.7.0] - 2022-10-21 12:39:50

### Added

- Metadata dictionary input to Variables.

## [1.6.0] - 2022-10-19 22:44:19

### Added

- Derivative calculation in simulations.

## [1.5.0] - 2022-10-19 16:01:51

### Added

- Branched simulations.

## [1.4.1] - 2022-10-18 18:13:09

### Added

- Plotly to dependencies

## [1.4.0] - 2022-10-18 15:28:22

### Added

- Performance improvements
- Import sorting and code health improvements

### Fixed

- Axes now work in simulations

## [1.3.0] - 2022-10-14 17:18:21

### Added

- Tests and documentation.

## [1.2.0] - 2022-10-14 10:14:40

### Added

- Parameter operations from openfisca-tools
- Dataset classes from openfisca-tools

## [1.1.3] - 2022-10-12 18:08:43

### Fixed

- Git blame reassigns OpenFisca-Core authors.

## [1.1.2] - 2022-10-12 16:25:33

### Fixed

- Bug in the country template microsimulation class failing to load data.

## [1.1.1] - 2022-10-12 15:58:34

### Added

- Behaviour to handle default datasets.

## [1.1.0] - 2022-10-12 15:33:59

### Added

- Microsimulation features
- Documentation for the CLI interface

## [1.0.6] - 2022-10-09 23:00:35

### Fixed

- Python 3.9 used in all GitHub actions.

## [1.0.5] - 2022-10-09 22:55:14

### Added

- Test that documentation builds

## [1.0.4] - 2022-10-09 22:33:24

### Added

- Type hints to most functions and classes.
- Jupyter book documentation for all modules.

## [1.0.3] - 2022-10-02 16:59:49

### Fixed

- Build action always runs.

## [1.0.2] - 2022-10-02 16:45:22

### Fixed

- Push action doesn't attempt to publish nonexistent documentation.

## [1.0.1] - 2022-10-02 16:38:34

### Fixed

- Version update action.

## [1.0.0] - 2022-10-02 16:35:17

### Changed

- Non-essential (for Python simulations) code removed.

## [0.1.0] - 2022-10-02 16:02:00

### Added

- OpenFisca-Core forked.



[1.10.11]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.10...1.10.11
[1.10.10]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.9...1.10.10
[1.10.9]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.8...1.10.9
[1.10.8]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.7...1.10.8
[1.10.7]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.6...1.10.7
[1.10.6]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.5...1.10.6
[1.10.5]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.4...1.10.5
[1.10.4]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.3...1.10.4
[1.10.3]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.2...1.10.3
[1.10.2]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.1...1.10.2
[1.10.1]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.0...1.10.1
[1.10.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.9.0...1.10.0
[1.9.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.8.1...1.9.0
[1.8.1]: https://github.com/PolicyEngine/policyengine-core/compare/1.8.0...1.8.1
[1.8.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.7.0...1.8.0
[1.7.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.6.0...1.7.0
[1.6.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.5.0...1.6.0
[1.5.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.4.1...1.5.0
[1.4.1]: https://github.com/PolicyEngine/policyengine-core/compare/1.4.0...1.4.1
[1.4.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.3.0...1.4.0
[1.3.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.2.0...1.3.0
[1.2.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.1.3...1.2.0
[1.1.3]: https://github.com/PolicyEngine/policyengine-core/compare/1.1.2...1.1.3
[1.1.2]: https://github.com/PolicyEngine/policyengine-core/compare/1.1.1...1.1.2
[1.1.1]: https://github.com/PolicyEngine/policyengine-core/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.0.6...1.1.0
[1.0.6]: https://github.com/PolicyEngine/policyengine-core/compare/1.0.5...1.0.6
[1.0.5]: https://github.com/PolicyEngine/policyengine-core/compare/1.0.4...1.0.5
[1.0.4]: https://github.com/PolicyEngine/policyengine-core/compare/1.0.3...1.0.4
[1.0.3]: https://github.com/PolicyEngine/policyengine-core/compare/1.0.2...1.0.3
[1.0.2]: https://github.com/PolicyEngine/policyengine-core/compare/1.0.1...1.0.2
[1.0.1]: https://github.com/PolicyEngine/policyengine-core/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/PolicyEngine/policyengine-core/compare/0.1.0...1.0.0

