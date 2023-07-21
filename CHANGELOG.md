# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.0] - 2023-07-21 14:55:15

### Added

- Chart formatting utilities.

## [2.3.0] - 2023-06-30 08:22:27

### Added

- Optional argument for saving datasets to a specific file path.
- Safety check for Path objects not being strings.

## [2.2.1] - 2023-06-02 15:22:29

### Fixed

- A bug causing variable not found errors to sometimes fail to throw.

## [2.2.0] - 2023-05-31 11:11:09

### Fixed

- Bug causing unhelpful errors when variables are not found.

## [2.1.1] - 2023-05-24 15:11:39

### Fixed

- Fixed pytest deprecated usages since version 7, and fixed a bug that prevented new package from publishing.

## [2.1.0] - 2023-05-23 13:22:04

### Added

- Python 3.10 support.

### Fixed

- PyTest syntax updated.

## [2.0.4] - 2023-05-23 07:35:11

### Fixed

- Bug causing simulations containing multiple group entities and axes to fail to simulate.

## [2.0.3] - 2023-04-14 10:06:10

### Changed

- Removed node homogenuity check for speed.

## [2.0.2] - 2023-03-31 18:27:52

### Changed

- Added ability to pull datasets from GitHub releases.

## [2.0.1] - 2023-03-23 08:55:46

### Fixed

- A bug where public datasets would fail to download.

## [2.0.0] - 2023-03-23 02:37:29

### Changed

- Datasets API simplified and made more consistent.

## [1.12.4] - 2023-03-08 21:35:17

### Fixed

- Removed bad dependency fixes.

## [1.12.3] - 2023-03-04 22:58:51

### Fixed

- Branches stay persistent between variables.

## [1.12.2] - 2023-02-09 21:59:16

### Added

- Option for loading non-year-specific datasets.

## [1.12.1] - 2023-01-28 15:58:38

### Added

- Abolition parameters for each variable under `gov.abolitions`.

## [1.12.0] - 2023-01-27 22:28:10

### Added

- Likely high time savings for simulations making heavy usage of reforms and branches.

## [1.11.4] - 2023-01-11 23:44:12

### Fixed

- Labels and names aren't propagated in parameters.

## [1.11.3] - 2023-01-10 17:32:23

## [1.11.2] - 2023-01-04 13:05:49

### Added

- Error message for an unknown variable.

## [1.11.1] - 2023-01-04 12:51:00

### Added

- A metadata option to specify which policies are modelled and not modelled.

## [1.11.0] - 2023-01-03 20:50:02

### Added

- Adds/subtracts option for parameter names.

## [1.10.24] - 2023-01-02 20:40:49

### Changed

- Pin all package versions.

## [1.10.23] - 2022-12-18 06:12:52

### Fixed

- Fix incorrect type annotation in policyengine_core/variables/variable.py

## [1.10.22] - 2022-12-18 06:07:32

### Added

- Fixed mypy errors in several files.

## [1.10.21] - 2022-12-17 21:51:59

### Changed

- Removed seemingly unused "N_" function

## [1.10.20] - 2022-12-17 16:36:53

### Fixed

- Fixed mypy errors in policyengine_core/tracers/tracing_parameter_node_at_instant.py

## [1.10.19] - 2022-12-17 09:23:10

### Fixed

- Fixed mypy errors in tests/core/test_entities.py

## [1.10.18] - 2022-12-15 15:27:35

### Fixed

- Bugs with Enum inputs.

## [1.10.17] - 2022-12-14 15:33:46

### Added

- Added `basic_inputs`, a metadata field for the basic inputs needed to get a country model running.

## [1.10.16] - 2022-12-13 23:40:47

### Added

- Added mypy to our test suite to prevent new type regressions

## [1.10.15] - 2022-12-12 15:28:47

### Added

- Hidden input field for variables.

## [1.10.14] - 2022-12-11 23:52:13

### Fixed

- Added documentation on populating changelog entries. Also created a CONTRIBUTING.md symlink to make contributing guidelines easier to find.

## [1.10.13] - 2022-12-11 23:46:33

### Fixed

- Git config now ignores JetBrains IDE config files

## [1.10.12] - 2022-12-07 13:41:29

### Fixed

- Uprating bug affecting the UK.

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



[2.4.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.3.0...2.4.0
[2.3.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.2.1...2.3.0
[2.2.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.2.0...2.2.1
[2.2.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.1.1...2.2.0
[2.1.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.1.0...2.1.1
[2.1.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.0.4...2.1.0
[2.0.4]: https://github.com/PolicyEngine/policyengine-core/compare/2.0.3...2.0.4
[2.0.3]: https://github.com/PolicyEngine/policyengine-core/compare/2.0.2...2.0.3
[2.0.2]: https://github.com/PolicyEngine/policyengine-core/compare/2.0.1...2.0.2
[2.0.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.0.0...2.0.1
[2.0.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.12.4...2.0.0
[1.12.4]: https://github.com/PolicyEngine/policyengine-core/compare/1.12.3...1.12.4
[1.12.3]: https://github.com/PolicyEngine/policyengine-core/compare/1.12.2...1.12.3
[1.12.2]: https://github.com/PolicyEngine/policyengine-core/compare/1.12.1...1.12.2
[1.12.1]: https://github.com/PolicyEngine/policyengine-core/compare/1.12.0...1.12.1
[1.12.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.11.4...1.12.0
[1.11.4]: https://github.com/PolicyEngine/policyengine-core/compare/1.11.3...1.11.4
[1.11.3]: https://github.com/PolicyEngine/policyengine-core/compare/1.11.2...1.11.3
[1.11.2]: https://github.com/PolicyEngine/policyengine-core/compare/1.11.1...1.11.2
[1.11.1]: https://github.com/PolicyEngine/policyengine-core/compare/1.11.0...1.11.1
[1.11.0]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.24...1.11.0
[1.10.24]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.23...1.10.24
[1.10.23]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.22...1.10.23
[1.10.22]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.21...1.10.22
[1.10.21]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.20...1.10.21
[1.10.20]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.19...1.10.20
[1.10.19]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.18...1.10.19
[1.10.18]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.17...1.10.18
[1.10.17]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.16...1.10.17
[1.10.16]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.15...1.10.16
[1.10.15]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.14...1.10.15
[1.10.14]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.13...1.10.14
[1.10.13]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.12...1.10.13
[1.10.12]: https://github.com/PolicyEngine/policyengine-core/compare/1.10.11...1.10.12
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

