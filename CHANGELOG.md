# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.6.0] - 2024-08-28 16:41:26

### Fixed

- Bugs in typing and Dataset saving.

## [3.5.3] - 2024-08-21 14:34:52

### Changed

- Fixed script to initialize a reform within the simulation

## [3.5.2] - 2024-08-17 19:28:38

### Changed

- change mypy version from 1.11.1 to <2

## [3.5.1] - 2024-08-08 15:17:05

### Fixed

- fixed bug where uprating was causing nan.

## [3.5.0] - 2024-08-08 15:16:12

## [3.4.0] - 2024-08-07 12:42:36

### Added

- Flat file export option to simulations.

## [3.3.0] - 2024-08-02 09:09:51

### Added

- Simulation loading from dataframes.
- Simulation `start_instant` attribute.

## [3.2.0] - 2024-07-28 19:28:05

### Changed

- Reform syntax to increase flexibility.

## [3.1.0] - 2024-07-26 00:28:31

### Added

- Flat file dataset option.
- Microdata weighting classes from microdf.
- Eternity dictionary reform options.

## [3.0.0] - 2024-06-26 17:08:24

### Changed

- Added support for Python 3.12

## [2.23.1] - 2024-06-06 09:09:41

### Added

- V
- i
- s
- u
- a
- l
- i
- z
- a
- t
- i
- o
- n
-  
- o
- p
- t
- i
- o
- n
-  
- w
- h
- e
- n
-  
- r
- u
- n
- n
- i
- n
- g
-  
- t
- e
- s
- t
- s

## [2.23.0] - 2024-06-04 19:43:13

### Added

- max_value and min_value in Variable class.

## [2.22.0] - 2024-05-31 11:12:13

### Fixed

- Bug causing reforms affecting data to not affect caches.

## [2.21.8] - 2024-05-14 15:33:07

### Fixed

- Uncap coverage.

## [2.21.7] - 2024-05-14 15:27:27

### Fixed

- Remove name metadata tag in docs.

## [2.21.6] - 2024-05-13 16:26:53

### Changed

- Replaced unsafe numpy-Python comparison with use of numpy dtype to convert byte-string arrays to Unicode ones within enums

## [2.21.5] - 2024-05-08 15:07:51

### Fixed

- Bug in macro cache logic.

## [2.21.4] - 2024-05-08 13:39:53

### Added

- Added `.vscode` and `venv` to `.gitignore` file.

## [2.21.3] - 2024-05-08 13:36:00

### Added

- Dependency upgrade for pytest, from 7 to 8.

## [2.21.2] - 2024-05-08 08:44:48

### Fixed

- Generation of enum-typed variables

## [2.21.1] - 2024-05-07 15:44:58

### Fixed

- Bug in macro caching logic.

## [2.21.0] - 2024-05-07 12:16:05

### Added

- Ability to turn off read and write features in the macro cache.

## [2.20.0] - 2024-05-06 14:24:08

### Added

- Macro impact caching.
- Dictionary-input start-stop date reform handling.

### Fixed

- Uprating bugs.

## [2.19.2] - 2024-05-06 12:47:34

### Added

- IPython to dependencies

## [2.19.1] - 2024-04-29 15:06:58

### Fixed

- FutureWarning issue with bools and enums.

## [2.19.0] - 2024-04-16 12:51:42

### Added

- Baseline parameter node.

## [2.18.0] - 2024-03-22 10:42:19

### Added

- Parameters can now be uprated based upon fixed cadences

## [2.17.1] - 2024-03-05 13:01:42

### Fixed

- Breakdown-added parameters no longer precede uprating indices.

## [2.17.0] - 2024-03-04 11:27:40

### Changed

- Add method for propagating units to descendants for ParameterScale objects

## [2.16.1] - 2024-02-29 16:03:09

### Fixed

- Fixed a bug in Core monthlyisation.

## [2.16.0] - 2024-02-14 14:49:15

## [2.15.0] - 2024-02-08 14:32:38

### Added

- Variable inheritance in declarations.

## [2.14.0] - 2024-02-06 17:28:06

### Changed

- A variable must have a label.
- If a variable has formulas then it must not have adds/subtracts, and vice versa.

## [2.13.1] - 2024-02-06 11:28:22

### Added

- Improvements to error handling in bad variable declarations.

## [2.13.0] - 2024-01-02 15:26:52

### Added

- Simulation helper to extract individual households from a microsimulation.

## [2.12.1] - 2023-12-27 13:28:08

### Fixed

- API to reform functionality made more editable.

## [2.12.0] - 2023-12-26 18:31:31

### Added

- API ID to reform object functionality.

## [2.11.5] - 2023-12-21 10:55:47

### Fixed

- Randomness issue.

## [2.11.4] - 2023-12-19 11:12:24

### Added

- Tools for branching

## [2.11.3] - 2023-12-16 15:34:37

### Fixed

- Random seed fixed for each simulation.

## [2.11.2] - 2023-12-14 11:38:52

### Fixed

- Bug causing random behaviour differences between situations with and without axes.

## [2.11.1] - 2023-11-30 18:45:01

### Fixed

- Bug causing `subtracts` to not accept strings.

## [2.11.0] - 2023-11-22 15:55:52

### Fixed

- Reforms always apply before parameter utilities.

## [2.10.0] - 2023-11-22 15:05:43

### Fixed

- Reforms always apply before parameter utilities.

## [2.9.0] - 2023-11-12 13:23:23

### Added

- PolicyEngine API integration for reforms.

## [2.8.2] - 2023-10-09 17:52:02

### Changed

- Unpinned lower NumPy pin.

## [2.8.1] - 2023-10-05 13:24:20

### Fixed

- Bug causing YAML tests to not populate `Simulation.input_variables`.

## [2.8.0] - 2023-10-05 11:41:53

### Added

- Support for adds/subtracts targeting parameters.
- Support for adds/subtracts targeting year ranges where no parameters exist.

### Changed

- Deprecated sum_of_variables.

## [2.7.1] - 2023-10-05 11:00:40

### Fixed

- Fix bug affecting microsimulation runs in countries which use automatic period adjustments.

## [2.7.0] - 2023-10-04 19:05:58

### Added

- Automatic period adjustment helper functionality.

### Changed

- Default error threshold for tests widened to 1e-3.

## [2.6.0] - 2023-10-04 14:51:40

### Fixed

- Bump numpy version
- Fix numpy record compatibility issue

## [2.5.0] - 2023-08-02 12:54:44

### Added

- Chart formatting utilities.

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



[3.6.0]: https://github.com/PolicyEngine/policyengine-core/compare/3.5.3...3.6.0
[3.5.3]: https://github.com/PolicyEngine/policyengine-core/compare/3.5.2...3.5.3
[3.5.2]: https://github.com/PolicyEngine/policyengine-core/compare/3.5.1...3.5.2
[3.5.1]: https://github.com/PolicyEngine/policyengine-core/compare/3.5.0...3.5.1
[3.5.0]: https://github.com/PolicyEngine/policyengine-core/compare/3.4.0...3.5.0
[3.4.0]: https://github.com/PolicyEngine/policyengine-core/compare/3.3.0...3.4.0
[3.3.0]: https://github.com/PolicyEngine/policyengine-core/compare/3.2.0...3.3.0
[3.2.0]: https://github.com/PolicyEngine/policyengine-core/compare/3.1.0...3.2.0
[3.1.0]: https://github.com/PolicyEngine/policyengine-core/compare/3.0.0...3.1.0
[3.0.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.23.1...3.0.0
[2.23.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.23.0...2.23.1
[2.23.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.22.0...2.23.0
[2.22.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.21.8...2.22.0
[2.21.8]: https://github.com/PolicyEngine/policyengine-core/compare/2.21.7...2.21.8
[2.21.7]: https://github.com/PolicyEngine/policyengine-core/compare/2.21.6...2.21.7
[2.21.6]: https://github.com/PolicyEngine/policyengine-core/compare/2.21.5...2.21.6
[2.21.5]: https://github.com/PolicyEngine/policyengine-core/compare/2.21.4...2.21.5
[2.21.4]: https://github.com/PolicyEngine/policyengine-core/compare/2.21.3...2.21.4
[2.21.3]: https://github.com/PolicyEngine/policyengine-core/compare/2.21.2...2.21.3
[2.21.2]: https://github.com/PolicyEngine/policyengine-core/compare/2.21.1...2.21.2
[2.21.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.21.0...2.21.1
[2.21.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.20.0...2.21.0
[2.20.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.19.2...2.20.0
[2.19.2]: https://github.com/PolicyEngine/policyengine-core/compare/2.19.1...2.19.2
[2.19.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.19.0...2.19.1
[2.19.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.18.0...2.19.0
[2.18.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.17.1...2.18.0
[2.17.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.17.0...2.17.1
[2.17.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.16.1...2.17.0
[2.16.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.16.0...2.16.1
[2.16.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.15.0...2.16.0
[2.15.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.14.0...2.15.0
[2.14.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.13.1...2.14.0
[2.13.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.13.0...2.13.1
[2.13.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.12.1...2.13.0
[2.12.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.12.0...2.12.1
[2.12.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.11.5...2.12.0
[2.11.5]: https://github.com/PolicyEngine/policyengine-core/compare/2.11.4...2.11.5
[2.11.4]: https://github.com/PolicyEngine/policyengine-core/compare/2.11.3...2.11.4
[2.11.3]: https://github.com/PolicyEngine/policyengine-core/compare/2.11.2...2.11.3
[2.11.2]: https://github.com/PolicyEngine/policyengine-core/compare/2.11.1...2.11.2
[2.11.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.11.0...2.11.1
[2.11.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.10.0...2.11.0
[2.10.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.9.0...2.10.0
[2.9.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.8.2...2.9.0
[2.8.2]: https://github.com/PolicyEngine/policyengine-core/compare/2.8.1...2.8.2
[2.8.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.8.0...2.8.1
[2.8.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.7.1...2.8.0
[2.7.1]: https://github.com/PolicyEngine/policyengine-core/compare/2.7.0...2.7.1
[2.7.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.6.0...2.7.0
[2.6.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.5.0...2.6.0
[2.5.0]: https://github.com/PolicyEngine/policyengine-core/compare/2.4.0...2.5.0
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

