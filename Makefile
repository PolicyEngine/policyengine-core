all: install format test build changelog

documentation:
	jb clean docs
	jb build docs

format:
	black . -l 79

install:
	pip install -e .[dev]

test-country-template:
	policyengine-core test policyengine_core/country_template/tests -c policyengine_core.country_template

mypy:
	mypy --config-file mypy.ini policyengine_core tests

test: test-country-template
	coverage run -a --branch -m pytest tests --disable-pytest-warnings
	coverage xml -i

build:
	python -m build

changelog:
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 0.1.0 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org PolicyEngine --repo policyengine-core --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml pyproject.toml
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml