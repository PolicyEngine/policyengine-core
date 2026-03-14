all: install format test build changelog

documentation:
	uv run jb clean docs
	uv run jb build docs

format:
	uv run black . -l 79

install:
	uv pip install -e .[dev]

test-country-template:
	uv run policyengine-core test policyengine_core/country_template/tests -c policyengine_core.country_template

mypy:
	uv run mypy --config-file mypy.ini policyengine_core tests

test: test-country-template
	uv run coverage run -a --branch -m pytest tests --disable-pytest-warnings
	uv run coverage xml -i

build:
	uv run python setup.py sdist bdist_wheel

changelog:
	uv run build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 0.1.0 --append-file changelog_entry.yaml
	uv run build-changelog changelog.yaml --org PolicyEngine --repo policyengine-core --output CHANGELOG.md --template .github/changelog_template.md
	uv run bump-version changelog.yaml setup.py
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml