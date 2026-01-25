all: install format test build changelog

documentation:
	jb clean docs
	jb build docs
	uv run python docs/add_plotly_to_book.py docs/_build

format:
	uv run black . -l 79

install:
	uv sync --all-extras

test-country-template:
	uv run policyengine-core test policyengine_core/country_template/tests -c policyengine_core.country_template

mypy:
	uv run mypy --config-file mypy.ini policyengine_core tests

test: test-country-template
	uv run coverage run -a --branch -m pytest tests --disable-pytest-warnings --reruns 2 --reruns-delay 5
	uv run coverage xml -i

build:
	uv build

changelog:
	uv run build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 0.1.0 --append-file changelog_entry.yaml
	uv run build-changelog changelog.yaml --org PolicyEngine --repo policyengine-core --output CHANGELOG.md --template .github/changelog_template.md
	uv run bump-version changelog.yaml pyproject.toml
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml
