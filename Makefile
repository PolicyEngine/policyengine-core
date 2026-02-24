all: install format test build changelog

documentation:
	jb clean docs
	jb build docs
	python docs/add_plotly_to_book.py docs/_build

format:
	black . -l 79

install:
	pip install -e ".[dev]" --config-settings editable_mode=compat

test-country-template:
	policyengine-core test policyengine_core/country_template/tests -c policyengine_core.country_template

mypy:
	mypy --config-file mypy.ini policyengine_core tests

test: test-country-template
	coverage run -a --branch -m pytest tests --disable-pytest-warnings --reruns 2 --reruns-delay 5
	coverage xml -i

build:
	python -m build

changelog:
	python .github/bump_version.py
	towncrier build --yes --version $$(python -c "import re; print(re.search(r'version = \"(.+?)\"', open('pyproject.toml').read()).group(1))")