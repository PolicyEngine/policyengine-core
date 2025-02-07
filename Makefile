all: install format test build changelog

documentation:
	jb clean docs
	jb build docs
	python docs/add_plotly_to_book.py docs/_build

format:
	black . -l 79

install:
	pip install -e ".[dev]" --config-settings editable_mode=compat
	pip install git+https://github.com/noman404/policyengine-us.git@noman404/python3.13
	pip install policyengine-uk

test-country-template:
	policyengine-core test policyengine_core/country_template/tests -c policyengine_core.country_template

mypy:
	mypy --config-file mypy.ini policyengine_core tests

test: test-country-template
	coverage run -a --branch -m pytest tests --disable-pytest-warnings
	coverage xml -i

build:
	python setup.py sdist bdist_wheel

changelog:
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 0.1.0 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org PolicyEngine --repo policyengine-core --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml setup.py
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml