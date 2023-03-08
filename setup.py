from pathlib import Path

from setuptools import find_packages, setup

# Read the contents of our README file for PyPi
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Please make sure to cap all dependency versions, in order to avoid unwanted
# functional and integration breaks caused by external code updates.

general_requirements = [
    "pytest>5,<6",
    "numpy",
    "black",
    "linecheck",
    "yaml-changelog",
    "coverage",
    "sortedcontainers",
    "numexpr",
    "dpath",
    "nptyping<2",
    "psutil",
    "wheel",
    "h5py",
    "microdf_python",
    "tqdm",
    "requests",
    "pandas",
    "plotly",
]

dev_requirements = [
    "autopep8",
    "black",
    "coverage",
    "furo",
    "jupyter-book",
    "linecheck",
    "markupsafe",
    "plotly",
    "setuptools",
    "sphinx",
    "sphinx-argparse",
    "sphinx-math-dollar",
    "wheel",
    "yaml-changelog",
    "mypy",
    "types-PyYAML",
    "types-requests",
    "types-setuptools",
    "types-urllib3",
]

setup(
    name="policyengine-core",
    version="1.12.3",
    author="PolicyEngine",
    author_email="hello@policyengine.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    description="Core microsimulation engine enabling country-specific policy models.",
    keywords="tax benefit microsimulation framework",
    license="https://www.fsf.org/licensing/licenses/agpl-3.0.html",
    license_files=("LICENSE",),
    url="https://github.com/policyengine/policyengine-core",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "policyengine-core=policyengine_core.scripts.policyengine_command:main",
        ],
    },
    extras_require={
        "dev": dev_requirements,
    },
    include_package_data=True,  # Will read MANIFEST.in
    install_requires=general_requirements,
    packages=find_packages(exclude=["tests*"]),
)
