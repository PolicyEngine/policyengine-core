from pathlib import Path

from setuptools import find_packages, setup

# Read the contents of our README file for PyPi
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Please make sure to cap all dependency versions, in order to avoid unwanted
# functional and integration breaks caused by external code updates.

general_requirements = [
    "pytest>=8,<9",
    "numpy~=1.26.4",
    "sortedcontainers<3",
    "numexpr<3",
    "dpath<3",
    "psutil>=6,<7",
    "wheel<1",
    "h5py>=3,<4",
    "requests>=2,<3",
    "pandas>=1",
    "plotly>=5,<6",
    "ipython>=8,<9",
    "pyvis>=0.3.2",
    "microdf_python>=0.4.3",
    "huggingface_hub>=0.25.1",
]

dev_requirements = [
    "black",
    "linecheck<1",
    "jupyter-book<1",
    "yaml-changelog<1",
    "coverage",
    "furo<2025",
    "markupsafe<3",
    "coverage",
    "furo",
    "mypy<2",
    "sphinx==5.0.0",
    "sphinx-argparse==0.4.0",
    "sphinx-math-dollar==1.2.1",
    "types-PyYAML==6.0.12.2",
    "types-requests==2.28.11.7",
    "types-setuptools==65.6.0.2",
    "types-urllib3==1.26.25.4",
]

setup(
    name="policyengine-core",
    version="3.17.1",
    author="PolicyEngine",
    author_email="hello@policyengine.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
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
    python_requires=">=3.10",
    extras_require={
        "dev": dev_requirements,
    },
    include_package_data=True,  # Will read MANIFEST.in
    install_requires=general_requirements,
    packages=find_packages(exclude=["tests*"]),
)
