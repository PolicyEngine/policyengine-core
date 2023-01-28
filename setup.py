from pathlib import Path

from setuptools import find_packages, setup

# Read the contents of our README file for PyPi
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Please make sure to cap all dependency versions, in order to avoid unwanted
# functional and integration breaks caused by external code updates.

general_requirements = [
    "pytest==5.4.3",
    "numpy==1.22.4",
    "black==22.12.0",
    "linecheck==0.1.0",
    "yaml-changelog==0.3.0",
    "coverage==6.5.0",
    "sortedcontainers==2.4.0",
    "numexpr==2.7.3",
    "dpath==2.0.1",
    "nptyping==1.4.3",
    "psutil==5.8.0",
    "wheel==0.38.4",
    "h5py==3.7.0",
    "microdf_python==0.3.0",
    "tqdm==4.62.3",
    "requests==2.28.1",
    "pandas==1.5.2",
    "plotly==5.11.0",
]

dev_requirements = [
    "jupyter-book==0.13.1",
    "furo==2022.9.29",
    "markupsafe==2.0.1",
    "mypy==0.991",
    "sphinx==4.5.0",
    "sphinx-argparse==0.4.0",
    "sphinx-math-dollar==1.2.1",
    "types-PyYAML==6.0.12.2",
    "types-requests==2.28.11.7",
    "types-setuptools==65.6.0.2",
    "types-urllib3==1.26.25.4",
]

setup(
    name="policyengine-core",
    version="1.12.1",
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
