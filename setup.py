"""Setup script for cig_bench.

Build a source distribution and wheel:
    python -m pip install --upgrade build
    python -m build

Upload to PyPI:
    python -m pip install --upgrade twine
    python -m twine upload dist/*

Local installation:
    pip install .
    pip install -e .          # editable / development install
"""

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent.resolve()


def _read(filename: str) -> str:
    path = HERE / filename
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


long_description = _read("README.md")


setup(
    name="cig_bench",
    version="0.1.0",
    description=(
        "CIG_Bench: a benchmark toolkit for seismic interpretation tasks "
        "(channel / fault / karst / property / RGT) built on HRNet."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CIG_Bench Authors",
    author_email="your_email@example.com",
    url="https://github.com/your-org/CIG_Bench",
    license="MIT",
    # ---------------------------------------------------------------
    # Packages
    # ---------------------------------------------------------------
    packages=find_packages(include=["cig_bench", "cig_bench.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    # ---------------------------------------------------------------
    # Runtime dependencies
    # ---------------------------------------------------------------
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.6",
        "torch>=1.10",
        "cigvis",
        "modelscope",
        "huggingface_hub",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "build",
            "twine",
            "wheel",
        ],
    },
    # ---------------------------------------------------------------
    # Classifiers / metadata
    # ---------------------------------------------------------------
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "seismic",
        "geophysics",
        "deep-learning",
        "hrnet",
        "benchmark",
        "fault-detection",
        "channel-detection",
        "rgt",
    ],
    zip_safe=False,
)
