"""
CIG-Bench dataset download demo (.py)
=====================================

One line of code downloads a dataset subset from ModelScope
(repo: douyimin/CIG-Bench-Dataset) and returns its local path.

Usage
-----
    python demo_dataset.py
"""

from __future__ import annotations

from cig_bench.dataset import cig_structureData, cig_geobodyData


def main() -> None:
    # Force the download directory (here: ./CIG-Bench-Dataset).
    # Each call returns the local path of the downloaded subset.
    download_dir = "./CIG-Bench-Dataset"

    structure_dir = cig_structureData(download_dir)   # -> ./CIG-Bench-Dataset/Structure
    print(f"[dataset] Structure downloaded to: {structure_dir}")

    geobody_dir = cig_geobodyData(download_dir)       # -> ./CIG-Bench-Dataset/Geobody
    print(f"[dataset] Geobody downloaded to: {geobody_dir}")


if __name__ == "__main__":
    main()
