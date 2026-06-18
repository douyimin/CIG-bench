"""
CIG-Bench Fault demo (.py)
==========================

End-to-end fault interpretation on a 3D seismic volume:
    1. Load a seismic volume (T, H, W) from disk.
    2. Construct FaultPredictor — weights are auto-downloaded from ModelScope
    3. Run multi-scale memory-bounded inference.
    4. Visualize the fault probability volume overlaid on the seismic using cigvis.

Usage
-----
    python demo_fault.py                          # uses ../RealData/your_seis.npy
    python demo_fault.py path/to/seis.npy         # custom seismic file
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cig_bench.predictor.fault import FaultPredictor


def main(seis_path: str = "../RealData/your_seis.npy") -> None:
    # ------------------------------------------------------------------
    # 1. Load seismic volume.  Shape convention: (T, H, W)
    #    - T : depth / time samples
    #    - H : inline  (axis 0 in section view)
    #    - W : crossline (axis 1 in section view)
    # ------------------------------------------------------------------
    seis_path = Path(seis_path)
    if not seis_path.exists():
        raise FileNotFoundError(
            f"Seismic file not found: {seis_path}\n"
            "Pass a path to a (T, H, W) .npy seismic volume on the command line."
        )
    seis = np.load(seis_path).astype(np.float32)
    print(f"[fault] loaded seismic: shape={seis.shape}, dtype={seis.dtype}")

    # ------------------------------------------------------------------
    # 2. Build predictor.  Weights download themselves on first use.
    # ------------------------------------------------------------------
    predictor = FaultPredictor(device="cuda")

    # ------------------------------------------------------------------
    # 3. Run inference.
    #    - rank / chunk_size : split the inference along the depth axis to
    #      keep GPU memory bounded on large field volumes.
    #    - scale_t/h/w       : anisotropic rescaling for non-square sampling.
    #    - threshold         : zeros out probabilities below this value.
    # ------------------------------------------------------------------
    prob, seis_used = predictor.predict(
        seis,
        rank=4,
        chunk_size=64,
        threshold=0.5,
        scale_t=0.5, scale_h=0.85, scale_w=0.85,
        resize_back=True,
    )
    print(f"[fault] prob volume: shape={prob.shape}, "
          f"min={prob.min():.3f} max={prob.max():.3f}")

    # ------------------------------------------------------------------
    # 4. 3D visualization with cigvis.
    # ------------------------------------------------------------------
    predictor.visualize(seis_used, prob)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../RealData/your_seis.npy")
