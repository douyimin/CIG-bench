"""
CIG-Bench Karst demo (.py)
==========================

End-to-end karst-cavity segmentation on a 3D seismic volume:
    1. Load a seismic volume (T, H, W) from disk.
    2. Construct KarstPredictor — weights are auto-downloaded from ModelScope
    3. Run multi-scale ensemble inference.
    4. Threshold the score volume and remove small connected components.
    5. Visualize the binary mask overlaid on the seismic using cigvis.

Usage
-----
    python demo_karst.py                         # uses ../RealData/your_seis.npy
    python demo_karst.py path/to/seis.npy        # custom seismic file
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cig_bench.predictor.karst import KarstPredictor


def main(seis_path: str = "../RealData/your_seis.npy") -> None:
    # ------------------------------------------------------------------
    # 1. Load seismic volume.  Shape convention: (T, H, W)
    # ------------------------------------------------------------------
    seis_path = Path(seis_path)
    if not seis_path.exists():
        raise FileNotFoundError(
            f"Seismic file not found: {seis_path}\n"
            "Pass a path to a (T, H, W) .npy seismic volume on the command line."
        )
    seis = np.load(seis_path).astype(np.float32)
    print(f"[karst] loaded seismic: shape={seis.shape}, dtype={seis.dtype}")

    # ------------------------------------------------------------------
    # 2. Build predictor.  Weights download themselves on first use.
    # ------------------------------------------------------------------
    predictor = KarstPredictor(device="cuda")

    # ------------------------------------------------------------------
    # 3. Multi-scale ensemble inference.  Karst features are usually small,
    #    so the default 7-scale grid (0.25× — 1.5×) covers them well.
    # ------------------------------------------------------------------
    scores, seis_used = predictor.predict(seis)
    print(f"[karst] score volume: shape={scores.shape}, "
          f"min={scores.min():.3f} max={scores.max():.3f}")

    # ------------------------------------------------------------------
    # 4. Threshold + small-component removal.  min_size is smaller than
    #    for channels because karst bodies are typically more compact.
    # ------------------------------------------------------------------
    mask = predictor.postprocess(scores, threshold=0.75, min_size=2000)
    print(f"[karst] mask volume: positive voxels = {int(mask.sum())}")

    # ------------------------------------------------------------------
    # 5. 3D visualization with cigvis.
    # ------------------------------------------------------------------
    predictor.visualize(seis_used, scores, mask)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../RealData/your_seis.npy")
