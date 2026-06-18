"""
CIG-Bench Channel demo (.py)
============================

End-to-end channel-geobody segmentation on a 3D seismic volume:
    1. Load a seismic volume (T, H, W) from disk.
    2. Construct ChannelPredictor — weights are auto-downloaded from ModelScope
    3. Run multi-scale ensemble inference (7 scales by default).
    4. Threshold the score volume and remove small connected components.
    5. Visualize the binary mask overlaid on the seismic using cigvis.

Usage
-----
    python demo_channel.py                       # uses ../RealData/your_seis.npy
    python demo_channel.py path/to/seis.npy      # custom seismic file
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cig_bench.predictor.channel import ChannelPredictor


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
    print(f"[channel] loaded seismic: shape={seis.shape}, dtype={seis.dtype}")

    # ------------------------------------------------------------------
    # 2. Build predictor.  Weights download themselves on first use.
    # ------------------------------------------------------------------
    predictor = ChannelPredictor(device="cuda")

    # ------------------------------------------------------------------
    # 3. Multi-scale ensemble inference.  Setting `accumulate="sum"`
    #    accumulates sigmoid probabilities across scales; the threshold
    #    used in step 4 is therefore relative to the sum, not to [0,1].
    # ------------------------------------------------------------------
    scores, seis_used = predictor.predict(
        seis,
        scales=[0.5, 0.75, 1.0, 1.25, 1.5],
        accumulate="sum",
    )
    print(f"[channel] score volume: shape={scores.shape}, "
          f"min={scores.min():.3f} max={scores.max():.3f}")

    # ------------------------------------------------------------------
    # 4. Threshold + small-component removal.
    # ------------------------------------------------------------------
    mask = predictor.postprocess(scores, threshold=0.75, min_size=50000)
    print(f"[channel] mask volume: positive voxels = {int(mask.sum())}")

    # ------------------------------------------------------------------
    # 5. 3D visualization with cigvis.
    # ------------------------------------------------------------------
    predictor.visualize(seis_used, scores, mask)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../RealData/your_seis.npy")
