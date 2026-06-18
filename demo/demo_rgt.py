"""
CIG-Bench RGT demo (.py)
========================

End-to-end relative geological time (RGT) estimation on a 3D seismic volume:
    1. Load a seismic volume (T, H, W) from disk.
    2. Construct RGTPredictor — weights are auto-downloaded from ModelScope
    3. Run inference to obtain a dense RGT volume.
    4. Extract horizons as iso-surfaces of the RGT volume.
    5. Visualize the seismic together with the horizons using cigvis.

Usage
-----
    python demo_rgt.py                       # uses ../RealData/your_seis.npy
    python demo_rgt.py path/to/seis.npy      # custom seismic file
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cig_bench.predictor.rgt import RGTPredictor


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
    print(f"[rgt] loaded seismic: shape={seis.shape}, dtype={seis.dtype}")

    # ------------------------------------------------------------------
    # 2. Build predictor.  Weights download themselves on first use.
    #    infer_shape is the size the model was trained on; the predictor
    #    resamples to/from it automatically.
    # ------------------------------------------------------------------
    predictor = RGTPredictor(device="cuda", infer_shape=(400, 512, 512))

    # ------------------------------------------------------------------
    # 3. Run inference.  Two optional auxiliary channels can be supplied
    #    (horizon_rgt, horizon_mask); leaving them as None disables
    #    horizon constraints.
    # ------------------------------------------------------------------
    rgt_vol, seis_used = predictor.predict(seis)
    print(f"[rgt] rgt volume: shape={rgt_vol.shape}, "
          f"min={rgt_vol.min():.3f} max={rgt_vol.max():.3f}")

    # ------------------------------------------------------------------
    # 4. Extract horizons (returns a (H, W, T) mask, value = RGT on
    #    each horizon, 0 elsewhere).
    # ------------------------------------------------------------------
    horizon_mask = predictor.extract_horizons(rgt_vol, n_horizons=100)

    # ------------------------------------------------------------------
    # 5. 3D visualization with cigvis.
    # ------------------------------------------------------------------
    predictor.visualize(seis_used, rgt_vol, horizon_mask)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../RealData/your_seis.npy")
