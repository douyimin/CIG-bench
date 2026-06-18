"""
CIG-Bench Property demo (.py)
=============================

End-to-end rock-property modeling on a 3D seismic volume conditioned on
sparse well-log values:
    1. Load a seismic volume (T, H, W) and a sparse property volume of the
       same shape, with zeros where no well is present.
    2. Construct PropertyPredictor — weights are auto-downloaded from
    3. Run inference: the predictor internally stacks three channels —
       seismic, sparse property, well mask — and feeds them to HRNet.
    4. Visualize the predicted dense property volume together with the
       seismic and well points using cigvis.

Usage
-----
    python demo_property.py                                 # uses defaults below
    python demo_property.py path/to/seis.npy path/to/prop.npy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cig_bench.predictor.property import PropertyPredictor


def main(seis_path: str = "../RealData/your_seis.npy",
         prop_path: str = "../RealData/your_log.npy") -> None:
    # ------------------------------------------------------------------
    # 1. Load seismic and sparse well-log property volumes.
    #    Both are (T, H, W); `prop` is zero everywhere except at well
    #    trace locations, where it holds the measured property value.
    # ------------------------------------------------------------------
    seis_path, prop_path = Path(seis_path), Path(prop_path)
    if not seis_path.exists():
        raise FileNotFoundError(f"Seismic file not found: {seis_path}")
    if not prop_path.exists():
        raise FileNotFoundError(
            f"Property (well-log) file not found: {prop_path}\n"
            "It must have the same (T, H, W) shape as the seismic, with zeros "
            "at non-well locations."
        )

    seis = np.load(seis_path).astype(np.float32)
    prop = np.load(prop_path).astype(np.float32)
    assert seis.shape == prop.shape, \
        f"seis {seis.shape} and prop {prop.shape} must have the same shape"
    print(f"[property] loaded seismic / prop: shape={seis.shape}")
    print(f"[property] well voxels: {int((prop != 0).sum())}")

    # ------------------------------------------------------------------
    # 2. Build predictor.  Weights download themselves on first use.
    # ------------------------------------------------------------------
    predictor = PropertyPredictor(device="cuda")

    # ------------------------------------------------------------------
    # 3. Run inference.
    #    `infer_shape` is the size the model was trained on; the predictor
    #    resamples to/from it automatically.
    # ------------------------------------------------------------------
    prop_vol, seis_used, wells = predictor.predict(
        seis, prop,
        infer_shape=(640, 512, 512),
    )
    print(f"[property] property volume: shape={prop_vol.shape}, "
          f"min={prop_vol.min():.3f} max={prop_vol.max():.3f}")

    # ------------------------------------------------------------------
    # 4. 3D visualization with cigvis.
    # ------------------------------------------------------------------
    predictor.visualize(seis_used, prop_vol, wells)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        main(args[0], args[1])
    elif len(args) == 1:
        main(args[0])
    else:
        main()
