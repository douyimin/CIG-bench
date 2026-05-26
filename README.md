<div align="center">

# CIG-Bench

**A Comprehensive Benchmark Toolkit for AI-Driven Subsurface Imaging Understanding**

[![PyPI](https://img.shields.io/badge/pip-cig__bench-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/cig-bench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.8-blue.svg)](#requirements)
[![Project Page](https://img.shields.io/badge/project-page-2D5F8A.svg)](https://douyimin.github.io/CIG-bench)

</div>

`cig_bench` is the official inference library accompanying the paper
**"CIG-Bench: A Comprehensive Survey and Benchmark for AI-Driven Subsurface Imaging Understanding"**.
It provides ready-to-use, pretrained deep-learning baselines for the five core seismic-interpretation
tasks of CIG-Bench:

| Task | Predictor | Backbone |
| :--- | :--- | :--- |
| Fault segmentation | `FaultPredictor` | HRNet w/ skip-connection (opt.) |
| Relative Geologic Time (RGT) | `RGTPredictor` | HRNet w/ skip-connection (opt.) |
| Channel segmentation | `ChannelPredictor` | HRNet w/ skip-connection (opt.) |
| Karst-cave segmentation | `KarstPredictor` | HRNet w/ skip-connection (opt.) |
| Property modeling (Vp / Density / Impedance / GR / Lithology …) | `PropertyPredictor` | HRNet w/ skip-connection (opt.) |

All five predictors share a uniform API and load weights automatically from
[ModelScope](https://www.modelscope.cn/) (default) or
[Hugging Face Hub](https://huggingface.co/) on first use.

> 📄 Paper / project page: <https://douyimin.github.io/CIG-bench>
> 💻 Source code: <https://github.com/douyimin/CIG-bench>

---

## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
  - [Fault segmentation](#fault-segmentation)
  - [RGT estimation](#rgt-estimation)
  - [Geobody segmentation (channel / karst)](#geobody-segmentation-channel--karst)
  - [Property modeling](#property-modeling)
- [Weight sources](#weight-sources)
- [Using local weights or a custom repo](#using-local-weights-or-a-custom-repo)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Citation](#citation)
- [License](#license)

---

## Installation

From PyPI:

```bash
pip install cig_bench
```

From source:

```bash
git clone https://github.com/douyimin/CIG-bench.git
cd CIG-bench/cig_bench_pkg
pip install .
```

For development (editable install with dev extras):

```bash
pip install -e ".[dev]"
```

You will additionally need at least one weight backend:

```bash
# Recommended for users in mainland China
pip install modelscope

# Or the international default
pip install huggingface_hub
```

---

## Quick start

The first time you build a predictor, the corresponding `.pth` checkpoint is downloaded into a local
cache. Subsequent constructions of the same predictor reuse the cached weights.

### Fault segmentation

The fault model predicts a probability volume that is thresholded into a fault mask. Anisotropic
rescaling (`scale_t`, `scale_h`, `scale_w`) adapts the model to surveys with non-square spatial
sampling. The optional `rank` and `chunk_size` arguments split inference along the depth axis
to keep GPU memory bounded on large field volumes.

```python
from cig_bench.predictor.fault import FaultPredictor

fault_predictor = FaultPredictor(device="cuda")
prob, used = fault_predictor.predict(
    seis,
    rank=4, chunk_size=64,       # memory-bounded chunked inference
    threshold=0.5,
    scale_t=0.5, scale_h=0.85, scale_w=0.85,
    resize_back=True,            # return result at the original (T, H, W)
)
fault_predictor.visualize(used, prob)
```

### RGT estimation

The RGT model regresses a smooth relative-geologic-time volume; horizons are then extracted as
iso-surfaces of that volume. Optional sparse horizon annotations may be passed as two auxiliary
channels (`horizon_rgt`, `horizon_mask`) to constrain the prediction.

```python
from cig_bench.predictor.rgt import RGTPredictor

rgt_predictor = RGTPredictor(device="cuda")
rgt_vol, used = rgt_predictor.predict(seis)
horizons      = rgt_predictor.extract_horizons(rgt_vol, n_horizons=100)
rgt_predictor.visualize(used, rgt_vol, horizons)
```

### Geobody segmentation (channel / karst)

Both geobody predictors share the same multi-scale ensemble strategy. By default inference is run
at seven spatial scales (from 0.25× to 1.5× the input size) and the resulting probability volumes
are accumulated. A configurable post-processing step removes small connected components.

```python
from cig_bench.predictor.channel import ChannelPredictor

channel_predictor = ChannelPredictor(device="cuda")
scores, used = channel_predictor.predict(
    seis,
    scales=[0.5, 0.75, 1.0, 1.25, 1.5],   # custom scale set
    accumulate="sum",
)
mask = channel_predictor.postprocess(scores, threshold=0.75, min_size=50000)
channel_predictor.visualize(used, scores, mask)
```

The karst predictor is used identically — only the checkpoint changes:

```python
from cig_bench.predictor.karst import KarstPredictor

karst_predictor = KarstPredictor(device="cuda")
scores, used = karst_predictor.predict(seis)
mask = karst_predictor.postprocess(scores, threshold=0.75, min_size=50000)
```

### Property modeling

The property predictor follows the GEM-style conditional paradigm: it takes a seismic volume and
a sparse well-log property volume (zeros where no well is present) and outputs a dense 3D
property volume. Internally it stacks three channels — seismic, sparse property, binary well mask —
and feeds them to the HRNet backbone. The number and location of wells are not fixed; passing
more wells generally improves accuracy.

```python
import numpy as np
from cig_bench.predictor.property import PropertyPredictor

prop_predictor = PropertyPredictor(device="cuda")
vp_vol, used, wells = prop_predictor.predict(
    seis, vp_log,
    infer_shape=(640, 512, 512),
)
prop_predictor.visualize(used, vp_vol, wells)
```

---

## Weight sources

`cig_bench` ships with two registries — `MODELSCOPE_REGISTRY` and `HF_REGISTRY` — that map each
task to a `(model_id, file_path)` pair. The active source is resolved in the following order:

1. The `source=` keyword argument passed to a predictor (`"modelscope"` or `"huggingface"`).
2. The environment variable `CIG_BENCH_WEIGHT_SOURCE`.
3. The default: `"modelscope"`.

Switch globally for all predictors:

```bash
export CIG_BENCH_WEIGHT_SOURCE=huggingface
```

Or per predictor:

```python
predictor = FaultPredictor(device="cuda", source="huggingface")
```

Aliases supported: `ms` / `model_scope` → `modelscope`; `hf` / `hugging_face` / `huggingface_hub` →
`huggingface`.

---

## Using local weights or a custom repo

```python
# 1) Use a local checkpoint (no download)
predictor = FaultPredictor("/path/to/fault.pth", device="cuda")

# 2) Override the default repo / filename / cache directory
predictor = FaultPredictor(
    model_id="your-group/CIG-Benchmark",
    file_path="fault.pth",
    cache_dir="./weights_cache",
    source="huggingface",          # or "modelscope"
    device="cuda",
)
```


To change the default repository IDs, edit `MODELSCOPE_DEFAULT_MODEL_ID` /
`HF_DEFAULT_MODEL_ID` (or the per-task entries) in
`cig_bench/predictor/_download.py`.

---

## Project layout

```text
cig_bench/
├── __init__.py
├── main.py
├── networks/                       # HRNet variants
│   ├── hrnet.py
│   ├── hrnet_skipconect.py
│   └── hrnet_skipconect_opt.py
└── predictor/                      # Inference pipelines
    ├── _download.py                # Auto-download from ModelScope / HF
    ├── channel.py
    ├── fault.py
    ├── karst.py
    ├── property.py
    ├── rgt.py
    └── utils.py
```

---

## Requirements

- Python ≥ 3.8
- `numpy ≥ 1.20`, `scipy ≥ 1.6`
- `torch ≥ 1.10` (GPU recommended; the predictors expose `rank` / `chunk_size` to bound memory)
- `cigvis` (for built-in `visualize(...)` methods)
- `modelscope` *and/or* `huggingface_hub` (depending on chosen weight source)

---

## Citation

If you use `cig_bench` in your research, please cite the accompanying survey & benchmark paper:

```bibtex
@article{dou2025cigbench,
  title       = {CIG-Bench: A Comprehensive Survey and Benchmark
                 for AI-Driven Subsurface Imaging Understanding},
  author      = {Dou, Yimin and Wu, Xinming},
  year        = {2025},
  institution = {University of Science and Technology of China}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
