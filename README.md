# CIG_Bench

A benchmark toolkit for seismic interpretation tasks built on HRNet.

Provides ready-to-use inference pipelines for:

- **channel** detection
- **fault** detection
- **karst** detection
- rock **property** estimation (Vp / Density / Impedance / GR ...)
- **RGT** (Relative Geologic Time) estimation

## Installation

```bash
pip install cig_bench
```

Or install from source:

```bash
git clone https://github.com/your-org/CIG_Bench.git
cd CIG_Bench
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick start

权重会在第一次使用时自动下载到本地缓存，之后再次构造 Predictor 不会重复下载。
默认从 [魔搭 ModelScope](https://www.modelscope.cn/) 下载，可切换到
[Hugging Face Hub](https://huggingface.co/)。

```python
from cig_bench.predictor.fault import FaultPredictor

# 默认从魔搭下载
predictor = FaultPredictor(device="cuda")

# 改用 Hugging Face
predictor = FaultPredictor(device="cuda", source="huggingface")
```

也可以通过环境变量全局切换默认源(对所有 Predictor 生效)：

```bash
export CIG_BENCH_WEIGHT_SOURCE=huggingface
```

也可以手动指定本地权重或覆盖远端坐标：

```python
# 1) 用本地已有权重
predictor = FaultPredictor("/path/to/fault.pth", device="cuda")

# 2) 覆盖默认仓库 / 文件路径
predictor = FaultPredictor(
    model_id="your-group/CIG-Benchmark",
    file_path="fault.pth",
    cache_dir="./weights_cache",   # 可选：自定义缓存目录
    source="huggingface",          # 或 "modelscope"
    device="cuda",
)
```

可用的 Predictor: `ChannelPredictor` / `FaultPredictor` / `karstPredictor` /
`PropertyPredictor` / `RGTPredictor`。

> 注意：所有权重统一为 `.pth` (state_dict) 格式，内部走
> ``model = ModelCls(); model.load_state_dict(torch.load(path, map_location='cpu'))``
> 的加载方式，不再支持 `torch.jit.load` 模式。

如需修改默认仓库 ID，编辑 `cig_bench/predictor/_download.py` 中的
``MODELSCOPE_DEFAULT_MODEL_ID`` / ``HF_DEFAULT_MODEL_ID`` 或对应注册表。

## Project layout

```
cig_bench/
├── networks/       # HRNet variants
└── predictor/      # Inference pipelines
    ├── channel.py
    ├── fault.py
    ├── karst.py
    ├── property.py
    ├── rgt.py
    └── utils.py
```

## Requirements

- Python >= 3.8
- numpy, scipy, torch, cigvis
- modelscope (用于魔搭下载)
- huggingface_hub (用于 Hugging Face 下载)

## License

MIT
