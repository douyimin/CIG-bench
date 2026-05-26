"""Network architectures used by CIG_Bench predictors."""

from .hrnet import HRNet as HRNet
from .hrnet_skipconect_opt import HRNet as HRNetSkipOpt

__all__ = ["HRNet", "HRNetSkipOpt"]
