import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# numpy 版本
# ---------------------------------------------------------------------------
def normalization(data: np.ndarray) -> np.ndarray:
    """线性归一化到 [0, 1]。"""
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + 1e-6)


def z_score_clip(data: np.ndarray, clp_s: float = 3.0) -> np.ndarray:
    """对数据做 z-score 标准化后按 ±clp_s 截断，再归一化到 [0, 1]。"""
    z = (data - np.mean(data)) / np.std(data)
    return normalization(np.clip(z, a_min=-clp_s, a_max=clp_s))


# ---------------------------------------------------------------------------
# torch 版本
# ---------------------------------------------------------------------------
def tensor_normalization(data: torch.Tensor) -> torch.Tensor:
    """线性归一化到 [0, 1]（torch 版本）。"""
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / (_range + 1e-6)


def tensor_z_score_clip(data: torch.Tensor, clp_s: float = 3.0) -> torch.Tensor:
    """z-score 标准化 + ±clp_s 截断 + 归一化到 [0, 1]（torch 版本）。"""
    z = (data - torch.mean(data)) / torch.std(data)
    return tensor_normalization(torch.clip(z, min=-clp_s, max=clp_s))


def normalization_expzero(data):
    mask = data != 0
    valid_data = data[mask]
    if valid_data.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    _range = max_val - min_val
    norm_data = np.zeros_like(data, dtype=np.float32)
    norm_data[mask] = (valid_data - min_val) / (_range + 1e-6) + 1e-4
    return norm_data


def z_score_clip_expzero(data, clp_s=3):
    mask = data != 0
    valid_data = data[mask]
    if valid_data.size == 0:
        return np.zeros_like(data)
    mean = np.mean(valid_data)
    std = np.std(valid_data)
    z = np.zeros_like(data, dtype=np.float32)
    z[mask] = (valid_data - mean) / std
    z = np.clip(z, a_min=-clp_s, a_max=clp_s)
    z[mask] = z[mask] * std + mean
    return normalization_expzero(z)


# ---------------------------------------------------------------------------
# 二值分割后处理
# ---------------------------------------------------------------------------
def remove_small_instances(segmentation: np.ndarray,
                           min_size: int,
                           connectivity: int = 1) -> np.ndarray:
    """
    删除体积小于 min_size 的连通实例。

    Args:
        segmentation: 3D 二值分割结果（布尔数组或 0/1 数组）。
        min_size:     最小体素数阈值，小于它的连通域会被移除。
        connectivity: 连通性，1=6-连通，2=18-连通，3=26-连通（默认 1）。

    Returns:
        bool 数组，形状与输入相同。
    """
    from skimage.measure import label  # 延迟 import
    labeled = label(segmentation, connectivity=connectivity)
    counts = np.bincount(labeled.ravel())
    keep = np.where(counts >= min_size)[0]
    keep = keep[keep != 0]  # 排除背景标签 0
    return np.isin(labeled, keep)


def horizons_from_rgt(sig, hu, ux):
    """
    Extract a set of horizons (iso-surfaces) from an RGT volume.

    Parameters
    ----------
    sig : float
        Smoothing width applied to each extracted horizon (sig=0 means no smoothing).
    hu : 1D array_like
        RGT values at which horizons (iso-surfaces) are extracted.
    ux : 3D ndarray, shape (n3, n2, n1)
        The RGT volume. n1 is the vertical axis.

    Returns
    -------
    hzs : 3D ndarray, shape (nh, n3, n2)
        For each requested RGT value hu[ih], a horizon surface giving the
        vertical (i1) position as a function of (i3, i2).
    """
    ux = np.asarray(ux, dtype=np.float32)
    hu = np.asarray(hu, dtype=np.float32)

    n3, n2, n1 = ux.shape
    nh = hu.shape[0]

    # ---------------------------------------------------------------
    # Step 1: build a "monotonic mask" for every trace, vectorized.
    #
    # Java logic per trace (n3, n2):
    #     keep ux[0]; then keep ux[i1] iff ux[i1] > max(kept so far).
    # That is exactly: keep the running cumulative-maximum strict-increase points.
    #
    # A sample at (i3,i2,i1) is kept  <=>  ux[i3,i2,i1] > cummax(ux[i3,i2,:i1]).
    # The first sample (i1=0) is always kept.
    # ---------------------------------------------------------------
    cmax = np.maximum.accumulate(ux, axis=-1)  # shape (n3, n2, n1)
    keep = np.empty_like(ux, dtype=bool)
    keep[..., 0] = True
    keep[..., 1:] = ux[..., 1:] > cmax[..., :-1]  # strict > vs running max so far

    # ---------------------------------------------------------------
    # Step 2: mirror the Java `copy(k1-1, ...)` step (drops the LAST kept sample).
    # We do that by clearing the last True in each trace.
    # ---------------------------------------------------------------
    # Index of the last True along axis=-1 for each (i3,i2).
    # Trick: argmax on the reversed boolean array gives the offset from the end.
    rev = keep[..., ::-1]
    last_offset = rev.argmax(axis=-1)  # 0 means the very last sample is True
    last_idx = (n1 - 1) - last_offset  # absolute index of last True in each trace
    i3_idx, i2_idx = np.indices((n3, n2))
    keep[i3_idx, i2_idx, last_idx] = False  # drop it, matching `copy(k1-1, ...)`

    # ---------------------------------------------------------------
    # Step 3: per-trace linear interpolation hu -> x1, vectorized.
    #
    # Trick: we sort kept u-values to the front of each trace by replacing
    # not-kept positions with +inf (so they get pushed to the tail by argsort).
    # Then for each query hu[ih], we find the segment via vectorized searchsorted
    # WITHOUT building a (nh, n3, n2, n1) array -- we iterate only over nh,
    # which is small (typically a handful). Inside the nh loop we still operate
    # on the entire (n3, n2) plane in one shot, so it's still vectorized.
    # ---------------------------------------------------------------
    INF = np.float32(np.inf)
    u_kept = np.where(keep, ux, INF)  # (n3, n2, n1)
    # Sort each trace ascendingly along axis=-1; sentinels go to the tail.
    order = np.argsort(u_kept, axis=-1, kind="stable")  # (n3, n2, n1)
    u_sorted = np.take_along_axis(u_kept, order, axis=-1)  # (n3, n2, n1)
    # Apply the same permutation to the original sample indices (0..n1-1).
    idx_orig = np.broadcast_to(np.arange(n1, dtype=np.float32),
                               (n3, n2, n1))
    x_sorted = np.take_along_axis(idx_orig, order, axis=-1)  # (n3, n2, n1)

    n_keep = keep.sum(axis=-1)  # (n3, n2)

    hzs = np.empty((nh, n3, n2), dtype=np.float32)
    # Loop only over the (small) nh axis. Each iteration is fully vectorized
    # over the (n3, n2) plane and avoids any (nh, n3, n2, n1) intermediates.
    for ih in range(nh):
        q = hu[ih]
        # Number of kept u-values strictly less than q, per trace.
        # Equivalent to per-row np.searchsorted(u_sorted, q, side='left'),
        # but works on the full 3D array via broadcasting + sum.
        # We mask sentinels by counting only "u < q AND kept".
        k = (u_sorted < q).sum(axis=-1).astype(np.int32)  # (n3, n2)

        # Form segment endpoints [lo, hi] = [k-1, k], clamped into [0, n_keep-1].
        hi = np.clip(k, 1, n_keep - 1)  # ensure hi >= 1
        lo = hi - 1

        u_lo = np.take_along_axis(u_sorted, lo[..., None], axis=-1)[..., 0]
        u_hi = np.take_along_axis(u_sorted, hi[..., None], axis=-1)[..., 0]
        x_lo = np.take_along_axis(x_sorted, lo[..., None], axis=-1)[..., 0]
        x_hi = np.take_along_axis(x_sorted, hi[..., None], axis=-1)[..., 0]

        du = u_hi - u_lo
        du_safe = np.where(du == 0, np.float32(1.0), du)
        t = (q - u_lo) / du_safe
        result = x_lo + t * (x_hi - x_lo)

        # Degenerate traces (length < 2) fall back to x_lo (= 0 on those traces),
        # matching the loop version.
        degenerate = n_keep < 2
        result = np.where(degenerate, x_lo, result)
        hzs[ih] = result.astype(np.float32)

    # ---------------------------------------------------------------
    # Step 4: optional lateral smoothing (matches ref.apply1 then ref.apply2).
    # ---------------------------------------------------------------
    if sig > 0:
        hzs = gaussian_filter1d(hzs, sigma=sig, axis=1, mode="nearest")
        hzs = gaussian_filter1d(hzs, sigma=sig, axis=2, mode="nearest")

    return hzs


def horizon_image(n1, n2, n3, d, sig, x1):
    """
    Paint a set of horizons into a 3D image volume for visualization.

    Parameters
    ----------
    n1, n2, n3 : int
        Dimensions of the output image volume (n1 is vertical).
    d : int
        Thickness of each horizon (allowed: 0, 1, 2, 3, 4, 5, 7) - same as Java.
    sig : float
        Smoothing width applied to each horizon before painting (sig=0 disables).
    x1 : 3D ndarray, shape (ns, n3, n2)
        The set of horizons (vertical positions) to paint.

    Returns
    -------
    hx : 3D ndarray, shape (n3, n2, n1)
        Background = -10. Horizon `is_` painted with value (is_ + 1),
        matching the Java overload that takes no `vs` array.
    """
    x1 = np.asarray(x1, dtype=np.float32)
    ns = x1.shape[0]

    # Optional pre-smoothing along (n3, n2).
    xs = x1.copy()
    if sig > 0:
        xs = gaussian_filter1d(xs, sigma=sig, axis=1, mode="nearest")
        xs = gaussian_filter1d(xs, sigma=sig, axis=2, mode="nearest")

    hx = np.full((n3, n2, n1), -10.0, dtype=np.float32)

    # ---------------------------------------------------------------
    # Per-horizon vectorized painting. We compute, for each (is_, i3, i2),
    # which i1 indices need to be painted, then do one fancy-index assignment.
    #
    # The Java logic painting locations:
    #   - center: i1 = round(x1i)
    #   - d == 2: + nearest of (i1-1, i1+1) to x1i
    #   - d == 3: + (i1-1, i1+1)
    #   - d == 4: + nearest of (i1-1, i1+1) plus two outer extension points
    #   - d == 5: + (i1-2, i1+2)
    #   - d == 7: + (i1-3, i1+3)
    #   - d in {0,1}: only center
    # ---------------------------------------------------------------
    # Pre-build a (n3, n2) grid of (i3, i2) for vectorized indexing.
    I3, I2 = np.indices((n3, n2))  # both (n3, n2)

    for is_ in range(ns):
        val = np.float32(is_ + 1)
        x1i = xs[is_]  # (n3, n2)

        valid = (x1i >= 0) & (x1i < n1)  # samples in range
        if not valid.any():
            continue

        # Java's Math.round: half away from zero. For non-negative x1i this is floor(x+0.5).
        # We mimic it with int(x+0.5) for the valid locations.
        i1 = np.empty_like(x1i, dtype=np.int32)
        i1[valid] = (x1i[valid] + 0.5).astype(np.int32)

        # Helper: paint a 2-D map of target i1 indices (with an in-range mask) into hx.
        def paint(target_i1, mask):
            # Combine masks: only paint where mask is True AND target is in [0, n1).
            ok = mask & (target_i1 >= 0) & (target_i1 < n1)
            if ok.any():
                hx[I3[ok], I2[ok], target_i1[ok]] = val

        # Center sample (always painted when valid).
        paint(i1, valid)

        if d == 2:
            m1 = i1 - 1
            p1 = i1 + 1
            # Nearest of (m1, p1) to x1i:
            choose_p = np.abs(m1 - x1i) > np.abs(p1 - x1i)
            k1 = np.where(choose_p, p1, m1)
            paint(k1, valid)

        elif d == 4:
            m1 = i1 - 1
            p1 = i1 + 1
            choose_p = np.abs(m1 - x1i) > np.abs(p1 - x1i)
            k1 = np.where(choose_p, p1, m1)
            # If k1 > i1 (i.e. chose p1): k2 = i1-1, k3 = k1+1
            # else                       : k2 = k1-1, k3 = i1+1
            k1_above = k1 > i1
            k2 = np.where(k1_above, i1 - 1, k1 - 1)
            k3 = np.where(k1_above, k1 + 1, i1 + 1)
            paint(k1, valid)
            paint(k2, valid)
            paint(k3, valid)

        elif d == 3:
            paint(i1 - 1, valid)
            paint(i1 + 1, valid)

        elif d == 5:
            paint(i1 - 2, valid)
            paint(i1 + 2, valid)

        elif d == 7:
            paint(i1 - 3, valid)
            paint(i1 + 3, valid)
        # d in {0, 1}: only the center, already painted above.

    return hx


# ---------------------------------------------------------------------------
# Smoke test + numerical comparison vs the loop-based reference implementation.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    # Reference (loop) implementation, copied verbatim from the loop version.
    from scipy.interpolate import interp1d


    def horizons_from_rgt_loop(sig, hu, ux):
        ux = np.asarray(ux, dtype=np.float32)
        hu = np.asarray(hu, dtype=np.float32)
        n3, n2, n1 = ux.shape
        nh = hu.shape[0]
        hzs = np.zeros((nh, n3, n2), dtype=np.float32)
        for i3 in range(n3):
            for i2 in range(n2):
                uxi = ux[i3, i2]
                u1 = np.empty(n1, dtype=np.float32)
                x1 = np.empty(n1, dtype=np.float32)
                u1[0] = uxi[0]
                x1[0] = 0.0
                k1 = 1
                for i1 in range(1, n1):
                    if uxi[i1] > u1[k1 - 1]:
                        u1[k1] = uxi[i1]
                        x1[k1] = float(i1)
                        k1 += 1
                length = k1 - 1
                if length < 2:
                    hzs[:, i3, i2] = x1[0] if length >= 1 else 0.0
                    continue
                u1 = u1[:length];
                x1 = x1[:length]
                ci = interp1d(u1, x1, kind="linear", bounds_error=False,
                              fill_value="extrapolate", assume_sorted=True)
                hzs[:, i3, i2] = ci(hu).astype(np.float32)
        if sig > 0:
            hzs = gaussian_filter1d(hzs, sigma=sig, axis=1, mode="nearest")
            hzs = gaussian_filter1d(hzs, sigma=sig, axis=2, mode="nearest")
        return hzs


    def horizon_image_loop(n1, n2, n3, d, sig, x1):
        x1 = np.asarray(x1, dtype=np.float32)
        ns = x1.shape[0]
        xs = x1.copy()
        if sig > 0:
            xs = gaussian_filter1d(xs, sigma=sig, axis=1, mode="nearest")
            xs = gaussian_filter1d(xs, sigma=sig, axis=2, mode="nearest")
        hx = np.full((n3, n2, n1), -10.0, dtype=np.float32)
        for is_ in range(ns):
            val = float(is_ + 1)
            for i3 in range(n3):
                for i2 in range(n2):
                    x1i = xs[is_, i3, i2]
                    if x1i < 0 or x1i >= n1: continue
                    i1 = int(x1i + 0.5)
                    if 0 <= i1 < n1: hx[i3, i2, i1] = val
                    if d == 2:
                        m1, p1 = i1 - 1, i1 + 1
                        k1 = m1 if abs(m1 - x1i) <= abs(p1 - x1i) else p1
                        if 0 <= k1 < n1: hx[i3, i2, k1] = val
                    elif d == 4:
                        m1, p1 = i1 - 1, i1 + 1
                        k1 = m1 if abs(m1 - x1i) <= abs(p1 - x1i) else p1
                        if k1 > i1:
                            k2, k3 = i1 - 1, k1 + 1
                        else:
                            k2, k3 = k1 - 1, i1 + 1
                        for kk in (k1, k2, k3):
                            if 0 <= kk < n1: hx[i3, i2, kk] = val
                    elif d == 3:
                        for kk in (i1 - 1, i1 + 1):
                            if 0 <= kk < n1: hx[i3, i2, kk] = val
                    elif d == 5:
                        for kk in (i1 - 2, i1 + 2):
                            if 0 <= kk < n1: hx[i3, i2, kk] = val
                    elif d == 7:
                        for kk in (i1 - 3, i1 + 3):
                            if 0 <= kk < n1: hx[i3, i2, kk] = val
        return hx


    # ---- Build a non-trivial RGT cube (mostly-monotonic + small noise) ----
    rng = np.random.default_rng(0)
    n3, n2, n1 = 25, 35, 120
    base = np.arange(n1, dtype=np.float32)[None, None, :]
    lateral = (2.0 * np.sin(0.2 * np.arange(n2))[None, :, None]
               + 1.5 * np.cos(0.15 * np.arange(n3))[:, None, None])
    noise = rng.normal(0, 0.05, size=(n3, n2, n1)).astype(np.float32)
    ux = base + lateral + noise  # not strictly monotonic everywhere -> exercises the filter

    hu = np.array([10.0, 30.0, 60.0, 90.0, 110.0], dtype=np.float32)

    # ---- horizons_from_rgt: vectorized vs loop ----
    t0 = time.perf_counter()
    hzs_v = horizons_from_rgt(sig=0.0, hu=hu, ux=ux)
    t_v = time.perf_counter() - t0

    t0 = time.perf_counter()
    hzs_l = horizons_from_rgt_loop(sig=0.0, hu=hu, ux=ux)
    t_l = time.perf_counter() - t0

    diff_h = np.max(np.abs(hzs_v - hzs_l))
    print(f"horizons_from_rgt :  vectorized {t_v * 1000:7.1f} ms   "
          f"loop {t_l * 1000:7.1f} ms   speedup x{t_l / t_v:5.1f}   "
          f"max|diff|={diff_h:.3e}")

    # ---- horizon_image: vectorized vs loop, all relevant d values ----
    for d in (0, 1, 2, 3, 4, 5, 7):
        t0 = time.perf_counter()
        img_v = horizon_image(n1=n1, n2=n2, n3=n3, d=d, sig=0.0, x1=hzs_v)
        t_v = time.perf_counter() - t0

        t0 = time.perf_counter()
        img_l = horizon_image_loop(n1=n1, n2=n2, n3=n3, d=d, sig=0.0, x1=hzs_v)
        t_l = time.perf_counter() - t0

        diff_i = np.max(np.abs(img_v - img_l))
        print(f"horizon_image d={d}:  vectorized {t_v * 1000:7.1f} ms   "
              f"loop {t_l * 1000:7.1f} ms   speedup x{t_l / t_v:5.1f}   "
              f"max|diff|={diff_i:.3e}")
