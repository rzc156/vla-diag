"""统计工具：置信区间、配对检验、功效分析。

设计原则：
  - 每个函数都只依赖 numpy，不引入 scipy（便于在任何环境跑）
  - 所有区间默认 95%
  - 报告里**必须**带区间，禁止只报点估计

为什么需要这个模块
------------------
VLA 评测最常见的错误是「拿一次 n=5 的成功率比较两个模型」。
本模块提供最小可检测效应（MDE）计算，让你在跑之前就知道
「这个样本量能测出多大的差异」——而不是跑完才发现测不出。
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "wilson_ci", "wilson_interval", "two_proportion_z", "two_proportion_p",
    "mcnemar", "mcnemar_exact", "paired_bootstrap",
    "required_n_two_proportions", "min_detectable_effect",
    "format_ci", "summarize_binary",
]


# ---------------------------------------------------------------- 置信区间

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 区间（推荐替代 Wald 区间）。

    小样本或极端比例（0/5、5/5）时 Wald 区间会给出 [0,0] 或 [1,1] 这种
    无意义的区间，Wilson 不会。VLA 评测里 n 常常只有 5~20，必须用 Wilson。
    """
    if n <= 0:
        return (0.0, 0.0)
    k = max(0, min(k, n))
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


wilson_interval = wilson_ci


def format_ci(k: int, n: int, digits: int = 1) -> str:
    """'12/20 = 60.0% [38, 79]' 一行式。"""
    if n == 0:
        return "0/0 = —"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} = {100*k/n:.{digits}f}% [{100*lo:.0f}, {100*hi:.0f}]"


# ---------------------------------------------------------------- 两比例

def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """两独立比例 z 统计量（合并比例版本）。"""
    if min(n1, n2) <= 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    return (p2 - p1) / se


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """两比例检验的 p 值（正态近似，双尾）。"""
    z = two_proportion_z(k1, n1, k2, n2)
    if math.isnan(z):
        return float("nan")
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


# ---------------------------------------------------------------- 配对检验

def mcnemar(b: int, c: int) -> dict:
    """McNemar 检验（配对二分类），返回卡方近似 + 精确二项 p 值。

    b = A 成功 & B 失败 的配对数
    c = A 失败 & B 成功 的配对数

    为什么配对：真机评测里固定初始位姿后，两次运行的配对差异远小于
    独立比较。用配对检验可以把方差降一个量级——**但前提是真配对**
    （同一初始状态）。如果初始状态不可复现，配对无效，自由度白丢。
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "chi2": 0.0, "p": 1.0,
                "note": "无不一致对，检验无信息"}
    chi2 = (abs(b - c) - 1) ** 2 / n
    p_chi2 = math.exp(-chi2 / 2) if chi2 > 0 else 1.0   # chi2(1) 近似
    # 精确二项（双尾），b ~ Binom(n, 0.5)
    k = min(b, c)
    cum = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_exact = min(1.0, 2 * cum)
    return {"b": b, "c": c, "n_discordant": n,
            "chi2": chi2, "p": p_exact, "p_chi2_approx": p_chi2}


mcnemar_exact = mcnemar


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """配对 bootstrap：估计差值 (b - a) 的置信区间。

    a, b 是等长的 0/1 序列（同一批初始状态下的两个方法）。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("a, b 必须是等长非空序列")
    rng = np.random.default_rng(seed)
    n = a.size
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = (b[idx] - a[idx]).mean(axis=1)
    obs = float(b.mean() - a.mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # 双尾 p：差值分布中跨过 0 的比例
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta": obs, "ci": (float(lo), float(hi)), "p": float(min(1.0, p)),
            "n": n, "n_boot": n_boot}


# ---------------------------------------------------------------- 功效分析

def required_n_two_proportions(p1: float, p2: float,
                               alpha: float = 0.05, power: float = 0.80) -> int:
    """检测 p1 → p2 差异所需的**每组**样本量。

    用标准两比例样本量公式（含反正弦变换修正）：
        n = (z_{α/2} + z_β)² · (p1(1-p1) + p2(1-p2)) / (p1-p2)²
    """
    if p1 == p2:
        raise ValueError("p1 与 p2 相同，无法计算样本量")
    z_a = _z_from_alpha(alpha)
    z_b = _z_from_power(power)
    num = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    return int(math.ceil(num / (p1 - p2) ** 2))


def min_detectable_effect(n: int, p: float = 0.8,
                          alpha: float = 0.05, power: float = 0.80) -> float:
    """给定每组 n 和基线比例 p，**能检测出的最小差异**（绝对百分点，0-1）。

    二分搜索：找最小的 delta 使 required_n(p, p+delta) <= n。

    这是「n=5 能不能下结论」的量化答案。
    """
    if n <= 0:
        return float("nan")
    lo, hi = 0.0, 1.0 - p
    # 快速判定：即使最大差异也测不出
    if required_n_two_proportions(p, 1.0 - 1e-6, alpha, power) > n:
        return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2
        try:
            need = required_n_two_proportions(p, p + mid, alpha, power)
        except ValueError:
            need = 10 ** 9
        if need <= n:
            hi = mid
        else:
            lo = mid
    return hi


def _z_from_alpha(alpha: float) -> float:
    """双尾 α 对应的 z（正态近似）。"""
    return _norm_ppf(1 - alpha / 2)


def _z_from_power(power: float) -> float:
    return _norm_ppf(power)


def _norm_ppf(q: float) -> float:
    """标准正态分位数（Acklam 近似，精度 ~1e-9）。"""
    if not 0 < q < 1:
        raise ValueError("q 必须落在 (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        t = math.sqrt(-2 * math.log(q))
        return (((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / \
               ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    if q > phigh:
        t = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / \
               ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    t = q - 0.5
    r = t * t
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*t / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------- 汇总

def summarize_binary(records, key: str = "success") -> dict:
    """把一批 episode 记录汇总成 k/n + Wilson 区间。"""
    vals = [bool(r[key]) for r in records]
    n = len(vals)
    k = sum(vals)
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": (k / n if n else float("nan")),
            "ci": (lo, hi), "text": format_ci(k, n)}
