"""余量诊断：在投入算力做「测试时选择」之前，先测有没有东西可选。

核心问题
--------
「用世界模型/验证器在 K 个候选动作里挑一个」——这个方法值不值得做？

直觉答案是「值」，但直觉是错的。在一个已经训练好的策略上：

  - 大多数决策点**所有候选都失败** → 早就输了，选谁都一样
  - 很多决策点**所有候选都成功** → 早就赢了，选谁都一样
  - 只有「有成功也有失败」的决策点，选择才有意义

这个比例叫**判别性（discriminability）**。它决定了选择类方法的收益上限。

实测（LIBERO，4 个候选，1598 个决策点）：
    全失败  83.3%
    全成功  12.7%
    判别性   4.0%   ← 只有这 4% 有得选
    → 完美选择器只比随机高 2.4 个百分点

所以正确顺序是：**先测余量，再决定要不要投算力**。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .stats import wilson_ci

__all__ = ["Headroom", "headroom", "headroom_curve", "required_k",
           "verdict_text"]


@dataclass
class Headroom:
    """一个决策点矩阵的余量诊断结果。"""
    n_decisions: int
    K: int
    winner_hist: dict          # 每个决策点有几个候选成功
    pct_all_fail: float
    pct_all_success: float
    pct_discriminative: float
    random_rate: float         # 随机选的期望成功率
    oracle_rate: float         # 完美选择器的成功率
    oracle_gap_pp: float       # 完美选择器的增益（百分点）
    per_task: dict | None = None

    def __str__(self) -> str:
        return (f"Headroom(n={self.n_decisions}, K={self.K}, "
                f"disc={100*self.pct_discriminative:.1f}%, "
                f"oracle_gap={self.oracle_gap_pp:+.2f}pp)")

    def table(self) -> str:
        n = self.n_decisions
        lo, hi = wilson_ci(int(round(self.oracle_gap_pp / 100 * n)), n)
        return "\n".join([
            f"决策点 {n}，每点候选 K={self.K}",
            f"  全失败    {100*self.pct_all_fail:>6.1f}%   ← 早就输了，选谁都一样",
            f"  全成功    {100*self.pct_all_success:>6.1f}%   ← 早就赢了，选谁都一样",
            f"  判别性    {100*self.pct_discriminative:>6.1f}%   ← ★ 只有这里有得选",
            "",
            f"  随机选     {100*self.random_rate:>6.2f}%",
            f"  完美选择器  {100*self.oracle_rate:>6.2f}%",
            f"  收益上限   {self.oracle_gap_pp:>+6.2f} pp",
        ])

    def verdict(self, min_gap_pp: float = 1.0) -> str:
        return verdict_text(self.pct_discriminative, self.oracle_gap_pp, min_gap_pp)


def headroom(winners: list[int] | np.ndarray, K: int,
             task_ids: list | None = None) -> Headroom:
    """从「每个决策点有几个候选成功」的数组算余量。

    参数
    ----
    winners : 长度 = 决策点数，每个元素是 [0, K] 的整数
    K       : 每个决策点的候选数
    task_ids: 可选，用于分组（如 "libero_10/4"）
    """
    w = np.asarray(winners, dtype=int)
    if w.size == 0:
        raise ValueError("winners 为空")
    if w.max() > K or w.min() < 0:
        raise ValueError(f"winners 越界：应在 [0, {K}]")

    n = int(w.size)
    hist = {int(k): int((w == k).sum()) for k in range(K + 1)}
    all_fail = hist.get(0, 0) / n
    all_succ = hist.get(K, 0) / n
    disc = 1.0 - all_fail - all_succ
    rand = float(w.sum()) / (n * K)
    oracle = 1.0 - all_fail

    per_task = None
    if task_ids is not None:
        per_task = {}
        for tid in sorted(set(task_ids)):
            m = np.array([t == tid for t in task_ids])
            sub = w[m]
            per_task[str(tid)] = {
                "n": int(sub.size),
                "disc": float(1 - (sub == 0).mean() - (sub == K).mean()),
                "all_fail": float((sub == 0).mean()),
                "all_success": float((sub == K).mean()),
            }

    return Headroom(
        n_decisions=n, K=K, winner_hist=hist,
        pct_all_fail=all_fail, pct_all_success=all_succ,
        pct_discriminative=disc,
        random_rate=rand, oracle_rate=oracle,
        oracle_gap_pp=100 * (oracle - rand),
        per_task=per_task,
    )


def verdict_text(disc: float, gap_pp: float, min_gap_pp: float = 1.0) -> str:
    """给出「该不该做测试时选择」的建议。"""
    if disc < 0.05:
        return (f"判别性 {100*disc:.1f}% < 5%：**没有余量**。"
                f"选择类方法（世界模型/验证器/多数投票）理论上限仅 {gap_pp:+.1f}pp。"
                f"优先改策略本身，而不是加选择器。")
    if gap_pp < min_gap_pp:
        return (f"判别性 {100*disc:.1f}%，但完美选择器只有 {gap_pp:+.1f}pp < "
                f"{min_gap_pp:.1f}pp：收益太小，不值得投入。")
    if disc < 0.15:
        return (f"判别性 {100*disc:.1f}%：余量有限，"
                f"完美选择器上限 {gap_pp:+.1f}pp。可以做，但要清楚上限不高。")
    return (f"判别性 {100*disc:.1f}%：**余量充足**，"
            f"完美选择器上限 {gap_pp:+.1f}pp。值得投入。")


def headroom_curve(winners: list[int] | np.ndarray, K: int) -> dict:
    """如果只能保留 k 个候选（k=1..K），余量如何变化？

    用途：K 越大越贵。这条曲线告诉你「加候选」的边际收益在哪饱和。
    """
    w = np.asarray(winners, dtype=int)
    n = w.size
    out = {}
    for k in range(1, K + 1):
        # 保留 k 个候选时：成功概率 = P(前 k 个里至少一个成功)
        # 无放回近似（把 winners 视为可挑出的成功候选数）
        p = float(np.minimum(w, k).sum()) / (n * k)
        oracle = float((w >= 1).mean())          # 只要能挑到 1 个成功即可
        out[k] = {"random_rate": p, "oracle_rate": oracle,
                  "gap_pp": 100 * (oracle - p)}
    return out


def required_k(winners: list[int] | np.ndarray, K: int,
               target_gap_pp: float) -> int | None:
    """要达到 target_gap_pp 的收益，最少需要多少个候选？"""
    curve = headroom_curve(winners, K)
    for k in sorted(curve):
        if curve[k]["gap_pp"] >= target_gap_pp:
            return k
    return None


def effect_size_from_discriminative(disc: float, K: int) -> float:
    """理论估计：判别性比例为 disc、K 个候选取最大值时，相对随机的增益。

    对均匀随机的成功概率 p，取 K 次最大值会把 E[max] 抬到
    1-(1-p)^K 附近。这里用判别性比例做一阶近似。
    """
    if K <= 1:
        return 0.0
    # 判别性决策点里，随机选的成功概率约为 1/K（只有 1 个成功候选的情形）
    p_rand = (1 - disc) * 1.0 + disc * (1.0 / K)     # 全成功点贡献 1
    p_oracle = (1 - disc) * 1.0 + disc * 1.0
    return float(p_oracle - p_rand)
