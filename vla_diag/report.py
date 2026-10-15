"""报告生成：把评测数据变成带区间和失败形态的 markdown。"""
from __future__ import annotations

import collections

from .failures import FailureTaxonomy, classify_episodes
from .stats import (format_ci, min_detectable_effect,
                    required_n_two_proportions, wilson_ci)

__all__ = ["rollout_report", "comparison_report", "power_note"]


def power_note(n: int, p: float = 0.8) -> str:
    """给一个样本量一个「能测出多大差异」的说明——防止过度解读小样本。

    样本量太小时不报 NaN，而是给出**做不到的原因和所需样本量**。
    """
    mde = min_detectable_effect(n, p)
    if mde != mde:      # NaN：连最大差异都测不出
        need = required_n_two_proportions(p, 1.0, power=0.8) if p < 1 else None
        return (f"n={n}/组：**样本量过小**。基线 {100*p:.0f}% 时，"
                f"连「{100*p:.0f}%→100%」这种极端差异都达不到 80% 功效"
                + (f"（需 ≥{need}/组）" if need else ""))
    return (f"n={n}/组：在基线 {100*p:.0f}% 下，"
            f"只能检测 ≥{100*mde:.0f}pp 的差异（α=0.05, power=0.8）")


def rollout_report(manifest, title: str = "评测报告",
                   max_steps: int = 500) -> str:
    """单个运行的报告：逐任务成功率 + 区间 + 失败形态。"""
    per = collections.defaultdict(list)
    for r in manifest:
        per[str(r["task"])].append(r)

    L = [f"# {title}", ""]
    L.append("| 任务 | 成功/总数 | 成功率 | 95% CI | 平均步数(成功) |")
    L.append("|---|---|---|---|---|")
    tk = tn = 0
    for task in sorted(per):
        v = per[task]
        k = sum(1 for r in v if r["success"])
        n = len(v)
        tk += k
        tn += n
        ok = [r["steps"] for r in v if r["success"]]
        avg = f"{sum(ok)/len(ok):.0f}" if ok else "—"
        L.append(f"| {task} | {k}/{n} | {100*k/n:.0f}% | "
                 f"[{100*wilson_ci(k,n)[0]:.0f}, {100*wilson_ci(k,n)[1]:.0f}] | {avg} |")
    L.append(f"| **合计** | **{tk}/{tn}** | **{100*tk/tn:.1f}%** | "
             f"**[{100*wilson_ci(tk,tn)[0]:.0f}, {100*wilson_ci(tk,tn)[1]:.0f}]** | |")
    L += ["", "## 失败形态", "", "```",
          FailureTaxonomy(classify_episodes(manifest, max_steps)).table(), "```", "",
          "**诊断**：" + FailureTaxonomy(classify_episodes(manifest, max_steps)).diagnose(),
          "", f"> ⚠ {power_note(min(len(v) for v in per.values()))}", ""]
    return "\n".join(L)


def comparison_report(runs: dict[str, list], max_steps: int = 500,
                      label_a: str = None, label_b: str = None) -> str:
    """两个运行的头对头对比（含失败形态差异）。"""
    from .stats import two_proportion_p, two_proportion_z

    names = list(runs)
    if len(names) != 2:
        raise ValueError("comparison_report 目前支持恰好两个运行")
    a, b = names
    pa = collections.defaultdict(list)
    pb = collections.defaultdict(list)
    for r in runs[a]:
        pa[str(r["task"])].append(r)
    for r in runs[b]:
        pb[str(r["task"])].append(r)

    L = [f"# 对比：{a} vs {b}", ""]
    L.append(f"| 任务 | {a} | {b} | 差值 |")
    L.append("|---|---|---|---|")
    tk1 = tn1 = tk2 = tn2 = 0
    for task in sorted(set(pa) | set(pb)):
        v1, v2 = pa.get(task, []), pb.get(task, [])
        k1 = sum(1 for r in v1 if r["success"])
        k2 = sum(1 for r in v2 if r["success"])
        n1, n2 = len(v1), len(v2)
        tk1 += k1; tn1 += n1; tk2 += k2; tn2 += n2
        if n1 and n2:
            d = 100 * k2 / n2 - 100 * k1 / n1
            L.append(f"| {task} | {format_ci(k1,n1)} | {format_ci(k2,n2)} | {d:+.0f}pp |")
        else:
            L.append(f"| {task} | {format_ci(k1,n1)} | {format_ci(k2,n2)} | — |")
    lo1, hi1 = wilson_ci(tk1, tn1)
    lo2, hi2 = wilson_ci(tk2, tn2)
    L.append(f"| **合计** | **{100*tk1/tn1:.1f}% [{100*lo1:.0f},{100*hi1:.0f}]** | "
             f"**{100*tk2/tn2:.1f}% [{100*lo2:.0f},{100*hi2:.0f}]** | "
             f"**{100*tk2/tn2-100*tk1/tn1:+.1f}pp** |")
    z = two_proportion_z(tk1, tn1, tk2, tn2)
    p = two_proportion_p(tk1, tn1, tk2, tn2)
    L += ["", f"两比例 z 检验：z = {z:.2f}，p = {p:.2e}", ""]

    t1 = FailureTaxonomy(classify_episodes(runs[a], max_steps))
    t2 = FailureTaxonomy(classify_episodes(runs[b], max_steps))
    L += ["## 失败形态对比", "",
          f"- {a}: " + t1.diagnose(),
          f"- {b}: " + t2.diagnose(), ""]

    nmin = min(min(len(v) for v in pa.values()), min(len(v) for v in pb.values()))
    L += [f"> ⚠ {power_note(nmin)}", ""]
    return "\n".join(L)
