"""失败分类：把「失败」拆成可操作的类别。

动机
----
「成功率 67%」这个数字几乎不携带信息。同样的 67% 可能来自：
  - 33% 的回合**把动作做错了**（能力问题 → 需要更好的策略）
  - 33% 的回合**做对了但没做完**（效率问题 → 需要更短的路径/更快的推进）
  - 33% 的回合**卡住了**（恢复问题 → 需要重试机制）

三者的改进方向完全不同，但成功率看不出区别。
本模块从 rollout 日志里把失败拆开。
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from .stats import wilson_ci

__all__ = ["EpisodeOutcome", "classify_episodes", "FailureTaxonomy",
           "failure_summary_table"]


@dataclass
class EpisodeOutcome:
    task: str
    ep: int
    success: bool
    steps: int
    max_steps: int

    @property
    def category(self) -> str:
        if self.success:
            return "success"
        # 到达步数上限 = 超时（任务没做完），而不是「做错了」
        if self.steps >= self.max_steps:
            return "timeout"
        return "terminated"      # 提前结束（env 主动终止）


def classify_episodes(manifest, max_steps: int = 500) -> list[EpisodeOutcome]:
    """把 manifest（list[dict]）转成带分类的回合列表。"""
    out = []
    for r in manifest:
        out.append(EpisodeOutcome(
            task=str(r["task"]), ep=int(r.get("ep", 0)),
            success=bool(r["success"]), steps=int(r.get("steps", 0)),
            max_steps=max_steps,
        ))
    return out


class FailureTaxonomy:
    """把一个套件的所有回合拆成 success / timeout / terminated。"""

    def __init__(self, outcomes: list[EpisodeOutcome]):
        self.outcomes = outcomes
        self.counts = collections.Counter(o.category for o in outcomes)
        self.by_task: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        for o in outcomes:
            self.by_task[o.task][o.category] += 1

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def success_rate(self) -> float:
        return self.counts["success"] / self.n if self.n else float("nan")

    @property
    def timeout_share_of_failures(self) -> float:
        """失败中「超时」的占比——这个数字决定了该往哪个方向优化。"""
        nfail = self.counts["timeout"] + self.counts["terminated"]
        return self.counts["timeout"] / nfail if nfail else float("nan")

    def timeout_step_histogram(self, bins: int = 5) -> dict:
        ts = [o.steps for o in self.outcomes
              if o.category == "timeout"]
        if not ts:
            return {}
        lo, hi = min(ts), max(ts)
        edges = np.linspace(lo, hi + 1, bins + 1)
        h, _ = np.histogram(ts, bins=edges)
        return {f"{int(edges[i])}-{int(edges[i+1])}": int(h[i]) for i in range(bins)}

    def diagnose(self) -> str:
        """给出**该往哪个方向优化**的建议。"""
        if self.n == 0:
            return "无数据"
        to = self.timeout_share_of_failures
        nfail = self.counts["timeout"] + self.counts["terminated"]
        if nfail == 0:
            return "全部成功，无需失败分析"
        if np.isnan(to):
            return "失败全为提前终止（env 判定失败），需查判据"
        if to >= 0.9:
            return ("失败几乎全是**超时**（任务未完成，而非动作错误）"
                    "→ 优先做效率优化：缩短路径、调整重规划节奏、提前终止重启")
        if to <= 0.1:
            return ("失败几乎全是**提前终止**（动作导致 env 判定失败）"
                    "→ 优先做能力优化：数据、策略、接触建模")
        return (f"失败混合：超时 {to:.0%} / 提前终止 {1-to:.0%}"
                "→ 两条线都要做，建议先解决占比高的一侧")

    def table(self) -> str:
        c = self.counts
        lo, hi = wilson_ci(c["success"], self.n)
        lines = [
            f"回合总数 {self.n}",
            f"  成功      {c['success']:>4}  ({100*c['success']/self.n:.1f}%)  "
            f"95% CI [{100*lo:.0f}, {100*hi:.0f}]",
            f"  超时      {c['timeout']:>4}  ({100*c['timeout']/self.n:.1f}%)  "
            f"← 任务未完成",
            f"  提前终止  {c['terminated']:>4}  ({100*c['terminated']/self.n:.1f}%)  "
            f"← 动作导致失败",
        ]
        return "\n".join(lines)


def failure_summary_table(manifests: dict[str, list], max_steps: int = 500) -> str:
    """多份 manifest → 一张对比表（含失败形态拆解）。"""
    rows = ["| 运行 | 成功率 | 失败=超时 | 失败=终止 | 诊断 |",
            "|---|---|---|---|---|"]
    for name, m in manifests.items():
        t = FailureTaxonomy(classify_episodes(m, max_steps))
        nfail = t.counts["timeout"] + t.counts["terminated"]
        to_pct = f"{100*t.counts['timeout']/nfail:.0f}%" if nfail else "—"
        te_pct = f"{100*t.counts['terminated']/nfail:.0f}%" if nfail else "—"
        diag = t.diagnose().split("→")[0].strip()
        rows.append(f"| {name} | {100*t.success_rate:.1f}% | {to_pct} | {te_pct} | {diag} |")
    return "\n".join(rows)
