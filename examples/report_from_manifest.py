#!/usr/bin/env python3
"""示例：从 rollout manifest 生成一份「诚实」的评测报告。

用法:
    python examples/report_from_manifest.py videos/manifest.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vla_diag import (  # noqa: E402
    FailureTaxonomy, check_actions, classify_episodes, format_ci, power_note,
)

# 典型的 LIBERO 动作维度（绝对 EEF 位姿 + 夹爪）
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def main(path: str):
    manifest = json.load(open(path))
    print(f"载入 {len(manifest)} 个回合: {path}\n")

    # ---------- 1. 失败形态 ----------
    t = FailureTaxonomy(classify_episodes(manifest, max_steps=500))
    print("=" * 60)
    print("  1. 失败形态拆解")
    print("=" * 60)
    print(t.table())
    print(f"\n  {t.diagnose()}\n")

    # ---------- 2. 逐任务 + 区间 ----------
    print("=" * 60)
    print("  2. 逐任务成功率（带 95% 区间）")
    print("=" * 60)
    per = {}
    for r in manifest:
        per.setdefault(str(r["task"]), []).append(r)
    for task in sorted(per):
        v = per[task]
        k = sum(1 for x in v if x["success"])
        print(f"  {task:<30} {format_ci(k, len(v))}")

    # ---------- 3. 样本量警告 ----------
    print("\n" + "=" * 60)
    print("  3. 这个样本量能下什么结论")
    print("=" * 60)
    n_min = min(len(v) for v in per.values())
    print(f"  {power_note(n_min, p=0.8)}")
    print(f"  ⚠ 逐任务比较时不要只看点估计——区间往往会重叠\n")

    # ---------- 4. 失败回合的步数分布 ----------
    timeouts = [x for x in manifest if not x["success"] and x["steps"] >= 500]
    if timeouts:
        print("=" * 60)
        print("  4. 超时回合的步数分布（全部贴着上限 = 效率问题）")
        print("=" * 60)
        steps = sorted(x["steps"] for x in timeouts)
        print(f"  失败回合 {len(steps)} 个，步数 {steps[0]}–{steps[-1]}")
        if steps[0] == steps[-1]:
            print("  ⚠ 全部等于步数上限：说明这些失败都是「没做完」而非「做错了」")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
