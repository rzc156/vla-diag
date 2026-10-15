"""自检：确认核心函数在已知输入下给出已知答案。

跑法:  python tests/test_vla_diag.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vla_diag import (  # noqa: E402
    FailureTaxonomy, check_actions, classify_episodes, format_ci, headroom,
    mcnemar, power_note, required_n_two_proportions, two_proportion_p,
    verify_checkpoint, wilson_ci,
)

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def main():
    print("=== stats ===")
    lo, hi = wilson_ci(0, 5)
    check("wilson(0,5) 上界 > 0（Wald 会给 [0,0]）", hi > 0.3, f"得到 {hi:.3f}")
    lo, hi = wilson_ci(5, 5)
    check("wilson(5,5) 下界 < 1（Wald 会给 [1,1]）", lo < 0.7, f"得到 {lo:.3f}")
    check("wilson(18,20) ≈ [0.70, 0.97]",
          abs(wilson_ci(18, 20)[0] - 0.701) < 0.01 and abs(wilson_ci(18, 20)[1] - 0.966) < 0.01,
          str(wilson_ci(18, 20)))
    check("format_ci 可读", format_ci(2, 5) == "2/5 = 40.0% [12, 77]", format_ci(2, 5))

    p = two_proportion_p(81, 100, 95, 100)
    check("81/100 vs 95/100 显著 (p<0.01)", p < 0.01, f"p={p:.4f}")

    m = mcnemar(5, 20)
    check("McNemar b=5,c=20 显著", m["p"] < 0.01, f"p={m['p']:.4f}")
    check("McNemar 无不一致对时 p=1", mcnemar(0, 0)["p"] == 1.0)

    n5 = required_n_two_proportions(0.80, 0.85)
    check("检测 5pp 需 ~900/组", 850 <= n5 <= 950, f"得到 {n5}")

    print("\n=== headroom（对齐已发表的数字）===")
    hist = {0: 1331, 1: 40, 2: 12, 3: 12, 4: 203}
    winners = [k for k, c in hist.items() for _ in range(c)]
    hr = headroom(winners, K=4)
    check("决策点数 = 1598", hr.n_decisions == 1598, str(hr.n_decisions))
    check("全失败 = 83.29%", abs(100 * hr.pct_all_fail - 83.29) < 0.1,
          f"{100*hr.pct_all_fail:.2f}%")
    check("判别性 = 4.005%", abs(100 * hr.pct_discriminative - 4.005) < 0.01,
          f"{100*hr.pct_discriminative:.3f}%")
    check("完美选择器增益 = 2.44pp", abs(hr.oracle_gap_pp - 2.44) < 0.01,
          f"{hr.oracle_gap_pp:.2f}pp")
    check("判别性低时判定为「没有余量」", "没有余量" in hr.verdict())

    hr2 = headroom([0, 1, 2, 4] * 25, K=4)
    check("判别性高时判定为「余量充足」", "余量充足" in hr2.verdict(), hr2.verdict())

    print("\n=== failures ===")
    man = [{"task": "t/0", "ep": i, "success": i < 7, "steps": 200 if i < 7 else 500}
           for i in range(10)]
    t = FailureTaxonomy(classify_episodes(man, max_steps=500))
    check("成功率 70%", abs(t.success_rate - 0.7) < 1e-9, str(t.success_rate))
    check("失败全分类为超时", t.counts["timeout"] == 3 and t.counts["terminated"] == 0,
          str(dict(t.counts)))
    check("诊断为效率问题", "效率" in t.diagnose(), t.diagnose())

    man2 = [{"task": "t/0", "ep": i, "success": i < 7,
             "steps": 200 if i < 7 else 150} for i in range(10)]
    t2 = FailureTaxonomy(classify_episodes(man2, max_steps=500))
    check("提前终止诊断为能力问题", "能力" in t2.diagnose(), t2.diagnose())

    print("\n=== integrity ===")
    import random
    random.seed(0)
    healthy = [[0.1 + 0.01 * i, 0.2 - 0.005 * i, -1.0 if i % 3 else 1.0]
               for i in range(10)]
    r = check_actions(healthy, action_dim=3)
    check("健康动作通过", bool(r), r.detail)
    r = check_actions([[0.1 + 0.01 * i, float("nan"), -1.0] for i in range(10)], action_dim=3)
    check("NaN 被抓到", not r and "NaN" in r.detail, r.detail)
    r = check_actions([[0.1 + 0.01 * i, 0.2, 0.3] for i in range(10)], action_dim=3)
    check("死维度被抓到", not r and "方差为 0" in r.detail, r.detail)

    with tempfile.TemporaryDirectory() as d:
        check("空目录被拒", not verify_checkpoint(d))

    print("\n" + "=" * 46)
    print(f"  {PASS} 通过 / {FAIL} 失败")
    print("=" * 46)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
