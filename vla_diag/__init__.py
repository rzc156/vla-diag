"""vla_diag — 在你相信一个 VLA 评测结果之前，先确认它不是假的。

五个模块
--------
stats      置信区间、配对检验、功效分析（「n=5 能测出多大差异」）
integrity  检查点/环境/动作输出的完整性检查（防静默失败）
failures   失败形态拆解（超时 vs 动作错误——改进方向完全不同）
headroom   余量诊断（值不值得做测试时选择）
report     带区间和失败形态的报告生成

最短可用路径
------------
    from vla_diag import verify_checkpoint, headroom, FailureTaxonomy
    from vla_diag import classify_episodes, format_ci, power_note

    # 1. 跑评测前：确认检查点不是半成品
    assert verify_checkpoint("~/models/xxx")

    # 2. 跑完：报告必须带区间
    print(format_ci(18, 20))          # '18/20 = 90.0% [70, 97]'

    # 3. 小样本要标出可检测下限
    print(power_note(5))              # 提醒 n=5 测不出小差异

    # 4. 失败要拆形态
    print(FailureTaxonomy(classify_episodes(manifest)).diagnose())
"""
from .stats import (
    wilson_ci, wilson_interval, format_ci, summarize_binary,
    two_proportion_z, two_proportion_p, mcnemar, mcnemar_exact,
    paired_bootstrap, required_n_two_proportions, min_detectable_effect,
)
from .integrity import (
    CheckResult, verify_checkpoint, verify_safetensors,
    action_stats, check_actions, determinism_check, canary_check,
)
from .failures import (
    EpisodeOutcome, classify_episodes, FailureTaxonomy, failure_summary_table,
)
from .headroom import (
    Headroom, headroom, headroom_curve, required_k, verdict_text,
)
from .report import rollout_report, comparison_report, power_note

__version__ = "0.1.0"
__all__ = [
    # stats
    "wilson_ci", "wilson_interval", "format_ci", "summarize_binary",
    "two_proportion_z", "two_proportion_p", "mcnemar", "mcnemar_exact",
    "paired_bootstrap", "required_n_two_proportions", "min_detectable_effect",
    # integrity
    "CheckResult", "verify_checkpoint", "verify_safetensors",
    "action_stats", "check_actions", "determinism_check", "canary_check",
    # failures
    "EpisodeOutcome", "classify_episodes", "FailureTaxonomy", "failure_summary_table",
    # headroom
    "Headroom", "headroom", "headroom_curve", "required_k", "verdict_text",
    # report
    "rollout_report", "comparison_report", "power_note",
]
