"""完整性检查：在相信一个评测结果之前，先确认它不是假的。

这个模块的存在理由是一个真实事故
----------------------------------
在一次 GR00T 评测中，某个套件的所有回合 100% 失败，且**不报任何错**。
排查后确认：评测启动时检查点权重仍在下载中，`safetensors` 用 mmap 读取，
未落盘的区域被读成零值 → 权重半坏 → 模型输出垃圾动作。

症状看起来完全像「这个模型不行」，而不是「文件没下完」。
这类故障会浪费几天到几周，因为没人会怀疑基础设施。

本模块把所有「先确认评测本身没问题」的检查集中起来。
"""
from __future__ import annotations

import collections
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "CheckResult", "verify_checkpoint", "verify_safetensors",
    "action_stats", "check_actions", "determinism_check", "canary_check",
]


# ---------------------------------------------------------------- 检查结果

@dataclass
class CheckResult:
    ok: bool
    name: str
    detail: str = ""
    data: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok

    def line(self) -> str:
        mark = "✅" if self.ok else "❌"
        return f"{mark} {self.name}" + (f" — {self.detail}" if self.detail else "")


# ---------------------------------------------------------------- 检查点

def verify_safetensors(path: str | Path) -> CheckResult:
    """校验单个 safetensors 文件可完整读取。

    重点不是「文件存在」，而是「**能读出所有张量**」。
    正在下载中的文件会存在但读失败（或读出零）。
    """
    p = Path(path)
    if not p.exists():
        return CheckResult(False, f"safetensors 存在", f"{p.name} 不存在")
    try:
        from safetensors import safe_open
        with safe_open(str(p), framework="np") as sf:
            keys = list(sf.keys())
        if not keys:
            return CheckResult(False, f"safetensors 非空", f"{p.name} 无张量")
        return CheckResult(True, f"safetensors 可读", f"{p.name}: {len(keys)} 张量")
    except Exception as e:
        return CheckResult(False, f"safetensors 可读",
                           f"{p.name} 读取失败({type(e).__name__}) "
                           f"— 常见原因：仍在下载中")


def verify_checkpoint(path: str | Path, min_shard_bytes: int = 1_000_000) -> CheckResult:
    """校验分片检查点完整：index 声明的张量数 == 实际可读出的张量数。

    用法：
        r = verify_checkpoint("~/models/GR00T-N1.7-LIBERO/libero_object")
        if not r: raise SystemExit(r.detail)
    """
    ck = Path(os.path.expanduser(str(path)))
    if not ck.is_dir():
        return CheckResult(False, "检查点目录", f"{ck} 不存在")

    idx = ck / "model.safetensors.index.json"
    single = ck / "model.safetensors"

    # --- 单文件模型 ---
    if not idx.exists():
        if single.exists():
            return verify_safetensors(single)
        shards = sorted(ck.glob("*.safetensors"))
        if not shards:
            return CheckResult(False, "检查点权重",
                               f"{ck} 下没有 .safetensors（是否是 LeRobot 格式？）")
        for s in shards:
            r = verify_safetensors(s)
            if not r:
                return r
        return CheckResult(True, "检查点权重", f"{len(shards)} 个分片均可读")

    # --- 分片模型：逐 shard 比对张量数 ---
    try:
        wm = json.loads(idx.read_text())["weight_map"]
    except Exception as e:
        return CheckResult(False, "index.json 可解析", f"{type(e).__name__}: {e}")

    need = collections.Counter(wm.values())
    bad: list[str] = []
    total = 0
    for shard, n in sorted(need.items()):
        sp = ck / shard
        if not sp.exists():
            bad.append(f"{shard} 缺失")
            continue
        if sp.stat().st_size < min_shard_bytes:
            bad.append(f"{shard} 仅 {sp.stat().st_size/1e6:.1f}MB（疑似未下完）")
            continue
        try:
            from safetensors import safe_open
            with safe_open(str(sp), framework="np") as sf:
                got = len(list(sf.keys()))
        except Exception as e:
            bad.append(f"{shard} 读取失败({type(e).__name__})")
            continue
        if got != n:
            bad.append(f"{shard} 张量数 {got} != index 声明的 {n}")
        else:
            total += got

    if bad:
        return CheckResult(False, "检查点完整性", "; ".join(bad),
                           {"shards": len(need), "ok_shards": len(need) - len(bad)})
    return CheckResult(True, "检查点完整性",
                       f"{len(need)} 个分片 / {total} 个张量全部通过",
                       {"shards": len(need), "tensors": total})


# ---------------------------------------------------------------- 动作输出

def action_stats(actions) -> dict:
    """诊断一组动作张量的健康度。

    能抓到的退化模式：
      - NaN / Inf（精度或网络问题）
      - 某些维度方差为 0（夹爪卡死、某轴不动）
      - 全常数（模型没在推理）
    """
    a = np.asarray(actions, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    fin = np.isfinite(a)
    out = {
        "shape": tuple(a.shape),
        "n_nan": int((~fin).sum()),
        "global_std": float(np.nanstd(a)) if fin.any() else float("nan"),
        "per_dim_std": [float(x) for x in np.nanstd(a, axis=0)],
        "per_dim_min": [float(x) for x in np.nanmin(a, axis=0)],
        "per_dim_max": [float(x) for x in np.nanmax(a, axis=0)],
    }
    out["dead_dims"] = [i for i, s in enumerate(out["per_dim_std"]) if s < 1e-8]
    return out


def check_actions(actions, action_dim: int | None = None,
                  dim_names: list[str] | None = None) -> CheckResult:
    """检查动作张量是否健康（NaN、死维度、全常数）。"""
    st = action_stats(actions)
    problems: list[str] = []

    if st["n_nan"] > 0:
        problems.append(f"{st['n_nan']} 个 NaN/Inf")

    if action_dim is not None and st["shape"][-1] != action_dim:
        problems.append(f"最后一维 {st['shape'][-1]} != 期望 {action_dim}")

    if st["global_std"] < 1e-8:
        problems.append("整体方差为 0（模型可能没在推理）")

    for i in st["dead_dims"]:
        nm = dim_names[i] if dim_names and i < len(dim_names) else f"dim{i}"
        problems.append(f"{nm} 方差为 0（该轴不动）")

    if problems:
        return CheckResult(False, "动作输出健康", "; ".join(problems), st)
    return CheckResult(True, "动作输出健康",
                       f"{st['shape']}, 全局 std={st['global_std']:.4f}", st)


# ---------------------------------------------------------------- 确定性

def determinism_check(fn, n: int = 3, atol: float = 1e-6) -> CheckResult:
    """跑同一输入 n 次，看输出是否一致。

    **重要**：对 diffusion / flow-matching 策略，不一致是**正常的**——
    它们每次采样不同的噪声。此时应当用 atol=np.inf 跳过，
    或改用「多次运行的方差」来判断，而不是强制确定性。
    """
    try:
        outs = [np.asarray(fn(), dtype=float) for _ in range(n)]
    except Exception as e:
        return CheckResult(False, "确定性检查", f"调用失败: {type(e).__name__}: {e}")
    base = outs[0]
    maxdiff = max(float(np.max(np.abs(o - base))) for o in outs[1:]) if n > 1 else 0.0
    ok = maxdiff <= atol
    return CheckResult(ok, "确定性",
                       f"{n} 次调用最大差 {maxdiff:.2e}（容差 {atol:.0e}）",
                       {"max_diff": maxdiff, "n": n})


# ---------------------------------------------------------------- 金丝雀

def canary_check(rollout_fn, n_episodes: int = 3,
                 min_success_rate: float = 0.5) -> CheckResult:
    """金丝雀任务：在正式评测前，先跑一个**已知模型能做**的简单任务。

    `rollout_fn(i) -> bool` 应当运行第 i 个金丝雀回合并返回是否成功。

    如果金丝雀（一个训练集内、模型必然会的任务）都失败，
    那问题在基础设施而不是模型——**不要浪费几小时跑正式评测**。

    这正是本次事故中缺失的一环：0% 成功率本来一眼就该可疑。
    """
    try:
        res = [bool(rollout_fn(i)) for i in range(n_episodes)]
    except Exception as e:
        return CheckResult(False, "金丝雀任务", f"运行失败: {type(e).__name__}: {e}")
    k = sum(res)
    rate = k / n_episodes
    ok = rate >= min_success_rate
    msg = (f"金丝雀成功率 {k}/{n_episodes} = {rate:.0%}"
           f"（要求 >= {min_success_rate:.0%}）")
    if not ok:
        msg += " — ⚠ 基础设施可疑，先不要跑正式评测"
    return CheckResult(ok, "金丝雀任务", msg, {"k": k, "n": n_episodes, "rate": rate})
