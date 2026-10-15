# vla_diag

[![tests](https://github.com/rzc156/vla-diag/actions/workflows/test.yml/badge.svg)](https://github.com/rzc156/vla-diag/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

**在你相信一个 VLA 评测结果之前，先确认它不是假的。**

一个轻量的评测诊断工具库。不训练模型，不跑仿真——只回答一个问题：

> **这个成功率数字，有没有可能是在测别的东西？**

---

## 为什么需要它

VLA 评测里最常见的失败不是「模型不行」，而是**评测本身在测别的东西**：

| 看起来像 | 实际在测 |
|---|---|
| 模型能力差（0% 成功率） | 权重文件没下完 |
| 动作能力不足（67% 成功率） | 推进效率不够（失败全是超时） |
| 方法 A 比 B 强 23 个点 | 重规划节奏不同（8 步 vs 50 步） |
| 换设备精度有差异 | 流匹配的采样噪声 |
| 世界模型能预警失败 | 画面变化量（成功回合变化更大） |

这类故障**不报错**，只会让你得出错误结论，然后基于它做几周无效优化。

详细的 7 个真实案例和原始数据见 **[CHECKS.md](CHECKS.md)**。

---

## 安装

```bash
pip install -e .
```

零依赖（只有 numpy）。不绑定任何仿真器或模型框架。

---

## 五个模块

| 模块 | 回答的问题 |
|---|---|
| `stats` | 这个差异是真的吗？n 要多大才够？ |
| `integrity` | 评测本身有没有崩？（检查点/动作/金丝雀） |
| `failures` | 失败是「超时」还是「做错了」？ |
| `headroom` | 值不值得做测试时选择？ |
| `report` | 生成带区间和失败形态的报告 |

---

## 快速开始

### 1. 跑评测前：确认检查点完整

```python
from vla_diag import verify_checkpoint

r = verify_checkpoint("~/models/GR00T-N1.7-LIBERO/libero_object")
if not r:
    raise SystemExit(f"❌ {r.detail}")
# ✅ 检查点完整性 — 2 个分片 / 1031 个张量全部通过
```

### 2. 报告成功率：永远带区间

```python
from vla_diag import format_ci
print(format_ci(18, 20))     # 18/20 = 90.0% [70, 97]
print(format_ci(2, 5))       #  2/5  = 40.0% [12, 77]   ← 宽度 65pp
```

### 3. 小样本：先知道能测出什么

```python
from vla_diag import power_note, required_n_two_proportions

print(power_note(10, p=0.8))
# n=10/组：样本量过小。连「80%→100%」都达不到 80% 功效（需 ≥32/组）

print(required_n_two_proportions(0.80, 0.85))
# 903   ← 想检测 5pp 提升，需要 900 回合/组
```

### 4. 跑完：拆开失败形态

```python
from vla_diag import classify_episodes, FailureTaxonomy
import json

manifest = json.load(open("videos/manifest.json"))
t = FailureTaxonomy(classify_episodes(manifest, max_steps=500))
print(t.table())
print(t.diagnose())
# 失败几乎全是超时（任务未完成，而非动作错误）
# → 优先做效率优化：缩短路径、调整重规划节奏、提前终止重启
```

### 5. 对比两个模型

```python
from vla_diag import comparison_report
print(comparison_report({"π0.5": man_a, "GR00T": man_b},
                        label_a="π0.5", label_b="GR00T"))
```

### 6. 决定要不要做选择类方法

```python
from vla_diag import headroom

# winners[i] = 第 i 个决策点上有几个候选成功（K 个候选）
hr = headroom(winners, K=4)
print(hr.table())
print(hr.verdict())
```

---

## 实战数据

本库的方法被用于以下实际评测（全部可复现）：

| 评测 | 规模 | 发现 |
|---|---|---|
| LIBERO 余量诊断 | 1598 决策点 / 900 配对回合 | 96% 的决策点无可选空间，完美选择器 +0.22pp (p=0.93) |
| π0.5 vs GR00T N1.7 | 4 套件 × 400 回合 | GR00T 平均 94.3%；差异归因需扣除重规划节奏 |
| 世界模型失败预警 | 3 模型 × 100 回合 | 任务内 AUC 0.25–0.53（等于随机） |
| GPU 部署验证 | 三方对比 | 跨设备差异 (0.014) < 采样噪声 (0.025)，加速 17.4× |

---

## 设计原则

1. **零依赖** —— 只有 numpy，能塞进任何环境
2. **不预设框架** —— 输入是普通的 list/dict，不绑定 LeRobot/robosuite
3. **默认保守** —— 区间默认 95%，功效默认 0.8
4. **不报无信息的数字** —— 样本量不足时明确说「测不出」，而不是给个点估计

---

## 还没做的事（欢迎补充）

- 真机数据的配对检验辅助（初始位姿不可复现时的降级方案）
- 多任务套件的分层 bootstrap
- 与 W&B / TensorBoard 的对接

---

## 参考

- [CHECKS.md](CHECKS.md) —— 7 个真实踩坑案例与检出手段
- Wilson (1927) —— score interval，小样本比例区间的正确做法
- McNemar (1947) —— 配对二分类检验
