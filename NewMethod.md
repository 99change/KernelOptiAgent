# 二、系统核心结构（最终形态）

你的系统应该是：

```text
CUDA Kernel
→ Nsight Profiling
→ Bottleneck Extractor（核心模块）
→ Structured Optimization State（IR）
→ LLM Optimizer
→ New Kernel
```

---

# 三、你现在要实现的版本（最小可行方案）

## Step 1：定义 bottleneck schema（约 10 个）

建议从这几个开始（够用了）：

```text
- non_coalesced_memory
- memory_bound
- low_occupancy
- high_register_pressure
- warp_divergence
- compute_underutilized
- shared_memory_underused
- memory_latency_bound
```

---

## Step 2：设计结构化输出（关键）

不要用文本，用 JSON：

```json
{
  "non_coalesced_memory": {
    "score": 0.8,
    "evidence": {
      "stride": 32,
      "dram_transactions": "high"
    }
  },
  "low_occupancy": {
    "score": 0.3,
    "evidence": {
      "occupancy": 0.35
    }
  }
}
```

必须包含三点：

* score（不要用 true/false）
* evidence（来自 Nsight）
* 固定 schema（字段不能变）

---

## Step 3：用 LLM 做“填表器”（临时方案）

输入：

* Nsight 输出（可以简化）
* kernel 代码（可选）

输出：

→ 上面的 JSON

注意：

> **LLM 只负责“结构化解释”，不能自由生成文本分析**

---

## Step 4：接入优化器（LLM）

给 LLM 的输入不再是 Nsight，而是：

```text
Detected bottlenecks:
- non-coalesced memory (score=0.8)
- memory bound (score=0.7)

Goal:
- reduce global memory transactions

Constraints:
- shared memory < 48KB
```

👉 这一步会显著提升优化质量。

---

# 四、一个关键升级（强烈建议加）

## 👉 加 aggregation（防止 LLM 不稳定）

做法：

* 同一输入跑 LLM 3–5 次
* 对 score 取平均

```text
final_score = mean(scores)
```

👉 这一步可以显著提升稳定性。

---






# 五、你后续的演进路线（暂时不管很重要）

## Phase 1（现在）

* LLM 填 bottleneck
* 手动规则很少

---

## Phase 2（关键跃迁）

* 收集数据：

  * (Nsight → bottleneck)
* 训练 classifier（替代 LLM）

👉 到这里你就已经超过大多数工作了

---

## Phase 3（真正研究点）

* bottleneck → optimization policy
* 或：
* bottleneck → search space reduction

---

# 六、你这个方案的本质创新点（如果你做出来）

> **“构建了 hardware-aware structured feedback representation”**



