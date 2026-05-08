# KernelOptiAgent

自动优化 CUDA kernel 的 3-Agent 系统。输入 naive kernel，输出带注释的优化版本。

核心思路：用 `nvcc --ptxas-options=-v` + `ncu` 采集真实硬件数据，驱动 LLM 做 hardware-grounded 的瓶颈分析，替代"让 LLM 读代码猜瓶颈"的做法。

---

## 项目结构

```
KernelOptiAgent/
├── agents/
│   ├── base.py              # Agent 基类（LLM 调用封装）
│   ├── profiler.py          # 硬件 Profiling Agent（纯工具流，零 LLM）
│   ├── analyzer.py          # 瓶颈分析 Agent（LLM 填表，以硬件数据为输入）
│   └── optimizer.py         # 优化执行 Agent
├── tools/
│   ├── kernel_tools.py      # nvcc 编译 + ptxas 解析 + ncu 采集 + 计时
│   └── knowledge_retrieval.py  # strategy → 示例代码检索
├── knowledge/               # CUDA 优化模式示例库
│   ├── float4_vectorized.cu
│   ├── latency_hiding.cu
│   ├── shared_memory_tiling.cu
│   ├── loop_unrolling.cu
│   ├── occupancy_tuning.cu
│   └── restrict_qualifiers.cu
├── core/
│   ├── models.py            # 数据模型（HardwareProfile / BottleneckIR）
│   ├── config.py            # LLM 配置（Qwen / Dashscope）
│   └── memory.py
├── main.py                  # 主入口
├── baseline_e2e.py          # 端到端 baseline 对比（裸 LLM 直接改）
└── results/                 # 输出目录
```

---

## 数据流

```
Input Kernel Code
        ↓
  [ProfilerAgent]  ← 纯工具流，零 LLM
  ├─ nvcc --ptxas-options=-v → PtxasInfo { registers, smem_bytes, spill }
  ├─ 编译 + 3次计时 → baseline_time_ms
  └─ ncu → NcuMetrics { memory_throughput%, compute_throughput%, dram%, occupancy% }
  → HardwareProfile
        ↓
  [AnalyzerAgent]
  - 将 HardwareProfile 真实数字注入 prompt
  - LLM 以"填表"方式输出 BottleneckIR（固定 schema，跑 3 次取均值）
  → BottleneckIR { non_coalesced_memory: {score, evidence}, ... }
  → strategies（按 score 排序，score >= 0.4 激活）
        ↓
  [OptimizerAgent]
  ├─ For each strategy:
  │  ├─ knowledge_retrieval(strategy) → 注入示例代码
  │  ├─ LLM 改写代码（hardware-aware prompt）
  │  ├─ 编译 + 实测，编译失败最多 2 次 self-repair
  │  ├─ 异常检测（>10x 慢则丢弃）
  │  └─ improvement > 5% 才保留
  ├─ E2E pass：无约束让 LLM 自由优化，与策略结果竞争
  → best optimized_code + speedup
        ↓
Output: results/optimized_kernel.cu（含修改说明注释）+ report
```

---

## HardwareProfile 数据结构

`ProfilerAgent` 一次采集，产出两类数据：

**PtxasInfo（编译期，来自 `nvcc --ptxas-options=-v` stderr）**

| 字段 | 含义 |
|------|------|
| `registers` | 每线程寄存器数 |
| `smem_bytes` | 静态 shared memory 字节 |
| `spill_stores` / `spill_loads` | 寄存器 spill 到 local memory 的量 |

**NcuMetrics（运行时，来自 `ncu --metrics ...`）**

| 字段 | 含义 |
|------|------|
| `memory_throughput_pct` | 整体内存吞吐 % of peak |
| `compute_throughput_pct` | SM 计算吞吐 % of peak |
| `dram_throughput_pct` | DRAM 带宽 % of peak |
| `l2_throughput_pct` | L2 Cache 吞吐 % of peak |
| `achieved_occupancy_pct` | 实际 warp occupancy % |

---

## Structured Bottleneck IR

LLM 不写自由文本，只填固定 schema 的 JSON，每个字段输出 `score`（0~1）和 `evidence`（含真实测量值）：

| 字段 | 含义 | 主要依据 |
|------|------|---------|
| `non_coalesced_memory` | 非合并访存 | 代码访问模式 |
| `memory_bound` | 内存带宽瓶颈 | `memory_throughput_pct` |
| `low_occupancy` | GPU 占用率低 | `achieved_occupancy_pct` + `registers` |
| `high_register_pressure` | 寄存器压力大 | `registers` + `spill_stores` |
| `warp_divergence` | Warp 分支分歧 | 代码 branch 结构 |
| `compute_underutilized` | 计算资源未充分利用 | `compute_throughput_pct` |
| `shared_memory_underused` | Shared memory 未利用 | `smem_bytes` + 代码 |
| `memory_latency_bound` | 内存延迟瓶颈 | `memory_throughput_pct` vs `compute_throughput_pct` |

- **3 次聚合**：同一 kernel 跑 LLM 3 次，score 取均值，防止单次不稳定
- **阈值激活**：`score >= 0.4` 才触发对应优化策略

---

## 知识库注入

`OptimizerAgent` 构建 prompt 时，按策略关键词检索 `knowledge/` 目录中的示例代码并注入，让 LLM 做"代码改写"而非"知识发明"。

---

## 快速上手

```bash
export DASHSCOPE_API_KEY=your_key

python main.py --input examples/vector_add.cu
python main.py --input your_kernel.cu --model qwen-max --rounds 5
```

---

## 当前进度

### ✅ 已完成

- **Profiler 纯工具流**：零 LLM，三步流水线（ptxas → 计时 → ncu），输出完整 `HardwareProfile`
- **Hardware-grounded Analyzer**：删除所有正则假静态分析，LLM prompt 直接注入真实测量数字；evidence 字段写实测值而非代码猜测
- **流程顺序调整**：Profiler 先跑，Analyzer 拿到硬件数据后再分析（原来是 Analyzer 先跑，手里没有任何硬件数据）
- **Self-repair**：编译失败时 LLM 最多自动修复 2 次
- **异常检测**：>10x 慢的结果自动拒绝

### 📋 下一步

- **Optimizer CoT 规划**：代码生成前让 LLM 先输出结构化修改方案（tile size、unroll factor 等），再按方案编码，便于 ablation "有规划 vs 无规划"
- **Best-of-N TTS**：同一策略生成 N 个变体，取最优。`nvcc` 编译 + 实测 speedup 本身就是完美 Verifier，无需训练 Reward Model
- **KernelBench 批量实验**：选 20-30 个 kernel，报告 speedup 分布；与 `baseline_e2e.py` 对比作为 ablation

---

## Test-Time Scaling

本项目天然适合 TTS——GPU 实测 speedup 本身就是完美的 Verifier。

| TTS 要素 | 本项目中的对应物 |
|----------|----------------|
| 候选生成（Best-of-N） | 同一策略让 LLM 生成 N 种变体 |
| Verifier | `nvcc` 编译通过 + 实测 speedup（无需训练） |
| 搜索策略 | 当前贪心串行；可升级为 beam search |
| 计算预算控制 | 最多尝试 K 次，或总时间上限 T 秒 |
