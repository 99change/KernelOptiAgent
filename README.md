# KernelOptiAgent（已上线部分）

自动优化 CUDA kernel 的 3-Agent 系统。输入 naive kernel，输出带注释的优化版本。

---

## 项目结构

```
KernelOptiAgent/
├── agents/
│   ├── base.py              # Agent 基类
│   ├── analyzer.py          # 瓶颈分析 Agent
│   ├── profiler.py          # 基准测评 Agent
│   └── optimizer.py         # 优化执行 Agent
├── tools/
│   ├── kernel_tools.py      # nvcc 编译 + GPU 测评
│   └── knowledge_retrieval.py  # strategy → 示例代码检索
├── knowledge/               # CUDA 优化模式示例库
│   ├── float4_vectorized.cu     # float4 向量化 + __ldg 正确用法
│   ├── latency_hiding.cu        # 延迟隐藏：__ldg / ILP / 软件流水线
│   ├── shared_memory_tiling.cu  # Tiled matmul / shared memory 数据复用
│   ├── loop_unrolling.cu        # #pragma unroll
│   ├── occupancy_tuning.cu      # __launch_bounds__ / block size 调优
│   └── restrict_qualifiers.cu   # __restrict__ 指针别名消除
├── core/
│   ├── models.py            # 数据模型（含 BottleneckIR）
│   ├── config.py            # LLM 配置（Qwen / Dashscope）
│   └── memory.py
├── main.py                  # 主入口
├── baseline_e2e.py/         # 端到端对比
└── results/                 # 输出目录
```

---

## 数据流

```
Input Kernel Code
        ↓
  [AnalyzerAgent]
  - LLM 以"填表"方式输出 BottleneckIR（固定 schema，跑 3 次取均值）
  → BottleneckIR { non_coalesced_memory: {score, evidence}, ... }
  → strategies（按 score 排序）
        ↓
  [ProfilerAgent]
  - 编译 + GPU 实测
  → baseline_time_ms
        ↓
  [OptimizerAgent]
  ├─ For each strategy (按 score 排序):
  │  ├─ knowledge_retrieval(strategy) → 注入示例代码
  │  ├─ 构建 hardware-aware prompt（IR score + constraints + 示例）
  │  ├─ LLM 改写代码
  │  ├─ 编译 + 实测
  │  ├─ 异常检测（>10x 慢则丢弃）
  │  └─ 超过阈值才保留
  → best optimized_code + speedup
        ↓
Output: results/optimized_kernel.cu（含修改说明注释）+ report
```

---

## Structured Bottleneck IR

LLM 不写自由文本描述瓶颈，而是填写固定 schema 的 JSON，每个字段输出 `score`（0~1）和 `evidence`：

| 字段 | 含义 |
|------|------|
| `non_coalesced_memory` | 非合并访存 |
| `memory_bound` | 内存带宽瓶颈 |
| `low_occupancy` | GPU 占用率低 |
| `high_register_pressure` | 寄存器压力大 |
| `warp_divergence` | Warp 分支分歧 |
| `compute_underutilized` | 计算资源未充分利用 |
| `shared_memory_underused` | Shared memory 未利用 |
| `memory_latency_bound` | 内存延迟瓶颈 |

- **3 次聚合**：同一 kernel 跑 LLM 3 次，score 取均值，防止单次不稳定
- **阈值激活**：`score >= 0.4` 才触发对应优化策略
- **约束推导**：高 score 的瓶颈影响 prompt 约束（如 `high_register_pressure` 高 → 提示 LLM 避免新增变量）

---

## 知识库注入

`OptimizerAgent` 构建 prompt 时，按策略关键词检索 `knowledge/` 目录中的示例代码并注入，让 LLM 做"代码改写"而非"知识发明"。每个示例文件带详细注释，包含常见错误用法说明（如 `__ldg` 必须传指针而非值）。

---

## 快速上手

```bash
# 需要环境变量
export DASHSCOPE_API_KEY=your_key

python main.py --input examples/vector_add.cu
python main.py --input your_kernel.cu --model qwen-max --rounds 5
```

---

# 待完成

## Test-Time Scaling（TTS）

> 核心洞察：训练时模型固定，但推理时投入更多计算量可持续提升效果。
> 本项目天然适合 TTS——`nvcc` 编译 + GPU 实测 speedup 本身就是完美的 Verifier，无需训练 Reward Model。

| TTS 要素 | 本项目中的对应物 |
|----------|----------------|
| 候选生成（Best-of-N） | 同一策略让 LLM 生成 N 种变体 |
| Verifier | `nvcc` 编译通过 + 实测 speedup（无需训练） |
| 搜索策略 | 当前贪心串行；可升级为 MCTS / Beam Search |
| 计算预算控制 | 最多尝试 K 次，或总时间上限 T 秒 |

**Best-of-N（最易落地）**：同一策略生成 N 个变体，全部编译测评，取最优保留。计算量增加 N 倍，优化质量可观提升，不需要改模型。

**Profile 数据反馈**：将 ProfilerAgent 的 bandwidth/occupancy 等指标传入 OptimizerAgent prompt，基于具体数字选策略而非静态分析。

## Phase 2（关键跃迁）

* 收集数据：

  * (Nsight → bottleneck)
* 训练 classifier（替代 LLM）

---

## Phase 3（真正研究点）

* bottleneck → optimization policy
* 或：
* bottleneck → search space reduction