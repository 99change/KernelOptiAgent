# GPU Kernel 优化 Agent - 简化实用架构

> 这是一个可以直接编码实现的架构设计，保留核心功能，摒弃过度设计。

## 核心理念

- **3 个 Agent**：分工简单，职责清晰
- **工具即函数**：不要过度抽象
- **流程驱动**：main.py 中清晰可见
- **增量扩展**：先能跑，再优化

---

## 项目结构

```
KernelOptiAgent/
│
├── agents/
│   ├── __init__.py
│   ├── base.py                  # [1] Agent 基类（~50 行）
│   ├── analyzer.py              # [2] 代码分析 Agent
│   ├── profiler.py              # [3] 性能测评 Agent  
│   └── optimizer.py             # [4] 优化执行 Agent
│
├── tools/
│   ├── __init__.py
│   └── kernel_tools.py          # [5] 所有工具函数集合
│
├── core/
│   ├── __init__.py
│   ├── models.py                # [6] 数据模型定义
│   ├── memory.py                # [7] 简单记忆系统
│   └── config.py                # [8] 配置管理
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                # 日志
│   └── errors.py                # 错误处理
│
├── main.py                      # [9] 主程序入口
├── SIMPLE_DESIGN.md             # 本文件
└── requirements.txt
```

## 数据流（简化版）

```
Input Kernel Code
        ↓
  [AnalyzerAgent]
  - 代码解析
  - 识别优化机会
  → AnalysisResult {strategies, bottlenecks}
        ↓
  [ProfilerAgent]
  - 基准测评
  - 性能数据采集
  → ProfileResult {baseline_time, metrics}
        ↓
  [OptimizerAgent]
  ├─ For each strategy:
  │  ├─ 生成优化代码
  │  ├─ 编译 + 验证
  │  ├─ 测评性能
  │  └─ 决策保留/丢弃
  → OptimizationResult {optimized_code, speedup}
        ↓
Output Optimized Kernel + Report
```

---

## 关键特点

| 特性 | 实现方式 |
|------|--------|
| **模块化** | 3 个 Agent + 工具库，职责清晰 |
| **可实现** | 每个文件 < 200 行代码 |
| **可扩展** | 新增 Agent 只需继承 BaseAgent |
| **可测试** | 每个 Agent 独立，容易单测 |
| **可观测** | main.py 中清晰看到全流程 |
| **核心完整** | 有分析、测评、优化三个阶段 |

---

## 知识补偿方案（解决 LLM CUDA 能力不足问题）

> 背景：通用 LLM 没有深度 CUDA 优化知识，直接让它"生成优化代码"效果差。
> 解法：在推理时把专业知识外挂进去，让 LLM 做"代码改写"而非"知识发明"。

### 三层补偿策略

**1. 知识库 + Few-shot 注入（主要手段）**

新增 `knowledge/` 目录，存放带完整注释的 CUDA 优化模式示例代码。
`AnalyzerAgent` 识别出策略后，`OptimizerAgent` 在构建 prompt 时自动检索并注入：

```
prompt = """
这是 shared memory tiling 的标准实现模式：
--- 示例开始 ---
[knowledge/shared_memory_tiling.cu 的内容]
--- 示例结束 ---

现在，按照上面的模式，将以下 naive kernel 改写：
[目标 kernel 代码]
"""
```

新增文件：
```
knowledge/
├── shared_memory_tiling.cu      # Tiled matmul 示例 + 注释
├── memory_coalescing.cu         # Coalesced access 示例
├── loop_unrolling.cu            # #pragma unroll 示例
├── warp_primitives.cu           # __shfl_down_sync 等示例
└── occupancy_tuning.cu          # block size / register 调优示例

tools/
└── knowledge_retrieval.py       # strategy → 相关示例代码检索
```

**2. Profile 数据反馈驱动（迭代提升手段）**

将 `ProfilerAgent` 的真实 ncu/nvprof 数据反馈给 `OptimizerAgent`，
每轮优化不再盲猜，而是基于具体数字做决策：

```
数据流变化：
  ProfileResult { exec_time_ms, bandwidth_util, occupancy, ... }
        ↓ (作为 context 传入)
  OptimizerAgent 的下一轮 prompt：
  "当前 bandwidth 利用率 40%，occupancy 62%，
   请针对 bandwidth 瓶颈选择下一步优化策略"
```

**3. 模板填充模式（最稳健的 fallback）**

对高频优化场景（如 matmul tiling），提供预写好的代码模板，
让 LLM 只负责填入参数（tile size、block dim 等），不负责生成结构：

```python
# tools/kernel_tools.py 中新增
def apply_tiling_template(kernel_code: str, tile_size: int) -> str:
    """用 LLM 抽取参数 + 填入标准 tiling 模板"""
    ...
```

### 更新后的数据流

```
Input Kernel Code
        ↓
  [AnalyzerAgent]
  - 静态分析
  - 识别策略列表
  → AnalysisResult {strategies, bottlenecks}
        ↓
  [ProfilerAgent]
  - 基准测评
  → ProfileResult {baseline_time, bandwidth_util, occupancy}
        ↓
  [OptimizerAgent]
  ├─ For each strategy:
  │  ├─ knowledge_retrieval(strategy) → 示例代码          ← 新增
  │  ├─ 构建 prompt（含示例 + profile 数据）               ← 新增
  │  ├─ LLM 改写代码（基于示例，非凭空发明）
  │  ├─ 编译 + 验证
  │  ├─ 测评性能
  │  └─ 将结果写回 profile context，驱动下一轮             ← 新增
  → OptimizationResult {optimized_code, speedup}
        ↓
Output Optimized Kernel + Report
```

### 实现优先级

| 优先级 | 方案 | 工作量 | 效果 |
|--------|------|--------|------|
| P0 | 知识库 + Few-shot 注入 | 小（写示例代码） | 高（直接提升代码质量） |
| P1 | Profile 数据反馈 | 小（修改数据流） | 中（迭代更精准） |
| P2 | 模板填充 | 中（写模板） | 高但覆盖场景有限 |

---

## Test-Time Scaling（TTS）方向

> 核心洞察：训练时模型固定了，但推理时投入更多计算量可以持续提升效果。
> 在有明确验证标准的任务（数学、代码）上，性能 ∝ log(推理计算量)。

### 本项目天然适合 TTS 的原因

NLP 任务做 TTS 的最大障碍是需要训练 Reward Model 来评估输出质量。
**本项目不需要**——`nvcc` 编译器 + GPU 实测 speedup 本身就是完美的硬指标 Verifier。

| TTS 要素 | 本项目中的对应物 |
|----------|----------------|
| 候选生成（Best-of-N） | 同一策略让 LLM 生成 N 种优化代码变体 |
| **Verifier** | `nvcc` 编译通过 + 实测 speedup 数值（无需训练） |
| 搜索策略 | 当前是贪心串行；可升级为 MCTS / Beam Search |
| 计算预算控制 | 设定最多尝试 K 次，或总时间上限 T 秒 |

### 三种 TTS 实现路径

**Best-of-N（最易落地）**
```
同一策略 → LLM 生成 N 个变体（temperature > 0）
         → 全部编译 + 测评
         → 取 speedup 最高的保留
```

**Beam Search**
```
维护 top-K 个当前最优代码版本
每轮对每个版本继续生成下一步优化
剪枝 speedup 低于阈值的分支
```

**MCTS（计算预算充足时）**
```
节点 = 某个优化状态的代码
展开 = LLM 生成子优化
评分 = 实测 speedup（UCB 公式平衡探索/利用）
适合参数搜索空间大的场景（tile_size × block_dim × unroll_factor）
```

### 对 OptimizerAgent 的影响

当前实现是每个策略只生成一次代码（greedy）：
```python
optimized_code = self._generate_optimized_code(best_code, strategy)  # 只调用一次
```

引入 TTS 后改为 Best-of-N：
```python
candidates = [self._generate_optimized_code(best_code, strategy) for _ in range(N)]
# 全部编译测评，取最优
best_candidate = max(candidates, key=lambda c: compile_and_test(c).speedup)
```

计算量增加 N 倍，但优化质量可观地提升，且完全不需要改模型。


