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

---

## [1] Agent 基类 (base.py)

**大小**：~50 行代码

```
BaseAgent:
├── __init__(name, llm_client)
├── _think(context) → reasoning      # LLM 推理
├── _call_tool(tool_name, **args)   # 调用工具
├── _store_memory(key, value)       # 存储信息
├── execute(task) → result          # 主执行
└── logger, memory 实例
```

**简化点**：
- ✅ 只有 1 个 LLM 客户端（OpenAI/Copilot）
- ✅ Tool 调用是直接函数调用，不是复杂的序列化
- ✅ 同步执行（没有 async）

---

## [2] Analyzer Agent (analyzer.py)

**职责**：分析 kernel 代码，找出优化机会

```python
class AnalyzerAgent(BaseAgent):
    """分析代码特征和优化机会"""
    
    def execute(self, kernel_code: str) -> AnalysisResult:
        
        # 1. 用工具做静态分析
        complexity = self._call_tool('analyze_syntax', kernel_code)
        memory_pattern = self._call_tool('detect_memory_pattern', kernel_code)
        
        # 2. 用 LLM 推理
        context = {
            "code": kernel_code[:500],  # 代码摘要
            "complexity": complexity,
            "memory_pattern": memory_pattern
        }
        analysis = self._think("""
            分析这个 CUDA kernel，找出优化机会：
            {context}
            
            返回 JSON: {
                "bottlenecks": [...],
                "optimization_strategies": [...]
            }
        """)
        
        # 3. 返回结构化结果
        return AnalysisResult(
            bottlenecks=analysis["bottlenecks"],
            strategies=analysis["optimization_strategies"],
            code_snippet=kernel_code
        )
```

**使用的工具**：
- `analyze_syntax(code)` → complexity score
- `detect_memory_pattern(code)` → memory access pattern
- `estimate_parallelism(code)` → parallelism level

---

## [3] Profiler Agent (profiler.py)

**职责**：在真实硬件上测评 kernel 性能

```python
class ProfilerAgent(BaseAgent):
    """获取性能基准数据"""
    
    def execute(self, kernel_code: str) -> ProfileResult:
        
        # 1. 编译 kernel
        compiled = self._call_tool('compile_cuda', kernel_code)
        if not compiled.success:
            raise CompilationError(compiled.error)
        
        # 2. 运行测评（3 次）
        metrics_list = []
        for i in range(3):
            metrics = self._call_tool('run_kernel', compiled.binary)
            metrics_list.append(metrics)
        
        # 3. 取平均值
        avg_metrics = self._aggregate_metrics(metrics_list)
        
        # 4. 用 LLM 分析瓶颈
        analysis = self._think(f"""
            kernel 性能数据：
            - 执行时间: {avg_metrics.exec_time} ms
            - 内存带宽利用率: {avg_metrics.memory_bw}%
            - 寄存器使用: {avg_metrics.register_usage}
            
            这表示什么瓶颈？
        """)
        
        return ProfileResult(
            metrics=avg_metrics,
            bottleneck=analysis,
            baseline_time=avg_metrics.exec_time
        )
```

**使用的工具**：
- `compile_cuda(code)` → binary
- `run_kernel(binary)` → KernelMetrics
- `extract_metrics(output)` → metrics dict

---

## [4] Optimizer Agent (optimizer.py)

**职责**：自动优化 kernel，迭代改进

```python
class OptimizerAgent(BaseAgent):
    """执行优化策略，迭代改进"""
    
    def execute(self, 
                kernel_code: str, 
                strategies: List[str],
                baseline_time: float) -> OptimizationResult:
        
        best_code = kernel_code
        best_time = baseline_time
        history = []
        
        # 1. 逐个尝试策略
        for strategy in strategies:
            
            # 2. 生成优化代码
            optimized = self._think(f"""
                针对这个 kernel，应用 {strategy} 优化：
                
                原代码：
                {kernel_code}
                
                返回优化后的完整代码。
            """)
            
            # 3. 验证和测评
            try:
                result = self._call_tool('compile_and_test', optimized)
                
                if result.success:
                    improvement = (baseline_time - result.time) / baseline_time
                    history.append({
                        "strategy": strategy,
                        "speedup": improvement,
                        "time": result.time,
                        "code": optimized
                    })
                    
                    # 4. 决策：保留还是丢弃
                    if improvement > 0.05:  # 至少 5% 改进
                        best_code = optimized
                        best_time = result.time
                        self._store_memory(f"strategy_{strategy}", optimized)
                        
            except Exception as e:
                self.logger.warn(f"Strategy {strategy} failed: {e}")
        
        return OptimizationResult(
            optimized_code=best_code,
            speedup=(baseline_time - best_time) / baseline_time,
            history=history
        )
```

**使用的工具**：
- `compile_and_test(code)` → execution result
- `validate_correctness(original, optimized)` → bool

---

## [5] 工具库 (tools/kernel_tools.py)

**特点**：所有工具集中在一个文件，都是简单函数

```python
# ========== 静态分析工具 ==========

def analyze_syntax(code: str) -> dict:
    """解析 CUDA 代码结构"""
    # 用正则表达式或 AST 解析
    # 返回: {"functions": [...], "kernels": [...], "complexity": score}

def detect_memory_pattern(code: str) -> str:
    """检测内存访问模式"""
    # 返回: "coalesced" / "strided" / "random"

def estimate_parallelism(code: str) -> int:
    """估计并行度"""
    # 返回: 线程块大小


# ========== 编译工具 ==========

def compile_cuda(code: str, gpu_arch="sm_80") -> CompileResult:
    """使用 nvcc 编译 CUDA kernel"""
    # 调用 nvcc
    # 返回: CompileResult(success, binary, error_msg)

def compile_and_test(code: str) -> TestResult:
    """编译 + 运行 + 测评"""
    # 1. 编译
    # 2. 运行测试用例
    # 3. 提取指标
    # 返回: TestResult(success, time, metrics)


# ========== 测评工具 ==========

def run_kernel(binary_path: str, num_runs=3) -> KernelMetrics:
    """在 GPU 上实际运行 kernel"""
    # 使用 CUDA Runtime + Profiler
    # 返回: KernelMetrics(exec_time, memory_bw, register_usage, ...)

def validate_correctness(original: str, optimized: str) -> bool:
    """验证优化后的代码是否正确"""
    # 对比输出结果
    # 返回: True/False


# ========== 工具函数 ==========

class CompileResult:
    success: bool
    binary: str
    error: str

class KernelMetrics:
    exec_time: float      # ms
    memory_bw: float      # %
    register_usage: int
    occupancy: float

class TestResult:
    success: bool
    time: float
    metrics: KernelMetrics
```

---

## [6] 数据模型 (core/models.py)

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AnalysisResult:
    """分析结果"""
    bottlenecks: List[str]
    strategies: List[str]
    code_snippet: str

@dataclass
class ProfileResult:
    """测评结果"""
    metrics: Dict
    bottleneck: str
    baseline_time: float

@dataclass
class OptimizationResult:
    """优化结果"""
    optimized_code: str
    speedup: float
    history: List[Dict]

@dataclass
class OptimizationReport:
    """最终报告"""
    original_kernel: str
    optimized_kernel: str
    speedup: float
    strategies_applied: List[str]
    analysis: AnalysisResult
    metrics_before: Dict
    metrics_after: Dict
```

---

## [7] 简单记忆 (core/memory.py)

```python
class AgentMemory:
    """简单的 dict 基础记忆"""
    
    def __init__(self):
        self.storage = {}
    
    def save(self, key: str, value):
        """保存信息"""
        self.storage[key] = value
    
    def retrieve(self, key: str):
        """检索信息"""
        return self.storage.get(key)
    
    def search(self, query: str) -> List:
        """搜索相关项"""
        return [v for k, v in self.storage.items() if query in k]
    
    def clear(self):
        """清空"""
        self.storage.clear()
```

> 第一版就这样。后续如果需要持久化可以加 SQLite 或 Redis。

---

## [8] 配置 (core/config.py)

```python
from dataclasses import dataclass

@dataclass
class AgentConfig:
    name: str
    model: str = "gpt-4"
    temperature: float = 0.3
    timeout: int = 60

@dataclass
class SystemConfig:
    gpu_device: int = 0
    max_optimization_rounds: int = 5
    min_improvement_threshold: float = 0.05  # 5%
    output_dir: str = "./results"
    enable_logging: bool = True

# 全局配置
CONFIG = SystemConfig(
    gpu_device=0,
    max_optimization_rounds=5,
    min_improvement_threshold=0.05
)
```

---

## [9] 主程序入口 (main.py)

**这是整个系统的核心流程**

```python
import os
from agents.analyzer import AnalyzerAgent
from agents.profiler import ProfilerAgent
from agents.optimizer import OptimizerAgent
from core.config import CONFIG
from core.models import OptimizationReport

def main(kernel_code: str):
    """GPU Kernel 优化主流程"""
    
    print("=" * 60)
    print("GPU Kernel Optimization Agent")
    print("=" * 60)
    
    # ===== Phase 1: 分析 =====
    print("\n[Phase 1] Analyzing kernel...")
    analyzer = AnalyzerAgent()
    analysis = analyzer.execute(kernel_code)
    
    print(f"  ✓ Bottlenecks: {analysis.bottlenecks}")
    print(f"  ✓ Strategies: {analysis.strategies}")
    
    
    # ===== Phase 2: 基准测评 =====
    print("\n[Phase 2] Profiling baseline...")
    profiler = ProfilerAgent()
    profile = profiler.execute(kernel_code)
    
    print(f"  ✓ Baseline time: {profile.baseline_time:.2f} ms")
    print(f"  ✓ Bottleneck: {profile.bottleneck}")
    
    
    # ===== Phase 3: 优化 =====
    print("\n[Phase 3] Optimizing kernel...")
    optimizer = OptimizerAgent()
    optimization = optimizer.execute(
        kernel_code=kernel_code,
        strategies=analysis.strategies[:CONFIG.max_optimization_rounds],
        baseline_time=profile.baseline_time
    )
    
    print(f"  ✓ Speedup: {optimization.speedup:.2f}x")
    print(f"  ✓ Applied {len(optimization.history)} optimizations")
    
    
    # ===== 生成报告 =====
    print("\n[Result] Optimization Summary")
    report = OptimizationReport(
        original_kernel=kernel_code,
        optimized_kernel=optimization.optimized_code,
        speedup=optimization.speedup,
        strategies_applied=[h["strategy"] for h in optimization.history],
        analysis=analysis,
        metrics_before={"time": profile.baseline_time},
        metrics_after={"time": profile.baseline_time / (1 + optimization.speedup)}
    )
    
    # 保存结果
    with open(f"{CONFIG.output_dir}/optimized_kernel.cu", "w") as f:
        f.write(report.optimized_kernel)
    
    print(f"\n✓ Optimization Complete!")
    print(f"  - Speedup: {report.speedup:.2f}x")
    print(f"  - Result saved to: {CONFIG.output_dir}/optimized_kernel.cu")
    
    return report


if __name__ == "__main__":
    # 示例 kernel 代码
    example_kernel = """
    __global__ void matmul(...) {
        ...
    }
    """
    
    result = main(example_kernel)
```

---

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

## 开发顺序（强烈推荐）

### Week 1
- [ ] 搭建目录结构
- [ ] 写 BaseAgent
- [ ] 写 kernel_tools.py（先用 mock 数据）
- [ ] 写 models.py

### Week 2
- [ ] 实现 AnalyzerAgent
- [ ] 实现 ProfilerAgent
- [ ] 测试前两个 Agent

### Week 3
- [ ] 实现 OptimizerAgent
- [ ] 写 main.py
- [ ] 集成测试

### Week 4+
- [ ] 优化性能
- [ ] 加更多策略
- [ ] 加 memory 持久化

---

## 需要的依赖

```
openai               # 或 github-copilot-sdk
pydantic            # 数据验证
pycuda              # CUDA 编程
numpy               # 数据处理
```

---

## 与完整架构的对比

| 方面 | 简化架构 | 完整架构 |
|------|--------|--------|
| **Agent数量** | 3 | 6+ |
| **文件数** | ~10 | ~25+ |
| **代码量** | ~500 行 | ~2000+ 行 |
| **开发时间** | 2-3 周 | 8-10 周 |
| **扩展性** | 80% | 100% |
| **上手难度** | 低 | 中-高 |

---

**这个架构既保留了实用性，又足够简洁，可以直接开始写代码！**
