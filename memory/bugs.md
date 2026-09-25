# Wafer Compiler Bug Patterns

本页按主题索引可复用的现象、根因、修复与防复发方法。只读当前问题相关条目，不把全部经验作为开工前置。
任务状态见[progress](../tasks/progress.md)，工程流程见[AGENTS](../AGENTS.md)，常用命令见[general_dev](general_dev.md)。
历史现场和已退役接口保存在任务归档；以下条目以当前编号设计和源码为准。

| 主题 | 范围 |
| --- | --- |
| [Frontend、数值与输入语义](bugs/frontend.md) | 13条根因 |
| [索引关系、切分与Tensor语义](bugs/relations-tiling.md) | 48条根因 |
| [搜索、候选与成本](bugs/search.md) | 34条根因 |
| [布局、bufferization与搬运](bugs/layout-movement.md) | 34条根因 |
| [内存、通信与完成](bugs/memory-completion.md) | 26条根因 |
| [Package、runtime与执行模型](bugs/package-runtime.md) | 31条根因 |
| [MLIR变换与C++所有权](bugs/mlir-cpp.md) | 14条根因 |
| [构建、验证与文档维护](bugs/build-validation.md) | 18条根因 |

可按实际机制检索，例如：

```bash
rg -n 'completion|SPM|bufferization|StringRef' memory/bugs
```
