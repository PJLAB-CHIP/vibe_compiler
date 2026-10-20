# GEMM 优化对照的实际 IR

对应[2026-10-20 实测报告](../../../board-performance/2026-10-20.md)。
`M=4096、N=4096、K=1024`，batch=1、BF16、16 Tile。
以下文件从本轮已测 package 对应的 compiler dump 原样复制，共七档 × 三层 × 16 Tile = **336 个文本 IR 文件**。

| 方案 | 目录 | Kernel 中位数（ms） |
| --- | --- | ---: |
| 基础分块 | [tensor-candidate-37](tensor-candidate-37/instruction/tile_00000.mlir) | 3.537 |
| A 本地驻留复用 | [tensor-candidate-39](tensor-candidate-39/instruction/tile_00000.mlir) | 2.807 |
| 非蛇形 NoC + 驻留 | [tensor-candidate-38](tensor-candidate-38/instruction/tile_00000.mlir) | 2.186 |
| 蛇形 NoC + NCx | [ncx-winner](ncx-winner/instruction/tile_00000.mlir) | 1.794 |
| 仅调整分块 | [tensor-candidate-35](tensor-candidate-35/instruction/tile_00000.mlir) | 3.887 |
| 普通 NoC，无驻留 | [tensor-candidate-23](tensor-candidate-23/instruction/tile_00000.mlir) | 3.610 |
| 蛇形 NoC，Tensor I/O | [tensor-winner](tensor-winner/instruction/tile_00000.mlir) | 2.239 |

每个目录包含：

- `tile-dataflow/tile_00000.mlir` 至 `tile_00015.mlir`：实际 Tile 数据流与存储/通信操作。
- `instruction/tile_00000.mlir` 至 `tile_00015.mlir`：最终 Instr，含分块循环、allocation offset、DMA、DTE 和同步。
- `target-llvm/tile_00000.ll` 至 `tile_00015.ll`：同次编译的 target LLVM 输出。

七档 source 逐字节相同，在 `source-program/` 保存一份原始 portable StableHLO 和 metadata。
[source.stablehlo.mlir](source.stablehlo.mlir) 是该 bytecode 经同一 build 的 `stablehlo-translate --deserialize`
产生的人类可读文本，不替代三层实际 compiler dump。

[manifest.json](manifest.json) 保存文件 hash；编译器/runtime 身份、原始样本、完整命令、
每 Tile 通信边及数值检查在[测量 JSON](../gemm-optimization-ladder-20261020.json)。
各层文件保持原样，包括 compiler dump 自带的末尾空行；本目录 Git 属性仅对这三层关闭末尾空行提示。
后续分析可以直接搜索或 diff，不需重新编译。
