# Compiler 命名整改计划

状态：`done`。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  active dialect、analysis、transformation、conversion、compiler driver、target/runtime接口和diagnostic。
- Current stage responsibility:
  根据每个对象处理的IR、实际变换和输出，删除错误抽象、拆分混合职责并修正误导名称。
- Output IR / files:
  语义和ABI不变、名称与实现职责一致的active source、IR schema、tests和current docs。
- Downstream consumer:
  后续compiler、runtime和工具开发。
- User-level driver / named pipeline:
  现有`wafer-compile`、`wafer-opt`和named pipelines；本任务不新增driver。
- Explicit non-goals:
  不做关键词机械替换，不改变IR语义或用户ABI；archive不作为current架构合同，只有用户明确要求的术语修正才同步文件名和引用。
- Done criteria:
  受影响的声明、实现、CMake、测试和current docs同步完成，并通过fresh build与对应测试。
```

## 执行方法

1. 沿active build实际包含的代码检查ODS、public API、pass/pipeline、文件名、diagnostic和current docs。
2. 对每个可疑名称读取实现和consumer，写清它处理的IR、实际变换、输出和ABI影响。
3. 无独立职责的抽象直接删除；混合多个变换的对象先拆分；职责清楚但名称错误的对象再重命名。
4. 每批沿一条完整调用链同步声明、实现、CMake、测试和文档，不保留内部兼容别名。
5. 每批运行fresh build和直接受影响的测试；全部批次完成后再做全仓构建与测试。

命名采用仓库pinned LLVM/MLIR的常见形式：type使用语义名词，function使用说明实际动作的动词短语，conversion说明
源表示和目标表示，optimization说明实际算法。同一概念在ODS、C++、pass、文件和diagnostic中使用同一名称。

## 当前命名边界

- `builtin.module`是共同symbol/verifier scope；top-level `TileModuleOp`直接携带`card_id`和`tile_id`。
- 单个Tile的cost为`InstructionProgramCost`，完整Tile集合的聚合结果为`InstructionProgramAggregateCost`。
- `StructuredProgramAnalysis`只分析current structured program；`ExecutableLowering`和`ExecutableCompilation`只说明动作。
- `DeviceExecutable`是all-and-only Tile executable与program data的最终内存owner。
- `DDR*`表示所有Tile共享的DDR资源；`SharedWorkspace*`表示该DDR中的共享workspace allocation。
- `CardId`、`CardCoordinate`、`card_id`和`card_count`只在确实表示物理身份或外部wire domain时保留。

## Device-scope terminology closure

```text
Pipeline position:
- Upstream IR / input:
  single-card TensorProgram、placed TileRegion/per-Tile modules、final instruction modules及current executable/package/runtime API。
- Current stage responsibility:
  让IR hierarchy、analysis、lowering与最终产物名称分别对应真实职责，删除没有独立语义的中间wrapper和范围限定。
- Output IR / files:
  builtin module中的top-level `wafer.tile.module(card_id, tile_id)`或后续直接fan-out的per-Tile builtin modules；
  `StructuredProgramAnalysis`、`InstructionProgramCost`、`InstructionProgramAggregateCost`、`ExecutableLowering`、
  `ExecutableCompilation`与`DeviceExecutable`。
- Downstream consumer:
  physical-dataflow materialization、Instr/memory/target lowering、package writer、model和runtime。
- User-level driver / named pipeline:
  current `wafer-compile` library entry及同实现的focused tests。
- Explicit non-goals:
  不删除真实physical identity `CardId`、`CardCoordinate`或外部schema `card_id/card_count`；
  不用typedef、alias、wrapper或双reader保留旧名；不借重命名改变数值、memory或completion语义。
- Completion criteria:
  source/header/CMake/test/current docs只保留下表current形式；真实Card字段未改变；完整本地gate通过。
```

current形式：

| 概念 | current表示 | 理由 |
| --- | --- | --- |
| shared module scope与Tile ownership | `builtin.module` + top-level `TileModuleOp(card_id, tile_id)` | builtin module已经提供共同scope；Tile op提供实际execution ownership |
| per-Tile fan-out | `WaferTileModuleFanout` | 直接消费top-level Tile modules并按typed identity稳定排序 |
| structured analysis | `StructuredProgramAnalysis` | 输入是current structured function、topology和Tile domain |
| instruction cost | `InstructionProgramCost` / `InstructionProgramAggregateCost` | 分别表示单Tile program与完整Tile集合，类型边界明确 |
| executable action | `ExecutableLowering*` / `ExecutableCompilation*` | namespace、输入和输出已经确定作用范围 |
| final owner | `DeviceExecutable` | 唯一持有全部Tile executables与program data |
| shared memory | `DDR*`、`SharedWorkspace*`、`shared_workspace` | DDR是memory space；workspace是其中的共享allocation role |
| physical identity | `CardId`、`CardCoordinate`、external `card_id/card_count` | 确实表示物理Card或外部wire domain |

覆盖矩阵：

| 输入等价类 | 结构 | failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| top-level Tile module set | 16个unique `(card_id=0,tile_id)`、顺序扰动、2x1024x64与2x1025x64 shared declarations、缺失/重复/负值 | local verifier只查自身；collection check拒绝duplicate；executable stage拒绝missing/foreign/unavailable Tile | fan-out按typed ID稳定排序；每个body只move一次；共享declaration复制一次；partial set保留给caller stage判定 | Executable lowering消费16个standalone modules |
| whole-device final owner | ordinary/profile、program data empty/nonempty、target-model consumer | incomplete Tile/target module/argument closure保持原typed failure | `DeviceExecutable`唯一持有all-and-only Tile executables及program data | package writer、target modules、model/runtime |
| shared DDR/workspace | DDR resource/binding/movement、shared workspace、package `card_id/card_count`、topology multi-card verifier | 任何字段丢失、scope改变或旧reader仍接受即失败 | DDR行为byte-equivalent；repo-owned workspace spelling只接受`shared_workspace`；physical card identity不变 | package strict readback与runtime invocation |
| residual scan | source/header/CMake/test/current docs | 任一旧wrapper/API仍有current caller即失败 | 旧名0；archive不改 | source/IR organization、public link与完整build |
