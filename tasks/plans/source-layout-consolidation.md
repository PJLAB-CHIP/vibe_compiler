# 源码布局收敛实施计划

状态：`doing`。动态状态只读`tasks/progress.md`，稳定目录与依赖合同由18号设计拥有，MLIR作用域与
rewrite/conversion规则由19号设计和`AGENTS.md`拥有。本计划只安排本次迁移，不复制各IR、target、package或runtime语义。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  current Wafer-owned include/lib/runtime/tools/test/unittests/cmake/python filesystem、实际CMake source/target/test graph、
  public headers、current numbered docs和current plans；编译器IR与产品输出保持现状。
- Current stage responsibility:
  将源码按18号最终目标树迁移；同步移动definition、direct consumer、CMake owner和test mirror；拆除旧目录、宽聚合target、
  产品test hook、未注册测试和source-tree generated artifacts；保持每项能力唯一实现。
- Output IR / files:
  不改变IR、ABI、package或runtime格式；输出职责明确、include/lib镜像、依赖单向、source/test exact registered的源码树及
  与其一致的current文档。
- Downstream consumer:
  Q52 physical-dataflow实现、named pipelines、wafer-compile、wafer-opt、wafer-run、Simulator backends、package/runtime和tests。
- User-level driver / named pipeline:
  不增加或重命名用户CLI/pass/pipeline；路径和CMake target只作为实现组织，不进入IR或产品协议。
- Explicit non-goals:
  不借迁移修正数值行为、搜索算法、SPM/completion、target ABI或package schema；不为保持旧路径添加forwarding header、
  target alias、双source list或compatibility wrapper；不改写archive中的历史路径。
- Completion criteria:
  18号目标树与filesystem/CMake/tests/current docs一致，旧owner及空目录为零；完整canonical build、全部registered本地测试、
  component header/link closure、source/IR organization和代表下游witness fresh通过并提交。
```

## 2. 施工方法

每个迁移批次都执行同一闭环：

1. 重读`AGENTS.md`、`progress.md`、18/19号设计及该批次涉及的语义owner；
2. 从definition、constructor、direct consumer、verifier/lowering、CMake owner和registered test建立逐文件对应；
3. 对照官方MLIR/CIRCT/IREE同类component和仓库pinned LLVM/MLIR接口，确认该文件属于analysis、transform、conversion、
   codegen还是driver；
4. 使用一次breaking migration移动实现、public include、CMake和test，不保留旧入口；
5. 运行该批次direct build/test、旧路径扫描和source registration检查；
6. 重读对应设计与完整diff，确认能力、failure分类、current-IR owner和MLIR工程边界未改变后才进入下一批。

移动优先使用`git mv`保持历史；内容修改使用`apply_patch`。任一批次发现文件职责与18号目标owner不符时，先修正文档，
不能为了完成目录表强行迁移。

## 3. 旧owner到新owner

| 当前owner/文件组 | 最终owner | 处置边界 |
| --- | --- | --- |
| `IR/Program/ModuleOps.*` | `IR/Tile/TileModuleOps.*` | 只定义`wafer.tile.module`；同步ODS include、generated dependency、实现和Dialect tests |
| `IR/Resource/SPMOps.*` | `IR/Tile/StorageOps.*` | `tile.load/store`是Tile movement；effect resource仍由Wafer interface集中定义 |
| `IR/Target` | `IR/Topology` | 只移动typed topology/mesh IR；pure C++ ID/fact留在Target |
| `Analysis/ControlFlow` | 同名generic analysis owner | 只保留跨IR复用的standard RegionBranch/CFG query；不扩成generic execution owner |
| `Analysis/{Structured,PhysicalDataflow,ScheduleCost,Scheduling,Executable,CallGraph}` | `Analysis/{Module,Linalg,Tile,Instr}` | IndexRelation/DAG/SemanticRoot等choice-independent事实按anchor拆分；analysis不得依赖Conversion/Planning |
| `Planning/PhysicalDataflow`及choice-dependent demand文件 | 同名收窄owner | 保留SpatialAssignment、ExactDemand/RootRegionWork、choice/domain/PBQP/search；actual IR、mutation和future/shadow事实迁出或删除 |
| mutable structured materialization/buffer relations | `Transforms/Tile` | caller-owned current-IR transaction bookkeeping，不由Analysis拥有且不跨IR epoch |
| `Conversion/StructuredTiling` | `Transforms/Linalg` | interface-driven tiling是同层变换helper，不是conversion |
| structural Tile materialization | `Transforms/Linalg/TileFormation` | closed choice立即产生actual TileModule/TileRegion；baseline/search调用同一atomic transform但各有owner |
| `Transforms/{Execution,Bufferization,SPM,DDR,MemoryPlanning,Scheduling,Transport}` | `Transforms/{Tile,Instr}` | 按变换输入IR拆分；不保留generic execution/scheduling/memory owner |
| `Conversion/WaferTileRegionToInstr` | `Conversion/TileToInstr` | 同步Passes、pipeline、DRR/C++ patterns、lit和public factory |
| `Transforms/Target`中的Instr lowering | `Conversion/InstrToLLVM` | conversion legality、TargetCall creation和postcheck同owner |
| `CodeGen/Target`中的LLVM translation/link/readback | `CodeGen/LLVM` | 不包含Instr conversion、package writing或Simulator execution |
| `CodeGen/Executable` | `CodeGen`、`Transforms`或`Driver` | DeviceExecutable/target module typed owner留CodeGen；memory/IR mutation迁Transforms；admission/orchestration迁Driver |
| `Conversion/StandaloneTileModules` | `Driver` | module multiplicity与result lifetime由compiler driver拥有，不是MLIR conversion |
| `Target/Core` | `Target`根目录 | pure target contract展平；检查public header自包含和Target无MLIR依赖 |
| `Target/Layout`及raw scalar codec | `Target/PhysicalTensor` | physical geometry/descriptor/codec/materialization统一owner；与PBQP layout assignment分离 |
| `Target/Execution`、`Model` | `Simulator` | shared host JIT/invocation/kernel在根；formal/reference、OneDNN、SystemC按backend拆分 |
| `Target/Numeric/Formal` | `Simulator/Reference` | formal/MPFR/SoftFloat reference arithmetic不进入Target基础library |
| `Target/Numeric/Qualification`和`Model/Qualification` | `Simulator/OneDNN` | 共享OneDNN backend进入Simulator；CLI/calibration driver仍由tool持有 |
| 顶层`Program` | `Frontend`、`Driver`、`Simulator`、`Target/PhysicalTensor` | 逐类型按source ownership、transaction、invocation/comparison和format拆分；不保留Program umbrella |
| `TestSupport`及产品源码中的failure injection | test-only support | 产品library不include测试header；unit/tool test分别链接窄test support |
| `tools/test_*.py` | `test/Tools` | 每个测试实际注册；产品脚本与开发helper分开 |
| build/dependency/check Python | `utils` | end-user/install tool仍留`tools`；同步CMake、docs和tests |
| `runtime/wafer_crt` | `runtime/crt` | 外部ABI文件名保持；安装和tool resolver同步更新 |

无current consumer、重复实现、空目录和Q52已经退役的shadow source不在表中寻找新位置；按18号能力映射和06号current设计直接删除。

## 4. 线性迁移批次

| 顺序 | 批次 | 输入 | 完成输出 | 直接验证 |
| --- | --- | --- | --- | --- |
| 1 | 文档与registration基线 | current filesystem/CMake/test inventory | 18号设计、本计划、progress；Q64旧计划归档；旧→新逐文件inventory冻结 | Markdown链接、`git diff --check`、inventory脚本dry run |
| 2 | IR与generated owner | current Wafer ODS/IR实现 | `IR/{LinalgExt,Tile,Instr,Topology}`，Program/Resource/Target旧IR目录消失 | TableGen、WaferIR、Dialect parser/printer/verifier/lit |
| 3 | Analysis与Planning | current analysis/planning APIs及tests | `Analysis/{Module,Linalg,Tile,Instr}`；Planning只保留纯choice/search | analysis/planning unit、invalidation/direct query、dependency scan |
| 4 | Transforms与Conversion | current pass/pipeline/helper实现 | 按输入IR的Transforms；仅三个`XToY` Conversion；旧pass path为零 | affected targets、`verify-each`、named pipeline/lit、1024/1025 witness |
| 5 | CodeGen、Driver与Frontend | current executable/target/program source | CodeGen只保留typed output+LLVM；module fan-out/orchestration归Driver；source data归Frontend | compiler/link closure、direct codegen/package tests、product tool smoke |
| 6 | Target与Simulator | current Target/Model/numeric/SystemC/OneDNN | pure Target+PhysicalTensor；Simulator Reference/OneDNN/SystemC；反向link清零 | target/simulator unit、optional dependency gates、same-target-LLVM differential |
| 7 | Package、Runtime、tools与test tree | current remaining source/test/tool paths | root tree、runtime/crt、utils、test/unittest mirror和Board support/case分离 | CTest registration mirror、Tools/lit、runtime/no-card focused tests |
| 8 | current文档与最终门禁 | 完成后的source/API/CMake | 所有current numbered docs/plans/memory路径和owner同步；archive不改写 | full residual scan、canonical build/CTest、second-build no-op、完整diff复审 |

每批提交前保持主工程可configure/build。若一个breaking public include迁移无法在中间状态保持build，则将其definition、全部consumer、
CMake和tests放在同一批patch，不引入过渡header。

## 5. 覆盖矩阵

| 覆盖类 | 输入等价类/结构路径 | exact断言 | 直接下游witness |
| --- | --- | --- | --- |
| source registration | production C/C++、public header、unit、lit、Python test、external-helper manifest | 每个translation unit/test恰有一个active owner；新增未注册fixture使checker失败 | fresh CMake source graph和CTest/lit discovery |
| public/component边界 | IR、Analysis、Planning、Transforms、Conversion、CodeGen、Target、Simulator、Package、Runtime headers | header独立include/link；禁止的反向include/link为零 | 对应component unit executable实际链接运行 |
| IR搬迁 | TileModule、Tile load/store、Topology/mesh的roundtrip和negative verifier | generic/custom roundtrip一致；局部verifier结果和diagnostic类别不变 | Tile formation、TileToInstr focused pipeline仍消费新header/generated owner |
| Analysis/Planning | rank≥3 structured DAG；主要维度1024与1025/1031；multiple Tile、remainder、tail；unsupported/indeterminate | demand/domain/choice集合、deterministic order和typed failure不变；不新增IR mutation | actual TileFormation direct API消费planning choice |
| Transforms/Conversion | StableHLO/Linalg、Tile、Instr三层正负例；整除/非整除；unknown source op | exact output legality、source-op absence、failure atomicity、`verify-each` | named pipeline与production builder调用同一实现 |
| CodeGen/Target | finalized Instr、target LLVM、linked/readback module；invalid ABI/metadata | target call/module/entry inventory、identity、digest和typed failure不变 | DeviceExecutable→target module→package direct tests |
| Simulator backend | Reference、OneDNN、SystemC；supported和typed unsupported family | 同一TargetCall input的functional result/failure分类不变；Target不反向link backend | existing differential/unit及tool qualification tests |
| Runtime/tool | package readback、no-card、profile、installed/build-tree tool | CLI和manifest/runtime行为不变；测试实际执行且无新skip | registered CTest/lit和focused no-card |
| 文档/残留 | current numbered docs、current plans、README、memory、CMake/install paths | old owner/path为零；archive历史路径不计current残留 | 新开发者仅沿AGENTS→progress→设计→current plan得到唯一结构 |

本任务不建立新的数值oracle。已有numeric tests只作为源码迁移回归，不修改comparator、reassociation或策略。

## 6. 文档同步清单

源码批次完成后必须按实际影响更新：

- 01架构中的library/pipeline索引；
- 02/03 frontend、program data和SPMD实现索引；
- 05–13 analysis/planning/transform/conversion实现索引；
- 14 target codegen、15 package/runtime、16 verification、17 Simulator backend的owner/path；
- 19 MLIR pass/analysis/conversion路径和build target；
- 20 current interface owner；
- current `tasks/plans/`中仍被队列消费的实现路径；
- `tasks/README.md`、`tasks/progress.md`、`memory/general_dev.md`、`memory/bugs.md`中真实受影响的稳定入口。

`tasks/archive/`保留当时路径，只更新导航使Q64旧计划明确为历史。文档不能在代码迁移前声称某个批次已完成；完成后也不能
保留旧路径作为“曾经实现”的current说明。

## 7. 最终验证

使用canonical `build/`和实际`nproc`：

1. `cmake --preset default`；
2. `cmake --build --preset default -j$(nproc)`；
3. `ctest --preset default -j$(nproc)`并核对实际执行、unsupported和skip；
4. configured `check-wafer-lit`/registered lit CTest；
5. public header/link、source/IR organization、dependency layering、旧路径和未注册测试负例；
6. 受影响的1024/1025或1031 named-pipeline/direct downstream tests；
7. 再次完整build必须为Ninja no-op；
8. `git diff --check`、`git status`、完整diff和18/19号合同复审。

真实板端不在本任务范围。源码组织不改变package/runtime/board合同，因此只运行受影响host/no-card gate，不重放历史板端输出。
