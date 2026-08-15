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

## 最终批次：IR 层级与限定词

本批继续处理第一轮后仍存在的范围型名称：

1. 核对`wafer.card.module`与`wafer.tile.module`的SymbolTable、`IsolatedFromAbove`、SSA ownership、split和lowering
   consumer，按真实module层级统一ODS、C++和文本IR名称。
2. 核对设备级candidate search/evaluation/selection、lowering、resource verification对象的输入输出；删除只说明遍历范围、
   不说明compiler职责的前缀，混合职责不以改名掩盖。
3. 限定词仅在同一API确有logical/target、virtual/register allocation或其它真实对照且强类型无法消歧时保留；已经由namespace、
   ID type、parent op、anchor或容器确定的限定不进入type、function、文件和diagnostic。
4. 沿parser/printer、verifier、conversion、named pipeline、public API、CMake、测试和current docs逐批迁移，不提供内部双名兼容层。

本批不改变placement、tiling、movement、memory planning、ABI或runtime语义，不改只读archive和外部API拼写，也不把所有
`physical`、`whole`、`device`或`target`做机械替换。每个名称先通过定义与consumer审计，再决定保留、重命名或拆分。

### 语义核对结果

| 当前对象 | 收敛方向 | 依据 |
| --- | --- | --- |
| `CardModuleOp` / `TileModuleOp` | `CardModuleOp` / `TileModuleOp` | 两者都是`SymbolTable`、`IsolatedFromAbove`的module-like container；`card_id`和`tile_id`仍是target topology中的真实层级 |
| `CardId` / `TileId`及coordinate/link | `CardId` / `TileId`及对应coordinate/link | 强类型已经与logical partition、launch slot分离，不需重复限定 |
| `TargetTopology` | `TargetTopology` | analysis直接且只从`wafer.target.topology`派生 |
| standalone Tile module/executable/trace | `TileModule` / `TileExecutable` / `TileIRTrace` | module、entry和typed `TileId`已经确定硬件绑定 |
| Tile memory planning | Tile memory planning | anchor和输入已经是独立Tile module |
| `CardDAG*` / `WholeDAG*` | `StructuredDAG*` | DAG从一个structured function的SSA派生；完整图由对象本身表示，不需要card/whole范围前缀 |
| `TileMapping` | `TileMapping` | 对象实际描述structured work到Tile的placement、temporal tile和edge action输入 |
| `WholeCardInstructionProgramCost` / `WholeCardResourceDurationEstimate` | `CardInstructionProgramCost` / `ProgramDurationEstimate` | instruction cost存在Card/Tile两个真实强类型层级；duration只有一个program范围，不重复限定 |
| `PhysicalTileExecutables` / internal `WholeCardExecutable` | public `CardExecutable` / internal `CardExecutableLoweringResult` | public owner原子持有一个card的all-and-only `TileExecutable`；internal result只表示lowering结果，不再造第二个executable概念 |
| `WholeCardExecutableLowering*` | `CardExecutableLowering*` | lowering边界产生一个`CardExecutable`；`whole`只重复容器已表达的范围 |
| optional card rank sources | `RankCandidate*`、`RankInstrLowering*`、`RankResourceCostValidation*` | 这些未构建source实际消费完整logical-rank tuple，不能伪装成current Card/Tile module实现 |

`CardResourceScope`、`TileResourceScope`、card/Tile topology字段、logical-to-Tile placement边界以及logical range与target byte
range的真实对照继续保留必要限定；它们不是本批要消除的重复范围词。

profile activation保留真实format version；correlation basis、static-cost scope和硬件校准claim key使用稳定语义名，
不再把内部算法revision伪装成独立版本。新的
type、function、pass option、pipeline timing和diagnostic均使用`Card`、`Tile`、`CardExecutable`等已由IR和强类型
定义的名称。

## 已完成批次

本轮按定义和直接consumer逐项核对，不以关键词替换代替语义判断：

| 范围 | 收敛结果 |
| --- | --- |
| compiler public API | target侧收敛为`CardExecutable`、`TargetLLVMModules`、`LinkedTargetModules`和`CompiledProgram`；函数名直接说明compile、link、readback或write package |
| Tile lowering | 原来混合转换、bufferization和memory planning的入口拆成明确的Instr lowering、SPM/DDR assignment与resource verification；文件和测试使用同一职责名 |
| whole-device selection | search、candidate evaluation、resource verification和selection分别命名；不再用含糊的协调、前沿或阶段结束术语代替实际动作 |
| target/runtime | profile数据写入、运行参数构造、target command验证和模型执行分别命名；C/C++符号与diagnostic同步 |
| MLIR source | verifier、target execution facts、instruction verification和StableHLO collective lowering按实际IR层命名；声明、实现、CMake和lit文件同步 |
| tools/tests/docs | Python字段和函数使用file、record、evidence、write、validation等具体对象或动作；current tasks、memory和presentations同步到现有API |
| profile与优化资格化 | runtime数据收集使用`WaferProfileCollection`；重复采样使用`PROFILE_MEASUREMENT_COUNT`；A/B用例和入口使用`OptimizationComparisonCases`与`PAIRED_COMPARISON_TEST` |

没有为名称建立源码黑名单。dependency snapshot由repo内producer/consumer同步演进，不拥有独立版本线；
上游StableHLO/PyTorch-XLA类名和环境变量属于外部API；archive不是current设计事实源。所有一手C++/Python抽象、
文件名、diagnostic和current文档已经退出旧术语。

## 已完成批次验证

- `build/q45-fresh`独立配置并完成419个构建步骤；没有复用`build/wafer-dev`生成物。
- fresh `WaferUnitTests`排除Q49/Q52已知长搜索后为629/629通过；长搜索不属于命名整改门禁。
- fresh CTest 1–11为11/11通过，包含lit、dependency snapshot、PyTorch/no-card和配置测试。
- fresh CTest 13–20为8/8通过，包含runtime IO、public-header/link smoke和feature-off link closure。
- source/IR organization、Python compileall、profile report、dependency helpers及受影响Board协议的host-only测试通过。
- `git diff --check`通过；未执行真实板端测试。

限定词与IR层级最终批次使用`build/q45-qualifier-fresh`独立验证：

- fresh configure后419/419构建通过，包含新ODS/TableGen、renamed source和public link targets；
- topology、CardModule/TileModule conversion、StructuredDAG、program resources、CardExecutable lowering、Tile memory
  planning、target ABI/codegen、named pipelines和Instr-to-target conversion定向单测138/138通过；
- CTest 1–11为11/11通过，CTest 13–20为8/8通过；Wafer lit 216/216由CTest实际执行；
- IR/source organization、Python compileall和`git diff --check`通过。

StructuredDAG placement enumeration和CardExecutable candidate synthesis属于Q49/Q52搜索行为，本批没有用
长搜索代替命名合同验证；没有执行真实板端测试。

未注册到CMake/CTest且依赖已退役实现的旧Board calibration、catalog和probe已经删除，不再作为后续任务的隐式入口。
Q50.S若需要性能比较，必须从current compiler pipeline建立有生产实现、有CTest注册且可重放的测试入口，不能恢复旧catalog
或用字符串匹配伪造compiler能力。
