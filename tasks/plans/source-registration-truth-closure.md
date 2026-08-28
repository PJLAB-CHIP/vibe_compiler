# Q64 Source registration truth闭合实施计划

状态：`done`。全仓source ownership、重复顶层目录清理和repo-wide registration mirror已闭合；稳定library依赖由18号设计文档
拥有，动态任务状态只看`tasks/progress.md`。Q62/Q63仍分别拥有numeric与NCC语义重构，并在当前Target/Model owner内继续。

当前`tools/check_source_organization.py`只对少数目录执行active/dormant集合闭包，仓库其余C++ translation unit和test可能既未进入
CMake，也未在18号dormant owner表出现而仍获得green结果。旧optimization comparison “source contract”又反向读取源码marker，
把未注册旧search文件伪装成必须保留的当前资产。CMake、filesystem、手写allowlist和Python catalog因此不是同一事实源。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  repository source/test filesystem、current CMake target/test registration、generated declarations及18号明确的task-owned dormant迁移表。
- Current stage responsibility:
  建立repo-wide registration mirror：每个可编译translation unit和测试要么属于一个current CMake target/CTest/lit owner，要么由
  一个current编号任务明确记录唯一能力承接与删除门禁；其它文件fail closed并删除或注册，不能被隐式忽略。
- Output IR / files:
  不改变compiler IR或产品output；输出唯一source/test inventory检查、收敛后的CMake registration和最小必要task-owned dormant表。
- Downstream consumer:
  fresh configure/build、unit/lit/CTest、public link closure以及Q50/Q51/Q62后续删除门禁。
- User-level driver / named pipeline:
  current CMake configure和source-organization检查；不新增compiler CLI、pass或runtime mode。
- Explicit non-goals:
  不把所有dormant source机械加入build，不为通过检查创建空target/stub，不保留retired symbol/source marker test，
  不把archive或generated build tree当source registration。
- Done criteria:
  repo-wide所有C/C++ source与测试均有且只有一个active registration或current task-owned dormant disposition；新增未注册文件使检查
  失败；registered test不读取implementation symbol/source文本来保活旧实现；CMake/source/test mirror、fresh configure/build和
  direct organization tests通过。
```

## 施工规则

- filesystem枚举排除build/generated/archive后，与CMake实际target source、unit target、lit/CTest registration做双向核对；
  checker不能只维护几个目录的硬编码集合。
- dormant记录必须包含current task owner、独有能力和删除门禁；只有“以后可能有用”或source未进CMake不能进入allowlist。
- Q51.Core删除Rank/coordinated search、paired optimization catalog/source-marker test；Q62删除numeric umbrella与迁出managed dependency
  conformance；Q63删除NCC free-switch旧owner。Q64不为这些文件重新建立dormant例外。
- source与test同步：实现未注册但test读取marker、test未注册但被文档称为gate、public header可include但无link symbol都作为错误。
- organization checker只证明注册/owner事实，不把FileCheck、源码marker或catalog计数冒充功能正确性。

## 当前源码分类与目标目录

扫描范围是source-controlled Wafer-owned `include/lib/tools/runtime/test/unittests/cmake`；build/generated、`third_party`和
`docs`原始资料不重排，但其current include/path引用必须随迁移更新。目标树不再存在`include/Wafer/Compiler`、
`lib/Wafer/Compiler`或顶层`WaferPipelines`：

| 当前文件职责 | 唯一目标owner | 文件组 |
| --- | --- | --- |
| current-IR/query-local事实 | `Analysis` | DAG、node-use、footprint、buffer relation、exact-demand/index/physical-layout relation及executable call/resource analysis；旧rank-global `GlobalTileRelation`已由current typed relation承接并删除 |
| physical assignment、baseline与search | `Planning` | 原`Compiler/Baseline`、`Compiler/Search`及placement/edge/temporal assignment；attention/decode纯proof单独进入`Analysis/Structured` |
| actual DeviceExecutable与Tile lowering | `CodeGen/Executable` | DeviceExecutable ownership、standalone Tile modules、Instr exact compile、fixed memory planning和bounded Tile execution |
| target LLVM/ABI/link/readback | `CodeGen/Target` | 原`Compiler/Target`除host TargetCall execution外的全部target code generation |
| actual communication/transport rewrite | `Transforms/Transport` | 原`Compiler/Transport`；它们修改/验证已选IR，不是driver或analysis |
| source-to-product orchestration | `Driver` | 原`Compiler/Pipeline`、SPMD helper bridge和outer publication transaction |
| program tensor/data ownership | `Program` | 原`Compiler/Program`及对应public API |
| package schema/read/write | `Package` | 既有manifest support与原`Compiler/Package` assembly/writer；outer rename仍归`Driver` |
| host TargetCall execution | `Target/Execution` | `TargetCallFrontend`，供model/tool消费且不依赖driver private search |
| named MLIR subpipeline | 各conversion/transform owner | StableHLO→Linalg、TileRegion→Instr、Instr memory preparation和SPMD propagation各回原owner；单pass wrapper删除，注册由owner提供，顶层`Pipelines` library删除 |
| test-only public hooks | `TestSupport` | 原`Compiler/Testing.h`及共享compiler test helpers；unit目录镜像production owner |

已经处于正确顶层owner的`IR`、`Frontend`、`Conversion`、`Transforms`、`Target`、`Package`、`Runtime`和`Model`文件逐项核对
definition与直接consumer后原位保留；仅按同一owner内部清晰子域整理，不为“目录整齐”建立无消费者的library层。

迁移顺序：先移动private implementation与unit mirror并保持行为不变，再拆public header/include和CMake target，最后删除
`WaferPipelines`与空目录。任一步都不得用compatibility include、转发header或旧target alias维持双owner。

### 2026-08-19 owner迁移checkpoint

逐文件inventory覆盖352个`lib/Wafer` tracked files、106个public/generated source headers以及tool/runtime/unit/lit/CTest入口；
全部Wafer-owned C++ translation unit现在由便宜的通用mirror检查为exact一个CMake owner、明确Q50/Q63 dormant owner或显式XLA
external-helper manifest。原2200余行、把numeric/frontend/model/package具体文件名复制进检查器的第二套架构表已删除；新检查只验证
registration、dormant disposition、旧owner零残留和unit注册，不代替各owner语义测试。

`include/lib/Wafer/Compiler`、`include/lib/Wafer/Pipelines`与`unittests/Compiler`已退出。现行目录为Analysis、Planning、
Conversion、Transforms、CodeGen、Driver、Program、Package、Target、Runtime、Model；public/private include和unit mirror同步迁移。
Target内部进一步按Core、Layout、Numeric/{Dependency,Formal,Qualification}、Execution归类；Model、Frontend、Package、Runtime与
Analysis也按其实际子域归类。`TargetCallFrontend`因不执行frontend职责已原位更名为`TargetCallExecution`。

顶层`WaferPipelines` library已删除：StableHLO→structured和TileRegion→Instr builder回到对应conversion owner，Instr memory与
SPMD builder回到Transforms；`wafer-opt`只拥有named registration。只包一层`pm.addPass`的helper已删除，production直接添加atomic
pass。fresh验证为主构建通过、729/729 C++ unit、250/250 supported lit（5项按feature配置unsupported）、4/4 public
header/link、IR/source organization、dependency layering以及轻量public search source-to-package/no-card通过。Q64不替代
Q50/Q62/Q63各自语义能力迁移门禁；明确dormant表继续使这些文件不能被误认为active coverage。

## 验证

- 为unregistered source、unregistered test、active+dormant overlap、unknown dormant owner和source-marker保活test提供checker负例；
- fresh configure后从实际CMake graph重算inventory，运行source organization、public header/link smoke和受影响direct tests；
- 不运行Q49.P/Q51长搜索、历史board资格或与source registration无关的全量integration。
