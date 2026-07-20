# MLIR-native Implementation / IndexRelation Foundation 实施记录

状态：historical / completed（2026-07-20）。Tracking ID为Q32.I；当前状态只看
`tasks/progress.md`。

设计owner：`tasks/01-architecture.md`、`tasks/05-local-compute-normalization.md`、
`tasks/06-physical-dataflow-synthesis.md`、`tasks/08-physical-realization.md`、
`tasks/10-compute-movement.md`、`tasks/13-communication.md`、
`tasks/16-verification-contract.md`和`tasks/18-source-organization.md`。

本记录保存Q32 fresh baseline、source implementation纵向、MLIR-backed IndexRelation foundation和
custom interface盘点证据。它不是新的总体设计、candidate wire schema或跨pass relation协议。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: normalized rank-local structured tensor program；compute root由LinalgOp、DestinationStyleOpInterface、TilingInterface、indexing maps、scalar region和typed collective fields表达。
- Current stage responsibility: 从source op-local语义与target capabilities枚举有界implementation；在isolated complete candidate clone中立即物化选中implementation；从当前IR按需构造可失效、可重算的index relation。
- Output artifact / IR: typed wafer.tile compute/movement clone及Exact/SoundBound/Unsupported/Invalid/ResourceExhausted的transformation-local IndexRelation；只有选中的implementation kind作为candidate临时参数继续到完整clone评估。
- Downstream consumer: tile-region→instruction conversion、SPM/DDR planning、execution cost、candidate selection/commit、all-rank finalization以及target/package/SystemC gate；Q32.R继续消费IndexRelation扩展physical realization。
- User-level driver / named pipeline: production wafer-compile structured tensor program scheduling与其完整source-to-package/target-model pipeline；wafer-opt只复用注册和debug入口。
- Explicit non-goals: 不新增planner descriptor/provider/query/key、relation attr或wire format；不在本checkpoint完成encoding/route/residency联合选择、StorageLoad迁移、默认producer切换或全部custom interface退役。
- Completion gate: fresh baseline和改后1/16-rank相关主线无回退；所有current production compute root有typed implementation或结构化unsupported；真实非baseline implementation产生不同actual clone并通过Tile→Instr及完整candidate gates；IndexRelation覆盖identity/permutation/broadcast/slice/reshape/composition、ValueBounds和precision/failure/budget negative；重复DPS/Tiling interface删除。
```

## Fresh baseline与现有owner矩阵

开工前development配置为226/229 lit通过、3条按feature配置unsupported，基础单测279/279；
target-model Release配置为227/229 lit通过、2条按feature配置unsupported，基础/numeric/bulk分别
279/279、54/54、18/18，6条SystemC integration通过。标准Llama-2 7B单block TP16 corpus以
16个CPU PJRT devices重新导出；未提供设备数时exporter以“requires 16 XLA devices, got 1”拒绝，
显式`CPU_NUM_DEVICES=16`后成功。开工baseline的完整target-model replay为16/16 rank输出匹配，
19,696 transactions、17个SystemC threads、53,257,728 managed-reference scalars和672个
bulk matmul/reorder commands，real约57.969秒。

| 功能面 | 当前source/owner | 本checkpoint结论或后续缺口 |
| --- | --- | --- |
| implementation | `linalg.fill`、`linalg.matmul`、`linalg.batch_matmul`、`linalg.generic`；`linalg.map`在normalization后归入generic | 四类production compute root均由external model提供typed baseline；pointwise reciprocal generic另有division implementation；其它Linalg root无interface时fail closed |
| collective | Wafer LinalgExt destination-style collective + standard DPS/Tiling/effect | 属于communication/movement root，不伪装成compute implementation；rank group、axis和combiner仍由typed op/interface消费 |
| encoding/view | `MemLayout` Tensor/Cx/NCx、typed memref、view/layout materialization ops | 当前固定Cx/NCx和view路径保留；relation驱动的zero-copy/DMA/GS/staged actual-clone选择交Q32.R/M |
| movement/route | storage load/store、tile move/layout materialize、RDMA/WDMA、gather-scatter、staged path | 当前route、descriptor、invalid-lane和geometry gate是后续rewrite必须重放的事实源，不提前复制成candidate descriptor |
| buffering/order | current complete traversal、lifetime、SPM/DDR placement、instruction/event顺序 | 保留existing final-clone重算与atomic commit seam；Q32.M/S才增加有真实owner的邻居 |
| communication | logical collective、tile communication、Direct DTE all-rank acceptance | current direct/ring/tree入口与缺口由Q32.M统一闭合；Q32.I不新增transport signature |
| finalization seam | isolated candidate clone→Tile→Instr→SPM/DDR→cost→rank frontier→whole-variant commit→ABI/package | 非baseline implementation已重放同一complete evaluation路径；失败clone不写回source |

## Source implementation interface与真实纵向

新增的`WaferTargetImplementationOpInterface`只拥有source op-local target implementation枚举和
selected materialization hook。Linalg op本身不可修改，因此通过dialect extension安装external model；source
semantics、DPS tie、tiling、indexing map和effect继续以标准MLIR接口为准。conversion-owned materializer拥有
physical operand/result及typed `wafer.tile.*`构造，interface不返回scalar-region副本、layout snapshot或
长期candidate payload。compiler、candidate evaluation和`wafer-opt`统一注册同一external models。

baseline候选分别是Fill、Gemm、BatchGemm和Generic。首个真实非baseline点是静态全parallel、exact
pointwise `linalg.generic`中的`1.0 / x`：baseline materialize为`ComputeElementwise Recip`，alternative
materialize为实际`ComputeFill(1)`加`ComputeElementwise Div`。两条clone产生不同typed Tile IR和不同
instruction count，并各自通过Tile→Instr verifier；candidate selection测试把两者送入complete traversal、
instruction lowering、SPM/DDR planning和execution cost。target capabilities关闭division时只枚举baseline。
repo-owned formal numeric与managed-reference model均把Recip定义为`1/x`，因此这个点在当前target语义内等价；
interface仍要求exact静态index relation，SoundBound/unsupported不会授权alternative。

当前临时`CandidateSpec`只保存semantic implementation kind，不写IR attr、artifact或wire schema。若一个task
含多个同kind可替换op，本checkpoint把该kind视为一次complete-clone implementation point；后续Q32.M/S在需要
独立组合时必须由实际choice owner给出op-local邻居，不能把op名称、序列化路径或全局shadow plan提升成协议。

## IndexRelation foundation

`IndexRelation`是transformation-local的MLIR Presburger adapter：domain为destination logical indexes，range为
source logical indexes；内部使用Affine map flattening、`PresburgerRelation`和`ValueBoundsConstraintSet`，不持有
Operation/Value handle。它当前提供：

- 静态shape identity、permutation、broadcast affine relation；
- static slice，以及由ValueBounds解析mixed `OpFoldResult`后的exact slice；
- 等元素数静态row-major reshape；
- composition和point membership；
- Exact、SoundBound、Unsupported、Invalid、ResourceExhausted五态与变量/disjunct budget。

只有Exact允许rewrite。dynamic domain形成SoundBound，composition传播最弱精度，不能把bound重新提升为Exact；
symbolic affine、未知ValueBounds、rank/shape错误、slice越界、element-count/stride溢出和budget耗尽均保持不同结果。
测试逐点覆盖identity/permutation/broadcast、slice→reshape composition、ValueBounds常量、dynamic bound传播、
symbolic unsupported、invalid/overflow和resource budget。每次actual IR mutation后由owner重新构造relation，旧对象
不得缓存或跨pass携带。image/preimage、injectivity/bijectivity、valid-domain intersection、equivalence/implication和
concat/segmented view留给Q32.R的真实consumer驱动扩展。

## Custom interface盘点

| interface | 真实consumer与结论 |
| --- | --- |
| `WaferTilingInterface` / `WaferTilingDemand` | 只重复枚举DPS operands/results和tiling字段；已删除。collective demand直接读取MLIR `TilingInterface`、`DestinationStyleOpInterface`和typed collective fields |
| `WaferLinalgExtCollectiveOpInterface` | CompilationStages、tiling demand、candidate support和tensor-control lowering仍消费rank-group/axis/combiner聚合事实；当前保留，Q32.M迁移后删除aggregate snapshot，只留标准接口无法表达且有generic consumer的最小查询 |
| layout / layout-materialization interfaces | typed ops自身verifier和interface tests存在，但尚无独立production analysis消费collect方法；在Q32.R encoding/view relation落地并由Q32.M迁移consumer前不仓促删除 |
| resource-effect interface | ExecutionCost、DDR planning和shared LifetimeAnalysis真实消费bytes/resource/access；Q32.M只有在typed op、standard effects、custom resources及SSA token/fence保住lifetime root、pending completion、DDR detection和cost semantics后才能删除重复记录 |
| instruction interface | target-format/preflight、cost、SPM high-water、candidate commit和Tile→Instr legality消费instruction family marker；family保留，重复`verifyInstructionContract` boilerplate由Q32.M审计删除 |
| target implementation interface | candidate enumeration与complete-clone BodyEmitter均为production consumer；填补上游Linalg op不可修改且target implementation不是标准Tiling/DPS接口职责的真实gap，保留 |

## 改后验证与剩余边界

- development `check-wafer`：226/229 lit通过、3条按配置unsupported；基础单测285/285。
- target-model Release `check-wafer`：227/229 lit通过、2条按配置unsupported；基础/numeric/bulk
  285/285、54/54、18/18，6条SystemC integration通过。
- 改后标准Llama-2 7B单block TP16：16/16 rank完整输出匹配，19,696 transactions、17 threads、
  final delta 1,232、2,032 managed-reference commands、53,257,728 managed scalars、672 bulk
  commands/matmuls/reorders、0 formal commands和0 bulk formal FMA，最终fresh replay real 56.933秒。
- reciprocal baseline/alternative actual-clone、capability boundary、完整candidate evaluation、IndexRelation
  precision/property/negative和删除接口后的collective demand均有focused回归；文本一致性与diff检查通过。

Q32.I没有完成StorageLoad destination-style迁移、relation完整运算、encoding/view/route actual alternatives、
resident handoff movement删除、production默认winner或bounded joint selection。上述边界按队列转入Q32.R及后续
Q32.B/V/M/S/G；本记录中的性能数字只证明本次无明显回退，不是新的优化目标。
