# Physical Relation / Realization 收口记录

状态：Q32.R 已完成并归档。当前任务状态以 `tasks/progress.md` 为准，后续 production-shaped
candidate seam 由 Q32.B 继续推进。

设计 owner：`tasks/06-physical-dataflow-synthesis.md`、`tasks/08-physical-realization.md`、
`tasks/10-compute-movement.md`、`tasks/16-verification-contract.md`、
`tasks/18-source-organization.md`；施工 checkpoint 为
`tasks/plans/physical-dataflow-synthesis.md` 的 Checkpoint C。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: Q32.I 已建立 source implementation interface/external model、MLIR-backed IndexRelation foundation，以及当前 verifier-legal structured tensor / tile-dataflow clone；shape、dtype、view、SSA、memory space/encoding、effect和target profile均已显式。
- Current stage responsibility: 补齐当前 view/transfer/resident mechanism 所需的 relation 查询和证明；由 physical encoding attr interface解释static footprint、alignment、logical valid cardinality、padding cardinality和逐logical-element physical bit segment；从两端typed memref与relation现场证明metadata view、compact DMA、GS和staged movement；把选择物化为真实view/movement/temp IR；将load迁移为explicit destination；用relation-backed full-buffer resident rewrite删除真实spill/reload。
- Output artifact / IR: explicit-source/destination wafer.tile.load、standard memref view、typed wafer.instr RDMA/WDMA/GS、显式SPM temporary，以及跨sibling tile-region的SPM SSA resident handoff；IndexRelation和transfer proof只在本次analysis调用中存在，不进入IR或artifact。
- Downstream consumer: tile-to-instruction conversion、whole-rank SPM planning、whole-variant DDR/event/transport、target ABI/package和repo-owned SystemC；Q32.B继续把这些mechanism接入production-shaped actual-clone frontier。
- User-level driver / named pipeline: wafer-compile source-to-bundle/package/target-model production pipeline；wafer-opt只复用同一scheduler/conversion作局部审计。
- Explicit non-goals: 本checkpoint不建立candidate wire schema、route report、shadow physical plan或第二套relation IR；不完成Q32.B的test seam、Q32.V mapped target能力、Q32.M全mechanism closure、Q32.S联合selection或Q32.G默认cutover。
- Completion gate: relation/property、encoding interface、direct/staged conversion、resident正负例、SPM/DDR/descriptor/completion/ABI回归通过；非7B production-shaped source删除真实中间WDMA/RDMA；标准7B source实际选择resident handoff，完整TP16 package/SystemC/PyTorch differential通过；双配置build/lit/unit/CTest、unsupported清单和组织/依赖/CRT检查通过。
```

## 实现结果

### Relation 与 physical encoding

- `IndexRelation` 现支持 static domain、destination/source domain 交、image/preimage、inverse、concat piece，
  以及 functional、injective、bijective、equivalence 和 implication 查询。proof API继续区分 exact、
  sound bound、unsupported、invalid和resource exhaustion；只有exact结果授权rewrite，查询预算耗尽时fail closed。
- property/unit coverage包含 identity、permutation、broadcast、slice、reshape、concat、composition、domain、
  image/preimage、等价/蕴含、invalid和budget negative；大canonical identity/reshape采用代数证明，不按元素数展开。
- `MemoryAttr` 实现 `WaferPhysicalEncodingAttrInterface`。static logical valid domain仍由memref shape唯一拥有；
  interface验证并返回valid/padding cardinality、physical footprint、natural alignment和每个valid logical index的
  physical bit segment。padding内容不被定义，Q32.V的physical-fill纵向继续拥有padding/unused-bit写入合同。

### Transfer realization 与 destination-style load

- `TransferRealizability` 从当前source/destination memref、encoding interface和`IndexRelation`现场证明
  metadata view、compact cross-space DMA、SPM gather/scatter和two-step staged movement；证明值不保存route选择。
- `wafer.tile.load` 已迁移为DestinationStyleOpInterface：显式DDR source和预先创建的SPM destination、零result；
  所有builder、source-to-tile conversion、tile-to-instruction lowering、effect/layout verifier和测试同步迁移。
- current compact Tensor reshape可物化metadata view；Cx/NCx physical map不等价时保留真实GS；external
  DDR Tensor到非Tensor SPM使用显式Tensor temporary形成DMA+GS staged route。permutation、broadcast、slice、
  reshape、tail和proof-budget负例均fail closed。

### Resident handoff

- full-buffer handoff对producer/consumer的collapse、expand、cast和full subview chain构造exact
  `IndexRelation`，要求functional/injective/bijective与完整source-domain cover，并从encoding interface重证
  footprint和valid元素数。
- accepted rewrite把producer SPM allocation作为tile-region result，经SSA传给dependent consumer，删除producer
  WDMA与consumer RDMA；必要的local physical copy显式成为GS。partial view、非Tensor encoding、effect barrier、
  external output和不完整fanout保持spill baseline。
- 新增production-shaped source回归强制形成producer/collective两个scope，最终IR保留一个SPM handoff、删除中间
  WDMA/RDMA并保留终端WDMA；现有producer-chain和transpose fusion回归同时通过。

## Fresh 证据

- development：230项lit中227 pass、3 configured unsupported；base unit 290/290；CTest 12/12。
  unsupported精确为StableHLO-disabled、target-model-bulk和target-model-source三个feature边界测试。
- target-model Release：230项lit中228 pass、2 configured unsupported；CTest 23/23，覆盖base、numeric、bulk和
  6个SystemC integration/negative入口。unsupported精确为StableHLO-disabled和target-model-disabled测试。
- 标准Llama-2 7B单block的rank-local scheduler在真实structured source上选择26条full-buffer SPM handoff；
  36个task全部形成complete selected IR。非7B source的final IR直接证明一组中间WDMA/RDMA消失。
- fresh Release TP16 production重新发布16-rank verified package并完成SystemC/PyTorch differential：
  `transactions=19696`、`systemc_threads=17`、`final_delta=1232`、`formal_commands=0`、
  `managed_reference_commands=2032`、`managed_reference_scalars=53257728`、`bulk_commands=672`、
  `bulk_matmuls=672`、`bulk_reorders=672`、`bulk_formal_fmas=0`；每rank 65,536个F16输出在既有
  `atol=0.02, rtol=0.01`合同下匹配同一最终long-period finite corpus的PyTorch eager expected。
- source/IR organization、dependency consistency、clang-format、`git diff --check`、109项CRT symbol closure和
  CRT conformance（formats/encoding rows/convert routes=`13/65/36`，groups=`4/23/9`）通过。

## 完成边界

Q32.R完成的是relation/encoding/transfer基础设施、destination-style load和首条真实resident mechanism纵向。
这些mechanism尚未作为Q32.B production-shaped test seam中的baseline/optimized actual clones进入all-rank frontier；
Q32.V target extension、Q32.M剩余choice producers、Q32.S bounded joint selection与Q32.G默认production cutover仍按队列顺序推进。

