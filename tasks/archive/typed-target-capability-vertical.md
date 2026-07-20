# Typed Target-Capability Vertical 收口记录

状态：Q32.V已完成并归档。当前任务状态以`tasks/progress.md`为准，mandatory mechanism producers由Q32.M继续推进。

设计owner：`tasks/06-physical-dataflow-synthesis.md`、`tasks/11-instruction-ir.md`、
`tasks/14-target-conversion-module-publication.md`、`tasks/15-launch-runtime-package.md`、
`tasks/16-verification-contract.md`、`tasks/17-target-execution-model.md`；施工checkpoint为
`tasks/plans/physical-dataflow-synthesis.md`的Checkpoint E。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR: Q32.B提交的selected structured/tile/instruction actual clone；source侧使用MLIR Linalg named contraction或精确indexing maps，movement侧使用destination-style DDR/SPM typed views和shared physical encoding。
- Current stage responsibility: 从current IR重算mapped logical-to-physical transfer cover并物化显式双端root-relative offsets；为fill物化logical-valid/physical-footprint typed domain和raw scalar；把source/Tile GEMM orientation无损送到Instr、v2 TargetCall、CRT和formal/SystemC consumer，同时保持v1 compact/implicit-normal合同不变。
- Output artifact / IR: verifier-legal mapped wafer.instr.rdma/wdma descriptors、typed wafer.tile/instr.fill、typed wafer.tile/instr.gemm；closed wafer-tx81-single-card-kernel-v2 / wafer-tx81-kernel-v2 profile row及wafer_tx81_gemm_oriented_v2 exact call；最终TargetTransaction只包含下游执行所需的address/count/format/orientation。
- Downstream consumer: Q32.M/S共同candidate owner、target LLVM/module publication、shared TargetCall decoder、repo-owned formal numeric kernel、plain/SystemC functional model；external board/provider admission仍属于later gate。
- User-level driver / named pipeline: 默认wafer-compile在selected winner选择相应registered profile后消费这些typed rows；wafer-opt只提供source→Instr和Instr→target LLVM focused replay，不形成第二条用户pipeline。
- Explicit non-goals: 不引入planner capability query、route sidecar、RequiredCapabilitySet或package schema升级；不把external model/board admission反馈给compile-time choice；不实现Count、quantized GEMM、packet provenance或timing。
- Completion gate: source/ODS/parser/printer/verifier、mapped descriptor cover/range/overflow、physical count/raw scalar、v1/v2 exact signature、CRT closure、formal NN/NT/TN/TT、repo-owned SystemC和current v1 regression fresh通过；非法双侧stride、offset pair/range、stored shape及wrong ABI在effect前拒绝。
```

## 实现结果

### Mapped RDMA / WDMA

- `TransferRealizability`对DDR Tensor与SPM Tensor/Cx/NCx之间的static byte-addressable identity relation建立exact
  mapped proof；非canonical、bitpacked或无法静态闭合的relation保持fail closed。
- tile load/store先保留既有compact descriptor；compact不成立时从current typed views枚举、合并logical segments，并为每条
  RDMA/WDMA物化成对的`src_offset`/`dst_offset`，包括显式0。每条descriptor独立闭合payload、target字段和两端range。
- target lowering在allocation base之后checked加入offset，仍复用既有RDMA/WDMA CRT signature；DDR planning和full-buffer
  handoff也消费相同offset事实。RDMA拒绝destination stride，WDMA拒绝source stride，不把DMA伪装成双侧layout engine。

### Physical-footprint fill

- Tile/Instr新增closed `FillDomain`。attr缺省保持v1 `logical_valid` Tensor语义；显式`physical_footprint`从shared physical
  encoding计算count，普通dtype使用physical elements，bitpacked BOOL使用physical bytes乘8。
- raw scalar ABI保存storage bits：窄整数zero-extend到32-bit字段，浮点bitcast为APInt。BOOL true固定传`1`，不依赖
  signed truncation；target model再次按logical format canonical width验证。
- TDMA BOOL encoding只在`BitpackedPhysicalFootprintFill`约束下开放。model canary证明Cx对齐块padding/tail和BOOL unused
  physical bits均被覆盖，logical-valid path不能冒充full-footprint initialization。

### Versioned oriented GEMM

- structured source支持`linalg.matmul_transpose_a/b`、batched named variants和精确rank-2 GEMM indexing maps；
  orientation从当前Linalg maps重算并进入typed Tile/Instr attrs，不从op名、shape猜测或保存shadow plan。normal/normal仍省略attrs。
- v1 profile和`wafer_tx81_gemm`保持implicit normal/normal。新增closed v2 profile/Kernel ABI和
  `wafer_tx81_gemm_oriented_v2`十字段exact signature；v2显式引用v1 format/numeric compatibility profile，但两个ABI identity不合并。
- Tile/Instr verifier、numeric key和functional transaction按orientation验证stored shapes与M/K/N/batch；CRT把两个closed enum
  直接传给`SetTransflag`。wrong v1 ABI、单侧attr、stored-shape mismatch和unknown enum均在target effect前拒绝。
- formal非方阵非对称payload逐一执行NN/NT/TN/TT；SystemC从compiler-generated v2 module经shared decoder执行四个typed
  transactions并核对最终physical output。oriented bulk/board admission仍需各自later exact record，不能从formal结果外推。

## Package 与 capability 边界

当前repo-owned consumers已从exact TargetCall和profile/Kernel ABI直接获得全部必需事实，不需要逐row package capability
preflight。因此Q32.V没有制造`RequiredCapabilitySet`或schema-v4；schema-v3、TargetArtifactBundle和no-card runtime合同保持不变。
若后续真实consumer需要per-row environment admission，只能在post-selection从winner Instr/TargetCall派生并另行闭合readback。

## Fresh 证据

- focused source/IR/lowering lit覆盖structured NT/TN/TT、batched NT/TN、mapped RDMA/WDMA多descriptor及双端0/非0 offset、
  physical Cx/BOOL fill、v2 oriented call和v1 wrong-ABI negative。
- formal/model unit覆盖非方阵NN/NT/TN/TT、stored-shape legality、Cx完整128-lane physical fill、BOOL 16-bit footprint和raw scalar。
- repo-owned SystemC由compiler-generated v2 module执行NN/NT/TN/TT并比较非对称F16结果；同一source的Tensor↔Cx movement
  实际经过mapped descriptor path。
- shared registry/decoder逐字段覆盖110项production TargetCall；CRT symbol closure和conformance覆盖13 formats、65 encoding rows、
  36 convert routes以及v1 fixed/v2 explicit GEMM transflag。
- development feature-off：239项lit中236 pass、3项configured unsupported；base unit 295/295；CTest 12/12。
  unsupported精确为StableHLO-disabled、target-model-bulk和target-model-source三个feature边界测试。
- SystemC feature-on独立配置：239项lit中200 pass、39项按该feature matrix configured unsupported；base unit
  294 pass、1项StableHLO-disabled skip，numeric unit 56/56，7条SystemC integration/negative入口通过，CTest 21/21。
- target-model Release feature-on：239项lit中237 pass、2项configured unsupported；base/numeric/bulk unit分别
  295/295、56/56、18/18，7条SystemC integration/negative入口通过，CTest 24/24。unsupported精确为
  StableHLO-disabled和target-model-disabled两个反向配置测试。
- 标准Llama-2 7B单block使用冻结的long-period finite corpus重新执行默认Release production driver：16-rank verified
  package原子发布后完成repo-owned SystemC/PyTorch differential，得到`transactions=20060`、
  `systemc_threads=17`、`final_delta=1258`、`formal_commands=0`、
  `managed_reference_commands=2032`、`managed_reference_scalars=53257728`、`bulk_commands=672`、
  `bulk_matmuls=672`、`bulk_reorders=672`、`bulk_formal_fmas=0`。每rank 65,536个F16输出在Q31收紧的
  `atol=0.004, rtol=0.002`合同下匹配同一payload的PyTorch eager expected。
- source/IR organization、dependency consistency、clang-format、`git diff --check`、110项CRT symbol closure和
  CRT conformance（formats/encoding rows/convert routes=`13/65/36`，groups=`4/23/9`）通过。

## 完成边界

Q32.V完成的是三类target能力的typed compiler/model纵向，不是它们进入joint search的证明。Q32.M负责构造实际mechanism
alternatives并迁移剩余layout/resource consumers，Q32.S负责有界组合和winner evidence，Q32.G负责默认production cutover及
旧decision owner退役。external board、exact-package execution、packet和timing仍不在本checkpoint claim内。
