# Wafer 单卡纵向切片实施计划

状态：active。任务状态和直接依赖以 `tasks/progress.md` 为准；本文件只拆解施工 checkpoint，不声明新的
IR/ABI 合同。设计边界由 `tasks/01-16` 对应 owner 文档承担，审计证据见
`tasks/archive/12-architecture-evidence-reset.md`。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  verified StableHLO program directory、显式 target topology/execution ranks 和当前 local tensor/group IR。
- Current stage responsibility:
  先闭合 instruction/target correctness，再以显式 per-rank clone形成完整 RankExecutable[]，验证后原子发布
  ExecutableBundle、typed C++ manifest/canonical JSON，并由 no-card runtime和reference executor消费。
- Output artifact / IR:
  rank-count=1或16的完整 executable bundle、target modules、verified manifest、runtime plan和reference结果。
- Downstream consumer:
  no-card RuntimeSession、reference executor、后续真实board adapter和长期多卡扩展。
- User-level driver / named pipeline:
  新的 wafer-compile 用户入口；wafer-opt只保留IR/pass调试。
- Explicit non-goals:
  本计划不实现跨卡/MPMD/hybrid rank class、Protobuf/WCRE、cross-model migration、cost calibration或板端性能。
- Completion gate:
  单tile与单卡16-rank linear/MLP均经同一driver生成完整原子bundle并与CPU reference一致；随后tiny Llama
  经同一16-rank路径；任何失败不发布partial output。板端gate保持外部阻塞。
```

## Global Rules

- correctness任务直接消费当前IR，不得被package或长期identity基础设施阻塞。
- candidate可抽样早期拒绝，但accepted artifact必须覆盖完整traversal和完整entry。
- V0采用显式per-rank static clone；禁止默认rank 0和production pass-only rank identity。
- package不复制instruction schedule；runtime不重做sharding、candidate、memory或transport planning。
- 所有输出先写transaction-owned staging root，验证完成后一次发布。
- reference executor只证明中层语义，不冒充target packet、board completion或性能证据。

## Checkpoint 1: Architecture Baseline

- 固化代码/测试事实、P0/P1风险和保留/删除/延后判定。
- 将旧7份long-horizon plans退出active索引并保留历史审计。
- 重写tasks/progress为短DAG；同步01/14/15/16近期合同。

完成：active任务不再引用不存在的Proto/WCRE/registry作为correctness前置；旧对象没有被伪报实现。

## Checkpoint 2: Target Correctness

状态：已完成。`check-wafer`新鲜执行25个C++ unit和225个lit（224 pass、1个feature-inverse
unsupported），CTest 3/3通过；CRT conformance与109-symbol closure checker通过。

- target lowering在mutation前preflight，并在clone上执行；临时拒绝尚不能结构保持的nested/multiblock/call。
- 建立共享physical geometry与ABI narrowing检查，先覆盖RDMA/WDMA、DTE、convert、GEMM和shape-bearing family。
- 修复async issue的read/write resource lifetime；无completion proof不复用。
- selector在完整traversal commit落地前只接受full-traversal tile，防止partial output。
- `check-wafer`实际执行lit和C++ unit tests。

完成：control-flow、partial traversal、OOB/narrowing、premature reuse定向negative tests全部通过且失败无mutation。

## Checkpoint 3: Typed Grouped-Program Driver

状态：已完成，对应Q15。`check-wafer`新鲜执行28个C++ unit和229个lit（228 pass、1个feature-inverse
unsupported），CTest 3/3通过；dependency、IR organization、CRT conformance和109-symbol closure checks通过。

- 定义最小move-only `CompilationRequest`和factory-only `ExecutionConfig`；rank-count无默认值且只接受1或16。
- 新增`wafer-compile`，让`wafer-opt`退出program-directory I/O、stage selector和final publication。
- 在transaction-owned source snapshot和唯一staging中完成frontend admission、exact topology/mesh、
  pinned helper SPMD、post-SPMD metadata/payload验证、local normalization和logical group formation。
- 保留helper未typed rewrite的non-IR program members；final directory重新parse/verify后才no-replace发布。

完成：真实configured helper的mandatory cases实际执行；helper/pass/readback/publication任一失败不改变source或
既有final且无staging残留。输出只到verified grouped program，不声称per-rank bundle或target artifact。

## Checkpoint 4: Per-Rank Executable Bundle

状态：已完成，对应Q16。`check-wafer`新鲜执行28个C++ unit和229个lit（228 pass、1个feature-inverse
unsupported）；rank-15 late failure、真实column-sharded 16-rank正例和row/HF transport负例通过。

- 直接消费Q15 verified grouped program；rank-count=1/16分别建立all-and-only isolated rank clones。
- rank通过typed API显式传入；所有rank执行相同group candidate、full traversal、layout、instruction、
  SPM/DDR、completion和resource validation；当前transport合同明确为`None`，collective fail closed。
- 每个passing clone形成typed C++ `RankExecutable`；全部rank及resource/completion coverage验证后才构造
  atomic `ExecutableBundle`。
- 禁止默认rank 0、代表rank、filename identity或未实现的executable dialect。

完成：rank 0/1的local slice与payload binding可区分；任一rank或late legality failure不形成partial bundle，
旧final不变。physical peer-positive bundle留给communication owner扩展，不作为当前Q16伪完成项。

## Checkpoint 5: Target Artifact Bundle

状态：已完成，对应Q17；Q0、Q16已完成。

- 只消费Q16 atomic `ExecutableBundle`，对每个rank执行target conversion、object/CRT/device link。
- device link只写transaction staging；验证all-and-only module set、required/undefined symbols、entry symbol、
  module format、ABI摘要和content digest。
- 全部rank readback通过后一次发布typed `TargetArtifactBundle`；manifest由下一checkpoint另行构造。

完成：任一compile/link/symbol/digest/publication late failure无final `.so`或partial rank set，Q16 bundle保持不变。

## Checkpoint 6: Manifest And No-Card Runtime

状态：已完成，对应Q18。

- 建立唯一C++ typed `PackageManifest`、semantic verifier和canonical JSON parser/serializer。
- 用typed slot/resource双射表示ABI；all-and-only关联Q16 ranks/resources/completion与Q17 modules/entries/digests。
- 删除production instruction/LLVM文本解析和package instruction schedule。
- runtime只消费verified manifest形成typed preflight/session plan；Python只作薄CLI或历史converter。

完成：Python/C++不再有不同acceptance；schema version、unknown field、bogus format和重复slot均fail closed；
manifest/package late failure不发布partial package。

## Checkpoint 7: Reference Core, Transport Activation And Vertical Gates

状态：进行中；Q19 core和Q16.T已完成，当前Q19.M为唯一doing，Q20/Q21按直接前置blocked。

- Q19 core只接受Q16 `ExecutableBundle`和typed role/index invocation tensors，先把accepted rank all-and-only投影为
  invocation-local immutable `ReferenceProgram`，再执行memref/instruction/control-flow；projection不序列化、不进入
  package，也不保存planner或collective schedule。当前supported flat/tile-region op、static view、single-block
  `scf.if`/`scf.for`和acyclic `cf.br`/`cf.cond_br`已投影为value-id/block graph；执行期不再读取mutable MLIR。
- single-rank engine已用APInt/APFloat闭合非zero-point convert和RND_MODE 0..4，整条convert先计算后commit；
  对旧wrapper、public header、register资料及`libinstr_tx81.a`的审计只证明`zp`写入`param.src1`、mode写入
  `ctrl.rnd_mode`，没有zero-point公式或hardware stochastic seed/state/推进合同；zero-point继续在projection fail closed。
  stochastic按用户要求采用reference-only common policy：显式execution seed、invocation-local SplitMix64逐dynamic element
  推进，并按上下相邻可表示值距离比例选择；同seed可重放但不冒充hardware-equivalent oracle。穷举gate覆盖全部
  zero-point kind拒绝和每个rounding kind的mode4执行，Q19 core由此按显式accepted capability收口。Q16/Q17已删除
  “module恰好一个func.func”的过度约束，统一消费唯一typed entry加private non-recursive direct-call closure；
  reference projection把同一closure复制为function/block/value-id graph并执行call forwarding。convert kind的
  source/destination type pair和parameter policy由Wafer IR typed helper唯一拥有，executor从它派生numeric format；
  TableGen全枚举gate执行全部非zero-point kind并在input/arena前拒绝全部zero-point kind，且以非零FP32→TF32→FP32
  roundtrip证明19-bit APFloat语义与4-byte physical storage的显式编码。executor代码已拆为internal immutable graph、
  accepted-IR projection、numeric/storage、immutable interpreter和薄public orchestration；只有projection读取MLIR，
  interpreter只消费投影。fixed-seed非平凡lowered-group differential已用独立
  CPU loop oracle证明所有hidden channel和两层非零bias均影响完整输出；
  test-only独立slow layout oracle已跨compact/Cx/NCx、rank/dtype、全部tail对齐台阶和channel block边界逐坐标
  对照production footprint/offset，并证明logical坐标映射唯一、位于physical range且越界统一失败；它不进入production
  artifact或协议。
- Q16.T的logical identity与physical acceptance checkpoints已完成：`channel_id`贯穿collective/tile，全部现有p2p
  materialization生成communication/phase/round/payload-slice typed identity；memory planning后完整rank domain按
  source/destination/message核对peer/bytes/range/wait，验证normal sender profile、分配receiver FSM，并原子补
  `DirectDTEBindingAttr`与`TransportContract::DirectDTE`。CRT opaque event/status lowering、atomic target publication、
  schema-v2 manifest和no-card runtime requirements已同步闭合；identity不从op顺序或名字猜，package不复制
  p2p body或per-op binding。
- Q19.M只消费Q16.T accepted bundle，以deterministic event scheduler执行DTE send/recv/wait并检测peer mismatch、
  duplicate recv、unmatched token和no-progress/deadlock；禁止手写DTE module绕过bundle gate。
- Q20/Q21真实exporter固定source revision/config/seed/dtype/shape和独立NumPy CPU reference；gate顺序仍为
  rank-count=1 linear/MLP、16-rank linear/MLP、16-rank tiny Llama。

完成：Q19/Q16.T/Q19.M各自的component gate先闭合；随后三条纵向gate均只经`wafer-compile`，reference结果与独立
CPU oracle比较。board test未运行时不声称board numeric/completion完成。

## Checkpoint 8: Structural Cleanup

- 按稳定职责拆candidate generation/materialization/legality/cost/commit。
- 按leaf family拆group-to-tile和tile-to-instr implementation；按package/runtime/provider拆runtime。
- 只有dependency/registration证据要求时才拆dialect，禁止纯目录美化。

完成：模块依赖单向，公共interface有consumer，旧matcher/text bypass和重复validator已删除。

## Verification

- 每批运行定向lit/unit tests、完整`check-wafer`、CTest和unsupported/skipped清单。
- 纵向gate记录source、上游artifact、accepted bundle成员、manifest、reference和下游consumer。
- 每批同步受影响设计、progress和必要memory，并独立提交。
