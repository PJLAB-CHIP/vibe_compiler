# Target Bulk Qualification 实施计划

状态：historical/completed。Q22.B已于2026-07-14按本计划闭合；当前合同以tasks/16、tasks/17和
`tasks/progress.md`为准，本文件只保留施工与审计记录，不再作为active入口。

本计划原对应Q22.B，只拆解受管oneDNN、bulk adapter、离线资格记录和runtime admission的施工顺序。formal numeric语义与
codec由Q22.N及`tasks/17`拥有，物理layout由`tasks/08`拥有，证据口径由`tasks/16`拥有；状态只看`tasks/progress.md`。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q22.N已验证的ModelProfileId、ResolvedNumericCommand、NumericSemanticsProfile、13-format codec、formal GEMM kernel与
  checked FormalNumericWorkBudget；tasks/08唯一physical layout helper。Q22.B不消费TargetLLVMModuleBundle、packet或SystemC。
- Current stage responsibility:
  受管引入唯一oneDNN CPU dependency；在完整command/profile/domain/environment preflight后，把target raw snapshot经
  target-owned codec/layout变为logical dense temporary，调用一次已准入MatMul和必要reorder，再由target-owned写回原子commit。
  独立离线producer按calibrate -> freeze -> validate签发可readback资格，runtime只接受record exact-match的admission。
- Output artifact / IR:
  immutable BulkBackendQualificationPolicy/Record、process-local BulkBackendAdmission registry、BulkExecutionEnvironment identity和
  atomic bulk tensor result/dispatch evidence。record是离线验证artifact，不进入compiler IR、ExecutableBundle或package。
- Downstream consumer:
  Q22.V source vertical在formal budget外的mandatory large GEMM必须命中admitted row；Q22.S不是本任务前置，未来只同步调用
  已准入bulk kernel，不为每个MAC创建event。
- User-level driver / named pipeline:
  受管host tool `wafer-cmodel-qualify-bulk`以独立no-replace calibrate/freeze/validate mode生成和验证records；正式target-model
  runtime以后只读已冻结registry。wafer-opt/pass、临时benchmark和runtime debug slow override不能签发资格。
- Explicit non-goals:
  不修改NumericSemanticsProfile、compiler legality、target comparator或Q22.L bundle；不把oneDNN默认行为、有限random corpus、
  effective ISA或implementation name单独当语义证明；不接Host CRT、packet、SystemC或board。
- Completion gate:
  默认关闭的feature-off基础链无oneDNN链接；feature-on受管source/build/license/loaded identity通过。adapter覆盖logical-dense与
  Cx/NCx tail、alias/canary、resolved descriptors、scratch/reorder budget和atomic failure。至少一条mandatory large GEMM在
  runtime formal budget外由冻结record exact-match后一次MatMul命中，缺record/domain/environment或reference implementation
  资格不符时fail closed且无scalar fallback；required tests不得skip/unsupported。
```

## Checkpoints

### 1. Typed bulk schema、预算与拒绝优先

- 定义无默认值的backend identity、environment fingerprint、value domain、qualification policy/record、admission kind和稳定
  failure reason；所有identity使用typed fields和canonical digest，不从filename、implementation字符串或cache命中恢复语义。
- formal runtime MAC预算继续由Q22.N唯一计算；bulk dispatcher在input/output分配前完成command、profile、shape/layout、domain、
  environment和record exact-match。预算内可显式选择formal，预算外无admission必须返回`bulk_backend_unavailable`。
- `bit-exact`只接受解析/穷举/implementation-source证明；有限corpus资格编码为payload/domain digest精确收缩的
  `profile-bounded`，未匹配输入拒绝。target comparator与backend envelope分离。

### 2. 受管oneDNN closure

- 固定readiness已验证的oneDNN 3.12 source commit/full digest、Apache-2.0 license、CPU SEQ/INFERENCE/MATMUL/REORDER静态构建
  options和唯一CMake imported target；基础compiler feature-off不链接，feature-on缺root/record或identity mismatch配置失败。
- clean build运行上游/project smoke，record source/build/options、headers/library、version、loaded artifact、transitive thread
  runtime和toolchain identity。首个SEQ profile把caller视为唯一实际worker并显式readback fenv/MXCSR；未来pool runtime必须新增
  profile和逐worker hook，不继承该结论。
- process-global max-ISA/hints/cache设置先于其它oneDNN API并核对status；冲突profile使用独立subprocess。

### 3. Target-owned adapter与一次MatMul

- 输入先完整snapshot；target codec/layout helper负责compact/Cx/NCx decode、block/tail/padding，oneDNN只看同dtype logical dense
  memory。destination先写private f32/s32 temporary，再由target formal finalize/codec/layout pack提交。
- MatMul只开放Q22.N已发布的F16/BF16/F32同dtype语义候选；逐row核对F32 fused accumulator、K递增与oneDNN实际reduction差异，
  不能证明的row保持rejected。TF32/I8只保留typed candidate，不因library支持dtype自动开放。
- primitive显式deterministic、strict fpmath、user scratchpad、无bias/post-op/scale/zero-point；检查2..12D、batch broadcast、轴和
  stride、plain N-contiguous destination。`any`只用于immutable weights，创建PD后读取resolved descriptor并纳入identity。
- 统计每command MatMul/reorder调用、formal MAC和allocated bytes；alias、padding/tail/canary、overflow、descriptor failure和
  memory budget失败均无architectural output或partial flags。

### 4. 三阶段资格producer与readback

- calibrate只读calibration manifest并产生immutable raw record；freeze在任何held-out执行前固定backend comparator/envelope、
  proof、semantic/domain/environment、calibration digest、预注册held-out digest及disjoint proof；validate只读policy执行held-out。
- final record绑定三阶段顺序/log/artifact digest、formal/profile/tool/source、payload/domain、formal/backend output、冻结阈值与
  实际误差/special classification、resolved descriptors/attrs/implementation和完整HostPlatformFingerprint。
- parser拒绝unknown/duplicate/missing field、noncanonical JSON、路径别名、digest/options/environment/loaded-object变化和
  calibration/held-out重叠；publication使用atomic no-replace。

### 5. Runtime admission、large-GEMM和验证

- release registry只导入已readback record；runtime exact-match才构造admission。same semantic在不同shape/domain/environment是
  独立row，cache key包含完整profile、adapter、descriptor、thread/fenv/platform和backend identity。
- generated non-batched/batched corpus覆盖M/N/K、tail、normal/near-zero/subnormal/special及adversarial边界；mandatory large case
  超过formal budget，断言一次MatMul、formal MAC为零、无slow fallback并与共同formal oracle按冻结policy比较。
- pinned-host只把实际非reference implementation且cold create/JIT、reorder和warm execute证据通过的row标
  performance-qualified；correctness gate不使用易抖动绝对wall time。
- 运行feature-on/off full unit、lit、CTest和显式unsupported审计；同步tasks/16/17、queue和必要memory，计划归档后提交。

## 完成判据

Q22.B完成只签发特定semantic/profile/domain/environment下可审计、可重复且fail-closed的bulk admission。它不表示source
workload已自动dispatch（由Q22.V证明），也不表示Host CRT、SystemC、packet、board numeric或timing完成。

## 完成记录

- 受管oneDNN 3.12固定commit、archive/library digest、license、SEQ/INFERENCE/MATMUL/REORDER build与API smoke；feature-on
  缺numeric/root/record/identity时configuration fail，feature-off binary无oneDNN/OpenMP/TBB closure。
- 首批F16/BF16/F32同dtype GEMM由target-owned Cx/NCx codec/layout提升为F32 dense，一次oneDNN MatMul、最多一次weights
  reorder，再经formal GEMM finalize和target pack提交；64³ row超过runtime formal budget仍无scalar fallback。
- canonical absolute no-alias/no-replace `calibrate -> freeze -> validate`、finite-payload `profile-bounded` policy、完整
  environment/descriptor/evidence readback及runtime exact-match admission已闭合；admission携带冻结的actual implementation和
  resolved descriptor digest，执行后与当前primitive evidence逐项复核。malformed/tampered、implementation/descriptor、
  environment、budget或reference implementation drift均fail closed。
- 2026-07-15综合重放中feature-on base/numeric/bulk分别164/164、47/47、14/14，lit为250 pass/2个预期
  feature-inverse unsupported，CTest 22/22；feature-off base 164/164，lit为249 pass/3个明确feature unsupported，
  CTest 12/12。
