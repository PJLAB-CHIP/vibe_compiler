# Target ABI 退役实施计划

状态：设计已收敛，待实现。任务状态以`tasks/progress.md`中的`target-abi-retirement`为准。

本任务只收口TX81 target profile、Kernel Runtime ABI、TargetCall/CRT和对应artifact consumer。它先于
`semantic-superoptimization`独立实施、验证和提交；不引入SMT、候选生成、优化axis或新的数值语义表示。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已完成target-abstract选择、TileRegion到Instr转换、SPM/DDR planning、worker placement、completion
  normalization和final rank verification的typed Instr IR，以及显式选择的TX81 target profile。
- Current stage responsibility:
  将所有current target-admitted Instr唯一映射到V3 TargetCall/CRT ABI，形成verified Target LLVM；使
  target profile、format/numeric registry、model、profiler、package和runtime只接受同一个current ABI。
- Output artifact / IR:
  只引用current V3和七个共享NCC/DTE lifecycle symbols的Target LLVM module、schema-v6 package及其
  可验证TargetCall/SystemC/board执行输入。
- Downstream consumer:
  device linker、repo-owned TargetCall frontend、SystemC/TargetModel、profile companion、wafer-run/no-card
  和TX board provider。
- User-level driver / named pipeline:
  wafer-compile --target-profile=wafer-tx81-single-card-kernel-v3；wafer-run只消费已验证package。
- Explicit non-goals:
  不做operator propagation、instruction synthesis、SMT证明、candidate selection、numeric representation
  cleanup或新的package capability schema；不改变RankLocalPointerBlockV1等与TargetProfile版本无关的entry ABI。
- Completion gate:
  V1/V2 profile、Kernel Runtime ABI、legacy ordinary TargetCall和LocalFence在current源码/IR/artifact入口中
  全部消失；V3从source到package、model/no-card和真实板端形成fresh纵向闭环。
```

## 1. Current Facts 与终态

计划状态：未实现。当前代码仍以三profile、217个descriptor和LocalFence/NCCJoin双表示运行；本表右列是
实现完成后的目标，不能在实现前当作current evidence。

| 边界 | 当前事实 | 目标 |
| --- | --- | --- |
| target profile | V1/V2/V3三个closed row，共享target identity和module format | 只保留`wafer-tx81-single-card-kernel-v3` |
| Kernel Runtime ABI | `wafer-tx81-kernel-v1/v2/v3` | 只保留`wafer-tx81-kernel-v3` |
| format/numeric ownership | V2/V3通过compatibility profile指向V1 registry | registry事实直接归V3，不保留compatibility indirection |
| TargetCall | 217 = 原始112 + 104个V3 ordinary counterpart + 1个V3 DTE issue | 112 = 7个共享NCC/DTE row + 104个V3 ordinary + 1个V3 DTE issue |
| completion | `instr.local_fence`和`instr.ncc_join`并存 | 只保留typed `instr.ncc_join` |
| legacy package | schema-v6可解析V1/V2 profile spelling | V1/V2 spelling在typed profile parser处直接unsupported |

原始112中有104个legacy ordinary row和8个共享row；8个共享row由LocalFence、NCCJoin及6个DTE lifecycle
row组成。因此先删104个legacy ordinary，再删LocalFence，得到112。这个112不是旧112-prefix，也不是217种
独立指令语义。

## 2. Single Current Target Profile

- 删除`TargetProfileId`和`KernelRuntimeABIId`的V1/V2 factory、enum value、registry row、parser spelling及
  所有三分支判断；V3 factory和canonical spelling保持不变，不发明V4或无版本alias。
- `TargetProfileRecord`删除`formatCompatibilityProfile`和`numericCompatibilityProfile`。format、convert、
  target numeric capability、numeric semantics/model policy直接以V3为key；因target identity改变而产生的
  internal digest/qualification变化按当前生成流程重建，不保留V1 alias。
- `TargetIdentityId::waferTx81SingleCard()`、`elf-riscv64`和显式profile选择继续保留。单profile不意味着删除
  target identity、package中的profile join或future target扩展边界。
- compiler model launch、TargetModel、TX board provider、runtime compatibility、schedule/cost policy、工具默认值和
  fixtures中的V1/V2硬编码统一迁移到V3。
- 不机械删除其它名字中的`V1`/`V2`：entry ABI、Direct-DTE status schema、loader policy、formal model policy
  等只有在其自身合同被本任务替代时才变化。

## 3. V3-only TargetCall 与 CRT

### 3.1 Closed registry

- 删除104个legacy ordinary descriptor、CRT compatibility wrapper、header declaration、symbol checker row和测试。
- 删除`TargetCallProfileAvailability`、descriptor `availability`字段、profile bit过滤和按profile选择同一semantic的
  lookup分支。lowering仍携带`TargetProfileId`做format/numeric/target legality，但TargetCall lookup只有一个current surface。
- 删除`TargetCallBuiltin::LocalFence`及`wafer_tx81_local_fence`；`TargetCallBuiltin::GemmOrientedV2`改为稳定语义名
  `GemmOriented`，真实current symbol继续是`wafer_tx81_gemm_oriented_v3`。
- 保留所有真实V3 `_v3` symbol spelling。共享NCC/DTE lifecycle symbol继续使用现有无版本拼写，因为它们本身就是
  current ABI，不通过重命名制造另一轮ABI。
- 新registry按过滤当前registry后的稳定顺序形成：先是NCCJoin和6个共享DTE lifecycle row，再是现有104个V3
  ordinary row及末尾V3 Direct-DTE send issue。测试锁定exact symbol/signature/semantic序列和总数112。
- 不新增TargetCall registry digest。descriptor ordinal只在profile companion中序列化，由对应schema/versioned
  correlation basis负责兼容性。

### 3.2 Completion canonicalization

- 所有producer先把worker0 completion materialize为`SyncNCCJoinOp participants=[worker0]`，再删除
  `SyncLocalFenceOp`的ODS、builder、parser/printer、verifier、analysis、cost和conversion分支。
- TargetCall decoder只产生`TargetNCCJoinTransaction`；model和profiler不再保留LocalFence variant/site kind。
- 当前typed decoder把LocalFence解释为worker0 participant，但CRT分别调用`TsmWaitfinish()`与
  `TsmWaitfinish_bywork(0)`。实现不能仅凭host模型宣称硬件等价：必须以fresh板端worker0和multi-worker隔离case
  证明typed NCCJoin实现正确；失败时任务保持未完成，不能用扩大participant scope掩盖差异。

### 3.3 Artifact compatibility

- package manifest wire shape保持schema v6；删除profile row后，V1/V2 manifest自然在typed parser处失败，不为纯
  accepted-value收缩机械升级schema。
- current V3 schema-v6 package若只含保留call仍可消费；含LocalFence或已删除symbol的module由Target LLVM/
  TargetCall exact closure拒绝。没有V1/V2 translator、reader alias或worker0 wrapper fallback。
- profiler site map确实序列化`target_call_ordinal`，因此profile companion schema从v6升级到v7，
  `typed-target-call-ordinal-ssa-identity-occurrence-v1`升级到v2，并写入registry size 112。其它package schema、
  Target LLVM metadata schema和record ABI没有字段/语义变化时不连带升级。

## 4. 实施 Checkpoints

1. 先把format、convert、target numeric capability、numeric/model policy和所有driver/provider默认值迁到V3，证明不再需要
   V1 compatibility key；更新由target key变化导致的internal qualification/digest fixtures。
2. 删除V1/V2 profile和Kernel Runtime ABI row，使CLI、manifest、readback、runtime/model environment对旧spelling
   在任何device effect前稳定失败。
3. 将全部LocalFence producer canonicalize为typed NCCJoin，补齐worker0/multi-worker host正反例后删除LocalFence IR和
   TargetCall/CRT surface。
4. 删除104个legacy ordinary call/wrapper及profile availability，收口112项registry并更新device link、TargetCall frontend、
   SystemC、profiler companion和conformance工具。
5. 重放source→Instr→TargetLLVM→package→no-card/model纵向，形成FP16/BF16 board-ready artifacts；最后串行执行
   fresh板端completion和ordinary/V3 worker/DTE定向case。

每个checkpoint都必须在独立工作树现状上保持V3 baseline可生成；不得通过同时引入superoptimizer来补偿ABI迁移失败。

## 5. Verification Contract

- profile/parser：只注册V3；V1/V2 target profile和Kernel Runtime ABI spelling、manifest及module metadata均拒绝；
- registry：exact 112 row、symbol唯一、signature/semantic/worker issue domain完整，无legacy ordinary、`_v2`和LocalFence；
- conversion：ordinary worker0、worker1/2、oriented GEMM、NCCJoin、Direct-DTE begin/prepare/issue/wait/finish均通过同一V3 preflight；
- artifact：schema-v6 V3 package roundtrip保持，V1/V2 package pre-effect失败；profile companion v7/v2 ordinal basis
  roundtrip，旧companion明确unsupported；
- consumers：TargetCall/SystemC transaction、model launch、TX provider、profiler和conformance checker只消费V3；
- no-card：rank 1/16完整package、guard、entry/slot/profile/ABI exact join；
- board：FP16/BF16普通worker0，以及至少一个nonzero worker或Direct-DTE case；串行验证output、guard、completion和
  lifecycle。无板时只能标`board-ready`，不能标`done`。

只修改parser、只把默认值换成V3、只得到112这个数字或只通过host decoder都不算完成。实现提交必须独立于
`semantic-superoptimization`，建议提交标题为`compiler: retire legacy target ABIs`。
