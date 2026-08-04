# Target ABI 收口实施计划

状态：board-ready。任务状态以 `tasks/progress.md` 中的 `target-abi-retirement` 为准；真实板端验证完成前不标 done。

本任务把当前 Wafer 后端收口到唯一可执行 ABI，并同时删除错误的 target profile 抽象和重复的
指令数据类型白名单。当前 compiler 只有一个 Wafer target backend；target 身份、runtime ABI、format
encoding 和 instruction legality 分别由其真实 owner 维护，不再由一个全局 registry row 捆绑。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已完成 target-abstract 选择、TileRegion 到 Instr 转换、SPM/DDR planning、worker placement、completion
  normalization 和 final rank verification 的 typed Instr IR。
- Current stage responsibility:
  按唯一 current TargetCall/CRT ABI 将 target-admitted Instr 转成 verified Target LLVM；format owner验证
  dtype 可编码性，instruction owner验证本指令的附加限制，artifact/runtime owner只接受当前 wire schema。
- Output artifact / IR:
  只引用 current ordinary TargetCall 和共享 NCC/DTE lifecycle symbol 的 Target LLVM module，以及只使用
  当前 manifest schema、entry ABI、Direct-DTE status ABI 和 profile-companion schema 的 package。
- Downstream consumer:
  device linker、repo-owned TargetCall frontend、SystemC/TargetModel、profiler companion、wafer-run/no-card
  和 TX board provider。
- User-level driver / named pipeline:
  `wafer-compile` 固定进入 Wafer target lowering；不提供 `--target-profile`。`wafer-run` 只消费已验证的
  current package。
- Explicit non-goals:
  不做 operator propagation、instruction synthesis、SMT 证明、candidate selection 或新的数值语义表示；
  不把不同 launch form 错当成同一 ABI 的历史版本。
- Completion gate:
  TargetProfile、V1/V2 Kernel Runtime ABI、legacy ordinary TargetCall、LocalFence 和旧 schema reader 在
  current 源码/IR/artifact 入口全部消失；GEMM 明确拒绝 FP32，其余指令消费统一可编码 dtype；source 到
  package、model/no-card 和真实 Llama compile 形成 fresh 纵向闭环。真实板端通过前状态只能是 board-ready。
```

## 1. 唯一事实源

| 事实 | Owner | 终态 |
| --- | --- | --- |
| compiler target | Wafer production driver / named lowering pipeline | 当前只有 Wafer 后端，不存在用户可选 profile |
| target identity | package publication | 固定 current target identity，只用于 artifact/runtime exact join |
| Kernel Runtime ABI | TargetCall/CRT | 只保留当前 worker-aware ordinary ABI 和共享 lifecycle ABI |
| dtype encoding | `TargetFormat` | 一份 current engine encoding registry，无 profile key/compatibility indirection |
| instruction dtype legality | Instr verifier / target preflight | GEMM 拒绝 FP32；其它指令不再用私有 op×dtype 子表缩窄统一 dtype domain |
| package schema | package parser/serializer | serializer写当前值，parser只接受当前值，无旧 reader/translator |
| launch entry ABI | runtime launch owner | 保留语义不同的 launch form；删除旧 accepted spelling，current 名称不带伪历史后缀 |
| Direct-DTE status ABI | status owner | 只保留 cache-line-owned current layout，无 V1 常量或 fallback |
| profiler companion schema | profiler owner | registry ordinal变化后只读写新的 current schema/basis |

`TargetProfileId`、`TargetProfileRecord`、`formatCompatibilityProfile` 和
`numericCompatibilityProfile` 全部删除。format、convert、numeric capability、scheduling、model、artifact 和
runtime API 不再接收一个恒定且无法改变行为的 target-profile 参数。不得用 `TargetContract`、`TargetMachine`
或其它新总包装替代它。

## 2. Current TargetCall 与 completion

- 删除 legacy ordinary descriptor、CRT compatibility wrapper、header declaration、symbol checker row和测试；
  current worker-aware symbol的外部 spelling 保持 ABI 实际值，内部 builtin 使用稳定语义名。
- 删除 `TargetCallProfileAvailability`、descriptor availability 字段和 lookup 的 profile 分支。
- 删除 `instr.local_fence`、`TargetCallBuiltin::LocalFence` 和 `wafer_tx81_local_fence`；所有 producer 使用
  `instr.ncc_join participants=[worker0]`，decoder/model/profiler只保留 typed participant join。
- registry 只包含 current ordinary calls、NCC join 和 Direct-DTE lifecycle calls；测试锁定 exact
  symbol/signature/semantic 序列，不再按旧/new prefix 形成两套表。

## 3. ABI 与 schema 收口

- package manifest保留 `schema_version` 作为 wire mismatch gate，但只接受当前 schema；删除 target profile字段，
  target identity、runtime ABI、module format和launch事实由各自字段表达。
- runtime ABI只接受当前 worker-aware spelling，不保留 V1/V2 parser、enum或compatibility分支。
- rank-local、grid row-table和cluster row-table是不同launch形态，不是三个历史ABI；保留需要的形态，但类型和值
  改为稳定语义名，parser只接受current spelling。
- Direct-DTE status只保留当前64-byte cache-line-owned layout；删除V1 spelling以及常量上的V2后缀。
- TargetCall ordinal改变时，profile companion升级一次并只接受新schema/basis；不保留旧ordinal translator。
- 其它带版本号的独立wire/numeric schema只有确实存在旧reader、旧执行分支或同职责重复实现时才进入本任务；
  单一current schema的版本字段用于fail-closed，不因字符串含`v1`机械删除。

## 4. 实施 Checkpoints

1. 删除target profile的用户选择和CompilationRequest传播，使production driver固定进入Wafer target lowering。
2. 将format、convert、numeric capability、scheduling和model registry改成无profile key的current事实源。
3. 修正dtype legality：统一format encoding负责可编码性，GEMM verifier/preflight单独拒绝FP32，删除reduce/pool等
   擅自缩窄的op×dtype表；补FP16/BF16/FP32 reduce正例和FP32 GEMM负例。
4. canonicalize NCC join并删除LocalFence；删除legacy ordinary call/wrapper和profile availability。
5. 收口package/runtime/entry/status/profile-companion schema，只保留current parser与serializer。
6. 重放source→Instr→TargetLLVM→package→no-card/model纵向；用本轮完整Release构建重跑实际Llama block并dump
   layout movement前后final Instr IR。

## 5. Verification Contract

- CLI：`--target-profile` 不存在，production compile无需隐藏默认值；
- dtype：F16/BF16/F32 reduce通过current target preflight，FP32 GEMM在target effect前稳定失败；
- registry：无profile availability、legacy ordinary、`_v2` builtin或LocalFence，symbol/signature唯一；
- conversion：ordinary worker0、worker1/2、oriented GEMM、NCCJoin和Direct-DTE lifecycle走同一current surface；
- artifact：package与profile companion只接受current schema，旧profile/ABI/status spelling在device effect前失败；
- consumers：TargetCall/SystemC transaction、model launch、TX provider、profiler和conformance checker只消费current ABI；
- no-card：rank 1/16完整package、guard、entry/slot/ABI exact join；
- scale：实际Llama-2 7B block由本轮fresh compiler完成production compile，并生成可比较的final Instr IR。

只删CLI、只改默认值、只让Llama越过reduce gate或只通过局部decoder测试都不算完成。实现不保留迁移alias、
旧reader、fallback wrapper或第二份能力事实源。

## 6. Fresh 纵向结果

- actual Llama-2 7B block使用hidden size 4096、intermediate size 11008、sequence length 16、FP16和16 ranks；
  production compile、schema-7 package、current runtime ABI及完整no-card launch preflight fresh通过。
- `--dump-compiler-ir`对production与`--optimization-preset=none`各生成16份final Instr MLIR和16份Target LLVM，
  all-and-only rank文件均非空。两边都能形成verified package并通过no-card。
- fresh host验证包括742项unit test、20项独立SystemC/model CTest、208项默认lit、14项compiler/runtime CTest、
  8项focused Tools/Runtime lit、112-symbol CRT闭包和65-row format encoding一致性检查。focused lit另有1项
  target-model-disabled反向配置case因当前build启用target model而按设计unsupported，不计为执行通过。
- production相对none的16-rank final Instr统计：总instruction 11648降到6852（-4796，-41.2%）；
  gather/scatter 2352→1568，DTE send/recv分别960→60，DTE wait 960→120，elementwise 1520→1070，
  NCC join 1792→870；GEMM 144、RDMA 1312、WDMA 832、convert 624和reduce 32保持不变。Instr dump总字节
  4986338→3289270（-34.0%）。这些是结构计数，不是板端性能结论。
- full-shape source额外暴露两个与ABI/dtype无关的通用lowering缺口：dynamic Tensor-layout SPM slice现用标准
  `memref.subview`加已有compact copy表达；rank-zero endpoint现由`PhysicalAccessRelation`返回唯一空坐标，使原有
  relation descriptor planner生成zero-source-stride广播。两者都有focused IR回归，不增加私有layout或case-specific op。

当前host/source/package/no-card边界达到board-ready；真实板端ordinary/nonzero-worker或Direct-DTE completion尚未执行，
因此不能标done，也不能从上述IR计数声称硬件收益。
