# Target Command Legality Closure 实施计划

本计划对应`tasks/progress.md`中的Q0.L，只拆解施工顺序和验证checkpoint；架构与IR/artifact合同仍由
`tasks/01`、`tasks/06`、`tasks/08`、`tasks/10`、`tasks/11`、`tasks/14`、`tasks/15`和`tasks/16`拥有，
任务状态只看`tasks/progress.md`。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  production CompilationRequest、validated ExecutionConfig、Q15 verified grouped program、Q16 accepted all-rank
  instruction/memory/completion IR，以及tasks/08 layout facts和tasks/14 target/ABI事实。
- Current stage responsibility:
  让显式registered target profile无损贯穿production transaction；在target effect前按profile×engine×logical-format
  拒绝无证据encoding；把tile-level elementwise map和reduce init/combiner完整materialize成无旁路语义的terminal IR。
- Output artifact / IR:
  profile-bearing ExecutableBundle、transaction-local prepared target LLVM/ABI artifact、TargetArtifactBundle和
  PackageManifest；terminal instruction IR不含indexing_maps或reduce init，且每个format-bearing command已通过typed gate。
- Downstream consumer:
  Q22.N读取唯一logical-format/profile事实，Q22.L消费prepared target LLVM/ABI artifact，Q17/Q18继续消费同一
  profile-bearing bundles，Q6.B只重放本gate后的fresh package。
- User-level driver / named pipeline:
  production只经required --target-profile=<registered-id>的wafer-compile；debug target named pipeline使用同一registry
  和typed conversion request，只补IR-local正反例。
- Explicit non-goals:
  不实现Q22 numeric codec、oneDNN、Host CRT或SystemC；不把profile写入module attr、名字、sidecar或隐式default；
  不用CModel补做compiler已经丢失的map/init语义；不宣称board或vendor-exact numeric。
- Completion gate:
  typed profile和identity逐层readback；registry全枚举及无证据row fail closed；map/reduce source语义在effect前完整展开，
  terminal budget checked；full conversion、CRT conformance、late-failure atomicity、Q20/Q21 source/package/reference/
  no-card链路以本轮fresh结果通过，mandatory tests实际执行而非unsupported/skipped。
```

## Checkpoints

### 1. Closed target profile 与 format registry

- 在tasks/14唯一owner下定义typed `TargetProfileId`、registered CLI spelling及到typed target identity/runtime ABI的闭映射。
- 首个opaque canonical spelling固定为`wafer-tx81-single-card-kernel-v1`；它只组合已有identity/ABI事实，不表示未知
  silicon revision或Q22 numeric profile，也不得被拆分恢复字段。
- 定义shared logical-format descriptor和profile×instruction-family×logical-format encoding/legality查询；layout只引用
  tasks/08事实，不复制Cx/NCx/BOOL footprint规则。准入集合唯一引用
  `tasks/14`“Q0.L 首个 closed profile”的`TargetFormatEngine`矩阵和独立typed convert 36-route whitelist；
  本计划不复制第二份format事实源。
- 先固定全枚举unit test和unknown/unsupported negative，再接入任何consumer。

完成判据：registry没有default/fallback；每个公开row唯一、可枚举、可诊断，`S`不被解读为数值正确性，
CT BOOL受op-kind gate约束，typed convert不放宽通用CT row，UINT/64-bit/TF32等无证据组合明确拒绝。

### 2. Profile-bearing production artifacts

- `ExecutionConfig` factory同时要求rank-count和typed profile；`CompilationRequest`、Q16 bundle和Q17 artifact bundle只复制
  这个typed value，不从IR、CLI string或环境重建。
- `wafer-compile`要求`--target-profile`并在任何helper/输出side effect前解析；所有production与test callsite显式传入。
- target preparation按registry得到target/runtime ABI identity；Q16/Q17 config join、Q18 schema-v3 manifest在target object中
  显式保存canonical profile spelling，parse后由typed profile反查并逐字段核对identity/runtime ABI/module format；runtime
  environment preflight同样exact-match。
- debug target named pipeline将required option在pipeline construction时解析为同一typed request；其它IR-local pipeline不伪造
  profile-bearing completion证据。

完成判据：缺失、unknown、duplicate和跨bundle mismatch在effect前失败；rank-15 target/package late failure无partial publication。

### 3. Map-free terminal elementwise

- `wafer.tile.elementwise`继续显式表达已验证的permutation/broadcast map；tile→instruction先用movement和same-shape
  temporary materialize该关系，再创建terminal elementwise。
- `wafer.instr.elementwise`的ODS/verifier/parser/printer不接受`indexing_maps`，target conversion另作defensive residual check。
- identity map同样strip；不能用当前movement IR稳定表达的dynamic/non-permutation case在任何effect前拒绝。

完成判据：正例覆盖identity、permutation及可表示broadcast，负例覆盖残留attr和不可表示map；target LLVM不读取map。

### 4. Init-first ordered reduce composite

- `wafer.tile.reduce`保持SSA init或typed `init_value`的唯一语义；验证两者互斥、类型一致和combiner精确可识别。
- selector在本基线不生成`reductionSplitSizes`；旧的neutral-init partial-reduce再combine会重关联source顺序，完整输入不能
  通过SPM或terminal预算时直接拒绝，未来顺序carry split另立可验证合同。
- tile→instruction按canonical lexicographic reduction order生成init-first fill、result-shaped slice movement和map-free
  elementwise ping-pong accumulator，最后移动到destination；每个engine command和completion都计入独立per-rank 4096预算。
- `wafer.instr.reduce`移除init operand/attr；source path不生成native reduce。只有未来compiler-owned全输入域等价证明才能替换
  composite，不能消费Q22 model policy授权。

完成判据：sum/max/min/avg中能由typed combiner精确表达的row有数值/顺序证据；dynamic init、unsupported combiner、overflow和
预算超限在effect前拒绝；terminal op无init。

### 5. Target legality 与纵向闭合

- target conversion在call emission前同时执行profile/format gate、残留map/init gate和既有geometry/narrowing gate。
- 修正把无证据i64/unsigned/TF32 format-bearing command当正例的测试；保留logical/reference-only能力与target-emittable能力
  的明确区分。
- 运行unit、lit、CTest和unsupported审计；用同一`wafer-compile`重放Q20 rank-count=1/16 linear/MLP及Q21 16-rank
  tiny Llama的compile、all-rank target modules、package、reference differential和no-card preflight。

完成判据：上述证据全部fresh且mandatory vertical实际执行；设计文档、队列和稳定memory同步后提交，Q0.L才可标done。

## 施工依赖与并行边界

```text
registry definition
  -> profile-bearing artifacts
  -> target profile/format preflight

tile elementwise/reduce contract
  -> terminal ODS/verifier cleanup
  -> map/reduce lowering
  -> target residual-illegal checks

profile/format lane + map/reduce lane
  -> production vertical replay
  -> Q0.L completion
```

registry/profile lane和tile semantic lane可以并行审计及在不重叠文件中施工；二者都可能修改target lowering和production
integration tests，因此这些join点由单一owner顺序合并并统一验证。Q22.N和Q22.L只有Q0.L完整通过后才进入`next`。
