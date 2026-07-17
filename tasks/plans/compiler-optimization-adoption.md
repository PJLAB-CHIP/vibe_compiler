# Compiler Optimization Adoption 实施计划

状态：active；Q33 `compiler-optimization-adoption`是当前唯一`next` row。本文只拆施工checkpoint、验证和退出门槛；
长期pipeline/优化分层由01、05、06、16和18拥有，动态状态只看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前verified post-SPMD StableHLO与official legalization产生的rank-local Linalg/Tensor/SCF/Arith/Math program，
  以及现有candidate/finalization、target backend和rank-count=1/16 source/model baselines。
- Current stage responsibility:
  建立全compiler upstream pass/utility adoption inventory；修复structured semantic recovery、DPS init、view/alias/effect/
  lifetime对偶然producer拓扑和generic canonicalizer的正确性依赖；资格化required normal form、fixed target-independent
  optimization和candidate-local mechanism边界，并把通过验证的fixed subset接入现有named production pipeline。
- Output artifact / IR:
  optimizer-ready rank-local structured tensor program、显式required-normalization postcondition、可重放的adoption records，
  以及供后续physical-dataflow planner复用的typed upstream mechanism边界。adoption record不是IR/package sidecar。
- Downstream consumer:
  Q32.I SemanticOpDescriptor/TargetImplementationProvider、Q32.M policy-free rewrite mechanisms、Q32.S bounded search，
  以及09/12/16 exact resource和equivalent-IR gates。
- User-level driver / named pipeline:
  production仍只经wafer-compile；wafer-opt注册完整相关upstream debug pass families用于replay，但不能由用户手拼成
  第二条production pipeline。
- Explicit non-goals:
  不建立Wafer -O3 pass soup，不向pinned XLA helper加入general HLO optimizer，不把whole-tensor CSE/fusion/pack/vector/
  loop scheduling无条件放入fixed pipeline，不改变physical layout/implementation/residency，不实现Transform control plane，
  不因upstream存在某个pass就承诺采用。
- Completion gate:
  adoption inventory覆盖全部compile cut points；required normalization不依赖generic canonicalizer；mandatory equivalent-IR
  families进入同一production consumer并通过complete rank/all-rank、SPM/DDR/completion/ABI及source-to-SystemC/PyTorch gate；
  每个qualified fixed mechanism在真实corpus发生非零改写且资源/host性能不回退。无效、no-op、blocked或rejected机制有
  明确结论，不能伪装成已采用。
```

## 2. 施工原则

1. 优化单位是IR cut point、pre/postcondition和consumer gate，不是pass名字。
2. availability、adoption mode和qualification正交记录；linked/registered/debug-replayable不等于production采用。
3. required normalization用确定性rewrite与postcondition verifier拥有正确性；generic canonicalizer只做best-effort cleanup。
4. whole pass只有全部rewrite都满足fixed合同才可进入默认pipeline；否则抽取安全pattern subset，或作为candidate-local mechanism。
5. 任何改变SSA sharing、alias、effect、allocation root或lifetime的改写都使旧analysis失效，并从当前IR fresh重算。
6. 每个机制先证明实际发生改写，再证明下游接受和数值正确；只跑pass exit 0、op count或手写fixture不算采用。

## 3. Checkpoints

### Checkpoint A：全 pipeline adoption inventory 与 fresh baseline

审计frontend/SPMD、StableHLO-to-Linalg、structured optimization、candidate materialization、function-boundary finalization、
instruction/target lowering和device publication的全部pass、pattern和library utility。每条record至少包含：

```text
mechanism + upstream revision
IR cut point / operation domain
availability
adoption mode
precondition + numeric/effect contract
actual rewrite count on mandatory corpus
equivalent-IR / downstream exact-gate / numeric result
compile-work / static-cost / host-wall observation
decision + owner
```

首轮必须明确登记：official StableHLO-to-Linalg、canonicalizer、OneShotBufferize、`TilingInterface`、
`linalg::makeTiledShapes`、`scf::tileAndFuseProducerOfSlice`、CSE/SCCP、Linalg/Tensor/SCF/Bufferization transform families，
以及target device publication使用的backend optimization。`wafer-opt`补齐相关upstream pass-family debug registration；
registration不改变production pipeline。

Gate：从现有named production入口fresh记录rank-count=1/16与冻结7B source-to-bundle/SystemC/PyTorch、unsupported/skipped、
compile work和host wall；inventory能区分当前已library-integrated、只debug可用、no-op observed、downstream blocked和rejected。

### Checkpoint B：Equivalent-IR stability 与 deterministic required normalization

先修语义恢复再采用新优化：

- reduction/DPS init从当前SSA、DPS tie、ConstantLike/fill semantics证明，不依赖direct `linalg.fill` defining op或emitter访问顺序；
- canonical semantic descriptor接受named/generic、共享/非共享producer、scalar capture/inline和unit-extent等价形态；
- standard collapse/expand/transpose/extract-slice、`to_tensor`/`to_memref` alias由ViewLike、type、reassociation和SSA重算；
- 把one-trip/full-subview等accepted/rejected所必需的折叠改成显式bounded rewrite与postcondition verifier；generic
  canonicalizer开关或worklist顺序不改变支持性；
- 审计Wafer op memory effects、completion和observable store，确保DCE/CSE/LICM不能删除或越过issue/wait/fence/collective；
- 每次改写后fresh重跑semantic、alias/effect、SPM/DDR lifetime和exact cost，不复用旧analysis。

mandatory metamorphic families由16 §5.1拥有。测试从同一verified module构造等价transaction-local clones，二者进入同一
production scheduling/finalization consumer，不形成用户可见第二pipeline。

Gate：所有等价形态都有合法reserved baseline；canonical descriptor/family domain在语义等价时一致，真实sharing机会可导致
不同cost但不能导致matcher/visitation失败；complete rank/all-rank gates和完整source expected differential通过；failure不泄漏
partial mutation。

### Checkpoint C：fixed hygiene qualification 与 post-adoption baseline

按05合同逐项判定：

- scalar/shape/identity scaffolding的窄CSE或其它cleanup只有在不改变tensor sharing/lifetime时才可评fixed；
- whole-tensor CSE、elementwise/producer fusion、unit-dim/view propagation、inline scalar、empty-tensor elimination和loop
  transforms默认交给Q32.M candidate-local机制；
- pack/vector/convert-to-loops等会抢占physical owner或销毁structured semantics的pass保持rejected/deferred；
- mandatory corpus零改写或无下游收益的SCCP等机制记录`no-op-observed`，不象征性加入production；
- qualified fixed subset接入05同一pipeline builder，debug和production复用body；每项有rewrite telemetry与关闭对照。

Gate：fixed-on/off都通过Equivalent-IR和完整纵向；fixed-on在mandatory corpus发生非零预期改写，compile work/static movement/
host wall无不可解释回退。重新冻结post-adoption rank-count=1/16与7B baseline，作为Q32.I及后续physical-dataflow比较起点。

## 4. Completion Audit

- 01/05/06/16/18、README和progress与live调用点一致；
- current correctness不再依赖generic canonicalizer或direct producer形态；
- adoption records覆盖已链接/注册/实际调用的上游机制，并记录qualified/no-op/blocked/rejected结论；
- fixed pipeline只含qualified target-independent机制，candidate-only清单直接交给Q32.M；
- 双配置build、lit/unit/CTest、dependency/organization checks和mandatory source/model vertical有fresh结果；
- 稳定经验同步memory，计划完成后移入archive并提交。

任一项未满足时Q32.I保持blocked；不得以“pass可运行”或“标准MLIR通常支持”替代该gate。
