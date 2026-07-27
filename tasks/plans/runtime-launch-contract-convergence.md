# Runtime Launch Contract 收口计划

本计划只拆解 `tasks/15-launch-runtime-package.md` 已收敛的实现工作；状态仍以
`tasks/progress.md` 为准。

## Pipeline position

- Upstream artifact / IR:
  带 `TargetProfileId`、两值 `RuntimeLaunchKind`、完整 rank domain、typed ABI slots、
  module exports 和 transport requirements 的 `ExecutableBundle` / `TargetArtifactBundle`。
- Current stage responsibility:
  compiler 一次性形成 tagged `RuntimeLaunchContract`；package schema v6 精确序列化并验证；
  `wafer-run` 只按 kernel 或 model 顶层分派，provider 在 kernel 内部执行已验证的 form 和
  ordered phases。
- Output artifact / IR:
  不含旧 `TargetLaunchABIId` 的 compiler/runtime API、schema-v6 package、no-card plan 和
  board submission lifecycle。
- Downstream consumer:
  package loader、RuntimeSession、BoardRuntime、TX provider、profiling evidence 和板端 case。
- User-level driver / named pipeline:
  `wafer-compile --launch-kind=kernel|model` 生成 package；唯一 `wafer-run` 入口消费 package。
- Explicit non-goals:
  不新增 case-specific runtime 入口；不把 cluster、Direct DTE、parameter layout 或
  prepare/main 提升成第三种 launch；不兼容迁移 schema v5。
- Completion gate:
  production 代码不存在旧 launch ABI API/字符串；旧 schema 和旧 CLI fail closed；kernel/model
  正例、Direct DTE kernel 正例、no-card 与 runtime lifecycle 回归通过，且 main submit 不藏在 wait。

## 实施 checkpoint

1. 建立 closed `RuntimeLaunchContract`，只允许 kernel/model 顶层 kind，并显式区分 kernel form、
   entry ABI、ordered phases 与独立 transport requirements。
2. 将 compiler request、target publication 与 `wafer-compile` 迁移到 `--launch-kind`；在完整
   accepted IR/rank domain仍可检查时形成唯一 resolved contract并由 `ExecutableBundle` 持有，
   transport acceptance作为并列事实而不成为runtime launch选择器。
3. 将 package、profile companion、RuntimeSession 和 JSON wire form 升级到 schema v6
   `target.launch`，明确拒绝 schema v5、`launch_abi` 和四个旧 spelling。
4. 将 BoardRuntime/driver/provider 收口到 kernel phase 与 model 两个 submit 面；同一绝对 deadline
   串行完成 ordered phases，transport status 只由 entry transport 决定。
5. 迁移测试、profile evidence 与板端 driver，执行旧接口全局清零、并行构建、unit/lit/CTest；
   同步设计、队列和稳定 memory 后提交。
