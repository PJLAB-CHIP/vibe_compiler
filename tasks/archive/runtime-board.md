# Board Runtime实施计划（已完成）

本计划拆解`tasks/progress.md`中的`runtime-board`执行项；状态唯一以任务队列为准，artifact合同由
`tasks/15-launch-runtime-package.md`和`tasks/16-verification-contract.md`拥有。

```text
Pipeline position:
- Upstream artifact / IR:
  production wafer-compile fresh发布并重新验证的当前typed package、显式target launch ABI、typed ResourceId invocation bindings和
  configured board inventory。
- Current stage responsibility:
  完成package/binding静态preflight后选择device并读取只读inventory；在allocation/load/launch前验证显式runtime library
  digest、runtime/PCI/device/tile identity和aggregate free-memory admission，再按launch ABI执行H2D、kernel module
  load/resolve或type-6 graph load、对应launch、trusted completion和D2H；
  成功路径及provider明确保持usable的失败才逆序cleanup，provider context poison后立即隔离且不再发任何低层TX API。
- Output artifact / IR:
  invocation-local typed board result和完整host-visible output bytes；不修改package或compiler IR。
- Downstream consumer:
  board完整CPU comparison、后续model/board numeric correlation和correctness通过后的profile校准。
- User-level driver / named pipeline:
  board-capable wafer-run；默认feature-off wafer-run保持vendor-free no-card入口。
- Explicit non-goals:
  不在runtime重新planning，不从名字或路径恢复ABI，不把sample、host target model或性能结果当作board correctness；不自动reset、power、
  retry或把context恢复混入invocation cleanup；one-shot board进程在显式lifecycle和flush后不运行vendor DSO finalizer；
  production runtime/tool不解析或特判Add、collective、shape、payload、resource名或test path，case构造、CPU expected、
  canary、结果比较和failure injection只放在`test/`或`unittests/`。
- Completion gate:
  同版本qualification baseline健康后，rank-one kernel bootstrap、单次grid16 kernel SPMD Add与type-6 load + type-7/BPM
  model SPMD Add在真实TX board完成logical tile 0..15执行依据、完整输出比较和重复稳定性验证；该gate不声明physical tile坐标；
  Direct DTE package另以完整16-rank通信、transport status、完整CPU exact和重复运行闭合真实placement/readiness/completion。
```

## Checkpoints

1. 构建/执行边界：board feature只构建按需加载vendor DSO的adapter，不授权live execution；hardware CTest还需默认关闭的独立
   registration option、完整预期identity和执行时`WAFER_EXECUTE_HARDWARE_TESTS=1`，普通CTest不得触卡。
2. typed provider lifecycle：ResourceId all-and-only binding、exact ELF/entry、stage failure suppression、cleanup-safe逆序释放及
   poisoned fail-stop由unit test闭合；TX首错后不调用error-string、copyback、unload、free、reset或power。
3. rank-one bootstrap：production StableHLO add fresh发布kernel package，使用非平凡输入、独立CPU expected、真实board至少连续
   两次完整exact。它只验证单算子、module pointer-block ABI和环境，不证明多tile。
4. 环境资格：driver/public runtime与宿主boot-source Kcore payload、module toolchain字节级闭合，debugfs cached
   version/status对应且module imports由匹配SDK Kcore exports闭合；没有device-RAM dump时不声明运行中payload逐字节一致。
   缺符号且无完整数值比较的vendor sample被明确排除。
5. multi-launch回归：既有16-rank `transport:none` fresh Add以16个stream分别执行`grid=(1,1,1)`，保留完整domain preflight、
   aggregate capacity、ownership、deadline、原子D2H、cleanup和exact回归。当前V5.6调度静态确认这16次均由tile0执行，因此该项
   只证明provider多提交/lifecycle，不是16-tile gate。
6. kernel SPMD Add：一次普通`txLaunchKernel`使用`grid=(16,1,1)`、`block=(1,1,1)`和rank-major pointer table；device entry以
   `__get_pid(0)`选择互不重叠的rank slice。每个output先H2D预填为逐字节不同于expected的`~expected` canary，要求16个slice
   all-and-only写回并完整Add exact；结合full-good logical inventory、限定V5.6固定logical tile调度和一次aggregate launch，形成
   logical tile 0..15执行依据，不声明physical tile坐标。该项是kernel单算子qualification，不是最终模型发射边界。
7. model ABI静态闭合：以当前V5.6 exact library/firmware为资格对象，由typed builder、artifact export/readback和fake provider验证
   `txLoadGraph`的type-6 load、type-7 run payload、
   56-byte head + 72-byte dyninfo、16份tile-specific module共享symbol、`entry(head)`、device-address ownership和module-name identity；
   fake/static gate先覆盖layout、overflow、缺失/额外module、错误ordinal、cleanup和poison；kernel-grid与model都另有非hardware
   production compile→当前schema package→all-rank no-card CTest，避免只有armed硬件脚本才覆盖用户入口。
8. model SPMD Add：由compiler-owned all-rank target artifact形成`tile0..tile15/kcore_fw.so`，type-6只加载，随后以包含48个
   rank-local input/output descriptor（其集合完整覆盖global boundary）及type-7 TLV的BPM调用一次`txLaunchModel`；每tile entry消费
   同一head并只处理自己的slice。要求与kernel gate相同的`~expected` canary、16个互斥slice完整exact、logical tile 0..15执行依据和
   重复稳定性，不声明physical tile坐标。model launch是最终模型发射方向，不能由kernel multi-launch冒充。
9. Direct DTE：使用closed `tx81-cluster-direct-dte-prepare-main-v1` launch ABI。Q17将完整16-rank
   target LLVM body聚合成一个payload，由typed `prepare`/`main` export和16个rank interface显式表示；Q18
   schema-v5以16个entry共同引用同一`ModuleId`表示shared execution，module不再复制rank/scope事实。
   provider必须在同一loaded module和同一自定义stream上执行cluster prepare，观测共同terminal后才可发射
   cluster main；两阶段共用一个host deadline。prepare在每个tile上以`__get_pid(0)`取得full-16 C-INS逻辑索引，先执行
   `init_tile_id(pid, 4)`建立TX81单卡4×4逻辑/物理映射所需的row-length状态，再执行`direct_sync_init(16)`；prepare不创建
   FSM/DTE handle，main的per-rank begin不得再次init ready slots。accepted binding只携带peer receiver SPM offset；main内
   TX81 CRT发送端用`get_tile_spm_addr_base(remote_tile,4,4)+offset`形成firmware destination，本地source和receiver FSM保持raw
   SPM offset。current status-v2以64-byte storage/alignment独占cache line，offset 0为逻辑`u32`。main terminal后先D2H
   并校验16个预填storage的offset-0 status全为`SUCCESS`，再允许user output D2H。cluster rank-major argument table不得超过
   `0x7d0` bytes。任一prepare/main query error、deadline、status pending/error/unknown都立即quarantine，不再
   D2H/unload/free/destroy，不reset/power/retry。live gate还必须用大于provider共同deadline的one-shot子进程外层
   deadline约束潜在阻塞的vendor调用；外层超时只终止并回收该子进程，停止本次gate且不重试或调用reset/power。
   先以静态聚合ELF、fake trace和无卡production replay闭合，再以
   StableHLO Direct DTE case证明logical/remote tile、receiver readiness、共同completion、16个rank-specific sentinel和完整
   CPU exact；NoTransport结果不能替代。
10. 批次收尾：显式armed执行对应hardware CTest并确认未skip/unsupported，同步任务队列、编号设计和memory，提交相关改动；
    kernel/model Add通过仍不能替代Direct DTE gate。

## Completion evidence

2026-07-22，rank-one、16×grid1 provider smoke、单次grid16 kernel、type-6/type-7 model与cluster Direct DTE均从
production source和schema-v5 package进入已注册hardware CTest并实际通过。kernel/model各以16个互斥64-byte Add slice连续两轮
完整exact；Direct DTE current status-v2以16-rank tree reduction+broadcast、每rank 64个f32（256 bytes）、16个
64-byte/64-aligned status storage的terminal success和完整
CPU exact连续两轮通过。最终ELF反汇编确认recv destination只在对应wait后消费，CRT反汇编确认status publication包含C908
cache clean/invalidate。各成功运行均完成显式cleanup，执行后只读SMI回到运行前memory/process/utilization基线，全程未调用
retry/reset/power；只形成logical tile `0..15`执行证据，不声明physical coordinate。status-v2收尾批次的
host gate为379/379 unit、214 lit passed + 10 feature-configured unsupported、Direct DTE no-card/outer-deadline 2/2，真实
Direct DTE hardware CTest 1/1。Q6.B据此完成，本计划归档。
