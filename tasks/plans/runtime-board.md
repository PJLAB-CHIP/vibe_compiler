# Board Runtime实施计划

本计划拆解`tasks/progress.md`中的`runtime-board`执行项；状态唯一以任务队列为准，artifact合同由
`tasks/15-launch-runtime-package.md`和`tasks/16-verification-contract.md`拥有。

```text
Pipeline position:
- Upstream artifact / IR:
  production wafer-compile fresh发布并重新验证的schema-v4 package、显式target launch ABI、typed ResourceId invocation bindings和
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
  retry或把context恢复混入invocation cleanup；one-shot board进程在显式lifecycle和flush后不运行vendor DSO finalizer。
- Completion gate:
  同版本qualification baseline健康后，rank-one kernel bootstrap、单次grid16 kernel SPMD Add与type-6 load + type-7/BPM
  model SPMD Add在真实TX board完成logical tile 0..15执行依据、完整输出比较和重复稳定性验证；该gate不声明physical tile坐标；
  Q6.B整体还须由Direct DTE package闭合真实placement/readiness/completion。
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
   production compile→schema-v4 package→all-rank no-card CTest，避免只有armed硬件脚本才覆盖用户入口。
8. model SPMD Add：由compiler-owned all-rank target artifact形成`tile0..tile15/kcore_fw.so`，type-6只加载，随后以包含48个
   rank-local input/output descriptor（其集合完整覆盖global boundary）及type-7 TLV的BPM调用一次`txLaunchModel`；每tile entry消费
   同一head并只处理自己的slice。要求与kernel gate相同的`~expected` canary、16个互斥slice完整exact、logical tile 0..15执行依据和
   重复稳定性，不声明physical tile坐标。model launch是最终模型发射方向，不能由kernel multi-launch冒充。
9. Direct DTE：只有取得可验证且受支持的BPM/placement/parameter ABI，或另行闭合cluster-compatible artifact/ABI后，才允许
   进入device effect；必须证明logical/remote tile关系、真实receiver readiness、timeout和共同completion，NoTransport结果不能替代。
10. 批次收尾：显式armed执行对应hardware CTest并确认未skip/unsupported，同步任务队列、编号设计和memory，提交相关改动；
    kernel/model Add通过仍不把缺Direct DTE的Q6.B标为done。

当前环境资格缺口和下一动作只记录在`tasks/progress.md`，不在本计划复制动态状态。
