# Board Runtime实施计划

本计划拆解`tasks/progress.md`中的`runtime-board`执行项；状态唯一以任务队列为准，artifact合同由
`tasks/15-launch-runtime-package.md`和`tasks/16-verification-contract.md`拥有。

```text
Pipeline position:
- Upstream artifact / IR:
  production wafer-compile fresh发布并重新验证的schema-v3 package、typed ResourceId invocation bindings和configured board inventory。
- Current stage responsibility:
  完成package/binding静态preflight后选择device并读取只读inventory；在allocation/load/launch前验证显式runtime library
  digest、runtime/PCI/device/tile identity和aggregate free-memory admission，再执行H2D、exact module load/resolve、launch、trusted completion和D2H；
  成功路径及provider明确保持usable的失败才逆序cleanup，provider context poison后立即隔离且不再发任何低层TX API。
- Output artifact / IR:
  invocation-local typed board result和完整host-visible output bytes；不修改package或compiler IR。
- Downstream consumer:
  board完整CPU comparison、后续model/board numeric correlation和correctness通过后的profile校准。
- User-level driver / named pipeline:
  board-capable wafer-run；默认feature-off wafer-run保持vendor-free no-card入口。
- Explicit non-goals:
  不在runtime重新planning，不从名字或路径恢复ABI，不把sample、model或性能结果当作board correctness；不自动reset、power、
  retry或把context恢复混入invocation cleanup；one-shot board进程在显式lifecycle和flush后不运行vendor DSO finalizer。
- Completion gate:
  同版本qualification baseline健康后，fresh rank-count=1/16 NoTransport package在真实TX board完成all-and-only lifecycle、
  完整输出比较和重复稳定性验证；Q6.B整体还须由Direct DTE package闭合真实placement/readiness/completion。
```

## Checkpoints

1. 构建/执行边界：board feature只构建按需加载vendor DSO的adapter，不授权live execution；hardware CTest还需默认关闭的独立
   registration option、完整预期identity和执行时`WAFER_EXECUTE_HARDWARE_TESTS=1`，普通CTest不得触卡。
2. typed provider lifecycle：ResourceId all-and-only binding、exact ELF/entry、stage failure suppression、cleanup-safe逆序释放及
   poisoned fail-stop由unit test闭合；TX首错后不调用error-string、copyback、unload、free、reset或power。
3. rank-one bootstrap：production StableHLO add fresh发布package，使用非平凡输入、独立CPU expected、真实board至少连续两次完整exact。
4. 环境资格：driver/public runtime与宿主boot-source Kcore payload、module toolchain字节级闭合，debugfs cached
   version/status对应且module imports由匹配SDK Kcore exports闭合；没有device-RAM dump时不声明运行中payload逐字节一致。
   缺符号且无完整数值比较的vendor sample被明确排除。
5. rank-count=16：先以`transport:none` fresh Add闭合provider-owned all-rank session：完整domain静态preflight、只读16-tile
   inventory、aggregate capacity、全部resource/module/entry ownership、独立stream共同submit、query deadline、原子D2H发布、逆序
   cleanup和完整输出比较。该gate只证明16个logical rank均执行，不把stream当作tile selector，也不宣称rank到physical tile的
   固定映射或16-tile并行利用率。Direct DTE继续在device effect前拒绝，直到vendor BPM/placement ABI或另行设计的
   cluster-compatible artifact/ABI能表达16份独立module与argument block。
6. Direct DTE：只有取得可验证且受支持的BPM/placement/parameter ABI，或另行闭合cluster-compatible artifact/ABI后，才允许
   进入device effect；必须证明logical/remote tile关系、真实receiver readiness、timeout和共同completion，NoTransport结果不能替代。
7. 批次收尾：显式armed执行对应hardware CTest并确认未skip/unsupported，同步任务队列、编号设计和memory，提交相关改动；
   checkpoint 1-5的完成不把仍缺checkpoint 6的Q6.B标为done。

当前环境资格缺口和下一动作只记录在`tasks/progress.md`，不在本计划复制动态状态。
