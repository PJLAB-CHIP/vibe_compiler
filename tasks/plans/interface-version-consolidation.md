# Q55 接口版本收敛实施计划

设计合同见`tasks/20-interface-evolution.md`，状态只看`tasks/progress.md`。本计划不重新定义package、runtime、
profiler、frontend或target语义。

状态：`done`。

## Checkpoint

1. **边界清单**：为active Wafer版本常量和编号语义字符串标注真实外围边界、外部事实或current-only内部表示；
   明确保留清单，未知项fail closed而不是机械替换。
2. **外围接口收敛**：package/ABI/evidence边界只保留一个current identity和集中parser/loader检查；identifier、symbol、
   wire record和nested record都不携带编号。
3. **内部表示收敛**：删除frontend nested版本、repo-owned helper/dependency snapshot版本、profile重复版本字段，
   并把算法、model、hash domain、loader policy和current schema字符串改为稳定语义名；不保留旧reader。
4. **测试与文档同步**：更新current fixture、golden、negative test和02、11、14-17文档；外部依赖、NPY、license、
   vendor规格、device runtime版本及archive/raw evidence保持不变。Board calibration保留硬件观测、输入、oracle和校验价值，
   迁移到current package/runtime/status接口；依赖未闭合lowering的source vertical只保留current source/oracle合同并明确阻塞。
5. **完成检查**：`rg`只剩allowlist；运行diff check、source organization、fresh并行build、受影响unit/lit/tool tests，
   更新memory与Q55状态后提交。

## 外部事实例外

- device runtime、第三方/release、外部文件格式、license和vendor specification版本；
- `tasks/archive/`与historical raw evidence中的原始外部事实。

Wafer-owned编号schema、nested `version`、带编号identifier、symbol和语义字符串全部删除；
不因已有测试或ABI名称已经存在而保留。

## 完成结果

- Checkpoint 1–4 已完成：frontend、package、profile、target ABI/CRT、dependency record、model、workload和测试fixture
  均收敛为一种无编号current表示；真实外部版本只保留在allowlist边界。
- package manifest以schema identity和exact fields验证，target ABI/CRT symbol使用无编号canonical名称，profiler与Direct-DTE
  record以magic、size、offset、alignment和guard验证唯一布局；没有旧reader、compatibility alias或编号分支。
- raw Board probe已迁到current package/runtime/status接口；22个raw probe加complete-Tile add普通/profile共24个
  `current-interface` no-card CTest证明各driver能生成并验证完整package。10个真实Board CTest进入同一串行runner，
  但本任务不执行真实板测。
- source-level calibration已改为current global source与host oracle。当前global lowering尚不能承接的optimizer/collective
  case不恢复退役SPMD/manifest路径，只保留受测source/oracle合同并在inventory中标明阻塞条件；full-4096 K-tiled
  current-source的SPM失败已fresh复现，M-tiled profile的package equality、full-output和Primary/Count/Trace要求已迁入catalog。
- M-tiled profile的current global-source runner继续保留普通/profile编译、package equality、target GEMM call、no-card、完整输出和
  profile报告校验；在global lowering闭合前保持未注册并fail closed，而不是删除校准实现。
- Board CMake/CTest只登记真实host/no-card/Board入口，inventory测试反向检查这些名称确实存在；catalog仍是case语义唯一来源，
  CMake不复制case注册表。旧reader、旧CLI、旧manifest字段和版本分叉保持零残留。
- Checkpoint 5的fresh build、核心unit/link、Board host/no-card、source organization、残留扫描和memory同步已经闭合；真实板端
  execution及被当前global lowering阻塞的source case不属于Q55完成条件。
