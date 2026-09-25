# Wafer Compiler通用开发方法

本文件只保存跨任务可复用的构建、测试和调试方法。IR语义、pipeline、package/runtime合同、硬件事实、
任务状态和历史case分别由编号设计、`docs/`、`tasks/progress.md`和`tasks/archive/`拥有。

## 开始与收尾

开工、授权、变更边界和提交流程只维护在[AGENTS](../AGENTS.md)，当前任务从[progress](../tasks/progress.md)进入。

设计事实的阅读入口：架构与pipeline看01；frontend/data看02；structured/physical transformation看05--10；
Instr/memory/communication看11--13；target/package/model看14--17；源码与MLIR工程看18--20；硬件行为看`docs/`。

## Canonical build

构建只使用preset `default`的`build/`；配置时机、并行度和完成检查遵守[AGENTS](../AGENTS.md#canonical主机构建)。
XPASS、UNRESOLVED和TIMEOUT同样不是通过。以下记录容易漏掉的依赖与工具细节。

- Structured e-graph依赖用`utils/deps/bootstrap_deps.py --egraph-sources`同步pinned source和vendor record；CMake只运行
  `cargo build --locked --offline`。缺source、Cargo/rustc或版本不匹配应在configure/build边界失败，compiler invocation
  不启动外部optimizer进程。
- Release LLVM依赖在关闭assertion时需显式设置`LLVM_FORCE_ENABLE_STATS=ON`；否则`--mlir-pass-statistics`可能只有报告标题，
  pass计数不会实际累加。检查安装的`llvm/Config/llvm-config.h`，修正后重建依赖并增量重建同一主工程，不放宽统计断言。
- Shardy依赖检查使用Git ancestry验证已知base。浅克隆即使HEAD正确，也可能缺少ancestor而检查失败；确认缺少的是历史后，
  补全该submodule历史并重新运行`utils/checks/check_deps.py`，不移动pinned HEAD或绕过检查。

常用命令：

```bash
cmake --preset default
cmake --build --preset default -j"$(nproc)"
cmake --build --preset default --target <target> -j"$(nproc)"
ctest --test-dir build -R '<affected-regex>' --output-on-failure -j"$(nproc)"
cmake --build --preset default --target check-wafer -j"$(nproc)"
ctest --preset default -j"$(nproc)"
```

- Lit case从configured build tree或registered CTest启动，让`lit.site.cfg.py`注入tool和dependency；不要直接把
  source-tree `.test`交给lit。
- GTest参数化case先用`--gtest_list_tests`确认展开后的suite identity；filter命中普通定义名不保证命中instances。
- 长命令经`tee`保存日志时启用pipeline failure propagation；package目录、success diagnostic和退出码共同构成结果。
- Install验证从同一build执行`cmake --install`并直接运行install-tree工具；Compiler/Runtime component不建立第二build。

## 测试输入与断言

Shape、整除/tail、tiny oracle和逐项矩阵要求见[测试覆盖](../AGENTS.md#测试覆盖)。

- 对coverage、owner、demand、merge、tail、copy、movement或completion的承诺使用精确计数/关系断言；不要把shape选取
  写入legality或策略。

## Pinned MLIR与API定位

- MLIR行为先查官方文档，具体API以configured build实际使用的pinned include和TableGen路径为准。旁置源码树含有某header
  不能证明当前构建可用。
- 修改已有对象前阅读definition、constructor、direct user、verifier、canonicalization和lowering；用`rg`沿完整调用链查找。
- Crash/assert先用verifier-valid input复现并定位哪个pass首次产生invalid IR；不要在下游增加兜底verifier或名称恢复。

## Source、CMake与命名

Source、header、CMake和测试owner见[18号设计](../tasks/18-source-organization.md)，MLIR组件边界见[19号设计](../tasks/19-mlir-engineering.md)。

- 命名前先用一句话说明对象语义，再核对typed domain和真实对照。Namespace、parent op或强类型已消歧时不重复范围词。
- 反引号路径、Markdown链接和current symbol改名后用`rg`检查残留；多义词逐项分类，不做无语义的全局替换。
- Python子进程和测试使用`-B`或`PYTHONDONTWRITEBYTECODE=1`；源码树不得留下`__pycache__`、`.pyc`或`.pyo`。

## Current-IR变换定位

- 先画出`input current IR -> typed choice -> actual transformation -> verifier -> fresh analysis -> consumer`。
- Relation side state只属于当前IR epoch。Rewrite通过`IRMapping`、listener或显式old→new map更新；找不到replacement时
  fail closed，不能按类型、邻接位置、Location或名称猜测。
- `none`和`search`调试时分别统计controller、candidate/attempt、atomic stage和actual leaf调用次数，确认二者互不调用或fallback。

## Actual SPM feedback定位

- 在讨论合法性前，先确认current Instr中已有actual allocation、alias/effect、completion、lifetime和完整owner relation。
  缺任一项时结果是unknown，不是capacity rejection。
- 只读取唯一MiniMalloc/placement validator的typed结果。Actual capacity rejection可以反馈controller；resource、timeout、
  unsupported和compiler failure不能伪装成容量不足。
- High-water、footprint、working-set、shape公式、buffer count和前一candidate结果只作诊断，不参与admission、pruning或fallback。
- 无owner或stale relation是compiler contract failure；不要按shape、名称或“Region只有一个root”补归因。

## IR膨胀定位

- 从source、post-normalization、Tensor、Tile、bufferized、Instr到target-call逐stage记录op family、wall、RSS和是否实际到达，
  找到首次异常增长；未到达stage不能记成零开销。
- 同时统计logical node、selected execution和actual compute occurrence。Fanout增加时，除explicit replica外producer compute
  不应增加；必要view和transport可增加但必须有current SSA/effect witness。
- 分开统计metadata view、materializing conversion、copy、allocation、global、movement和target call。View不应自动变成movement；
  copy不能依赖后置DCE才消失。
- Instrumentation显式开启且不得改变IR、choice或failure。计数只用于回归与归因，不进入legality或expected-inventory verifier。
- 优先消除重复walk、重复materialization和错误scope，再考虑memo、并行或低层micro-optimization。

## Search与profile定位

- Pre-structural frontier只保存immutable typed choice和continuation；后续choice必须从candidate current IR重算并立即apply。
- Memo key覆盖query实际读取的全部typed field，value只保存pure result。Cache-off、eviction、hash seed和resume切分只能改变work，
  不能改变successor、accepted set或winner。
- Profile只在完整new path通过后开启。记录transition/actualization次数、time-to-first、wall、RSS、IR inventory和publication。
- Priority、memo、component DP和LNS不能签发legality。有损beam/Top-k/LNS-only只能报告budgeted result，不能声称exact。

## Descriptor与completion定位

- Descriptor从current relation、loop bounds和typed layout构造；已有total/bounded proof时直接编码，不先逐元素展开再压缩。
- 修改join/wait前读取current硬件、target lowering和CRT/runtime事实，标注`supported`、`board-observed`、`unknown`或`excluded`。
- Same-worker ordinary RAW/WAR/WAW只保持issue order。Direct-DTE recv wait放在first read/FSM reuse前，send/relay wait放在
  last release/resource reuse前；issue不是默认wait位置，NCC join与DTE wait不能互相替代。
- Operation类别、WDMA、loop/Region/materialization boundary和“保守同步”都不是completion证据。Unknown保持typed unknown。

## Program data与package定位

- Compiler transaction拥有content-stable `ProgramDataSource`和checked `ProgramDataRange`；验证后不重新打开用户path。
  Large-file mmap不防其它进程原地改inode，必要时复制到transaction-owned file并边读边hash。
- Source snapshot只复制IR/metadata/目录结构；parameter/constant按range读取并按selected target representation转换一次。
  Payload I/O账本区分open、read window、bytes、hash、helper和materialization用途。
- Strict package loader从typed descriptor和共享codec重算bytes/alignment，并检查manifest、module、program data、entry和argument
  all-and-only closure。Python断言只补测试，不维护第二reader。
- 所有validation、readback和digest binding在staged root内完成，最后一次no-replace rename原子发布。失败不留partial final目录。
- No-card在provider side effect前完成完整memory/binding/transport/completion验证；历史package和输出不作本轮输入。

## Product source与设备复现

- Product source只使用02号定义的portable StableHLO artifact；text IR只属于明确的focused入口，不是第二reader。
- Advisory verifier与compiler可复用ingestion实现，但compiler仍独立snapshot、拥有并验证输入，不能信任隐藏的“已验证”状态。
- 比较none/search时两次fresh parse/import，各自拥有ProgramData、IR、output和package；source identity只证明输入一致。
- 设备运行与异常停止遵守[板端验证流程](../AGENTS.md#板端验证)；下列方法用于占用核对和故障证据定位。
- 占用检查先对实际设备节点使用有全系统可见权限的`fuser -v`，再以`/proc/<pid>/fd`的字符设备号、exe、UID和cgroup核对持有者。
  不能只搜进程名或只看当前用户；不同容器仍可能持有同一设备。日志服务的设备FD不等于计算任务，但不能仅凭名称将其忽略。
  计算占用存在时等待并重查，权限或身份不明时保持unknown；每次launch前重新确认，检查结果不构成全系统排他租约。
- Runtime成功与完整数值通过后仍检查该次执行窗口的驱动/固件错误事件。LSU TDMA fatal、XID或AP异常是设备故障，
  不能因为host completion成功继续批次或签发健康性能基线。事件必须与boot、PCI设备和执行时间关联。
- 故障后优先只读保留上一boot的kernel journal、已落盘固件日志和runner执行时间。固件单调时间与host墙钟分别记录，
  用明确事件和实际协议顺序对齐；AP reset若发生在host超时退出后的context清理阶段，不能倒推成最初的卡死原因。
- 历史raw/log只用于审计。代码、package或environment改变后，仅使用本轮新build、新launch和新output形成结论。

## 文本与提交检查

```bash
rg --files
rg '<symbol-or-path>'
git diff --check
git status --short
```

提交前检查完整diff、旧名称/路径残留、测试实际执行和共同署名；纯文档修改运行引用、路径、authority和格式检查，
不机械执行无关compiler构建。
