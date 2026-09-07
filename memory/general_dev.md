# Wafer Compiler通用开发方法

本文件只保存跨任务可复用的构建、测试和调试方法。IR语义、pipeline、package/runtime合同、硬件事实、
任务状态和历史case分别由编号设计、`docs/`、`tasks/progress.md`和`tasks/archive/`拥有。

## 开始与收尾

- 开工先运行`git status`，识别共享worktree中的已有改动；不回滚、不覆盖无关工作。
- 依次读`AGENTS.md`、`tasks/progress.md`、当前编号设计和实施计划，再读定义、构造位置、直接consumer和测试。
- 非小修先确认pipeline contract和本项覆盖矩阵；缺少输入、输出、consumer或完成门禁时先补设计。
- 修改按一个IR或文件格式边界推进。Late stage暴露上游缺口时回到真实producer修复，不在consumer临时repair。
- 收尾使用本轮生成的构建/测试结果，检查完整diff和工作树，只更新受影响文档并提交相关文件。

设计事实的阅读入口：架构与pipeline看01；frontend/data看02；structured/physical transformation看05--10；
Instr/memory/communication看11--13；target/package/model看14--17；源码与MLIR工程看18--20；硬件行为看`docs/`。

## Canonical build

- 唯一主工程binary dir是`build/`，由checked-in preset `default`拥有。普通任务不创建按任务、日期或配置命名的
  第二build。CMake输入、toolchain或managed dependency改变时重新configure同一目录。
- 修改期间可先构建受影响target；形成完成结论或提交代码前，在同一build执行一次无target的完整增量构建。
  Ninja只重建失效节点，不通过删除build掩盖dependency或registration问题。
- Host build、unit、CTest、lit、catalog和no-card默认使用`nproc`并行；只有已证内存、锁或共享写目录限制时降并发。
- Test的`UNSUPPORTED`、skip、未注册、XPASS、UNRESOLVED和TIMEOUT都不算通过。报告结果时同时确认关键case实际执行。
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

- IR变换正例使用rank至少3且主要迭代维不小于1024的static shape；partition、tiling或loop成对覆盖1024与
  1025/1031，并实际经过multi-Tile、multi-block/wave、remainder和tail。
- Tiny shape只用于有界oracle、最小verifier负例、scalar/zero-rank或单点故障定位，并注明原因；同一机制仍需真实规模case。
- 每项矩阵写清输入等价类、结构分支、typed failure、exact输出和direct downstream witness。正例不能只断言pass成功。
- 对coverage、owner、demand、merge、tail、copy、movement或completion的承诺使用精确计数/关系断言；不要把shape选取
  写入legality或策略。

## Pinned MLIR与API定位

- MLIR行为先查官方文档，具体API以configured build实际使用的pinned include和TableGen路径为准。旁置源码树含有某header
  不能证明当前构建可用。
- 修改已有对象前阅读definition、constructor、direct user、verifier、canonicalization和lowering；用`rg`沿完整调用链查找。
- 新op/type/attr前检查standard dialect、trait、interface、canonicalization和conversion能否表达。新增pass依赖dialect时显式声明。
- Crash/assert先用verifier-valid input复现并定位哪个pass首次产生invalid IR；不要在下游增加兜底verifier或名称恢复。
- Rewrite在第一次mutation前完成所有可失败preflight；同一clone transaction使用`IRMapping`。分析结果在IR mutation后默认失效。

## Source、CMake与命名

- Public header只暴露稳定typed API；query-local recipe、staging builder和failure bookkeeping留在`lib/`内部header。
- CMake显式列source和library dependency。新增、删除或迁移能力时同步header/source/CMake/test，fresh configure检查漏列。
- Test目录镜像被测component；fixture不是第二套schema、parser、ABI或产品pipeline。
- 命名前先用一句话说明对象语义，再核对typed domain和真实对照。Namespace、parent op或强类型已消歧时不重复范围词。
- 反引号路径、Markdown链接和current symbol改名后用`rg`检查残留；多义词逐项分类，不做无语义的全局替换。
- Python子进程和测试使用`-B`或`PYTHONDONTWRITEBYTECODE=1`；源码树不得留下`__pycache__`、`.pyc`或`.pyo`。

## Current-IR变换定位

- 先画出`input current IR -> typed choice -> actual transformation -> verifier -> fresh analysis -> consumer`。
- Analysis只读current IR和显式target config。Choice只保存transformation参数；operation、SSA、buffer、alias、movement、
  lifetime、event和order只有物化后才是事实。
- Search candidate只复制最近的`IsolatedFromAbove` owner。失败/loser整体销毁；Accepted owner直接向下游移动，不能按plan重建。
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
- 无卡阶段生成package、host expected和runner；真实设备只在明确任务中单进程逐case运行。Timeout或设备异常后停止批次，
  不自动retry、reset或power cycle。
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
