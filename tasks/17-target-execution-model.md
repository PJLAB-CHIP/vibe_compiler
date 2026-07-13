# Wafer Target Execution Model（CModel）

状态：2026-07-13初步设计。本文固定target execution model的证据边界、推荐实现分层和板端校准计划；
当前尚无production实现，任务状态看`tasks/progress.md`。本文不复制总体架构、instruction schedule或完整register/
packet硬件事实表；第3节只给出用于界定模型claim的non-normative摘要。compiler、target、runtime和verification的
既有合同分别仍由`tasks/01`、`tasks/14`、`tasks/15`和`tasks/16`拥有。

本文把日常所称的CModel限定为**目标相关执行模型**：它位于accepted executable之后，以当前target ABI、
packet、address space、engine和completion事实执行程序，并与独立reference/CPU结果比较。它不是Q19
ReferenceExecutor的别名，也不因采用SystemC就自动获得packet、timing或cycle accuracy。

## 1. 目标和非目标

目标：

- 让同一份accepted rank program在RISC-V device link之外多一个目标相关consumer，尽早暴露instruction-to-target
  lowering、typed CRT call、地址、descriptor、engine和Direct DTE错误；
- 建立plain C++功能核，使target-call、packet/MMIO、未来exact ELF/ISS和可选SystemC/TLM adapter消费同一份
  target command语义；
- 保持Q19 ReferenceExecutor和CPU oracle独立，以differential定位compiler lowering、模型实现和硬件行为差异；
- 对已由静态证据支持的功能/事务行为显式建模，对未知numeric、queue、bank和timing行为在执行前fail closed或
  保持参数化；
- 用真实board microbench、PMU和source-backed package逐级校准模型，并保留原始证据、环境身份和held-out验证结果。

非目标：

- 不新增CModel dialect、instruction sidecar、packet list、shadow schedule或第二套package manifest；
- 不从op、buffer、symbol、文件名或trace文本恢复rank、resource、binding或transport语义；
- 不复用Q19的interpreter、numeric kernel或Direct DTE scheduler作为target model实现；
- 不把host target-call execution称为exact CRT/packet/ELF execution；
- 不把model completion称为board completion，也不在板端证据前声明hardware-bit-exact、performance-accurate或
  cycle-accurate；
- 不让SystemC成为基础compiler/runtime的强制依赖，也不把SystemC对象写入compiler IR、ExecutableBundle或package。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  近期入口消费Q16 ExecutableBundle经过tasks/14 target ABI preparation和full conversion形成的all-and-only、
  owner-backed fully legal target LLVM modules，以及与每rank entry精确双射的typed ABI slots。该target-LLVM
  bundle不可序列化，不进入package。exact-module扩展另行消费Q18 VerifiedPackageManifest和其all-and-only
  Q17 RISC-V ELF modules。Q19结果和CPU expected只作独立比较，不作为target model执行输入。
- Current stage responsibility:
  在任何执行副作用前完成model capability preflight；通过host target ABI、packet/MMIO或未来ISS frontend把
  exact lower-level调用投影为typed target commands；执行rank/tile address spaces、CT/NE/RDMA/WDMA/TDMA、
  local completion和Direct DTE/FSM event。模型不重做sharding、candidate、layout、memory或transport planning。
- Output artifact / IR:
  invocation-local、owner-backed target model execution result，包含all-and-only rank terminal status、typed
  output tensors、model/profile provenance和可选typed diagnostic trace。结果不进入compiler IR、
  ExecutableBundle、TargetArtifactBundle或PackageManifest，也不作为后续planning输入。
- Downstream consumer:
  tasks/16 differential/CI、board bring-up、exact-module闭合后的同package model/board correlation，以及板端证据
  闭合后的独立timing calibration；target model结果不是compiler transformation输入。
- User-level driver / named pipeline:
  首个target-call gate由wafer-compile在Q17/Q18原子发布后、同一invocation仍持有target LLVM bundle时进入显式
  target-model execution mode；wafer-opt/pass chain只补局部测试。只有exact package/ELF执行闭合后，wafer-run才通过typed
  RuntimeProvider选择target model并消费verified package，不能用host-only模式冒充该入口。
- Explicit non-goals:
  不复制instruction schedule，不改变manifest语义，不用Q19执行结果驱动target model，不把untimed或
  loosely-timed结果升级为board/cycle证据，不用代表rank、手写LLVM、手写packet或单个kernel替代真实纵向链。
- Completion gate:
  target-call/transaction基线必须执行同一wafer-compile产生的rank-count=1/16 source-backed program，覆盖
  all-and-only fully legal target LLVM ranks、完整typed ABI、SPM/DDR、当前supported engine和Direct DTE状态，
  与独立Q19/CPU oracle按显式dtype tolerance比较完整输出；unsupported symbol/profile、地址/descriptor错误、
  deadlock和任一rank late failure均在model result publication前fail closed且无partial result。model mismatch不回滚
  已验证Q17/Q18 artifacts。exact package/ELF、
  board correlation和timing accuracy分别保持更高独立gate。
```

## 3. 当前事实基线

### 3.1 已有compiler和runtime边界

- Q16 `ExecutableBundle`已经拥有all-and-only static rank modules、typed program bindings、accepted memory/
  transport和terminal completion；target model不得重新选择candidate或重算跨rankbinding。
- tasks/14已经在private `PreparedTargetRank`中完成entry output/workspace/status ABI preparation，随后把instruction
  结构保持地lower为fixed `void(i64...)` target LLVM CRT calls；RISC-V device link在该lowering之后发生。
- 当前Wafer CRT header、lowering、source和checker形成109个production symbol的闭合surface。symbol存在只证明
  ABI closure，不证明packet、numeric或hardware completion。
- Q17正式交付物是all-and-only RISC-V ELF `TargetArtifactBundle`；Q18 package只包含typed resource/slot/module/
  entry/completion/transport requirement，不包含instruction schedule。
- Q18当前只实现pure no-card `RuntimeSessionPlan`。真实`RuntimeProvider`生命周期和board执行尚未实现。
- Q19直接消费accepted instruction/memory facts；其reference-only numeric和deterministic DTE policies不得成为
  target model的实现事实源。

近期实现应把tasks/14中已完成ABI preparation和full target conversion的结果提升为owner-backed、不可序列化的
**target LLVM bundle**。它是实际fully legal LLVM IR及typed slots的all-rank集合，同时被现有RISC-V link和host
target-call model消费；它不是新IR层、packet artifact或package成员。具体C++类名和文件布局只作实现索引，不属于
artifact合同。

该bundle必须显式拥有canonical all-rank domain、每rank logical rank/entry/module、`ExecutionConfig`、ordered typed
ABI slots、target identity/revision、target/kernel ABI facts和覆盖全部module的context/owner lifetime。所有rank完成ABI
preparation、full conversion和readback后才能原子构造bundle并交给任一consumer；不能让device link和host model分别
从默认值恢复target facts，也不能让passing ranks先行执行。

### 3.2 已确认的硬件功能和事务事实

当前静态证据足以支撑以下受支持子集：

| 范围 | 已确认事实 | 初版可声明 | 仍不可声明 |
| --- | --- | --- | --- |
| topology/memory | 单卡4×4、16 tile；每tile 3 MiB SPM；普通SPM、保留64 KiB、DDR和Kcore alias范围已知 | rank/tile隔离、bounds、reservation和byte-level memory | 所有SKU容量、bad-tile/PG行为均可由静态值硬编码 |
| physical layout | compact、Cx/NCx block/tail/fold和256B footprint规则已恢复 | logical coordinate到physical byte offset及越界检查 | 任意dtype/op组合均可执行 |
| packet/ABI | CT、NE、RDMA、WDMA、TDMA packet字段、worker window、trigger和range/end字段已恢复 | typed decode、字段宽度、descriptor/range conformance | vendor CRT实际packet与host重实现天然一致 |
| movement | contiguous/strided RDMA/WDMA、基础TDMA/gather-scatter/memset的byte语义较完整 | 受支持descriptor的功能执行 | 所有alignment/stride组合的性能公式 |
| local queues | 每tile三个worker window；五类NCC queue；parallel模式存在range/busytable依赖 | 显式issue、local drain和保守event ordering | queue深度、多发射、精确仲裁和worker物理独立性 |
| Direct DTE | unicast register、receiver-ready、issue、busy/done/error、wait/release和部分resource限制已知 | 当前accepted single-destination profile的功能/event模型 | broadcast/gather等完整模式及精确contention |
| PMU | DTE/SPM/NCC base、record种类和counter shape已知 | 保留raw counter和建立校准实验 | counter单位、wrap、workload correlation已证明 |

底层事实的长期owner仍是`docs/wafer-hardware-instruction-set-and-programming-model.md`、
`docs/wafer-register-level-instruction-spec.md`和`docs/tx8-deps-reverse-engineering/`。模型实现必须引用共享
geometry/ABI定义或生成的typed定义，不能把本文表格复制成第二套常量。

### 3.3 缺失事实

以下缺口要求初版fail closed、保守event关系或明确的model-only profile：

- per-op/dtype/shape latency、throughput、pipeline depth和clock-domain关系；
- queue depth、同queue多发射、cross-queue arbitration和三个worker的真实共享资源拓扑；
- exact SPM address-to-bank函数、RAM_ACC replay成本、LSU/NoC/DDR arbitration和route/hop contention；
- DTE setup、packet、alignment、route和并发传输周期函数；
- PMU单位、enable/clear边界、wrap/saturation和host/device时间相关性；
- 浮点NaN/Inf/subnormal/overflow、部分fused optional field、zero-point公式、stochastic seed/state/推进合同；
- provider/firmware实际初始化的`serial_mode`和错误恢复行为。

历史allocator使用的64 KiB coloring粒度目前只是heuristic，不是SPM bank或hard ABI；它与Kcore占用SPM末尾
64 KiB这一真实reservation是两个不同事实。公开峰值只可作上界检查，不得直接成为模型延迟或带宽参数。

### 3.4 Vendor simulator seam

依赖中声明了`initTsmOpPointer_cmodel`、`instr_tick_cc`和cycle-mode接口，simulation interface的host分支却只
建立INTERFACE target；当前仓库和附带archive没有这些host定义，附带instruction/common-util archive为RISC-V
object。因此当前不能把vendor CModel当作可链接依赖。

进入实现前应向vendor确认：host simulator library/source、支持的target revision、CRT/packet入口、SystemC版本、
license、numeric profile、threading/time contract和可重放要求。若取得可信实现，可作为另一个frontend/backend
接入本合同，但不得绕过capability、differential和board correlation gate。

## 4. 模型架构

### 4.1 Target frontend分层

模型至少保留三个互不冒充的入口：

1. **Target-call frontend**：host LLVM JIT执行同一fully legal target LLVM module，并为当前
   `wafer_tx81_*` fixed ABI注册host实现。它验证target ABI preparation、control flow、call graph、typed arguments、
   address formation和CRT call sequence；因为没有执行RISC-V CRT/archive，所以不证明实际vendor packet。
2. **Packet/MMIO frontend**：消费exact packet words或register transaction，统一decode为typed target command。
   它验证packet字段、worker/engine、address/range/trigger和completion relation；positive必须来自exact Q17 ELF经
   ISS执行真实CRT/archive产生的register trace、board capture，或vendor提供且版本可审计的host packet builder。
   项目自行重写的builder在与这些golden逐字段相关前只算model-internal/negative coverage。
3. **Exact-module frontend**：从Q18 verified package加载并执行Q17 RISC-V ELF，通过ISS、loader ABI、MMIO/custom
   instruction、Direct DTE和completion adapter进入同一模型核。只有此入口完成后，才可称package-facing target
   model provider。

target-call和packet gate可以独立发现不同错误；二者通过不等于exact-module通过。任何测试和diagnostic都必须记录
实际入口与capability profile。

当前fixed CRT ABI没有额外model-context参数。host frontend必须把每次entry call显式绑定到对应invocation/rank context，
可用per-JIT symbol trampoline或有严格scope/lifetime的thread-local call frame；不得使用跨invocation共享的process-global
mutable singleton，也不能从thread id、调用顺序或symbol名字推断rank。具体binding机制是实现选择，但必须覆盖并发rank、
异常退出和nested call后的恢复测试。

target LLVM中的i64地址参数仍是Wafer resource base、SPM/DDR offset或provider-managed model address，不是可任意解引用的
host pointer。host frontend只能通过typed ABI slot和checked address registry解析；未知base、overflow、跨resource访问
或把SPM alias当host虚拟地址都必须在memory effect前拒绝。

### 4.2 Plain C++功能核

功能核按稳定硬件责任拆分：

- model context：target revision、logical/physical rank/tile mapping和invocation-local状态；
- memory system：每rank/tile SPM、reserved region、card DDR/resource slots、checked address translation；
- typed command decoder：统一承载engine kind、worker、operands、descriptor、geometry和completion，不序列化；
- CT/NE/RDMA/WDMA/TDMA engines：只执行explicit supported profile，未知optional/numeric组合fail closed；
- deterministic event engine：queue issue、resource occupation、local drain和typed no-progress诊断；
- Direct DTE fabric/FSM：rank-local endpoint/status、payload snapshot、receiver-ready、send/recv/wait/release；
- result/diagnostic collector：typed terminal status、完整output和可选trace。

typed target command只是一次ABI call或一次packet decode产生的瞬时transaction；它不预构造整程序command vector，
不进入compiler artifact或package，不能成为长期shadow program。event engine读取transaction执行时状态，不复制
compiler completion DAG或planner trace。exact ELF路径必须经过ISS产生raw packet/register transaction再进入decoder，
不能由ISS hook直接合成高层command绕过packet证据。

### 4.3 数值实现独立性

Q19和target model可以共享稳定dtype enum、physical geometry、packet field definition和ABI常量，但不能共享完整
compute kernel、rounding policy或DTE scheduler。否则differential会让同源bug同时通过。

target model对numeric profile使用以下规则：

- 有硬件/ABI直接证据并有区分性测试的行为进入supported profile；
- 只有数学名称而无edge behavior证据时，模型结果标记为model semantics，不能称hardware-bit-exact；
- zero-point等没有唯一公式的组合在执行副作用前拒绝；
- stochastic行为在取得seed/state合同前只可做独立统计correlation，不得用Q19的SplitMix64 policy冒充硬件；
- 每个profile都必须列出dtype、shape/descriptor、optional fields、completion和比较口径，不能从enum存在推导支持。

### 4.4 SystemC/TLM边界

SystemC是可选的调度/TLM实现技术，不是target model语义owner。初版保留plain C++功能核和确定性component tests；
需要多engine并发、ISS virtual platform、temporal decoupling或timing refinement时，再以adapter把memory/engine/DTE
暴露为SystemC modules和TLM transactions。

约束：

- 基础compiler、reference executor、manifest parser和no-card runtime不依赖SystemC；
- SystemC build必须是显式可选feature，版本/license由依赖管理拥有，不能从host偶然安装状态决定；
- untimed、loosely-timed和approximately-timed profile必须在结果provenance中可区分；
- SystemC scheduling order不能充当message identity、resource binding或compiler completion语义；
- approximately-timed只有板端held-out correlation通过后才能发布；cycle-accurate必须另有RTL/per-cycle trace、
  vendor cycle model或完整微架构合同。

## 5. Runtime和exact ELF边界

### 5.1 近期target-call执行

近期入口是同一次`wafer-compile` invocation中的下游verification consumer：Q17/Q18先按各自合同原子发布verified
target/package artifacts，driver再用仍由invocation持有的owner-backed target LLVM bundle执行target-call model。
model mismatch或执行失败可让driver返回非零并保留diagnostic，但已验证package保持可审计，不回滚、不改写，也不让
Q22成为Q17/Q18 correctness前置。是否启用该gate属于用户显式请求或configured CI policy；不能变成wafer-opt
stop-stage或手拼pass。

该入口的完成证明止于target-call functional/transaction evidence。它不读取已发布package，不执行RISC-V CRT和
vendor archive，也不能使用`wafer-run`的provider语义。

### 5.2 Exact package provider

Q17 module固定为RISC-V64 ELF，并链接repo CRT、RISC-V `libinstr_tx81.a`/`libcommon_util.a`，同时保留SPM mapping、
Direct DTE、logging、allocator等loader symbols。host不能直接`dlopen`这些module。完整provider至少需要：

- RV64 CPU/ABI execution和module loader；
- 当前loader ABI及entry/slot/resource binding；
- SPM alias、MMIO/register和必要custom instruction；
- CT/NE/RDMA/WDMA/TDMA packet发射；
- Direct DTE/FSM、host watchdog、status和terminal completion；
- allocation/import、H2D、load/resolve、all-rank submit、wait、D2H和逆序cleanup。

因此单独接入通用RISC-V ISS并不足够。若未来exact-module frontend闭合，target model通过tasks/15定义的typed
`RuntimeProvider`消费原样Q18 package；manifest schema、resource/slot、transport requirement和module digest不增加
model专用分支。provider输出invocation-local result/diagnostic，不能写回package或生成partial successful result。

当前`RuntimeSessionPlan`按一个entry/rank做no-card preflight；Direct DTE exact execution不能把16个entry plan简单顺序
循环。provider落地前必须由tasks/15补充owner-backed all-rank invocation/session，把manifest中all-and-only entries、
typed invocation bindings、transport capability、共同submit/progress/status和atomic cleanup组成一个执行域。该session只
引用package已提交事实和provider-owned runtime handles，不复制target module内的message/packet schedule。

## 6. Event、completion和failure合同

模型至少区分：

- target call/packet已提交；
- 当前tile/worker NCC queue local drain；
- Direct DTE receiver ready、busy、done/error和payload visible；
- multi-rank no-progress/deadlock；
- provider terminal status和copyback eligibility。

`TsmExecute` success不表示完成，`TsmWaitfinish[_bywork]`只证明对应local NCC domain，不能替代DTE wait或multi-tile
arrival。保守模型可以延后event完成，但不能合并没有证据的completion domain。

所有入口执行前完成all-rank capability和resource preflight。unknown symbol/packet/profile、地址越界、descriptor错误、
missing endpoint、unsupported numeric或exact-module环境不匹配，必须在input import或model state mutation前整体失败。
运行中错误必须：

- 停止dependent event和copyback；
- 标记受影响invocation/rank，不产生可误认为成功的partial result；
- 逆序释放已获取资源；
- 保留稳定stage、rank、entry、command/event类别和model/profile provenance；
- no-progress使用确定性state snapshot诊断，不依赖wall-clock或thread调度决定语义。

## 7. 模型精度和发布标签

| Profile | 最低事实和gate | 允许声明 | 禁止声明 |
| --- | --- | --- | --- |
| target-call functional | same fully legal target LLVM、typed ABI、supported host CRT calls、完整输出differential | target lowering/ABI functional | actual vendor packet、package ELF、board |
| packet functional | exact-ELF register trace、board capture或versioned vendor builder的decode/range/engine/numeric/completion conformance | supported packet functional | 完整loader/provider lifecycle、timing、board |
| untimed event | completion-domain、queue ordering和DTE lifecycle经component及板端相关性验证 | target event relation | latency、throughput、cycle |
| exact-module functional | verified package、all-and-only RISC-V ELF、ISS/loader/MMIO/provider lifecycle | package target-model execution | real board、hardware timing |
| loosely-timed | PMU measurement basis、single-engine和single-flow held-out校准 | 指定profile的粗粒度timing | contention/cycle accuracy |
| approximately-timed | bank/worker/queue/DDR/NoC/DTE contention经held-out验证 | 指定环境/profile的事务时序估计 | RTL/cycle等价、跨revision泛化 |
| cycle-accurate | RTL/per-cycle trace、vendor cycle model或完整微架构合同及correlation | 明确revision的cycle行为 | 无证据的其它SKU/revision |

profile是execution result provenance，不是compiler legality或package semantic branch。高层profile失败不能反向改变
accepted instruction支持范围；如果硬件可表达但model未覆盖，应扩target model capability或保持该profile拒绝。

## 8. Verification Gates

### 8.1 Capability和target-call gate

- 自动从当前lowering/CRT typed surface得到all-and-only host implementation coverage，不复制109项字符串表；
- 同一`ExecutableBundle`分别进入Q19和target-call model，整数exact，浮点按显式dtype/op tolerance比较完整output；
- 真实rank-count=1 linear/MLP是首个纵向gate，手写LLVM/MLIR只补negative；
- unknown symbol、wrong ABI slot、地址/descriptor/narrowing和unsupported numeric在input/model allocation前拒绝；
- target LLVM bundle构造的任一rank late failure不形成bundle或partial model result；Q17/Q18各自仍按原合同失败。
  bundle构造和package均成功后，model mismatch只让verification返回非零并保留已验证package供审计。

### 8.2 Packet和memory gate

- CT/NE/RDMA/WDMA/TDMA的typed args、packet fields、worker、trigger、range/end和return/status逐family覆盖；
- SPM/DDR bounds、reserved region、Cx/NCx、bitpacked i1、subview/strided descriptor用独立slow oracle或board bytes验证；
- target-call positive来自compiler-generated target LLVM；packet positive只接受exact-ELF register trace、board capture或
  versioned vendor builder。项目host builder在逐字段golden correlation前只补model-internal/negative coverage；
- host target-call与packet frontend进入同一command core时，比较command/observable memory effect，不用共享builder
  自证正确。

### 8.3 Multi-rank和provider gate

- 16个rank拥有独立SPM/context/status，all-and-only accepted rank都执行；
- Direct DTE覆盖receiver-ready、payload snapshot、send/recv/wait/release、duplicate/missing/mismatch和deadlock；
- target model不能读取Q19 logical message schedule或用reference coordinator实现transport；
- exact provider按allocate/import、H2D、load/resolve、submit、wait/status、D2H、cleanup逐阶段注入失败；
- wait/status失败后禁止copyback，任一rank失败无partial successful result。

### 8.4 Source-backed和correlation gate

- 原样重放Q20 rank-count=1/16 linear/MLP和Q21 16-rank tiny Llama产物，不建立model专用fixture或计划；
- 检查全部rank/module/output、typed status和可观察event/packet数量，不用rank 0、shape或digest代替执行；
- target model与Q19/CPU差异先按target call、packet、memory、numeric、event分类；
- configured board可用后，exact-module frontend闭合时用同一package比较model/board完整output、status、trace和PMU；
  在此之前target-call model只能和board做同source/accepted program的cross-frontend correlation，不能称same-package执行；
- tests为unsupported/skipped时对应profile gate保持未完成。

Q20/Q21的具体模型、shape、dtype和tolerance只是source-backed gate参数，不定义target model协议。通用合同来自fully
legal target modules、typed ABI/capability、packet和address/completion关系；其它workload按同一关系执行，超出profile
时结构化拒绝，不能把这些case的调用序列固化进模型。

## 9. 板端证据与校准计划

### 9.1 测量纪律

每次板端运行必须记录：

- device SKU/revision、DDR容量、good-tile bitmap、logical/physical tile map；
- firmware、driver、runtime、compiler revision、package和module digest；
- 板卡独占状态和后台负载，以及可查询的clock/power/temperature；无法查询的变量标记为uncontrolled，不假定固定频率；
- input seed、worker/tile、SPM/DDR地址、compiler/CRT-derived command provenance和issue order；只有经过验证的capture
  才记录为actual packet bytes，无法捕获时显式标记而不补造；
- warmup、正式重复、trial随机顺序和全部raw samples；
- host wall time、provider phase、device user timer和raw PMU delta分别记录，在单位确认前不互换；
- 每个sample的完整功能输出；数据错误的性能sample作废；
- calibration和held-out validation shape/descriptor/workload严格分离。

常规calibration不发raw非法MMIO/address，也不运行receiver-not-ready或未知non-unicast。queue overflow等可能hang的
实验只进入独立destructive qualification，并要求设备独占、watchdog、timeout、reset和cleanup先验证。任一timeout/
reset后必须先做health check；未恢复设备的后续样本无效。所有失败样本保留并标记invalid，不能删除。若SKU/revision、
tile map或good-tile bitmap无法取得，只能发布受限local functional evidence，不能发布跨tile或可泛化timing profile。
验收阈值例如重复CV、held-out median/p95误差必须在看结果前固定，并明确属于项目阈值而非硬件事实。

### 9.2 实验矩阵

| 实验 | 待回答问题 | 主要控制变量和观测量 | 证据输出 | 升级gate |
| --- | --- | --- | --- | --- |
| provider correctness | 真实allocation/copy/load/launch/transport/completion/copyback/cleanup是否闭合 | 同一Q21 package；phase/status/timeout/output/cleanup | package/module digest、phase log、actual output | Q6.B完整输出和重复invocation先通过，性能sample才有效 |
| measurement basis | PMU enable/clear隔离、delta、width/unit、wrap和读取是否可信 | empty/read overhead、多个window长度、固定worker/block、requested issue与completed count、独立timer；raw start/end | raw PMU和sample序列 | clear隔离、单调性、count与timer ratio稳定；ambiguous wrap样本不拟合，单位未闭合只保留raw tick |
| completion domains | issue、IB counter、task_done、NCC wait、DTE wait各自证明什么 | 单长命令、分worker、NCC/DTE单独和组合；CSR/status/data-visible | typed event trace、CSR/PMU snapshot | 只把重复happens-before关系写入event model |
| single-engine curves | 各engine启动成本、吞吐和shape/dtype关系 | 单tile/worker/engine，size/shape/dtype sweep；exec/blocking/output | per-engine raw samples、fitted curve | 功能正确且held-out误差达预设目标才进入loosely-timed |
| DMA descriptors | contiguous/strided byte/iteration和segment成本 | 固定总bytes，扫inner/stride/iteration/alignment；输出和LSU PMU | descriptor、访问区间、output、PMU | held-out descriptor可预测后才加入timing公式 |
| queue overlap | 五queue并发、dependency和共享resource关系 | engine pair的disjoint/RAW/WAR/WAW、AB/BA；serial mode逐worker写入/readback并在实验后恢复 | relation matrix、per-engine exec/blocking和output | 只记录重复稳定的overlap/dependency class，不由此猜仲裁算法 |
| queue capacity | backpressure和该profile下的安全outstanding范围 | 同/跨queue递增N、IB counter/completed count、watchdog和recovery | bounded sweep、timeout/exception/health log | 只发布稳定safe outstanding；不得称真实queue depth，destructive失败后health check必过 |
| SPM bank | address-bit/offset period与conflict的真实关系 | 128B/256B及更大base delta扫描，64KiB仅作heuristic候选；无range overlap、多tile/worker | address-pair heatmap和SPM/engine raw counters | 稳定周期才形成versioned heuristic；否则只建模显式overlap，64KiB不升级为bank事实 |
| multi-worker | worker 0/1/2共享哪些engine/LSU/SPM资源 | 单worker baseline与两/三worker组合；per-worker PMU/wait状态 | worker resource matrix | 重复证明后才加入resource topology |
| Direct DTE | accepted unicast binding的receiver/FSM/status/data-visible、setup和byte语义 | known-good两tileendpoint、receiver-ready-first、provider-managed channel/FSM，扫size/alignment；两端状态和DTE PMU | binding/register/PMU/source/dest bytes | completion与数据可见稳定后才加入event/byte model；raw non-unicast不进入常规矩阵 |
| fabric contention | 多tile DDR/NoC/DTE aggregate和endpoint/topology placement差异 | 1/2/4/16 tile、地址和endpoint placement组合；per-tile blocking和bytes/time | topology-aware bandwidth surface | held-out组合达预设误差才归纳contention class；不反推出不可观测routing算法 |
| numeric correlation | rounding、zero-point、accumulation/fusion的真实语义 | tie/NaN/Inf/overflow、能排除候选公式的区分向量、独立held-out和重复随机payload；raw bits/status | 按op/dtype/optional-field/revision版本化的vector corpus和comparison | deterministic需bit-stable且唯一排除候选公式并通过held-out；stochastic无seed合同只建统计profile |
| vertical correlation | component参数能否解释真实program | Q20/Q21完整output/event/PMU/provider phases；exact frontend闭合前只作same-source cross-frontend | model/board differential summary | exact frontend后才称same-package；held-out真实workload达profile目标才发布对应timing标签 |

### 9.3 Evidence artifact

每次run形成独立、不可变的verification evidence directory，至少包含环境身份、package/module digest、invocation、
packet/event trace、raw PMU、sample、expected/actual和只引用raw evidence digest的summary。它不是package sidecar，
不被compiler legality、lowering或runtime launch消费。

板端校准参数不能作为散落magic number写入功能核。若后续timing model需要消费参数，应先定义按device/firmware/
runtime profile版本化、引用raw evidence digest且可验证的typed calibration profile；它只影响model timing，不能改变
IR legality、candidate acceptance或package语义。若板端结果与静态资料冲突，先回到对应hardware/ABI owner记录并
收敛冲突，不能静默调整模型常量。numeric/functional发现只有经hardware/ABI owner和conformance tests闭合后才能
扩model capability，不能自动扩大instruction legality。Q9若消费测量结果，应另行生成经评审的derived cost profile，
并且只排序已经合法的candidate。

## 10. 分阶段交付

### 10.1 Capability和依赖收敛

- 取得或确认不存在vendor host simulator/SystemC SDK；
- 固定首批target call、dtype/layout、engine、Direct DTE和completion capability matrix；
- 把tasks/14 private prepared target LLVM提升为owner-backed all-rank内部artifact；
- 固定SystemC为optional dependency，未取得dependency时plain C++路径仍可构建和测试。

完成：文档、typed capability和failure分类收敛；未实现symbol/profile在任何mutation前可被完整枚举拒绝。

### 10.2 Target-call functional slice

- host JIT执行same fully legal target LLVM；
- 建立rank-local virtual SPM/DDR、typed slots和supported host CRT calls；
- 先覆盖source-backed linear/MLP需要的RDMA、WDMA、gather/scatter、fill、GEMM、f32 elementwise和local fence；
- 与独立Q19/CPU比较完整rank-count=1输出。

完成：真实compiler chain通过target-call gate，unknown symbol/ABI/address/numeric preflight和atomic failure通过；明确
标记未验证actual packet/ELF。

### 10.3 Packet、event和16-rank transport

- 增加current CT/NE/RDMA/WDMA/TDMA packet decode/conformance；
- 建立三个worker、五queue、保守resource event、local drain和Direct DTE/FSM；
- 扩展当前production compute/movement surface和16-rank source-backed case；
- 需要时增加optional SystemC/TLM adapter，但plain C++ component gate保持可运行。

完成：rank-count=1/16完整输出、packet/memory/event negative、DTE no-progress和all-rank atomic result通过；仍不称
exact package/ELF或timing model。

### 10.4 Exact package execution

- 在取得vendor simulator或完成RV64 ISS、loader ABI、MMIO/custom instruction和provider lifecycle后接入
  Q18 verified package；
- 原样执行all-and-only Q17 modules，不发布host专用instruction list或修改manifest；
- 通过wafer-run typed provider入口执行Q20/Q21 package和阶段性failure injection。

完成：同一package在model provider中完整allocate到cleanup并产生可信status/output；否则该能力保持更高待解锁gate。

### 10.5 Board correlation和timing refinement

- 先完成board provider correctness和PMU measurement basis；
- 按single-engine、queue/SPM/DTE/fabric/numeric矩阵采样；
- 用独立held-out shapes/descriptors/workloads验证loosely/approximately-timed profile；
- cost calibration如需消费结果，另由tasks/06/16和Q9建立typed consumer，不反向污染correctness。

完成：只发布实际通过的profile标签、适用device/firmware/runtime identity、误差分布和未覆盖范围。没有RTL或vendor
cycle证据时cycle-accurate保持非目标。

## 11. 待讨论问题

以下问题不阻塞target-call functional设计，但会改变更高层实现成本，因此保留为显式待讨论问题：

1. **Vendor simulator交付**：现有header只有接口痕迹。需要确认是否能取得host library/source、版本和license；若可得，
   exact packet/ISS路径可能显著缩短，但仍需独立correlation。
2. **Exact package execution是否为近期产品要求**：如果主要目标是compiler lowering回归，target-call/packet gate已提供
   高价值；如果用户需要独立`wafer-run`虚拟平台，RV64/loader/provider必须提升为主线。
3. **Packet事实源**：需要确认可否复用vendor可审计builder或获得register trace。host CRT直接构造typed command只证明
   ABI，不证明packet；在事实源明确前两种gate保持分开。
4. **SystemC依赖策略**：需要确认CI平台、可接受版本、license和是否需要ISS co-simulation。该选择只影响event/timing
   backend，不改变artifact或功能核合同。
5. **板端环境和验收阈值**：需要固定可用SKU/revision、firmware/runtime组合、reset/watchdog能力和预先声明的LT/AT
   held-out误差目标；未固定前不发布timing accuracy。

## 12. 文档和实现归属

- tasks/14拥有target ABI preparation、fully legal target LLVM、CRT和RISC-V module publication；target LLVM bundle
  落地时应在同批同步其producer合同。
- tasks/15拥有exact target model `RuntimeProvider`生命周期和package消费；host target-call模式不是该provider。
- tasks/16拥有target-call、packet/event、exact-module、board和timing的证据分层及CI gate。
- 本文拥有target execution model内部边界、capability、SystemC选择、板端校准计划和各层不得冒充的声明。
- tasks/09/11/13仍分别拥有memory legality、instruction/packet legality和Direct DTE/completion；model不能改写这些
  compiler合同。

进入代码实现前应建立独立实施计划；本初步设计本身不表示target model、exact provider、board或calibration已完成。
