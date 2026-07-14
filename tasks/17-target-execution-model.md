# Wafer Target Execution Model（CModel）

状态：2026-07-14方案收敛。本文固定以数值正确性为近期目标的untimed target execution model、SystemC/TLM边界、
实现分层和板端numeric correlation计划；timing calibration仅作为deferred extension。当前尚无production实现，任务状态看
`tasks/progress.md`。本文不复制总体架构、instruction schedule或完整register/
packet硬件事实表；第3节只给出用于界定模型claim的non-normative摘要。compiler、target、runtime和verification的
既有合同分别仍由`tasks/01`、`tasks/14`、`tasks/15`和`tasks/16`拥有。

本文把日常所称的CModel限定为**目标相关、非cycle-accurate的数值功能模型**：它位于accepted executable之后，以当前
target ABI、packet、address space、engine和completion事实执行程序，并与独立reference/CPU及后续board结果比较。它不是
Q19 ReferenceExecutor的别名，也不因采用SystemC就自动获得vendor packet、hardware numeric、timing或cycle accuracy。

## 1. 目标和非目标

目标：

- 让同一份accepted rank program在RISC-V device link之外多一个目标相关consumer，尽早暴露instruction-to-target
  lowering、typed CRT call、地址、descriptor、engine和Direct DTE错误；
- 以SystemC作为正式target model的模块、并发和event容器，以TLM承载必要的MMIO、DDR和fabric事务；近期只发布
  untimed/delta-cycle profile，不建立任意时间参数；
- 把算子数值、packet decode和checked memory effect保留为不依赖SystemC的plain C++ kernel，使其可做独立unit/
  differential test，并由SystemC modules调用；
- 让同一份repo-local CRT源码的host build经过CModel-compatible `TsmNew*`/operator/execute/MMIO seam、实际
  `Tsm*Instr`构造和`TsmExecute`进入模型，
  不以绕过CRT/packet的host shim代替正式CModel路径；
- 保持Q19 ReferenceExecutor和CPU oracle独立，以differential定位compiler lowering、模型实现和硬件行为差异；
- 对已由静态证据支持的功能/事务行为显式建模；未知numeric在执行前fail closed，未知queue/bank/timing不进入近期
  correctness claim；
- 用真实board single-op区分向量、重复随机held-out和source-backed workload校准numeric profile，并保留原始bytes/status、
  环境身份和evidence digest。

非目标：

- 不新增CModel dialect、instruction sidecar、packet list、shadow schedule或第二套package manifest；
- 不从op、buffer、symbol、文件名或trace文本恢复rank、resource、binding或transport语义；
- 不复用Q19的interpreter、numeric kernel或Direct DTE scheduler作为target model实现；
- 不把direct host shim称为CRT/packet执行，也不把Host-CRT/SystemC的project packet称为vendor-exact packet或RISC-V ELF执行；
- 不把model completion称为board completion，也不在板端证据前声明hardware-bit-exact、performance-accurate或
  cycle-accurate；
- 不把PMU、latency、throughput、LT/AT或cycle calibration作为Q22或Q22.C的近期完成前置；
- 不让SystemC成为基础compiler、reference executor、package parser或no-card runtime的强制依赖；target-model feature
  一旦启用则SystemC是正式模型依赖，且SystemC对象不写入compiler IR、ExecutableBundle或package。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  近期入口消费Q16 ExecutableBundle经过tasks/14 target ABI preparation和full conversion形成的all-and-only、
  owner-backed fully legal target LLVM modules，以及与每rank entry精确双射的typed ABI slots。该target-LLVM
  bundle不可序列化，不进入package。exact-module扩展另行消费Q18 VerifiedPackageManifest和其all-and-only
  Q17 RISC-V ELF modules。Q19结果和CPU expected只作独立比较，不作为target model执行输入。
- Current stage responsibility:
  在任何执行副作用前完成model capability preflight；以同一repo CRT的host build和CModel-compatible Tsm factory/
  operator/execute/MMIO seam形成
  packet/register transaction，交给SystemC tile/worker/engine/memory/fabric modules执行rank/tile address spaces、
  CT/NE/RDMA/WDMA/TDMA、local completion和Direct DTE/FSM event。direct target-call shim只作ABI smoke；未来ISS
  frontend执行exact ELF。模型不重做sharding、candidate、layout、memory或transport planning。
- Output artifact / IR:
  invocation-local、owner-backed target model execution result，包含all-and-only rank terminal status、typed
  output tensors、model/profile provenance和可选typed diagnostic trace。结果不进入compiler IR、
  ExecutableBundle、TargetArtifactBundle或PackageManifest，也不作为后续planning输入。
- Downstream consumer:
  tasks/16 differential/CI和Q22.C board numeric correlation；Q22.E exact-module与deferred Q22.P timing calibration是
  独立consumer。target model结果不是compiler transformation输入。
- User-level driver / named pipeline:
  首个正式CModel gate由wafer-compile在Q17/Q18原子发布后、同一invocation仍持有target LLVM bundle时进入显式
  target-model execution mode：host执行target LLVM并调用同源host CRT，CRT再向SystemC model发packet/event。
  wafer-opt/pass chain和direct shim只补局部测试。只有exact package/ELF执行闭合后，wafer-run才通过typed
  RuntimeProvider选择target model并消费verified package，不能用host-only模式冒充该入口。
- Explicit non-goals:
  不复制instruction schedule，不改变manifest语义，不用Q19执行结果驱动target model，不把untimed结果升级为
  board/timing/cycle证据，不用代表rank、手写LLVM、手写packet或单个kernel替代真实纵向链。
- Completion gate:
  SystemC host-CRT/transaction基线必须执行同一wafer-compile产生的rank-count=1/16 source-backed program，覆盖
  all-and-only fully legal target LLVM ranks、同源CRT、实际project packet构造、完整typed ABI、SPM/DDR、当前
  supported engine和Direct DTE状态，
  与独立Q19/CPU oracle按显式op/dtype model profile policy比较完整输出；unsupported symbol/profile、地址/descriptor错误、
  deadlock和任一rank late failure均在model result publication前fail closed且无partial result。model mismatch不回滚
  已验证Q17/Q18 artifacts。Q22只发布model-only untimed functional-numeric profile；board numeric、exact package/ELF和
  timing accuracy分别由Q22.C、Q22.E和deferred Q22.P保持为更高独立gate。独立packet/MMIO事实源缺失只限制
  vendor-exact packet claim，不阻塞Q22或Q22.C。
```

## 3. 当前事实基线

### 3.1 已有compiler和runtime边界

- Q16 `ExecutableBundle`已经拥有all-and-only static rank modules、typed program bindings、accepted memory/
  transport和terminal completion；target model不得重新选择candidate或重算跨rankbinding。
- tasks/14已经在private `PreparedTargetRank`中完成entry output/workspace/status ABI preparation，随后把instruction
  结构保持地lower为fixed `void(i64...)` target LLVM CRT calls；RISC-V device link在该lowering之后发生。
- 当前Wafer CRT header、lowering、source和checker形成109个production symbol的闭合surface。CRT源码通过
  `TsmNew*`/operator function table填写`Tsm*Instr`并调用`TsmExecute`，而不是直接写模型结果；symbol存在仍只证明
  ABI closure，不证明packet、numeric或hardware completion。
- Q17正式交付物是all-and-only RISC-V ELF `TargetArtifactBundle`；Q18 package只包含typed resource/slot/module/
  entry/completion/transport requirement，不包含instruction schedule。
- Q18当前只实现pure no-card `RuntimeSessionPlan`。真实`RuntimeProvider`生命周期和board执行尚未实现。
- Q19直接消费accepted instruction/memory facts；其reference-only numeric和deterministic DTE policies不得成为
  target model的实现事实源。

近期实现应把tasks/14中已完成ABI preparation和full target conversion的结果提升为owner-backed、不可序列化的
**target LLVM bundle**。它是实际fully legal LLVM IR及typed slots的all-rank集合，同时被现有RISC-V link、direct
ABI-smoke和host-CRT/SystemC CModel消费；它不是新IR层、packet artifact或package成员。具体C++类名和文件布局只作
实现索引，不属于artifact合同。

repo CRT当前在源码内强制定义`USING_RISCV=1`，附带instruction/common-util archive又是RISC-V object，因此它不能原样
形成host CModel。实现时应把platform选择移到build contract，并让RISC-V和host构建共享同一份repo CRT wrapper与
command-invocation C源码；Tsm operator/packet builder只有RISC-V archive、没有可共享源码，host侧仍需独立的
project-owned实现；只有取得独立packet/MMIO事实源后才能增加vendor packet provenance。其它host差异只落在operator/
MMIO/memory/context adapter。当前Direct DTE
sender/receiver使用process-global static状态，host多rank执行不能共享该状态；必须改成明确绑定invocation/rank的state，
或证明每rank独立装载实例及其lifetime，
不能从OS thread、symbol名或调用顺序恢复rank。

该bundle必须显式拥有canonical all-rank domain、每rank logical rank/entry/module、`ExecutionConfig`、ordered typed
ABI slots、target identity/revision、target/kernel ABI facts和覆盖全部module的context/owner lifetime。所有rank完成ABI
preparation、full conversion和readback后才能原子构造bundle并交给任一consumer；不能让device link和host model分别
从默认值恢复target facts，也不能让passing ranks先行执行。

### 3.2 已确认的硬件功能和事务事实

当前静态证据足以支撑以下受支持子集：

| 范围 | 已确认事实 | 初版可声明 | 仍不可声明 |
| --- | --- | --- | --- |
| topology/memory | 当前目标单卡4×4、16 tile；每tile 3 MiB SPM，末64 KiB有Kcore/runtime占用证据；另有SPM alias和多个consumer-specific DDR mapping/address-view线索 | 当前profile的rank/tile隔离、SPM reservation和byte-level model window；DDR只按typed resource注册 | 所有SKU容量、bad-tile/PG行为或单一静态DDR范围可硬编码 |
| physical layout | compiler当前accepted compact、Cx/NCx规则可由typed layout重算；hardware helper另有256B alignment/footprint证据 | source-backed accepted layout的logical coordinate到physical byte offset及越界检查 | Cx/NCx名称或256B规则是所有dtype/op的通用硬件layout |
| packet/ABI | CT、NE、RDMA、WDMA、TDMA packet字段、worker window、trigger和range/end字段已恢复 | typed decode、字段宽度、descriptor/range conformance | vendor CRT实际packet与host重实现天然一致 |
| movement | contiguous/strided RDMA/WDMA、基础TDMA/gather-scatter/memset的混合单位和descriptor字段已恢复；element count、byte stride、iteration-minus-one及byte-count字段按family区分 | 受支持descriptor的功能执行和逐字段单位检查 | 把所有count当byte、所有alignment/stride组合的性能公式 |
| local issue | 每tile三个worker window；`serial_mode=0`语义把accepted CT/NE/RDMA/WDMA/TDMA分成五个逻辑NCC issue class并存在range/busytable依赖 | model-only per-worker config确认parallel mode时显式五个逻辑class；否则采用更保守serial profile；local drain与event ordering分离 | `3×5`物理queue实例、provider已逐worker设置/读回parallel mode、queue容量、多发射、精确仲裁和worker物理独立性 |
| Direct DTE | raw DTE register含模式/多目的字段；public helper只证明single-destination setup、receiver-ready、issue、busy/done/error、wait/release生命周期及部分resource限制 | 当前accepted single-destination profile的功能/event模型 | raw broadcast/gather等完整模式及精确contention |
| PMU | DTE/SPM/NCC base、record种类和counter shape已知；现有helper的worker scope不一致 | 保留raw counter、记录实际register/worker provenance并建立校准实验 | 所有counter均per-worker，或单位、wrap、workload correlation已证明 |

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

当前依赖暴露两级、尚未闭合的CModel seam：

1. **instruction/packet级**：`instr_operator.h`声明`initTsmOpPointer_cmodel`，`instr_adapter.h`在host分支声明
   `instr_tick_cc`和cycle-mode接口，`op_fw_sim_if` host CMake却只提供include-only INTERFACE target。checkout中没有
   `init/freeTsmOpPointer_cmodel`或`instr_tick_cc`定义；RISC-V `libkcorert.a`里的cycle-mode set/get只是兼容stub。
   当前repo CRT也不调用`initTsmOpPointer_cmodel`，而是直接`TsmNew* -> 填packet -> TsmExecute -> TsmDelete*`，所以
   仅取得该operator-table initializer仍不足以host化CRT。
2. **host runtime级**：x86 `libtx8_runtime.so`的`Runtime::SetCModelHandle`会尝试`dlopen`
   `libcmodel_runtime_api.so`并解析device、compile、launch、run、copy和tile-info等高层入口。被加载的library、其匹配
   header/resource以及该binary还依赖的host libraries均不在checkout；现有binary只证明`dlsym`结果被存入字段且library
   handle会被`dlclose`，没有证明普通launch路径读取/调用这些字段。因此这是vendor CModel存在的强线索，不是当前
   可运行provider，更不能证明其内部
   使用SystemC或能消费Q17/Q18 artifact。

附带`libinstr_tx81.a`、`libcommon_util.a`和`libkcorert.a`都是RISC-V object，不能链接进host model；当前vendor tree和
x86 runtime的公开依赖/symbol中也没有SystemC/TLM痕迹。SystemC仍可能封装在缺失的CModel library内，故结论只能是
“实现技术未知”，不能写成vendor已确认使用或不使用SystemC。

进入实现前应优先向vendor索取完整`libcmodel_runtime_api`套件或低层host instruction model、匹配headers/resources、
支持的target revision、artifact输入、SystemC及其它transitive dependency版本/license、numeric profile、threading/time
contract和可重放要求。高层套件只有证明能消费当前verified package/session后才作为typed `RuntimeProvider`；否则只作
外部differential oracle。低层套件在Tsm/packet/MMIO边界接入本文架构。两者都不得绕过capability、differential和board
correlation gate。

### 3.5 行业调研结论（非规范性）

SystemC官方把它定位为C++上的system-level design/modeling/verification语言和类库，用于architectural exploration、
virtual platform和HW/SW co-design；TLM-2.0重点覆盖memory-mapped bus/on-chip interconnect，并区分LT/AT coding style：

- <https://systemc.org/overview/systemc/>
- <https://systemc.org/overview/systemc-tlm/>
- <https://systemc.org/resources/standards/>
- <https://www.accellera.org/images/downloads/standards/systemc/TLM_2_0_LRM.pdf>

NVDLA的官方virtual platform是register-accurate SystemC/TLM-2.0平台，QEMU CPU包装成SystemC module，公开CModel按
硬件block使用`SC_THREAD`、FIFO、event和TLM transport；这证明SystemC在accelerator register/transaction model中是
成熟常见做法，不证明照搬其block划分、版本或精度标签适合Wafer：

- <https://nvdla.org/vp.html>
- <https://github.com/nvdla/hw/tree/nvdlav1/cmod>

反例同样重要：Spike等ISS可用plain C++实现，Verilator可生成plain C++或SystemC wrapper。TLM LRM也允许untimed功能
模型是普通C函数或单个SystemC process。因而SystemC提供的是模块、并发、离散事件、simulation time和组件互联，不自动
提供numeric、packet、bit或cycle accuracy；定宽类型本身也不决定舍入、饱和和累加语义。LT通常用blocking transport和较少
timing point，AT通常用non-blocking phase表达更多timing point，但二者是coding style而不是自动精度等级，TLM-2.0不定义
cycle-accurate coding style。本文选择SystemC主架构来自Wafer当前确有16 tile、三个worker window、条件化五个逻辑NCC
issue class、MMIO、
Direct DTE/FSM和多completion domain，而不是“CModel必须用SystemC”的语言规则；functional kernel仍保持plain C++。

- <https://verilator.org/guide/latest/overview.html>
- <https://github.com/riscv-software-src/riscv-isa-sim>

## 4. 模型架构

### 4.1 Target frontend分层

模型保留三个互不冒充的执行入口，以及一条可选的独立packet/MMIO证据路径：

1. **Direct target-call ABI smoke**：host LLVM JIT执行同一fully legal target LLVM module，并给`wafer_tx81_*` fixed ABI
   注册最小direct shim。它验证target ABI preparation、control flow、call graph、typed arguments和address formation；
   因为绕过repo CRT、Tsm method table和packet，只是快速诊断，不满足正式Q22 CModel gate。
2. **Host-CRT/SystemC frontend**：host执行同一target LLVM，调用与device build同源的repo CRT；host platform实现
   `TsmNew/Delete`、method table、`TsmExecute/TsmWaitfinish`和Direct DTE/FSM/memory hooks，将CRT实际构造的packet/
   event投递给SystemC model。这是近期正式functional-event CModel路径。自行实现的host operator仍是project-owned
   packet builder；逐字段独立packet/MMIO correlation前不能称vendor-exact packet。
3. **Exact-module frontend/provider**：从Q18 verified package加载并执行Q17 RISC-V ELF。project ISS adapter通过loader
   ABI、MMIO/custom instruction、Direct DTE和completion adapter复用SystemC architecture；独立vendor simulator不要求
   内部使用SystemC，但必须满足相同artifact、provider lifecycle、capability和evidence合同。只有此入口完成后，才可称
   package-facing target model provider。
4. **Optional independent packet/MMIO evidence**：消费exact packet words或register transaction，与host-CRT路径的packet
   decode、worker/engine、address/range/trigger和observable memory effect比较。positive必须来自exact Q17 ELF经ISS执行
   真实CRT/archive产生的register trace、board capture，或vendor提供且版本可审计的builder/CModel；手写packet只补
   negative。缺少这条证据不阻塞数值模型或板端数值相关，但禁止`vendor-exact packet`声明。

这些路径可以发现不同错误；任何一条通过都不能冒充其它证据。测试和diagnostic必须记录实际frontend、CRT/packet
provenance、capability、numeric profile和event profile。

当前fixed CRT ABI没有额外model-context参数。host frontend必须把每次entry call显式绑定到对应invocation/rank context。
SystemC通常在单个OS thread上cooperative调度多个process，`thread_local current_rank`不能单独代表rank；可行机制包括
per-JIT symbol trampoline加host-only context accessor、按`sc_process_handle`绑定context，或每rank独立CRT/JIT symbol
namespace。不得使用跨invocation共享的process-global mutable singleton，也不能从thread id、调用顺序或symbol名字推断
rank。机制必须覆盖并发rank、阻塞yield、异常退出和nested call后的恢复测试。

target LLVM中的i64地址参数按typed ABI角色分别表示NCC-visible tile-local SPM offset/address、compiler-planned DDR
arena base加offset，或
external DDR/resource的invocation-time registered device address；它们不是可任意解引用的host pointer，也不能被压成
一个固定DDR范围。host frontend只能通过typed ABI slot、resource identity和checked address registry解析；未知base、
overflow、跨resource访问或把SPM alias当host虚拟地址都必须在memory effect前拒绝。

当前repo CRT的多数public helper返回`void`且会丢弃`TsmExecute`返回值。host platform必须另有invocation-local error latch，
在entry结束、wait和all-rank terminal前汇总factory、packet、address、engine和event错误；该latch属于model diagnostic，
不能伪装成target ABI中原本不存在的return。板端exception/status仍作为独立证据源。

### 4.2 SystemC架构和plain C++ kernel

正式模型按稳定硬件责任拆分为SystemC modules/channels：

- model top/context：target revision、当前profile的16个physical tile slots、capability/good-tile map、logical/physical
  rank/tile mapping和invocation-local状态；
- tile memory：每tile SPM、reserved region、card DDR/resource slots和checked address translation；
- 每tile三个worker-facing issue domain；model-only per-worker parallel config声明`serial_mode=0`语义时，每个domain区分CT、NE、
  RDMA、WDMA、TDMA五个逻辑issue class，否则进入不声称五类并行的保守serial profile；该表示不宣称`3×5`物理
  queue实例或engine复制关系；
- tile-level CT/NE/RDMA/WDMA/TDMA engine endpoints及保守共享resource arbitration；
- Direct DTE fabric/FSM：rank-local endpoint/status、receiver-ready、source read/lifetime、send/recv/wait/release和destination
  visibility；
- completion/result：issue、local drain、destination visible、all-rank terminal和typed no-progress diagnostic。

当前静态证据没有证明target execution environment已设置`serial_mode=0`、queue depth、同queue多发射、三个worker是否各有独立engine、
cross-worker arbitration或SPM bank函数。第一版只能建立untimed/delta-cycle functional-event模型，对未知共享resource
保守串行；不能把任意`sc_fifo`容量、`3×5`物理queue或调度顺序升级成微架构事实。worker只能从最终packet的
`inter_type[9:8]`取得；若当前producer只生成
worker 0，source-backed gate只声明worker 0，多worker关系由packet component test覆盖，不能在frontend自行round-robin。

packet decode、地址检查和各engine的numeric/memory effect由不包含SystemC header、不链接SystemC的plain C++ kernels
实现，并可独立做unit/property/differential test。初版采用model-only保守observable-commit policy：kernel在SystemC
event确定的完成点一次提交可观察memory effect，不能在`TsmExecute` issue时就让结果可见；这不宣称硬件没有partial
write或相同visibility时刻。它们与Q19仍不共享compute kernel、rounding policy或DTE scheduler。

typed target command只是一次packet/register decode产生的瞬时transaction；CRT packet多数位于栈上，host
`TsmExecute`必须在返回前完成decode或复制异步处理所需的`Tsm*Instr`字段，绝不能保留caller栈指针。该复制要求只
处理instruction bytes，不授权在Direct DTE issue时snapshot payload；DTE source具体读取时刻仍由model profile和后续
board/vendor correlation决定。transaction不预构造整程序command vector，不进入
compiler artifact或package，不能成为长期shadow program。SystemC modules读取transaction执行时状态，不复制compiler
completion DAG或planner trace。exact ELF路径必须经过ISS产生raw packet/register transaction再进入decoder，不能由ISS
hook直接合成高层command绕过packet证据。

### 4.3 实现对象和调用边界

以下名称是实现索引，不是新的IR、package协议或稳定public ABI：

| 对象 | 必须拥有 | 禁止拥有 |
| --- | --- | --- |
| `TargetLLVMModuleBundle` | 共享MLIR/LLVM context owner、canonical all-rank domain、每rank logical rank/entry/fully legal module、ordered ABI slots、`ExecutionConfig`、target/ABI identity | packet list、schedule、model state或任何可序列化sidecar |
| `TargetModelCapability` | op family/kind、src/accum/dst dtype tuple、shape/layout/tail和descriptor范围、rounding/optional-field tuple、numeric comparator、event/transport支持 | 由symbol存在推导的隐式支持、扩大compiler legality的规则 |
| `TargetModelInvocation` | bundle引用、all-rank typed input/parameter/output binding、checked external resource registration、选定model profile | host pointer伪装的device address、从文件名恢复的rank/resource |
| `TargetTransaction` | `TsmExecute`返回前复制的packet字段、worker/engine、地址/descriptor、dtype/shape/optional fields和invocation-local sequence identity | caller栈指针、整程序command vector、Q19 schedule或DTE payload的无证据snapshot |
| `TargetModelResult` | all-and-only rank terminal status、完整output、stable failure stage/rank/transaction、frontend/CRT/packet/numeric/event provenance | partial successful rank集合、回写compiler/package的状态 |

capability row的有效key至少包含target revision、op family/kind、src/accum/dst dtype、rounding和optional/fusion/
zero-point tuple、shape/layout/tail class、descriptor/alignment class、worker和completion profile。实现支持谓词是
`compiler accepted tuple ∩ model kernel tuple`；板端profile仅在Q22.C进一步收窄，不能扩instruction legality。enum、CRT
symbol或packet decoder存在都不等于row已支持。首版状态至少区分`unknown`、`model-only`、`board-observed`、
`calibrated`、`supported`和`rejected`，并携带unsupported reason。

`TargetLLVMModuleBundle`在所有rank完成ABI preparation、full conversion和verification后原子形成。host执行把每rank module
翻译到独立LLVM context和独立ORC `JITDylib`，避免同名entry/private helper碰撞；JIT clone设置native triple/data layout，
并用显式symbol map注册repo CRT host symbols，不能依赖当前process偶然导出的符号。host retarget前拒绝target-specific
intrinsic、inline asm、未知address space或其它不能安全host materialize的LLVM结构；用于RISC-V link的module保持不变。
由于entry是动态slot数量的`void(i64...)` fixed ABI，JIT层生成统一签名的host-only thunk，例如
`void __wafer_model_invoke(const uint64_t *slots)`，按已验证slot数加载并调用原entry；禁止把可变参数函数指针强转调用。
direct shim和正式Host-CRT frontend复用同一bundle和slot materialization。所有rank JIT/thunk materialize成功后才发布
model executable。

正式entry必须在SystemC可yield process中调用，使`TsmWaitfinish`、DTE wait和FSM receive可等待`sc_event`；所有rank process
先创建再启动。invocation/rank context由显式trampoline或`sc_process_handle`绑定，不能只靠TLS。第一版使用单个长寿命
`ModelSystem`，不在每次invocation调用`sc_stop`；只在所有process/queue/event quiescent后清理invocation-local memory和
error latch，并先串行化invocation。后续并发只有在context隔离和reset测试闭合后才能开放。

执行前structural/capability preflight必须一次枚举并拒绝：任何reachable CRT symbol/factory family缺host实现；任何静态
可知的op/dtype/shape/optional tuple无kernel或comparator；address plan、ABI binding或rank/transport endpoint不闭合。
preflight通过后才在私有invocation backing中导入input并创建SystemC运行态。运行时builder产生的packet fields、dynamic
descriptor和computed address必须在每条transaction的read/effect前验证；失败整体丢弃私有invocation且不发布output，
但不虚构一次不具备通用性的dry-run来声称它们在input import前已知。plain C++ `NumericKernelRegistry`按完整typed key精确
查找，禁止按symbol/op名字fallback。

memory分为只含metadata的`InvocationAddressPlan`和通过plan后才建立的私有`InvocationMemoryRegistry`。registry为每个
typed ABI resource、workspace/status和rank-local SPM分配不重叠的synthetic device range；entry只得到这些device address。
resolve必须显式携带rank/tile、address space、resource、read/write role，并证明checked range完整落入唯一region，不能按
地址数值阈值猜SPM/DDR或直接解引用host pointer。每条transaction先读完整snapshot，plain kernel返回待提交byte effects；
所有range/payload通过后，SystemC completion event才一次提交。单command错误无partial write，整次invocation只有all-rank
terminal success后才extract/copy output。

### 4.4 数值实现独立性

Q19和target model可以共享稳定dtype enum、physical geometry、packet field definition和ABI常量，但不能共享完整
compute kernel、rounding policy或DTE scheduler。否则differential会让同源bug同时通过。

target model对numeric profile使用以下规则：

- 有硬件/ABI直接证据并有区分性测试的行为进入supported profile；
- 只有数学名称而无edge behavior证据时，模型结果标记为model semantics，不能称hardware-bit-exact；
- zero-point等没有唯一公式的组合在执行副作用前拒绝；
- stochastic行为在取得seed/state合同前只可做独立统计correlation，不得用Q19的SplitMix64 policy冒充硬件；
- 每个profile都必须列出dtype、accumulator、shape/descriptor、optional fields、completion和比较口径，不能从enum存在
  推导支持；
- movement、integer、boolean、guard/canary和其它确定性raw storage effect必须byte/bit exact；
- floating comparator按op/dtype/profile定义，不使用全局tolerance：只有重复执行、跨reset和held-out都bit-stable时才发布
  bit-exact；否则同时记录ULP和`abs <= atol + rtol * |board|`，near-zero、GEMM/reduction length另分policy；
- NaN classification、payload/sign，Inf sign，signed zero，subnormal/FTZ/DAZ和overflow/saturation/wrap/status分别比较，
  不交给普通`isclose`；
- profile阈值在查看held-out结果前冻结；Q19/CPU已有tolerance不自动成为board numeric tolerance。

### 4.5 SystemC/TLM和simulation-time边界

SystemC是Q22正式functional-event CModel的强制执行容器，这是项目工程选择，不是从vendor binary恢复出的事实，也不
自动提升精度。target-model feature在configured build中必须找到受管、版本固定的SystemC dependency并真实执行相应
tests；基础compiler、reference executor、manifest parser、no-card runtime和plain C++ kernels仍不依赖SystemC。

TLM-2.0不要求用于每条内部边界。推荐范围是：

- DDR、SPM aperture、MMIO和未来ISS/interconnect使用TLM generic payload及必要typed extension；
- NCC packet、worker queue、Direct DTE/FSM event使用保留字段身份的typed transaction/channel；
- 不把硬件packet再包装成opaque payload，不让socket/port拓扑进入compiler IR或package；
- 初版只发布untimed/delta-cycle functional-event profile；没有测量依据时不写任意`wait(N, SC_NS)`；
- `TsmWaitfinish`、`direct_sync_wait`、DTE wait和FSM receive必须从可yield的SystemC process等待event，不能busy-poll；
  所有rank process先注册再启动，不能顺序运行到第一个peer wait才创建其它rank；
- SystemC scheduling order不能充当message identity、resource binding或compiler completion语义；
- LT、AT、temporal decoupling、DMI及任何`wait(N, SC_NS)`只属于deferred Q22.P；恢复时必须另行更新设计和实施计划。
  cycle-accurate仍需RTL/per-cycle trace、vendor cycle model或完整微架构合同，并不是本文路线的默认终点。

## 5. Runtime和exact ELF边界

### 5.1 近期Host-CRT/SystemC执行

近期正式入口是同一次`wafer-compile` invocation中的下游verification consumer：Q17/Q18先按各自合同原子发布verified
target/package artifacts，driver再用仍由invocation持有的owner-backed target LLVM bundle，host执行target LLVM并
调用与device build同源的repo CRT wrapper/command-invocation C源码。host-only platform层负责rank context、Tsm factory/
operator、packet复制、checked address/MMIO和可yield wait；packet/event随后进入SystemC model。model mismatch或执行
失败可让driver返回非零并保留diagnostic，但已验证package保持可审计，不回滚、不改写，也不让Q22成为Q17/Q18
correctness前置。是否启用该gate属于用户显式请求或configured CI policy；不能变成wafer-opt stop-stage或手拼pass。

direct `wafer_tx81_*` shim可以保留为快速ABI smoke，但它绕过repo CRT、Tsm object和packet，不计入正式CModel完成。
Host-CRT/SystemC入口能证明compiler-produced target call经过project CRT/packet/event语义并产生正确完整输出；在与
RISC-V archive register trace、board capture或vendor builder逐字段相关前，它仍不证明vendor-exact packet。该入口也
不读取已发布package、不执行RISC-V ELF/vendor archive，不能使用`wafer-run`的provider语义。

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
- Direct DTE receiver ready、busy、done/error、source lifetime和destination visible；
- multi-rank no-progress/deadlock；
- provider terminal status和copyback eligibility。

`TsmExecute` success不表示完成，`TsmWaitfinish[_bywork]`只证明对应local NCC domain，不能替代DTE wait或multi-tile
arrival。每个issue domain使用显式、单调的invocation-local issue ordinal/watermark；wait捕获调用前watermark并等待相关
command完成，不能用delta scheduling order充当identity。保守模型可以延后event完成，但不能合并没有证据的completion
domain。所有rank/process均等待且SystemC event queue为空但仍非terminal时，形成确定性no-progress state snapshot并整体
失败，不用wall-clock决定语义。

所有入口在input import前完成all-rank structural capability、static profile、slot/resource/address-plan和endpoint preflight。
unknown symbol/factory/static profile或exact-module环境不匹配必须在此时整体失败。只有执行packet builder后才可知的
packet/op tuple、computed address、dynamic descriptor和runtime event错误，在私有invocation内、相应read/effect前拒绝。
这种失败仍必须：

- 停止dependent event和copyback；
- 标记受影响invocation/rank，不产生可误认为成功的partial result；
- 逆序释放已获取资源；
- 保留稳定stage、rank、entry、command/event类别和model/profile provenance；
- no-progress使用确定性state snapshot诊断，不依赖wall-clock或thread调度决定语义。

## 7. 模型精度和发布标签

| Profile | 最低事实和gate | 允许声明 | 禁止声明 |
| --- | --- | --- | --- |
| direct ABI smoke | same fully legal target LLVM、typed ABI、direct symbol shim和基本地址检查 | target lowering/ABI smoke | CRT、packet、event、package ELF、board |
| repo-CRT/SystemC model-only functional-numeric | same fully legal target LLVM、同源repo CRT host build、project packet、untimed SystemC event/completion、独立plain C++ kernel和完整输出differential | project CRT/packet路径在supported model profile中的functional-numeric correctness | vendor-exact packet、hardware numeric、package ELF、board、timing |
| optional packet/MMIO conformance | exact-ELF register trace、board capture或versioned vendor builder与project packet的逐字段decode/range/engine/register-effect conformance | 对应capture范围的packet/register provenance | 仅凭trace声明numeric、完整loader/provider lifecycle、timing或board等价 |
| Q22.C board numeric correlation | Q22 model-only通过、Q6.B有效board execution、区分向量、重复稳定性、冻结的comparator、独立held-out和source-backed完整输出 | 绑定环境和tested domain的board-output-correlated profile；对应独立packet/MMIO evidence闭合后才升级为hardware-correlated-numeric | vendor-exact packet、不可观测内部实现、未测domain、exact package、timing |
| exact-module functional | verified package、all-and-only RISC-V ELF、ISS/loader/MMIO/provider lifecycle | package target-model execution | real board、hardware timing |
| deferred Q22.P timing | 另行恢复后的PMU measurement basis和独立timing held-out | 实际通过的LT/AT profile | 改变numeric、IR legality或candidate acceptance；无RTL证据时称cycle-accurate |

profile是execution result provenance，不是compiler legality或package semantic branch。高层profile失败不能反向改变
accepted instruction支持范围；如果硬件可表达但model未覆盖，应扩target model capability或保持该profile拒绝。

## 8. Verification Gates

### 8.1 Capability、ABI smoke和Host-CRT/SystemC gate

- 自动从当前lowering/CRT typed surface得到all-and-only direct-shim与host-CRT platform coverage，不复制109项字符串表；
- direct shim只检查symbol、signature、control flow、typed slot和基本address formation，结果明确标记ABI smoke；
- 同一`ExecutableBundle`进入Q19；由它经tasks/14同一full conversion派生的owner-backed target LLVM bundle进入
  Host-CRT/SystemC model。两者整数exact，浮点按显式dtype/op tolerance比较完整output；
- 真实rank-count=1 f32 linear/residual MLP是首个纵向gate，手写LLVM/MLIR只补negative。该slice只覆盖它实际调用的
  RDMA、WDMA、gather/scatter、GEMM、elementwise add、tanh和local fence，以及对应
  `TsmNew{Rdma,Wdma,DataMove,Gemm,Arith,Activation}` factory；不以109-symbol surface一次宣称全部支持；
- 正式positive必须经过同源repo CRT的Tsm factory/operator/packet、`TsmExecute`、local wait及适用Direct DTE/FSM hooks；
  SystemC-enabled tests若unavailable/skipped则该gate未完成；
- unknown symbol、factory、wrong ABI slot、static profile和address plan在input/model allocation前拒绝；dynamic packet、
  地址/descriptor/narrowing和numeric tuple在对应transaction effect前拒绝，且不产生外部output；
- target LLVM bundle构造的任一rank late failure不形成bundle或partial model result；Q17/Q18各自仍按原合同失败。
  bundle构造和package均成功后，model mismatch只让verification返回非零并保留已验证package供审计。

### 8.2 Q22 project packet和memory gate

- CT/NE/RDMA/WDMA/TDMA的typed args、packet fields、worker、trigger、range/end、raw `TsmExecute` status、model error
  latch和target ABI可观察status逐family区分覆盖；
- SPM/DDR bounds、reserved region、Cx/NCx、bitpacked i1、subview/strided descriptor用独立slow oracle或board bytes验证；
- Host-CRT/SystemC positive来自compiler-generated target LLVM，并把repo CRT实际构造的project packet作为正式
  functional-event输入；Q22内用独立field/memory oracle覆盖raw bytes、decode和observable effect，但project host
  builder不得标记vendor-exact；
- 若取得exact-ELF register trace、board capture或versioned vendor builder，可另做逐字段packet/MMIO correlation并提升
  packet provenance；它是可选诊断证据，不是Q22或Q22.C前置，也不能替代numeric vectors。

### 8.3 Q22.C Board numeric correlation gate

Q22.C在Q22、Q6.B和configured numeric corpus均可用后执行，不要求PMU、packet capture或exact-module provider。Q6.B先
证明allocation/H2D/load/launch/wait/status/D2H/cleanup、watchdog/reset和重复invocation有效；任何provider、completion、
copyback或guard失败的sample均为invalid，不能用于调numeric。

板端corpus按capability row生成，而不是按一个op名字笼统通过。首批顺序是：

1. movement/fill/boolean/guard和未写区域byte/bit exact；
2. rank-count=1 f32 GEMM、add和tanh；
3. Tiny Llama实际需要的reduce、exp、rsqrt、mask/i1和其它已accepted tuple；
4. 16-rank Direct DTE与完整output；
5. f16/bf16/TF32、convert、zero-point和stochastic只在各自事实闭合后逐row扩展。

每个numeric family至少覆盖以下能区分预先列出的候选语义的向量；有限样本不证明全输入域等价：

- rounding：正负halfway、halfway±1 source ULP、最大有限值边界、min normal/subnormal和float-to-int的
  `N + 0.5`，测试值从raw bits构造；
- special values：±qNaN/sNaN及payload、±Inf、±0，max/min同时交换operand顺序；DAZ与FTZ分别测；
- overflow：float max附近及integer边界±1，区分Inf/max-finite、wrap/saturate/trap/status；
- GEMM/reduction：cancellation及排列、K/reduction length跨tile/tail边界、可区分accumulator width/order和
  fused/non-fused的向量；bias、psum和activation optional field一次只启用一项；
- zero-point：正负和边界输入与多个zero-point交叉，区分加、减、reinterpret、clamp和wrap；不能唯一排除候选公式前
  保持unsupported；
- layout/memory：非零base、stride/iteration、alias、Cx/NCx和bitpacked/tail边界，同时检查pre/post raw bytes和canary。

comparison policy逐op/dtype/shape/profile冻结：movement/integer/boolean/raw storage exact；deterministic float只有重复、
跨reset和held-out逐bit稳定后才标`bit-exact`，否则使用明确的ULP及abs/rel envelope；NaN/Inf/signed-zero/subnormal/
overflow/status单独处理；stochastic无seed/reset/state/advance合同只能发布统计profile，不能成为逐元素CI oracle。
calibration corpus、随机held-out和Q20/Q21完整output相互隔离，rank-count=1/16均比较all-and-only outputs。

profile identity分为semantic validity key和run provenance。前者至少绑定SKU/revision、firmware、driver/runtime、instruction
library、CRT/ABI和optional mode；后者记录compiler/model commit、package/module digest和SystemC版本。单块板只能标
`device-unit-observed`，同revision多块板复现后才可标`revision-correlated`。通过后仍只声明tested domain内的
`board-output-correlated` numeric；只有相应独立packet/MMIO证据存在时，才升级为`hardware-correlated-numeric`并增加
packet/opcode provenance，绝不由numeric output反推vendor-exact packet。

### 8.4 Multi-rank和provider gate

- 16个rank拥有独立SPM/context/status，all-and-only accepted rank都执行；
- Direct DTE覆盖receiver-ready、source在send completion前保持合法、send/recv/wait/release、receiver completion后的
  destination visibility、duplicate/missing/mismatch和deadlock；具体source read时刻在板端或vendor证据前只属于
  model profile，不能照搬Q19 snapshot policy冒充硬件；
- target model不能读取Q19 logical message schedule或用reference coordinator实现transport；
- exact provider按allocate/import、H2D、load/resolve、submit、wait/status、D2H、cleanup逐阶段注入失败；
- wait/status失败后禁止copyback，任一rank失败无partial successful result。

### 8.5 Source-backed和correlation gate

- 复用Q20 rank-count=1/16 linear/MLP和Q21 16-rank tiny Llama的同一source/config及accepted upstream chain，不建立
  model专用fixture或计划；Host-CRT/SystemC消费同一invocation的target LLVM，只有exact-module frontend才消费原样package；
- 检查全部rank/module/output、typed status和可观察event/packet数量，不用rank 0、shape或digest代替执行；
- target model与Q19/CPU差异先按target call、CRT/packet、memory、numeric、event分类；
- configured board可用后，Q22.C以同source/accepted program比较model/board完整output和status；exact-module frontend
  未闭合时明确标cross-frontend，闭合后才可增加same-package标签。packet capture和PMU都不是numeric gate前置；
- tests为unsupported/skipped时对应profile gate保持未完成。

Q20/Q21的具体模型、shape、dtype和tolerance只是source-backed gate参数，不定义target model协议。通用合同来自fully
legal target modules、typed ABI/capability、packet和address/completion关系；其它workload按同一关系执行，超出profile
时结构化拒绝，不能把这些case的调用序列固化进模型。

## 9. 板端数值相关设计

### 9.1 样本有效性和记录纪律

每次板端运行必须记录device SKU/revision和unit identity、good-tile map、firmware/driver/runtime/instruction-library/CRT/
ABI digest、compiler/model revision、package/module digest、input seed、typed invocation、完整input/parameter/output raw bytes、
pre/post DDR/SPM及guard bytes、provider phase、terminal status/exception、reset/repeat identity和comparison report。packet/
register capture若存在则记录来源、完整性和digest；不可用时明确写`unavailable`，不能补造。

Q6.B的allocation/import/H2D/load/launch/wait/status/D2H/cleanup、watchdog/reset和重复invocation先闭合。input未正确写入、
completion不可信、copyback失败、guard被破坏或reset后health check失败的sample必须保留但标`invalid`，不得进入numeric
calibration。calibration vectors、随机held-out、shape/layout held-out和真实workload corpus必须预先分区；comparison policy
和阈值在查看held-out结果前冻结。Q22.C数值校准不需要PMU、wall-time或固定频率。

### 9.2 Capability profile和graduation

每个profile row的semantic validity key至少包括：target SKU/revision、firmware、driver/runtime、instruction library、CRT/
ABI、op family/kind、src/accum/dst dtype、rounding、optional/fusion/zero-point tuple、shape/layout/tail、descriptor/alignment、
worker和completion profile。row value保存state、comparison/special-value/status policy、tested domain、packet evidence level、
calibration/held-out/evidence digest、repeat stability和unsupported reason。run provenance另存compiler/model commit、SystemC
版本及package/module digest，不能用一次run identity代替semantic validity key。

状态只按以下证据单向升级；失败时拆细row或回退，禁止放宽全局容差：

1. `unknown -> model-only`：独立plain C++ kernel unit/property、packet/memory component和Q22 source-backed链均通过；
2. `model-only -> board-observed`：Q6.B有效single-op microbench、完整raw bytes/status及多次重复通过；
3. `board-observed -> calibrated`：区分向量在预先列出的候选语义集合中唯一排除其它候选，或明确形成统计/approximate
   comparator，且calibration set冻结；
4. `calibrated -> supported`：预留的value、shape、layout、optional-field和真实MLP/Llama held-out全部输出通过，无silent
   exception，跨reset稳定。

profile永远只能收窄`compiler accepted tuple ∩ model kernel tuple`，不能反向扩大IR/instruction legality。单块板证据标
`device-unit-observed`；至少多块同revision板复现后才可标`revision-correlated`。如果packet provenance不可用，仍可发布
tested domain内的`board-output-correlated` profile，但不得把row归因为vendor-exact opcode/register effect。

### 9.3 黑盒板端的可证与不可证边界

只有launch、status和最终output的黑盒板端，仍可证明versioned环境、tested input/shape domain内的端到端I/O稳定性以及
CModel是否满足预注册comparator；若compiler能产生isolated single-op microbench，多数rounding、accumulation、fusion和
zero-point候选也可通过区分向量经验排除。

黑盒结果不能证明vendor-exact packet/register、观测等价候选中的内部accumulator精度/顺序/融合、未测试输入域、不可见
exception、DTE精确source-read/visibility点、stochastic seed/state或跨revision泛化。若只能运行整网且不能构造single-op
或读取中间buffer，只能发布workload-level correlation，不能升级per-op capability。有限held-out通过也只证明声明的
tested domain，不是数学上的全域等价证明。

### 9.4 Evidence artifact和事实回写

每次run形成独立、不可变的verification evidence directory，至少包含环境identity、package/module/model/profile digest、
typed invocation、input/expected/actual raw bytes、guard、provider status、可选packet/register capture、comparison report和
evidence digest；profile只引用这些digest。它不是package sidecar，不被compiler legality、lowering或runtime launch消费。

板端结果与静态hardware/ABI资料冲突时，先回到对应owner记录并收敛，不能静默调模型常量。numeric发现只有经owner和
conformance tests闭合后才能扩model capability，仍不能自动扩大instruction legality。PMU、latency、throughput、queue/
bank/contention及LT/AT参数全部属于deferred Q22.P；Q9若未来消费测量结果，应另行生成只排序已合法candidate的derived
cost profile。

## 10. 分阶段交付

### 10.1 Capability和依赖收敛

- 并行向vendor索取完整host CModel development package：匹配`host_runtime.h`/`runtime_api.h`/`tx_runtime.h`/TsmML headers、
  `libcmodel_runtime_api.so`、`libhpgr.so`、`libtsmml.so`、model resources和transitive dependency/license/version；同时确认
  是否存在低层x86 instruction/operator library；取得后作为可替换frontend或独立oracle，不阻塞project-owned实现；
- 固定首批target call、dtype/layout、engine、packet、Direct DTE和completion capability matrix；
- 把tasks/14 private prepared target LLVM提升为owner-backed all-rank内部artifact；
- 增加默认关闭的稳定target-model build feature；固定Accellera SystemC reference implementation版本、获取方式、license和
  单一CMake target。基础compiler与plain C++ kernels仍可独立构建；feature启用时缺SystemC必须configuration fail，未启用
  时正式profile明确unavailable且Q22 gate未完成，不能由direct shim代替；
- plain C++ kernel/property tests不链接SystemC；SystemC test executable使用唯一`sc_main`入口并实际运行，不能只编译、
  skip或用`gtest_main`替代；
- 形成同一repo CRT源码的device/host platform contract，移除源码内强制`USING_RISCV`选择，明确rank-local Direct DTE
  state、checked address和blocking-yield边界。

完成：文档、typed capability和failure分类收敛；未实现symbol/profile在任何mutation前可被完整枚举拒绝。

### 10.2 Host-CRT/SystemC functional slice

- 固定真实`linear-residual-mlp-f32` rank-count=1作为首个vertical：当前case参数是batch 2、input 16、hidden 32、
  output 16、f32、tanh和residual；这些值只是测试参数，不进入capability协议；
- host JIT执行same fully legal target LLVM；
- host执行同源repo CRT，经Tsm factory/operator形成packet并复制到SystemC tile/worker/queue入口；
- 建立rank-local virtual SPM/DDR、typed slots、checked address、invocation error latch和可yield local wait；
- 先覆盖该artifact实际调用的`rdma`、`gather_scatter`、`gemm`、`elementwise_add`、`elementwise_tanh`、`wdma`和
  `local_fence`，以及`Rdma/DataMove/Gemm/Arith/Activation/Wdma`六类Tsm factory；
- 与独立Q19/CPU比较完整rank-count=1输出。

该slice的completion gate是：

- `wafer-compile -> ExecutableBundle -> TargetLLVMModuleBundle -> Q17/Q18 publication -> model`真实链执行，手写LLVM/
  packet只补negative；
- bundle late-rank、host retarget、ORC missing symbol、factory/packet field、address exact-end/overflow/cross-resource/
  reserved-SPM、kernel和event failure均有atomic negative；
- SystemC证明issue后结果尚不可见，local fence按watermark等待，completion一次commit，failure唤醒且不copyback；
- 完整output同时与Q19及现有NumPy expected按case已有`atol=1e-6, rtol=1e-5`比较，并检查全部status和packet family
  count，不只抽查元素；这是model-only comparator，不是板端profile阈值；
- 最后一个WDMA/packet/kernel注入失败时不形成model result，已发布Q17/Q18 artifact仍保留；SystemC-enabled CI test
  实际执行，unavailable/skipped不算通过。

完成后结果标记`repo-CRT/SystemC model-only functional-numeric`，不标记hardware numeric、vendor-exact packet或exact
ELF。direct ABI smoke只作局部诊断。该slice明确不含rank-count=16、Direct DTE、tiny Llama、PMU或timing；它们按后续
阶段扩展，不反向固化首case的shape或call order。

### 10.3 Project packet、event和16-rank transport

- 增加current CT/NE/RDMA/WDMA/TDMA project packet decode和独立field/memory component checks；vendor-exact逐字段
  correlation只在配置独立packet/MMIO source后作为可选provenance gate；
- 在SystemC中建立三个worker window；model-only per-worker parallel config成立时区分五个逻辑issue class，否则走保守serial
  profile；不声明`3×5`物理queue或engine复制；同时建立
  resource event、local drain和Direct DTE/FSM；
- 扩展当前production compute/movement surface和16-rank source-backed case；
- DDR/SPM aperture、MMIO和未来ISS/interconnect采用受限TLM边界；plain C++ kernel component gate保持可独立运行。

完成：rank-count=1/16完整输出、packet/memory/event negative、DTE no-progress和all-rank atomic result通过；Q22完成，
仍不称hardware numeric、exact package/ELF或timing model。

### 10.4 Q22.C Board numeric correlation

- Q6.B先闭合board provider correctness、watchdog/reset和重复invocation；
- 按第8.3节的capability row、区分向量和comparison policy采集calibration corpus；
- 冻结policy后运行独立random/shape/layout/optional-field held-out及Q20/Q21完整output；
- 发布绑定environment identity、tested domain和raw evidence digest的profile；packet capture不可用时降级packet
  provenance，不阻塞board-output correlation。

完成：只有实际通过的row从`model-only`逐级升级，失败/未测tuple保持unsupported。Q22.C不声明exact ELF、timing或未测
输入域等价。

### 10.5 Q22.E Exact package execution

- 在取得vendor simulator或完成RV64 ISS、loader ABI、MMIO/custom instruction和provider lifecycle后接入
  Q18 verified package；
- 原样执行all-and-only Q17 modules，不发布host专用instruction list或修改manifest；
- 通过wafer-run typed provider入口执行Q20/Q21 package和阶段性failure injection。

完成：同一package在model provider中完整allocate到cleanup并产生可信status/output；否则该能力保持更高待解锁gate。

### 10.6 Q22.P Deferred timing calibration

- 本阶段不在近期numeric correctness范围内，只有另行恢复并配置可信PMU/timing environment后才执行；
- 先验证PMU measurement basis，再按single-engine、queue/SPM/DTE/fabric矩阵采样；
- 用独立timing held-out shapes/descriptors/workloads验证loosely/approximately-timed profile；numeric policy保持只读；
- cost calibration如需消费结果，另由tasks/06/16和Q9建立typed consumer，不反向污染correctness。

完成：只发布实际通过的profile标签、适用device/firmware/runtime identity、误差分布和未覆盖范围。没有RTL或vendor
cycle证据时cycle-accurate保持非目标。

## 11. 待讨论问题

以下问题不改变SystemC作为Q22正式functional-numeric容器的当前选择，但会改变dependency、packet provenance和板端
实现成本，因此保留为显式待讨论问题：

1. **Vendor simulator交付**：现有低层header只有接口痕迹，高层x86 runtime会动态寻找缺失的
   `libcmodel_runtime_api.so`。需要确认能否取得完整host-runtime开发包、instruction model、依赖、资源、版本和license，
   以及其输入究竟是Tsm model/session、packet/MMIO还是exact package；若可得，packet或exact-module路径可能显著缩短，
   但仍需独立correlation。
2. **Packet事实源**：需要确认可否复用vendor可审计builder或获得register trace。repo CRT host build产生的是
   project-derived packet；它验证当前compiler/CRT/model链，但在独立correlation前不证明vendor packet完全一致。该缺口
   不阻塞Q22.C的board-output numeric profile，只限制packet/opcode provenance。
3. **SystemC工程基线**：需要固定CI平台、受支持版本、获取方式、license、deterministic scheduling要求和未来ISS
   co-simulation边界。SystemC只拥有event/transaction实现，不改变artifact、packet、功能核或compiler合同。
4. **板端numeric环境和验收政策**：需要固定可用SKU/revision/unit、firmware/runtime/instruction-library/CRT组合、
   reset/watchdog能力、single-op可观测性、calibration/held-out corpus和逐op/dtype comparator；未固定前不发布
   hardware-correlated numeric profile。LT/AT阈值只在deferred Q22.P恢复时讨论。

## 12. 文档和实现归属

- tasks/14拥有target ABI preparation、fully legal target LLVM、CRT和RISC-V module publication；target LLVM bundle
  落地时应在同批同步其producer合同。
- tasks/15拥有exact target model `RuntimeProvider`生命周期和package消费；host-CRT/SystemC模式不是该provider。
- tasks/16拥有direct ABI smoke、host-CRT/SystemC、Q22 project packet/event、Q22.C board numeric、可选packet/MMIO、
  exact-module、board和deferred timing的证据分层及CI gate。
- 本文拥有target execution model内部边界、capability、SystemC选择、板端numeric correlation计划和各层不得冒充的声明。
- tasks/09/11/13仍分别拥有memory legality、instruction/packet legality和Direct DTE/completion；model不能改写这些
  compiler合同。

进入代码实现前应建立独立实施计划；本方案设计本身不表示target model、exact provider、board或calibration已完成。
