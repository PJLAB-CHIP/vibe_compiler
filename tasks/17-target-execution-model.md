# Wafer Target Execution Model（CModel）

状态：2026-07-20已完成Q22.N/L/B/H/S/V、Q22 model-only functional-numeric profile、Q28标准7B单block scale vertical及
Q31多seed数值表征。Q32.I/R/B先重放当前已完成的v1 target-call/SystemC gate；mapped DMA、physical fill和
versioned GEMM orientation由已排期Q32.V typed target vertical闭合。`RequiredCapabilitySet`/package schema upgrade只在真实
execution consumer需要逐row preflight时从winner派生。model/board admission仍由本文拥有，Q22.C board correlation和
Q22.E exact-package execution保持独立later gate。封闭vendor Host-CRT/packet/DWFC seam不作为数值CModel依赖。本文固定以数值正确性为
近期目标的untimed target execution model、SystemC/TLM边界、实现分层和板端numeric correlation计划；timing calibration
仅作为deferred extension。任务状态看
`tasks/progress.md`。本文不复制总体架构、instruction schedule或完整register/
packet硬件事实表；第3节只给出用于界定模型claim的non-normative摘要。compiler、target、runtime和verification的
既有合同分别仍由`tasks/01`、`tasks/14`、`tasks/15`和`tasks/16`拥有。

本文把日常所称的CModel限定为**目标相关、非cycle-accurate的数值功能模型**：它位于accepted executable之后，以当前
target-call ABI、address space、engine和completion事实执行程序，并与独立source CPU expected及后续board结果比较。
它不因采用SystemC就自动获得vendor packet、hardware numeric、timing或cycle accuracy。

## 1. 目标和非目标

目标：

- 让同一份accepted rank program在RISC-V device link之外多一个目标相关consumer，尽早暴露instruction-to-target
  lowering、typed CRT call、地址、descriptor、engine和Direct DTE错误；
- 以SystemC作为正式target model的模块、并发和event容器；当前private memory/event实现不建立无consumer的TLM socket，
  TLM只保留给未来ISS/MMIO/interconnect consumer；近期只发布untimed/delta-cycle profile，不建立任意时间参数；
- 把算子数值、packet decode和checked memory effect保留为不依赖SystemC的plain C++ kernel，使其可做独立unit/
  differential test，并由SystemC modules调用；
- 已在任何f32 workload vertical之前建立覆盖当前全部13种logical storage format、有证据的target-profile×engine×format encoding、7种公开
  compute/convert format及完整product/accumulator/intermediate/destination、舍入/overflow/special-value政策的数值基础层；
  kernel只消费唯一`NumericSemanticsProfile`，不保留先写f32、以后再补dtype的旁路；
- 让Q22.L同一份fully legal target LLVM在host执行；由共享typed target-call registry和exact-signature bridge把
  实际`wafer_tx81_*`调用形成invocation-local `TargetTransaction`并进入SystemC，不重新lower或预构造整程序
  command vector；
- 当前Q22只消费已经完成的v1 TargetCall；CModel不读取Instr IR、planner candidate或index relation补猜事实。
  Q32.V启用mapped DMA、physical fill或额外GEMM orientation后，也只能消费最终versioned TargetCall中的address/domain、
  direction-correct descriptor和typed orientation字段；
- 保持source CPU oracle、target model和board执行独立，以differential定位模型实现和硬件行为差异；compiler lowering
  由逐层IR legality、conversion和artifact readback gate定位，不再维护accepted-IR第二套解释器；
- 对已由静态证据支持的功能/事务行为显式建模；未知numeric在执行前fail closed，未知queue/bank/timing不进入近期
  correctness claim；
- 用真实board single-op区分向量、重复随机held-out和source-backed workload校准numeric profile，并保留原始bytes/status、
  环境身份和evidence digest。

非目标：

- 不新增CModel dialect、instruction sidecar、packet list、shadow schedule或第二套package manifest；
- 不从op、buffer、symbol、文件名或trace文本恢复rank、resource、binding或transport语义；
- 不另建accepted-IR interpreter、numeric kernel或Direct DTE scheduler作为target model实现；
- 不把host原生`float`、oneDNN/Eigen等CPU kernel的默认累加/舍入/量化行为或单一外部库当作target numeric合同；
- 不把repo-owned target-call frontend称为CRT/packet执行，也不把经授权独立实现的host packet称为vendor-exact packet或RISC-V ELF执行；
- 不在vendor许可范围和可用事实源经项目owner/法务确认前修改vendor材料、实现其派生operator/packet builder或把当前
  reverse-engineering资料当作实现授权；
- 不把model completion称为board completion，也不在板端证据前声明hardware-bit-exact、performance-accurate或
  cycle-accurate；
- 不把model/board admission、qualification record或correlation结果反馈给planner生成、过滤或排序candidate；
  它们只控制对应execution consumer是否接受已经发射的程序；
- 不把PMU、latency、throughput、LT/AT或cycle calibration作为Q22或Q22.C的近期完成前置；
- 不让SystemC成为基础compiler、package parser或no-card runtime的强制依赖；target-model feature
  一旦启用则SystemC是正式模型依赖，且SystemC对象不写入compiler IR、ExecutableBundle或package。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  近期入口消费同一production compilation product中的Q16 ExecutableBundle及其经过tasks/14 target ABI preparation和
  full conversion形成的all-and-only、owner-backed fully legal target LLVM modules，以及与每rank entry精确双射的typed ABI slots。该target-LLVM
  bundle还必须消费`CompilationRequest -> ExecutionConfig`端到端携带的typed target profile，以及该
  profile由tasks/14 registry唯一映射的target identity/Kernel Runtime ABI。Q32 candidate selection不要求
  planner-side `RequiredCapabilitySet`；Q32.V也只在真实execution consumer采用逐row package preflight时增加
  capability-bearing package revision。当前profile不表示silicon revision，也禁止在full conversion后补default。
  bundle不可序列化，不进入package。exact-module扩展另行消费其已实现版本对应的Q18 VerifiedPackageManifest和
  all-and-only Q17 RISC-V ELF modules。Q32.V仅在真实consumer需要时增加final TargetCall capability projection及扩展package readback。
  CPU expected只作独立比较，不作为target model执行输入。
- Current stage responsibility:
  在任何执行副作用前，从reachable typed target-call rows验证当前v1 command、ABI、model kernel和
  `ModelProfileId`均明确支持；host materialize并执行同一target LLVM，由共享typed
  target-call registry和exact-signature bridge形成invocation/rank-bound `TargetTransaction`，交给SystemC
  tile/engine/memory/fabric modules执行rank/tile address spaces、CT/NE/RDMA/WDMA/TDMA、local completion和
  Direct DTE/FSM event。当前target-call ABI没有worker identity，近期模型只发布单一保守logical issue domain。
  Q32.V对mapped DMA、physical fill与额外GEMM orientation执行repo-owned exact-signature、
  formal/SystemC语义验证；采用扩展package capability集合的consumer再做all-and-only readback。external model
  provider或board preflight只决定该consumer能否执行已选程序，不返回planner、不过滤或重排compiler candidates。
  未来获准Host-CRT/packet和ISS frontend分别增加packet provenance与exact ELF证据。模型不重做sharding、candidate、
  layout、memory或transport planning。
- Output artifact / IR:
  invocation-local、owner-backed target model execution result，包含all-and-only rank terminal status、typed
  output tensors、model/profile provenance和可选typed diagnostic trace。结果不进入compiler IR、
  ExecutableBundle、TargetArtifactBundle或PackageManifest，也不作为后续planning输入。
- Downstream consumer:
  tasks/16 differential/CI和Q22.C board numeric correlation；Q22.E exact-module与deferred Q22.P timing calibration是
  独立consumer。target model结果不是compiler transformation输入。
- User-level driver / named pipeline:
  用户通过`wafer-compile --target-profile=<registered-id> --target-model`显式选择tasks/14 typed
  profile和model consumer，且不存在target profile默认值；当前发布baseline是closed v1。Q32.V扩展profile
  必须另行显式选择，并由同一Q32 production candidate owner消费。`--model-input`和`--model-expected`只使用调用者提供的
  固定source corpus payload/expected。wafer-compile在Q17/Q18原子发布后、同一invocation仍持有target LLVM bundle时
  执行host target LLVM，exact-signature bridge再向SystemC model发typed transaction/event。GEMM policy显式选择
  `formal`、`prefer-admitted`或`managed-reference`：`prefer-admitted`只能消费verified bulk qualification record；
  `managed-reference`只用于受管环境下的scale functional-reference gate，不签发Q22.B exact admission，也不改变target capability registry。
  wafer-opt/pass chain和component fixture只补局部测试。只有exact package/ELF执行闭合后，wafer-run才通过typed
  RuntimeProvider选择target model并消费verified package，不能用host-only模式冒充该入口。
- Explicit non-goals:
  不复制instruction schedule，不改变manifest语义，不用CPU expected驱动target model执行，不把untimed结果升级为
  board/timing/cycle证据，不用代表rank、手写LLVM、手写packet或单个kernel替代真实纵向链。
- Completion gate:
  Q0.L、Q22.N、Q22.L、Q22.B、Q22.H、Q22.S与Q22.V已经完成。Q22.N闭合13种logical codec、有证据的target-profile×engine×format encoding、当前7种
  compute/convert format、36条convert route、4种确定性舍入和逐family formal conformance；Q22.L原子形成all-rank
  owner-backed target LLVM bundle；Q22.B闭合首批F16/BF16/F32 exact-payload `profile-bounded` oneDNN admission。Q22.L
  直接解锁Q22.H repo-owned target-call frontend；Q22.N+Q22.H+既有Q16.T已由Q22.S闭合SystemC functional-event model，Q22.B不是Q22.S前置。
  Q22.B+Q22.S+既有Q20/Q21已由Q22.V重放Q20 f32、source-produced f16/bf16 GEMM、Q21 16-rank tiny Llama和
  deterministic source-backed large GEMM，覆盖all-and-only ranks、typed target call/ABI、SPM/DDR、
  supported engine和Direct DTE，并与独立source CPU oracle比较完整输出；任一late failure无partial result，model mismatch
  不回滚已验证Q17/Q18 artifacts。Q22只汇总model-only untimed functional-numeric profile；board numeric、exact
  package/ELF和timing accuracy分别由Q22.C、Q22.E和deferred Q22.P保持为更高独立gate。缺许可兼容vendor host seam或
  独立packet/MMIO事实源只阻塞可选packet provenance升级，不阻塞Q22数值功能模型，并禁止CRT/packet/vendor-exact claim。
  Q28在同一production入口上进一步闭合标准7B单block TP16、managed-reference bulk/tensor lane和完整PyTorch eager
  output differential；它是Q22之后的scale consumer，不扩大Q22 capability claim，且不证明board、exact package/ELF、性能或timing。
  Q32.I/R/B沿同一production入口重放当前Q22 v1 model gate；Q32.V另行闭合mapped DMA、physical fill、额外GEMM
  orientation及其typed target-call/SystemC纵向。只有声明执行这些扩展且采用capability-bearing package的
  model/board/exact-package consumer才必须消费Q32.V package set，不能回退为profile-wide
  隐式支持。repo-owned formal/SystemC gate属于Q32.V完成条件；external exact-package/board admission仍是
  Q22.E/Q6.B/Q22.C gate，且不参与compiler candidate生成或选择。
```

## 3. 当前事实基线

### 3.1 已有compiler和runtime边界

- Q16 `ExecutableBundle`已经拥有all-and-only static rank modules、typed program bindings、accepted memory/
  transport和terminal completion；target model不得重新选择candidate或重算跨rankbinding。
- tasks/14现有per-rank transaction-local target preparation已完成entry output/workspace/status ABI preparation，随后把
  instruction结构保持地lower为fixed `void(i64...)` target LLVM CRT calls；RISC-V device link在该lowering之后发生。
- 当前Wafer CRT header、lowering、source和checker形成109个production symbol的闭合surface。CRT源码通过
  `TsmNew*`/operator function table填写`Tsm*Instr`并调用`TsmExecute`，而不是直接写模型结果；symbol存在仍只证明
  ABI closure，不证明packet、numeric或hardware completion。
- 当前109-symbol/v1 surface中的plain GEMM没有orientation字段，模型只发布implicit normal/normal canonical row；
  RDMA/WDMA transaction也没有planner-local offset字段，offset若存在必须先由target lowering折入最终address。
  versioned oriented call及mapped-transfer Instr字段由Q32.V实现；在tasks/14/11落地并通过exact-signature gate前
  均不是当前model capability，也不阻塞Q32.I/R/B的current-v1纵向。
- Q17正式交付物是all-and-only RISC-V ELF `TargetArtifactBundle`；Q18 package只包含typed resource/slot/module/
  entry/completion/transport requirement，不包含instruction schedule。
- Q18当前只实现pure no-card `RuntimeSessionPlan`。真实`RuntimeProvider`生命周期和board执行尚未实现。
- accepted instruction/memory facts由target lowering和target model直接消费；不存在独立reference-only numeric或
  deterministic DTE policy可成为target model的旁路事实源。
- Q0.L shared registry显式枚举13种logical format，但engine×format准入只开放tasks/14有静态编码证据的row；
  UINT、64-bit和通用TF32 format-bearing command均在target effect前fail closed。`wafer.instr.convert`是独立例外：
  typed `InstrConvertKind`通过opcode 139..174的36条closed route选择含TF32的wrapper，但这不会打开通用
  CT/RDMA/WDMA/GEMM的TF32 row。frontend可接收的f64在target profile中也没有descriptor。Q22 numeric foundation应
  实现TF32 raw32 codec并分别记录logical codec、convert可达性和engine-specific format legality；不能用model有codec扩大
  compiler legality，也不能让f64晚到packet阶段才失败。

当前实现已经把tasks/14中完成ABI preparation和full target conversion的结果提升为owner-backed、不可序列化的
**target LLVM bundle**。factory-only `TargetCompilationProduct`同时持有accepted executable和生成Q17 artifact时实际消费的
同一target LLVM bundle；下游不能公开构造该关系，model也不能从package或accepted IR再次lower。它是fully legal LLVM IR及
typed slots的all-rank集合，同时被现有RISC-V link和repo-owned target-call/SystemC CModel消费；它不是新IR层、packet
artifact或package成员。具体C++类名和文件布局只作实现索引，不属于artifact合同。

repo CRT当前在源码内强制定义`USING_RISCV=1`。fresh host probe证明同一C源码和public header可由GCC完整编成x86-64
object，但只调用`wafer_tx81_local_fence`的最小链接首先缺`TsmWaitfinish`；完整object还要求36个Tsm factory/delete/
execute/wait入口和11个Direct-DTE/SPM platform入口。把现有`libinstr_tx81.a`加入会因其成员为RISC-V ELF
（EM=243、RVC、double-float ABI）而`file in wrong format`；`libcommon_util.a`、`liblibc_stub.a`和runtime archive同样不能
成为host provider。`op_fw_sim_if`的host CMake只是include-only INTERFACE target，checkout没有这些host定义。因此“同源”
当前只证明wrapper/command-invocation源码可host-compile，不证明host link或packet执行闭合。

以后若为可选packet provenance取得合法交付并获授权实施，应把platform选择移到互斥build contract：device定义RISC-V platform，host真正不定义
`USING_RISCV`，不能传`USING_RISCV=0`冒充false，因为vendor header使用`#ifdef`。RISC-V和host只共享经确认可使用的repo CRT
wrapper源码；host operator/MMIO/memory/context必须来自vendor交付的许可兼容库，或来自项目owner/法务确认允许且具有
独立可审计规范的clean-room实现。当前Direct DTE sender/receiver还是process-global static，host多rank执行不能共享该
状态；必须绑定invocation/rank或证明每rank独立装载实例及lifetime，不能从OS thread、symbol名或调用顺序恢复rank。

仓库随附vendor最终用户许可协议第2.1节把许可描述为有限、自用、不可转让/分许可且可撤销，第2.2节在购买凭证没有另行
约定时禁止修改、逆向、反汇编/反编译、提取源码和创建衍生作品。本文不判断具体采购条款；任何repo CRT/host Tsm/
packet provenance施工前必须由项目owner/法务确认当前材料的实际授权范围，或取得vendor书面许可/公开独立规范。
该external gate不再阻塞repo-owned target-call/SystemC数值模型；Q22.H不得消费这些vendor派生事实，Q22.K才可在以后
获准时增加独立CRT/packet claim。

Q32.I/R/B沿用已完成的v1 identity链；Q32.V在同一链上增加typed extension：
`wafer-compile --target-profile=wafer-tx81-single-card-kernel-v1`把
tasks/14拥有的唯一`TargetProfileId`写入`CompilationRequest`/`ExecutionConfig`，并贯穿
profile-bearing `ExecutableBundle`、target conversion、transaction-local prepared target LLVM/ABI artifact、
`TargetArtifactBundle`和PackageManifest readback。`TargetLLVMModuleBundle`再原子拥有canonical all-rank domain、
每rank logical rank/entry/module、ordered typed ABI slots、该validated identity及全部LLVM context lifetime；所有rank完成
ABI preparation、full conversion和readback后才能交给device link或host model。任何consumer都不能从default、
自由字符串manifest或host环境恢复target facts。Q32.V只在该identity链上增加扩展capability/package验证，
不改变v1身份选择，也不把model/board admission变成planner输入。该profile/identity、Q22.N numeric和Q22.L bundle gate
现已闭合。

### 3.2 已确认的硬件功能和事务事实

当前静态证据足以支撑以下受支持子集：

| 范围 | 已确认事实 | 初版可声明 | 仍不可声明 |
| --- | --- | --- | --- |
| topology/memory | 当前目标单卡4×4、16 tile；每tile 3 MiB SPM，末64 KiB有Kcore/runtime占用证据；另有SPM alias和多个consumer-specific DDR mapping/address-view线索 | 当前profile的rank/tile隔离、SPM reservation和byte-level model window；DDR只按typed resource注册 | 所有SKU容量、bad-tile/PG行为或单一静态DDR范围可硬编码 |
| physical layout | compiler当前accepted compact、Cx/NCx规则可由typed layout重算；Cx/NCx是block-major而不是普通dense stride，INT8/UINT8 full block为128 lane、其它格式为64 lane，并存在compact C0 tail；hardware helper另有256B alignment/footprint证据 | source-backed accepted layout的logical coordinate到physical byte offset及越界检查 | Cx/NCx可直接表示成oneDNN stride；名称或256B规则是所有dtype/op的通用硬件layout |
| dtype/storage | `Data_Format`公开13种有效格式：INT8/16/32、UINT8/16/32、INT64/UINT64、FP16/BF16/FP32/TF32和bitpacked BOOL；storage width为1/2/4/8 byte，BOOL为1 bit/element | 13种logical raw codec；tasks/08拥有bit/byte footprint和layout geometry，tasks/14只对有证据的target-profile×engine×format发布ABI/register encoding legality | enum存在即证明每个engine都能搬运或计算该dtype；当前DMA helper只有0..7编码却把8..12当作已支持；把BOOL scalar helper的byte store当作bitpacked硬件合同 |
| compute/convert | 当前CRT公开INT8/INT16/INT32/FP16/BF16/FP32/TF32七种格式间36条convert route；代码检查显示typed convert lowering可选择含TF32的wrapper，但还没有target-lowering integration正例；`RND_MODE`有nearest-even、zero、positive-infinity、negative-infinity、stochastic五种值 | 七种格式的logical numeric descriptor、36条route结构和四种确定性rounding的model candidate；每条op另按完整语义profile和engine-specific format legality准入 | 通用format-bearing op已能发射TF32；code-reachable convert即已被集成验证；UINT/INT64 compute、任意op×dtype笛卡尔积、INT8-source zero-point公式或stochastic随机状态已由ABI证明 |
| packet/ABI | CT、NE、RDMA、WDMA、TDMA packet字段、worker window、trigger和range/end字段已有证据，但近期frontend不消费vendor packet | typed target-call descriptor/range conformance；packet只作Q22.K future oracle | repo-owned target-call已验证packet字段、worker或vendor CRT一致性 |
| movement | contiguous/strided RDMA/WDMA、基础TDMA/gather-scatter/memset的混合单位和descriptor字段已恢复；element count、byte stride、iteration-minus-one及byte-count字段按family区分 | 受支持descriptor的功能执行和逐字段单位检查 | 把所有count当byte、所有alignment/stride组合的性能公式 |
| local issue | 每tile三个worker window；`serial_mode=0`语义把accepted CT/NE/RDMA/WDMA/TDMA分成五个逻辑NCC issue class并存在range/busytable依赖，但target-call ABI没有worker字段 | 单一保守logical issue domain内区分五个engine family；local drain与event ordering分离 | worker identity、`3×5`物理queue实例、provider逐worker配置、queue容量、多发射、精确仲裁和worker物理独立性 |
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
2. **host runtime级**：fresh `RTLD_NOW` probe在x86 `libtx8_runtime.so`上首先因`libhpgr.so`缺失而失败；其dynamic
   dependency还要求缺失的`libtsmml.so`。该runtime的`Runtime::SetCModelHandle`会尝试`dlopen`
   `libcmodel_runtime_api.so`并解析device、compile、launch、run、copy和tile-info等高层入口。被加载的library、其匹配
   header/resource以及该binary还依赖的host libraries均不在checkout；现有binary只证明`dlsym`结果被存入字段且library
   handle会被`dlclose`，没有证明普通launch路径读取/调用这些字段。因此这是vendor CModel存在的强线索，不是当前
   可运行provider，更不能证明其内部
   使用SystemC或能消费Q17/Q18 artifact。

附带`libinstr_tx81.a`、`libcommon_util.a`和`libkcorert.a`都是RISC-V object，不能链接进host model；当前vendor tree和
x86 runtime的公开依赖/symbol中也没有SystemC/TLM痕迹。SystemC仍可能封装在缺失的CModel library内，故结论只能是
“实现技术未知”，不能写成vendor已确认使用或不使用SystemC。

以后恢复Q22.K或Q22.E前应向vendor索取完整`libcmodel_runtime_api`套件或低层host instruction model、匹配headers/resources、
支持的target revision、artifact输入、SystemC及其它transitive dependency版本/license、numeric profile、threading/time
contract和可重放要求。高层套件只有证明能消费当前verified package/session后才作为typed `RuntimeProvider`；否则只作
外部differential oracle。低层套件只有在其许可允许当前集成方式后才在Tsm/packet/MMIO边界接入本文架构。两者都不得
绕过capability、differential和board correlation gate，也不能用当前EULA覆盖的材料自行派生缺失实现来替代交付。

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

模型保留两个互不冒充的近期/远期执行入口，以及一条可选packet/MMIO provenance路径：

1. **Repo-owned target-call/SystemC frontend**：host LLVM JIT执行同一fully legal target LLVM module；共享typed
   target-call registry验证每个reachable `wafer_tx81_*` exact signature，per-rank exact-signature bridge显式携带
   invocation/rank context，并把每次真实动态调用形成typed `TargetTransaction`投递给SystemC model。这是近期正式
   functional-event CModel路径。它不编译repo CRT、不构造Tsm packet，只声明target-call/ABI provenance。
2. **Exact-module frontend/provider**：从Q18 verified package加载并执行Q17 RISC-V ELF。project ISS adapter通过loader
   ABI、MMIO/custom instruction、Direct DTE和completion adapter复用SystemC architecture；独立vendor simulator不要求
   内部使用SystemC，但必须满足相同artifact、provider lifecycle、capability和evidence合同。只有此入口完成后，才可称
   package-facing target model provider。
3. **Optional authorized CRT/packet/MMIO evidence**：取得owner-approved vendor host package或独立公开规范后，可让同一
   target LLVM经过获准repo CRT/host Tsm seam，或消费exact packet/register transaction，与repo-owned target-call路径的
   typed command、address/engine和observable memory effect比较。positive必须来自exact Q17 ELF经ISS执行
   真实CRT/archive产生的register trace、board capture，或vendor提供且版本可审计的builder/CModel；手写packet只补
   negative。缺少这条证据不阻塞数值模型或板端数值相关，但禁止`vendor-exact packet`声明。

这些路径可以发现不同错误；任何一条通过都不能冒充其它证据。测试和diagnostic必须记录实际frontend、CRT/packet
provenance、capability、numeric profile和event profile。

当前fixed target-call ABI没有额外model-context参数。host clone为每个reachable call生成exact-signature bridge，在bridge
body中显式嵌入其invocation/rank context并调用唯一typed dispatcher。完整symbol只作exact ABI-key lookup；SystemC
yield后不从前后缀、参数数量、TLS、OS thread或调用顺序恢复语义/rank。bridge、JIT和context共享invocation-local lifetime；
机制必须覆盖并发rank、阻塞yield和失败锁存，reentry/nested call在rank运行态显式拒绝并整体abort。

target LLVM中的i64地址参数按typed ABI角色分别表示NCC-visible tile-local SPM offset/address、compiler-planned DDR
arena base加offset，或
external DDR/resource的invocation-time registered device address；它们不是可任意解引用的host pointer，也不能被压成
一个固定DDR范围。host frontend只能通过typed ABI slot、resource identity和checked address registry解析；未知base、
overflow、跨resource访问或把SPM alias当host虚拟地址都必须在memory effect前拒绝。

多数target call返回`void`。host frontend使用invocation-local error latch，在entry结束、wait和all-rank terminal前汇总
registry、address、engine和event错误；该latch属于model diagnostic，不能伪装成target ABI中原本不存在的return。
Direct DTE event仍按原ABI返回opaque `i64`，失败先锁存错误再返回不可发布的dummy identity。板端exception/status仍作为
独立证据源。

### 4.2 SystemC架构和plain C++ kernel

正式模型按稳定硬件责任拆分为SystemC modules/channels：

- model top/context：target profile/identity、当前profile的16个physical tile slots、capability/good-tile map、logical/physical
  rank/tile mapping和invocation-local状态；
- tile memory：每tile SPM、reserved region、card DDR/resource slots和checked address translation；
- 每tile单一保守logical issue domain，按typed call明确区分CT、NE、RDMA、WDMA、TDMA engine family；当前target-call
  ABI不携带worker identity，因此不建模或声称三个worker window、`serial_mode=0`、`3×5`物理queue或engine复制关系；
- tile-level CT/NE/RDMA/WDMA/TDMA engine endpoints及保守共享resource arbitration；
- Direct DTE fabric/FSM：rank-local endpoint/status、receiver-ready、source read/lifetime、send/recv/wait/release和destination
  visibility；
- completion/result：issue、local drain、destination visible、all-rank terminal和typed no-progress diagnostic。

当前target-call ABI没有worker字段，静态证据也没有证明queue depth、同queue多发射、cross-worker arbitration或SPM bank
函数。第一版只能建立untimed/delta-cycle functional-event模型，对未知共享resource保守串行；不能把任意`sc_fifo`
容量、worker数或调度顺序升级成微架构事实。worker/packet行为只在以后Q22.K取得合法packet evidence后另行建模。

typed transaction检查、地址检查和各engine的numeric/memory effect由不包含SystemC header、不链接SystemC的plain C++ kernels
实现，并可独立做unit/property/differential test。初版采用model-only保守observable-commit policy：kernel在SystemC
event确定的完成点一次提交可观察memory effect，不能在target-call issue时就让结果可见；这不宣称硬件没有partial
write或相同visibility时刻。它们不与compiler-side consumer共享compute kernel、rounding policy或DTE scheduler。

SystemC调度粒度是一次target transaction/command，不是tensor element或单个MAC。GEMM transaction在plain kernel内一次
调用admitted oneDNN primitive并返回待提交buffer effect；绝不能为M×N×K循环创建`sc_event`、process或TLM transaction。
formal scalar loop只在checked work budget内服务小规模conformance、edge profile和失败诊断；大command超过budget且无admitted
bulk row时必须在compute前返回`bulk_backend_unavailable`，不能以隐式scalar fallback拖垮大网络。

typed target command只是一次真实target-call bridge产生的瞬时transaction；bridge必须在返回前把异步处理所需的
typed arguments复制进transaction，绝不能保留caller栈指针。该复制不授权在Direct DTE issue时snapshot payload；DTE
source具体读取时刻仍由model profile和后续board/vendor correlation决定。transaction不预构造整程序command vector，不进入
compiler artifact或package，不能成为长期shadow program。SystemC modules读取transaction执行时状态，不复制compiler
completion DAG或planner trace。exact ELF路径必须经过ISS产生raw packet/register transaction再进入decoder，不能由ISS
hook直接合成高层command绕过packet证据。

### 4.3 实现对象和调用边界

以下名称是实现索引，不是新的IR、package协议或稳定public ABI：

| 对象 | 必须拥有 | 禁止拥有 |
| --- | --- | --- |
| same-lowering compilation product | factory-only ownership relation，原子持有accepted executable及直接生成Q17 artifacts的同一target LLVM bundle | 从package/readback重建bundle、公开构造伪造同源关系或第二次lowering |
| `TargetLLVMModuleBundle` | 当前Q22拥有canonical all-rank domain、每rank独立LLVM context owner、logical rank/entry/fully legal module、ordered ABI slots、`ExecutionConfig`（内含唯一`TargetProfileId`），以及由module metadata经closed registry解析/readback的target identity/kernel ABI；Q32.V可在同一owner上增加per-rank capability projection和all-rank set digest | packet list、schedule、model state、从symbol/metadata digest反推typed facts、默认补出的revision或任何可序列化sidecar |
| `RequiredCapabilitySet`（Q32.V按真实consumer需要引入） | 从final instruction/TargetCall rows经tasks/14 owner派生的canonical sorted unique row keys及digest；采用capability-bearing package的model provider以显式`ModelProfileId`、board provider以显式environment在effect前逐key匹配 | Q32 candidate输入、无consumer时的Q32.V前置、command地址/顺序/次数、implementation/residency/movement choice、capability lease、由profile名猜出的隐式全集 |
| compact program invocation | typed `ProgramTensor`、完整logical user input、accepted per-rank slice和已验证package-relative parameter/constant payload | reference compute、model compute、target physical address、从文件名恢复resource role |
| `ModelProfileId` | 显式选择的一组确定性model-only semantics identity；Q22.C可另发布绑定target/environment的hardware-correlated profile | compiler legality、隐式default、hardware等价声明 |
| `NumericCommandKey` | static preflight和runtime从shared typed target-call registry/transaction投影的target profile/Kernel Runtime ABI、engine/op kind/variant、operand role/storage dtype、shape/layout、M/K/N/batch、convert kind及fixed optional fields；Q32.V target call另加入typed lhs/rhs orientation | erased Instr sidecar、任意symbol/string推断、numeric comparator、oneDNN选择、板端阈值 |
| `NumericSemanticsProfile` | 稳定typed identity/digest；完整operand storage/compute、product、accumulator、intermediate、destination、rounding points、FMA/reduction order、overflow、FTZ/DAZ、NaN/special/status及optional-field顺序 | 单个dtype、host library默认值、按workload临时覆盖的side table |
| `TargetModelCapability` | `(ModelProfileId, NumericCommandKey)`到唯一`NumericSemanticsProfile` identity的映射，以及shape/layout/descriptor、event/transport、compiler-emittable和hardware evidence状态 | 重复保存压缩的src/accum/dst key、由symbol存在推导支持、扩大compiler legality的规则 |
| `BulkBackendAdmission` | 完整semantic profile digest、shape/layout adapter、value domain、backend environment和已签发的`bit-exact/profile-bounded`类别；`rejected`表示不产生该对象 | target numeric semantics、target comparator或“library支持该dtype” |
| `FormalNumericExecutionContext` | non-yielding formal kernel作用域内的profile、model status与MPFR state save/set/clear/capture/restore和显式target status映射；APFloat/APInt每次调用显式传rounding | rank identity、跨SystemC wait的全局/TLS状态、SoftFloat oracle状态或oneDNN worker状态 |
| `BulkExecutionEnvironment` | oneDNN runtime/threads/ISA/implementation、当前SEQ caller-worker control readback、caller fenv恢复和admission provenance；未来非SEQ profile再要求逐worker initialization evidence | worker native flags到target status的映射、architectural memory或SystemC API |
| prepared target-model invocation | 两个same-lowering bundle引用、all-rank ordered slot value、按slot layout编码的read-only physical bytes、checked non-overlap address plan和prepared target-call executable | host pointer伪装的device address、从文件名恢复rank/resource、别名source NPY storage |
| `TargetTransaction` | exact-signature bridge返回前复制的typed call kind/ABI revision、engine、最终地址/descriptor、dtype/shape/optional fields、explicit rank context和invocation-local sequence identity；Q32.V transaction按versioned ABI增加orientation等扩展字段 | Instr local-offset字段、planner index relation、packet/worker字段、caller栈指针、整程序command vector、shadow schedule或DTE payload的无证据snapshot |
| `TargetModelResult` | all-and-only rank terminal status、完整output、numeric flags、transaction/thread/delta计数、formal/bulk command计数、MatMul/reorder/formal-FMA evidence和record digest | partial successful rank集合、回写compiler/package的状态 |

`NumericCommandKey`只包含typed target-call registry/transaction已经表达、并由typed target profile唯一解释的事实。
当前v1没有orientation字段；Q32.V启用额外GEMM orientation时，orientation和ABI revision必须参与
key/pattern/resolution/admission digest，不能与normal/normal记录共用。它
不保留被erase的Instr program或shadow command list。`NumericSemanticsProfile`是所选`ModelProfileId`下这些事实对应的唯一
完整数值解释；`BulkBackendAdmission`只决定该解释能否由host bulk library
加速。三者分离后，target comparator、板端校准政策和oneDNN环境都不会反向污染硬件语义。实现支持谓词是
`compiler accepted tuple ∩ model kernel tuple`；板端profile仅在Q22.C进一步收窄，不能扩instruction legality。enum、target-call
symbol或future packet decoder存在都不等于row已支持。capability不能压成单个布尔值，至少保留三个正交维度：
`model-implemented/absent`、`compiler-emittable/not-emittable`和`model-only/device-unit-observed/revision-correlated/
hardware-calibrated/rejected`；每个未闭合维度都携带reason。这样13种storage codec可以一次实现，同时TF32当前compiler
缺口、UINT/INT64 compute缺口和未校准硬件行为仍然如实可见。

该映射在实现中拆成三层：`NumericCommandKey`是一次dynamic command的exact facts和
key digest；`NumericCapabilityPattern`是有限、可重用且可证不重叠的typed selector/shape/layout
constraint，携带三能力轴、semantic/kernel/comparator/backend identity和pattern digest；
`ResolvedNumericCommand`原子绑定exact key与唯一pattern，并用resolution digest引用model policy、key、
pattern和semantics digest。`NumericSemanticsProfile`不持有exact key；executor只消费resolved object。
禁止使用任意predicate+优先级、最佳匹配或shape枚举建立无界profile集合；missing、duplicate或overlap必须在
effect前失败。dependency/actual-loaded-binary identity进execution provenance digest，不混入纯numeric semantics digest。

映射必须满足以下唯一性规则：如果accumulator、FMA、逐步rounding、quant或optional-field行为可由程序选择，先扩typed
tile/instruction IR和Wafer CRT ABI；如果它是target固定隐含行为，则每个显式`ModelProfileId`与完整command tuple必须恰好
映射一个semantic profile。板端校准前可并存多个命名model profile，但每次execution必须显式选择并记录一个，
不能进入compiler legality、隐式default或CModel side table。Q22只发布确定性model-only profile；Q22.C再把board evidence
绑定到target revision/environment，选择或拒绝对应row并发布hardware-correlated profile。当前v1 plain GEMM只有
normal/normal、单一format且要求lhs/rhs/dst同element type；Q32.V的v2每个lhs/rhs orientation tuple是不同exact key，
但仍由一个parameterized pattern family覆盖，不建立四套kernel。f16/bf16 narrow/wide、TF32 product和i8 accumulator等分别属于不同显式model profile，
不能在同一次execution内按数据猜测切换。这里的model/board row只控制execution admission，不能反馈给planner选择candidate。

`TargetLLVMModuleBundle`在所有rank完成ABI preparation、full conversion、LLVM translation和module readback后原子形成；每rank
module已经由独立LLVM context拥有。host执行后续把每rankmodule克隆/retarget到独立ORC `JITDylib`，避免同名entry/private helper碰撞；JIT clone设置native triple/data layout，
并为shared typed registry确认的每个reachable call生成exact-signature bridge，不能依赖当前process偶然导出的符号。host retarget前拒绝target-specific
intrinsic、inline asm、未知address space或其它不能安全host materialize的LLVM结构；用于RISC-V link的module保持不变。
由于entry是动态slot数量的`void(i64...)` fixed ABI，JIT层生成统一签名的host-only thunk，例如
`void(const uint64_t *slots)`，按已验证的完整ordered typed slots加载并调用原entry；禁止把可变参数函数指针强转调用。
所有rank JIT/thunk/bridge materialize成功后才发布
model executable。

正式entry必须在SystemC可yield process中调用，使local fence、DTE wait和FSM receive可等待`sc_event`；所有rank process
先创建再启动。invocation/rank context由per-rank exact-signature bridge显式绑定，不能只靠TLS或`sc_process_handle`推断。
首个正式入口在每个driver进程的initial elaboration中建立一次invocation-local model，不调用`sc_stop`，由`sc_start`运行到
process/queue/event quiescent后销毁私有memory和error latch；需要重复运行时重新启动driver进程。同进程长寿命
`ModelSystem`、重复或并发invocation只有在persistent process、context隔离和reset测试另行闭合后才能开放，不属于Q22完成能力。

执行前structural/capability preflight必须一次枚举并拒绝：任何reachable target-call symbol/signature缺typed bridge；任何静态
可知的op/dtype/shape/optional tuple无kernel或comparator；address plan、ABI binding或rank/transport endpoint不闭合。
preflight通过后才在私有invocation backing中导入input并创建SystemC运行态。动态
descriptor和computed address必须在每条transaction的read/effect前验证；失败整体丢弃私有invocation且不发布output，
但不虚构一次不具备通用性的dry-run来声称它们在input import前已知。plain C++ `NumericKernelRegistry`按完整typed key精确
查找，禁止按symbol/op名字fallback。

memory分为只含metadata的`InvocationAddressPlan`和通过plan后才建立的私有`InvocationMemoryRegistry`。registry为每个
typed ABI resource、workspace/status和rank-local SPM分配不重叠的synthetic device range；entry只得到这些device address。
resolve必须显式携带logical rank、address space和read/write role；返回值给出唯一slot/resource identity并证明checked range
完整落入唯一region，Direct DTE的tile identity另由typed endpoint绑定，不能按地址数值阈值猜SPM/DDR或直接解引用host pointer。
每条transaction先读完整snapshot，plain kernel返回待提交byte effects；
所有range/payload通过后，SystemC completion event才一次提交。单command错误无partial write，整次invocation只有all-rank
terminal success后才extract/copy output。

当前Q22 v1 RDMA transaction只执行strided DDR source到从最终destination address开始的sequential SPM bytes，
WDMA严格反向。Q32.V启用mapped DMA后，模型仍不增加“layout conversion”分支；Instr两端
root-relative offset必须先由target lowering折入最终source/destination address。plain kernel只按最终descriptor检查
payload、source/destination end、overlap、snapshot和canary，不接收logical index map。若compiler错误地把需要双侧
strided的关系编码成DMA，target-call/memory oracle必须暴露byte差异或range错误，不能用planner relation修正结果。

当前target-call/SystemC executable preflight直接枚举reachable typed calls，并对每个command查询fresh
`(NumericCommandKey, ModelProfileId)` model record；它不需要package capability set。Q32.V package consumer
才对manifest中每个required capability查询同版本record，旧版本record即使其它字段相同也不匹配。任何missing row都在
memory import/effect前拒绝，board row完全独立。

later Q3.6 Count当前只有opcode/wrapper和raw low-u32 writeback线索，尚未定义source predicate、format-specific numeric
semantics或独立golden，因此current source lowering、target model和board admission都必须在任何effect前拒绝，不能猜成
count-nonzero或从symbol/opcode名字恢复语义。

Q3.6若恢复实施，只建立真实consumer需要的typed纵向：

- Instr/TargetCall显式携带source、single-element i32 destination、element count、format和synchronous-writeback
  effect/completion；target conversion从typed operands验证range、alignment、disjointness和checked byte count；
- test-only mechanical adapter可以验证raw u32 little-endian store、canary、late-failure atomicity和local
  compute/movement ordering，但它不签发numeric、source或board资格；
- 只有source IR明确表达Count predicate，且有独立golden/formal-SystemC evidence和实际model kernel时，才在普通
  typed model capability table中增加对应row；board仍由独立environment row验收；
- 不预先定义独立的versioned capability、qualification或package协议；若真实package/model consumer以后确需
  跨进程capability projection，另按tasks/14-15的later边界同批设计；
- compiler candidate生成和Q32 selection永远不读取model/board admission结果。

ArgMax/ArgMin现有wrapper在destination store前等待local worker的事实可以复用于实现同步writeback effect，但不能
外推Count predicate、format支持或numeric semantics。

Q32.V oriented GEMM formal kernel按stored shape和typed fields取数：transpose lhs使用`stored[k,m]`，transpose rhs使用
`stored[n,k]`，destination仍写`stored[m,n]`；batch leading dimension保持显式。oneDNN adapter可以用logical strides/view
或受验证reorder实现同一关系，但orientation、stored shape、payload/destination template和adapter descriptor都进入
admission identity。已有normal/normal exact qualification record对NT/TN/TT不匹配；managed-reference也必须逐command
检查orientation，而不能因oneDNN支持transpose就扩大target capability。该admission只允许model执行已经选定并发射的
command，不参与planner candidate选择。

source invocation先保留compact row-major program tensor；prepared model invocation再按每个exact Kernel ABI slot携带的
shape/dtype/layout调用shared physical tensor codec，编码compact/Cx/NCx target bytes并核对physical footprint。output按同一slot
反向解码，driver按program output binding的global/local slice比较all-and-only rank。program-boundary BOOL的NPY byte表示与
target bitpacked表示尚未建立无歧义source合同，因此compact BOOL input当前在装配边界fail closed；这不影响target-call内部
i1临时值、select和bitpacked physical effect已经支持的事实，也不能用内部支持反向宣称BOOL program input已开放。

### 4.4 完整数值类型系统

首版先消费Q0.L由tasks/14单一拥有的typed format registry，再建立codec/数值类型系统并让GEMM、reduce、elementwise和
convert等kernel消费；Q22不另建registry，也不允许每个kernel自行解释`format`或让首个f32 vertical形成默认语义。
logical format、layout geometry与target format encoding必须分层；同一logical format在不同engine、layout或target revision
上可能有不同编码或根本不合法。数值基础层消费以下对象：

| 对象 | 必须显式表达 | 不能表达为 |
| --- | --- | --- |
| `LogicalFormatDescriptor` | format identity、raw container与semantic bit width、signedness、exponent/significand、canonical encoding、NaN/Inf/subnormal能力 | C++ `sizeof(T)`、CRT enum值范围、physical block或host native type |
| `TargetFormatEncodingRecord` | typed target profile、engine、logical format、public ABI enum/register code、format-specific constraint和evidence状态；实际layout另作typed command fact由tasks/08 helper验证 | 一个对所有engine通用的`Data_Format -> code` switch、Cx/NCx几何副本或logical format名称 |
| `NumericSemanticsProfile` | 每个operand的storage/compute type、product、accumulator和intermediate、destination、每个rounding point、FMA/reduction order、overflow/saturate/wrap、FTZ/DAZ、NaN/special/status、zero-point和stochastic policy | 单个`dtype`、symbol后缀或全局rounding flag |
| capability row | `ModelProfileId`、`NumericCommandKey`、唯一semantic profile identity、target format encoding及shape/layout/descriptor/optional-field范围、evidence status | “library支持该dtype”、压缩src/accum/dst tuple或13种格式的笛卡尔积 |

storage层从第一天覆盖13种logical format。INT8/16/32、UINT8/16/32、INT64/UINT64和FP16/BF16/FP32/TF32按公开storage
width读写little-endian raw bits；BOOL的physical bit order只在有证据的encoding profile中确定。TF32的32-bit container与
semantic precision分开，所有codec必须定义noncanonical低位如何拒绝或规范化。movement只复制raw bits，不经过host浮点值。
Cx/NCx block/tail/footprint、BOOL bitpack和alignment继续由tasks/08及唯一`computeWaferPhysicalTensorInfo`拥有；
`TargetFormatEncodingRecord`只定义engine/register legality及format-specific constraint，exact command中的typed layout
必须另行通tasks/08 helper验证并与record constraint交叉，不能让任一侧复制几何。当前DMA helper的0..7编码
证据不足以证明UINT8/16/32和INT64/UINT64对应8..12可直接搬运；这些
engine×format row在共享registry闭合前必须target-illegal，即使logical codec已经存在。

compute层从第一天为当前公开七种format建立完整descriptor，并把36条convert route作为一个受类型检查的surface，而不是
36份手写kernel。四种确定性rounding mode必须端到端生效；stochastic虽然进入profile和capability，但在取得target
seed/state/reset/advance合同前只能选择显式`model-only`政策或拒绝hardware-correlated执行。INT8-source四条zero-point
route同样保留显式字段和候选profile；没有公式证据时不能静默采用常见的`scale * (x - zp)`。convert的TF32 typed route与
其它engine的TF32 format encoding分别准入，不能互相证明。UINT/INT64当前只承诺logical storage codec；movement和compute
均按engine-specific capability决定，不由enum存在推导。

36条route必须从现有typed `InstrConvertKind`定义生成，并保持参数类别闭合：4条INT8→FP16/BF16/FP32/TF32携带
zero-point；9条widening/已声明plain route不消费额外参数；其余23条消费rounding mode。固定ABI中存在但该route不消费的
参数必须验证为ignored-by-contract，不能意外影响结果；新增route必须先更新唯一typed定义和closure test。float-to-int
还必须在profile中显式定义NaN、±Inf和overflow结果；float-to-float显式定义sNaN quiet、payload/sign传播、tininess
before/after-rounding和exception status。SoftFloat/MPFR的默认policy不得补齐这些字段；target ABI不能观察的status先保持
model-only evidence，不伪造成硬件可见状态。

算术kernel必须使用同一个`NumericSemanticsProfile`执行：

- integer逐步定义signedness、widening、overflow、saturate/wrap/status；禁止依赖C++ signed overflow；
- floating逐步定义输入规范化、乘法/加法/FMA精度、accumulator、reduction order和每个rounding point；
- GEMM/reduce不能因为host库默认使用f32或s32 accumulator就省略target accumulator；
- tanh、exp、rsqrt等超越函数先从高精度结果按profile舍入，host `libm`结果只能作非规范性fast-path候选；
- NaN分类/payload/sign、Inf sign、signed zero、subnormal/FTZ/DAZ和异常status均是独立字段，不能交给普通`isclose`。

首批formal backend不能只注册f32。至少实现并区分以下model candidate：f32乘加/累加/写回f32；f16和bf16分别以窄类型
或f32累加后写回原类型；TF32 operand以显式TF32 product policy、f32 accumulator写回TF32或f32；i8乘法、i32累加并以
i32 component result验证，量化写回另需显式quantization profile。它们是用于compiler验证和板端区分实验的候选，不是
硬件已确认矩阵。当前`InstrGemm`只验证lhs/rhs/dst同element type，target ABI也只传一个format且没有accumulator字段；
若target profile不能唯一固定隐含accumulator，就必须扩typed IR/ABI表达，不能把候选选择放进CModel side table或从dtype猜。

### 4.5 数值库分工和独立性

不存在一个现成库同时覆盖Wafer全部格式、量化、累加、超越函数和未知硬件edge behavior。采用“正式标量语义后端 +
受准入约束的bulk backend + 逐family独立differential或显式trusted-TCB gate”的组合：

| 组件 | 本项目角色 | 边界 |
| --- | --- | --- |
| LLVM APFloat/APInt | 受管LLVM pin内的正式基础标量后端；覆盖FP16/BF16/FP32/TF32、四种确定性rounding、逐op status、convert、FMA及无C++ UB固定位宽整数 | 不提供所需sqrt/exp/tanh/rsqrt等超越函数；同一APFloat实现不能自计第二oracle |
| Berkeley SoftFloat/TestFloat | FP16/FP32独立IEEE differential和adapter/build conformance；`testsoftfloat`以包内不同slowfloat实现检查SoftFloat，BSD-3-Clause便于受管引入 | 不进入production kernel，不覆盖BF16、TF32、超越函数或Wafer optional-field政策；普通`testfloat`以SoftFloat作expected，不能再多算一个oracle |
| MPFR/GMP | tanh/exp/sqrt/rsqrt等高精度formal production backend，并为BF16/TF32基础路径提供高精度differential/TCB；destination/temporary使用显式precision，operation显式传rounding并设置exponent range | production使用MPFR时同一wrapper不能再算独立oracle；MPFR只有一种NaN且不原生模拟目标subnormal/payload，必须由raw codec/profile包裹；依赖和LGPL合规在引入前固定 |
| oneDNN | Q22中已准入大规模GEMM/MatMul的主执行后端；source-backed大矩阵不能长期用scalar逐MAC执行 | primitive、ISA、accumulation及math mode会影响结果；strict/deterministic不证明target等价；它不拥有Wafer codec、Cx/NCx、rounding、quant或reduction-order语义 |

Eigen、gemmlowp或host `libm`可以加入非规范性performance/differential实验，但不进入首版语义owner：Eigen fast-math和
vectorization会改变边界行为，gemmlowp只覆盖低精度GEMM且带自己的quantization合同。oneDNN则不是“以后再做”的优化：
Q22首版必须为实际source-backed大GEMM提供admitted bulk path；依赖必须版本固定、license可审计，基础compiler和no-card
runtime不因此链接这些库。

oracle独立性按算术实现和target adapter/codec的来源判断；同一library换precision、driver、executable或随机seed都不形成
第二个oracle。逐family最小矩阵是：APFloat production FP16/FP32与独立SoftFloat adapter比较，`testsoftfloat`的slowfloat
路径验证SoftFloat本身；同样使用APFloat的另一条执行路径不能算独立oracle。APFloat production BF16/TF32与MPFR高精度结果及
test-only独立整数/raw-bit rounder比较；MPFR production transcendental以已知点、
metamorphic relation、version/digest/self-test和wrapper验证闭合trusted-TCB gate。这里的self-test至少绑定exact MPFR/GMP
source/build digest和configure options，在干净依赖build上分别执行上游`make check`，保存command、exit status、version、
config summary及test-suite logs；随后执行项目wrapper known-point/metamorphic/raw-codec tests。上游check只证明受管依赖
build conformance，不算第二算术oracle。另一实现或后续board作为升级证据；
MPFR本身标记trusted semantic TCB而不是“双软件oracle”，结果只声明trusted MPFR semantics；oneDNN bulk与
formal APFloat loop逐row比较，qualification corpus按format另以SoftFloat或MPFR检查关键区分向量；不能因再次调用APFloat
算第三路。target codec由test-only逐bit mapper产生expected，不能让production codec自证。target model只共享稳定dtype enum、
physical geometry、packet字段和ABI常量，不共享其它执行consumer的compute/rounding/codec helper或DTE scheduler。

### 4.6 oneDNN bulk准入、layout和执行生命周期

本节先区分当前已实现合同和以后新增profile时的约束。Q22.B当前只签发F16/BF16/F32同dtype GEMM的有限payload
`profile-bounded` row：三个格式经target-owned codec/layout精确提升为F32，oneDNN只执行一次F32 MatMul及最多一次
weights reorder，随后由formal GEMM finalize舍入回原格式并target-pack。当前不签发`bit-exact`、TF32、integer、跨payload
domain或timing/performance资格。下面关于解析/穷举证明、非SEQ worker pool、packed-weight/project cache、TF32/int8和
performance-qualified host profile的规则都是新增对应能力时必须另行实现和验证的扩展门槛，不能反推为Q22.B当前事实。

`NumericBackendPolicy`只决定如何执行已确定的`NumericSemanticsProfile`，不属于compiler IR或硬件语义。每条
`BulkBackendAdmission`由semantic profile、K/shape/value domain、oneDNN build/runtime、ISA、actual implementation、thread和
worker environment共同定址；同一semantic profile在不同shape或environment可以有不同结论。每条admission row只能是：

- `bit-exact`：需要数学与实现证明所有中间值、reduction和写回与formal semantics一致；有限random corpus不能证明bit exact。
  integer GEMM还要排除所有partial/compensation/merge中间s32 overflow；int8在部分AVX2/AVX-512实现上可能经
  `VPMADDUBSW`发生pairwise s16 saturation，s8×s8还可能通过u8 shift与compensation实现，必须由
  与实现无关的输入bound，或actual implementation加受管源码/build证据证明不会发生；effective ISA本身只表示可dispatch
  上限，不足以完成该证明；
- `profile-bounded`：只有解析证明或对离散有限domain穷举后，才可声明覆盖该完整value domain的数学上界；仅由独立held-out
  corpus得到的结果必须标记`empirically-qualified`，适用范围只到冻结的tested corpus/domain，不能外推为连续domain上界。
  `empirically-qualified`不是可泛化的第四种结论，而是value domain精确收缩到枚举corpus/payload digest的
  `profile-bounded` row；运行时输入不匹配该有限domain时必须rejected。通常浮点GEMM至多先进入这一类的经验资格；
- `rejected`：要求固定reduction tree、逐MAC窄舍入、专有中间截断或其它oneDNN不能表达的profile。

oneDNN `accumulation_mode`约束允许的累加精度与partial rounding，不证明reduction tree、FMA拓扑或逐MAC rounding。correctness profile
禁止`relaxed`和`any`，只能用`strict`或显式accumulator；每个primitive显式设`fpmath_mode::strict`。首版只构造无bias、
empty post-ops、无scale/zero-point/dropout/precomputed-reduction且target optional fields全关的primitive，不假设存在通用
“关闭fusion”API；未来bias/psum/scale/activation要么由target-owned epilogue按profile执行，要么该row rejected。
correctness MatMul必须显式设置`deterministic=true`；支持该attribute的reorder同样显式设置，不支持或返回
`unimplemented`则该admission row拒绝。它只表示同一固定platform/environment可重放，不证明跨ISA/version/thread或与
target bit-exact。oneDNN的
TF32 fpmath mode是允许f32隐式降精度，不是Wafer TF32 raw32 storage；TF32候选fast path必须先由target codec解码成f32，
以strict f32执行，再由target codec写回，只有product/accumulator profile相符才准入。low-precision destination优先让
MatMul输出f32/s32 temporary，再由target-owned codec舍入、饱和和pack。oneDNN reorder/narrowing即使通过准入也只是一种
implementation，target写回语义始终由semantic profile拥有。

bounded finite envelope覆盖normal、near-zero以及gradual-underflow profile中的subnormal；这些值先核对format/class，随后仍
必须通过该row的raw-exact或abs/rel/ULP数值判定，不能只因同属subnormal就接受。FTZ/DAZ row则按profile分别验证input
normalization、zero/sign、underflow/inexact/status，不把已flush值送入gradual公式。NaN/sNaN payload、Inf、signed zero和
overflow result另按显式classification/raw policy。若oneDNN要求caller sanitize或不能保持target special-value policy，
adapter必须用target-owned pre/post处理并单独证明，做不到则该domain rejected，不能用普通tolerance吞掉。

layout和commit固定为：

```text
target raw snapshot
  -> target-owned Cx/NCx decode/unpack
  -> logical dense tensor
  -> optional oneDNN-only reorder
  -> oneDNN MatMul into f32/s32 temporary
  -> target-owned rounding/codec/layout pack
  -> SystemC completion atomic commit
```

Cx/NCx是block-major并带compact tail，不能冒充oneDNN dense stride。`format_tag::any`只用于shape已知、内容稳定且reorder
结果可复用的immutable weights；per-invocation mutable、内容不可复用的input使用显式descriptor。primitive descriptor创建后
必须查询resolved `pd.weights_desc()`，按该descriptor的`get_size()`分配、作为reorder destination并纳入packed-cache identity；
禁止用`any` descriptor直接创建memory。oneDNN reorder的数值变换attributes必须完全缺省：不设置scales、zero-points或
sum，不能用显式scale=1、zero-point=0或sum=0冒充缺省；通用执行attributes仍按上文显式设置deterministic、strict fpmath
和user scratchpad，不支持即拒绝该row。reorder不定义Wafer quant、saturation、Cx/NCx或BOOL bit order，也只能对同一dtype、同一logical tensor/coordinate set和shape改变
physical memory format，不能承担broadcast、transpose、reshape或数值转换。MatMul preflight显式验证2..12D/same-rank relation、batch broadcast、M/K/N
axis relation、transpose view、source M/K与weights N/K至少一个相应轴stride=1，以及plain destination的N轴连续；不满足时在
primitive创建前拒绝。kernel先snapshot全部input并写private temporary；alias、padding、tail和canary均按physical profile保留，
codec和所有range通过后才提交。pack/reorder/scratch总内存先做checked budget，失败无architectural effect。

SystemC process同步调用`primitive.execute()`和`stream.wait()`；这段调用不推进simulation time，oneDNN worker不得调用
SystemC API或访问architectural memory。初版不把primitive任意offload到外部OS thread。使用
`scratchpad_mode::user`，每个in-flight execution在与primitive相同engine上创建并独占由primitive descriptor查询的
scratchpad。首版固定一个CPU engine kind/index，resolved primitive/memory descriptors、engine和stream的lifetime覆盖相关
primitive、memory与in-flight execute；依赖构建只允许一个兼容的thread/OpenMP runtime。固定thread runtime/count、
dynamic/nested/affinity、requested/effective ISA与hints、caller fenv，以及每个实际worker可验证的rounding mode、MXCSR
rounding-control bits、FTZ/DAZ和exception-mask initialization/readback。worker环境必须通过所选thread runtime的受管hook在
primitive执行所用pool内建立，不能假定caller的`fesetround`或MXCSR会被继承；无法覆盖并核对全部worker时，对应floating
row拒绝。`HostPlatformFingerprint`还固定CPU architecture/vendor/family/model/stepping/microcode、用于dispatch的CPUID/
feature leaves、OS/kernel、libc/loader、process affinity/NUMA policy，以及oneDNN、thread runtime和其它实际loaded DSO的
version/build-id或content digest。项目cache key包含完整
profile digest、M/N/K/batch、transpose/broadcast、logical/oneDNN strides/layout、all attrs、adapter version、effective ISA、
thread profile、worker-fenv profile/pool generation、`HostPlatformFingerprint` digest、CPU engine kind/index、resolved
descriptors和oneDNN version/build；不缓存invocation data handle。immutable
packed weights另以content/resource identity和resolved `weights_desc`为key。oneDNN primitive cache capacity按entry显式固定；project primitive/packed-weight cache另有checked byte
budget与eviction。每个MatMul/reorder分别记录`impl_info_str()`；它只作identity的一部分及审计/performance gate，不能推导
target语义。

process-global bootstrap顺序固定为：设置max ISA和CPU ISA hints并逐项检查status，随后设置primitive-cache capacity，再查询
effective configuration/version并创建engine/primitive descriptor。max ISA/hints setter必须发生在任何其它oneDNN API前。
同一process不切换冲突profile，需要验证另一全局配置时使用独立subprocess并记录全部启动配置。

admission provenance至少包含semantic profile digest、shape/layout adapter version、value domain、oneDNN version/build、
requested/effective ISA与hints、每个MatMul/reorder的`impl_info_str()`、thread runtime/count、caller fenv、worker-environment
evidence（逐worker rounding/RC、FTZ/DAZ、exception mask及pool identity）、`HostPlatformFingerprint`和全部primitive attrs。
解析证明只有在proof显式覆盖多个platform fingerprint时才能跨host复用；经验或implementation-bound row必须runtime exact
match全部platform字段。在admission/correlation qualification中formal backend是共同numeric oracle：backend result
和board result分别对formal result比较；正式large-command执行不要求每次同步重跑formal loop，但必须命中已冻结资格row，
不得让oneDNN成为唯一oracle。`T_target`记录target/board comparator，`B_backend`记录bulk-vs-formal envelope，二者在查看
held-out前分别冻结；exact target要求`T_target=0`且只能使用`B_backend=0`的bit-exact bulk。bounded target必须为backend预留
独立预算并验证`B_backend`未超额；model-vs-board的三角上界至多用于诊断，不能代替两条对共同formal oracle的acceptance。
切换backend不得静默放宽`T_target`，经验得到的`B_backend`只适用于其冻结tested corpus/domain。
若exact/bounded证据只对特定implementation成立，admission必须把`impl_info_str()`与oneDNN source/build digest、build
options、primitive descriptor和完整environment共同固定，并在descriptor创建后核对；implementation name本身不是证明，
这组字段仍只是backend admission identity，不得反向定义target semantics。

static shape在preflight创建primitive descriptor；`unimplemented`、unsupported descriptor、memory budget或admission缺失都在
input effect前失败。formal scalar path只允许在checked work/MAC budget内执行；超过budget且无admitted bulk时返回
`bulk_backend_unavailable`，不能悄悄跑慢。显式debug override可允许慢路径，但不计任何正式gate。

bulk qualification与runtime执行是两个独立stage。受管host tool `wafer-cmodel-qualify-bulk`强制
`calibrate -> freeze -> validate`三阶段no-replace workflow：calibration只读spec并记录formal/backend比较；freeze在任何
held-out执行前固定comparator、semantic、adapter、environment、calibration digest和预注册held-out spec/payload/
destination digest；validate只读policy执行held-out，不能回写或放宽envelope。同一次看完held-out再取最大误差的流程不得
签发资格。

当前tool使用显式formal和bulk byte/scratch/reorder budget运行。final canonical record绑定policy/calibration/spec、semantic、
adapter、value domain、target/backend comparator、formal/backend output、frozen/observed abs/rel、formal flags、resolved
descriptor、implementation、MatMul/reorder/formal-FMA计数及完整SEQ environment/backend identity；它不进入compiler IR、
package或target semantics。runtime readback该record后，只有command、payload、destination、semantic、adapter、environment
和预期backend output全部exact-match时才构造`BulkBackendAdmission`。当前经验row只能匹配record枚举的payload digest；
mandatory large GEMM资格必须来自这个offline producer，不能通过runtime debug slow path补做。phase log、wall-clock timeout、
ULP comparator、解析/穷举domain matcher或release registry若以后需要，必须扩schema并补closed parser/readback gate。

source vertical使用canonical schema v2 spec直接嵌入lower-case hex的lhs/rhs/destination-template target physical bytes，并逐项
验证format、layout、M/K/N/batch和exact footprint；tool不会在资格阶段重新打开source NPY或按seed再生成payload。`seed`只作为
source/config identity，嵌入bytes才是admission事实。calibration和held-out除spec digest外还必须具有不同payload digest，避免
同一输入重复签发伪held-out。driver只通过feature-independent abstract bulk seam提交完整resolved command和private physical
snapshots；record不匹配返回no-admission，超过formal budget时再稳定提升为`bulk-backend-unavailable`。

“一次MatMul call”配合formal MAC为零只证明bulk dispatch，不证明SystemC event缩放或性能；event缩放由Q22.V验证。
optional reorder primitive另行计数。未来可另建pinned-host bulk qualification，记录cold primitive/JIT、
pack/reorder和warm execute，覆盖M=1 decode、正常/批量GEMM和实际source workload；mandatory large profile若落入reference
implementation则performance不合格。correctness CI不绑定易抖动的绝对wall time，只有标记`performance-qualified`的固定
host profile使用冻结阈值。cold/JIT数据必须来自fresh subprocess或已证明清空/禁用global primitive cache的环境，不能把
cache hit标成cold；任一mandatory dynamic-source reorder落入reference implementation且主导端到端时间时同样不授予
performance-qualified。

oneDNN约束以受管版本官方资料为准；dependency manifest还必须记录与受管tag/commit匹配的文档版本，以下在线链接只作
当前导航：

- <https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html>
- <https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html>
- <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_accumulation_mode.html>
- <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_rounding_mode.html>
- <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_deterministic.html>
- <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_fpmath_mode.html>
- <https://uxlfoundation.github.io/oneDNN/dev_guide_reorder.html>
- <https://uxlfoundation.github.io/oneDNN/page_memory_format_propagation_cpp.html>
- <https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html>
- <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_scratchpad.html>
- <https://uxlfoundation.github.io/oneDNN/dev_guide_primitive_cache.html>
- <https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html>
- <https://uxlfoundation.github.io/oneDNN/group_dnnl_api_service.html>
- <https://uxlfoundation.github.io/oneDNN/struct_dnnl_primitive_desc_base.html>

### 4.7 FormalNumericExecutionContext、bulk environment和library state隔离

正式基础后端APFloat/APInt没有需要跨调用恢复的ambient rounding/flag状态，所有operation显式传入rounding并返回status；
MPFR的flags、emin/emax、default precision和rounding是global或OS-thread-local。独立SoftFloat conformance adapter的
rounding、tininess、extF80 precision和sticky exception flags同样是global或OS-thread-local，但不进入production context。
SystemC只保证process activation在显式suspension或
return前不被其它SystemC process抢占；OS-thread映射和affinity不是模型合同。因此formal scope可以依赖non-yielding
activation，却不能把TLS或thread id当作rank/process identity或隔离。每个formal numeric kernel使用以下作用域：

```text
save backend environment
  -> set complete NumericSemanticsProfile
  -> clear native flags
  -> execute without wait/yield/reentrant dispatch
  -> capture native flags and map target status
  -> restore caller environment on every exit path
```

APFloat/APInt基础scope只保存model-owned sticky status，并在每个operation后按profile聚合显式返回的status；不能读取host
fenv。DAZ由target-owned input normalization在调用前执行，FTZ由target-owned result codec在调用后执行。独立SoftFloat
adapter另用RAII保存/恢复`softfloat_roundingMode`、`softfloat_detectTininess`、`extF80_roundingPrecision`和全部flags，并固定
specialization、`THREAD_LOCAL`配置、source digest及non-trapping `softfloat_raiseFlags`实现；adapter通过不能替代production
profile gate。

MPFR scope保存/恢复emin/emax、default precision/rounding及全部flags，并检查每个environment setter返回值。每个destination/
temporary以`mpfr_init2`等显式precision初始化，每个operation显式传rounding mode，不依赖default precision/rounding。自定义
exponent range下必须保留operation返回的ternary值：IEEE-like subnormal路径直接把它传给`mpfr_subnormalize`；其它需要
range check的路径才传给`mpfr_check_range`，不能无条件先check再subnormalize或丢弃ternary。

configured Q22 formal backend要求独立SoftFloat adapter以`THREAD_LOCAL`构建、`mpfr_buildopt_tls_p()!=0`且MPFR
header/runtime版本一致；
依赖build的上游self-test还必须产生content-addressed `FormalDependencyConformanceRecord`，绑定source/config/build digest、
实际`libmpfr`和transitive `libgmp` artifact digest/build-id、MPFR version/patch/build options及`gmp_version`。production要么
静态链接这些exact tested artifacts并readback executable link manifest，要么启动时解析实际loaded MPFR/GMP objects并对
binary digest/build-id、version、patch/options、transitive linkage逐字段exact match；library search path或ABI-compatible替换
不能绕过。任一identity/TLS/header-runtime不匹配则configuration fail。仅component/debug build可在一个覆盖整个
non-yielding formal scope的global mutex下使用non-TLS依赖，且不计正式gate。即使TLS有效，每次scope仍save/restore，因为
logical process可能复用同一OS thread。

oneDNN不进入上述native flag capture。`BulkExecutionEnvironment`固定runtime、worker count、ISA、implementation及
per-worker initialization；caller thread的fenv/MXCSR仍用RAII恢复，但不能代表、初始化或捕获worker thread状态。对每个
floating row，qualification与runtime都必须在primitive实际使用的worker pool中设置并readback所需rounding mode（至少
`FE_TONEAREST`及MXCSR RC bits）、FTZ/DAZ和exception-mask policy，并把worker/pool identity纳入exact match；若某个受管
implementation明确忽略其中字段，只能用implementation-bound source/build证据记录其effective固定行为。无法控制并验证
全部worker、pool复用后环境漂移或runtime无法提供受管初始化hook时，该row rejected。oneDNN API status、output
classification和admission result可进入diagnostic；worker FP flags不得映射为target exception status。

formal context是invocation-owned聚合状态，scalar/tensor evaluator只在完整成功后commit；当前同步API没有scope未析构时的
nested dispatch seam。MPFR/SoftFloat adapter分别保存并恢复immediate caller ambient state，component test覆盖嵌套caller
scope的LIFO、normal/early special/error return、normal/subnormal和caller预置state恢复；工程以`-fno-exceptions`构建，不声明
C++ exception路径。另用两个真实OS thread分别验证SoftFloat adapter与MPFR TLS build和状态独立性。

依赖与调度事实以官方资料为准：

- <https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat-source.html>
- <https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat.html>
- <https://www.jhauser.us/arithmetic/TestFloat-3/doc/TestFloat-source.html>
- <https://www.jhauser.us/arithmetic/TestFloat-3/doc/TestFloat-general.html>
- <https://www.mpfr.org/mpfr-current/mpfr.html>
- <https://gmplib.org/manual/Installing-GMP>
- <https://standards.ieee.org/ieee/1666/7293/>
- <https://systemc.org/resources/standards/>

target model对numeric profile继续遵守：

- 有硬件/ABI直接证据并有区分性测试的行为进入supported profile；
- 只有数学名称而无edge behavior证据时，模型结果标记为model semantics，不能称hardware-bit-exact；
- zero-point等没有唯一公式的组合在执行副作用前拒绝hardware profile；候选model profile必须命名并进入provenance；
- stochastic行为在取得seed/state合同前只可做独立统计correlation，不得用已退役reference policy冒充硬件；
- floating comparator按op/dtype/profile定义，不使用全局tolerance。`reference`始终是formal result，`observed`分别是bulk或
  board result；两者各自对formal比较，model↔board只作诊断。bit-exact row按profile规定的raw-bit/classification equality；
  bounded finite row的abs/rel clause固定为
  `abs(observed - reference) <= atol + rtol * abs(reference)`。每个comparator显式声明`max_ulp`启用或禁用；启用时ULP clause
  与abs/rel clause取AND，禁用时ULP只记录不参与acceptance。ULP只比较同一declared destination format经target-owned codec
  处理后的canonical finite encoding：FP16/BF16/FP32分别使用16/16/32-bit storage bits；TF32先验证或按profile规范化raw32
  低13位，再提取`sign|exp8|frac10`为19-bit semantic encoding。对宽度`w`的encoding `b`定义单调key为
  `sign(b) ? ((~b) & (2^w-1)) : (b | 2^(w-1))`，ULP distance是两key的无符号绝对差；禁止跨dtype直接计算ULP。
  NaN/sNaN payload、Inf、signed zero和overflow先按独立classification/raw policy判断。gradual-underflow的subnormal和其它
  near-zero finite值在classification通过后仍进入abs/rel及启用的ULP clause；FTZ/DAZ row改按其zero/sign/status政策判断；
  GEMM/reduction length另分profile；
- profile阈值在查看held-out结果前冻结；source CPU corpus已有tolerance不自动成为board numeric tolerance。

### 4.8 SystemC/TLM和simulation-time边界

SystemC是Q22正式functional-event CModel的强制执行容器，这是项目工程选择，不是从vendor binary恢复出的事实，也不
自动提升精度。target-model feature在configured build中必须找到受管、版本固定的SystemC dependency并真实执行相应
tests；基础compiler、manifest parser、no-card runtime和plain C++ kernels仍不依赖SystemC。

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

### 5.1 近期target-call/SystemC执行

近期正式入口是同一次`wafer-compile` invocation中的下游verification consumer：Q17/Q18先按各自合同原子发布verified
target/package artifacts；driver再用仍由invocation持有的owner-backed target LLVM bundle做owner-safe native
materialization。shared typed registry验证reachable call/signature，per-rank exact-signature bridge显式绑定context并把
实际target call同步投递给SystemC model。model mismatch或执行
失败可让driver返回非零并保留diagnostic，但已验证package保持可审计，不回滚、不改写，也不让Q22成为Q17/Q18
correctness前置。是否启用该gate属于用户显式请求或configured CI policy；不能变成wafer-opt stop-stage或手拼pass。

repo-owned target-call/SystemC入口能证明compiler-produced target call经过typed ABI/transaction/event语义并产生正确完整
输出；它不执行repo CRT、不构造Tsm packet，因此不声明CRT/packet正确性。以后只有Q22.K与RISC-V archive register trace、
board capture或versioned vendor builder逐字段相关后才增加packet provenance。该入口也
不读取已发布package、不执行RISC-V ELF/vendor archive，不能使用`wafer-run`的provider语义。

当前driver只有一条数值比较边界：调用者通过`--model-input`/`--model-expected`提供固定source corpus payload和独立
CPU expected。不存在accepted-IR reference oracle或失败后的隐式切换；F16/BF16/F32 finite output使用case显式
`atol + rtol * abs(expected)`，整数、布尔和其它非浮点destination raw exact，NaN/Inf在当前source vertical gate拒绝。
model成功诊断同时发布
ranks、target transaction、SystemC thread/delta、formal/bulk command、MatMul/reorder/formal-FMA计数及每个bulk admission
record digest，便于CI证明large GEMM没有按MAC形成event或scalar fallback。这些provenance不进入package。

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
`RuntimeProvider`消费当次compiler原样发布并由Q18验证的package；manifest version、resource/slot、transport
requirement和module digest不增加model专用分支。Q22.E是独立later gate，不以Q32新增package格式为前置。
若执行Q32.V扩展且该provider采用capability-bearing package，provider必须额外消费并逐项验证同版本
`RequiredCapabilitySet`，不能把扩展command解释成
当前v1能力。Q22.V记录的旧package只作历史证据，不能替代fresh exact-module input。provider输出invocation-local
result/diagnostic，不能写回package或生成partial successful result。

当前`RuntimeSessionPlan`按一个entry/rank做no-card preflight；Direct DTE exact execution不能把16个entry plan简单顺序
循环。provider落地前必须由tasks/15补充owner-backed all-rank invocation/session，把manifest中all-and-only entries、
typed invocation bindings、transport capability、共同submit/progress/status和atomic cleanup组成一个执行域。该session只
引用package已提交事实和provider-owned runtime handles，不复制target module内的message/packet schedule。

## 6. Event、completion和failure合同

模型至少区分：

- target call已提交；
- 当前tile conservative issue domain local drain；
- Direct DTE receiver ready、busy、done/error、source lifetime和destination visible；
- multi-rank no-progress/deadlock；
- provider terminal status和copyback eligibility。

target-call bridge返回不表示model effect已经完成，local fence不能替代DTE wait或multi-tile
arrival。每个issue domain使用显式、单调的invocation-local issue ordinal/watermark；wait捕获调用前watermark并等待相关
command完成，不能用delta scheduling order充当identity。保守模型可以延后event完成，但不能合并没有证据的completion
domain。所有rank/process均等待且SystemC event queue为空但仍非terminal时，形成确定性no-progress state snapshot并整体
失败，不用wall-clock决定语义。

所有入口在input import前完成all-rank structural capability、static profile、slot/resource/address-plan和endpoint preflight。
unknown symbol/signature/static profile或exact-module环境不匹配必须在此时整体失败。动态command tuple、computed address、
descriptor和runtime event错误，在私有invocation内、相应read/effect前拒绝。
这种失败仍必须：

- 停止dependent event和copyback；
- 标记受影响invocation/rank，不产生可误认为成功的partial result；
- 逆序释放已获取资源；
- 保留稳定stage、rank、entry、command/event类别和model/profile provenance；
- no-progress使用确定性state snapshot诊断，不依赖wall-clock或thread调度决定语义。

## 7. 模型精度和发布标签

| Profile | 最低事实和gate | 允许声明 | 禁止声明 |
| --- | --- | --- | --- |
| repo-owned target-call/SystemC model-only functional-numeric | same fully legal target LLVM、shared typed call registry、exact-signature context bridge、untimed SystemC event/completion、formal scalar语义层、逐profile admitted oneDNN大GEMM backend和完整输出differential | supported model profile内的target-call/ABI/event functional-numeric correctness | repo CRT、Tsm/packet、hardware numeric、package ELF、board、timing |
| optional authorized CRT/packet/MMIO conformance | owner-approved host package或独立规范、exact-ELF register trace、board capture或versioned vendor builder的逐字段decode/range/engine/register-effect conformance | 对应合法source/capture范围的CRT/packet/register provenance | 仅凭trace声明numeric、完整loader/provider lifecycle、timing或board等价 |
| Q22.C board numeric correlation | Q22 model-only、Q32 integrated audit的当次verified package、Q6.B有效board execution、current emitted command rows、重复稳定性、冻结的comparator、独立held-out和source-backed完整输出；测试Q32.V扩展时另加同版本package set preflight | 绑定环境和tested domain的board-output-correlated profile；对应独立packet/MMIO evidence闭合后才升级为hardware-correlated-numeric | vendor-exact packet、不可观测内部实现、未测domain、exact package、timing |
| exact-module functional | Q22 model-only、Q32 integrated audit后当次Q18 verified package、all-and-only RISC-V ELF、ISS/loader/MMIO/provider lifecycle；Q32.V扩展package另要求required-capability preflight | package target-model execution | real board、hardware timing |
| deferred Q22.P timing | 另行恢复后的PMU measurement basis和独立timing held-out | 实际通过的LT/AT profile | 改变numeric、IR legality或candidate acceptance；无RTL证据时称cycle-accurate |

profile是execution result provenance，不是compiler legality或package semantic branch。高层profile失败不能反向改变
accepted instruction支持范围；如果硬件可表达但model未覆盖，应扩target model capability或保持该profile拒绝。

### 7.1 当前Q22发布能力矩阵

下表只描述已发布model-only profile；exact selector仍以shared typed registry为唯一事实源。数量是closure证据，不是新协议：

| 边界 | 当前发布支持 | 明确拒绝或未升级 |
| --- | --- | --- |
| storage / physical codec | 13种logical raw codec；compact、Cx、NCx及bitpacked BOOL的shared physical geometry/roundtrip；typed ABI slot显式携带layout | program-boundary compact BOOL NPY尚无bitpacked source合同；codec存在不开放无target encoding的engine row |
| formal convert | 101条确定性typed convert policy，四种确定性rounding及完整raw/special/status政策 | 4条INT8-source zero-point公式和stochastic state未证，保持命名candidate/rejected；f64不在target profile |
| formal elementwise | 88条selector（84条floating和4条BOOL logic）；f16/bf16/f32的已注册算术、关系、基础/MPFR transcendental按唯一profile执行 | integer elementwise、`exp_lp`/`sat_relu`/`leaky_relu`未闭合参数政策；未知selector无fallback |
| formal GEMM / reduce | 当前v1 normal/normal f16、bf16、f32同dtypeGEMM，F32 fused accumulator、+0 init、K递增、destination RNE；native F32 sum reduce按+0 accumulator、logical row-major input递增和逐step RNE执行 | Q32.V oriented GEMM需新的versioned TargetCall/key/formal语义；采用external bulk lane时再闭合该lane的exact admission，board执行另需独立board row；I8 accumulator政策、TF32 generic GEMM、其余15条native reduce selector拒绝；不从dtype猜窄/宽accumulator |
| admitted bulk GEMM | 当前component资格覆盖v1 normal/normal f16/bf16/f32同dtype、rank 2/3、Cx/NCx；Q22 source发布的admitted完整case为Q20首个f32 GEMM和64³ f32 GEMM | orientation是exact identity；无exact command/payload/destination/environment/expected-output record即no admission；超过formal budget时绝不scalar fallback；不外推连续输入域bit-exact |
| managed-reference bulk GEMM | Q28 scale gate逐command验证supported GEMM semantic、shape/layout、受管environment、finite inputs及byte budget，并强制完整PyTorch expected tolerance comparison | 不产生exact qualification record，不声明raw-exact target arithmetic、hardware correlation或未检查value-domain；provenance与exact admission分字段 |
| functional transaction / event | checked RDMA/WDMA、gather/scatter、memset、elementwise、convert、当前v1 GEMM、local fence及当前single-destination Direct DTE control；all-rank private SPM/DDR和atomic output | mapped DMA、physical-footprint fill与oriented GEMM由Q32.V实现：分别增加Instr/local-address、typed fill domain/raw scalar和versioned orientation gate；Count当前只有opcode/raw-writeback线索且无typed semantic/model consumer，必须pre-effect拒绝；field-valid但无kernel的conv/pool/unpool等family、未知地址/layout/endpoint；无worker/queue容量、packet或timing claim |
| source vertical | Q20 formal/admitted、f16/bf16 formal、64³ f32 admitted、Q21 16-rank Direct DTE；全部直接比较固定source CPU expected | 未固定expected的shape stress不算numeric evidence |

这张矩阵说明“支持多数据类型”由storage、engine legality、numeric selector、functional kernel和source evidence分层；不能把
13种codec、oneDNN公开dtype列表或单个source case分别冒充整个笛卡尔积。

## 8. Verification Gates

### 8.1 Q22.N/B/L/H/S/V 分项 Verification Gates

本节按queue row分组记录证据；同处一节不增加依赖。调度依赖以第10节和`tasks/progress.md`为准，任何下游positive都不能
反向成为Q22.N、Q22.B或Q22.L的完成前置。

#### Q22.N numeric foundation

- 从tasks/14唯一shared typed registry读取13种`LogicalFormatDescriptor`和有证据的target-profile×engine×format
  `TargetFormatEncodingRecord`；穷举INT8/UINT8/BOOL、FP16/BF16 raw pattern和TF32 canonical encoding，并以
  classification/boundary/random覆盖FP32、宽整数及noncanonical TF32。tasks/08独立验证BOOL physical bit ordinal、
  Cx/NCx block/tail/footprint；byte内LSB0/MSB0只由显式encoding/model policy选择，无证据UINT/64-bit
  DMA row和f64在target preflight拒绝；
- 36条convert route、四种确定性rounding、zero-point/stochastic命名候选及float special-result逐项区分；formal kernel覆盖
  f32、f16/bf16窄/宽累加、TF32-to-f32和i8-to-s32 component，逐family使用independent oracle或trusted-TCB gate；
- type-generic elementwise/GEMM/reduce覆盖operand/product/accumulator/intermediate/destination、FMA、reduction order和
  overflow；每个published `(ModelProfileId, NumericCommandKey)`唯一映射`NumericSemanticsProfile`，unknown/duplicate在
  effect前拒绝；
- `FormalNumericExecutionContext`覆盖target-owned sticky flag原子commit；SoftFloat/MPFR save/restore覆盖immediate caller
  ambient scope嵌套LIFO、normal/early/error return和双OS-thread TLS。工程`-fno-exceptions`，本row不声明C++ exception测试，
  也不依赖SystemC process或oneDNN output。

#### Q22.B bulk qualification

- semantic profile、shape/layout、冻结payload和完整oneDNN environment共同定址当前`profile-bounded/rejected` row；有限
  corpus即使观测raw-exact也不能外推为bit-exact；
- adapter重放`raw snapshot -> target decode/Cx/NCx unpack -> f32 dense -> optional weights reorder -> oneDNN f32 MatMul
  -> formal finalize -> target pack -> atomic commit`，覆盖padding/tail、scratch/reorder/total budget、resolved descriptor、
  destination template和异常路径；
- 正式row只能由独立`calibrate -> freeze -> validate`canonical no-replace记录签发，readback后runtime精确匹配command、
  semantic、adapter、payload、destination、environment和预期output。malformed/tampered record、reference implementation或
  environment control drift在effect前拒绝；
- generated F16/BF16/F32、batched NCx及64³ component corpus证明一次MatMul、最多一次reorder和零backend formal FMA；
  64³ row超过runtime formal budget后仍无scalar fallback。source-backed dispatch属于Q22.V，不由本component gate冒充。

#### Q22.L target LLVM module bundle

- Q0.L prepared target LLVM/ABI artifact形成owner-backed、move-only、不可序列化的all-rank bundle；逐rank readback logical rank、
  entry、profile/target identity/Kernel Runtime ABI、ordered typed slots和module identity，任一late failure均不形成bundle；
- direct shim只消费该bundle检查symbol、signature、control flow、typed slot和基本address formation，结果仅标ABI smoke；
  bundle不调用Host CRT、构造packet、链接SystemC或替代Q17/Q18 artifact。

#### Q22.H repo-owned target-call frontend

- Q22.L同一target LLVM经过owner-safe clone/native retarget和动态slot thunk；稳定Target层shared typed registry同时拥有
  lowering和frontend需要的symbol/signature/call family/field decoder，109项descriptor逐项形成对应typed payload，
  不能在JIT/SystemC/test复制字符串表；
- frontend在sink begin前交付完整ordered typed slot metadata/value bindings；per-rank exact-signature bridge显式绑定invocation/rank context并形成
  typed transaction。rank防yield重入，prepare-commit失败仍可abort，unknown symbol/signature、wrong ABI slot和
  native-illegal structure在执行前拒绝，callback/destruction/late-rank failure均无partial result；
- 本gate不编译repo CRT、不实现Tsm factory/operator或packet。它必须明确只证明target-call frontend，不能冒充Q22.K。

#### Q22.S SystemC functional-event model

- Q22.H实际transaction进入rank/tile memory、conservative issue event、local completion和Direct DTE/FSM；numeric effect只调用Q22.N，
  不能读取另一套accepted-IR numeric kernel或DTE scheduler；
- 唯一`sc_main`至少运行两个`SC_THREAD`跨delta覆盖issue/visibility/completion、failure wakeup和invocation-owned numeric
  status聚合；immediate caller ambient环境恢复由Q22.N单独证明；
  unavailable/skipped或plain C++ kernel test不算完成。本row不以Q22.B或完整source workload为前置。

#### Q22.V source-backed functional-numeric verticals

- 同一`wafer-compile`重放Q20 rank-count=1 f32 linear/residual MLP、source-produced f16/bf16 GEMM和Q21 16-rank tiny Llama，
  覆盖它们实际调用的typed target-call/numeric/event/Direct DTE surface并比较all-and-only完整输出；
- source-backed deterministic large GEMM必须超过formal budget、拥有固定source/config/seed或payload及独立expected/digest，
  并自动命中Q22.B冻结admission；现有未初始化4096 exporter升级前只算结构测试，不能证明dispatch或numeric；
- SystemC-enabled vertical必须真实执行；任一rank late failure无partial model result。Q17/Q18仍先按各自合同发布，model mismatch
  只让verification返回非零并保留已验证package供审计。

该gate现已完成：五个固定corpus case由正式source exporter进入同一driver；Q20 formal/admitted输出均通过，admitted路径
只替换其中一个GEMM；f16/bf16 formal raw output通过；64³ large GEMM在10000 FMA预算下命中一次exact source-payload
admission且bulk formal FMA为零；Q21执行全部16 ranks和17个SystemC thread process。错误expected及错误bulk record
均返回非零并保留已验证package。

unsupported-reason closure只是各row必要条件，不能替代相应positive matrix。

### 8.2 Q22 Current Gate 与 Q32.V Typed Target-Capability/Package Extension

既有Q22 v1 typed target-call/memory gate已经完成。Q32.I/R/B先在新的production compile结果上fresh重放该gate：

- CT/NE/RDMA/WDMA/TDMA的现有typed args、descriptor、address/end、model error latch和target ABI可观察status逐family覆盖；
- SPM/DDR bounds、reserved region、Cx/NCx、bitpacked i1及当前subview/strided descriptor由独立slow oracle验证；
- target-call/SystemC positive来自compiler-generated target LLVM，并把exact-signature bridge实际形成的typed transaction
  作为functional-event输入；
- preflight从current reachable target calls检查当前v1 model admission。结果只决定本次model consumer接受或拒绝
  已选程序，不进入planner、不生成候选，也不改变compiler legality。

下列compiler与repo-owned model项目属于已排期Q32.V，不改变Q22 done状态，但必须在Q32.M/S消费相应planner
choice前独立闭合：

- mapped RDMA/WDMA：非零SPM local offset已折入最终address、多descriptor、full/C0 tail、pre/post canary及非法
  双侧stride；CModel只凭最终`TargetTransaction`验证bytes，不读取planner relation；
- physical-footprint fill及其padding/tail可观察性；
- versioned oriented GEMM：NN/NT/TN/TT非方阵、非对称payload、stored-shape mismatch、wrong ABI revision和
  admission-record mismatch；orientation进入`NumericCommandKey`及formal/bulk/managed provenance；
- 若实际consumer需要，Q16/Q17/Q18对同一versioned `RequiredCapabilitySet`及其当批冻结encoding做all-and-only readback；missing/extra/tampered key、
  late-rank union failure和profile/key mismatch均无partial package/model result；
- 以显式`ModelProfileId`执行repo-owned TargetCall/SystemC extension gate；若存在external model provider，
  其admission只在effect前检查已经发射的扩展command，不参与physical-dataflow选择。

真实board environment admission属于Q6.B/Q22.C external gate，不阻塞Q32.V、Q32.M/S或Q32 umbrella
completion；它只能在effect前接受或拒绝已经发射的extension row。

若取得exact-ELF register trace、board capture或versioned vendor builder，可另做逐字段packet/MMIO correlation并提升
packet provenance；它属于Q22.K可选诊断证据，不是Q22、Q32.V compiler/SystemC或Q22.C前置，也不能替代numeric vectors。

### 8.3 Q22.C Board numeric correlation gate

Q22.C在Q22、Q32 integrated audit、Q6.B和configured numeric corpus均可用后执行，是独立later gate，不要求PMU、packet capture、
exact-module provider或新的package capability schema。当前v1 correlation消费当次compiler发布并验证的package/profile，
fresh重放Q6.B lifecycle；Q6.B证明allocation/H2D/load/launch/wait/status/D2H/cleanup、watchdog/reset和
重复invocation有效。任何provider、completion、copyback或guard失败的sample均为invalid，不能用于调numeric。

board provider必须在任何effect前把本次将执行的current command rows匹配到同environment的board-supported
allowlist；model row或profile名不能替代该gate。若correlate Q32.V扩展且其consumer采用capability-bearing package，才额外要求同一revision的
`RequiredCapabilitySet` all-and-only readback并逐key匹配。缺少Q32.V package set只阻止扩展row升级，不阻止当前v1
Q22.C设计继续作为later gate。

板端corpus按capability row生成，而不是按一个op名字笼统通过。首批顺序是：

1. 13种storage format的movement/fill/boolean/guard、layout tail和未写区域byte/bit exact；Q32.V mapped
   RDMA/WDMA再按direction、非零local offset、多descriptor和canary隔离校准；
2. 七种compute/convert format的generated single-op vectors，覆盖36条convert route、四种确定性rounding及已实现
   f16/bf16/f32/TF32/integer arithmetic和GEMM/reduce candidate；Q32.V oriented GEMM按每个
   ABI/format/orientation tuple独立row；
3. rank-count=1 f32 GEMM、add、tanh及完整MLP；
4. Tiny Llama实际需要的reduce、exp、rsqrt、mask/i1和其它已accepted tuple；
5. 16-rank Direct DTE与完整output；
6. zero-point和stochastic候选从第一版即存在，但只有公式、seed/state/advance事实闭合后才选择hardware row。

每个numeric family至少覆盖以下能区分预先列出的候选语义的向量；有限样本不证明全输入域等价：

- rounding：正负halfway、halfway±1 source ULP、最大有限值边界、min normal/subnormal和float-to-int的
  `N + 0.5`，测试值从raw bits构造；
- special values：±qNaN/sNaN及payload、±Inf、±0，max/min同时交换operand顺序；DAZ与FTZ分别测；
- overflow：float max附近及integer边界±1，区分Inf/max-finite、wrap/saturate/trap/status；
- GEMM/reduction：cancellation及排列、K/reduction length跨tile/tail边界、非方阵非对称payload区分NN/NT/TN/TT，
  并可区分accumulator width/order和
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
  model profile，不能照搬已退役reference snapshot policy冒充硬件；
- target model不能读取shadow logical message schedule或用另一套coordinator实现transport；
- exact provider按allocate/import、H2D、load/resolve、submit、wait/status、D2H、cleanup逐阶段注入失败；
- wait/status失败后禁止copyback，任一rank失败无partial successful result。

### 8.5 Source-backed和correlation gate

- 复用Q20 rank-count=1/16 linear/MLP和Q21 16-rank tiny Llama的同一source/config及accepted upstream chain，不建立
  model专用fixture或计划；target-call/SystemC消费同一invocation的target LLVM，只有exact-module frontend才消费原样package；
- source-produced f16/bf16 simple GEMM及deterministic large GEMM同样从正式frontend进入；large case必须有固定payload、
  独立expected/digest并超过formal budget。generated shape corpus只补dispatch，不替代source-backed numeric expected；
- 检查全部rank/module/output、typed status和可观察transaction/event数量，不用rank 0、shape或digest代替执行；
- target model与source CPU expected差异先按target call、memory、numeric、event分类；只有Q22.K启用时才增加CRT/packet类别；
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

### 10.0 Implementation readiness spike

```text
Pipeline position:
- Upstream artifact / IR:
  Q21固定source/config/payload及同一wafer-compile形成的accepted rank artifacts；当前tasks/14 target preparation、repo CRT/
  vendor dependency源码和本机toolchain/package environment。source CPU expected只作结果核对，不替代artifact census。
- Current stage responsibility:
  对source-produced reduce做kind/init/dimension/shape/layout census并checked估算Q0.L ordered composite的command、SPM和compile
  budget；对SoftFloat/TestFloat、MPFR/GMP、oneDNN、SystemC、repo host CRT/operator seam执行read-only或transaction-local
  configure/compile/link probe，记录exact command、版本、architecture、license入口和失败根因；把Q22拆成独立queue边界。
- Output artifact / IR:
  编号设计中的readiness evidence、更新后的任务队列和可复现probe workflow。它不是compiler IR、ExecutableBundle、package、
  numeric profile或长期binary artifact；临时probe build只在transaction-local目录存在。
- Downstream consumer:
  Q0.L实施计划及Q22.N/L/B/H/S/V各自的dependency和completion gate。
- User-level driver / named pipeline:
  真实Q21入口仍是wafer-compile；IR-local dump只从该正式producer派生。依赖probe使用当前CMake/toolchain、pkg-config和最小
  compile/link命令，不引入production CLI。
- Explicit non-goals:
  不实现Q0.L语义或CModel，不签发oneDNN admission，不选择hardware numeric policy，不把header presence/compile-only或手写IR
  当作纵向通过，不要求board/vendor simulator；允许修复阻塞source-backed readiness重放的窄小既有compiler缺陷并增加回归。
- Completion gate:
  Q21 reduction census能证明当前ordered composite在明确budget内可实现，或以可复现反例否决并修订设计；每项本地依赖/
  host seam都有source/version/architecture/link/license结论和新鲜命令；无法本地判定的vendor/board项明确归为external；
  Q22独立queue rows与后续plan前置已建立。
```

#### 10.0.1 Q21 source-backed reduce census

fresh 16-rank formal replay由固定tiny-Llama source/config/payload通过同一`wafer-compile`完成，reference完整输出匹配；verified
package含16个rank module、entry、completion和ELF。以下数据来自Q29之前的legacy scheduling debug
replay：它从当时正式producer为每个logical rank派生，不是手写fixture，也没有把debug dump当作
published package成员。其group ordinal/group-boundary只是历史census标签，不是当前production
artifact或调度合同；Q29/Q28必须从structured task/dataflow主线重放结论。当时16个rank的
`@main`均有相同四项：

| legacy group ordinal | kind / init | source dims | input -> result | tile layout | extent |
| --- | --- | --- | --- | --- | --- |
| `#0` | sum；legacy group-boundary scalar最终为f32 `+0` | `[2]` | `1x4x16xf32 -> 1x4xf32` | `NCx -> Cx` | 16 |
| `#20` | IEEE maximum；local f32 `-Inf` | `[3]` | `1x1x4x4xf32 -> 1x1x4xf32` | `NCx -> NCx` | 4 |
| `#21` | sum；legacy group-boundary scalar最终为f32 `+0` | `[3]` | `1x1x4x4xf32 -> 1x1x4xf32` | `NCx -> NCx` | 4 |
| `#28` | sum；legacy group-boundary scalar最终为f32 `+0` | `[2]` | `1x4x16xf32 -> 1x4xf32` | `NCx -> Cx` | 16 |

这些tail-dimension physical mappings中，每个fixed reduction tuple都可由一条`gather_scatter`把slice materialize为4个连续
f32；sum/max均有exact map-free elementwise combiner。令extent为`R`，correctness-first基线有`1 fill + R slice
movement + R elementwise + 1 final movement = 2R+2`条engine command，并在每条后保守形成completion，共`4R+4`
个terminal op。因而extent 16/4分别为68/20项，每个rank合计176，低于tasks/10/11新固定、独立于candidate guard的
4096个static reduce terminal-op预算；16 ranks合计2816只作规模说明，budget仍按每个accepted rank独立检查。

每项result-shaped A/B/scratch的compact payload为`3 x 16 = 48 B`；按当前256 B SPM alignment保守占三个slot即768 B。
当前四个reduce region的最高existing span为2064 B，加上不复用任何旧buffer的768 B后为2832 B，只占
`[65536, 3080192)`共3,014,656 B可规划窗口的约0.094%。因此Q21不否决ordered composite，也没有逼近SPM边界。

该结论最初只证明Q0.L可实施；当前正式lowering已经改为init-first canonical-order composite，并按每rank独立4096
terminal-op cap在selector和最终target边界重算。它仍需本轮fresh source-backed gate作为完成证据；Q21本身没有覆盖
dynamic init、所有非tail/multi-dim reduction、avg及完整target special-value政策。tasks/06 tile-dataflow candidate materialization
counter与terminal command counter语义不同，不能共用。

#### 10.0.2 Numeric和SystemC依赖实证

checkout/system探针与官方candidate源码探针得到以下边界。`candidate source probe通过`只说明该上游版本能在当前
x86-64/GCC 13环境构建运行，不等于已进入`WaferDependencyVersions.cmake`、通过完整上游self-test或成为Q22
production依赖：

| dependency | checkout/system事实 | transaction-local candidate结果 | 实施结论 |
| --- | --- | --- | --- |
| SoftFloat | 系统无source/header/library/package，Q22.N不依赖隐式系统包 | Q22.N受管Release 3e；固定`ARM-VFPv2-defaultNaN`、`THREAD_LOCAL=_Thread_local`和non-trapping raiseFlags，TLS/双thread/adapter gate通过 | 只由唯一`WaferNumeric::SoftFloat`用于F16/F32独立oracle，默认global-state build和identity不匹配均在numeric effect前拒绝 |
| TestFloat | 系统无source或可执行文件 | 受管Release 3e构建`testsoftfloat`，F16/F32 mulAdd及level-1/level-2 slowfloat conformance进入record | 只验证SoftFloat本身，不算第二个production numeric oracle |
| MPFR/GMP | 系统仅有不可作为开发依赖的x86-64 runtime；Q22.N不链接其SONAME | Q22.N现由受管GNU m4 1.4.21构建GMP 6.3.0与MPFR 4.2.2，共享库、header、TLS/version/transitive identity和上游self-test均进入完整conformance record | feature-on只导入record精确指向的artifact并readback实际loaded identity；缺root/record、任一digest/options/gate或路径别名均configuration fail，feature-off基础compiler不链接它们 |
| oneDNN | 无独立header/library/package；PyTorch 2.5内嵌3.5.3符号为local，空壳CMake target不可复用 | 官方v3.12 commit `80afa710...`以CPU SEQ、INFERENCE、MATMUL/REORDER、static配置构建；`DNNL::dnnl` 2x2 f32 MatMul返回3.12.0和正确结果 | Q22.B从受管source形成唯一target；该probe不签发任何bulk admission，也不证明dtype/profile等价 |
| SystemC/TLM | 无header/library/pkg-config/CMake package；不能隐式消费宿主包 | Q22.S已把官方3.0.2 commit `70b0fc8e...`及archive SHA-256固定到统一版本文件；受管bootstrap构建静态`SystemCLanguage` package，独立consumer经官方`SystemC::systemc`运行C++17 `sc_main`、两个`SC_THREAD`和delta-cycle event | 默认关闭；feature-on要求Q22.N和完整canonical record，缺root/record、source/install tree、library/header/package export/license或任一gate即configuration fail；不混用Ubuntu 2.3.4 ABI |

SoftFloat/TestFloat 3e采用U.C. Berkeley三条款式许可，SystemC 3.0.2参考实现为Apache-2.0，oneDNN为Apache-2.0；
MPFR/GMP分别涉及LGPL及GMP双许可，具体静态/动态分发、source offer和notice由引入任务在项目发布政策下确认。官方当前
资料确认MPFR 4.2.2要求GMP 5.0以上，故GMP 6.3.0满足版本关系；license文本存在不等于本项目已经完成合规审查。
readiness candidate source identity为SoftFloat 3e zip SHA-256 `21130ce8...c746`、TestFloat 3e
`6d4bdf00...ad6`、GMP 6.3.0 `a3c2b802...8898`、MPFR 4.2.2 `b67ba038...ce01`，以及表内两个Git commit；
Q22.N现已在统一版本文件记录并由bootstrap校验完整digest，表中截断值仍只作可读审计摘要；oneDNN已经由Q22.B正式
受管引入，SystemC依赖及functional-event component已由Q22.S完整闭合。bootstrap smoke只证明依赖可用，不能替代后续
16-rank numeric/Direct-DTE component evidence。

#### 10.0.3 Host CRT、vendor seam和replay边界

host CRT的可编译/不可链接边界及external授权gate见3.1/3.4。结论是wrapper层可复用候选已经被编译事实支持，但当前没有
host operator、Direct-DTE/SPM provider或可加载vendor CModel closure；不能靠RISC-V archive、include-only target或
repo-owned transaction冒充CRT/packet evidence。vendor交付和授权属于external，但不阻塞已经闭合的Q22.H typed
target-call frontend和Q22.S functional-event model；Q22.K才等待合法vendor seam。

历史readiness replay曾暴露旧materializer创建`async.token`却未声明Async dependent dialect的问题；
修复模式已经沉淀到`memory/bugs.md`。该旧debug结果不能代替当前structured task/dataflow或Q0.L/Q28证据，
退役入口也不作为compatibility replay保留。

#### 10.0.4 Readiness决议

- Q21 resource census通过并已由Q0.L按tasks/10/11 ordered reduce设计完成fresh source replay；readiness不再是前向blocker。
- Q22.N已经建立默认关闭的受管source/bootstrap、唯一CMake target、23项build/self-test/identity gate与license artifact
  closure；Q22.B随后独立闭合受管oneDNN、qualification record和发布政策，readiness probe本身没有被当作bulk admission。
- Q22.L已经独立形成owner-backed target LLVM bundle；Q22.H直接消费它建立repo-owned target-call frontend；
  Q22.S/Q22.V继续依赖Q22.H，external vendor seam只影响Q22.K packet provenance。
- vendor CModel套件、真实board和hardware numeric/packet/timing仍是external evidence；它们不否定model-only方案，也不能由
  文档、有限corpus或SystemC选择推断。

### 10.1 Capability和依赖收敛

- Q0.L已完成：production `CompilationRequest`贯穿typed `TargetProfileId`；debug named target pipeline只用同一registry
  显式解析required option。tasks/14单一拥有engine×format ABI/register legality，source reduce按init-first canonical-order
  composite展开，elementwise map不再静默丢义；Q22.N/L只消费这些已验证事实，CModel不得补救compiler已丢失或未证明
  合法的command语义；
- 当前不追索或逆向实现封闭vendor CModel package。只有项目owner以后主动提供可用host development package或独立公开规范，
  并由项目owner/法务确认允许host集成、修改或clean-room实现时，才重新审计对应headers、libraries、resources、transitive
  dependency、license和version；此前不实现repo CRT/host Tsm/packet，Q22.H只消费repo-owned typed target-call ABI；
- 固定首批target call、dtype/layout、engine、Direct DTE和completion capability matrix；packet字段只属于Q22.K；
- Q22.L已把tasks/14 private prepared target LLVM提升为owner-backed all-rank内部artifact，并让现有device link直接消费；
  该本地artifact不以vendor授权为前置，也不提前执行host CRT/packet；
- 默认关闭的`WAFER_ENABLE_SYSTEMC_MODEL`已经固定Accellera SystemC 3.0.2完整commit/archive digest、获取方式、
  Apache-2.0 license/notice、source/install tree、静态library/header、`SystemCLanguage` package、唯一
  `SystemC::systemc` target和五项build/consumer/delta-event gate。CMake只消费validator产生的canonical snapshot，不联网、
  不搜索宿主package；基础compiler与plain C++ kernels仍可独立构建。Q22.S在该依赖上继续闭合RTTI隔离bridge、private
  rank/tile memory、typed effects、SystemC process/event、Direct DTE和atomic result；bootstrap smoke或direct shim仍不能
  替代正式component gate；
- Q22.N已经把SoftFloat/TestFloat 3e、GNU m4 1.4.21、GMP 6.3.0和MPFR 4.2.2的完整source digest、license、
  thread/rounding环境、唯一CMake target和上游self-test纳入受管依赖；Q22.B又把oneDNN 3.12的commit、archive/library
  digest、SEQ build和MatMul/reorder smoke纳入独立受管record。
  formal numeric tests缺任一该family必需的independent oracle或trusted-TCB conformance dependency时明确unavailable，
  不能以host `float`替代。Q22.N已经固定SoftFloat specialization/`THREAD_LOCAL`、MPFR TLS/runtime版本、self-tested
  MPFR/GMP artifact到实际loaded shared binary的digest/version/transitive exact-match；Q22.B已经固定oneDNN SEQ
  dispatcher、caller-worker环境、host platform digest和静态binary identity，MPFR/GMP的LGPL交付义务仍由发布配置承担。
  基础compiler
  不依赖oneDNN，但完整Q22 profile缺oneDNN时必须标记bulk execution unavailable、正式大GEMM gate未完成；
- 消费Q0.L在tasks/11/14 owner中闭合的shared typed logical-format/physical-encoding registry和typed target
  profile/target identity/Kernel Runtime ABI，供verifier、target lowering、CRT conformance与model共同使用；
- plain C++ kernel/property tests不链接SystemC；SystemC test executable使用唯一`sc_main`入口并实际运行，不能只编译、
  skip或用`gtest_main`替代；
- external授权/事实源gate通过后才形成同一repo CRT wrapper的device/host platform contract，移除源码内强制
  `USING_RISCV`选择，明确rank-local Direct DTE state、checked address和blocking-yield边界；未通过时不修改vendor材料或
  实现其派生operator/packet seam。

完成：文档、typed capability、依赖决策和failure分类收敛；未实现symbol/profile在任何mutation前可被完整枚举拒绝。

### 10.2 Q22.N Multi-dtype numeric foundation

```text
Pipeline position:
- Upstream artifact / IR:
  Q0.L已提交并验证的TargetProfile、LogicalFormatDescriptor、TargetDataFormatCodeRecord、
  TargetFormatEncodingRecord和TargetConvertRoute registries，tasks/08唯一layout helper，以及tasks/11当前typed
  instruction command surface；不依赖Q22.L target LLVM bundle、ELF或package。
- Current stage responsibility:
  建立显式ModelProfileId、validated NumericCommandKey、完整NumericSemanticsProfile、13-format raw codec、
  capability/comparator closure、formal numeric kernel和invocation/thread隔离的execution context；任何numeric effect前拒绝
  unknown、duplicate、unsupported或缺少policy的tuple。
- Output artifact / IR:
  immutable process-local numeric registry/profile、FormalNumericResult/status和可readback dependency-conformance evidence；
  不形成compiler IR、package、serialized shadow program或新的command schedule。
- Downstream consumer:
  Q22.B查询codec、semantic profile和formal backend requirement；Q22.S调用同一formal kernel；Q22.V重放完整source vertical。
- User-level driver / named pipeline:
  Q22.N没有独立production CLI；configured component gate直接验证该library。用户链路只在Q22.S完成后经同一
  wafer-compile target-model mode消费，wafer-opt/pass不能成为numeric旁路。
- Explicit non-goals:
  不扩大compiler target legality，不接oneDNN、SystemC、Host CRT或Target LLVM，不复用其它执行consumer的kernel，
  不把model-only profile、SoftFloat/MPFR默认行为或有限corpus声明为hardware numeric事实。
- Completion gate:
  tasks/16 §11.1和本文§8.1/§10.2的codec、36-route、rounding、formal family、dependency、state isolation、
  capability closure及unsupported审计全部实际执行；required test不得unsupported/skipped。
```

- 消费Q0.L在tasks/14建立的唯一`LogicalFormatDescriptor`和按target profile×engine×format分派的
  `TargetFormatEncodingRecord`，实现由descriptor索引的13种raw codec与TF32 container/semantic-width处理；Q22.N不复制
  registry。tasks/14 encoding只拥有ABI/register code、engine legality与format-specific constraint；exact command的
  typed layout另行由tasks/08唯一helper验证并与constraint交叉，Cx/NCx/BOOL几何不进入numeric
  registry。对invalid f64/unused及无证据engine×dtype row显式拒绝；
- 建立`(ModelProfileId, NumericCommandKey) -> NumericSemanticsProfile`唯一映射和capability三维状态；不从
  op/symbol/string恢复compute、product、accumulator、rounding、overflow或optional-field顺序。Q22.N只提供供Q22.B查询的
  typed backend需求和默认rejected状态，不创建bulk admission；
- 用受管LLVM pin内的APFloat/APInt完成FP16/BF16/FP32/TF32基础算术、convert、FMA、逐op status及无C++ UB固定位宽整数；
  production不得调用已退役accepted-IR helper。独立SoftFloat adapter交叉FP16/FP32，TestFloat的slowfloat路径验证SoftFloat自身；
  MPFR-backed formal path完成sqrt/tanh/exp/rsqrt等高精度结果并为BF16/TF32提供高精度differential；
- 一次闭合36条convert route、101条四种确定性rounding/plain执行row、88条elementwise（84条floating加4条BOOL logic）和
  F16/BF16/F32三条GEMM formal row；zero-point和stochastic保留命名candidate policy及区分向量。native F32 sum
  reduce闭合+0 accumulator、logical row-major input递增和逐step RNE formal row，其余15条native selector保持拒绝；
- `FormalNumericExecutionContext`只聚合invocation-owned model status，effect-free scalar/tensor evaluator在完整成功后原子commit；
  APFloat每次调用显式传入rounding，首个profile固定gradual、no-DAZ、no-FTZ。MPFR wrapper与SoftFloat oracle adapter各自
  保存/恢复完整环境；component tests覆盖immediate caller ambient scope嵌套LIFO、normal/early/error return和双OS-thread
  TLS，工程`-fno-exceptions`且不声明C++ exception测试，不在本阶段依赖SystemC process；
- 执行第8.1节Q22.N子项的exhaustive、boundary、property、metamorphic、independent-oracle和capability closure tests；生成的matrix
  必须区分model-implemented、compiler-emittable和hardware evidence，不能以f32 workload代替。

完成：13种logical storage codec和有证据的physical engine/layout row闭合，当前七种compute/convert format及36条convert
route无missing/duplicate；每个published `(ModelProfileId, NumericCommandKey)`有唯一semantic profile/formal kernel/comparator
或静态unsupported reason；formal backend逐family independent/trusted-TCB gate和execution-context isolation通过。该阶段
不产生oneDNN admission，也不声明任一未知edge policy为hardware事实。

完成证据：feature-on受管依赖记录包含20个artifact、9份license文本和23项conformance gate。2026-07-15综合重放中
base/numeric分别164/164、47/47，lit为250 pass/2个预期feature-inverse unsupported，CTest 22/22；feature-off base
164/164，lit为249 pass/3个明确feature unsupported，CTest 12/12。没有required numeric test被skip/unsupported。
该证据只签发model-only numeric foundation，不签发bulk、SystemC或hardware profile。

#### 10.2.1 首个model-only policy closure

首个且无默认值的opaque `ModelProfileId`固定为`wafer-model-formal-deterministic-v1`。它只选择以下确定性model语义，
不进入compiler `ExecutionConfig`、target legality、package或hardware evidence。完整typed record及其digest才是定义，调用方
不得解析spelling推导字段：

- 所有multi-byte scalar codec使用little-endian。numeric TF32使用32-bit container的bits 31:13作为`s1e8f10`，encode清零
  low 13，numeric decode遇noncanonical low 13非零即拒绝；raw movement仍保留全部bytes。BOOL physical bit
  ordinal只能来自tasks/08 owner helper，byte内LSB0/MSB0必须由显式target encoding或model profile
  选择；首个LSB0候选只标model-only，不能复用已退役reference私有mapper。Cx/NCx bitpacked block/tail事实尚未
  固定时必须在codec effect前拒绝；
- rounding mode 0/1/2/3分别为RNE/RTZ/RTP/RTN。23条rounding route逐mode发布，9条plain route固定RNE且禁止额外attr。
  mode 4只保留未发布`wafer-model-seeded-stochastic-v1`候选，在显式seed、PRNG、reset/state及每dynamic element advance
  合同闭合前以`stochastic-state-unproven`拒绝；
- 四条I8-source zero-point route保留typed key与subtract/add/raw等区分向量，但当前全部以
  `zero-point-formula-unproven`拒绝，不能猜测常见`x-zp`公式；
- float-to-int只在finite且目标范围内发布；NaN、Inf或overflow使整条command `reject-no-write`。float numeric op把sNaN
  quiet并置model invalid，qNaN不置invalid，输出统一canonical positive qNaN且不传播payload/sign；gradual underflow、
  tininess-after，flags只作model diagnostic，不映射真实Tsm status。确定性formal comparator为raw-bit与model flags exact；
- max/min任一NaN产生上述canonical qNaN；max的混合signed-zero为+0且仅双-0保留-0，min的混合signed-zero为-0且仅双+0
  保留+0。其它exact/convert路径保留signed zero；
- 首个plain GEMM capability只发布F16/BF16/F32：operand exact decode，F32 fused FMA accumulator从+0开始按K递增，
  每次FMA写点RNE，destination按RNE写回原format。narrow-fused、narrow-unfused、wide-unfused、TF32-to-F32和I8-to-S32
  只作为不同显式candidate/profile的component区分语义；当前NE×TF32及same-type I8 destination合同不允许它们冒充command
  capability；
- native F32 sum reduce按init-first +0、logical row-major input递增和逐step RNE执行；其它kind/format因
  init/identity/order未闭合而拒绝。`exp_lp`、`satrelu`、`leakyrelu`以及conv/pool/unpool/rand/LUT/argmax等缺少参数、
  tie、coordinate、RNG或accumulator政策的family同样静态拒绝，不能由wrapper存在推导支持。

36-route closure严格来自`TargetConvertRoute`：4条zero-point、9条plain、23条rounding；numeric层只保存validated route
identity和optional field，不复制opcode/src/dst/parameter-kind事实。capability同时保留`model-implemented`、
`compiler-emittable`、`hardware-evidence`三轴；codec存在不得扩大tasks/14的65-row engine legality。

首个profile的finite selector closure固定如下。数量是registry/verifier合同，不是workload覆盖率；shape、layout和具体参数仍留在
exact `NumericCommandKey`，不会被折叠进selector或由名字恢复：

| family | selector closure | model-implemented | 静态拒绝 |
| --- | ---: | ---: | ---: |
| CT convert | 128 | 101（9条plain RNE + 23条rounding route × 4种确定性mode） | 23条stochastic mode、4条zero-point route |
| CT elementwise | 128 | 88（51条F16/BF16/F32 APFloat、33条F16/BF16/F32 MPFR、4条BOOL logic） | 31条integer policy、9条缺参数policy的op |
| NE GEMM | 4 | F16、BF16、F32共3条 | I8 destination/accumulator policy未闭合 |
| native CT reduce | 16 | F32 sum共1条 | 其余15条kind/format因init/identity/order未闭合拒绝 |

因此registry总计276个不重叠selector。formal tensor executor只消费已resolve且supported的convert、elementwise、GEMM和
F32 sum reduce；它在
output分配前完成profile/command、arity/count、canonical encoding及caller-owned scalar/FMA双预算preflight，整张tensor
成功后才一次commit aggregate model flags。MPFR的Sqrt/Rsqrt/Log2/Ln/Pow2/Exp/Sin/Cos/Tanh/Sigmoid/Softplus
published row只接受F16/BF16/F32同格式RNE；TF32和directed direct-op路径只作component evidence，不能扩大compiler surface。

### 10.3 Q22.B oneDNN bulk qualification

- 受管oneDNN 3.12由唯一`WaferBulk::oneDNN`静态target提供，canonical dependency record绑定commit、source/archive、
  headers/library digest、license、SEQ/INFERENCE/MATMUL/REORDER options和smoke结果。该feature默认关闭；启用时numeric依赖、
  root或record不完整立即configuration fail，关闭时基础binary无oneDNN/OpenMP/TBB closure。
- 独立`BulkExecutionEnvironment`固定首版x86-64 SEQ caller-worker、default max ISA、no hints、cache 0、RNE、FTZ=0、DAZ=0
  和全部异常masked，并绑定且在复用时重验host CPU/features、effective ISA、kernel/libc/loader、affinity及CPU/NUMA
  topology。执行RAII恢复caller
  fenv/MXCSR；sticky flags不进入identity，也不映射为target status。
- 首版`BulkBackendAdmission`只覆盖Q22.N已经发布的F16/BF16/F32同dtype GEMM。target-owned codec/layout先把Cx/NCx physical
  snapshot精确提升为F32 dense input；oneDNN执行一次F32 MatMul及最多一次weights reorder；Q22.N formal finalize再舍入为原
  dtype并target-pack。native host FP16/BF16 availability不改变该contract，TF32和integer仍保持未准入。
- `calibrate -> freeze -> validate`以canonical、absolute、no-alias、no-replace artifacts冻结held-out spec/payload、destination、
  semantic、adapter、resolution、environment和comparator。有限F32-exact corpus只签发payload精确收缩的
  `profile-bounded` row，不把raw-exact observation冒充bit-exact证明；final record readback后runtime才可构造exact-match
  admission。
- formal和bulk分别使用显式work/byte/scratch/reorder budget。64³ component row超过runtime formal FMA budget后仍只通过
  admission调用一次MatMul，formal FMA为零；缺record、identity/payload漂移、reference implementation、budget或descriptor
  失败均无scalar fallback和partial result。

完成：F16/BF16/F32、Cx/NCx tail、batched row、environment drift、artifact tamper、多字段malformed JSON、budget和
feature-on/off link closure均已通过；真实CLI三阶段及readback admission已执行。source-backed自动dispatch、SystemC事件和
完整输出仍由Q22.V闭合，板端numeric及timing分别属于Q22.C/Q22.P。

### 10.4 Q22.L Owner-backed target LLVM module bundle

- 把tasks/14 transaction-local prepared target LLVM/ABI artifact提升为owner-backed、move-only、不可序列化的all-rank
  `TargetLLVMModuleBundle`；不重新运行另一套lowering，也不从Q17 ELF或manifest反推module语义；
- 每rank独立`LLVMContext`拥有module；module-owned typed metadata记录schema、logical rank、entry、profile/target identity、
  Kernel Runtime ABI和ordered typed slots，并连同module identifier、closed RISC-V triple和fixed `void(i64...)` entry从
  LLVM module本体readback。全部rank通过后才原子形成bundle，late failure不保留partial owner；
- 现有Q17 device link直接消费bundle中的同一module形成`TargetArtifactBundle`，不重复ABI preparation/lowering/translation。
  bundle同时保留给Q22.H和direct ABI smoke，不调用host CRT、不构造packet、不链接SystemC，也不进入serialized package。

完成：真实rank-count=1/16 producer形成all-and-only owner-backed target LLVM modules，identity/ABI/module readback和late-rank
atomic negative通过。该row不需要vendor授权，也不证明Host CRT或model execution。

新鲜完成证据：1-rank direct producer验证context lifetime、move-only ownership及module/slot readback，missing profile/entry/slot
metadata均被拒绝；rank-15 target failure无bundle/ELF/package。正式driver已拆成
`ExecutableBundle -> TargetLLVMModuleBundle -> TargetArtifactBundle`，rank1/rank16 linear和16-rank tiny Llama的ELF、manifest、
reference及no-card纵向重放通过；2026-07-15综合重放中base 164/164、lit 250 pass/2个预期feature-inverse unsupported、
CTest 22/22。
Q22.H现已从该bundle解锁并按repo-owned target-call frontend推进。

### 10.5 Q22.H Repo-owned target-call frontend

- 只消费Q22.L `TargetLLVMModuleBundle`；host clone/JIT执行same fully legal target LLVM，不形成另一份instruction lowering或
  改写bundle；host retarget前拒绝target intrinsic、inline asm、未知address space和非registry external call；
- 稳定Target层的shared typed target-call registry由lowering和frontend共同消费，拥有symbol、exact signature、call
  family和field decoder；109项descriptor均需通过all-and-only signature/payload gate。完整symbol只允许exact ABI-key
  lookup，不能在JIT/SystemC/test复制字符串表，也不能从前后缀、参数数量或任意字符串启发式恢复语义；
- 每rank生成`void(const uint64_t *slots)` fixed thunk和exact-signature bridge；bridge显式携带invocation/rank context，
  实际动态call形成typed transaction并同步投递transaction-local sink，Direct DTE返回invocation-local opaque identity；
- 全部rank先materialize/preflight并复制完整ordered typed slot metadata/value bindings，再由下游先创建全部rank process、每process调用一次
  `executeRank`；rank使用not-started/running/terminal防止yield重入。sink提供
  begin/issue/terminal/prepare-commit/infallible-commit/abort，host retarget、missing/wrong signature、slot、reentry、
  callback、destruction和late-rank failure均无partial executable/result。

完成：同一Q22.L all-rank target LLVM实际经过repo-owned typed target-call frontend，symbol/signature/context/sink closure完整且
failure无partial result。该gate不编译repo CRT、不构造Tsm packet，不证明CRT/packet/vendor-exact或SystemC numeric。

### 10.6 Q22.S SystemC functional-event model

- 以受管SystemC 3.0.2和唯一`SystemC::systemc` target建立默认关闭的feature；feature启用时缺依赖必须configuration fail，
  compiler和plain C++ numeric kernels保持可独立构建；
- 消费Q22.H实际typed transaction，建立rank-local virtual SPM/DDR、typed slots、checked address、invocation error latch、
  单一保守logical issue domain、resource event、local drain、可yield wait及Direct DTE/FSM；不声明worker window、
  `3×5`物理queue或engine复制；
- shared decoder对109项descriptor逐ABI字段形成typed payload，所有payload family进入统一field-valid validator；
  RDMA/WDMA、gather/scatter、memset、elementwise、convert、GEMM和Direct DTE control
  具有checked functional effect，其它field-valid family保持结构化unsupported。首个profile的DDR/SPM都在invocation-private
  registry内完成，不建立没有consumer的TLM socket；future ISS/interconnect只能通过另行设计的受限TLM边界接入，packet/MMIO
  correlation仍是Q22.K可选provenance gate；
- numeric effect只调用Q22.N formal profile；Q22.B bulk在本stage不是完成前置。plain C++ kernel component tests不链接SystemC；
  五个SystemC integration executable各有唯一`sc_main`，16个rank `SC_THREAD`加control process跨delta执行numeric和DTE；
- delta-cycle component gate证明issue后结果尚不可见、local fence检查已完成ordinal watermark、completion一次commit、failure
  唤醒且不copyback；unknown transaction、address exact-end/overflow/cross-resource/reserved-SPM和DTE no-progress均结构化失败。

完成：同一Q22.H正式producer的16-rank elementwise和collective-permute实际进入SystemC；numeric测试观察sticky inexact，DTE在
sender/receiver匹配完成点读取source并发布destination，metadata mismatch和missing endpoint/no-progress均唤醒waiter且无partial
result。unknown event同样必须latch failure并唤醒waiter。2026-07-15综合重放中feature-on base/numeric/SystemC component
分别164/164、47/47、5/5，lit为250 pass/2个预期feature-inverse unsupported，CTest 22/22；feature-off base 164/164，
lit为249 pass/3个明确feature unsupported，CTest 12/12且numeric/bulk/SystemC link closure通过；shared physical codec的
bulk 14/14回归通过。该证据不要求完整source workload，
也不发布repo CRT/packet、RISC-V ELF执行、board numeric、性能或timing claim。

### 10.7 Q22.V Source-backed functional-numeric verticals

- 固定真实`linear-residual-mlp-f32` rank-count=1作为首个vertical：当前case参数batch 2、input 16、hidden 32、output 16、
  f32、tanh和residual都只是测试参数，不进入capability协议；
- 重放产生Q22.L artifact的正式producer chain，经`wafer-compile -> ExecutableBundle -> TargetLLVMModuleBundle -> Q17/Q18 publication ->
  repo-owned target-call frontend -> SystemC model`真实链执行；手写LLVM/transaction只补negative；
- Q20 GEMM必须实际命中Q22.B冻结的admitted oneDNN row；另以强制formal backend的小shape重放同一semantic profile，按该row
  exact/bounded policy比较，证明backend选择不改变target semantics。完整output与NumPy expected按case已有
  `atol=1e-6, rtol=1e-5`比较并检查全部status/transaction family count，不只抽查元素；
- 增加source-produced f16、bf16 simple GEMM；把large GEMM exporter固定为source/config/payload、独立expected和digest，
  mandatory shape超过formal budget并自动命中Q22.B admission，且SystemC event/transaction数量不随M×N×K按per-MAC增长；
  4096 shape只保留为stress参数；
- 执行Q21 16-rank tiny Llama，覆盖batched GEMM、reduce、exp/rsqrt、i1/select、Direct DTE和all-rank output；
- 最后一个WDMA/transaction/kernel及任一rank late failure不形成model result，已发布Q17/Q18 artifact仍保留；SystemC-enabled CI
  必须实际执行，unavailable/skipped或unsupported-reason closure不算通过。

完成：Q20 formal/admitted双路径、f16/bf16 formal、deterministic 64³ admitted large GEMM和Q21完整输出已经通过；bulk
自动dispatch、错误record/no-scalar-fallback、output mismatch、DTE no-progress及all-rank atomic result已有覆盖。在Q29
structured scheduling / SPM residency接入后的当前主线中，large case形成10个target transaction、一次MatMul和零bulk
formal FMA；Q21形成12656个transaction、17个SystemC thread process。计数变化来自显式tile-region fence和以SPM gather
替代部分DDR movement，数值输出仍按同一source/reference合同匹配；因此这里记录当前确定性target-call拓扑，不把历史调度
的事务数当作长期协议。
Q21同一source/driver链还在已发布package与reference comparison之后于logical rank 15 terminal注入失败；该test-only seam
只存在于`wafer-compile-test`，验证稳定stage/rank诊断、无matched model result及manifest/rank module保留。
Q22据此只发布`target-call/SystemC model-only functional-numeric`，仍不称repo CRT、packet、hardware numeric、exact
package/ELF、性能或timing model。最终fresh suite数量保留在对应归档证据，不复制到任务队列。

### 10.7.1 Q28 Llama-2 7B 单 Block Scale Vertical

Q28把source vertical从语义最小case提升到标准Llama-2 7B单层shape：H=4096、I=11008、32 heads、
head_dim=128、FP16、batch=1、sequence=16、TP16。硬件单卡64/128 GB DDR足以容纳7B级resident weight，
而每tile 3 MiB SPM要求projection、attention和MLP用显式DDR slice与时间tile执行；因此该case用于验证空间切分、
时间分块、memory planning、instruction/completion、target LLVM和CModel组合，不以整体weight放入SPM。
它是规模consumer，不是planner通用性或orientation capability证明；CModel不得包含Llama op序列/名字/shape fast path，
其它图只要形成相同最终TargetCall就走同一transaction/kernel。

大GEMM不得走formal逐MAC路径。Q28增加独立`managed-reference`执行政策：逐command核对已发布GEMM semantic、
shape/layout、受管oneDNN environment、finite value-domain和显式byte budget，再由完整PyTorch expected tolerance约束
最终结果。它不签发Q22.B exact qualification record，也不声明raw-exact target arithmetic或hardware correlation；结果中
必须分别记录managed environment digest和exact admission digest。不能把任意oneDNN支持、host dtype或未检查payload
当作target能力。accepted time tiling必须由structured IR和SSA completion表达，不在CModel内重建compiler schedule，
也不通过提高4096 eager常量伪造规模支持；本固定case若实际展开低于预算，无需为形式上的scale预先引入loop。
F16/BF16输入到oneDNN F32 descriptor的widening是精确bit mapping，不调用逐元素APFloat；输出仍按已解析target
finalize语义舍入。当前受管artifact固定`DNNL_CPU_RUNTIME=SEQ`且关闭primitive cache，因此Q28 Release耗时是功能慢测
基线，不代表多核性能；未来threaded artifact必须作为新的受管依赖/环境身份闭合，不能只继承ambient线程设置。

当前Q28使用v1 implicit normal/normal plain target GEMM：plain form只接受exact rank-2 `Cx`；batched form只接受exact rank-3、single-leading-batch、canonical
`[B,M,K] x [B,K,N] -> [B,M,N]`的`NCx`。target call不携带自由rank/layout/dimension-map字段，因此rank >= 4或
permuted batch axis会丢失physical bank boundary，必须在target ABI前显式canonicalize，否则verifier拒绝。
`B=1`时`NCx[1,M,C]`和`Cx[M,C]`的footprint/offset完全相同，已由跨两个channel block和`C0` tail的property
回归证明，所以CModel按target call的`batch_count`选择layout不会在该特例产生byte歧义。
Q32.V的目标v2在保持rank/layout边界的同时用typed orientation决定stored operand relation。tasks/14
exact-signature及repo-owned formal/SystemC语义闭合后compiler才可把该versioned command交给Q32.M/S消费；
external bulk-model admission只决定相应model consumer能否执行已经发射的command，对应board row只决定board
consumer能否执行及升级profile。external admission不参与planner candidate生成、过滤或排序；Q28当前19,696条
transaction和既有误差结果不能外推为
oriented GEMM证据。

Q28的movement功能执行保留descriptor本身，而不是把descriptor提前展开成per-segment effect。规则nested stride先以
checked span、单resource resolve和non-overlap stride合同完成快速验证，再从一份source snapshot提交strided destination；
其它ABI合法排列允许线性枚举并按区间排序验证。两条路径都必须在任一写入前完成全部range/resource/overlap检查，保持
gather/scatter source-before-write及命令级atomic effect；不能用性能优化放宽descriptor预算或静默接受重叠destination。

同一显式policy还拥有非GEMM的tensor functional lane，但不扩大target capability registry：只消费已经resolve的F16/F32
elementwise、F16/F32 nearest-even convert和native F32 sum reduce，在command级检查arity、shape/layout、non-NaN值域及
scalar/byte budget后使用host native tensor kernel批量执行；attention causal mask所需有符号infinity按IEEE运算保留，
GEMM仍保持finite-only。`formal`继续作为小规模raw-exact scalar oracle，
`prefer-admitted`继续只允许exact-record GEMM，二者的非GEMM均不得进入该lane；`managed-reference`遇到不支持的
dtype/op/rounding或该non-NaN值域外输入输出必须在effect前失败，不能静默回退formal。SystemC仍只调度transaction/event，
functional kernel不把逐元素操作建成SystemC event。结果分别记录formal、managed-reference tensor和bulk command数量，
完整PyTorch tolerance comparison是该lane的必要下游consumer，不把它解释为raw-exact或硬件相关证据。

固定source corpus在artifact/digest publication前必须拒绝任一nonfinite input、parameter或expected。F16全张量comparison
按解码后的finite数值应用显式atol/rtol，不得沿非F32分支退化为raw bytes；raw-exact仍只属于非浮点storage和另行冻结的
bit-exact capability row。

该scale consumer已经实际完成：16-rank production target LLVM形成19,696个typed transaction并由17个SystemC thread
执行；managed tensor/bulk分别为2,032/672条command，672条bulk均实际调用MatMul，formal command/FMA为零。完整
65,536-element F16 output通过固定PyTorch eager tolerance；错误expected和scalar budget负例都在不回滚已发布package的
同时拒绝matched model result。这里的计数用于证明没有scalar fallback，不是性能或cycle模型；完整corpus、负例和suite
证据由tasks/16及归档实施记录拥有。

完成边界和负例由tasks/16的Q28 gate拥有；完整32层、KV/autoregressive、board correlation和timing仍分别后续闭合。

### 10.7.2 Production vertical性能边界

Q30优化Q28已经发布的完整host production vertical，不预设瓶颈位于compiler还是functional-reference backend。
2026-07-17 fresh profile先定位tile-region到instruction的静态movement descriptor构造；收口该热点后的stage timer又定位
ProgramTensor physical encoding和bulk/tensor command中的重复physical codec traversal。oneDNN实际MatMul/reorder不是主要成本，
因此不修改thread runtime或受管dependency identity。本节规定跨stage不变量，shared geometry由tasks/08、instruction lowering
由tasks/11拥有。

任何性能修改都必须保持formal/tensor/bulk计数、exception/evidence、SystemC delta、DTE/completion、pending effect原子提交、
完整output bytes和PyTorch expected differential。若后续profile显示physical layout、logical value、finite/non-NaN检查、
dense adapter或oneDNN descriptor在一个command中成为主要重复工作，应由command-local typed plan合并；跨command reuse只能由
backend/invocation拥有，并以environment、resolved command、tensor key和immutable payload identity构成完整key及显式容量
预算。进程全局cache、buffer地址、slot名和ambient线程变量都不能成为正确性或reuse协议。

ProgramTensor invocation encoding、managed tensor lane和bulk GEMM adapter统一消费tasks/08的
`WaferStaticPhysicalOffsetCalculator`；byte-addressable physical traversal流式读写logical values，不建立element-count大小的
offset side table。bulk F16/BF16/F32输入经strict physical decode后可以直接做位级F32 widening，不得在adapter中对同一
`RawLogicalValue`再跑一遍canonicalization；noncanonical/value-domain拒绝仍由decode和managed admission在backend effect前
完成。该共享只覆盖layout/codec事实，不共享PyTorch expected、compute result或command effect。

threaded oneDNN属于受管依赖身份变化，只有在fresh测量证明SEQ compute是主要剩余瓶颈后才能引入，并需固定worker政策、
link closure和environment digest，重新通过qualification及完整7B数值。Q30实施与完成证据归档于
`tasks/archive/llama-block-production-performance.md`。

### 10.7.3 Source/model numeric statistics

最终ProgramTensor comparator可以在调用者显式请求时报告decoded finite output相对同一source expected的
numeric-exact、absolute-error和destination-format ULP分布。statistics是只读验证artifact：不参与tolerance结果、backend
admission、SystemC调度、effect提交、package或environment identity。absolute quantile和ULP quantile均包含exact元素并使用
nearest-rank；ULP从F16/BF16/F32 canonical raw encoding构造sign-aware monotonic key，当前numeric comparator视为相等的
`+0/-0`距离为0。NaN/Inf和未发布floating dtype仍在统计发布前fail closed。

Q31用Q28固定seed和两个预先冻结、明确`admission=false`的diagnostic seed验证了更严格的source/model tolerance。三个seed
的全部rank均通过`atol=0.004, rtol=0.002`，该值现为scale case的source/model comparator policy；逐rank完整统计中的共同
`p99_abs`为`0.0009765625`，跨seed/rank最坏`max_abs`为`0.0029296875`。ULP在near-zero处会放大，继续只作diagnostic。
该结果只收缩同shape/dtype/payload算法的有限empirical regression policy；不能改写formal/managed semantics、Q22.B exact admission，
也不能提前成为Q22.C board profile。具体seed、阈值和完成证据由tasks/16及
`tasks/archive/llama-block-numeric-characterization.md`拥有。

### 10.8 Q22.C Board numeric correlation

- 当前v1先以当次compiler发布并验证的package/profile fresh重放Q6.B board provider correctness、
  watchdog/reset和重复invocation；它不等待条件性Q32.V package capability字段；
- board environment在任何effect前匹配本次current command rows；若测试Q32.V扩展且已采用capability-bearing package，再对同一revision的
  `RequiredCapabilitySet`逐key匹配allowlist；
- 按第8.3节的capability row、区分向量和comparison policy采集calibration corpus；
- 冻结policy后运行独立random/shape/layout/optional-field held-out及Q20/Q21完整output；
- 发布绑定environment identity、tested domain和raw evidence digest的profile；packet capture不可用时降级packet
  provenance，不阻塞board-output correlation。

完成：只有实际通过的row从`model-only`逐级升级，失败/未测tuple保持unsupported。Q22.C不声明exact ELF、timing或未测
输入域等价。

### 10.9 Q22.E Exact package execution

- 在取得vendor simulator或完成RV64 ISS、loader ABI、MMIO/custom instruction和provider lifecycle后接入
  当次compiler生成且Q18验证的package identity；Q22.E不以无consumer的Q32.V package schema为前置；
- 若执行Q32.V mapped DMA、oriented GEMM或其它扩展command，且对应consumer采用capability-bearing package，必须消费并验证同revision的
  `RequiredCapabilitySet`；当前v1 exact-package路径不得冒充已支持这些扩展；
- 原样执行all-and-only Q17 modules，不发布host专用instruction list或修改manifest；
- 通过wafer-run typed provider入口对本次package实际声明的能力做preflight，再执行Q20/Q21 package和阶段性
  failure injection。

完成：同一package在model provider中完整allocate到cleanup并产生可信status/output；否则该能力保持更高待解锁gate。

### 10.10 Q22.P Deferred timing calibration

- 本阶段不在近期numeric correctness范围内，只有另行恢复并配置可信PMU/timing environment后才执行；
- 先验证PMU measurement basis，再按single-engine、queue/SPM/DTE/fabric矩阵采样；
- 用独立timing held-out shapes/descriptors/workloads验证loosely/approximately-timed profile；numeric policy保持只读；
- cost calibration如需消费结果，另由tasks/06/16和Q9建立typed consumer，不反向污染correctness。

完成：只发布实际通过的profile标签、适用device/firmware/runtime identity、误差分布和未覆盖范围。没有RTL或vendor
cycle证据时cycle-accurate保持非目标。

## 11. 后续外部门槛和待讨论问题

以下问题不改变SystemC作为Q22正式functional-numeric容器的当前选择，但会改变dependency、packet provenance和板端
实现成本，因此保留为显式待讨论问题：

1. **Vendor simulator交付**：现有低层header只有接口痕迹，高层x86 runtime会动态寻找缺失的
   `libcmodel_runtime_api.so`，DWFC/vendor CModel属于封闭交付且当前项目没有可用、可链接、可授权审计的完整开发包。
   本路线不再主动依赖或逆向该seam，Q22据repo-owned target-call/SystemC完成。只有项目owner以后明确提供完整host runtime、
   instruction model、依赖、资源、版本和license时，才重新把它作为Q22.K或Q22.E的候选，并仍需独立correlation。
2. **Packet事实源与授权**：需要确认可否复用vendor可审计builder或获得register trace，并确认当前采购条款是否允许host
   集成或独立实现。只有vendor交付的许可兼容provider，或经项目owner/法务确认、基于独立可审计规范的clean-room seam才能
   产生可用于Q22.K的host packet；它验证对应compiler/CRT/model链，但在独立correlation前不证明vendor packet完全一致。
   授权缺口不阻塞Q22.H/Q22.S/Q22.V，只限制CRT/packet/opcode provenance。
3. **SystemC工程基线**：当前Q22已固定受管SystemC 3.0.2、source/build digest、license、唯一CMake target和真实
   `sc_main` gate；未来若增加其它CI平台或ISS co-simulation，仍需单独固定其deterministic scheduling和TLM边界。
   SystemC只拥有event/transaction实现，不改变artifact、packet、功能核或compiler合同。
4. **板端numeric环境和验收政策**：需要固定可用SKU/revision/unit、firmware/runtime/instruction-library/CRT组合、
   reset/watchdog能力、single-op可观测性、calibration/held-out corpus和逐op/dtype comparator；未固定前不发布
   hardware-correlated numeric profile。LT/AT阈值只在deferred Q22.P恢复时讨论。
5. **数值依赖发布政策**：Q22.N已经固定SoftFloat/TestFloat 3e、GNU m4 1.4.21、GMP 6.3.0、MPFR 4.2.2及受管LLVM
   APFloat/APInt的职责、完整source digest、SoftFloat specialization/`THREAD_LOCAL`、MPFR TLS/runtime和实际MPFR/GMP
   binary identity readback。MPFR/GMP的LGPL动态交付、notice、relink/source-offer流程仍需在发布配置中确认；
   未确认时可以完成内部component gate但不能发布不满足许可义务的binary。这些只改变构建与交付方式，不允许改变
   `NumericSemanticsProfile`或用host native语义降级。oneDNN是完整Q22 profile的大GEMM执行依赖；其受管版本、CPU
   dispatcher/thread runtime和binary发布方式由Q22.B固定，但基础compiler不链接它。

## 12. 文档和实现归属

- tasks/14拥有target ABI preparation、fully legal target LLVM、CRT和RISC-V module publication；target LLVM bundle
  落地时应在同批同步其producer合同。
- tasks/15拥有exact target model `RuntimeProvider`生命周期和package消费；target-call/SystemC模式不是该provider。
- tasks/16拥有repo-owned target-call/SystemC、Q22 typed transaction/event、Q22.C board numeric、可选CRT/packet/MMIO、
  exact-module、board和deferred timing的证据分层及CI gate。
- 本文拥有target execution model内部边界、capability、SystemC选择、板端numeric correlation计划和各层不得冒充的声明。
- tasks/09/11/13仍分别拥有memory legality、instruction/packet legality和Direct DTE/completion；model不能改写这些
  compiler合同。

Q22 target-call/SystemC model-only实现已经完成。后续exact provider、board numeric、packet provenance或timing施工前仍应
建立独立实施计划；本文不表示这些更高gate已经完成。
