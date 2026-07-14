# Wafer Target Execution Model（CModel）

状态：2026-07-14已完成真实Q21 readiness、Q22.N formal numeric foundation和Q22.L owner-backed target LLVM bundle，
当前推进Q22.B oneDNN bulk qualification；Host-CRT/SystemC整体执行链仍未实现并受授权gate约束。本文固定以数值正确性为
近期目标的untimed target execution model、SystemC/TLM边界、实现分层和板端numeric correlation计划；timing calibration
仅作为deferred extension。任务状态看
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
- 在任何f32 workload vertical之前先建立覆盖当前全部13种logical storage format、有证据的target-profile×engine×format encoding、7种公开
  compute/convert format及完整product/accumulator/intermediate/destination、舍入/overflow/special-value政策的数值基础层；
  kernel只消费唯一`NumericSemanticsProfile`，不保留先写f32、以后再补dtype的旁路；
- 在vendor许可/采购条款经项目owner确认允许后，让同一份repo-local CRT wrapper源码的host build经过已授权的
  `TsmNew*`/operator/execute/MMIO seam、实际
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
- 不把host原生`float`、oneDNN/Eigen等CPU kernel的默认累加/舍入/量化行为或单一外部库当作target numeric合同；
- 不把direct host shim称为CRT/packet执行，也不把经授权独立实现的host packet称为vendor-exact packet或RISC-V ELF执行；
- 不在vendor许可范围和可用事实源经项目owner/法务确认前修改vendor材料、实现其派生operator/packet builder或把当前
  reverse-engineering资料当作实现授权；
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
  bundle还必须消费`CompilationRequest -> ExecutionConfig`端到端携带的typed target profile，以及该
  profile由tasks/14 registry唯一映射的target identity/Kernel Runtime ABI；当前profile不表示silicon revision，
  也禁止在full conversion后补default。bundle不可序列化，不进入package。exact-module扩展另行消费Q18 VerifiedPackageManifest和其all-and-only
  Q17 RISC-V ELF modules。Q19结果和CPU expected只作独立比较，不作为target model执行输入。
- Current stage responsibility:
  在任何执行副作用前完成model capability preflight；在许可/事实源gate通过后，以同一repo CRT wrapper的host build和
  已授权的CModel-compatible Tsm factory/
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
  用户通过`wafer-compile --target-profile=wafer-tx81-single-card-kernel-v1`显式选择tasks/14 typed
  profile，且不存在target profile默认值；首个正式CModel gate由
  wafer-compile在Q17/Q18原子发布后、同一invocation仍持有target LLVM bundle时进入显式
  target-model execution mode：host执行target LLVM并调用同源host CRT，CRT再向SystemC model发packet/event。
  wafer-opt/pass chain和direct shim只补局部测试。只有exact package/ELF执行闭合后，wafer-run才通过typed
  RuntimeProvider选择target model并消费verified package，不能用host-only模式冒充该入口。
- Explicit non-goals:
  不复制instruction schedule，不改变manifest语义，不用Q19执行结果驱动target model，不把untimed结果升级为
  board/timing/cycle证据，不用代表rank、手写LLVM、手写packet或单个kernel替代真实纵向链。
- Completion gate:
  Q0.L、Q22.N与Q22.L已经完成。Q22.N闭合13种logical codec、有证据的target-profile×engine×format encoding、当前7种
  compute/convert format、36条convert route、4种确定性舍入和逐family formal conformance；Q22.L原子形成all-rank
  owner-backed target LLVM bundle；该边界现已完成。Q22.N单独解锁Q22.B oneDNN qualification；Q22.L与external authorization/spec gate
  共同解锁Q22.H host CRT。Q22.N+Q22.H+既有Q16.T再解锁Q22.S SystemC functional-event model，Q22.B不是Q22.S前置。
  Q22.B+Q22.S+既有Q20/Q21最后由Q22.V重放Q20 f32、source-produced f16/bf16 GEMM、Q21 16-rank tiny Llama和
  deterministic source-backed large GEMM，覆盖all-and-only ranks、authorized host packet、typed ABI、SPM/DDR、
  supported engine和Direct DTE，并与独立Q19/CPU oracle比较完整输出；任一late failure无partial result，model mismatch
  不回滚已验证Q17/Q18 artifacts。Q22只汇总model-only untimed functional-numeric profile；board numeric、exact
  package/ELF和timing accuracy分别由Q22.C、Q22.E和deferred Q22.P保持为更高独立gate。缺许可兼容host seam或允许
  clean-room实现的独立规范只阻塞Q22.H及其下游；缺独立packet/MMIO事实源只限制vendor-exact packet claim。
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
- Q17正式交付物是all-and-only RISC-V ELF `TargetArtifactBundle`；Q18 package只包含typed resource/slot/module/
  entry/completion/transport requirement，不包含instruction schedule。
- Q18当前只实现pure no-card `RuntimeSessionPlan`。真实`RuntimeProvider`生命周期和board执行尚未实现。
- Q19直接消费accepted instruction/memory facts；其reference-only numeric和deterministic DTE policies不得成为
  target model的实现事实源。
- Q0.L shared registry显式枚举13种logical format，但engine×format准入只开放tasks/14有静态编码证据的row；
  UINT、64-bit和通用TF32 format-bearing command均在target effect前fail closed。`wafer.instr.convert`是独立例外：
  typed `InstrConvertKind`通过opcode 139..174的36条closed route选择含TF32的wrapper，但这不会打开通用
  CT/RDMA/WDMA/GEMM的TF32 row。frontend可接收的f64在target profile中也没有descriptor。Q22 numeric foundation应
  实现TF32 raw32 codec并分别记录logical codec、convert可达性和engine-specific format legality；不能用model有codec扩大
  compiler legality，也不能让f64晚到packet阶段才失败。

近期实现应把tasks/14中已完成ABI preparation和full target conversion的结果提升为owner-backed、不可序列化的
**target LLVM bundle**。它是实际fully legal LLVM IR及typed slots的all-rank集合，同时被现有RISC-V link、direct
ABI-smoke和host-CRT/SystemC CModel消费；它不是新IR层、packet artifact或package成员。具体C++类名和文件布局只作
实现索引，不属于artifact合同。

repo CRT当前在源码内强制定义`USING_RISCV=1`。fresh host probe证明同一C源码和public header可由GCC完整编成x86-64
object，但只调用`wafer_tx81_local_fence`的最小链接首先缺`TsmWaitfinish`；完整object还要求36个Tsm factory/delete/
execute/wait入口和11个Direct-DTE/SPM platform入口。把现有`libinstr_tx81.a`加入会因其成员为RISC-V ELF
（EM=243、RVC、double-float ABI）而`file in wrong format`；`libcommon_util.a`、`liblibc_stub.a`和runtime archive同样不能
成为host provider。`op_fw_sim_if`的host CMake只是include-only INTERFACE target，checkout没有这些host定义。因此“同源”
当前只证明wrapper/command-invocation源码可host-compile，不证明host link或packet执行闭合。

后续若获授权实施，应把platform选择移到互斥build contract：device定义RISC-V platform，host真正不定义
`USING_RISCV`，不能传`USING_RISCV=0`冒充false，因为vendor header使用`#ifdef`。RISC-V和host只共享经确认可使用的repo CRT
wrapper源码；host operator/MMIO/memory/context必须来自vendor交付的许可兼容库，或来自项目owner/法务确认允许且具有
独立可审计规范的clean-room实现。当前Direct DTE sender/receiver还是process-global static，host多rank执行不能共享该
状态；必须绑定invocation/rank或证明每rank独立装载实例及lifetime，不能从OS thread、symbol名或调用顺序恢复rank。

仓库随附vendor最终用户许可协议第2.1节把许可描述为有限、自用、不可转让/分许可且可撤销，第2.2节在购买凭证没有另行
约定时禁止修改、逆向、反汇编/反编译、提取源码和创建衍生作品。本文不判断具体采购条款；Q22.H施工前必须由项目owner/
法务确认当前材料的实际授权范围，或取得vendor书面许可/公开独立规范。该external gate未通过时，可以继续实现不消费
vendor派生事实的Q22.N numeric、Q22.B bulk和Q22.L bundle；SystemC只可保留dependency/bootstrap candidate probe，
不能把Q22.S基础组件另行标成doing或完成，也不得施工host Tsm operator/packet seam。

该bundle必须显式拥有canonical all-rank domain、每rank logical rank/entry/module、`ExecutionConfig`（内含唯一
`TargetProfileId`）、ordered typed ABI slots、由该ID经registry解析并readback的target identity/kernel ABI facts，以及
覆盖全部module的context/owner lifetime。所有rank完成ABI
preparation、full conversion和readback后才能原子构造bundle并交给任一consumer；不能让device link和host model分别
从默认值恢复target facts，也不能让passing ranks先行执行。

当前`ExecutionConfig`需要能唯一选择target/ABI profile的typed identity。Q0.L由tasks/14 registry拥有
`TargetProfileId`，通过`wafer-compile --target-profile=wafer-tx81-single-card-kernel-v1`显式选择并写入
`CompilationRequest`/
`ExecutionConfig`，再贯穿profile-bearing ExecutableBundle、target conversion、transaction-local prepared target LLVM/ABI
artifact、`TargetArtifactBundle`和PackageManifest readback；它不能由target model、自由字符串manifest或host环境在末端恢复。
Q22后续创建的`TargetLLVMModuleBundle`只消费并再次readback该已验证identity，
不是Q0.L提前创建的artifact。该profile/identity、Q22.N numeric和Q22.L bundle gate现已闭合。

### 3.2 已确认的硬件功能和事务事实

当前静态证据足以支撑以下受支持子集：

| 范围 | 已确认事实 | 初版可声明 | 仍不可声明 |
| --- | --- | --- | --- |
| topology/memory | 当前目标单卡4×4、16 tile；每tile 3 MiB SPM，末64 KiB有Kcore/runtime占用证据；另有SPM alias和多个consumer-specific DDR mapping/address-view线索 | 当前profile的rank/tile隔离、SPM reservation和byte-level model window；DDR只按typed resource注册 | 所有SKU容量、bad-tile/PG行为或单一静态DDR范围可硬编码 |
| physical layout | compiler当前accepted compact、Cx/NCx规则可由typed layout重算；Cx/NCx是block-major而不是普通dense stride，INT8/UINT8 full block为128 lane、其它格式为64 lane，并存在compact C0 tail；hardware helper另有256B alignment/footprint证据 | source-backed accepted layout的logical coordinate到physical byte offset及越界检查 | Cx/NCx可直接表示成oneDNN stride；名称或256B规则是所有dtype/op的通用硬件layout |
| dtype/storage | `Data_Format`公开13种有效格式：INT8/16/32、UINT8/16/32、INT64/UINT64、FP16/BF16/FP32/TF32和bitpacked BOOL；storage width为1/2/4/8 byte，BOOL为1 bit/element | 13种logical raw codec；tasks/08拥有bit/byte footprint和layout geometry，tasks/14只对有证据的target-profile×engine×format发布ABI/register encoding legality | enum存在即证明每个engine都能搬运或计算该dtype；当前DMA helper只有0..7编码却把8..12当作已支持；把BOOL scalar helper的byte store当作bitpacked硬件合同 |
| compute/convert | 当前CRT公开INT8/INT16/INT32/FP16/BF16/FP32/TF32七种格式间36条convert route；代码检查显示typed convert lowering可选择含TF32的wrapper，但还没有target-lowering integration正例；`RND_MODE`有nearest-even、zero、positive-infinity、negative-infinity、stochastic五种值 | 七种格式的logical numeric descriptor、36条route结构和四种确定性rounding的model candidate；每条op另按完整语义profile和engine-specific format legality准入 | 通用format-bearing op已能发射TF32；code-reachable convert即已被集成验证；UINT/INT64 compute、任意op×dtype笛卡尔积、INT8-source zero-point公式或stochastic随机状态已由ABI证明 |
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

进入实现前应优先向vendor索取完整`libcmodel_runtime_api`套件或低层host instruction model、匹配headers/resources、
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

模型保留三个互不冒充的执行入口，以及一条可选的独立packet/MMIO证据路径：

1. **Direct target-call ABI smoke**：host LLVM JIT执行同一fully legal target LLVM module，并给`wafer_tx81_*` fixed ABI
   注册最小direct shim。它验证target ABI preparation、control flow、call graph、typed arguments和address formation；
   因为绕过repo CRT、Tsm method table和packet，只是快速诊断，不满足正式Q22 CModel gate。
2. **Authorized Host-CRT/SystemC frontend**：external授权/事实源gate通过后，host执行同一target LLVM，调用与device build
   同源且获准用于该用途的repo CRT wrapper；host platform由vendor许可兼容host库，或经确认允许的独立规范clean-room实现
   提供`TsmNew/Delete`、method table、`TsmExecute/TsmWaitfinish`和Direct DTE/FSM/memory hooks，将CRT实际构造的packet/event
   投递给SystemC model。这是近期正式functional-event CModel路径。独立实现仍只标记其实际provenance；逐字段独立
   packet/MMIO correlation前不能称vendor-exact packet。
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
SystemC只规定cooperative process语义，不固定OS-thread映射；`thread_local current_rank`不能代表rank。可行机制包括
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

- model top/context：target profile/identity、当前profile的16个physical tile slots、capability/good-tile map、logical/physical
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

SystemC调度粒度是一次target transaction/command，不是tensor element或单个MAC。GEMM transaction在plain kernel内一次
调用admitted oneDNN primitive并返回待提交buffer effect；绝不能为M×N×K循环创建`sc_event`、process或TLM transaction。
formal scalar loop只在checked work budget内服务小规模conformance、edge profile和失败诊断；大command超过budget且无admitted
bulk row时必须在compute前返回`bulk_backend_unavailable`，不能以隐式scalar fallback拖垮大网络。

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
| `TargetLLVMModuleBundle` | canonical all-rank domain、每rank独立LLVM context owner、logical rank/entry/fully legal module、ordered ABI slots、`ExecutionConfig`（内含唯一`TargetProfileId`）和由module metadata经closed registry解析并readback的target identity/kernel ABI facts | packet list、schedule、model state、默认补出的revision或任何可序列化sidecar |
| `ModelProfileId` | 显式选择的一组确定性model-only semantics identity；Q22.C可另发布绑定target/environment的hardware-correlated profile | compiler legality、隐式default、hardware等价声明 |
| `NumericCommandKey` | static preflight从typed CRT call registry投影、runtime从decoded packet transaction投影的target profile、engine/op kind/variant、operand role/storage dtype、shape/layout、M/K/N/batch、convert kind及fixed optional fields | erased Instr sidecar、任意symbol/string推断、numeric comparator、oneDNN选择、板端阈值 |
| `NumericSemanticsProfile` | 稳定typed identity/digest；完整operand storage/compute、product、accumulator、intermediate、destination、rounding points、FMA/reduction order、overflow、FTZ/DAZ、NaN/special/status及optional-field顺序 | 单个dtype、host library默认值、按workload临时覆盖的side table |
| `TargetModelCapability` | `(ModelProfileId, NumericCommandKey)`到唯一`NumericSemanticsProfile` identity的映射，以及shape/layout/descriptor、event/transport、compiler-emittable和hardware evidence状态 | 重复保存压缩的src/accum/dst key、由symbol存在推导支持、扩大compiler legality的规则 |
| `BulkBackendAdmission` | 完整semantic profile digest、shape/layout adapter、value domain、backend environment和`bit-exact/profile-bounded/rejected`结论 | target numeric semantics、target comparator或“library支持该dtype” |
| `FormalNumericExecutionContext` | non-yielding formal kernel作用域内的profile、model status与MPFR state save/set/clear/capture/restore和显式target status映射；APFloat/APInt每次调用显式传rounding | rank identity、跨SystemC wait的全局/TLS状态、SoftFloat oracle状态或oneDNN worker状态 |
| `BulkExecutionEnvironment` | oneDNN runtime/threads/ISA/implementation、worker initialization能力、caller fenv恢复和admission provenance | worker native flags到target status的映射、architectural memory或SystemC API |
| `TargetModelInvocation` | bundle引用、all-rank typed input/parameter/output binding、checked external resource registration、选定model profile | host pointer伪装的device address、从文件名恢复的rank/resource |
| `TargetTransaction` | `TsmExecute`返回前复制的packet字段、worker/engine、地址/descriptor、dtype/shape/optional fields和invocation-local sequence identity | caller栈指针、整程序command vector、Q19 schedule或DTE payload的无证据snapshot |
| `TargetModelResult` | all-and-only rank terminal status、完整output、stable failure stage/rank/transaction、frontend/CRT/packet/numeric/event provenance | partial successful rank集合、回写compiler/package的状态 |

`NumericCommandKey`只包含typed CRT registry或decoded transaction已经表达、并由typed target profile唯一解释的事实；它
不保留被erase的Instr program或shadow command list。`NumericSemanticsProfile`是所选`ModelProfileId`下这些事实对应的唯一
完整数值解释；`BulkBackendAdmission`只决定该解释能否由host bulk library
加速。三者分离后，target comparator、板端校准政策和oneDNN环境都不会反向污染硬件语义。实现支持谓词是
`compiler accepted tuple ∩ model kernel tuple`；板端profile仅在Q22.C进一步收窄，不能扩instruction legality。enum、CRT
symbol或packet decoder存在都不等于row已支持。capability不能压成单个布尔值，至少保留三个正交维度：
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
映射一个semantic profile。板端校准前可并存多个命名model/candidate profile，但每次execution必须显式选择并记录一个，
不能进入compiler legality、隐式default或CModel side table。Q22只发布确定性model-only profile；Q22.C再把board evidence
绑定到target revision/environment，选择或拒绝对应row并发布hardware-correlated profile。当前plain GEMM只有单一format且
要求lhs/rhs/dst同element type；f16/bf16 narrow/wide、TF32 product和i8 accumulator等分别属于不同显式model profile，
不能在同一次execution内按数据猜测切换。

`TargetLLVMModuleBundle`在所有rank完成ABI preparation、full conversion、LLVM translation和module readback后原子形成；每rank
module已经由独立LLVM context拥有。host执行后续把每rankmodule克隆/retarget到独立ORC `JITDylib`，避免同名entry/private helper碰撞；JIT clone设置native triple/data layout，
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
| LLVM APFloat/APInt | 受管LLVM pin内的正式基础标量后端；覆盖FP16/BF16/FP32/TF32、四种确定性rounding、逐op status、convert、FMA及无C++ UB固定位宽整数 | 不提供所需sqrt/exp/tanh/rsqrt等超越函数；不得调用Q19 helper或继承其policy，Q19因同库只能算integration cross-check |
| Berkeley SoftFloat/TestFloat | FP16/FP32独立IEEE differential和adapter/build conformance；`testsoftfloat`以包内不同slowfloat实现检查SoftFloat，BSD-3-Clause便于受管引入 | 不进入production kernel，不覆盖BF16、TF32、超越函数或Wafer optional-field政策；普通`testfloat`以SoftFloat作expected，不能再多算一个oracle |
| MPFR/GMP | tanh/exp/sqrt/rsqrt等高精度formal production backend，并为BF16/TF32基础路径提供高精度differential/TCB；destination/temporary使用显式precision，operation显式传rounding并设置exponent range | production使用MPFR时同一wrapper不能再算独立oracle；MPFR只有一种NaN且不原生模拟目标subnormal/payload，必须由raw codec/profile包裹；依赖和LGPL合规在引入前固定 |
| oneDNN | Q22中已准入大规模GEMM/MatMul的主执行后端；source-backed大矩阵不能长期用scalar逐MAC执行 | primitive、ISA、accumulation及math mode会影响结果；strict/deterministic不证明target等价；它不拥有Wafer codec、Cx/NCx、rounding、quant或reduction-order语义 |

Eigen、gemmlowp或host `libm`可以加入非规范性performance/differential实验，但不进入首版语义owner：Eigen fast-math和
vectorization会改变边界行为，gemmlowp只覆盖低精度GEMM且带自己的quantization合同。oneDNN则不是“以后再做”的优化：
Q22首版必须为实际source-backed大GEMM提供admitted bulk path；依赖必须版本固定、license可审计，基础compiler和no-card
runtime不因此链接这些库。

oracle独立性按算术实现和target adapter/codec的来源判断；同一library换precision、driver、executable或随机seed都不形成
第二个oracle。逐family最小矩阵是：APFloat production FP16/FP32与独立SoftFloat adapter比较，`testsoftfloat`的slowfloat
路径验证SoftFloat本身；Q19因同样使用APFloat只作integration cross-check。APFloat production BF16/TF32与MPFR高精度结果及
test-only独立整数/raw-bit rounder比较；MPFR production transcendental以已知点、
metamorphic relation、version/digest/self-test和wrapper验证闭合trusted-TCB gate。这里的self-test至少绑定exact MPFR/GMP
source/build digest和configure options，在干净依赖build上分别执行上游`make check`，保存command、exit status、version、
config summary及test-suite logs；随后执行项目wrapper known-point/metamorphic/raw-codec tests。上游check只证明受管依赖
build conformance，不算第二算术oracle。另一实现或后续board作为升级证据；
MPFR本身标记trusted semantic TCB而不是“双软件oracle”，结果只声明trusted MPFR semantics；oneDNN bulk与
formal APFloat loop逐row比较，qualification corpus按format另以SoftFloat或MPFR检查关键区分向量；Q19只作source integration
cross-check，不能因再次调用APFloat算第三路。target codec由test-only逐bit mapper产生expected，不能让production codec自证。
Q19和target model只共享稳定dtype enum、physical geometry、packet字段和ABI常量，不能共享compute/rounding/codec helper或
DTE scheduler。

### 4.6 oneDNN bulk准入、layout和执行生命周期

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

bulk qualification与runtime执行是两个独立stage。Q22实现受管host tool `wafer-cmodel-qualify-bulk`，并强制
`calibrate -> freeze -> validate`三阶段no-replace workflow：calibration invocation只能读取calibration manifest并产生
immutable calibration record；freeze invocation在任何held-out执行前产生content-addressed
`BulkBackendQualificationPolicy`，固定`B_backend` comparator/envelope、proof类别、semantic/domain/environment digest、
calibration manifest/record digest、预注册held-out manifest digest及两集合不重叠证明；validate invocation只读该policy，
按冻结阈值运行held-out，不能回写或放宽envelope。final record保存三个phase artifact/log digest和顺序校验；同一次看完
held-out再取最大误差的流程不得签发资格。

该tool消费冻结的`NumericSemanticsProfile` registry、shape/layout/value-domain或解析证明、分区corpus和完整backend
environment，在独立checked offline CPU/time/memory budget下运行formal和oneDNN；该budget可高于runtime MAC budget，但
仍有timeout/resource limit，debug override结果不得签发资格。成功只产生immutable `BulkBackendQualificationRecord`，包含
policy/calibration/held-out manifest与phase-log digest、formal/profile/tool/source digest、domain或payload digests、formal
expected digest、backend output digest、冻结及实际abs/rel/ULP与special-value结果、证明类别、resolved
descriptors/attrs/implementation、`HostPlatformFingerprint`，以及逐worker fenv/MXCSR readback在内的完整environment。它不进入compiler IR/package、
不定义target semantics。release/CI把readback验证过的records编入backend registry；runtime只有在profile/domain/environment
及qualification-policy digest exact match时构造`BulkBackendAdmission`，否则rejected。经验row只能匹配record枚举的payload digest；解析/穷举row按其
已验证domain matcher。mandatory large GEMM资格必须来自这个offline producer，不能通过runtime debug slow path补做。

“一次MatMul call”配合formal MAC为零只证明bulk dispatch，不证明SystemC event缩放或性能；event缩放由Q22.V验证。
optional reorder primitive另行计数。另建pinned-host bulk qualification，记录cold primitive/JIT、
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

formal context必须RAII恢复，支持stack-safe nesting或显式拒绝nested dispatch；SystemC `wait`、async callback和reentrant
numeric dispatch在scope中非法。强制测试让两个`SC_THREAD`跨delta cycle交替不同rounding/tininess/exponent profile，前一
process制造overflow/inexact而后一process执行exact op，并覆盖nested、early return、exception、normal/subnormal及caller
预置state恢复；这只证明logical-process隔离。另用两个真实OS thread分别验证SoftFloat adapter与MPFR TLS build和状态独立性。

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
- stochastic行为在取得seed/state合同前只可做独立统计correlation，不得用Q19的SplitMix64 policy冒充硬件；
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
- profile阈值在查看held-out结果前冻结；Q19/CPU已有tolerance不自动成为board numeric tolerance。

### 4.8 SystemC/TLM和simulation-time边界

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
target/package artifacts；external授权/事实源gate通过后，driver再用仍由invocation持有的owner-backed target LLVM
bundle，host执行target LLVM并调用获准host使用、与device build同源的repo CRT wrapper/command-invocation C源码。
host-only platform层由许可兼容provider负责rank context、Tsm factory/operator、packet复制、checked address/MMIO和
可yield wait；packet/event随后进入SystemC model。model mismatch或执行
失败可让driver返回非零并保留diagnostic，但已验证package保持可审计，不回滚、不改写，也不让Q22成为Q17/Q18
correctness前置。是否启用该gate属于用户显式请求或configured CI policy；不能变成wafer-opt stop-stage或手拼pass。

direct `wafer_tx81_*` shim可以保留为快速ABI smoke，但它绕过repo CRT、Tsm object和packet，不计入正式CModel完成。
Host-CRT/SystemC入口能证明compiler-produced target call经过其明确provenance的CRT/packet/event语义并产生正确完整输出；在与
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
model专用分支。Q22.E completion只接受Q22.V fresh source replay记录的Q0.L profile-bearing package identity，不能选择
Q0.L前历史package。provider输出invocation-local result/diagnostic，不能写回package或生成partial successful result。

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
| authorized repo-CRT/SystemC model-only functional-numeric | external授权/事实源gate、same fully legal target LLVM、获准host使用的同源repo CRT wrapper、authorized host packet seam、untimed SystemC event/completion、formal scalar语义层、逐profile admitted oneDNN大GEMM backend和完整输出differential | 对应授权和supported model profile内的host CRT/packet functional-numeric correctness | vendor-exact packet、hardware numeric、package ELF、board、timing |
| optional packet/MMIO conformance | exact-ELF register trace、board capture或versioned vendor builder与authorized host packet的逐字段decode/range/engine/register-effect conformance | 对应capture范围的packet/register provenance | 仅凭trace声明numeric、完整loader/provider lifecycle、timing或board等价 |
| Q22.C board numeric correlation | Q22 model-only通过、Q6.B有效board execution、区分向量、重复稳定性、冻结的comparator、独立held-out和source-backed完整输出 | 绑定环境和tested domain的board-output-correlated profile；对应独立packet/MMIO evidence闭合后才升级为hardware-correlated-numeric | vendor-exact packet、不可观测内部实现、未测domain、exact package、timing |
| exact-module functional | verified package、all-and-only RISC-V ELF、ISS/loader/MMIO/provider lifecycle | package target-model execution | real board、hardware timing |
| deferred Q22.P timing | 另行恢复后的PMU measurement basis和独立timing held-out | 实际通过的LT/AT profile | 改变numeric、IR legality或candidate acceptance；无RTL证据时称cycle-accurate |

profile是execution result provenance，不是compiler legality或package semantic branch。高层profile失败不能反向改变
accepted instruction支持范围；如果硬件可表达但model未覆盖，应扩target model capability或保持该profile拒绝。

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
- `FormalNumericExecutionContext`覆盖SoftFloat/MPFR save/restore、target-owned FTZ/DAZ、sticky flag、双OS-thread TLS及
  nested/early-return/exception；本row不依赖SystemC process或oneDNN output。

#### Q22.B bulk qualification

- semantic profile、shape/value domain和完整oneDNN environment共同定址`bit-exact/profile-bounded/rejected` row；exact由
  中间值/overflow/reduction证明，bounded由解析证明、离散穷举或冻结的有限tested domain支持，finite corpus不得外推；
- adapter重放`raw snapshot -> target decode/Cx/NCx unpack -> dense -> optional reorder -> oneDNN temp -> target codec/pack
  -> atomic commit`，覆盖alias、padding/tail/canary、scratch/pack budget、resolved descriptor、cache identity和异常路径；
  `BulkExecutionEnvironment`另验证caller/实际worker state边界；
- formal scalar执行使用checked MAC budget；超过budget且无admission时在compute前失败。generated large-GEMM component corpus
  只证明dispatch和adapter；正式row必须由独立`calibrate -> freeze -> validate`记录签发并在runtime exact-match；
- pinned-host qualification记录`impl_info_str()`、effective ISA、threads、cold create/JIT、pack/reorder和warm execute；mandatory
  profile落入reference implementation时不授予`performance-qualified`，correctness不依赖易抖动的绝对wall time。

#### Q22.L target LLVM module bundle

- Q0.L prepared target LLVM/ABI artifact形成owner-backed、move-only、不可序列化的all-rank bundle；逐rank readback logical rank、
  entry、profile/target identity/Kernel Runtime ABI、ordered typed slots和module identity，任一late failure均不形成bundle；
- direct shim只消费该bundle检查symbol、signature、control flow、typed slot和基本address formation，结果仅标ABI smoke；
  bundle不调用Host CRT、构造packet、链接SystemC或替代Q17/Q18 artifact。

#### Q22.H authorized Host CRT

- external authorization/spec gate满足后，Q22.L同一target LLVM经过获准host使用的同源repo CRT wrapper和许可兼容Tsm
  factory/operator；all-and-only symbol/platform surface从lowering/header/typed registry生成，不复制109项字符串表；
- unknown symbol、factory、wrong ABI slot和static profile在mutation前拒绝，dynamic packet/address/descriptor在effect前拒绝；
  direct shim、RISC-V archive和SystemC component均不能替代host provider/symbol closure。

#### Q22.S SystemC functional-event model

- Q22.H实际transaction进入rank/tile memory、worker event、local completion和Direct DTE/FSM；numeric effect只调用Q22.N，
  不能读取Q19 numeric kernel或DTE scheduler；
- 唯一`sc_main`至少运行两个`SC_THREAD`跨delta覆盖issue/visibility/completion、failure wakeup和numeric context恢复；
  unavailable/skipped或plain C++ kernel test不算完成。本row不以Q22.B或完整source workload为前置。

#### Q22.V source-backed functional-numeric verticals

- 同一`wafer-compile`重放Q20 rank-count=1 f32 linear/residual MLP、source-produced f16/bf16 GEMM和Q21 16-rank tiny Llama，
  覆盖它们实际调用的CRT/packet/numeric/event/Direct DTE surface并比较all-and-only完整输出；
- source-backed deterministic large GEMM必须超过formal budget、拥有固定source/config/seed或payload及独立expected/digest，
  并自动命中Q22.B冻结admission；现有未初始化4096 exporter升级前只算结构测试，不能证明dispatch或numeric；
- SystemC-enabled vertical必须真实执行；任一rank late failure无partial model result。Q17/Q18仍先按各自合同发布，model mismatch
  只让verification返回非零并保留已验证package供审计。

unsupported-reason closure只是各row必要条件，不能替代相应positive matrix。

### 8.2 Q22 authorized host packet和memory gate

- CT/NE/RDMA/WDMA/TDMA的typed args、packet fields、worker、trigger、range/end、raw `TsmExecute` status、model error
  latch和target ABI可观察status逐family区分覆盖；
- SPM/DDR bounds、reserved region、Cx/NCx、bitpacked i1、subview/strided descriptor用独立slow oracle或board bytes验证；
- Host-CRT/SystemC positive来自compiler-generated target LLVM，并把authorized host seam实际构造的packet作为正式
  functional-event输入；Q22内用独立field/memory oracle覆盖raw bytes、decode和observable effect，但独立实现的packet
  builder不得标记vendor-exact；
- 若取得exact-ELF register trace、board capture或versioned vendor builder，可另做逐字段packet/MMIO correlation并提升
  packet provenance；它是可选诊断证据，不是Q22或Q22.C前置，也不能替代numeric vectors。

### 8.3 Q22.C Board numeric correlation gate

Q22.C在Q22、Q6.B和configured numeric corpus均可用后执行，不要求PMU、packet capture或exact-module provider。Q6.B先
证明allocation/H2D/load/launch/wait/status/D2H/cleanup、watchdog/reset和重复invocation有效；任何provider、completion、
copyback或guard失败的sample均为invalid，不能用于调numeric。

板端corpus按capability row生成，而不是按一个op名字笼统通过。首批顺序是：

1. 13种storage format的movement/fill/boolean/guard、layout tail和未写区域byte/bit exact；
2. 七种compute/convert format的generated single-op vectors，覆盖36条convert route、四种确定性rounding及已实现
   f16/bf16/f32/TF32/integer arithmetic和GEMM/reduce candidate；
3. rank-count=1 f32 GEMM、add、tanh及完整MLP；
4. Tiny Llama实际需要的reduce、exp、rsqrt、mask/i1和其它已accepted tuple；
5. 16-rank Direct DTE与完整output；
6. zero-point和stochastic候选从第一版即存在，但只有公式、seed/state/advance事实闭合后才选择hardware row。

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
- source-produced f16/bf16 simple GEMM及deterministic large GEMM同样从正式frontend进入；large case必须有固定payload、
  独立expected/digest并超过formal budget。generated shape corpus只补dispatch，不替代source-backed numeric expected；
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

### 10.0 Implementation readiness spike

```text
Pipeline position:
- Upstream artifact / IR:
  Q21固定source/config/payload及同一wafer-compile形成的accepted rank artifacts；当前tasks/14 target preparation、repo CRT/
  vendor dependency源码和本机toolchain/package environment。Q19 expected只作结果核对，不替代artifact census。
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
package含16个rank module、entry、completion和ELF。随后只从该正式producer的grouped artifact为每个logical rank派生
accepted debug replay；不是手写fixture，也没有把debug dump当作published package成员。16个rank的`@main`均有相同四项：

| group ordinal | kind / init | source dims | input -> result | tile layout | extent |
| --- | --- | --- | --- | --- | --- |
| `#0` | sum；group-boundary scalar最终为f32 `+0` | `[2]` | `1x4x16xf32 -> 1x4xf32` | `NCx -> Cx` | 16 |
| `#20` | IEEE maximum；local f32 `-Inf` | `[3]` | `1x1x4x4xf32 -> 1x1x4xf32` | `NCx -> NCx` | 4 |
| `#21` | sum；group-boundary scalar最终为f32 `+0` | `[3]` | `1x1x4x4xf32 -> 1x1x4xf32` | `NCx -> NCx` | 4 |
| `#28` | sum；group-boundary scalar最终为f32 `+0` | `[2]` | `1x4x16xf32 -> 1x4xf32` | `NCx -> Cx` | 16 |

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
dynamic init、所有非tail/multi-dim reduction、avg及完整target special-value政策。tasks/06 candidate materialization
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
| SystemC/TLM | 无header/library/pkg-config/CMake package；feature-on在当前环境必须configuration fail | 官方3.0.2 commit `70b0fc8e...`构建并安装`SystemCLanguage`/`SystemC::systemc`；C++17 `sc_main`的两个`SC_THREAD`经delta-cycle event同步并正常退出 | Q22.S以3.0.2作为qualified candidate，正式pin仍需进入统一版本文件并跑上游/项目测试；不混用Ubuntu 2.3.4 ABI |

SoftFloat/TestFloat 3e采用U.C. Berkeley三条款式许可，SystemC 3.0.2参考实现为Apache-2.0，oneDNN为Apache-2.0；
MPFR/GMP分别涉及LGPL及GMP双许可，具体静态/动态分发、source offer和notice由引入任务在项目发布政策下确认。官方当前
资料确认MPFR 4.2.2要求GMP 5.0以上，故GMP 6.3.0满足版本关系；license文本存在不等于本项目已经完成合规审查。
readiness candidate source identity为SoftFloat 3e zip SHA-256 `21130ce8...c746`、TestFloat 3e
`6d4bdf00...ad6`、GMP 6.3.0 `a3c2b802...8898`、MPFR 4.2.2 `b67ba038...ce01`，以及表内两个Git commit；
Q22.N现已在统一版本文件记录并由bootstrap校验完整digest，表中截断值仍只作可读审计摘要；oneDNN/SystemC的正式引入
分别留给Q22.B/Q22.S。

#### 10.0.3 Host CRT、vendor seam和replay阻塞

host CRT的可编译/不可链接边界及external授权gate见3.1/3.4。结论是wrapper层可复用候选已经被编译事实支持，但当前没有
host operator、Direct-DTE/SPM provider或可加载vendor CModel closure；Q22.H不能靠RISC-V archive、include-only target或
direct shim冒充完成。vendor交付和授权属于external，Q22.N、Q22.B和Q22.L仍可独立推进；plain SystemC在此期间只作
readiness probe，Q22.S实现仍等待Q22.H实际transaction。

readiness replay还发现`wafer-convert-group-to-tile-region`会创建`async.token`却没有声明Async dependent dialect，导致只跑
`wafer-lower-groups-to-tile-region`的Q21 artifact abort。本轮已补dependent dialect和all-to-all named-pipeline回归；修复后
同一Q21 grouped artifact可产生36个tile region及2个DTE endpoint。这个修复只恢复debug replay，不改变production artifact
语义或替代Q0.L。

#### 10.0.4 Readiness决议

- Q21 resource census通过并已由Q0.L按tasks/10/11 ordered reduce设计完成fresh source replay；readiness不再是前向blocker。
- Q22.N已经建立默认关闭的受管source/bootstrap、唯一CMake target、23项build/self-test/identity gate与license artifact
  closure；Q22.B仍需独立受管oneDNN、qualification record和发布政策，不能把readiness probe当bulk admission。
- Q22.L已经独立形成owner-backed target LLVM bundle；Q22.H仍受external vendor授权/host-seam事实源阻塞；
  Q22.S/Q22.V继续依赖Q22.H。
- vendor CModel套件、真实board和hardware numeric/packet/timing仍是external evidence；它们不否定model-only方案，也不能由
  文档、有限corpus或SystemC选择推断。

### 10.1 Capability和依赖收敛

- Q0.L已完成：production `CompilationRequest`贯穿typed `TargetProfileId`；debug named target pipeline只用同一registry
  显式解析required option。tasks/14单一拥有engine×format ABI/register legality，source reduce按init-first canonical-order
  composite展开，elementwise map不再静默丢义；Q22.N/L只消费这些已验证事实，CModel不得补救compiler已丢失或未证明
  合法的command语义；
- 并行向vendor索取完整host CModel development package：匹配`host_runtime.h`/`runtime_api.h`/`tx_runtime.h`/TsmML headers、
  `libcmodel_runtime_api.so`、`libhpgr.so`、`libtsmml.so`、model resources和transitive dependency/license/version；同时确认
  是否存在低层x86 instruction/operator library，并让项目owner/法务确认采购条款是否允许host集成、修改和派生实现；
  未确认时Q22.H保持blocked，但不阻塞不消费vendor派生事实的Q22.N/Q22.L/Q22.B；
- 固定首批target call、dtype/layout、engine、packet、Direct DTE和completion capability matrix；
- Q22.L已把tasks/14 private prepared target LLVM提升为owner-backed all-rank内部artifact，并让现有device link直接消费；
  该本地artifact不以vendor授权为前置，也不提前执行host CRT/packet；
- 增加默认关闭的稳定target-model build feature；以readiness通过的Accellera SystemC 3.0.2作为candidate，在统一版本文件
  固定完整commit/digest、获取方式、Apache-2.0 notice、`SystemCLanguage` package和唯一`SystemC::systemc` target。基础
  compiler与plain C++ kernels仍可独立构建；feature启用时缺SystemC必须configuration fail，未启用时正式profile明确
  unavailable且Q22 gate未完成，不能由direct shim代替；
- Q22.N已经把SoftFloat/TestFloat 3e、GNU m4 1.4.21、GMP 6.3.0和MPFR 4.2.2的完整source digest、license、
  thread/rounding环境、唯一CMake target和上游self-test纳入受管依赖；oneDNN 3.12仍只是Q22.B的readiness-qualified candidate。
  formal numeric tests缺任一该family必需的independent oracle或trusted-TCB conformance dependency时明确unavailable，
  不能以host `float`替代。Q22.N已经固定SoftFloat specialization/`THREAD_LOCAL`、MPFR TLS/runtime版本、self-tested
  MPFR/GMP artifact到实际loaded shared binary的digest/version/transitive exact-match；仍待Q22.B固定的是oneDNN
  dispatcher/thread runtime、完整`HostPlatformFingerprint`和binary发布方式，MPFR/GMP的LGPL交付义务仍由发布配置承担。
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
  不扩大compiler target legality，不接oneDNN、SystemC、Host CRT或Target LLVM，不复用Q19 production kernel，
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
  production不得调用Q19 helper。独立SoftFloat adapter交叉FP16/FP32，TestFloat的slowfloat路径验证SoftFloat自身；
  MPFR-backed formal path完成sqrt/tanh/exp/rsqrt等高精度结果并为BF16/TF32提供高精度differential；
- 一次闭合36条convert route、101条四种确定性rounding/plain执行row、88条floating elementwise、4条BOOL logic和
  F16/BF16/F32三条GEMM formal row；zero-point和stochastic保留命名candidate policy及区分向量。source reduce只消费
  Q0.L已materialize的普通composite，16条native reduce selector因init/identity/order未闭合全部静态拒绝；
- `FormalNumericExecutionContext`只聚合invocation-owned model status，effect-free scalar/tensor evaluator在完整成功后原子commit；
  APFloat每次调用显式传入rounding，首个profile固定gradual、no-DAZ、no-FTZ。MPFR wrapper与SoftFloat oracle adapter各自
  保存/恢复完整环境；component tests覆盖nested/exception restore和双OS-thread TLS，不在本阶段依赖SystemC process；
- 执行第8.1节Q22.N子项的exhaustive、boundary、property、metamorphic、independent-oracle和capability closure tests；生成的matrix
  必须区分model-implemented、compiler-emittable和hardware evidence，不能以f32 workload代替。

完成：13种logical storage codec和有证据的physical engine/layout row闭合，当前七种compute/convert format及36条convert
route无missing/duplicate；每个published `(ModelProfileId, NumericCommandKey)`有唯一semantic profile/formal kernel/comparator
或静态unsupported reason；formal backend逐family independent/trusted-TCB gate和execution-context isolation通过。该阶段
不产生oneDNN admission，也不声明任一未知edge policy为hardware事实。

新鲜完成证据：feature-on受管依赖记录包含20个artifact、9份license文本和23项conformance gate；numeric suite 37/37、
CTest 8/8通过。feature-off基础suite发现138项，137 pass、1个预期StableHLO importer skip，CTest 6/6通过。
完整`check-wafer`执行208项lit并全部通过，`--show-unsupported`列出的41项全部来自未启用的StableHLO/Shardy importer依赖，
没有required numeric test被skip/unsupported。该证据只签发model-only numeric foundation，不签发bulk、SystemC或hardware profile。

#### 10.2.1 首个model-only policy closure

首个且无默认值的opaque `ModelProfileId`固定为`wafer-model-formal-deterministic-v1`。它只选择以下确定性model语义，
不进入compiler `ExecutionConfig`、target legality、package或hardware evidence。完整typed record及其digest才是定义，调用方
不得解析spelling推导字段：

- 所有multi-byte scalar codec使用little-endian。numeric TF32使用32-bit container的bits 31:13作为`s1e8f10`，encode清零
  low 13，numeric decode遇noncanonical low 13非零即拒绝；raw movement仍保留全部bytes。BOOL physical bit
  ordinal只能来自tasks/08 owner helper，byte内LSB0/MSB0必须由显式target encoding或model profile
  选择；首个LSB0候选只标model-only，不能复用Q19私有mapper。Cx/NCx bitpacked block/tail事实尚未
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
- source reduce只消费Q0.L已经materialize的init-first、有序fill/movement/map-free elementwise sequence；native reduce因
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
| native CT reduce | 16 | 0 | 4种op × 4种已编码format全部因init/identity/order未闭合拒绝 |

因此registry总计276个不重叠selector。formal tensor executor只消费已resolve且supported的convert、elementwise和GEMM；它在
output分配前完成profile/command、arity/count、canonical encoding及caller-owned scalar/FMA双预算preflight，整张tensor
成功后才一次commit aggregate model flags。MPFR的Sqrt/Rsqrt/Log2/Ln/Pow2/Exp/Sin/Cos/Tanh/Sigmoid/Softplus
published row只接受F16/BF16/F32同格式RNE；TF32和directed direct-op路径只作component evidence，不能扩大compiler surface。

### 10.3 Q22.B oneDNN bulk qualification

- 由受管oneDNN source形成唯一CMake target、GEMM adapter和backend registry；使用独立`BulkExecutionEnvironment`，不把
  worker flags映射为target status；
- 建立独立`BulkBackendAdmission`。每个semantic profile×shape/value-domain×environment候选通过
  `calibrate -> freeze -> validate`后进入`bit-exact/profile-bounded/rejected`，分别处理f32、f16/bf16宽累加、
  TF32-to-f32和i8-to-s32；runtime只接受qualification record与当前host/profile exact-match；
- formal work budget和fail-fast固定；generated large-shape component gate以dispatch/instrumentation证明mandatory大矩阵命中
  admitted row且没有逐MAC scalar fallback；另做pinned-host implementation/performance qualification；
- adapter输入输出只使用Q22.N codec形成的logical dense temporary；oneDNN结果和envelope不能反向修改
  `NumericSemanticsProfile`或板端comparator。

完成：首批GEMM row具有可readback的bulk admission、budget failure和environment mismatch negative；mandatory qualification
large shape命中admitted backend，reference implementation不冒充performance-qualified。source-backed自动dispatch和完整输出
仍由Q22.V闭合。

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
reference及no-card纵向重放通过；全量138/138 unit、249项lit中248 pass/1个预期feature-inverse unsupported、CTest 6/6。
Q22.H仍因external authorization/spec gate未满足而保持blocked。

### 10.5 Q22.H Authorized Host CRT

- external授权/事实源gate是开工前置；项目owner/法务先确认采购条款允许host集成，或取得vendor书面许可/允许独立实现的
  公开规范。未满足时Q22.H保持blocked，不修改vendor材料或实现其派生operator/packet seam；
- 只消费Q22.L `TargetLLVMModuleBundle`；host JIT执行same fully legal target LLVM，不形成host专用lowering或改写bundle；
- device/host build共享经确认可使用的repo CRT wrapper源码并使用互斥platform contract，许可兼容Tsm factory/operator
  形成packet；
- host provider闭合readiness发现的36个Tsm和11个Direct-DTE/SPM platform入口；Direct DTE state绑定invocation/rank，
  不从process-global、OS thread或调用顺序恢复；
- host retarget、ORC missing symbol、factory/packet field和provider failure均在对外result前失败；packet sink component test
  只证明authorized host seam，不冒充SystemC event、完整numeric vertical或vendor-exact packet。

完成：同一Q22.L all-rank target LLVM经host CRT wrapper实际产生许可兼容packet transaction，symbol/provider closure完整且
failure无partial result。direct target-call shim仍只作ABI smoke。

### 10.6 Q22.S SystemC functional-event model

- 以受管SystemC 3.0.2和唯一`SystemC::systemc` target建立默认关闭的feature；feature启用时缺依赖必须configuration fail，
  compiler和plain C++ numeric kernels保持可独立构建；
- 消费Q22.H实际packet，建立rank-local virtual SPM/DDR、typed slots、checked address、invocation error latch、三个worker window、
  resource event、local drain、可yield wait及Direct DTE/FSM。model-only per-worker parallel config无证据时采用保守serial profile，
  不声明`3×5`物理queue或engine复制；
- 覆盖current CT/NE/RDMA/WDMA/TDMA authorized host packet decode和独立field/memory checks；DDR/SPM aperture、MMIO和未来
  ISS/interconnect只通过受限TLM边界，vendor-exact逐字段correlation仍是可选provenance gate；
- numeric effect只调用Q22.N formal profile；Q22.B bulk在本stage不是完成前置。plain C++ kernel component tests不链接SystemC；
  SystemC integration test使用唯一`sc_main`，至少两个`SC_THREAD`跨delta交替numeric profile，证明thread-local state恢复；
- delta-cycle component gate证明issue后结果尚不可见、local fence按watermark等待、completion一次commit、failure唤醒且不
  copyback；unknown packet、address exact-end/overflow/cross-resource/reserved-SPM、event error和DTE no-progress均结构化失败。

完成：实际SystemC executable闭合packet decode、memory effect、visibility、completion和Direct DTE/FSM component matrix；
feature unavailable/skipped或plain C++ unit不能冒充通过。该阶段不要求完整source workload，也不发布numeric/board/timing claim。

### 10.7 Q22.V Source-backed functional-numeric verticals

- 固定真实`linear-residual-mlp-f32` rank-count=1作为首个vertical：当前case参数batch 2、input 16、hidden 32、output 16、
  f32、tanh和residual都只是测试参数，不进入capability协议；
- 重放产生Q22.L artifact的正式producer chain，经`wafer-compile -> ExecutableBundle -> TargetLLVMModuleBundle -> Q17/Q18 publication ->
  authorized Host CRT -> SystemC model`真实链执行；手写LLVM/packet只补negative；
- Q20 GEMM必须实际命中Q22.B冻结的admitted oneDNN row；另以强制formal backend的小shape重放同一semantic profile，按该row
  exact/bounded policy比较，证明backend选择不改变target semantics。完整output同时与Q19及NumPy expected按case已有
  `atol=1e-6, rtol=1e-5`比较并检查全部status/packet family count，不只抽查元素；
- 增加source-produced f16、bf16 simple GEMM；把large GEMM exporter固定为source/config/payload、独立expected和digest，
  mandatory shape超过formal budget并自动命中Q22.B admission，且SystemC event/transaction数量不随M×N×K按per-MAC增长；
  4096 shape只保留为stress参数；
- 执行Q21 16-rank tiny Llama，覆盖batched GEMM、reduce、exp/rsqrt、i1/select、Direct DTE和all-rank output；
- 最后一个WDMA/packet/kernel及任一rank late failure不形成model result，已发布Q17/Q18 artifact仍保留；SystemC-enabled CI
  必须实际执行，unavailable/skipped或unsupported-reason closure不算通过。

完成：Q20、f16/bf16、deterministic large GEMM和Q21完整输出，bulk自动dispatch、packet/memory/event negative、DTE
no-progress及all-rank atomic result全部通过；Q22只发布`repo-CRT/SystemC model-only functional-numeric`，仍不称hardware
numeric、vendor-exact packet、exact package/ELF或timing model。

### 10.8 Q22.C Board numeric correlation

- Q6.B先闭合board provider correctness、watchdog/reset和重复invocation；
- 按第8.3节的capability row、区分向量和comparison policy采集calibration corpus；
- 冻结policy后运行独立random/shape/layout/optional-field held-out及Q20/Q21完整output；
- 发布绑定environment identity、tested domain和raw evidence digest的profile；packet capture不可用时降级packet
  provenance，不阻塞board-output correlation。

完成：只有实际通过的row从`model-only`逐级升级，失败/未测tuple保持unsupported。Q22.C不声明exact ELF、timing或未测
输入域等价。

### 10.9 Q22.E Exact package execution

- 在取得vendor simulator或完成RV64 ISS、loader ABI、MMIO/custom instruction和provider lifecycle后接入
  Q22.V fresh source replay记录的Q0.L profile-bearing Q18 verified package identity；
- 原样执行all-and-only Q17 modules，不发布host专用instruction list或修改manifest；
- 通过wafer-run typed provider入口执行Q20/Q21 package和阶段性failure injection。

完成：同一package在model provider中完整allocate到cleanup并产生可信status/output；否则该能力保持更高待解锁gate。

### 10.10 Q22.P Deferred timing calibration

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
2. **Packet事实源与授权**：需要确认可否复用vendor可审计builder或获得register trace，并确认当前采购条款是否允许host
   集成或独立实现。只有vendor交付的许可兼容provider，或经项目owner/法务确认、基于独立可审计规范的clean-room seam才能
   产生Q22正式host packet；它验证当前compiler/CRT/model链，但在独立correlation前不证明vendor packet完全一致。授权缺口
   阻塞Q22.H/Q22.S/Q22.V，不阻塞numeric/bulk component实现；packet capture缺失只限制packet/opcode provenance。
3. **SystemC工程基线**：需要固定CI平台、受支持版本、获取方式、license、deterministic scheduling要求和未来ISS
   co-simulation边界。SystemC只拥有event/transaction实现，不改变artifact、packet、功能核或compiler合同。
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
- tasks/15拥有exact target model `RuntimeProvider`生命周期和package消费；host-CRT/SystemC模式不是该provider。
- tasks/16拥有direct ABI smoke、authorized host-CRT/SystemC、Q22 host packet/event、Q22.C board numeric、可选packet/MMIO、
  exact-module、board和deferred timing的证据分层及CI gate。
- 本文拥有target execution model内部边界、capability、SystemC选择、板端numeric correlation计划和各层不得冒充的声明。
- tasks/09/11/13仍分别拥有memory legality、instruction/packet legality和Direct DTE/completion；model不能改写这些
  compiler合同。

进入代码实现前应建立独立实施计划；本方案设计本身不表示target model、exact provider、board或calibration已完成。
