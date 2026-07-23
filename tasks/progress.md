# Wafer Compiler Task Queue

更新时间：2026-07-23

本文件只记录当前调度状态、前置关系和紧凑完成索引，不保存逐轮测试数字、实现复盘或历史工作日志。
长期架构与pipeline contract以编号设计文档为准，详细完成证据与实施记录位于`tasks/archive/`，完整导航见
`tasks/README.md`；历史变化由Git保留。

当前发布基线：Q22 repo-owned target-call/SystemC model-only untimed functional-numeric链、Q28标准Llama-2 7B单block
TP16 scale vertical、Q30 production vertical host性能收口、Q31多seed数值表征及Q32 MLIR-native bounded
physical-dataflow synthesis已经完成；数值纵向直接比较固定source CPU
expected，不再维护
accepted-IR第二套解释器。SystemC受管依赖统一位于`third_party/systemc-model`。Q6.B板端runtime execution已完成；
更广的板端数值相关、exact package执行、packet provenance和timing仍是独立later/external gate。

共享SPM/DDR static memory packing已由默认MiniMalloc fixed-capacity canonical search闭合；Q32在该owner之上完成
implementation、relation/tiling、encoding/view/route、residency、buffering/order、direct/ring/tree communication、
current integer-domain variants和typed target capability的bounded actual-clone联合选择，并退役旧decision旁路。
当前没有平行provider/query/schema或独立语义求解器；删除机制没有删除功能轴。

## 队列规则

- `Q*`是稳定tracking ID，不表示pipeline层级；执行顺序由状态和前置关系决定。
- 至多一个row标`doing`；`next`表示前置已满足，`later`不会自动进入主线，`blocked`必须写明外部或任务前置。
- `done`只表示对应编号文档中的completion gate已有验证；详细证据不复制到本文件。
- 新任务必须指向编号设计owner；非小修还需在`tasks/plans/`建立实施计划，并确认对应设计文档中的pipeline contract。
- 设计、实现和本队列冲突时，先按`AGENTS.md`优先级收敛事实，再更新状态。

## 当前执行图

```text
已完成compiler/source-oracle/model主线：
Q14
Q0 -> Q15 -> Q16 -> Q17 -> Q18 -> Q16.T
Q5.C + Q16.T -> Q20 -> Q21 -> Q22.R
Q22.R -> Q0.L
  -> Q22.N -> Q22.B
  -> Q22.L -> Q22.H
Q22.N + Q22.H + Q16.T -> Q22.S
Q22.B + Q22.S + Q20 + Q21 -> Q22.V -> Q22

源码模块化：
Q23 -> Q24 -> Q25 -> Q26

static memory packing：
Q26 -> Q34 -> Q32.B

验证consumer收敛：
Q22 -> Q27

tile-dataflow scheduling与7B级单block纵向：
Q22 + Q27 -> Q29 -> Q28 -> Q30 -> Q31

MLIR-native physical-dataflow synthesis：
Q29 + Q28 + Q30 + Q31 -> Q32.I -> Q32.R -> Q32.B -> Q32.V -> Q32.M -> Q32.S -> Q32.G -> Q32

configured board runtime（已完成）：
Q0.L + Q21 + configured board -> Q6.B

已完成communication quality closure与待续large-shape board vertical：
Q32 -> Q36
Q32 + Q6.B + Q36 + configured board -> Q35

已完成float candidate支持：
Q32 -> Q32.N

active compiler-sensitive hardware calibration and multi-engine overlap：
Q32 + Q6.B + configured board -> Q37

later/external：
Q32 + Q6.B -> Q9
Q22 + Q32 + Q6.B + configured numeric corpus -> Q22.C
Q18 + Q22 + Q32 + configured simulator/ISS -> Q22.E
Q32 + Q22.C + validated PMU/timing environment -> Q22.P
Q22 + owner-approved packet evidence -> Q22.K
Q32 -> Q32.T (optional compiler control plane)
explicit Count semantic/target/model evidence -> Q3.6 (independent typed writeback/ABI/numeric closure)
```

## 当前实施队列

Q37 compiler-sensitive hardware calibration and multi-engine software pipelining正在执行。第一checkpoint先形成
`docs/tx81-compiler-hardware-calibration.md`独立证据台账，以current硬件资料、vendor header/library与安全板端
microcase闭合会改变compiler legality、planning、lowering、cost或runtime completion的TX81事实，包括instruction
packet/数值/layout、SPM/DDR与cache、NCC各engine/worker/queue/address dependency、同步/可见性、Direct DTE/
multi-tile arrival、launch ABI与PMU measurement basis。每个维度必须得到已验证结论，或得到带保守compiler
处理的明确Unknown/unsupported边界；这些事实未闭合前不修改production scheduling。随后才在complete instruction
IR上物化显式multi-buffer、prologue/steady/epilogue、resource-aware issue order和latest-legal wait/fence，并让
每个actual clone重新经过SPM/DDR、instruction、target、package与board correctness gate。计划见
`tasks/plans/multi-engine-software-pipelining.md`。
当前queue active occupancy与full行为仍在Q37内保持`unknown`：静态depth不直接作为occupancy证据。普通
calibration只运行1/2/4（TDMA 1/2）。修正builder生命周期和逐issue观察位置后，CT exact documented
`D=6`已由单engine、单case、单样本及前后known-good Add heartbeat闭合submission/completion/count/output/guard；
逐issue control均为`0x100`，因此不声明六条并发occupancy。本轮显式manual授权的CT typed tight `D+1=7`也在
单engine/case/sample、紧邻issue、前后Add heartbeat和完整oracle下通过，证明documented depth是pending
queue storage而非完整lifetime总提交上限；window后control为`0x100`且blocking为0，未观察到resident/full/
backpressure。current profile上的NE/RDMA/WDMA exact `D=6`与TDMA exact `D=4`也已分别由单engine、
单case、单样本、前后known-good Add heartbeat及完整oracle闭合：`instruction_delta=6/6/6/4`，对应
engine/full execution delta为`492/2101/1578/292` cycles，blocking delta均为0，全部boundary/final result与
guard mismatch均为0。该组结果不证明active occupancy、full或backpressure，任意更深提交继续禁用。
current profile的10个disjoint cross-engine pair也已按单进程串行执行完成：serial/window各3个样本，全部
instruction count、result和guard正确，blocking delta均为0，最终known-good Add heartbeat通过。已有
CT+RDMA r4正overlap现降级为`historical/inconclusive`：本轮canonical CT→RDMA r4 window excess为
`[78,0,0]`、median为0；新增同RAW顺序RDMA→CT r4对照中，serial/window每个样本都满足
`ct_exec + rdma_exec == full_exec`，median excess同样为0。两个方向均为serial/window各3样本；新增对照的
result、guard、instruction count全部正确且blocking为0。CT+WDMA、RDMA+WDMA、CT+NE、NE+RDMA、NE+WDMA在r2/r4的
serial/window pairwise-excess median均为0，四个含TDMA pair在r2也均为0，r4因TDMA静态depth为4未运行。
两次RAW exact/partial/adjacent请求都在hazard发射前被disjoint资格门禁拦截，未执行hazard。整批前后Add
heartbeat均通过、卡健康且未调用reset/power。当前profile和current workload下没有pair满足稳定正overlap门禁；
这不证明硬件永远不能并行，但compiler对所有pair默认保守串行。RAW hazard暂不适用，只有未来对照稳定达到
median正overlap后才重新执行。
CT worker1/2 routing与三worker matching join也已由低深度4KiB FP16 Add闭合：worker1/2单case的
`inter_type`分别为`0x100/0x200`，matching `bywork` mask为`0b010/0b100`，对应worker CT instruction
delta均为1；worker0/1/2 disjoint join使用mask `0b111`，三个worker CT delta各为1。三个case的
boundary/final result与guard均正确、blocking为0，前后Add heartbeat通过。当前将CT三worker routing、
matching `bywork`及disjoint join记为`board-observed`；跨worker并行、仲裁、同地址行为与
default/local-fence跨worker scope仍保持保守`unknown/excluded`。
instruction-family typed catalog的35个safe case也已全部逐个串行上板通过，覆盖f16/bf16 elementwise、
convert、reduce、select composite、f16 NE GEMM、f16 TDMA Pad与f16 peripheral ArgMax/ArgMin composite
writeback、f16 PoolMax、peripheral LUT16 raw-offset lookup，以及f16 TDMA Img2Col；每个case均有精确bit
oracle、SPM guard、
terminal和cleanup。首次`reduce-sum-f16`准确暴露catalog边界错误：逻辑结果是128B，但CT physical write
span为256B。将四个reduction row修正为`result_bytes=128`、`output_span=256`后，reduction和剩余case全部
通过。ArgMax在含负数普通值且唯一最大值`100@index73`时通过；ArgMin在全正普通值且唯一最小值
`0.5@index42`时通过，value/index分别写入同一slot的`[0:2]`与`[4:8]`，中间2B保持不变。ArgMin负数
对照会错误返回首元素`-30@index0`而不是`-100@index42`，因此负数域保持unsupported，不由正数case外推。
最终Add heartbeat正常。f32、special value、held-out tail以及其余deferred geometry/writeback family继续
留在Q37后续校准。PoolMax使用`[1,2,4,64] -> [1,1,2,64]`、无padding、2x2 kernel/stride，
两个输出窗口的128个FP16结果逐bit正确，256B physical output span和suffix guard均通过。
LUT16使用128个非顺序`uint16`字节偏移查找128项FP16 table，完整256B结果逐bit正确；该证据只闭合
raw 16-bit byte-offset语义，不能把source解释成FP16数值index。
Img2Col使用`[1,3,3,64]`、2x2 kernel、1x1 stride和零padding，vendor destination
`[1,4,4,64]`按`ky,kx,oh,ow,c`展开；1024个FP16结果和2048B physical span/guard全部通过。compiler
verifier已同步为`kernel_strides=[Kx,Ky,Sx,Sy]`及destination
`[N,Kx*Ky,outH*outW,C]`，并用非对称非1x1正反例和完整target ABI lowering golden闭合。
NE BF16 1x16x16 identity GEMM也已通过：32B logical result逐bit正确，256B physical output span和
suffix guard均正确；该结果只闭合BF16 format、当前normal/normal layout与identity数值，不外推累加舍入、
transpose或tail。

Q32.N numeric algebraic extension已完成。任务收缩为直接删除pass中不必要的float类型门槛：
现有algebraic candidate、generic reduction切分、named GEMM K切分和Ring collective均接受支持的
f16/bf16/f32，不增加frontend mode、fast-math协议或Tile/Instr附加字段。结构、shape、layout、资源、
target encoding以及integer overflow/no-wrap检查保持；f16/bf16是compiler主线验证dtype，模型测试保持自身类型。

Q36已闭合current topology/execution mesh到compiler-private Ring/ordered-Tree参数、explicit p2p instruction、
collective completion和whole-card minimum-hop cost的事实链；详细证据见
`tasks/archive/topology-aware-collective-lowering.md`。Q35已完成并消费
Q6.B已闭合的cluster Direct DTE launch/runtime，形成显式SPMD contracting shard、large-shape M/N tiling、
fixed-capacity SPM、local GEMM与all-reduce的完整production package及板端exact证据。Q6.B已闭合四条typed board launch/runtime路径，包括独立Direct DTE
placement/readiness/completion、重复完整CPU exact和清理后资源基线；详细合同与证据由tasks/13-16及
`tasks/archive/runtime-board.md`拥有。
Q35 full-4096 f16 production静态确认`M=256,K=256,N=512`、每rank128个tile和256 KiB DTE payload；
先后修复CRT RHS raw transpose映射和RDMA/WDMA byte-stride到vendor element-stride的边界转换。无sharding的
`4096x256 x 256x4096`纯tiling隔离case完整32 MiB raw exact；最终16-rank full case连续两轮均由16个rank
正常完成，16份32 MiB output逐字节exact，cleanup后设备回到`9248M / 65536M`、0% utilization、无进程基线。
完成记录见`tasks/archive/k-sharded-gemm-board-vertical.md`。
Q32已完成integrated completion audit；该完成不会让其它
later/external gate自动进入主线。

完成边界：Q32.I/R/B/V/M/S/G采用MLIR interface、可重算analysis、actual-clone rewrite、DialectConversion、现有exact
gates和atomic commit；保留implementation、tile、encoding/view/route、storage/residency、buffering/order、communication、
resource-aware selection及mapped/physical-fill/oriented target纵向，不再把provider/query/key、shadow frontier、版本化诊断
schema或model/board qualification作为planner协议。下表保存Q32各checkpoint的紧凑完成索引。board numeric correlation、
simulator/ISS、packet provenance和timing所需外部事实仍只保留在Later / External Gates。

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 窄边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q37 | `tx81-compiler-hardware-calibration-and-multi-engine-software-pipelining` | `doing` | Q32、Q6.B + configured board | 先以独立证据台账和安全板端microcase闭合会改变compiler legality/planning/lowering/cost/runtime completion的instruction、数值/layout、memory/cache、engine/worker/queue、address dependency、同步/可见性、DTE/multi-tile、launch与PMU事实；每个未闭合维度保留带保守处理的显式Unknown/unsupported。完成该gate后，才从current instruction SSA/effect/loop事实物化通用multi-buffer软件流水、跨engine issue order及latest-legal wait/fence，经过完整memory/target/package/correctness gate并以板端对照确认实际重叠；aggregate PMU不伪装成cycle-accurate模型。 | 06、08-17；`docs/tx81-compiler-hardware-calibration.md`；`tasks/plans/multi-engine-software-pipelining.md` |
| Q32.N | `numeric-algebraic-extension` | `done` | Q32 | algebraic candidate、generic reduction、named GEMM K切分和Ring collective已删除仅因float或缺少额外fast-math标注而拒绝的分支；f16/bf16无标注正向覆盖actual mutation、frontier和Tile/Instr lowering，integer overflow/no-wrap及真实结构、资源和target负例保持。未新增frontend mode、私有数值policy或IR carrier。 | 05-07、10-11、13、16；`tasks/plans/numeric-algebraic-extension.md` |
| Q36 | `topology-aware-collective-lowering` | `done` | Q32 | current typed topology/mesh派生rank placement、exact bounded Ring与保持rank_group中序的ordered Tree；collective correctness/completion、singleton identity和final p2p minimum-hop whole-card cost闭合，不声明route/cycle/timing。 | 04、06、11、13、16、18；`tasks/archive/topology-aware-collective-lowering.md` |
| Q35 | `k-sharded-gemm-board-vertical` | `done` | Q32、Q6.B、Q36 + configured board | full-4096 f16 GEMM由显式row/contracting SPMD形成16份local K=256 GEMM和sum all-reduce；production闭合M/N tiling、SPM/DDR、Direct DTE、shared ELF、no-card与完整板端raw exact。修复CRT GEMM raw orientation及strided RDMA/WDMA element-unit边界后，纯tiling 32 MiB exact，16-rank full case连续两轮16份32 MiB output全部exact并回到设备基线。只形成该case/environment的workload-level evidence，不新增ABI、不完成Q22.C或timing。 | 02、03、05-07、09、10、13-17；`tasks/archive/k-sharded-gemm-board-vertical.md` |
| Q32.I | `mlir-native-implementation-relation-foundation` | `done` | Q29、Q28、Q30、Q31 | source OpInterface/external models已让generic division与target reciprocal两种真实implementation进入complete clone；MLIR Affine/Presburger/ValueBounds IndexRelation foundation与precision/failure/property gate闭合；重复DPS/Tiling语义的WaferTilingInterface已删除。证据见`tasks/archive/mlir-native-implementation-relation-foundation.md`。 | 01、06、08、10、13、16、18；`tasks/archive/physical-dataflow-synthesis.md` A/B |
| Q32.R | `physical-relation-realization` | `done` | Q32.I | rich IndexRelation查询、physical encoding attr interface、TransferRealizability、destination-style StorageLoad和relation-backed resident handoff已闭合；非7B source删除真实中间WDMA/RDMA，标准7B source选择26条handoff并通过fresh TP16 package/SystemC/PyTorch gate。证据见`tasks/archive/physical-relation-realization.md`。 | 06-11、16、18；同计划C |
| Q32.B | `physical-dataflow-test-seam-vertical` | `done` | Q32.R、Q34 | compiler-private production-shaped seam已让conservative spill唯一reserved baseline与spill/resident optimized actual clones共同进入rank frontier；rank只做SPM，all-rank disposable tuple重做DDR及全部late gate，1/16-rank与标准7B source-to-package/SystemC/PyTorch及determinism/atomic gate通过。证据见`tasks/archive/physical-dataflow-test-seam-vertical.md`。 | 01、06-18；同计划D |
| Q32.V | `typed-target-capability-vertical` | `done` | Q32.B | mapped DMA/WDMA双端root-relative offset与descriptor、physical-footprint fill的padding/tail/bitpacked domain及oriented GEMM source/Tile/Instr/v2 TargetCall/CRT/formal/SystemC纵向已闭合；v1 ABI保持不变，Q32.V完成当时因无真实逐row consumer而保持schema v3，后续Q6.B launch ABI consumer已独立升级为当前schema v5。证据见`tasks/archive/typed-target-capability-vertical.md`。 | 06、08、10、11、14-18；同计划E |
| Q32.M | `physical-mechanism-choice-closure` | `done` | Q32.V | shared candidate owner已从verified source独立产生recompute、static LICM和integer modular reassociation/tree/distribution/factorization actual clones；partial fanout保留DDR spill并增加maximal-compatible SPM SSA result，spill/resident与movement-first ready-order分别形成完整rank alternatives；communication从同一tile parent产生ring/ring、direct/ring和ring/tree完整clone并逐个重跑Instr/SPM/DDR/verifier/cost。layout/resource与collective consumers已迁到typed op、value-associated standard effects、custom resources和SSA token/fence，重复layout/resource/collective-info/verifyInstructionContract合同及public communication selector已删除。证据见`tasks/archive/physical-mechanism-choice-closure.md`。 | 05-13、16、18；同计划F |
| Q32.S | `bounded-joint-physical-dataflow-selection` | `done` | Q32.M | source/recipe/scope-policy与spill/resident/ready-order均以actual clone有界组合；reserved baseline独立于source 16、recipe 12、rank evaluation 64、rank frontier 256、whole tuple 64+64及whole Pareto 16等optimization caps。validated placement/high-water及final DDR/SPM/NoC/collective/compute/instruction/event/dataflow facts进入whole-card exact Pareto与target-owned static policy；最终winner不读scalar time或producer计数。implementation、fusion/share/recompute、LICM、各current integer variant、fixed-Cx direct mapped route、resident reuse、ready-order及direct/tree collective均有production-shaped whole winner。证据见`tasks/archive/bounded-joint-physical-dataflow-selection.md`。 | 06-13、16、18；同计划G |
| Q32.G | `physical-dataflow-production-cutover` | `done` | Q32.S | 默认`wafer-compile`已成为唯一production decision owner并原子提交whole-variant winner；旧public scheduling pass/pipeline、scope-prefix、layout/demand影子结构、communication selector/options、scalar-time winner和discovery recovery已删除。rank frontier以semantic generation与physical artifact kind双键约束all-rank correspondence，默认source/bulk SystemC数值纵向通过。证据见`tasks/archive/physical-dataflow-production-cutover.md`。 | 01、06-18；同计划H |
| Q32 | `physical-dataflow-synthesis` | `done` | Q32.G | current功能面包含implementation、relation/tiling、encoding/view/route、GEMM/batched-GEMM fixed-Cx-NCx absorption、storage/residency、share-vs-recompute、static loop-invariant hoist、全部Q32.M current numeric variants、buffering/resource-aware ready-order、communication、resource-aware selection及Q32.V typed capability；每个choice producer具备production IR mutation、下游exact consumer、whole-variant winner和atomic bundle commit证据，无选择分支的required closure mutation保留在committed winner。rank-count=1/16、Q20/Q21、Q28 fixed-seed/Q31 held-out 7B PyTorch/SystemC和全部SPM/DDR/event/transport/instruction/ABI/package/atomic gates已fresh通过。不含floating reassociation/tree、generic online reduction、non-GEMM FMA contraction、超出current integer-domain exact/modular子集的algebraic distribution/factorization、board性能或timing。证据见`tasks/archive/physical-dataflow-synthesis-completion-audit.md`。 | 01、06-18；`tasks/archive/physical-dataflow-synthesis.md` completion audit |

下列later/external gate不会因Q32完成自动进入主线。

## Later / External Gates

| Tracking ID | Semantic key | 状态 | 必须满足的前置 / 外部 gate | 窄边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q9 | `cost-calibration` | `later` | Q32、Q6.B + profile environment | 只校准Q32合法候选排序，不改变语义合法性。 | 06、16 |
| Q22.C | `target-model-numeric-correlation` | `later` | Q22、Q32、Q6.B + configured numeric corpus | 按capability row用board区分向量和held-out冻结numeric comparator/profile；Q35可提供large K-sharded GEMM workload-level证据，但不是本gate的硬前置且不能单独满足它。 | 16、17 |
| Q22.E | `target-model-package-execution` | `later` | Q18、Q22、Q32 + configured simulator/ISS | 原样执行Q32 integrated audit冻结的verified package及all-and-only RISC-V ELF；任何未来schema升级必须先独立完成再作为该gate输入。 | 15、16、17 |
| Q22.K | `target-model-packet-provenance` | `later` | Q22 + owner-approved vendor package或独立公开规范 | 可选关联repo CRT/packet/MMIO；缺失不阻塞数值CModel。 | 14、16、17 |
| Q22.P | `target-model-timing-calibration` | `later` | Q32、Q22.C + validated PMU/timing environment | deferred LT/AT校准；没有RTL/vendor cycle证据不声明cycle accuracy。 | 16、17 |
| Q32.T | `compiler-transform-control` | `later` | Q32 + explicit external control-plane consumer | 只有出现真实wafer-opt/autotuning consumer后才设计；必须复用同一rewrite/conversion，Transform IR不保存candidate frontier、不替代all-rank coordinator，也不进入wafer-compile或artifact。当前没有冻结param/report schema。 | 01、05-08、10、16、18 |
| Q3.6 | `crt-writeback-scalar` | `later` | explicit Count predicate + wrapper/target/model consumer evidence | static compact-contiguous source到proven-disjoint single-element i32 SPM destination的Count writeback。必须独立闭合typed instruction、effect/completion、ABI/CRT、model evidence和必要package readback；当前predicate/golden/formal-SystemC evidence/board row absent，source/model/board admission保持关闭。它不依赖Q32/Q32.V planner或capability协议。 | 11、14-17 |

新model/distributed/executable dialect、MPMD/rank class、跨卡coherent variant、WCRE/global registry、capability lease、
跨model state migration、共享weight cache、segmented MoE和70B/100GB stress当前不在active DAG；恢复时必须先更新
编号设计和completion gate。

## Done Index

本表只提供状态和证据入口，不复述测试数字或实现过程。

| Tracking ID | Semantic key | 状态 | 完成边界 | 证据 owner |
| --- | --- | --- | --- | --- |
| Q14 | `architecture-baseline` | `done` | 当前单卡纵向架构、事实优先级和历史计划边界已重基线。 | 01、14、15、16；`tasks/archive/12-architecture-evidence-reset.md` |
| Q0 | `target-correctness` | `done` | target conversion、结构保持、physical legality和原子失败窄边界闭合。 | 06、07、09、11、14、16 |
| Q5.C | `workload-corpus` | `done` | 固定PyTorch/XLA source/config/payload/expected corpus及独立CPU oracle。 | 02、16 |
| Q15 | `compiler-driver` | `done` | source到verified rank-local structured tensor program directory及原子发布闭合。 | 01-06、16 |
| Q16 | `executable-bundle` | `done` | all-and-only rank executable与move-only bundle闭合。 | 03、04、06、09、12、13、16 |
| Q17 | `target-artifact-bundle` | `done` | single-lowering target module、device link及原子artifact发布闭合。 | 14、16 |
| Q18 | `manifest-runtime` | `done` | typed manifest、package readback和no-card preflight闭合。 | 15、16 |
| Q6.B | `runtime-board` | `done` | rank-one、provider multi-launch、grid16 kernel、type6/type7 model及16-rank 256-byte Direct DTE均由fresh production package闭合static/fake/no-card与重复真实板端exact lifecycle；Direct DTE当前使用独占cache line的status-v2；只声明logical tile 0..15，不声明physical coordinate。 | 13-16；`tasks/archive/runtime-board.md` |
| Q16.T | `direct-dte-transport-activation` | `done` | Direct DTE binding、completion、target activation和package投影闭合。 | 13-16 |
| Q20 | `single-card-linear-mlp` | `done` | linear/residual MLP的1/16-rank source/CPU-expected/package纵向链闭合。 | 01、05、06、10-13、15、16 |
| Q21 | `single-card-tiny-llama` | `done` | 16-rank tiny Llama完整source/CPU-expected/package纵向链闭合。 | 01、05、06、10-13、15、16 |
| Q22.R | `target-model-readiness` | `done` | Q21资源与numeric/bulk/SystemC/host seam readiness完成分级。 | 01、10、11、14-17；`tasks/archive/target-model-readiness.md` |
| Q0.L | `target-command-legality-closure` | `done` | typed target profile、format legality、map/reduce lowering和fresh source replay闭合。 | 01、03、04、06、08、10、11、14-16；`tasks/archive/target-command-legality-closure.md` |
| Q22.N | `target-numeric-foundation` | `done` | multi-dtype codec、formal numeric policy/kernel和受管oracle依赖闭合。 | 16、17；`tasks/archive/target-numeric-foundation.md` |
| Q22.L | `target-llvm-module-bundle` | `done` | owner-backed all-rank target LLVM bundle及single-lowering device-link闭合。 | 14、16、17；`tasks/archive/target-llvm-module-bundle.md` |
| Q22.B | `target-bulk-qualification` | `done` | oneDNN exact qualification、runtime admission和no-fallback bulk lane闭合。 | 16、17；`tasks/archive/target-bulk-qualification.md` |
| Q22.H | `target-host-call-frontend` | `done` | same-target-LLVM host frontend、typed decoder和atomic sink闭合。 | 14、16、17；`tasks/archive/target-call-functional-frontend.md` |
| Q22.S | `target-systemc-event-model` | `done` | SystemC functional-event、private memory、Direct DTE和atomic result闭合。 | 16、17；`tasks/archive/systemc-functional-event-model.md` |
| Q22.V | `target-model-source-verticals` | `done` | source-backed formal/bulk/multi-rank完整输出组合闭合。 | 01、16、17；`tasks/archive/target-model-source-verticals.md` |
| Q22 | `target-execution-model` | `done` | model-only untimed functional-numeric capability profile发布完成。 | 01、16、17；`tasks/archive/target-model-completion-audit.md` |
| Q23 | `source-modularity` | `done` | instruction、tile-region到instruction和target numeric按稳定职责拆分，build/test/组织gate闭合且公共语义不变。 | 18；`tasks/archive/source-organization-refactor.md` |
| Q24 | `remaining-source-modularity` | `done` | group/candidate、target LLVM、numeric conformance、frontend program与compiler driver按稳定职责拆分，双配置gate闭合且公共合同不变。 | 18；`tasks/archive/remaining-source-modularity.md` |
| Q25 | `residual-source-modularity` | `done` | reference/model、numeric/bulk、compiler/artifact/package和frontend bridge共11个聚合实现按稳定职责拆分，双配置及真实外部helper gate闭合且公共合同不变。 | 18；`tasks/archive/residual-source-modularity.md` |
| Q26 | `memory-lifetime-analysis` | `done` | instruction loop backedge completion、共享path-sensitive lifetime/packing core、DDR issue-to-fence lifetime及两侧原子offset commit闭合，SPM/DDR各自memory-space、DTE、descriptor和resource合同保持。 | 09、11、12、18；`tasks/archive/memory-lifetime-analysis.md` |
| Q34 | `static-memory-packing` | `done` | SPM/DDR共享packing默认使用受管MiniMalloc fixed-capacity canonical search；精确edge-clique conflict适配、确定性宽松全局work budget、三态result、独立validator和仅限`ResourceExhausted`的first-fit fallback已闭合。 | 09、12、18；`tasks/archive/static-memory-packing.md` |
| Q27 | `reference-executor-retirement` | `done` | accepted-IR第二套解释器、oracle分支和旧CLI退役；CPU expected、typed invocation及target CModel/board differential边界保留。 | 01、16-18；`tasks/archive/reference-executor-retirement.md` |
| Q29 | `tile-dataflow-scheduling` | `done` | structured tensor program直达bounded rank-local task/dataflow candidate、完整traversal、跨region SPM、whole-rank/whole-variant resource gate、旧group executable surface退役及TP16 7B compile-only all-rank package闭合。 | 01、06-13、16；`tasks/archive/tile-dataflow-scheduling.md` |
| Q28 | `llama-7b-block-vertical` | `done` | 标准Llama-2 7B单block TP16从真实source、task-dataflow package到repo-owned SystemC managed-reference执行及完整PyTorch eager output differential闭合；不包含board、exact ELF、性能或timing。 | 02、03、06、09、11、12、16、17；`tasks/archive/llama-7b-block-vertical.md` |
| Q30 | `llama-block-production-performance` | `done` | 保持accepted IR、all-rank package、target command、SystemC行为和完整PyTorch differential不变，收口static movement构造及physical codec重复遍历；7B Release wall time稳定下降。 | 08、10、11、16-18；`tasks/archive/llama-block-production-performance.md` |
| Q31 | `llama-block-numeric-characterization` | `done` | 最终ProgramTensor边界的逐rank abs/ULP统计、非admission多seed 7B重放及预冻结source/model gate收紧闭合；不改变arithmetic、corpus admission或板端policy。 | 02、16-18；`tasks/archive/llama-block-numeric-characterization.md` |
| Q1 | `crt-surface-audit` | `done` | compiler-emitted production CRT symbol/prototype surface审计完成。 | 11、14、16及对应archive |
| Q2-Q3 | `crt-device-symbol-closure` | `done` | production CRT symbol和device-link closure闭合。 | 11、14、16及对应archive |
| Q3.5 | `crt-extended-evidence` | `done` | 扩展CRT surface evidence已分级。 | 11、14、16及对应archive |
| Q13.T | `supporting-doc-tool-decoupling` | `done` | conformance工具从代码事实源推导，不再解析设计文档marker。 | 01、16 |
| Q13.W | `tool-workflow-consistency` | `done` | SystemC canonical third-party root和existing CMake cache切换闭合。 | 16；`tasks/archive/third-party-dependency-root-consistency.md` |
| Q10-Q13 | `historical-design-governance` | `done` | 历史审计、恢复和设计治理工作已归档。 | `tasks/README.md`、`tasks/archive/` |

## 实施计划入口

- 当前active implementation task为Q37，计划见
  `tasks/plans/multi-engine-software-pipelining.md`。
- 最新完成任务Q35见`tasks/archive/k-sharded-gemm-board-vertical.md`。
- Q6.B完成计划见`tasks/archive/runtime-board.md`。
- Completed task：Q32 `physical-dataflow-synthesis`，完成审计见
  `tasks/archive/physical-dataflow-synthesis-completion-audit.md`，实施计划见
  `tasks/archive/physical-dataflow-synthesis.md`。
- 新实施计划：`tasks/plans/`。
- 已完成计划和历史证据：`tasks/README.md`的“实施计划导航”和“归档文档”。
