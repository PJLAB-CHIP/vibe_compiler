# TX81 Compiler-Hardware Boundary Calibration

本文是TX81硬件行为对Wafer compiler边界的profile-scoped校准台账。它回答三类问题：

1. 哪些硬件事实会改变IR legality、physical planning、instruction lowering、candidate cost或runtime
   completion；
2. 当前证据处于静态可见、板端观察、已校准、已支持、未知还是排除状态；
3. 未校准时compiler必须采用什么保守边界。

本文不复制总体架构、register手册或编号设计合同。跨profile硬件事实仍由
`docs/wafer-hardware-instruction-set-and-programming-model.md`和
`docs/wafer-register-level-instruction-spec.md`拥有；IR、runtime和verification语义仍由`tasks/06`、
`tasks/08-17`拥有。本文只保存可复现probe方法、profile绑定证据、结果和consumer映射。单个case参数不是
长期协议，原始日志也不进入compiler artifact。

## Pipeline position

```text
Pipeline position:
- Upstream artifact / IR:
  version-matched target profile、完整rank instruction program、typed buffer/view/range/effect、worker/engine、
  async token、wait/fence、target module/package以及可独立启动的board probe package。
- Current stage responsibility:
  用静态证据和安全、强oracle的板端microcase校准会改变compiler决策的硬件行为；把结论绑定到完整profile，
  区分semantic legality、completion correctness和performance observation。
- Output artifact / IR:
  profile-scoped calibration evidence及其compiler consumer边界；不产生新的program IR，不把测量side table、
  case名、buffer名或raw packet塞入production artifact。
- Downstream consumer:
  physical-dataflow candidate selection、layout/transfer realization、SPM/DDR planning、target-abstract
  compute/movement、instruction legality、communication completion、target lowering、runtime package、
  verification和后续multi-engine scheduling。
- User-level driver / named pipeline:
  board probe复用正常wafer-compile到package及wafer-run执行路径；production consumer仍是现有
  source-to-bundle named pipeline。
- Explicit non-goals:
  不建立第二份总体架构，不穷举未暴露ISA，不用aggregate PMU宣称cycle accuracy，不把一次observed结果
  外推到其它firmware/runtime/SKU，不以reset/power cycle作为测试步骤，不用缺失typed gate与前后heartbeat的
  任意queue overflow或缺失participant的DTE case探索错误行为。
- Completion gate:
  本文compiler-sensitive矩阵每一项都有可追溯静态合同和板端校准结论，或者有明确Unknown/unsupported、
  风险及保守compiler处理；校准vector与held-out vector分离，正确性oracle先于性能解释。只有完成该gate后，
  Q37才进入production multi-buffer和resource-aware scheduling。
```

## 1. 证据成熟度与profile绑定

每条结论只使用下列状态，不用“应该”“大概”代替证据：

| 状态 | 含义 | compiler可否消费 |
| --- | --- | --- |
| `static` | header、source、register定义或version-matched binary控制流可见 | 可用于形成待验证合同、hard safety bound或negative；不能产生未观测性能常数 |
| `board-observed` | 一个有完整profile身份、强oracle的板端vector通过 | 只能描述该vector；不能外推shape/layout/value或升级为通用能力 |
| `calibrated` | 区分向量排除候选错误语义，重复稳定，measurement basis闭合 | 可形成该profile的窄capability/cost输入 |
| `supported` | 冻结后的独立value/shape/layout held-out和真实纵向继续通过 | 可进入对应编号owner的production profile；仍不能改变上层数学语义 |
| `unknown` | 静态或板端证据不足，或多个解释仍不能区分 | compiler保留保守排序/显式completion，不能伪造支持或时间 |
| `excluded` | 已知危险、ABI不接受、缺恢复路径或不属于该调度域 | verifier/provider拒绝或测试框架不发射 |

每个板端证据至少绑定：

- device SKU/revision、good-tile map和tile count；
- driver、firmware、runtime ABI/version、instruction library和CRT/compiler identity；
- `serial_mode`、PMU enable/scope及使用的worker/tile；
- probe spec、input/output digest、resource mapping、repeat ordinal和原始counter；
- timeout、terminal status、完整output、guard/canary和cleanup结果。

PCI BDF、临时build目录和安装路径只是单次运行信息，不是稳定profile identity。普通测试批次不重复反汇编
或重做版本调查；profile签名变化时才重新qualification。

## 2. Probe protocol

### 2.1 分层执行

每个case按固定顺序进入板端：

1. host/device源码编译、packet/schema validator和no-card build smoke；
2. 批次开始时一次只读设备空闲/identity检查；
3. 单engine或已知安全baseline；
4. 同workload显式串行control；
5. disjoint并行候选；
6. 只有disjoint已证明可重叠的engine pair才进入exact/partial/adjacent dependency校准；
7. matching worker/local completion、cross-worker join、DTE或multi-tile completion；
8. 完整readback、canary、terminal status和资源cleanup。

timeout、设备无响应或状态异常会立即停止该批次；测试不自动重试，不调用NPU power、reset或firmware替换。
管理面显示idle、0%利用率或无进程只说明inventory/accounting表面状态，不能证明execution/completion面仍健康。
后续queue边界probe使用独立进程、单case，并在case前后各运行一次既有known-good Add heartbeat；任一heartbeat
timeout就停止全部板测，由用户决定恢复方式。

### 2.2 强oracle

一个case至少同时满足：

- 输入非零且能区分“未执行”“只执行首段”“旧cache”“错误layout”和正确结果；
- 每个queue entry使用独立地址或独立pattern，避免重复packet掩盖丢entry；
- 对语义输出做全范围CPU golden，对movement做DMA round-trip；
- output前后guard/canary保持，partial overlap按逐字节或逐元素composition比较；
- packet routing、worker、inclusive range和实际issue count与PMU差值一致；
- matching completion后检查结果，最后再用显式safety drain保证测试进程可恢复；
- 性能case配对serial control和重复样本；先证明正确，再解释counter。

`TsmExecute`原始返回值、单次wall time、全零output、最终safety drain之后才做的可见性检查、untimed
target model和`blocking_time == 0`均不能单独作为成功或并行证据。

### 2.3 地址关系

所有地址关系使用半开区间推理，packet inclusive end只在ABI边界转换：

| 关系 | 第二个区间 |
| --- | --- |
| exact | `B = [A, A + L)` |
| partial | `B = [A + L/2, A + 3L/2)` |
| adjacent | `B = [A + L, A + 2L)` |
| disjoint | `B`位于独立、带guard的远端range |

首轮base和length至少256B对齐。64KiB只可作为offset sweep样本，不能预设为物理bank周期。

### 2.4 Schema 2 hazard与queue安全边界

parameterized NCC probe在wire中显式携带first/second operand角色；RAW/WAR/WAW/RAR不能从lane名、
engine名或buffer名恢复。host只在同一engine pair的rounds=2 disjoint serial/window对照已经观察到
可重叠后，才允许执行exact/partial/adjacent hazard case。device以实际packet range回传证据，并用逐段
composition golden、未选operand guard和最终copyback共同闭合正确性。

register/header中的queue depth只描述静态storage/register形状，不等于host可安全连续发射的outstanding上限，
也不证明IB会在该数值驻留。旧CT边界probe的issue limit 5连续三次得到完整正确结果且记录
`control_after_issue=0x100`，issue limit 6第一次timeout，随后known-good Add也timeout；但旧实现同时保留
全部`TsmNew` builder到case结束，并在每次issue后、下一次issue前插入多组MMIO观察。对象生命周期和观察间隔都
污染了连续提交路径，因此这组5/6结果不能用于归因硬件queue，也不能把issue6 timeout解释为静态depth不安全。

修正后的probe在packet构造完成后立即`TsmDelete` builder，只保留独立packet，并把worker control读取紧邻
`TsmExecute`之后。普通calibration仍只运行保守multi-issue 1/2/4，TDMA只运行1/2。静态documented depth由独立
`documented-depth-manual`校准：一次只选一个engine和一个case、只运行一遍，并在前后各用known-good Add heartbeat
确认execution/completion面；它只验证连续提交恰好`D`条后的completion、instruction count、完整结果和guard，
不把成功提升成并发occupancy证据。当前profile上，修正后的CT exact `D=6`单样本及前后Add heartbeat已经完整
通过，证明该CT submission/completion vector可用；六次issue后的control观察均为`0x100`，没有证明六条请求
同时active或resident。

同一manual合同下，NE/RDMA/WDMA exact `D=6`与TDMA exact `D=4`也已由单engine、单case、单样本及前后
known-good Add heartbeat完整通过：`instruction_delta=6/6/6/4`，对应engine/full execution delta为
`492/2101/1578/292` cycles，blocking delta均为0，全部boundary/final result与guard mismatch均为0。
这些向量闭合当前profile的submission/completion/count/output/guard，但不证明active occupancy、full或
backpressure。

### 2.5 Instruction、dtype、layout 与 shape 覆盖合同

instruction-family qualification不能再用“某条FP16 Add通过”代表CT，也不能用“同一数学shape在host上
连续”代表所有device layout。校准矩阵的一个row由下面的typed tuple唯一确定：

```text
(engine, opcode/kind, operand form, input/output dtype, input/output layout,
 logical shape class, physical tail class, numeric domain, oracle)
```

其中opcode universe以current version-matched vendor header和`instr_def.h`为静态输入：CT opcode
`0..186`覆盖arithmetic、relation、logic、transcendental、activation、reduce、pool/unpool、DataMove、
convert和peripheral；NE另覆盖GEMM、Conv与DepthwiseConv配置面；RDMA、WDMA、TDMA GatherScatter和Direct
DTE分别保留自己的descriptor/transport矩阵。catalog必须对每个公开entry给出以下三种 disposition 之一，
不能因没有实现case而从清单消失：

- `board-positive`：已具备固定输入、完整expected、physical write span和guard，可进入隔离板端suite；
- `static-negative`：该dtype/layout/form在当前typed compiler/ABI中明确非法，由verifier、lowering或
  required-symbol gate拒绝，不发射raw packet；
- `isolated-deferred`：header/opcode存在，但语义、writeback、随机性、别名或completion边界尚不足以形成
  强oracle；只做constructor/packet/link静态检查，待独立manual gate，不能计为功能覆盖。

浮点CT的基础dtype矩阵为FP16、BF16和FP32。普通值域、signed zero、Inf、subnormal、NaN分别记账；
没有NaN payload/quieting证据时不能由普通值结果外推。convert按opcode的完整source/destination pair记账，
integer、TF32和rounding mode不能折进“FP16/BF16/FP32已测”。relation同时区分value output和bitpacked bool
output；logic同时区分value logic和bitpacked bool logic。operand form至少包括：

| 指令组 | 必须区分的form |
| --- | --- |
| arithmetic binary | `VV`、`VS`、`VuV`、`VuVLoop` |
| relation | 六种predicate × `VV/VS/VuV/VuVLoop` × value/bool output |
| logic | value `V/VV/VuV/VuVLoop`与bitpacked bool `V/VV/VuV/VuVLoop` |
| unary/transcendental/activation | 各公开opcode的vector form |
| reduce/pool/unpool | kind、axis/window、indexed/non-indexed output及dtype |
| DataMove | mirror、transpose、rotate、NCHW/NHWC、concat、pad、Img2Col、TensorNom、
  GatherScatter、MaskMove/MaskGather；compiler broadcast另按GatherScatter materialization验证 |
| peripheral | count/arg、memset、bit2fp、bilinear、LUT、rand/factorize/elem-mask按各自arity和
  writeback/randomness disposition |

这里的三种“broadcast”不能合并：CT `VuV/VuVLoop`是短向量operand form；source/tile broadcast在
current production pipeline中materialize为一条或多条typed GatherScatter；raw Direct DTE broadcast属于
multi-destination communication ABI。三者的地址、completion和consumer不同。

layout矩阵以`Tensor/NTensor/Cx/NCx`的真实physical encoding为准，而不是为每个opcode盲做笛卡尔积：

| layout row | 正向检查 | 必须区分的negative/边界 |
| --- | --- | --- |
| compact `Tensor/NTensor` | logical element、physical span、tail和guard一致 | rank/shape/dtype不匹配、越界descriptor |
| `Cx` | full channel block、compact `C0` tail、256B physical padding逐段oracle | `C=B-1/B/B+1`及第二个full block后tail；padding不能按逻辑值外推 |
| `NCx` | 每个N slice独立Cx block/tail，跨N步进和整份footprint正确 | 不得把rank-3/4静默flatten成Cx；N slice间guard/poison不得串扰 |
| layout materialization | Tensor↔Cx/NCx、transpose、slice、broadcast、concat由显式movement覆盖
  all-and-only logical points | 不能把native transform、metadata view和GatherScatter composite混成一条能力 |

current production CT elementwise仍只接受`Tensor/no-invalid-lane`。Cx/NCx row只有在显式
Tensor↔Cx/NCx materialization后执行CT，或未来建立segmented/full-physical typed TargetCall、CRT和SystemC
纵向后，才能成为正向production case；raw packet直接扫physical footprint只能校准hardware行为，不能绕过
compiler legality。plain GEMM按rank-2 Cx验证，batched GEMM按单leading-batch NCx验证；Conv/Pool类按其
NHWC/NCx wrapper关系独立验证，不能从GEMM layout外推。

为避免只测启动开销，新的数值/布局row使用有诊断能力的中等shape：

- CT Tensor主向量使用至少8192个logical element，并另设非256B整除的held-out tail；`VuV`使用不超过
  64的非平凡unit，`VuVLoop`同时让full与tail字段非零；
- Cx/NCx至少覆盖`C=63/64/65/127/129`中的block边界和`N>1, H*W>1`，每个physical padding区保持独立
  canary；
- NE FP16/BF16至少覆盖`M=64,K=128,N=128`的非平凡累加、`M=65,K=129,N=129`的M/K/N tail、batch2及
  NN/NT/TN/TT中合法组合；每个case分别记录logical output与physical Cx/NCx span；
- transform/layout movement使用至少数千元素并采用非对称维度，防止1x1、方阵或全零输入掩盖轴顺序。

完整矩阵按opcode/form、dtype、layout/shape-class分suite串行执行，不要求一次进程跑完全部case。每个
board-positive row仍是独立launch和外层timeout；安全deterministic suite在批次前后做known-good heartbeat，
queue/deferred/random/multi-writeback等高风险manual row才逐case做前后heartbeat。no-card preflight只闭合
catalog、oracle、compiler verifier、device object/link和package协议，不能把row状态升级为
`board-observed`或`calibrated`。本节当前只冻结case规划和准入合同；probe、catalog与oracle实现放在规划
评审后，不在同一批文档修改中提前展开。

本轮取得显式manual授权以区分“静态depth”和“总提交数”后，五类engine分别执行typed tight `D+1`
manual gate：CT/NE/RDMA/WDMA各连续提交7条，TDMA连续提交5条；每个向量仍为单engine、单case、单样本，
packet builder预先释放，相邻`TsmExecute`之间不插入register MMIO，window后才执行matching wait和完整
result/guard oracle。五个向量的submission、completion、instruction count、完整结果和guard均通过，
且未观察到PMU blocking。它们证明当前profile可接受并完成超过静态pending depth的总提交数，因此
documented depth不是一次完整submission/completion lifetime的总提交上限。

这些短workload在唯一control观察前均已排空，没有观察到resident数量、queue full或backpressure；不能把
`D+1`成功写成对应数量的请求同时驻留，也不能据此选择production outstanding window。typed tight `D+1`
只保留为隔离manual gate；任意更深提交在真实occupancy/full行为闭合前仍不执行。单次cycle与engine
execution delta只保留为原始profile样本，不外推为固定cost。

### 2.6 DTE/SPM PMU有效样本边界

DTE/NCC有序交互probe使用独立transport PMU采样合同。设备按high-low-high读取DTE channel与SPM port
64-bit counter，同时记录window前的enable值、window内enable是否变化和stable mask；host只有在DTE
所需channel、SPM所需port均已启用、scope未变化、全部split counter稳定且至少一个transport delta非零时，
才把样本记为`raw_observation`。enable缺失、scope变化或全部delta为零统一记为`inconclusive`，不能进入
measurement-basis校准；未知metadata位、非只读采样或unstable counter直接拒绝。

transport counter不替代正确性oracle。每个rank分别保护DTE source、receive和NCC compute的SPM payload，
设备端在completion边界检查三组前后guard及FP16结果；host同时检查DDR输出payload前后guard和全量expected。
当前schema、cross-device build/link、synthetic parser/oracle和no-card package路径已通过，板端counter
measurement basis仍为`unknown`。

## 3. Compiler-sensitive calibration matrix

下表是Q37校准入口。每个row最终必须变为`calibrated/supported`，或保留带明确保守行为的
`unknown/excluded`。不要求所有字段笛卡尔积，但每个语义分支必须有能区分错误实现的vector。

| 域 | 待闭合事实 | 最小probe与oracle | compiler consumer | 当前状态 |
| --- | --- | --- | --- | --- |
| profile qualification | runtime/instruction/CRT身份、tile map、PMU、3个worker `serial_mode` | 一次只读identity/CSR，禁止写power/reset | 14-17 target/runtime profile | 当前profile `board-observed` |
| constructor ownership | instruction method-table对象可以位于local地址0；allocation成功不能由pointer truthiness判断 | 第一份raw constructor，精确PMU count和结果；显式ownership bit控制delete | 14 CRT lowering与probe infrastructure | `calibrated` |
| execute result | `TsmExecute`成功和invalid type路径都可返回1，raw rc不能区分成功 | packet legality、目标queue count和结果共同判定 | 14/15 structured error boundary | `calibrated`；rc只记录不判成功 |
| packet routing/range | `inter_type`映射、worker编码、begin/end materialization和inclusive end | 五类单engine，执行后读实际register/packet，完整结果 | 11/14 instruction legality | CT/NE/RDMA/WDMA部分`board-observed`；TDMA Memset routing/range `board-observed` |
| CT numeric/form | opcode 0..186；arithmetic/relation/logic的`VV/VS/VuV/VuVLoop`、value/bool output；f16/bf16/f32、convert pair、rounding、NaN/Inf/subnormal/signed-zero、large-shape tail | opcode/form×dtype typed catalog；至少8192元素普通值、非256B held-out tail、短向量unit/full-tail、完整bit oracle | 10/11/17 numeric capability | existing 60-row有限catalog及Add held-out仍为既有`board-observed`；完整opcode/form×dtype扩展仍为`unknown`，不由既有Add外推 |
| instruction × physical layout | Tensor/NTensor/Cx/NCx原生资格、materialization路径、C0 tail、N-slice步进、padding lane | valid组合exact output/span/canary；需materialize组合重放Tensor↔Cx/NCx movement；非法组合verifier/packet negative | 08/10/11 physical legality | current CT production只准Tensor/no-invalid-lane，plain/batched GEMM分别使用Cx/NCx；其它组合必须按row闭合，不能由同opcode Tensor结果外推 |
| DataMove/layout | mirror/transpose/rotate、NCHW/NHWC、concat、pad、Img2Col、TensorNom、GatherScatter/MaskMove/MaskGather及compiler broadcast materialization | 非对称large-shape、axis-sensitive payload、all-and-only logical point oracle、physical guard | 08/10/11 movement legality | Pad/Img2Col和部分GatherScatter composite有既有证据；concat、native transforms、MaskGather及large-shape broadcast仍为`unknown`，其中缺安全强oracle的row使用`isolated-deferred` disposition |
| NE numeric/layout | f16/bf16、accumulation、transpose、C0 tail、padding、K/M/N边界 | 非零identity/diagonal GEMM、完整padded range/canary | 08/10/11/17 | f16/bf16 accumulation、selected batch/M/K/N tail与orientation，以及f16 raw psum `board-observed`；未覆盖组合仍`unknown` |
| RDMA/WDMA descriptor | byte/logical-element stride转换、iteration、inclusive range、tail | contiguous + 1/2/3D stride，非零round-trip和guard | 08/11/14 | contiguous与既有large GEMM `supported`；FP16 1/2/3D stride round-trip、holes和guards `board-observed` |
| TDMA Memset | element count、byte stride、raw logical iteration、inclusive range和dtype packet encoding | whole/128B×32/64B×64 geometry，I8/F16/BF16 raw与CRT，全range和guard | 10/11/14 | 普通dtype descriptor `calibrated`；I8/F16/BF16 vectors `board-observed` |
| TDMA BOOL fill | native `Fmt_BOOL` completion与bitpacked physical-footprint实现 | native小range timeout隔离；production BOOL→I8 byte fill需独立raw register、全range和guard | 10/11/14 | native `Fmt_BOOL`在当前profile `excluded`；直接CRT I8 physical16向量`board-observed`，但BOOL→I8 production canonicalization仍待held-out |
| TDMA movement variants | GatherScatter和其它DataMove的byte count、stride/iteration、range与kind-specific geometry | 每个已准入kind使用能区分错误descriptor的非零pattern、全range和guard | 08/10/11/14 | f16 Pad vector `board-observed`；GatherScatter已有compiler/model路径，其它variant仍`unknown` |
| SPM capacity/reservation | allocatable range和保留区 | boundary-positive与verifier negative；不触碰保留区 | 09/11 | 静态hard bound；board边界held-out未闭合 |
| SPM alignment/bank | 256B legality、非1024-bit访问代价、bank/color映射 | disjoint offset sweep，固定长度/engine pair/serial control | 09 placement与06 cost | 256B静态；exact bank mapping `unknown` |
| DDR/cache/coherence | host H2D、Kcore cache、DMA completion和host publication是不同域 | Kcore read前invalidate对照、DMA round-trip、matching drain后D2H | 09/12/14/15 | stale-cache机制`calibrated`；完整direction matrix待闭合 |
| queue shape与连续提交边界 | register/header中CT/NE/RDMA/WDMA静态depth为6、TDMA为4；该数值描述pending storage，不是完整lifetime总提交上限，active occupancy和full行为不能由形状或总提交数推出 | 普通calibration只跑1/2/4（TDMA 1/2）；隔离manual gate先验证恰好`D`，再经显式manual授权执行typed tight `D+1`，均为单engine/case/sample、前后Add heartbeat和完整count/output/guard；禁止任意更深提交 | 10/11/16 | 五类engine的exact `D`与typed tight `D+1` submission/completion/count/output/guard均为当前profile `board-observed`；短workload在观察前已排空且blocking为0，只证明总提交可超过depth；所有engine的occupancy/full/backpressure仍`unknown` |
| worker scope | worker0/1/2 routing、default wait和`bywork` scope | CT三worker；matching wait后、safety drain前读CSR/result | 10/11/14/15 | CT worker0/1/2 routing与matching `bywork`均`board-observed`；version-matched静态反汇编显示default wait轮询worker0，现有worker1板测在观察前自然排空，故default跨worker scope仍`unknown` |
| cross-engine overlap | 五类engine全部10个pair的可重叠性和共享资源 | disjoint backlog2 + serial control；仅在不触及任一engine full-depth时增加backlog4；`Ea+Eb-FU`重复正值 | 06/10/11/16 cost/scheduling | 旧CT+RDMA正overlap降级为`historical/inconclusive`；本轮两个方向r4对照median均为0，其余pair当前workload也未观察到PMU overlap；不外推成硬件不支持并行，compiler全部串行 |
| address dependency | busytable对RAW/WAR/WAW/RAR及exact/partial/adjacent/stride envelope的处理 | 仅对已证实可重叠pair做composition golden和PMU对照 | 09-11 legality/scheduling | 显式operand/range schema与composition oracle已通过no-card；本轮两次RAW选择均在hazard发射前被资格门禁拦截，板端hazard未执行；当前暂不适用 |
| issue overhead | wrapper构包/heap间隔对短window的影响，prepared issue是否值得materialize | 同packet序列wrapper与prebuilt对照 | 06/14 candidate/lowering | 旧RDMA+CT差异为`historical/inconclusive`，本轮未复现稳定正overlap，不构成新IR语义或收益结论 |
| local completion | default wait、`bywork`、local fence的范围和visibility | matching wait后立即CSR/Kcore oracle，再做safety drain | 09-11/15 | worker0/1/2 matching `bywork`正向`board-observed`；三轮worker0对照均观察到逐指令wait比window末尾一次wait更慢，但不形成固定cost；default/local-fence跨worker scope仍`unknown` |
| cross-worker join | 多worker并行、仲裁与地址依赖是否跨worker | disjoint w0/w1/w2，逐workerjoin；同地址不做无序正向 | 10/11/15 | 三worker disjoint CT matching join `board-observed`；并行性、仲裁与同地址行为仍`unknown/excluded` |
| Direct DTE | source read、destination visibility、participant、channel/FSM、terminal status | 16-rank receiver-first；producer→DTE、DTE→consumer和disjoint顺序 | 13-16 | 四个有序case `board-observed`；sender overlap `unknown` |
| multi-tile arrival | full-card/subgroup barrier的participant与复用合同 | production 16-rank正向；缺participant/错误坐标不测试 | 13/15 | 两轮反向错峰的16-rank `hrt_barrier`均16/16 marker正确、0 mismatch/crosstalk，full-card复用`board-observed`；通用subgroup `unknown` |
| host launch/runtime | kernel/model launch、resource staging/readback、timeout、failure cleanup | 同package schema、exact output、terminal/cleanup | 14-16 | 已有kernel/model与16-rank路径`board-observed/supported`，按owner证据解释 |
| NCC PMU basis | instruction count、engine exec、global union、worker scope、wrap稳定读取 | 单engine等式、pair union、high-low-high和重复样本 | 16与后续Q9 | worker0 engine/union `calibrated` |
| DTE/SPM/TMNOC PMU | counter scope、unit和与workload相关性 | 独立单域workload和held-out payload sweep | 16与后续Q9 | register shape `static`；measurement basis `unknown` |
| SCALAR/CSR ordinary issue | 是否属于`TsmExecute` typed queue | 静态negative，不发送未知packet | 11/14 | 当前ABI `excluded` |

## 4. 当前profile已经闭合的事实

本节记录当前已取得的事实，不把精确cycle提升成跨profile硬件常数。

### 4.1 NCC mode、queue和cross-engine overlap

- 3个worker的`serial_mode`只读值均为0。
- worker 0上disjoint RDMA/CT使用64KiB强sentinel workload；低深度issue count `1/2/4`均在显式drain和
  WDMA round-trip后完整exact。旧sweep中的aggregate issue-count 6只能保留为该配对workload的历史观察，
  不能作为任一单queue可安全outstanding 6条的证据。
- 旧样本曾记录prebuilt packet在backlog 2出现正overlap，one-shot wrapper在较深backlog才形成明显窗口；
  issue-count-4重复样本的prebuilt overlap显著大于wrapper。这组旧正值在本轮资格复测中
  不可复现，现降级为`historical/inconclusive`；它不再证明当前profile的CT+RDMA并行能力，也不形成
  prepared-issue收益或cost输入。
- canonical CT→RDMA r4 disjoint资格对照的serial/window各运行3个样本；本轮window pairwise excess为
  `[78,0,0]`、median为0，未通过稳定正overlap资格门禁。新增同RAW顺序RDMA→CT r4 disjoint对照也各运行
  3个serial/window样本，全部result、guard和instruction count正确，blocking delta为0；但两组每个样本都满足
  `ct_exec + rdma_exec == full_exec`，median excess均为0。
- 其余9个disjoint pair随后在current profile以单进程串行运行；每个serial/window配置各取3个样本，
  instruction count、完整result和guard全部正确，blocking delta均为0，整批最终known-good Add heartbeat通过。
  CT+WDMA、RDMA+WDMA、CT+NE、NE+RDMA、NE+WDMA在r2与r4的serial/window
  pairwise-excess median均为0。CT+TDMA、NE+TDMA、RDMA+TDMA、WDMA+TDMA只运行r2，serial/window
  median也均为0；r4会触及TDMA静态depth 4，故未运行。
- 上述零值只表示当前profile和当前workload没有观察到PMU overlap，不证明这些pair在不同长度、布局或
  issue形态下永远不能并行。两次RAW exact/partial/adjacent选择都在hazard发射前被disjoint资格门禁拦截，
  没有执行hazard。整批前后known-good Add heartbeat均通过、卡健康，且未调用reset或power接口。当前profile
  下compiler对所有engine pair保守串行；只有未来同方向对照稳定达到median正overlap才重新开放对应hazard
  校准与并行候选。
- CT在worker 1和worker 2各执行一个4KiB FP16 Add；packet `inter_type`分别编码为`0x100/0x200`，
  matching `TsmWaitfinish_bywork` mask分别为`0b010/0b100`，对应worker CT instruction delta均为1，
  CT/full execution delta分别为`78/79` cycles。三worker disjoint join随后以mask `0b111`执行，
  worker0/1/2 CT instruction delta各为1，CT/full execution delta均为238 cycles。三个case在
  safety drain前的boundary result/guard及final result/guard均无mismatch，blocking均为0，前后Add
  heartbeat通过；这闭合当前profile的CT worker routing、matching `bywork`和disjoint join正确性，
  不证明跨worker并行、仲裁或default/local-fence跨worker scope。
- 旧single-engine CT边界样本中，issue limit 5连续三次完整正确且
  `control_after_issue=0x100`，issue limit 6第一次timeout，随后known-good Add也timeout。但该probe保留全部
  `TsmNew` builder，并在相邻issue之间读取多组MMIO，故不能从该timeout判断硬件queue depth或安全提交上限。
- 修正probe在packet构造后立即删除builder，并在`TsmExecute`后立刻读取control。重启后先运行known-good Add，
  1/1 exact且cleanup完成；随后`ct-raw-documented-depth6-window`单样本中六次`execute_rc`均为1，
  `worker0.ct instruction_delta=6`、CT与full execution delta均为473 cycles、blocking delta为0，六个
  boundary/final result及guard mismatch均为0，slot 6独立地址的结果也正确；最后一次known-good Add同样
  1/1 exact且cleanup完成。这把旧timeout归因于受污染probe路径，而不能归因于硬件documented depth。
- 六次issue后的`control_after_issue`均为`0x100`，观测点处IB为空闲，因此本次只把CT exact `D=6`的
  submission/completion/count/output/guard记为当前profile `board-observed`；它不证明六条同时active或
  resident，也不校准queue-full/backpressure。
- NE/RDMA/WDMA exact `D=6`与TDMA exact `D=4`随后也分别在独立单样本及前后known-good Add heartbeat下
  完整通过。四个向量的instruction delta分别为`6/6/6/4`，对应engine/full execution delta分别为
  `492/2101/1578/292` cycles，blocking delta均为0，全部boundary/final result与guard mismatch均为0。
  这些结果将五类engine的exact documented-depth submission/completion/count/output/guard记为当前profile
  `board-observed`，但同样不提供active occupancy、full或backpressure证据。
- 五类engine随后分别在明确manual gate中执行typed tight `D+1`：CT/NE/RDMA/WDMA各7条，TDMA 5条。
  每个向量都预先释放builder，相邻`TsmExecute`之间不做register MMIO，并在window后执行matching wait和
  完整oracle；五组instruction count、result、guard和completion均正确，blocking delta均为0。这证明
  documented depth不是完整lifetime总提交上限。
- 这些短workload在唯一control观察前都已排空，所以没有看到`D+1`条resident、queue full或backpressure；
  `TsmExecute`静态路径也没有software queue-full check。实际full行为继续为`unknown`，不得用本结果选择
  production outstanding window。单次cycle不外推为固定cost，任意更深overflow不执行。

overlap只按下面的PMU关系解释：

```text
pairwise_excess = engine_a_exec + engine_b_exec - fu_union_exec
```

`FU_EXE_TIME`是active engine时间的union；`*_BLOCKING_TIME`是queue backpressure，不是dependency stall。

### 4.2 单engine correctness与ABI观察

- CT f16 Add使用非零输入、完整fp16 golden、guard和DMA round-trip通过。
- instruction-family typed catalog的60个safe case已逐个串行launch并通过完整bit oracle、SPM guard、
  terminal与cleanup：f16/bf16 Neg/Add/Sub/Mul/Max/Min/Pow2/Relu，I8→f16/bf16、bf16→f16、
  f16→bf16/i16 convert，f16 Sum/Max、bf16 Min/Avg reduction，f16/bf16 Bit2FP+MaskMove select，
  f16 NE GEMM、f16 TDMA Pad、f16 TDMA Img2Col、f16 PoolMax、peripheral LUT16与f16 peripheral
  ArgMax/ArgMin、f16 indexedmax→maskunpool composite，以及下面的CT/NE held-out vectors。PoolMax使用
  `[1,2,4,64] -> [1,1,2,64]`、无padding、2x2 kernel/stride，两个输出窗口的128个FP16结果逐bit正确，
  256B physical output span和suffix guard均通过。ArgMax在含负数的128元素输入上返回
  `100@index73`；ArgMin在全正128元素输入上返回`0.5@index42`。两者都由CRT等待writeback后把FP16 value
  写到slot `[0:2]`、uint32 index写到`[4:8]`，并保持`[2:4]` poison不变。ArgMin负数对照返回了
  `-30@index0`，没有返回真实最小值`-100@index42`，因此current profile只闭合ArgMin正数普通值domain，
  负数域保持unsupported。该批只证明catalog中明确列出的value、dtype与geometry组合，不外推未覆盖opcode、
  NaN或其它instruction family。
- CT Add新增f16/bf16 logical tail130向量：260B逻辑结果逐bit正确，512B physical output span及suffix
  guard通过；它闭合当前Add packet对该非block-aligned tail的写范围，不外推其它CT opcode。finite f32
  Add的128元素512B结果也逐bit正确，只把当前Add opcode的FP32路径记为`board-observed`。
- f16/bf16 special Add均以加零向量覆盖正负零、正负无穷、正负max-finite、正负min-normal和正负
  min-subnormal，128元素结果逐bit正确。该向量明确不含NaN，因此不能形成NaN propagation、payload或
  quieting规则。
- LUT16 source是128个`uint16`字节偏移`2*((37*i+11)%128)`，不是FP16数值index；table/output才是
  FP16 payload。128项互异table经过非顺序lookup后的完整256B输出逐bit正确，physical span与suffix guard
  均通过。该证据只闭合raw 16-bit byte-offset lookup，不外推其它index编码或不同source/table count。
- TDMA Img2Col使用FP16 source `[1,3,3,64]`、2x2 kernel、1x1 stride和零padding，vendor destination
  `[1,4,4,64]`按`ky,kx,oh,ow,c`即`[N,Kx*Ky,outH*outW,C]`展开；1024个结果逐bit正确，2048B
  physical output span与suffix guard通过。既有compiler verifier曾按`[Kh,Kw,Sh,Sw]`解释参数并要求
  `[N,outH,outW,C*Kh*Kw]`，只因旧测试是1x1而没有暴露；当前已改为vendor合同并增加非对称非1x1正反例。
- NE BF16 1x16x16 identity GEMM返回32B logical output且逐bit等于BF16 lhs，256B physical span与
  suffix guard正确。该case隔离闭合BF16 format和已验证normal/normal layout，不证明非平凡accumulation
  rounding、transpose、batch或logical tail。
- BF16 PoolMax与TDMA Img2Col沿用已闭合FP16 geometry，但使用BF16可精确表示的小整数隔离format路径：
  PoolMax两个2x2/stride2窗口的256B结果逐bit正确；Img2Col 2x2/stride1 kernel-major重排的2048B结果
  逐bit正确。两者的physical span、suffix guard、terminal、cleanup和后置Add heartbeat均通过。
- Unpool composite先以indexedmax把FP16 `[1,2,2,64]`写成value与i16 index
  `[1,1,1,64]`，经local fence后由maskunpool恢复`[1,2,2,64]`。输入让64个channel的最大值位置按
  `channel % 4`分布到四个空间位置，因此能区分“传入i16 SPM index buffer地址”与旧scalar-index误解；
  512B输出逐bitexact，256B auxiliary index span的suffix guard、terminal、cleanup及后置Add heartbeat
  均通过。该结果把current FP16 2x2/stride2 indexedmax→maskunpool组合记为`board-observed`，不外推
  indexedmin、avg、其它geometry或dtype。compiler侧indexed pool第二个dest为same-shape i16 SPM memref，
  mask/unpool显式读取该operand；target lowering只在静态SPM range适配`uint32_t`时传其起始地址，avg传0。
- ordinary FP16 Conv使用非对称`Sx/Sy=2/1`、input `[1,1,3,4]`、HWOI weight `[1,1,4,4]`和output
  `[1,1,2,4]`，8个结果逐bit正确；16B logical result、256B physical span及suffix guard通过。该向量能
  区分旧HWIO channel轴和对称stride误解，并与已修正的compiler verifier/target ABI合同一致。
- 同一Conv geometry以BF16 tight payload独立上板，8个结果、16B logical result、256B physical span及
  guard同样逐bit正确；该证据只新增BF16 format资格，不外推非平凡accumulation rounding或其它geometry。
- NE BF16 `M1K16N16`非平凡累加向量让每个输出具有16个非零K贡献；输入和FP32点积均为精确二进制数，
  expected同时包含非tie round-up/down。32B结果逐bit正确，256B physical span与guard通过，因此排除逐项
  BF16累加和末端截断；该向量仍不证明transpose、batch或logical tail。
- NE的held-out矩阵随后逐项通过完整bit oracle、physical span及guard。FP16覆盖非平凡累加舍入、
  `M=4`、batch2/`M=8`、`K=17`、`N=17`、`N=65` retained-C0 tail、NT/TN/TT orientation和独立
  raw psum writeback；BF16覆盖batch2/`M=8`、`K=17`、`N=65`及NT orientation。raw psum向量同时验证
  非零GEMM result与非零psum auxiliary span，但它不证明跨tile reduction或communication。未出现于该矩阵的
  dtype、orientation与tail组合继续为`unknown`。
- reduction首次运行暴露的是probe ABI错误而非硬件错误：四个case的逻辑结果均为128B，但CT会写满256B
  physical block，后128B是padding。catalog把allowed output span误写成128B，因而准确报告128B guard
  mismatch；将`result_bytes=128`与`output_span=256`分开后，四个reduction及余下case全部通过。结果逻辑域、
  physical write span和suffix guard必须分别建模，不能用逻辑shape缩小硬件写范围。
- RDMA和WDMA分别以非零pattern、全range round-trip和精确instruction count通过。额外FP16
  1D/2D/3D strided round-trip分别验证compact payload、scatter位置、stride holes和双侧guard，三种
  descriptor均逐字节exact；这闭合当前向量的byte-to-element wrapper转换，不外推动态、负stride或超过
  三层descriptor。
- NE f16 16x16 identity GEMM通过；实际source/output register range均覆盖256B。retained logical
  `C0=16`使用compact stride 16，错误使用full block stride 64只会得到前4个正确对角元素。该case证明layout
  vector必须能区分logical tail和full physical block。
- vendor constructor的第一份method-table allocation可以返回数值0的有效local地址；probe和CRT ownership
  必须由显式状态表示，不能把地址0当作allocation failure。
- `TsmExecute`当前实现的成功分支与invalid-type分支都可返回1；成功必须由packet legality、目标queue PMU
  count、terminal status和结果共同证明。
- NE packet的end字段会在execute路径根据shape重新物化；只检查prepared packet会看到零end。需要在issue后
  读取实际NE register range。

### 4.3 TDMA Memset descriptor、dtype与BOOL边界

- `TsmPeripheral::Memset`把`St_StrideIteration`的iteration字段作为raw logical trip count写入TDMA
  source stride/iteration register，不使用RDMA/WDMA和Direct DTE路径中的`iteration - 1`编码。inactive
  dimension必须写1；把三层iteration清零会形成非法descriptor。
- 普通非bitpacked dtype的inclusive destination range为：

  ```text
  dst_end =
      dst
      + Σ((iteration_i - 1) * stride_i)
      + elem_count * element_bytes
      - 1
  ```

  canonical contiguous descriptor使用
  `{stride0 = physical_bytes, iteration0 = 1, stride1 = 0, iteration1 = 1,
  stride2 = 0, iteration2 = 1}`。先前4KiB请求只改首128B的根因是全零iteration，不是128B
  hardware上限；packet中的full-range `dst_end`本身不足以证明实际写满。
- I8的whole 4KiB、128B×32和64B×64 raw descriptor均经完整WDMA readback和前后guard精确通过；
  production CRT的I8 whole 4KiB也与raw path一致。FP16和BF16各有256B raw与CRT向量精确通过，
  `elem_count=128`且实际source descriptor为`stride0=256, iteration0=1`。
- 独立CRT I8 physical16向量也完成精确readback与guard验证；它只证明直接I8 packet的16B physical
  footprint，不等价于production BOOL→I8 canonicalization已闭合。
- 在当前profile的配对样本里，64B×64比whole 4KiB和128B×32用时更长。该结果只是一条
  profile-scoped performance observation，不是stride legality、固定代价或跨profile cost常数。
- 原生TDMA `Fmt_BOOL`的小range安全case在10秒内未完成；执行上下文被隔离且没有自动重试/reset/power。
  随后的只读设备状态回到idle且没有残留进程只说明管理面状态，不能证明execution面健康。该packet因没有
  可接受的completion证据而在当前profile标记为`excluded`，但不能据此推断其它profile或其它TDMA kind。
- production只对`physical_footprint` BOOL fill定义替代路径：Instr/TargetCall保持bit count和canonical
  bool scalar，TX81 CRT边界把完整physical bytes改写成`Fmt_INT8`、byte count和`0x00/0xff` byte splat，
  继续由TDMA Memset执行。该路径会有意覆盖最后一个byte中的unused bits，故不能用于
  `logical_valid` BOOL fill；其板端held-out通过前不升级为`supported`。

### 4.4 Cache、completion与DTE

- Kcore普通load读取复用的cacheable DDR input可能命中旧cache；对range执行匹配cache invalidate和fence后
  恢复exact。同一payload的NCC DMA round-trip不受该旧load影响。
- 因此host H2D、Kcore DDR load、NCC local drain和host-visible publication是不同visibility合同，不能互换。
- version-matched静态反汇编显示`TsmWaitfinish()`轮询default worker 0；
  `TsmWaitfinish_bywork(worker)`轮询指定worker，current `wafer_tx81_local_fence()`直接调用default
  `TsmWaitfinish()`。现有worker1 depth-6 RDMA向量，以及worker0短TDMA/worker1 depth-6 NE向量，
  都在default wait的boundary观察前自然排空；default和matching `bywork(1)`均得到正确结果，因而这些
  板测不能证明default wait会等待worker1。跨worker scope继续为`unknown`，compiler必须按实际participant
  逐worker join。
- worker0上6条CT的“每条后wait”和“window末尾一次wait”对照独立运行三轮；每轮结果、guard和completion
  都正确，且频繁wait的plan cycles均更高。该方向性观察支持compiler把wait放在latest-legal completion
  boundary并合并相邻wait，但样本不形成固定wait latency、比例或跨workload cost常数。
- 16-rank production Direct DTE基线、NCC producer经local drain后作为DTE source、DTE receive wait后由NCC
  consumer，以及两个disjoint安全顺序均16/16 exact、terminal success、canary正确。
- 当前Direct DTE helper把真实attach/send/wait/release集中在wait路径，不能用现有ABI证明sender与NCC
  overlap；该维度保持Unknown。
- 独立full-card `hrt_barrier`以两个epoch复用同一barrier；epoch1按rank递增错峰，epoch2反向错峰。
  两轮均得到16/16 rank-specific marker、0 mismatch和0 crosstalk，且各rank等待cycle随两次反向错峰呈
  相反次序，排除“marker自然完成而未等待最晚participant”的弱解释。该结果只把当前16-rank full-card
  participant与两次复用记为`board-observed`；subgroup barrier仍无证据。
- barrier probe首次运行在host报告Direct DTE terminal status `0xffffffff`，原因是手写cluster device
  entry声明了transport status resource，却遗漏
  `wafer_tx81_direct_dte_begin_after_prepare()`/`wafer_tx81_direct_dte_finish()`，不是
  `hrt_barrier`超时。补齐terminal publication ABI后同一两轮barrier通过；手写cluster probe必须把
  status begin/finish视为launch ABI，而不能用未发布的poison status判断硬件同步失败。

## 5. 校准 case 规划与执行批次

本节是后续实现和实卡执行的唯一case拆解入口。当前先完成inventory、case分层、shape/value选择、oracle和
执行顺序；不在规划尚未评审时写probe。已有板端事实不因新规划失效，但只作为对应row的既有证据，不替代
新增row。

### 5.1 Case row、分层和完成证据

每个catalog row必须显式记录下列字段。opcode编号和vendor method只用于关联静态证据，稳定语义仍由
family/form/dtype/layout描述。

| 字段组 | 必填内容 |
| --- | --- |
| semantic | engine、instruction family、opcode/kind、operand form、input/output角色 |
| typing | input/output dtype、rounding、zero-point、bitpacked BOOL或value output |
| physical | input/output layout、logical shape、physical span、tail/padding class、读写半开区间 |
| data | numeric domain、固定seed或固定pattern、input digest；不同operand不能复用同一pattern |
| oracle | CPU/formal expected、exact bits或预先声明的误差口径、全logical output、padding/guard/canary |
| execution | disposition、rank/tile/worker scope、static/no-card/board gate、profile identity、timeout、terminal、PMU issue count与cleanup |

每个合法语义分支分三层：

1. `calibration`使用诊断能力最强的main shape/value，排除未执行、首段执行、form解释错误、轴错误和layout错误；
2. `held-out`使用独立shape、value和tail，不从calibration seed机械缩放，只用于成熟度升级；
3. `production-vertical`从真实source/IR重放到package，证明typed consumer真的使用该能力。

所有公开opcode/form/layout组合都必须出现在inventory，但不是全部发raw packet：native合法组合进入
`board-positive`；需要显式movement的组合由materialization正向case闭合；typed非法组合进入
`static-negative`；无法形成安全强oracle的组合进入`isolated-deferred`。因此“覆盖全部组合”指每个组合都有
明确结论和gate，不是把非法或危险组合也做成板端笛卡尔积。

### 5.2 CT opcode、form 与 dtype suite

CT以current header的`0..186`为完整inventory。除bitpacked BOOL自身没有浮点dtype外，浮点入口均分别建立
FP16、BF16、FP32 row；不能由同opcode的另一dtype外推。

| opcode/family | 计划case | main oracle与held-out |
| --- | --- | --- |
| `0..5` unary arithmetic | Abs、Recip、Square、Sqrt、Rsqrt、Neg × FP16/BF16/FP32 | 8192普通值；8197 tail。Recip/Sqrt/Rsqrt分别含正域、零和负域隔离row |
| `6..29` binary arithmetic | Max/Min/Add/Sub/Mul/Div × `VV/VS/VuV/VuVLoop` × 3 dtype | 8192主向量；8197 tail；`VuV`用非平凡且非2次幂unit，`VuVLoop`让full与tail字段均参与结果 |
| `30..77` relation | Eq/Ne/Ge/Gt/Le/Lt × `VV/VS/VuV/VuVLoop` × value/bitpacked BOOL output × 3 input dtype | 同一truth pattern同时验证value codeword与BOOL逐bit结果；检查BOOL末字节未用bit和suffix guard |
| `78..97` logic | value Not/And/Or/Xor及公开`VV/VuV/VuVLoop` form；packed BOOL Not/And/Or/Xor及对应form | value按FP16/BF16/FP32 raw codeword做bit-exact；BOOL用交错、全0、全1和非整字节tail |
| `98..104` transcendental | Log2、Ln、Pow2、Exp、ExpLp、Sin、Cos × 3 dtype | 正常域用高精度reference记录误差；定义profile tolerance前只记`board-observed`，域外和special value独立 |
| `105..110` activation | Tanh、Sigmoid、Relu、SatRelu、LeakyRelu、Softplus × 3 dtype | 负/零/正、拐点两侧和饱和区；参数隐含行为未闭合时不升级为production支持 |
| `111..114` reduce | Sum/Avg/Max/Min × C/W/H/HW × 3 dtype | 非对称`[2,7,9,65]`，每个轴使用不同pattern；held-out覆盖C block边界、负值和tie |
| `115..120` pool | Avg/Sum/Max/Min及indexed Max/Min × 3 dtype | 非方kernel、非对称stride/pad；indexed case检查value、index编码、tie-breaking和双writeback guard |
| `121..123` unpool | scalar-index Unpool、Avg Unpool、mask/index Unpool × 3 dtype | 多个非零位置、重叠/非重叠窗口和完整destination footprint；index来源及重复index语义分别记账 |
| `124..138` DataMove | 由5.3节逐entry展开 | all-and-only logical point、完整physical span和每段padding canary |
| `139..174` convert | 36条source→destination route全部入表；有参数的route覆盖round-to-nearest-even、zero、+Inf、-Inf | 普通值、正负halfway、饱和/溢出、zero-point；stochastic rounding单独统计suite，不与确定性exact混跑 |
| `175..186` peripheral | Count、BitCount、ArgMax/Min、Memset、Factorize、Bit2Fp、Bilinear、LUT16/32、RandGen、ElemMask | deterministic entry验证全部writeback；RandGen/随机ElemMask在seed、分布和可恢复性合同闭合前为`isolated-deferred` |

算术/比较的form参数还要有专门的区分向量：`VS`的scalar bit pattern不能等于任一vector首元素；`VuV`的unit
不能整段常量；`VuVLoop`的每个outer chunk使用不同unit pattern，并让最后一个不完整chunk参与expected。
这样同一case能区分“误当VV”“只重复首unit”“忽略full字段”和“只执行第一段”。

数值域按下列顺序分suite，避免一个NaN或随机case把整个基础资格变成不可诊断：

1. 普通有限、可精确表示的正负数和非零operand；
2. rounding-sensitive halfway、长累加和cancellation；
3. `+0/-0`、最小normal、subnormal、overflow和`+Inf/-Inf`；
4. quiet/signaling NaN与payload/quieting传播；没有明确oracle时只记录raw bits，不升级支持；
5. integer min/max、饱和、zero-point和随机模式。

### 5.3 Layout、Concat、Broadcast 与 DataMove suite

instruction与layout组合先按physical行为分类，再为每个公开entry登记disposition：

| 组合类 | 必须执行的正向case | production含义 |
| --- | --- | --- |
| CT vector × Tensor/NTensor | 每个CT vector opcode/form/dtype的main与tail | native正向资格 |
| CT vector × Cx | lane-independent代表覆盖full-physical raw span；另以Tensor↔Cx显式movement只处理logical-valid points | raw结果只校准硬件；production仍需materialize或未来typed segmented consumer |
| CT vector × NCx | 至少两个N slice，slice使用不同pattern和独立padding poison；逐slice重放Cx逻辑 | 不得把NCx静默flatten成单个Cx |
| reduce/pool/unpool × Cx/NCx | 原生wrapper声明的shape/layout逐kind正向；compact Tensor版本若需转换则重放movement | 不由GEMM或CT vector结果外推 |
| GEMM × Cx/NCx | rank-2 Cx与single-leading-batch NCx分别闭合，见5.4节 | current production native路径 |
| 任意不受typed ABI支持的direct组合 | verifier/lowering/required-symbol negative | 不上板、不以raw packet绕过 |

Cx/NCx的共同shape族为`[2,7,9,65]`，边界族覆盖`C=63/64/65/127/129`；每个N slice、full block、
compact `C0` tail和256B physical padding使用不同poison。DataMove使用下列非对称case，所有FP16/BF16/FP32
可接受entry分别记账：

| DataMove能力 | calibration shape/关系 | 必须区分的错误 |
| --- | --- | --- |
| Mirror、Rotate90/180/270 | `[2,7,9,65]`，H/W位置编码不同 | H/W轴颠倒、只处理首N、padding被当逻辑值 |
| Transpose | header定义的固定映射使用非对称H/W/C shape，held-out更换三轴长度 | metadata-only、固定轴映射错误、Cx block步进错误 |
| NCHW↔NHWC | `[2,65,7,9] ↔ [2,7,9,65]` | 仅改shape未搬数据、C/H/W顺序错误 |
| raw Concat | vendor公开的C/W/H/HW维度分别使用不等长两输入；C用33+32→65，H/W使用非对称切分 | 两输入反序、轴错误、第二输入tail/padding串扰 |
| compiler concat | N轴或任意raw wrapper不能原生表达的合法concat由typed GatherScatter composite闭合 | 不能把raw Concat可用性外推成全部source语义 |
| Pad | `[2,5,7,65]`到非对称pad后的`[2,7,10,65]` | top/bottom/left/right次序、pad值、内部physical padding |
| TensorNom/channelnorm | `[2,7,9,65]`且各channel统计不同 | 归一化轴、统计范围和padding参与错误 |
| GatherScatter | contiguous、1D/2D/3D stride、holes与边界tail | byte/element单位、iteration off-by-one、inclusive end |
| compiler broadcast | scalar→full、`[65]` channel→`[2,7,9,65]`、`[9,65]` row→full | broadcast轴、stride0解释、只填首slice |
| MaskMove/MaskGather/value-to-BOOL MaskGather | 稀疏非周期mask、非整字节BOOL tail | mask/index单位、bit order、未选位置被覆盖 |
| Img2Col | `[2,9,11,65]`、非方kernel/stride与非对称pad | kernel-major顺序、X/Y轴、输出shape与padding |
| Tensor↔Cx/NCx | 同一logical payload的双向round-trip和单向physical golden | C0 tail、N步进、padding publication与逻辑值混淆 |

CT `VuV/VuVLoop`、compiler GatherScatter broadcast和Direct DTE broadcast分别属于operand form、local
materialization和multi-rank transport；三组case不能共用一个“broadcast passed”状态。Direct DTE broadcast
仍要求receiver-first、participant、channel/FSM、terminal和每个destination独立guard。

### 5.4 NE FP16/BF16 compute suite

NE先把BF16当独立numeric profile，而不是“FP16换format字段”。GEMM基础矩阵如下：

| 维度 | calibration | held-out/边界 |
| --- | --- | --- |
| shape | `M=64,K=128,N=128` | `M=65,K=129,N=129`，M/K/N tail同时出现 |
| orientation | NN、NT、TN、TT逐项使用非对称operand | 第二组非方M/K/N，防止转置flag错误被方阵掩盖 |
| batch/layout | rank-2 Cx；batch2 NCx且两个batch pattern不同 | 左右batch broadcast/不等batch只在typed ABI明确后进入正向 |
| dtype | FP16→FP16、BF16→BF16；其它input/output pair逐项登记 | 不由input BF16推导accumulator或output rounding |
| accumulation | 每个output含至少128个非零贡献 | cancellation、round-up/down区分、K=129尾项非零 |
| options | bias、psum、ReLU/LeakyReLU、axis scale各做one-factor-at-a-time | quant、sparse和组合option在单项语义闭合后才组合 |

BF16专门建立普通精确值、长累加、正负cancellation、round-up/down、signed zero、normal/subnormal、
overflow/Inf和NaN传播row。每个row同时检查logical result、Cx/NCx physical span、padding和guard；只要
accumulator宽度、flush-to-zero或NaN传播仍有多个解释，就保持`board-observed/unknown`而不写成通用BF16
语义。

Conv、DepthwiseConv和BackwardConv分别建inventory，不能由ordinary Conv外推。ordinary Conv主shape采用
`N=2,H=17,W=19,I=65,O=96`、非方`Kx=3,Ky=2`、非对称stride/pad/dilation，held-out把O改为65；
Depthwise按其独立channel relation选择同级别非64整除shape，不复用ordinary Conv的O。每种kind先闭合bare
FP16/BF16，再对bias、psum、ReLU/LeakyReLU、axis scale、quant、sparse、pad/unpad做
one-factor-at-a-time。header可配置但typed compiler尚无consumer的option进入`isolated-deferred`；
shape relation或wrapper根本不接受的组合进入`static-negative`。

### 5.5 Shape、资源与oracle基线

| suite | calibration shape | held-out | 目的 |
| --- | --- | --- | --- |
| CT compact vector | 8192 logical elements | 8197 elements | 排除启动开销主导并覆盖非256B tail |
| CT `VuV` | 8192与非2次幂unit | 8197与另一legal unit | 区分unit重复、full/tail |
| Cx/NCx | `[2,7,9,65]` | C=63/64/127/129族 | 覆盖block、C0 tail、N stride |
| reduce/pool/movement | 至少数千元素、非对称H/W/C | 独立axis/window/pad | 排除1x1、方阵和全零弱oracle |
| GEMM | 64×128×128 | 65×129×129、batch2 | 非平凡累加和三维tail |
| Conv | N2、17×19、3×2 kernel、I65/O96 | 独立stride/dilation、O65 | 轴、channel tail和option |
| RDMA/WDMA/TDMA | 64KiB强sentinel加1D/2D/3D holes | 独立length/offset/tail | descriptor单位、range和cache |

任何具体shape在进入board-positive前都必须由SPM/DDR planner证明输入、全部writeback、padding和前后guard不
越过当前可分配范围。shape是诊断向量，不成为compiler支持上限；更大shape由分块后的相同typed合同覆盖。

### 5.6 Execution、completion 与 transport case

instruction qualification之外，Q37其余硬件边界继续按下列case维护；已闭合项只在profile签名变化或需要
held-out时重跑：

1. **TDMA held-out**：独立value/length复验I8/F16/BF16 descriptor；native `Fmt_BOOL`保持excluded，
   production BOOL physical-footprint只测试I8 byte-fill canonicalization。
2. **single-engine/worker completion**：CT/NE/RDMA/WDMA/TDMA低深度1/2/4，TDMA只到2；worker1/2只做
   routing与matching wait。
3. **documented depth与tight `D+1`（已闭合）**：保留单engine/case/sample及前后heartbeat；不继续更深，
   不把总提交完成解释成occupancy/full/backpressure。
4. **cross-engine disjoint/dependency**：10个pair当前保持串行；只有同方向serial/window稳定正overlap后，
   才执行对应RAW/WAR/WAW/RAR exact/partial/adjacent composition。
5. **SPM offset**：固定workload只改变相对offset，形成经验conflict class，不提前命名物理bank。
6. **worker/join/visibility**：补default wait跨worker区分向量；同地址只跑显式ordered正向。
7. **DTE/SPM PMU与multi-tile**：补counter basis和安全subgroup正向；错误坐标、缺participant和资源提前
   复用只做negative，不上板。

### 5.7 下一轮实卡执行顺序

实现阶段按suite生成共享catalog/oracle和共享device dispatcher；同一ABI/resource class只编译、链接和打包
一次，由typed request选择case，不为每个row重复编译。所有static-negative、host oracle、device compile/link
和no-card package在进入实卡环境前一次性闭合，因此实卡时间主要用于execution/readback，不随case数线性重复
编译。

本地CT/NE/DataMove、layout和descriptor资格默认只在一个known-good tile的worker0运行；worker routing只用
代表case覆盖worker1/2。只有Direct DTE、multi-tile arrival、collective或明确rank-dependent语义才启动多rank，
不能为了“每rank workload相同”把同一个local instruction oracle在所有rank重复16遍。

实卡按下面批次串行执行。每个board-positive row独立launch和外层timeout；safe deterministic suite在批次
前后heartbeat，高风险manual row逐case前后heartbeat。任一case或heartbeat异常立即停止，不自动
重试/reset/power：

1. profile identity、只读状态和known-good Add heartbeat；
2. CT Tensor普通有限值：unary、binary四种form，再做FP16/BF16/FP32 tail；
3. relation value/BOOL与logic value/BOOL，优先闭合bit order、write span和guard；
4. Tensor↔Cx/NCx、Concat、GatherScatter broadcast及其它deterministic DataMove；
5. NE FP16/BF16 GEMM main/tail/orientation/batch，再做Conv/Depthwise和one-factor options；
6. reduce、pool/unpool、convert确定性rounding和deterministic peripheral；
7. transcendental/activation的普通域与误差观察；
8. zero/subnormal/Inf/NaN、multi-writeback、随机或其它`isolated-deferred`，每项单独manual gate；
9. 尚未闭合的SPM、visibility、PMU和安全multi-tile case。

一批case只有在完整expected、physical span、padding/guard、terminal、issue count和cleanup全部通过时才记
`board-observed`；calibration与独立held-out均通过后才可记`calibrated`；production source-to-package纵向
再通过后才可记`supported`。原始日志按profile和case row关联保存，不把一次板端输出直接改写成跨profile
compiler常数。

## 6. 明确禁测或保守处理

- `depth + 1`不得混入普通calibration；它只允许进入typed tight manual gate：单engine、单case、单样本，
  packet builder预先释放，相邻execute之间不插入register MMIO，并要求前后known-good Add heartbeat和完整
  count/output/guard均通过。除这个精确`D+1`向量外，任意更深或未证明occupancy/full的overflow仍不执行。
- 不发射SCALAR、DTE或CSR raw packet到`TsmExecute`。
- 当前profile不发射TDMA Memset native `Fmt_BOOL` packet；BOOL full-footprint只允许先canonicalize为I8
  byte fill，logical-valid BOOL fill保持fail closed。
- 不使用大于2的worker id；底层`% 3`会静默别名，production必须先拒绝。
- 不执行跨worker同地址且没有producer completion的case。
- 不执行DTE receiver未准备、错误tile坐标/FSM/channel、缺participant或资源提前复用。
- 不把历史Atomic Barrier、firmware替换型instruction profiler或power-cycle脚本作为普通probe。
- 不把aggregate PMU升级为per-cycle trace或cycle-accurate model；没有measurement basis的counter保持Unknown。

## 7. Compiler消费规则

- semantic legality只能消费稳定instruction/layout/numeric结果，不能由性能probe反推。
- memory planning从当前IR的SSA、view/range/effect和completion重算；hardware busytable不是lifetime proof。
- scheduling只移动dependency DAG中ready的指令。当前profile没有pair通过稳定正overlap资格门禁，所有
  engine pair保守串行。某engine pair可重叠不等于alias case可重排，当前workload的零overlap也不等于
  硬件永久不支持并行；未来只有同方向对照稳定达到median正overlap才允许重新校准对应hazard和并行候选。
- bank/latency/bandwidth在未校准时保持Unknown；支持时也只排序已经通过exact legality的candidate。
- local drain、cross-worker join、DTE completion、multi-tile arrival和host publication保持不同typed边界。
- wait只在消费结果、复用地址或跨visibility域前的latest-legal位置materialize；同一worker相邻wait应合并，
  非default worker使用matching `bywork`。当前没有证据允许把default/local fence当作跨worker join。
- `wafer.instr.fill`的BOOL count仍按typed physical bit domain解释；native packet exclusion只影响TX81
  target/CRT mapping，不能把上层BOOL改写成I8语义。I8 byte fill仅对完整`physical_footprint`成立。
- profile-specific数据进入target capability/calibration consumer，不进入上层StableHLO、Shardy或数学IR。
- probe源码、case名、sample count和raw packet不是production协议；通用实现只读取当前IR和typed target profile。
