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
  风险及保守compiler处理；校准vector与held-out vector分离，正确性oracle先于性能解释。已有保守fallback
  只关闭对应legality/correctness风险，不等价于硬件行为完备；新增可区分case及其pending/board结果持续在
  本文同一row演进，不另建最终结果文档。
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
6. 对全部有界descriptor执行exact/partial/adjacent/strided dependency correctness观察；disjoint正overlap
   只决定能否取得compiler并行资格，不决定安全case是否发板；
7. matching worker/local completion、cross-worker join、DTE或multi-tile completion；
8. 完整readback、canary、terminal status和资源cleanup。

timeout、设备无响应或状态异常会立即停止该批次；测试不自动重试，不调用NPU power、reset或firmware替换。
管理面显示idle、0%利用率或无进程只说明inventory/accounting表面状态，不能证明execution/completion面仍健康。
后续queue边界probe使用独立进程、单case；同一software/runtime身份的合格会话只在总runner首尾各运行一次
known-good Add heartbeat，不为每个case重复资格化。任一heartbeat或case timeout都停止全部板测，由用户决定恢复方式。

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
DTE分别保留自己的descriptor/transport矩阵。catalog必须对每个公开entry给出以下四种 disposition 之一，
不能因没有实现case而从清单消失：

- `board-positive`：已具备固定输入、完整expected、physical write span和guard，可进入隔离板端suite；
- `board-observation`：可以安全执行并检查完整span、guard、completion和request echo，但NaN quieting、
  隐式参数、饱和或其它profile行为尚无独立expected；只保存raw result，不取得semantic资格；
- `static-negative`：该dtype/layout/form在当前typed compiler/ABI中明确非法，由verifier、lowering或
  required-symbol gate拒绝，不发射raw packet；
- `isolated-deferred`：owned ABI虽能到达该路径，但当前请求可能永久阻塞、破坏不受控range，或缺少能够
  在同一runtime session中保证timeout、safety drain和cleanup的生命周期owner；必须同时写明最接近的
  安全替代probe和恢复缺口，不发板。

缺少exact numeric oracle本身不再构成`isolated-deferred`理由。只要descriptor/write span有界、请求能在
外层timeout内进入matching completion或safety drain、资源能正常cleanup，就构造成`board-observation`：
随机指令重复采样并保留每次raw output，未知layout/饱和/量化行为保留完整physical span、padding、guard、
request echo、execute返回和PMU。owned typed ABI根本没有对应setter/field的组合属于`static-negative`，
不能用相近family的raw packet冒充。

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

current board-positive catalog为普通`VuV`选择
`1 <= unit_elem_count <= 64`、`elem_count % unit_elem_count == 0`且不写`full_*`的完整-unit矩形。
`VuVLoop`的supported interface legality则固定为
`unit_elem_count == 64`且
`full_elem_count * unit_elem_count == elem_count * full_unit_elem_count`；host以扩宽或checked
multiplication验证该关系，不能用截断除法代替。任一关系不满足时只构造host negative，不生成raw板端
packet。既有`unit_elem_count=32/37`逐bit exact结果只记为out-of-contract hardware observation，
不授权compiler、runtime或production扩大准入。合同内main/held-out可使用不同geometry，但catalog、
device decode、packet、payload、record echo和host oracle必须从同一shape geometry派生。

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

- CT Tensor主向量使用至少8192个logical element，并另设非256B整除的held-out tail；`VuV`当前正向
  使用不超过64的非平凡unit且只覆盖完整unit，`VuVLoop` board-positive固定使用64-element unit并满足
  checked base/full乘积关系，同时让各outer unit pattern可区分；违反任一合同关系的geometry只进host
  negative，不再作为板端qualification；
- Cx/NCx至少覆盖`C=63/64/65/127/129`中的block边界和`N>1, H*W>1`，每个physical padding区保持独立
  canary；
- NE FP16/BF16至少覆盖`M=64,K=128,N=128`的非平凡累加、`M=65,K=129,N=129`的M/K/N tail、batch2及
  NN/NT/TN/TT中合法组合；每个case分别记录logical output与physical Cx/NCx span；
- transform/layout movement使用至少数千元素并采用非对称维度，防止1x1、方阵或全零输入掩盖轴顺序。

完整矩阵按opcode/form、dtype、layout/shape-class分suite串行执行，不要求一次进程跑完全部case。每个
board-positive row仍是独立launch和外层timeout；安全deterministic suite在批次前后做known-good heartbeat，
queue边界、安全random observation和multi-writeback等高风险board row才逐case做前后heartbeat。所有
可安全构造的row进入实卡suite，`isolated-deferred`只保留上述不可恢复/无生命周期owner的隔离输入；
no-card preflight只闭合
catalog、oracle、compiler verifier、device object/link和package协议，不能把row状态升级为
`board-observed`或`calibrated`。当前28个导航域下面的叶子均由具体catalog对象或带理由的非执行对象解析，
最终叶子总数与处置计数以`wafer_hardware_calibration_matrix.py`及其机器审计结果为准，不在本文复制一份
易失汇总。所有board-positive/board-observation进入对应实卡批次，delegated项解析到明确的concrete
board case；negative/deferred只表示非执行处置已经准备，不因门禁通过而改变production capability。
catalog分组均被manifest记账；允许复用的共享case使用显式白名单，其余分组只能被引用一次。

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
measurement basis按counter分域记账。四种基础有序DTE/NCC模式、event后安全复用和两destination
broadcast共六种模式分别展开成16B calibration及32B/64B/256B/4KiB held-out，共30个full-card case；
每rank的264448B host resource包含四个66048B guarded slot，设备以NCC RDMA seed SPM、在matching
completion后以NCC WDMA回读完整payload/非active后缀和前后256B guard，host再检查四个slot及剩余
canary，从而同时区分传输长度错误、越界写和错误completion-domain可见性。六个已解码
DTE/SPM split counter逐mode、逐payload保留16-rank raw delta并自动
报告median趋势，不预设byte/cycle/event单位。current version-matched header对TMNOC只公开两个base address，
没有只读counter offset或measurement contract；该项因此是带证据的`static-negative`，禁止猜offset读寄存器。
同一full-card package另构造四种有界错误观察：source在首个send event前重复prepare、destination在首个
receive event前重复prepare、invalid send/recv FSM、wait未知event。CRT均同步返回错误，device先把错误写入
shadow status，再完成合法生命周期或空生命周期并用真正runtime status正常SUCCESS收尾，避免把预期错误误当成
BoardRuntime quarantine。receiver未prepare会进入没有device timeout的`direct_sync_wait`。独立mode 13
已经用3s host completion deadline实测：cluster main未完成，runtime报告`context=poisoned`，外层8s终止；
全程没有retry、reset或power操作。因此该路径取得“会阻塞且污染当前execution context”的直接证据，
production继续fail closed。

2026-07-24首次执行扩展DTE/NCC批次时，cluster main在completion内timeout；保存产物只包含
264448B host resource carrier package以及production baseline的input/expected raw file，没有
`probe-build`或任何`probe-raw`目录。按host driver顺序，这证明timeout发生在probe module替换和任一
DTE/NCC catalog case之前：当时carrier的`RESOURCE_BYTES / sizeof(f32)`被直接用作production
Direct DTE baseline的local shape，使原本只应资格确认的baseline膨胀为每rank 66112个`f32`元素的
full-card collective。该事故不能归因于30个PMU case、有界错误观察、raw async sender或新增
RDMA/WDMA readback，也不形成这些路径已经上板的证据；它只证明这次把carrier容量耦合进baseline workload的
cluster main没有在外层timeout内发布terminal。

最小修复已经把两个职责拆开：production资格确认单独编译并执行固定64元素baseline；264448B carrier只用于
提供probe所需host resource，随后替换为probe module，不再执行carrier原始giant collective。修复后的独立
64元素production baseline已在16 ranks取得exact结果。随后modes 1--6的
`16/32/64/256/4096B`共30个full-card case全部通过完整expected、status、SPM readback guard和正常cleanup。
DTE channel0 transfer raw median在每个mode内均随payload严格递增，modes 1--4呈scale 1、
modes 5--6呈scale 2；该结果只提供profile内相对传输量相关性，不命名counter绝对单位。execution raw
不随payload单调，不能进入latency/cost；SPM PMU没有enable，因此SPM counter样本为`inconclusive`。
modes 7、8、12的16B有界错误观察也已逐项完成16-rank执行：预期TRANSPORT_ERROR被shadow status观察到，
随后runtime以clean SUCCESS收尾，payload/guard exact且正常cleanup。该证据只闭合source提前复用、
destination提前复用和invalid FSM三条同步拒绝路径。mode 9 unknown-event wait同样在16 ranks观察到
预期TRANSPORT_ERROR，随后clean SUCCESS、payload/guard exact并正常cleanup。modes 10--11以64KiB
transport各执行3 samples × 16 ranks：serial control和在`send_async`/`wait_done`之间插入CT的window均
exact，`send_async`、`wait_done`、`release`返回码全0，guard exact且正常cleanup。至此modes 1--12均已
通过各自board oracle；async window只证明该调用序列和结果正确，不单凭正确性宣称DTE与CT发生时间重叠。

## 3. Compiler-sensitive calibration matrix

下表是Q37校准入口。每个row最终必须变为`calibrated/supported`，或保留带明确保守行为的
`unknown/excluded`。不要求所有字段笛卡尔积，但每个语义分支必须有能区分错误实现的vector。

| 域 | 待闭合事实 | 最小probe与oracle | compiler consumer | 当前状态 |
| --- | --- | --- | --- | --- |
| profile qualification | runtime/instruction/CRT身份、tile map、PMU、3个worker `serial_mode` | 一次只读identity/CSR，禁止写power/reset | 14-17 target/runtime profile | 当前profile `board-observed` |
| constructor ownership | `TsmNewArith`返回由vendor一参`rt_malloc/rt_free`拥有的host method-table allocation；空指针是allocation failure，packet materialize后builder可以释放 | 最终module保留exact `rt_malloc/rt_free` UND；记录constructor返回`uintptr_t`和nonnull位，立即delete builder后仍执行exact packet/count/result/guard | 14 CRT lowering、device-link与probe infrastructure | `board-observed`：loader直接解析system heap ABI后constructor非空，builder释放后的CT packet仍完成且count/result/guard正确；不得由CRT猜scope并改写为`csi_kernel_malloc/free` |
| execute result | `TsmExecute`成功和invalid type路径都可返回1，raw rc不能区分成功 | packet legality、目标queue count和结果共同判定 | 14/15 structured error boundary | `calibrated`；rc只记录不判成功 |
| packet routing/range | `inter_type`映射、worker编码、begin/end materialization和inclusive end | 五类单engine，执行后读实际register/packet，完整结果 | 11/14 instruction legality | CT/NE/RDMA/WDMA部分`board-observed`；TDMA Memset routing/range `board-observed` |
| CT numeric/form | opcode 0..186；arithmetic/relation/logic的`VV/VS/VuV/VuVLoop`、value/bool output；f16/bf16/f32、convert pair、rounding、NaN/Inf/subnormal/signed-zero、large-shape tail | opcode/form×dtype typed catalog；至少8192元素普通值、非256B held-out tail、短向量unit/full-tail、完整bit oracle；`VuVLoop`先过supported interface host legality | 10/11/17 numeric capability | convert 204/204已按当前catalog完成板端gate。instruction-family保留160个safe case，raw N/HWC Reduce八项因隔离timeout改为fail-closed `isolated-deferred`。Pool当前16个exact与8个bounded observation均有板端结论；Unpool按实际numeric行为分exact与observation。本轮CT vector为op010 tail与op011--013共23项保存了板端结果，但其中`VuVLoop unit=32/37` exact只算out-of-contract hardware observation。合同内`unit=64`的两个区分geometry已在clean session通过exact result、guard和completion，前后ordinary Add也通过。违反`unit==64`或base/full乘积关系的packet一律host negative，不再上板 |
| instruction × physical layout | Tensor/NTensor/Cx/NCx原生资格、materialization路径、C0 tail、N-slice步进、padding lane | valid组合exact output/span/canary；需materialize组合重放Tensor↔Cx/NCx movement；非法组合verifier/packet negative | 08/10/11 physical legality | CT/NE/RDMA/WDMA/TDMA × Tensor/NTensor/Cx/NCx共20个组合均已有处置：8个native positive、3个materialize-then-consume concrete positive和9个static-negative；Tensor↔Cx/NCx movement及CT Cx/NCx、NE Tensor materialize→consume composite均已完成no-card准备。current CT production仍只准Tensor/no-invalid-lane |
| DataMove/layout | mirror/transpose/rotate、NCHW/NHWC、concat、pad、Img2Col、TensorNom、GatherScatter/MaskMove/MaskGather及compiler broadcast materialization | 非对称large-shape、axis-sensitive payload、all-and-only logical point oracle、physical guard | 08/10/11 movement legality | base与extended安全路径已有板端结论。large Pad/Img2Col及TensorNom证明NCx footprint不能由compact logical bytes或单次末尾对齐代替：每个N slice都服从aligned-C与batch stride，internal/batch padding和外部guard分别记账。raw Concat C/W/H只有bounded completion/guard；native `dims=HW`是错误指令用法，永久`static-negative`且测试面不得发射。合法source-level HW concat继续走typed GatherScatter |
| NE numeric/layout | f16/bf16、accumulation、transpose、C0 tail、padding、K/M/N边界 | 非平凡多项累加GEMM、完整padded range/canary | 08/10/11/17 | 当前73个row为32 exact、38 safe observation、3 static-negative、0 deferred；GEMM PSUM和左右不等batch保持exact。FP16 ReLU观察为no-op，ordinary Conv的current NCx/HWOI oracle未闭合，二者均不再宣称exact。修正schema的BackwardConv FP16/BF16各3个板端样本均按8192B footprint完成，request/result echo、完整physical span、suffix guard和completion通过；它取得bounded observation，但不升级尚无唯一numeric oracle的通用BackwardConv exact资格。GEMM sparse因typed `TsmGemm`无setter归入static-negative |
| RDMA/WDMA descriptor | byte/logical-element stride转换、iteration、inclusive range、tail；DDR descriptor与local SPM footprint分域 | contiguous + 1/2/3D stride，非零round-trip、DDR holes、compact SPM和guard | 08/11/14 | contiguous与既有large GEMM `supported`。修正oracle后的3个standalone DMA strict compact-SPM case及36个NCC strided dependency case全部通过板端result/composition、DDR holes、SPM/output guard、count和completion；确认stride只作用DDR endpoint，local SPM endpoint连续使用`transfer_bytes`。旧mismatch来自错误oracle，不是mapped-SPM cache问题 |
| TDMA Memset | element count、byte stride、raw logical iteration、inclusive range和dtype packet encoding | whole/128B×32/64B×64 geometry，I8/F16/BF16 raw与CRT，全range和guard | 10/11/14 | 普通dtype descriptor `calibrated`；I8/F16/BF16 vectors `board-observed` |
| TDMA BOOL fill | native `Fmt_BOOL` completion与bitpacked physical-footprint实现 | native小range timeout隔离；production BOOL→I8 byte fill需独立raw register、全range和guard | 10/11/14 | native `Fmt_BOOL`在当前profile `excluded`；直接I8 physical16与BOOL→I8的136 logical bits→17 physical bytes held-out均完整readback/guard通过，替代路径在完整physical-footprint域`board-observed` |
| TDMA movement variants | GatherScatter和其它DataMove的byte count、stride/iteration、range与kind-specific geometry | 每个已准入kind使用能区分错误descriptor的非零pattern、全range和guard | 08/10/11/14 | base 46与extended默认17项均按各自exact/observation合同取得板端结论；large Pad/Img2Col、TensorNom使用修正后的NCx physical footprint。native `dims=HW` Concat永久`static-negative`，不得生成packet、board CTest、runner step或其它执行入口；raw C/W/H、MaskGather等仍只消费bounded observation，不由完成性外推语义 |
| SPM capacity/reservation | allocatable range和保留区 | boundary-positive与verifier negative；不触碰保留区 | 09/11 | 4个boundary/held-out positive均已板端exact，count、result与guard通过；6个越界/保留区项保持static-negative |
| SPM alignment/bank | 256B preferred alignment、非1024-bit访问资格/代价、bank/color映射 | bounded非preferred geometry round-trip、disjoint offset sweep、多offset engine-pair serial/window control | 09 placement与06 cost | 当前SPM inventory有146行：原19个本地case及新增5个`base+64/+128/+192`、`length=128/384`均已有板端证据。用于性能区分的16个sustained cell已组成4个同invocation matched board group；probe保存并设置owner-backed PMU enable，按high-low-high读取port0/6 T2/T3，执行matched WDMA readback，并在partial accept失败时先做matching-worker safety drain。完整SPM/canary、count、address、PMU、readback和lifecycle oracle当前仅为`board-ready/pending-board`。physical bank/color identity仍没有owner-backed bank/port mapping或bank-specific attribution；offset、port counter或aggregate execution方向都不得代签bank编号 |
| DDR/cache/coherence | host H2D、Kcore cache、DMA completion和host publication是不同域 | Kcore read前invalidate对照、DMA round-trip、matching drain后D2H | 09/12/14/15 | memory-descriptor 155行、独立cache 58行、16-rank tile-offset与40GiB sparse probe已有板端证据。新增DDR active-rank contention为RDMA-only、WDMA-only、双向三类 × 4KiB/64KiB/64KiB-stride三种workload × active rank 1/2/4/8/16，共45个真实case、9个五点matched board group；完整active/inactive rank oracle和16-rank lifecycle已接入runner，当前`board-ready/pending-board`，不形成bank/controller结论 |
| queue shape与连续提交边界 | register/header中CT/NE/RDMA/WDMA静态depth为6、TDMA为4；该数值描述pending storage，不是完整lifetime总提交上限，active occupancy和full行为不能由形状或总提交数推出 | 普通calibration只跑1/2/4（TDMA 1/2）；隔离manual gate验证`D`、tight `D+1`和较长workload观察点，完整检查count/output/guard | 10/11/16 | 五类engine的`D`与tight `D+1`已有板端证据；30个`short/sustained × D-1/D/D+1`区分case已成为逐case CTest和runner步骤，当前`board-ready/pending-board`，用于闭合full/backpressure response。当前versioned record没有head/tail/occupancy/resident字段，因此resident数量由`queue-active-resident-count-unobservable` host gate typed fail closed，不能由task-done、静态depth或完成总数代签 |
| worker scope | worker0/1/2 routing、default wait和`bywork` scope | CT三worker；matching wait后、safety drain前读CSR/result与completion marker | 10/11/14/15 | 既有matching `bywork(1)`和自然排空对照仍只形成原窄结论。18个wait-scope、12个subset-scope逐case实验，以及44个characterization case组成的14个matched group均已接入真实device protocol、CTest和runner：fixed-total placement 14、bounded progress 18、逐worker low/high outstanding 12。强oracle固定同一physical sentinel slot，在safety drain前保存observer boundary并将target pending与raw `CONTROL`交叉核对，同时检查whole-output tail canary、join返回码/最终idle和device deadline；partial issue或observation timeout按全部attempted worker只做一次bounded cleanup，失败即poison停批且不retry。当前均为`board-ready/pending-board`。absolute per-worker completion timestamp与physical arbiter policy没有owner-backed观测面，分别由typed host boundary拒绝；跨worker仍必须显式join |
| cross-engine overlap | 五类engine全部10个pair的可重叠性和共享资源 | disjoint sustained window、serial control、FU union、三次重复；dependency pair另保留两种issue order | 06/10/11/16 cost/scheduling | expanded pair关闭了“所有pair只能串行”的旧结论：16KiB同worker CT+NE/CT+RDMA/CT+WDMA/CT+TDMA/NE+RDMA/NE+WDMA/NE+TDMA、cross-worker CT+RDMA及64KiB RDMA+WDMA/RDMA+TDMA/WDMA+TDMA均为三次重复正`Ea+Eb-FU`且window plan更低，可进入窄并行capability。4KiB RDMA+WDMA无论disjoint/exact/half-partial及A→B/B→A都为excess=0，只允许减少issue/wait开销，不计engine overlap收益 |
| address dependency | busytable对RAW/WAR/WAW/RAR及exact/partial/adjacent/stride envelope的处理 | 对全部有界descriptor做composition/guard/completion；PMU正overlap只控制并行资格 | 09-11 legality/scheduling | contiguous RAW/WAR/WAW/RAR × exact/partial/adjacent的serial/window共24个case、72个样本均通过composition/count/guard；pure same-worker NCC链在RAW/WAR/WAW之间不插wait，按IR edge保持issue order并由busytable落实。修正oracle后的3个standalone DMA及36个NCC strided case也全部通过：1D/2D/3D RAW/WAR/RAR serial/window与WAW exact/partial/adjacent serial/window均完成composition/count/guard。该结果支持DDR-strided/compact-SPM分域descriptor和显式issue order，不授权删除IR依赖或宣称overlap |
| issue overhead | wrapper构包/heap间隔对短window的影响，prepared issue是否值得materialize | 同packet序列wrapper与prebuilt对照 | 06/14 candidate/lowering | 旧RDMA+CT差异为`historical/inconclusive`，本轮未复现稳定正overlap，不构成新IR语义或收益结论 |
| local completion | default wait、`bywork`、local fence的范围和visibility | completion-domain boundary返回后立即CSR/Kcore oracle，再做safety drain | 09-11/15 | wait-overhead 5类engine的wait-each/wait-once、raw/wrapper issue各10项均通过；RDMA→CT、CT/NE→WDMA、TDMA→CT/NE的pure same-worker NCC链由issue order+busytable完成，不要求中间wait。CT/NE→Kcore需要在离开NCC completion domain前matching drain；NCC→Direct DTE、跨worker join、barrier与terminal/host publication同样是显式boundary。default/local-fence跨worker scope仍`unknown` |
| cross-worker join | 多worker并行、仲裁与地址依赖是否跨worker | disjoint w0/w1/w2，逐workerjoin与六个proper-subset mask；同地址不做无序正向 | 10/11/15 | full join与六个proper-subset mask均保证mask内worker在boundary完成；但六例中mask外worker也都已自然排空，未观察到unjoined pending，故只闭合included-worker正向，不证明subset join会等待或排除mask外worker。并行性、仲裁与同地址无序行为仍`unknown/excluded` |
| Direct DTE | source read、destination visibility、participant、channel/FSM、terminal status | 16-rank receiver-first；producer→DTE、DTE→consumer、disjoint顺序、event后复用、multidestination mode及同步错误返回 | 13-16 | 修复后的独立64元素production baseline已在16 ranks exact。modes 1--6的16/32/64/256/4096B共30个full-card case全部通过expected、status、guarded SPM readback和cleanup。modes 7、8、9、12的16B同步错误观察均为16-rank exact。modes 10--11的64KiB raw async serial/window各3 samples × 16 ranks exact。mode 13 receiver-unprepared在3s host deadline内未完成并使runtime context进入`poisoned`。新增25个独立board-pending case覆盖四源fan-in，以及raw broadcast/scatter/shuffle × fanout 2/4/8/15 × adjacent/interleaved；`dest_num`编码和destination mapping只作板端观察。8/15源fan-in、device phase、physical route和跨卡transport仍fail closed；sender时间重叠仍`unknown` |
| multi-tile arrival | full-card/subgroup barrier的participant与复用合同 | production 16-rank正向；缺participant/错误坐标不测试 | 13/15 | 两轮反向错峰的16-rank `hrt_barrier`均16/16 marker正确、0 mismatch/crosstalk，full-card复用`board-observed`；version-matched实现固定观察16个slot，故1/2/4/8/15 subgroup已落成compile/submission前typed-negative，不能作为正向发包；未来只有独立participant-aware primitive才能新增subgroup positive |
| host launch/runtime | kernel/model launch、resource staging/readback、timeout、failure cleanup | 同package schema、exact output、terminal/cleanup | 14-16 | 已有kernel/model与16-rank路径`board-observed/supported`，按owner证据解释 |
| NCC PMU basis | instruction count、engine exec、global union、worker scope、wrap稳定读取 | 单engine等式、pair union、high-low-high和重复样本 | 16与后续Q9 | worker0 engine/union `calibrated` |
| DTE/SPM/TMNOC PMU | counter scope、unit和与workload相关性 | 独立单域workload和held-out payload sweep | 16与后续Q9 | modes 1--6 × 16/32/64/256/4096B的30个full-card case已有板端raw。DTE channel0 transfer median在每个mode内随payload严格递增，modes 1--4为相对scale 1、modes 5--6为scale 2；只可用于当前profile的相对传输量相关性，不命名绝对单位。execution raw非单调，不能用于latency/cost；这30个既有DTE样本没有启用SPM PMU，故其中SPM数据`inconclusive`，不代表SPM port counter不存在。新增SPM sustained probe已接入owner-backed port0/6 T2/T3读取但尚未上板。TMNOC因只有base、无decoded只读offset而`static-negative` |
| SCALAR/CSR ordinary issue | 是否属于`TsmExecute` typed queue | 静态negative，不发送未知packet | 11/14 | 当前ABI `excluded` |

## 4. 当前profile已经闭合的事实

本节记录当前已取得的事实，不把精确cycle提升成跨profile硬件常数。

### 4.0 行为模型、编译器结论与未决边界

单个case通过只说明该请求在给定输入、地址、布局和completion路径下完成。硬件校准的交付物不是通过数，
而是能够排除哪些错误行为、形成什么profile-scoped机制判断，以及编译器因此可以或不可以做什么。本节聚合
后续各小节的原始证据；细节case仍留在4.1--4.4，避免把一次输出直接升级成跨shape、跨dtype或跨profile合同。

当前总判断是：TX81已经具备支撑保守正确lowering所需的基本执行、显式completion、DDR cache publication、
SPM容量和主要movement路径；expanded sustained矩阵还形成了十个profile/shape/schedule scoped正overlap
cell，足以让candidate selection有条件生成并行窗口。queue response、SPM conflict proxy、worker placement
及其它未匹配shape/pair已有真实board-ready区分case但尚未上板；active resident数量、physical SPM/DDR
bank identity、absolute per-worker completion timestamp等缺少ABI观测面的事实则由typed host gate明确拒绝。
因此Checkpoint A允许实现显式依赖、窄capability驱动的
software-pipeline候选，不能把“queue可提交”“`serial_mode=0`”或任意case完成泛化成全局并行收益。

先把“是否已经板测”和“compiler是否据此开放能力”分开写清楚：

| 证据/能力 | 板端事实（YES/NO） | 当前compiler决策（YES/NO） |
| --- | --- | --- |
| 合同内`VuVLoop unit=64` | **YES。** 两个区分geometry均通过exact result、guard和completion，前后ordinary Add也通过，未污染execution会话 | **YES，窄合同开放。** 只接受`unit_elem_count==64`且checked base/full乘积关系成立的已验证form/dtype/shape-class；合同外geometry仍由host legality拒绝 |
| BackwardConv修正footprint | **YES。** FP16、BF16各3个样本按8192B footprint完成，request/result echo、physical span、suffix guard和completion通过 | **YES，允许bounded observation；NO，不宣称通用numeric exact。** planner按weight-owned shape保护8192B，未有唯一numeric oracle的组合不进入exact白名单 |
| SPM非preferred geometry | **YES。** `base+64/+128/+192`及`length=128/384`共5个case均完成round-trip、guard和completion | **YES，允许这5种bounded geometry；NO，不把256B删除为preferred alignment，也不外推任意byte alignment** |
| memory-descriptor与cache矩阵 | **板端执行YES，current-schema全量闭合。** memory 155行已在current 2MiB schema下全部通过result、guard、count、completion和current validator；独立cache 58行有自己的板端证据 | **YES，按155行各自已验证的descriptor/range/schedule消费资格。** 该结果仍不授权bank coloring、固定cost或未匹配overlap |
| expanded pair | **YES，correctness与窄并行资格。** 34项全部通过；10组disjoint sustained window三次重复正FU excess且plan更低，6组4KiB RDMA+WDMA dependency/control excess恒为0但plan更低 | **YES，按profile+pair+worker+transfer+relation白名单选择window。** 正excess组可计入并行收益；4KiB RDMA+WDMA只计issue/wait开销收益，不能计engine overlap |
| NCC→Kcore depth-4 A/B | **YES，两个case均完成。** no-local-wait与local-wait都得到exact boundary marker/result/guard；两次snapshot前NE均已自然完成，`boundary_control_distinguishing=false` | **NO，不能据此判定local wait必要性或scope。** 保留NCC→Kcore completion-domain出口的matching completion合同 |
| strided descriptor/dependency | **YES。** 3个standalone DMA strict case与36个NCC dependency case全部通过修正oracle后的composition/result/guard/count/completion | **YES，开放已验证1D/2D/3D DDR-strided/compact-SPM descriptor与issue-order correctness；NO，不由完成性推导重排或overlap** |
| DDR tile-offset | **YES。** 16 rank的actual allocation base × 全部已列相对offset × RDMA/WDMA矩阵完成round-trip、guard和raw timing记录 | **YES，允许这些compiler-managed allocation内的相对offset；NO，不命名physical bank/controller/hop** |
| DDR sparse high-offset | **YES。** 单个40GiB workspace的`0/4/16/32/38GiB`及`end-768B`六个窗口均exact round-trip且guard通过 | **YES，允许已验证40GiB allocation内相对地址及allocation-end边界；NO，不外推64GiB、其它allocator状态或physical bank** |

算子与movement不再用“case通过”代替行为结论：

| 域 | 硬件行为 | 证据范围 | compiler决策 | 禁止外推 |
| --- | --- | --- | --- | --- |
| CT | Tensor vector对已测FP16/BF16/FP32 unary、`VV/VS/VuV` main/tail按current exact oracle完成；合同内两个`VuVLoop unit=64` geometry exact。reduce的128B logical result实际写满256B physical block。ArgMin全正普通值返回`0.5@index42`，负数输入稳定返回错误结果；tie/NaN尚无板端结论 | current catalog 653项现已全部通过result、span、guard、record和lifecycle oracle，`PASS=653, FAIL=0, MISSING=0`。旧审计的63项mismatch来自host oracle语义错误：op78--87的value logic不是对FP bit-pattern做位运算，而是先把每个数值解释为truth，执行NOT/AND/OR/XOR后写回数值0/1；op92--94的bitpacked BOOL VuV按byte-addressed RHS storage工作，logical unit 37占5 bytes、物理周期为40 bits，最后不足一个物理周期的tail对RHS按0补齐。修正后目录`board-ct-vector-corrected63-v2`中id `10468..10527,10553,10559,10565`全部`execute_result=1, mismatches=0`且process exit 0。另有convert 204-row gate、reduce/pool/peripheral focused raw及clean-session `VuVLoop` A/B。新增F16 tie/NaN各一个真实board case，每个含3个区分输入、coherent value/index分类、padding/guard和lifecycle oracle，已经接入逐case CTest与runner但尚未上板 | 已有653个row按其实际板端证据消费；`VuVLoop`只接受unit64和checked base/full关系；ArgMin仍只开放已验证全正普通值，负数拒绝；tie/NaN等待board observation，不预设winner/NaN规则 | 不由已有CT special-value或F16四域外推BF16/F32、signed zero、Inf、subnormal或其它ArgMin域；这些未列域由`argmin-unqualified-domain` typed host gate拒绝，必须先新增独立typed row |
| NE GEMM/options | FP16/BF16 GEMM对main、K/N tail、batch、NN/NT/TN/TT及左右不等batch取得bit oracle；local psum改变并产生独立writeback。bias、ReLU、LeakyReLU和正/负axis-scale在FP16/BF16 large NN三次样本中均与bare输出逐bit相同，说明current wrapper option未产生声明语义 | 当前保存raw重新校验70个NE case/146个样本：32个GEMM exact case；15个GEMM observation各3样本；所有physical padding和slot guard闭合。BF16 signed-zero/subnormal/overflow/NaN与quant raw三次hash稳定但只属observation | GEMM只开放已验证dtype/orientation/tail/batch；psum作为显式aux result规划。上述no-op option不得做fusion或消除独立算子；BF16 special/quant只保存profile behavior，不进入通用数值重写 | 不由FP16推BF16，不由稳定hash推NaN payload、FTZ、accumulator宽度或量化公式，不把local psum当跨tile collective |
| NE Conv family | ordinary Conv的feature-index与output-index fingerprint各3/3唯一匹配NCx，错误Cx候选分别有604/763 byte mismatch；weight fingerprint对Cx/NCx候选均不匹配（684/678 byte mismatch）。BackwardConv由weight shape拥有`tfr_1`，FP16/BF16各3样本确认8192B bounded footprint；Depthwise与BackwardConv均有独立有界writeback | 19个ordinary Conv observation、2个Depthwise和2个BackwardConv case各3样本，完整physical span、padding、suffix guard与completion已校验 | ordinary Conv的feature/output按NCx规划，weight layout未恢复前Conv保持observation；BackwardConv按weight-owned 8192B footprint分配；三种kind分别建capability，不互相代签 | 不把小非对称Conv exact外推large Conv；不把feature/output的NCx结论外推weight；不把bounded completion升级为通用Conv/Depthwise/BackwardConv numeric exact |
| DataMove/layout | Tensor↔Cx/NCx在`C=63/64/65/127/129`证明C0 tail、每个N slice的aligned-C和batch stride都是allocation footprint。large Pad/Img2Col/TensorNom分别写到19456B/88576B/17408B，排除compact logical bytes和“整份末尾只对齐一次”。GatherScatter可正确materializeconcat/broadcast/holes/tail。native raw Concat C/W/H只有bounded writeback；`dims=HW`不是可校准能力，而是错误指令用法 | base 46 case、extended安全17 case及large修正case，均检查all-and-only logical point、internal/batch padding、allocation外guard和completion；`dims=HW`只保留static-negative合同，不存在板端case | physical planner与oracle共用layout codec；concat/broadcast统一选择typed GatherScatter composite；native Concat不进入默认lowering；logical result、padding和guard分别规划 | 不把metadata view当搬运，不把C/W/H bounded completion当exact concat，不由Cx/NCx movement外推任意NTensor/native CT/NE consumption；不得以header可编码为理由重新发射`dims=HW` |
| Pool/Unpool | Pool的16个exact覆盖BF16/F32 symmetric及FP16 asymmetric unpadded；8个padding/tie只形成bounded behavior。indexed Pool的value与index是双writeback，F32 index占u32宽度、完整span 1024B。Unpool的index参数是same-shape i16 SPM buffer地址而非scalar；Avg与已验证symmetric mask组合exact，ordinary indexed及F32/large组合仅bounded，其中large FP16 mask只命中最后64-channel window；重复overlap尚无板端规则 | Pool 24个当前case及既有Unpool exact/observation有板端结果。新增`unpool-index-f16-repeated-overlap-observed`与`unpool-mask-f16-repeated-overlap-observed`两个真实device case：先由indexed-max产生四个都指向同一global位置、local index分别为5/3/2/0的真实aux，再在producer完成后把四组按sample轮换的不同FP16 sentinel写入pooled value；host要求aux逐bit正确、64个collision channel各自由候选四组sentinel的非空subset解释，并分别记录uniform或lane-varying的value mask与histogram；其它logical position与physical tail只取zero/seed并保持slot guard。两个case已经接入逐case CTest与runner但尚未上板 | IR显式携带index SSA buffer及其dtype；Pool按value/index分别分配。Unpool只开放明确exact组合；ordinary indexed、large及新重复overlap在得到板端分类前不进入通用numeric lowering | 不把F32 Pool的u32 index直接接到i16 mask-Unpool，不由完成性或case准备推导collision winner/accumulate规则，不由一个window外推任意padding/tie语义 |

先给当前profile可以直接执行的硬结论；后文的证据限制不改变这些compiler决策：

| 问题 | 当前硬结论 | compiler/runtime立即执行的决策 |
| --- | --- | --- |
| same-worker NCC依赖链是否每条都要wait | **不要。** 已测RAW/WAR/WAW链在相邻issue间无`TsmWaitfinish`时按issue order正确完成 | 保留SSA/effect dependency edge与issue order；删除链内逐edge wait |
| `TsmWaitfinish`放在哪里 | **只放completion-domain出口，并合并。** wait-each比wait-once稳定增加plan cycles | 只在NCC→Kcore/Direct DTE、跨worker join、barrier、terminal/host publication物化matching completion |
| 当前是否启用自动跨engine overlap | **有条件启用。** 只对expanded矩阵中三次重复正FU excess、correctness和plan同时通过的profile/shape/schedule cell启用 | candidate selection按pair、worker关系、transfer class、address relation和issue order查窄capability；未匹配cell仍串行 |
| queue depth能否直接当pipeline window | **不能。** documented `D`和tight `D+1`均能正确完成，证明静态depth不是总提交上限 | 不从header depth选择window；window由显式resource/lifetime gate约束 |
| 256B是否SPM硬对齐 | **不是。** `base+64/+128/+192`及`length=128/384`均已完成正确round-trip | 256B保留为preferred alignment；上述已测geometry允许通过legality，不建立未测byte alignment白名单 |
| 是否实施SPM/DDR bank coloring | **现在不实施。** 已测offset/phase/cache矩阵没有形成稳定可消费bank class | planner不做bank coloring，cost model不写入伪造bank周期或固定offset收益 |
| mapped-SPM是否需要dcache操作 | **不需要且不允许。** `0x30400000+offset`是uncached weak-order alias | mapped-SPM只用有序load/store与fence/sync；cacheable DDR按owned range clean/invalidate |
| ArgMin是否支持浮点负数 | **不支持当前负数域。** 全正普通值exact，负数对照稳定给出错误结果；F16 tie/NaN的两个区分case为`board-ready/pending-board` | 只准已验证全正普通值域；负数域由capability/verifier拒绝；tie/NaN等待真实board observation，其它dtype/domain由typed host gate拒绝 |
| Unpool是否通用exact | **不是。** 已列Avg和symmetric mask组合exact；ordinary indexed及其它已有组合只取得bounded observation；index/mask两条重复overlap case为`board-ready/pending-board` | 只对明确exact组合开放numeric资格；重复overlap未上板前不定义winner、覆盖或累加语义 |
| native Concat/BOOL是否进入production | **不进入。** native `dims=HW` Concat是错误指令用法，永久static-negative；native TDMA BOOL没有可靠completion | 合法source-level HW concat继续lower为typed GatherScatter，且不得为native `dims=HW`建立任何执行入口；BOOL physical fill使用已验证I8替代路径 |
| DTE与barrier怎么用 | **只用已验证的有序DTE和16-rank full-card barrier** | 保留显式event/wait；subgroup和sender overlap不进入production |
| timeout后是否继续试卡 | **不继续。** timeout后的管理面idle不代表execution健康 | 首个timeout立即停批，不retry/reset/power；新会话只由known-good Add重新建立资格 |

所以本轮已经解决的不是“硬件是否理论上还能更快”，而是production当前必须采用的边界：
**same-worker依赖链无逐条wait、completion出口合并wait、跨engine只按十个正excess cell有条件开放、
4KiB RDMA+WDMA只降plan不计overlap、256B非硬对齐、
不做bank coloring、mapped-SPM不用dcache、数值和DataMove按上述白名单准入。**

| 机制 | 从板端证据得到的行为结论 | 当前compiler/runtime处理 | 仍未得到的事实与区分性probe |
| --- | --- | --- | --- |
| NCC queue与连续提交 | 五类engine的documented `D`及tight `D+1`均完成count/result/guard。较长tight `D+1`在每类engine的3个样本中都于紧邻issue的control观察点尚未task-done，随后正确完成且无PMU blocking；这只证明观察点至少还有active work，不给出同时resident数量 | 不把header depth或`control=0`直接写成production outstanding window；候选必须保留matching completion，且每个window重新过result/guard/resource gate | queue-full/backpressure已有30个`D-1/D/D+1 × short/sustained`真实case待板端；resident数缺少record字段，已由typed host gate拒绝，不能靠再加packet闭合 |
| 跨engine执行 | 旧短workload的零excess不能代表所有长度。expanded sustained窗口中，16KiB七个same-worker compute/movement pair的window excess中位数为`61..1438` cycles，cross-worker CT+RDMA为533，64KiB三种movement pair为`2679..7471`；各组三次都为正且window plan均降低。4KiB RDMA+WDMA的disjoint/exact/half-partial、A→B/B→A共六组excess均为0但plan降低`490..890` cycles | 对十个正excess cell开放profile-scoped overlap candidate并保留真实双buffer/range/completion合同；4KiB RDMA+WDMA允许形成无中间wait的window以减少plan开销，但cost中的engine overlap收益必须为0。任何其它shape/pair/relation不外推 | 后续cost只需增加独立held-out transfer/shape验证和production source vertical；不再把已闭合cell写成Unknown |
| 地址依赖与busytable | contiguous RAW/WAR/WAW/RAR × exact/partial/adjacent及修正后的1D/2D/3D strided serial/window均按issue-order composition完成，count/guard正确。3个standalone DMA strict case同时确认DDR endpoint strided、local SPM endpoint compact。pure same-worker NCC链不插中间wait也正确完成；这不证明compiler可省略edge | scheduler从SSA、显式effect和实际半开range建立RAW/WAR/WAW edge并保持issue order；RAR只在另有resource/control edge时保序。DDR descriptor envelope与SPM compact footprint分开建模；hardware ordering实现IR edge，不替代IR dependency或lifetime | 跨worker同地址和超出current descriptor/range的busytable覆盖未开放；已测strided组合不再列为Unknown |
| wait与worker scope | `bywork`对matching worker有效；静态实现显示default wait只轮询worker0。default/local-fence与bywork对照返回时worker1均已完成，且六个proper-subset join返回时mask外worker也均已排空，因此这些样本不能区分跨worker scope。wait-each与wait-once方向性样本确认频繁wait增加plan cycles | pure same-worker NCC地址依赖不插wait；只在NCC→Kcore/Direct DTE、跨worker join、barrier、terminal/host publication等completion-domain exit放置并合并matching drain。每个跨worker boundary显式join真实participant | 18个wait-scope与12个subset include/exclude真实case已接入runner并要求pre-boundary pending，当前待板端；若仍自然排空则继续Unknown，不扩大scope |
| completion与cache visibility | SDK的`0x30400000 + offset`是uncached weak-order mapped-SPM alias，使用有序load/store与`fence`/`sync`，不得执行dcache操作；cacheable DDR的Kcore/host publication仍需按owned range clean/invalidate。depth-4 no-local-wait和local-wait在boundary snapshot前都已自然完成，marker/result/guard均exact，故这对样本不区分wait scope或必要性 | 完整Instr IR以typed MemoryEffects和SSA alias/root/path建立RAW/WAR/WAW dependency issue order，不在first conflict物化local fence；runtime仍在NCC→Kcore completion-domain boundary建立matching completion，且不把诊断用`bywork(0)`硬编码成长期handshake | default/local-fence跨worker scope及最小可靠completion handshake仍未由区分观察闭合。same-session DDR stale对照不阻塞当前一次性runtime；未来引入persistent session并跨launch复用同一allocation时重新启用 |
| SPM容量、alignment与reuse | allocatable boundary、64KiB transfer、5个nonpreferred geometry及1--4 iteration双slot均完成。256B CT+RDMA phase/offset sweep的window `Ea+Eb-FU≈0`、blocking=0，与16KiB稳定正overlap形成明确size effect；base phase和offset主要落在相近常态但夹杂单样本长尾，6144三次相对较慢也没有形成周期 | 256B作为preferred而非hard alignment；双slot由真实allocation/view、SSA lifetime与completion edge表达。短256B pair不计overlap；planner不做bank coloring，cost不写固定phase penalty | 4/16KiB sustained candidate/control共16个cell、4组已board-ready待执行；probe读取owner-backed port0/6 T2/T3并做matched WDMA readback，partial accept后先安全drain。它们最多形成offset-equivalence proxy；缺的是physical bank/port mapping与bank-specific attribution，不是port counter本身 |
| physical layout与DataMove | base movement、Tensor↔Cx/NCx以及large Pad/Img2Col/TensorNom闭合了logical points与NCx physical footprint：aligned-C padding和每个N slice的batch stride都是allocation的一部分。native Concat C/W/H只有bounded completion；native `dims=HW`是错误指令用法 | physical planner从layout codec计算每个slice与整份span；logical result、internal/batch padding和外部guard分开。合法source-level HW concat继续lower为typed GatherScatter，不选择native Concat | 不外推任意shape/dtype、NTensor或CT/NE native consumption；native `dims=HW`永久static-negative，测试和runner不得包含该packet |
| CT数值与physical write | current 653项catalog全部通过current exact result/span/guard/record oracle，0 missing；全部launch/lifecycle正常。旧63项mismatch已由corrected oracle重跑为63/63 `execute_result=1, mismatches=0`：op78--87使用数值truth逻辑，op92--94的BOOL VuV使用byte-addressed 40-bit physical RHS周期并对末尾RHS zero-fill。reduction另表明logical result 128B时硬件仍写256B physical block。raw N/HWC轴不在version-matched enum且首个隔离case未完成，现已从dispatcher移除 | legality按opcode/form/dtype/numeric-domain/shape逐项消费653个通过row；value logic和bitpacked BOOL VuV的logical/physical合同进入共享oracle与lowering。result bytes、physical output span和suffix guard独立建模；`VuVLoop`仅在`unit_elem_count==64`且checked base/full乘积关系成立时supported | 40-bit RHS周期与zero-fill只对已验证BOOL VuV合同成立；不能外推其它unit/layout/form。历史unit32/37 VuVLoop observation与raw N/HWC Reduce不进入production |
| NE数值与option | FP16/BF16 GEMM已覆盖非平凡累加、tail、batch、orientation和local psum。ReLU实际等于bare且保留负值；large ordinary Conv与当前host physical indexing不符。修正schema的BackwardConv FP16/BF16各3个样本均按8192B footprint完成，physical span、suffix guard和completion通过 | 只对已验证GEMM组合给exact资格；ReLU和ordinary Conv保持observation，不把wrapper enable bit当语义；BackwardConv按weight-owned shape和8192B footprint保护range，允许bounded observation但不宣称通用numeric exact | BackwardConv/ordinary Conv已有真实bounded board row；缺唯一typed NCx/HWOI numeric oracle时catalog保持observation/fail closed，不把另一个无oraclecase登记为待上板 |
| Pool/Unpool与peripheral | Pool的16个exact向量闭合当前BF16/F32 symmetric与FP16 asymmetric unpadded geometry；8个padding/tie向量只取得bounded observation。indexed F32的value后跟同format宽度的u32 indices，完整span为1024B。Unpool证明index参数是i16 SPM buffer地址而非scalar；F32 mask与large asymmetric FP16 mask虽有界完成，但数值分别存在producer/consumer宽度不一致与只命中最后64-channel window。ArgMin全正普通值域exact，负数域错误；Bilinear已有3样本bounded observation | Pool按value/index各自dtype宽度规划双writeback；IR用same-shape i16 SSA buffer连接已验证indexed pool/unpool。ArgMin tie/NaN各1个case与Unpool index/mask repeated-overlap各1个case已经真实接入board CTest和runner，当前均`pending-board` | F32 index/u32与i16 Unpool producer-consumer宽度不一致由IR verifier拒绝；large Unpool和Bilinear已有bounded板端证据但不能升级为通用exact语义。只有重复overlap规则及ArgMin tie/NaN仍待本批板端分类；其它ArgMin dtype/domain由typed gate fail closed |
| Direct DTE与barrier | 有序producer→DTE、DTE→consumer、两种disjoint顺序、event后复用和两destination broadcast均完成payload矩阵；四种同步错误观察得到预期transport error后clean success；raw async serial/window各3×16 exact且返回码全0；full-card barrier两个反向错峰epoch均16/16正确 | DTE以显式event/token和wait表达；raw async允许在send/wait之间发射独立CT，但正确性证据不等于时间重叠收益；当前只准16-rank full-card barrier，subgroup在发包前拒绝 | 25个raw multidestination/fan-in case已接入board CTest：broadcast/scatter/shuffle的2/4/8/15 fanout各测adjacent/interleaved，另测四源fan-in；均仍pending，`dest_num`/mapping只作观察。8/15源fan-in、可信device phase、physical route和跨卡transport继续typed fail closed |
| PMU与cost | NCC per-engine count可用于确认实际发射；只有稳定的FU union关系可判overlap。单次execution delta、blocking=0或offset sweep不能成为固定latency | PMU先用于资格和归因；只有重复、同scope、counter enable稳定且correctness通过的对照才能进入cost model | DTE saved mode仍缺可信device phase basis；SPM sustained probe已有owner-backed port0/6 T2/T3读取，但缺physical bank/port mapping与bank-specific attribution，不能据此命名bank或直接写cost。TMNOC无decoded只读offset，保持static-negative |
| timeout与设备上下文 | native BOOL表明“header可编码”不等于matching completion；错误发射native `dims=HW`也曾触发timeout。timeout后管理面idle不能证明execution context健康。独立clean session中的合同内`VuVLoop unit=64`及前后Add均正常 | native `dims=HW`不是待恢复能力，而是永久static-negative；其它危险packet在current profile excluded。首个timeout即停批，不自动retry/reset/power，也不从邻近enum外推能力 | 无动态恢复动作；新session只用一次ordinary Add建立执行资格；native `dims=HW`永不发射，其它危险packet保持excluded |

由上述行为模型直接得到当前策略：

1. **已经允许**：已验证opcode/form/dtype的保守单engine lowering；显式Tensor↔Cx/NCx materialization；
   当前SPM容量/对齐下的静态packing；matching wait、逐worker join、mapped-SPM有序访问以及DDR owned-range
   cache clean/invalidate；已验证的DDR-strided/compact-SPM descriptor、40GiB workspace内相对地址；
   expanded矩阵十个正excess cell的profile-scoped overlap candidate；有序Direct DTE和
   16-rank full-card barrier。
2. **暂不允许**：未匹配expanded白名单的自动overlap、用queue depth选择pipeline window、SPM bank coloring、
   default wait代替跨worker join、任意轴native Concat、原生TDMA BOOL、subgroup barrier，以及把NE
   ReLU/Conv option当作exact语义；raw N/HWC Reduce与未进入两个已验证
   `VuVLoop unit=64` geometry的其它numeric/tail组合同样保持fail closed。历史op013→op014是
   out-of-contract observation，不再列为production待重放边界。
3. **后续执行第5.10节已接入runner的pending区分实验时，目标也不是再累计通过数**：每个case必须明确区分一个仍可能
   成立的行为模型，例如“default wait是否覆盖worker1”或“同一SPM冲突分类能否跨base/tile复现”；
   不能区分机制的重复smoke不进入执行批次。

本轮已经保存的板端证据按“结论而不是通过数”收敛如下；未列为板端结论的新增资产仍只是离线准备：

| 证据域 | 已经得到的结论 | 对compiler/runtime的直接约束 | 仍待板端区分 |
| --- | --- | --- | --- |
| CT / peripheral | current CT catalog 653/653通过current exact oracle；旧63项mismatch在修正value-truth及BOOL VuV 40-bit physical period/tail zero-fill oracle后全部重跑为`execute_result=1, mismatches=0`。合同内`VuVLoop unit=64`的两个区分geometry及其前后Add已闭合；ArgMin只在全正普通值域得到`0.5@index42` exact，负数域返回错误结果；普通indexed Unpool只得到有界执行观察，FP16 indexed-max→mask-unpool组合得到exact | 按实际已有板端证据准入；ArgMin tie/NaN两个case及Unpool index/mask repeated-overlap两个case已接入board CTest和runner但尚未上板，不能混入本列“已经得到的结论” | BOOL VuV 40-bit周期不外推其它unit/layout/form；ArgMin tie/NaN和Unpool repeated-overlap待板端分类；其它ArgMin dtype/domain由typed gate fail closed |
| NE | FP16/BF16 GEMM的main、K/N tail、batch、orientation和FP16 psum已经闭合各自bit oracle；BackwardConv修正footprint后的FP16/BF16各3个样本已闭合bounded span/guard/completion；ReLU没有表现出clamp，普通Conv只允许消费已由非对称vector恢复的窄合同 | GEMM只开放已验证组合；BackwardConv按weight-owned 8192B footprint规划并允许bounded observation；ReLU与未唯一恢复physical indexing的Conv option保持observation | BackwardConv与ordinary Conv已有真实bounded board case；缺唯一typed numeric/physical oracle的组合由catalog disposition保持observation/fail closed，不伪造成另一组无oracle板测 |
| DataMove / layout | base与extended安全路径、Tensor↔Cx/NCx、Pad、Img2Col和TensorNom已经区分logical span、physical padding及batch stride；native Concat C/W/H只证明有界完成；native `dims=HW`是错误指令用法 | 合法source-level HW concat继续使用typed GatherScatter；layout planner必须按真实physical codec规划padding和跨N步进 | native `dims=HW`永久static-negative，测试面不保留任何执行入口 |
| SPM / DDR / cache | SPM 24个本地board row、5个DataMove-backed delegated row及107个memory-backed delegated row均有current-schema证据；memory 155行已按current 2MiB schema全量重放通过。独立cache 58行、16-rank tile-offset矩阵和40GiB sparse workspace有各自板端结果 | 256B仍是preferred alignment而非硬限制；mapped-SPM只使用有序访问，cacheable DDR按owned range做publication；只让已验证row进入当前资格 | SPM sustained 16 cell/4 group、DDR active-rank 45 case/9 group及SPM/DDR equivalence CTest均已board-ready待执行。SPM probe已有owner-backed port0/6 T2/T3与matched WDMA readback，但physical bank/port mapping和bank-specific attribution仍typed fail closed；DDR physical bank/controller/hop同样未闭合 |
| NCC / 同步 / 并行 | 五类engine的documented depth、tight `D+1`、same-worker依赖链、逐worker join、wait-each/wait-once、large backlog、手写双slot及34个expanded pair均通过各自count/result/guard。3个DMA和36个NCC strided case亦通过。expanded中十个disjoint sustained cell取得三次重复正FU excess，4KiB RDMA+WDMA六个cell excess恒为0但plan更低。depth-4 A/B都exact，但snapshot前已自然完成，未区分wait | RAW/WAR/WAW保持显式edge和issue order；正excess cell可生成窄overlap candidate，4KiB RDMA+WDMA只合并issue/wait而不计engine overlap；NCC→Kcore等completion-domain exit仍物化matching completion | queue response 30、wait scope 18、subset scope 12和worker characterization 44个case/14组均已board-ready待执行；worker oracle在drain前捕获boundary并交叉核对`CONTROL`，同时检查同slot sentinel、tail canary、join RC/idle与deadline。resident数、absolute worker timestamp及arbiter policy为typed fail-closed |
| DTE / barrier | 修复后的64元素production baseline为16-rank exact；modes 1--6的30个full-card payload case全部通过expected/status/SPM guard/cleanup；modes 7、8、9、12的16B错误观察均得到预期transport error后clean success、exact guard和正常cleanup；modes 10--11的64KiB raw async serial/window各3×16 exact、返回码全0、guard与cleanup正常；至此modes 1--12均通过各自oracle。16-rank full-card barrier也已闭合。DTE channel0 transfer raw只取得严格payload单调与1×/2×相对scale；execution非单调；这些历史mode没有启用SPM PMU | 保留显式event/wait，只支持当前16-rank full-card participant合同；已验证同步错误必须由shadow status保留错误并由runtime clean owner正常收尾；raw async window只开放调用顺序正确性，不据此写时间重叠收益；DTE transfer raw只作profile内相对量证据，不写绝对单位或latency cost | 25个raw multidestination/fan-in case已board-ready但尚未上板；`dest_num`/mapping只记录观察。8/15源fan-in、device phase、physical route和跨卡transport保持fail closed；subgroup继续typed-negative |
| failure boundary | out-of-contract `VuVLoop unit=32` timeout后，后置Add也timeout；native BOOL、错误发射的native `dims=HW`和未证明SPM range的诊断也出现过不可靠completion。独立clean session中的合同内`unit=64`及前后Add均正常 | 任一timeout立即停批；不retry/reset/power。合同外VuVLoop与native `dims=HW`永不发板；每个新session只用一次ordinary Add建立执行资格 | 无待恢复动作；native `dims=HW`保持static-negative，其它危险packet保持excluded，安全合同按独立结果消费 |

因此当前结论不是“硬件已经可以全面并行”，而是：保守单engine lowering、显式依赖顺序和
completion-domain boundary继续作为正确性基线，同时十个expanded cell允许进入production候选选择；
4KiB RDMA+WDMA和其它未匹配cell不能计engine overlap收益。SPM/DDR bank-aware cost仍无physical分类证据。

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
  issue形态下永远不能并行。hazard correctness现已与性能资格解耦：contiguous
  RAW/RDMA→CT、WAR/CT→TDMA、WAW/RDMA→TDMA、RAR/CT→WDMA分别覆盖exact/partial/adjacent的
  serial/window，共24个case、72个样本，全部满足issue-order composition、instruction count、完整结果和
  guard。pure same-worker NCC的window在相邻RAW/WAR/WAW之间没有插`TsmWaitfinish`，只在最终
  completion-domain exit drain；它证明current有界packet可按IR issue order由busytable正确完成，不授权
  compiler删除RAW/WAR/WAW edge，也不形成overlap收益。RAR不因地址关系本身增加edge；只有未来同方向
  对照稳定取得正median才重新开放对应并行候选。
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
- 修正probe在packet构造后立即删除builder，并在`TsmExecute`后立刻读取control。独立clean session先运行known-good Add，
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
- 随后的large tight `D+1` active-observation把CT/NE/RDMA/WDMA的7条与TDMA的5条各独立重复3次；15个样本在
  紧邻issue的control观察点均为未task-done，最终count/result/guard/completion全对且blocking delta为0。
  因而当前profile确实能在该观察点保留至少一个active request，但control没有resident count，且没有出现
  queue-full或backpressure；不得据此选择production outstanding window，任意更深overflow仍不执行。
- large backlog的64KiB RDMA/WDMA、large CT/NE及其四个compute/movement serial/window组合均有界完成。
  保存的RDMA+NE三次window仍满足`rdma_exec + ne_exec == full_exec`，没有稳定overlap。1--4 iteration的
  RDMA→CT→WDMA双slot serial/window八项也都保持两slot、count/result/guard正确；它们只闭合手写issue顺序和
  显式completion下的复用，不证明production software pipeline或多buffer调度已经成立。
- raw/wrapper各覆盖五类engine的两次window提交，十项issue-path都完成；wait-each/wait-once的十项配对同样
  完成且继续显示频繁wait增加plan cycles。builder构造方式和wait位置因此可以进入candidate cost，但不能成为
  IR语义或跨profile固定延迟。
- expanded sustained pair使用预构packet、4轮双buffer、serial/window各3个样本，所有lane均满足独立
  result、双侧guard、worker instruction count和final control。16KiB同workerdisjoint窗口的三次
  `Ea+Eb-FU`全部为正：CT+NE `[382,534,537]`、CT+RDMA `[533,532,527]`、
  CT+WDMA `[540,536,544]`、CT+TDMA `[59,61,45]`、NE+RDMA `[1296,1529,1495]`、
  NE+WDMA `[1321,1246,1300]`、NE+TDMA `[603,605,605]`；cross-worker CT+RDMA同样为
  `[534,531,537]`。64KiB movement窗口的RDMA+WDMA、RDMA+TDMA和WDMA+TDMA分别为
  `[7740,7471,6981]`、`[2782,2812,2799]`和`[2669,2679,2680]`。所有这些cell的三次
  window plan都低于配对serial control，因此它们取得窄overlap capability，不再沿用旧短workload的零结论。
- 4KiB RDMA+WDMA的far-disjoint、exact和half-partial分别按A→B/B→A执行；六个cell的三次
  `Ea+Eb-FU`全部为0，而window plan每次仍低于serial。硬件在该transfer class没有表现出engine execution
  overlap，但去掉中间completion仍降低issue/wait计划开销；compiler必须保留issue order和range edge，
  只把plan收益计入cost，不能伪造FU overlap收益。

overlap只按下面的PMU关系解释：

```text
pairwise_excess = engine_a_exec + engine_b_exec - fu_union_exec
```

`FU_EXE_TIME`是active engine时间的union；`*_BLOCKING_TIME`是queue backpressure，不是dependency stall。

### 4.2 单engine correctness与ABI观察

- CT f16 Add使用非零输入、完整fp16 golden、guard和DMA round-trip通过。
- vendor instruction archive的一参`rt_malloc/rt_free`由匹配Kcore的
  `__rtmsym_rt_malloc/__rtmsym_rt_free`直接解析。删除CRT中臆造的scope-0
  `csi_kernel_malloc/free`桥后，最终module保留exact system-heap UND；constructor返回非零method-table
  地址，builder在packet materialize后释放，随后CT issue/count/result/guard仍完整正确。这闭合当前loader
  与vendor builder的ownership路径，不允许CRT从相邻三参API猜scope或free配对。
- 旧CT vector catalog的首个`VuV`实卡case在logical element 8177开始保留output canary：
  `8192 = 37 * 221 + 15`，首差与完整unit边界精确重合，且probe还给非loop opcode写入了`full_*`。
  该case超出当时catalog已闭合的完整-unit矩形，不能据此把非整除geometry判为硬件非法，也不是Max opcode
  的负向能力证据。修复后的catalog使用shape-owned unit，普通`VuV`不写`full_*`。`VuVLoop`的
  supported interface不是由这些raw结果反推，而是固定要求`unit_elem_count == 64`以及
  `full_elem_count * unit_elem_count == elem_count * full_unit_elem_count`。本轮op010的剩余5个
  tail/dtype向量，以及op011 `VS`、op012 `VuV`、op013 `VuVLoop`的18个main/tail/dtype向量曾逐bit通过；
  其中op013使用的32/37-element unit只能作为out-of-contract hardware observation，不能证明对应packet
  属于production合同。clean reboot后op014 f16 `VV Add`隔离执行逐bit exact，但紧接out-of-contract
  op013执行时只在首128B失败，后置known-good Add也错误；该序列同样不能推出production first-conflict
  wait或数值资格。
- 本次又将`unit_elem_count=32`作为独立raw case执行，case在completion内timeout，随后的既有Add
  heartbeat也timeout，说明该次execution会话已经污染；事故会话随即停止。这个失败只记录
  out-of-contract hardware behavior。今后违反上述任一`VuVLoop`关系的raw packet在host legality处拒绝，
  不再以隔离、observation或held-out名义上板。
- instruction-family typed catalog原有60个safe case已逐个串行launch并通过完整bit oracle、SPM guard、
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
- ArgMin现在另有F16 tie与NaN两个真实board case：tie用`(5,69)`、`(0,127)`和`(31,32,96)`三组候选，
  NaN用positive quiet、positive signaling和negative quiet三种bit pattern并保留独立finite minimum；
  host要求返回value/index彼此coherent、index有界、padding poison不变、guard、terminal和cleanup完整。
  两项已经注册逐case CTest并进入`pending-execution-boundaries` runner batch，但尚未上板，故不预写
  tie-breaking或NaN selection/quieting结论。F16 signed-zero/Inf/subnormal及BF16/F32等其它ArgMin域没有
  独立typed row，由`argmin-unqualified-domain`准备门禁fail closed，不能拿相邻CT special-value代签。
- catalog随后新增第61个`unpool-index-f16`：独立执行indexed-max产生index、matching fence和opcode 121
  ordinary Unpool，并以完整512B bounded raw/guard/completion observation与既有opcode 123 mask-unpool
  分开记账。旧板端执行在exact oracle下失败，因旧seed不能区分no-op与写零；当前改用非零poison并降为
  bounded observation；非零poison的current板端复验已完成，ordinary indexed Unpool继续只取得有界执行资格。
- instruction-family concrete inventory仍有168项，但只有160项可进入普通板端：137个exact、23个bounded
  observation；8个raw N/HWC Reduce已经从safe dispatcher移除。对应117个typed Reduce/Pool/Unpool row现在为
  80个board-executable、18个board-observation、11个static-negative和8个isolated-deferred，0 Unknown。
  version-matched `Reduce_Dim`只公开C/W/H/HW；历史raw dimension 3的首个N-axis case超过外层completion
  deadline并污染execution context，因此N/HWC × Sum/Avg/Max/Min全部fail closed，不能由raw数值编码或邻近轴
  外推。
- Pool当前24个新增向量均已取得板端结论。BF16/F32 symmetric和FP16非对称无padding的16项按完整numeric
  expected通过；8个padding/tie项有界完成但只保留observation。F32 indexed Max/Min的index writeback是
  128个`uint32_t`，不是`uint16_t`：512B value加512B index构成1024B完整span，旧768B span会把正确后256B
  误报成guard写坏。compiler必须按index output dtype规划第二个writeback，不能固定成i16宽度。
- Unpool按实际语义分层：Avg的FP16/BF16/F32与symmetric mask composite的FP16/BF16保留exact；
  ordinary indexed FP16/BF16/F32只保留bounded observation。F32 mask虽然1024B result、
  256B auxiliary及guard都完整，但其producer写u32 index而consumer按i16 mask解释，故只能记observation；
  FP16 `k3x2/s2x1` large mask的四个64-channel source window中只有最后一窗符合semantic scatter，前三窗为零，
  同样降为bounded observation。它们证明请求有界完成，不证明通用mask/index canonicalization。
  重复overlap不再只是文档待测项：`unpool-index-f16-repeated-overlap-observed`和
  `unpool-mask-f16-repeated-overlap-observed`先用真实indexed-max生成`5/3/2/0`四组aux，再在明确
  producer completion后注入按sample轮换的四组不同FP16 sentinel。host逐bit检查aux，要求64个重叠
  target channel各自都能由候选四组sentinel的非空subset解释，并记录uniform或lane-varying的
  position/value mask与histogram；同时限制其它logical position、physical tail、slot guard、matching
  terminal和cleanup；padding-only写入、aux篡改或单个target corruption均有host反例，
  因而不会再把“任意bounded writeback”冒充collision观察。两个case已经进入逐case CTest及
  `pending-execution-boundaries` runner batch；当前尚未上板，因此collision winner/覆盖/累加规则仍未知。
  既有Factorize、LUT32、RandGen、ElemMask各3样本bounded observation通过；
  Bilinear曾按旧exact oracle失败；当前3个板端样本已按bounded observation通过，但数值语义仍未取得exact资格。
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
  这两个非对称小geometry只闭合current wrapper和FP16/BF16 format的窄板端向量；large ordinary Conv从
  logical element 97开始与current oracle不符，因此不能把小geometry外推成通用Conv exact/supported资格。
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
- RDMA和WDMA分别以非零pattern、全range round-trip和精确instruction count通过。保存的FP16
  1D/2D/3D strided raw结果确认DDR sink的scatter位置、holes和双侧guard；进一步失败归因撤回了
  “mapped-SPM stale cache”假说：descriptor stride只作用DDR endpoint，local SPM endpoint始终是连续
  `transfer_bytes`，packet的保守end envelope不改变其数据布局。旧device oracle错误按DDR scatter offset读取
  SPM，分别产生26/32/44个mismatch。修正后的3个standalone DMA strict case全部完成DDR scatter、
  compact-SPM、holes、guard和count oracle；36个NCC case进一步完成1D/2D/3D RAW/WAR/RAR
  serial/window及WAW exact/partial/adjacent serial/window的composition、guard、count和completion。
  因此已验证组合取得strided round-trip与issue-order correctness资格，但不产生重排或overlap资格。
- NE f16 16x16 identity GEMM通过；实际source/output register range均覆盖256B。retained logical
  `C0=16`使用compact stride 16，错误使用full block stride 64只会得到前4个正确对角元素。该case证明layout
  vector必须能区分logical tail和full physical block。
- 旧probe曾把`TsmNewArith`返回值错误解释为“允许数值0的local address”。该接口实际返回host
  method-table allocation，空指针是allocation failure；不得解引用或delete空指针。system heap ABI修正后的
  constructor observation已在板端记录non-null `uintptr_t`，并证明packet materialize后立即delete builder不影响
  独立packet的count/result/guard。
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
  `logical_valid` BOOL fill。136 logical bits→17 physical bytes的BOOL→I8 held-out与独立I8 physical16
  control均完成全range readback和guard，因此该替代路径在完整physical-footprint域`board-observed`；
  它仍不授权native BOOL packet或logical-valid尾bit保留语义。

### 4.4 Cache、completion与DTE

- Kcore普通load读取复用的cacheable DDR input可能命中旧cache；对range执行匹配cache invalidate和fence后
  恢复exact。同一payload的NCC DMA round-trip不受该旧load影响。
- 因此host H2D、Kcore DDR load、NCC local drain和host-visible publication是不同visibility合同，不能互换。
- SDK把`get_spm_memory_mapping(offset)`定义为
  `KUIPER_L1SPM_UNCACHE_WEAKORDER_BASE + offset`，当前base为`0x30400000`。这是uncached weak-order
  mapped-SPM alias：CPU通过该alias读写只需用`fence iorw,iorw`/`sync`建立顺序，不得把mapped pointer传给
  `dcache.cipa/civa/ipa/iva`。只有raw `0x0 + offset` cacheable SPM alias才适用对应cache操作；DDR
  publication/readback仍按实际owned cache line执行clean/invalidate。此前把ArgMin seed保留解释成
  mapped-SPM cache publication失败的结论已经撤回。
- version-matched静态反汇编显示`TsmWaitfinish()`轮询default worker 0；
  `TsmWaitfinish_bywork(worker)`轮询指定worker，current `wafer_tx81_local_fence()`直接调用default
  `TsmWaitfinish()`。worker0短TDMA与worker1 large NE的default/byworker/local-fence三向对照中，三个
  boundary的worker1 marker、result和task-done都已完成；这只说明请求在观察前自然排空，不能证明
  default/local fence覆盖worker1。六个proper-subset join也都保证mask内worker完成，但mask外worker同样已
  排空，没有观察到pending exclusion。跨worker scope继续为`unknown`，compiler必须按实际participant逐worker
  join。
- 五类engine各自的两次issue均完成wait-each/wait-once对照，全部result、guard和completion正确；频繁wait
  的plan cycles方向性更高。该观察支持compiler把wait放在latest-legal completion boundary并合并相邻wait，
  但样本不形成固定wait latency、比例或跨workload cost常数。
- RDMA→CT、CT/NE→WDMA、TDMA→CT/NE五个ordered producer/consumer均以pure same-worker NCC链在相邻
  instruction之间不插wait，完整result、count和guard正确；current verified descriptor域由IR dependency
  issue order与worker busytable完成RAW/WAR/WAW。CT/NE写后由Kcore直接读取mapped SPM的两个case则在
  NCC→Kcore completion-domain exit执行matching drain后正确，不能把这个domain boundary反写成前述NCC
  链内每个first consumer的要求。mapped-SPM本身的cache属性由SDK地址域定义，不再由单个数值case推断。
  本轮DMA strided mismatch的cache解释也已撤回，原因是DDR-strided/compact-SPM布局oracle错误。
- clean reboot后，`op014 F16 VV Add main`隔离执行逐bit exact；先执行
  `op013 F32 VuVLoop min tail`再执行同一`op014`时，后者仅bytes `[0,128)`错误、byte 128以后exact，
  且错误字节既不是canary也不是完整前一输出。128B等于一个CT/SPM 1024-bit beat，不是C908的64B cache
  line；随后其它CT及known-good Add也可数值错误。该现象排除普通cache，但由此推导“RDMA→CT与CT→WDMA
  的first conflict前必须插`TsmWaitfinish`”是错误结论：上述无中间wait的same-worker NCC dependency
  向量已经按issue order和busytable正确完成。完整Instr IR仍从typed MemoryEffects和SSA alias/root/view
  path建立RAW/WAR/WAW edge并保持issue order，但不为链内edge物化local fence。该op013使用32/37-element
  unit，位于当前supported `VuVLoop`合同之外；它的exact输出与后续异常都只保留为out-of-contract
  hardware observation，不再作为production completion或geometry的待闭合正向case。违反合同的geometry
  由host negative闭合，不再上板探测。
- runtime local fence需要建立issue到completion poll以及poll返回到后续issue的机器顺序；当前
  `TsmWaitfinish()`只轮询CSR的事实不足以闭合完整handshake。它只在NCC→Kcore/Direct DTE、跨worker
  join、barrier、terminal/host publication等completion-domain boundary使用并尽量合并；诊断中加入
  `bywork(0)`与default wait只提供保守区分证据，不能硬编码成跨worker或长期ABI合同。
- 地址交换诊断曾在未证明整个range属于可用SPM arena前，对`0x70000`起始地址执行64KiB WDMA，造成真正
  30秒completion timeout并使BoardRuntime将context标记为poison/quarantine。该诊断已经从probe撤销，
  `0x70000..0x7ffff`不得复用；后续任何诊断地址必须先证明整个半开range的owner、reservation和legality。
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
  status begin/finish视为entry transport lifecycle，而不能把它提升为新launch kind，或用未发布的poison
  status判断硬件同步失败。

## 5. 校准 case 规划与执行批次

本节是实卡执行的唯一case拆解入口。第3节28行只按compiler consumer提供稳定导航，不能直接作为准备
完成单位；真正门禁是每行下面的叶子需求。每个叶子必须由机器索引解析到具体catalog entry或带原因的
`static-negative`/`isolated-deferred`，并独立绑定shape/value、oracle、physical span/guard、执行域、
shared dispatcher/package与no-card gate。整行状态从叶子自动汇总，不能手填。

旧版索引只检查了28行的文件存在、CTest注册和空`remaining_preparation`，因此曾产生错误的
`28/28 ready`。该结论已经撤回；已有板端事实和已实现case不失效，但只作为对应叶子的证据，不能为
concat轴、SPM/DDR bank、worker mask或其它叶子代签。本轮为28个导航域补齐叶子级catalog对象或带理由
的negative/deferred对象；最终叶子总数和处置计数由机器矩阵单一事实源给出。对应host oracle、target C
build/link、shared-package no-card与矩阵一致性门禁闭合，只表示case可以按明确处置执行或跳过；新增row
仍须在相同profile身份下实际执行，no-card通过不升级证据成熟度。

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
`board-positive`；可以安全捕获但尚无独立语义expected的组合进入`board-observation`；由另一份具体
board case拥有payload/oracle/completion的组合进入`delegated-positive`并解析该证据对象；typed非法组合
进入`static-negative`；只有可能永久等待、写出owned range或缺少同一runtime session生命周期owner的组合
进入`isolated-deferred`。因此“覆盖全部组合”指每个组合都有明确结论和gate，不是把非法或危险组合也做成
板端笛卡尔积。

### 5.2 CT opcode、form 与 dtype suite

CT以current header的`0..186`为完整inventory。除bitpacked BOOL自身没有浮点dtype外，浮点入口均分别建立
FP16、BF16、FP32 row；不能由同opcode的另一dtype外推。packed BOOL logic明确编码为hardware
`Fmt_BOOL=7`，不能借用FP16/BF16/FP32 format；relation的BOOL output同时保留其输入dtype和独立bitpacked
输出合同。

| opcode/family | 计划case | main oracle与held-out |
| --- | --- | --- |
| `0..5` unary arithmetic | Abs、Recip、Square、Sqrt、Rsqrt、Neg × FP16/BF16/FP32 | 8192普通值；8214 held-out tail。Recip/Sqrt/Rsqrt分别含正域、零和负域隔离row |
| `6..29` binary arithmetic | Max/Min/Add/Sub/Mul/Div × `VV/VS/VuV/VuVLoop` × 3 dtype | 普通`VuV`的8192主向量使用32-element unit、8214 tail使用37-element unit；`VuVLoop` board-positive必须改用64-element unit并满足checked base/full乘积关系。已保存的32/37-element `VuVLoop` exact只作out-of-contract hardware observation，合同外row只跑host negative |
| `30..77` relation | Eq/Ne/Ge/Gt/Le/Lt × `VV/VS/VuV/VuVLoop` × value/bitpacked BOOL output × 3 input dtype | 同一truth pattern同时验证value codeword与BOOL逐bit结果；检查BOOL末字节未用bit和suffix guard |
| `78..97` logic | value Not/And/Or/Xor及公开`VV/VuV/VuVLoop` form；packed BOOL Not/And/Or/Xor及对应form | value按FP16/BF16/FP32 raw codeword做bit-exact；BOOL用交错、全0、全1和非整字节tail |
| `98..104` transcendental | Log2、Ln、Pow2、Exp、ExpLp、Sin、Cos × 3 dtype | 正常域用高精度reference记录误差；定义profile tolerance前只记`board-observed`，域外和special value独立 |
| `105..110` activation | Tanh、Sigmoid、Relu、SatRelu、LeakyRelu、Softplus × 3 dtype | 负/零/正、拐点两侧和饱和区；参数隐含行为未闭合时不升级为production支持 |
| `111..114` reduce | Sum/Avg/Max/Min × C/W/H/HW × 3 dtype | 非对称`[2,7,9,65]`，每个轴使用不同pattern；held-out覆盖C block边界、负值和tie |
| `115..120` pool | Avg/Sum/Max/Min及indexed Max/Min × 3 dtype | 非方kernel、非对称stride/pad；indexed case检查value、index编码、tie-breaking和双writeback guard |
| `121..123` unpool | scalar-index Unpool、Avg Unpool、mask/index Unpool × 3 dtype | 多个非零位置、重叠/非重叠窗口和完整destination footprint；index来源及重复index语义分别记账 |
| `124..138` DataMove | 由5.3节逐entry展开 | all-and-only logical point、完整physical span和每段padding canary |
| `139..174` convert | 36条source→destination route全部入表；有参数的route覆盖round-to-nearest-even、zero、+Inf、-Inf | 普通值、正负halfway、饱和/溢出、zero-point；stochastic rounding单独统计suite，不与确定性exact混跑 |
| `175..186` peripheral | Count、BitCount、ArgMax/Min、Memset、Factorize、Bit2Fp、Bilinear、LUT16/32、RandGen、ElemMask | deterministic entry验证全部writeback；RandGen/随机ElemMask重复采样raw output并验证有界write span、guard、completion和cleanup |

实卡前的shared-package资产已把opcode `0..110`展开为653个可筛选row：FP16/BF16/FP32、
`VV/VS/VuV/VuVLoop`、value/bitpacked BOOL、8192-element calibration/8214-element held-out以及zero、
signed-zero、normal/subnormal、Inf、quiet/signaling NaN和domain boundary共用一个device dispatcher；
542个为exact、81个为预先声明tolerance、30个为只保存raw result的`board-observation`。convert
`139..174`的36条route共有204行：72个main/tail nearest-even、69个directed-rounding、36个halfway/extrema、
4个zero-point和23个stochastic；其中158个exact、46个observation，23个stochastic row使用directed
halfway输入并默认重复3次，分别保存raw result和physical-span hash。`111..138`和`175..186`复用
instruction-family/DataMove concrete case或给出显式非执行处置；最终`0..186`的187个opcode全部入表，
并由typed profile、board executable/observation、static-negative或isolated-deferred逐项处置。
上述资产行数不是板端准入计数；其中历史`VuVLoop unit=32/37`行即使保存exact结果，也必须在current
supported interface gate处归为out-of-contract evidence，不能进入board-positive filter。
instruction-family concrete catalog共有168项，其中160项safe；Reduce/Pool/Unpool的117项为80个
board-executable、18个board-observation、11个static-negative和8个raw-axis isolated-deferred，0 Unknown。
raw N/HWC Reduce不再进入device dispatcher；处置完备不等于相邻dtype/axis取得能力。

算术/比较的form参数还要有专门的区分向量：`VS`的scalar bit pattern不能等于任一vector首元素；`VuV`的unit
不能整段常量；`VuVLoop`先以host negative锁定`unit_elem_count == 64`和checked base/full乘积关系，合同内
每个outer chunk再使用不同64-element unit pattern，并让最后一个完整chunk参与expected，held-out另保留
非256B physical span tail。这样正向case能区分“误当VV”“只重复首unit”“忽略full字段”和“只执行第一段”，
而不把违反supported interface的raw packet带到板端。

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
| CT vector × Tensor/NTensor | Tensor执行每个CT vector opcode/form/dtype的main与tail；NTensor缺独立direct ABI，做static-negative并先materialize compact Tensor | 只有Tensor是current native正向资格 |
| CT vector × Cx | lane-independent代表覆盖full-physical raw span；另以Tensor↔Cx显式movement只处理logical-valid points | raw结果只校准硬件；production仍需materialize或未来typed segmented consumer |
| CT vector × NCx | 至少两个N slice，slice使用不同pattern和独立padding poison；逐slice重放Cx逻辑 | 不得把NCx静默flatten成单个Cx |
| reduce/pool/unpool × Cx/NCx | 原生wrapper声明的shape/layout逐kind正向；compact Tensor版本若需转换则重放movement | 不由GEMM或CT vector结果外推 |
| GEMM × Cx/NCx | rank-2 Cx与single-leading-batch NCx分别闭合，见5.4节 | current production native路径 |
| 任意不受typed ABI支持的direct组合 | verifier/lowering/required-symbol negative | 不上板、不以raw packet绕过 |

NE实卡前资产使用同一host physical codec生成Cx/NCx block-major payload、C0 tail、N-slice步进和padding
poison，当前共73个row：32个exact、38个安全raw observation和3个static-negative，0 deferred。既有
FP16/BF16 × `NN/NT/TN/TT` × main/tail/batch基础GEMM、长累加/cancellation、BF16 special、
one-factor option和large/held-out Conv全部保留；新增I8 quant raw observation，FP16/BF16
DepthwiseConv与BackwardConv raw observation，以及FP16/BF16左右不等batch
`L1/R2`、`L2/R1` exact。基础GEMM使用独立salt的32-bit mixer生成全非零±1 operands，每个output都包含K个
非零贡献；每个batch的N个跨M column signature互异，每行至少有8个不同结果，并由N轴循环置换故障注入
证明physical oracle可检出channel/C0-tail顺序错误。所有safe row完整校验logical/physical output、
padding和slot guard；sparse因`TsmGemm`没有对应setter、pad/unpad因typed ABI无field而static-negative。
70个safe row均已有current保存板端证据；exact/observation处置仍只按各自窄证据消费，不由相邻dtype/kind外推。

Cx/NCx的共同shape族为`[2,7,9,65]`，边界族覆盖`C=63/64/65/127/129`；每个N slice、full block、
compact `C0` tail和256B physical padding使用不同poison。DataMove使用下列非对称case，所有FP16/BF16/FP32
可接受entry分别记账：

| DataMove能力 | calibration shape/关系 | 必须区分的错误 |
| --- | --- | --- |
| Mirror、Rotate90/180/270 | `[2,7,9,65]`，H/W位置编码不同 | H/W轴颠倒、只处理首N、padding被当逻辑值 |
| Transpose | header定义的固定映射使用非对称H/W/C shape，held-out更换三轴长度 | metadata-only、固定轴映射错误、Cx block步进错误 |
| NCHW↔NHWC | `[2,65,7,9] ↔ [2,7,9,65]` | 仅改shape未搬数据、C/H/W顺序错误 |
| raw Concat | C/W/H分别使用不等长两输入；C用33+32→65，H/W使用非对称切分。三者已有bounded completion/guard板端观察，catalog保持observation且production走GatherScatter；native `dims=HW`是错误指令用法，永久static-negative，不生成packet或执行case | 两输入反序、轴错误、第二输入tail/padding串扰；packet可编码不能替代合法性、可完成性和数值资格 |
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

DataMove base共享dispatcher有46个board case。除原有large transform和NCHW↔NHWC外，compiler concat按
C/W/H/HW/N五种source语义分别使用非对称两输入并由有界GatherScatter materialize；broadcast分别覆盖
scalar、C=65 channel和`W9×C65` row；Cx/NCx覆盖C=63/64/65/127/129的四向转换；gather覆盖contiguous、
1D/2D/3D holes和16385-element tail。所有case检查all-and-only logical points、完整physical span和slot
canary；instruction count是case预算而非单指令能力声明。extended默认dispatcher增加17个case：raw
Concat C/W/H、large Pad/Img2Col、MaskGather/MaskGather_bV、TensorNom、Cx/NCx materialize→CT Add、
Tensor materialize→NE identity GEMM、I8 `128B×32`/`64B×64` strided TDMA，以及FP16/BF16 raw-vs-CRT
Memset。公开DataMove opcode `121..138`均有处置：14个exact、4个observation、0 deferred；四个
observation opcode为raw Concat、TensorNom和两种MaskGather，均保留raw result、expected diff、
physical guard和completion。CT/NE/RDMA/WDMA/TDMA与Tensor/NTensor/Cx/NCx的20个组合分类为8个native
positive、3个materialize-then-consume positive和9个static-negative，未留空白。native `dims=HW` Concat
是错误指令用法，永久归入static-negative：catalog、dispatcher、CTest和runner均不得保留任何发射或重新资格
入口。C/W/H已取得raw completion和guard板端观察但尚未exact-qualified；合法source-level HW concat由
typed GatherScatter闭合，extended安全路径按各自exact/observation合同完成板端校验。

large NCx case还闭合了不能从logical bytes推导physical span的三个反例：Pad
`N2H5W7C65→N2H7W10C65`使用9728B source和19456B destination；Img2Col
`N2H9W11C65→N2H6W54C65`使用27136B source和88576B destination；TensorNom从16380B compact Tensor读取，
写17408B NCx。旧的compact或只在整份末尾对齐算法分别会少保护Pad和TensorNom的1024B physical后缀，并忽略
Img2Col每个N slice的batch padding。planner与oracle因此必须共用physical layout：logical-valid point、
C0/internal padding、batch suffix和allocation外guard分别处理。

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
overflow/Inf和NaN传播row。四个special observation分别带`+0/-0`乘法方向、正负最小/最大subnormal与
最小normal、正负max-finite overflow和`+Inf/-Inf` operand，以及正负quiet/signaling NaN和不同payload；
同一输入还保留finite control lane。每个row同时检查logical result、Cx/NCx physical span、padding和guard；只要
accumulator宽度、flush-to-zero或NaN传播仍有多个解释，就保持`board-observed/unknown`而不写成通用BF16
语义。

Conv、DepthwiseConv和BackwardConv分别建inventory，不能由ordinary Conv外推。ordinary Conv主shape采用
`N=2,H=17,W=19,I=65,O=96`、非方`Kx=3,Ky=2`、非对称stride/pad/dilation，held-out把O改为65；
Depthwise按其独立channel relation选择同级别非64整除shape，不复用ordinary Conv的O。每种kind先闭合bare
FP16/BF16，再对bias、psum、ReLU/LeakyReLU、axis scale、quant、sparse、pad/unpad做
one-factor-at-a-time。owned typed ABI能表达且range有界的option进入exact或raw observation；没有
对应setter/field、shape relation或wrapper根本不接受的组合进入`static-negative`，不能用相近Conv family
代签。

当前profile的FP16 GEMM ReLU raw完整结果逐bit等于bare baseline，8192个logical结果中3828个负值仍未
clamp；因此FP16 row只记录no-op observation，BF16因共享同一wrapper option路径且尚无独立反证也保持
observation。large ordinary Conv raw从logical element 97开始与current NCx/HWOI host expected不符，
总计21632个logical mismatch；四个已执行option的physical result又逐bit等于bare baseline。该证据不足以
在feature、weight和output physical indexing候选间唯一归因，所以ordinary Conv bare和全部option均只保留
bounded raw observation，不能作为exact Conv或fusion资格。

BackwardConv与ordinary/depthwise的transfer-shape owner不同：type-2下`AddWeight`把full weight shape写入
`tfr_1`，`AddOutput`的shape参数不再拥有该字段。因此`[1,1,64,64]` FP16 weight-gradient footprint是
8192B；旧catalog按`[1,4,4,64]` AddOutput参数只允许2048B，板端恰报告6144个越界guard byte。修正后的
catalog/probe从weight shape和Cx dtype宽度推导8192B，8192B之后的suffix guard继续严格。旧raw离线按
该边界可完整重放。修正schema的FP16/BF16各3个板端样本均完成request/result echo、8192B physical
span、suffix guard和completion，因此weight-owned footprint取得bounded board-observed资格；当前vector
没有唯一numeric golden，故不把这6个完成样本升级为通用BackwardConv exact语义。

### 5.5 Shape、资源与oracle基线

| suite | calibration shape | held-out | 目的 |
| --- | --- | --- | --- |
| CT compact vector | 8192 logical elements | 8214 elements | 排除启动开销主导并覆盖非256B tail |
| CT `VuV` | 8192与非2次幂unit | 8214与另一legal unit | 区分unit重复、full/tail |
| Cx/NCx | `[2,7,9,65]` | C=63/64/127/129族 | 覆盖block、C0 tail、N stride |
| reduce/pool/movement | 至少数千元素、非对称H/W/C | 独立axis/window/pad | 排除1x1、方阵和全零弱oracle |
| GEMM | 64×128×128 | 65×129×129、batch2 | 非平凡累加和三维tail |
| Conv | N2、17×19、3×2 kernel、I65/O96 | 独立stride/dilation、O65 | 轴、channel tail和option |
| RDMA/WDMA/TDMA | 64KiB强sentinel加1D/2D/3D holes | 独立length/offset/tail | descriptor单位、range和cache |

任何具体shape在进入board-positive前都必须由SPM/DDR planner证明输入、全部writeback、padding和前后guard不
越过当前可分配范围。shape是诊断向量，不成为compiler支持上限；更大shape由分块后的相同typed合同覆盖。

### 5.6 SPM、physical range 与复用 suite

SPM测试不能只看最终数值；每个case同时记录allocation/view的半开区间、packet inclusive end、logical
result、完整physical footprint和前后guard。测试分成下面八类：

| case类 | calibration vector | oracle、准入与禁止外推 |
| --- | --- | --- |
| capacity/reservation | 从profile静态可分配区推导最高合法256B block、跨界1 block和保留区相邻block | 合法边界做round-trip；越界和保留区只做planner/verifier negative，不发板端packet |
| alignment | 同一4KiB/64KiB payload放在256B对齐base；另以`base+64/+128/+192`和`length=128/384`建立有界非preferred geometry observation | 256B只作为preferred alignment；五个observation各有独立payload expected、完整round-trip和前后guard，实卡结果只关闭对应geometry，不外推所有byte alignment |
| relative-offset sweep | 固定engine、length和pattern，只改变A/B相对offset：0、256、512、1KiB、2KiB、4KiB、8KiB、16KiB、32KiB、64KiB及独立held-out offset | 完整output/guard先通过；PMU只形成经验conflict class，不把64KiB或某个峰值命名成bank周期 |
| address relation | exact、half-partial、adjacent、far-disjoint和strided envelope；分别构造RAW/WAR/WAW/RAR | 全部有界descriptor都执行composition/guard/completion观察；disjoint正overlap只决定是否开放compiler并行资格 |
| physical layout | Tensor、Cx、NCx的full block、C0 tail、N-slice步进和每段padding poison | logical point、padding和guard分别比较；不能把padding写入当作logical正确 |
| lifetime/reuse | producer完成前后复用同一slot、双slot轮转、奇偶iteration和最后一次tail | pure same-worker NCC复用由typed RAW/WAR/WAW edge保持issue order，跨Kcore/DTE/worker/terminal domain才要求matching drain/token；缺dependency/boundary completion、悬空view和capacity不足做IR/verifier negative |
| engine access | CT read/write、NE双读单写、RDMA DDR→SPM、WDMA SPM→DDR、TDMA SPM→SPM | 五类均解析到memory-descriptor concrete payload；exact result、range、count和guard先于PMU解释 |
| bank/engine pair | 五类engine的10个无序pair，每个pair使用`4352/8192/65536`三个不重叠相对offset并分别登记serial/window control | 每格3次重复并已闭合exact结果/guard/raw delta；这些短/旧cell只作地址与size对照，overlap能力以expanded sustained FU-union矩阵为准，不得用offset-only round-trip冒充bank conflict |

SPM单engine基础矩阵覆盖CT read/write、NE双读单写、RDMA DDR→SPM、WDMA SPM→DDR和TDMA
SPM→SPM；每类先做contiguous，再做其ABI支持的stride/tail。cross-engine offset sweep保持两条workload、
issue count和地址以外的变量不变；有界correctness case不等待正overlap资格。所有slot由planner证明落在
当前profile可分配区，测试自身不探测未知保留区。

当前`wafer_spm_calibration_catalog.py`有146个typed row：24个本地board、10个static-negative、112个
指向具体DataMove/memory-descriptor case的delegated row和0 deferred。本地board部分由4个capacity boundary、
11个相对offset `0/256/512/1024/2048/4096/8192/16384/32768/65536/65792`、1个64KiB transfer以及
single iteration、双slot偶数4轮、双slot奇数5轮三个lifetime case组成；使用非零byte pattern检查
RDMA→SPM→WDMA exact、SPM前后guard、DDR canary和instruction count。另5个本地case以同样的强oracle
观察`base+64/+128/+192`和`length=128/384`；它们已通过板端完整round-trip、guard和completion，只是
对应geometry的bounded legality observation，独立expected
不等于提前supported。10个negative只覆盖真实reservation/range非法及缺wait、悬空view、capacity不足的
lifetime；112个delegated row由20个address relation、Tensor/Cx/NCx五个physical layout、五类engine access、
10个engine pair × 三个relative offset × serial/window，以及代表性CT+RDMA的16个新增bank-phase row和
6个新增base-residue row组成。
每项都解析到concrete bounded request，而非只贴标签。offset性能与bank class必须等实卡原始PMU，不能由
no-card或静态地址公式预判。保存的256B CT+RDMA phase/offset raw中，所有window样本
`CT_exec + RDMA_exec - FU_full_exec≈0`且blocking为0，说明短请求没有取得overlap；这与16KiB expanded
CT+RDMA三次稳定正excess形成size effect。`base mod 256 = 0/64/128/192`的常态约为CT `47--50`、
RDMA `233--235` cycles，64/128只出现偶发长尾，192三样本total约285；offset `4096..5888`多数total
`280--282`，夹杂单样本长尾。offset 6144的serial为`413/364/307`、window为`360/328/307`，只记为
repeatable slower observation；这些数据没有稳定周期或可消费phase class。planner因此不做bank coloring，
cost不写固定phase penalty，需更大payload和held-out重复后才可能建立地址性能类。

DDR/cache可见性使用另一份共享package和16KiB payload，共四个独立方向、四次单invocation：
host H2D→Kcore在同一次device invocation中执行matching invalidate后要求exact；Kcore store→NCC RDMA执行
clean+fence后要求RDMA exact；NCC WDMA→Kcore执行invalidate后要求Kcore exact；WDMA→host D2H要求matching
completion后完整readback exact。SPM/output guard、RDMA/WDMA count和cache-control mask同时作为强gate。
跨两个独立`wafer-run`无法保留同一runtime allocation/cache状态，因此不会把两次进程输出拼成
stale/invalidate因果oracle。当前一次性BoardRuntime在每次launch后释放allocation，该状态不可达，不阻塞
现有compiler/runtime。未来支持模型常驻或persistent session、允许Host跨launch覆写同一输入/workspace
allocation时，重新启用该条件性case：先让Kcore以值A prime cache，Host原地址写入B，再比较invalidate前后
读取，决定`HostWrite -> DeviceKcoreRead`所有权转换是否必须插入invalidate/fence。

既有cache catalog另有54个DDR bank case：RDMA/RDMA、WDMA/WDMA、RDMA/WDMA三种pair各使用
`0/256/512/1024/2048/4096/8192/16384/32768`九个same-allocation相对offset，并分别执行serial/window
control，payload固定4KiB且逐case检查output/guard/completion。它们用于实卡聚类bank/conflict趋势，不预设
bank公式。普通pair路径已改为host payload经RDMA seed、被测pair、完整slot WDMA回读和一次terminal
completion；serial control只保留定义schedule的中间completion，cases 0--3的真实Kcore/cache边界继续使用
mapped-SPM。独立memory-descriptor package当前提供155个case：14个DMA/DDR exact覆盖source/destination offset、
4097/8193/16385及65535 tail、64KiB contiguous、1D/2D/3D holes、255/256/257与4095/4096/4097
default-burst boundary；5个single-engine access exact；20个RAW/WAR/WAW/RAR ×
exact/partial/adjacent/disjoint/strided observation；60个engine-pair exact覆盖10个pair的
`4352/8192/65536`相对offset及serial/window；另有代表性CT+RDMA的
`4096 + k * 256 (k=0..8)`相位扫描和固定8192B间距下`base mod 256 = 0/64/128/192`扫描。所有case包含
descriptor echo、compact/envelope range、SPM/output guard、expected instruction count和PMU raw delta。
owned CRT没有显式burst配置field，因此只跨默认boundary，`ddr-configurable-burst-knob`为static-negative。
全155个case已在current `RESOURCE_BYTES=2MiB` schema下重新上板，current validator对result、descriptor
echo、range、guard、instruction count和completion全部通过；旧262144B artifact不再承担当前资格证明。
54个纯NCC cache pair使用独立package/schema并有板端correctness/guard/completion结果。新增34个
steady-state/cross-worker/dependency pair也已全部通过strict result、guard、count和completion gate。每个
pair/offset/schedule cell逐样本验证exact结果后发布serial/window summary、per-engine execution和FU
full-execution delta。十个disjoint sustained cell在三次重复中同时取得正`Ea+Eb-FU`与较低plan，可作为
窄overlap capability；六个4KiB RDMA+WDMA cell的excess恒为0但plan降低，只形成issue/wait收益。
这些结果仍不能命名bank/color class或外推未测transfer/shape。

新增16-rank DDR tile/allocation probe以256B请求形成
`tile × actual allocation base × offset × RDMA/WDMA`矩阵：每rank有两组独立input/output endpoint，
offset为`0/256/512/1K/2K/4K/8K/16K/32K/64K/128K/256K/512K/1M`，每格3个样本。
full-card launch内部每个phase只有一个rank发NCC，其余rank只参与barrier；结果记录logical/physical tile、
四个实际64-bit allocation base、instruction/blocking/engine/FU/window及issue-to-completion cycles，并以
engine execution作为主速度指标。runtime没有DDR bank/controller placement selector，rank之间也不能共享
manifest resource，因此该probe只发布地址类别和延迟矩阵，不预设bank位、controller或hop公式；全rank并发
contention必须另列，不能与单active-rank路径结果混合。

该16-rank probe已在全部rank完成actual 64-bit allocation base、上述相对offset、RDMA/WDMA round-trip、
guard和raw timing记录，因此compiler可以消费“manifest allocation内相对地址可达”这一窄结论；结果不提供
physical bank/controller/hop命名。独立sparse probe又在单个40GiB compiler-managed workspace内完成
`0/4/16/32/38GiB`及`allocation_end-768B`六个窗口的exact round-trip和guard，闭合已验证allocator状态下的
高位相对地址与allocation-end边界。该结果不外推64GiB、其它分配顺序/碎片状态或physical bank。

### 5.7 Synchronization、visibility 与 completion suite

同步测试按completion domain分开，不能用最后一次全局drain把错误边界掩盖掉。每个正向case在“目标同步
刚返回、safety drain之前”读取区分性结果或状态；safety drain只负责进程恢复。

| domain/case | producer→consumer与区分向量 | 成功oracle |
| --- | --- | --- |
| same-worker NCC dependency | RDMA→CT、CT/NE→WDMA、TDMA→CT/NE及同engine复用；RAW/WAR/WAW按指定issue order连续发射，不插中间wait | 只在序列离开NCC completion domain时drain并检查queue count、逐段composition、result和guard；证明busytable落实issue order，不删除IR edge |
| worker-specific wait | worker0/1/2各发独立CT pattern，分别调用matching `bywork`；现有default/local-fence对照只保留已取得的自然排空事实，真正pending exclusion的18个case已按第5.10节接入runner待执行 | 只允许对应worker结果可见；default/local-fence scope无法区分时保持`unknown` |
| cross-worker join | w0/w1/w2 disjoint输出，单worker join、两worker mask和三worker mask；mask外worker保持pending的12个include/exclude A/B case已按第5.10节接入runner待执行 | join mask覆盖的worker全部完成，未覆盖worker不被伪称完成；最终safety join后全部guard正确 |
| NCC completion-domain exit | NCC→Kcore、NCC→Direct DTE、structured barrier/backedge及terminal/host publication | 在latest-legal boundary使用matching local drain并合并相邻boundary；boundary返回后立即验证consumer/terminal，不能把最终safety drain代签 |
| Kcore/cache visibility | host H2D→Kcore read、Kcore store→NCC read、NCC write→Kcore read、WDMA→host D2H | invalidate/clean/fence前后对照、DMA round-trip和host全量expected分别闭合，不能互相替代 |
| Direct DTE | NCC producer→local drain→DTE send；DTE receive→event wait→NCC consumer；source/destination复用在event后；同步错误路径独立执行 | 正向要求每rank source/receive/compute guard、DDR output、event、terminal与participant正确；错误观察要求shadow status保留TRANSPORT_ERROR且正常runtime terminal/cleanup |
| multi-tile arrival | full-card两个epoch反向错峰；安全subgroup使用显式rank group和两次复用 | 每个participant得到rank-specific marker，0 mismatch/crosstalk；缺participant/错坐标不上板 |
| runtime publication | device terminal→runtime completion→D2H→cleanup；失败路径使用外层timeout | terminal schema、完整output、cleanup和下一次heartbeat；管理面idle不替代execution heartbeat |

跨worker同地址无ordered producer、错误coordinate/participant和host readback早于terminal继续在IR/verifier、
package validator或host gate拒绝。DTE source或destination提前复用、invalid FSM与wait未知event已有
CRT同步错误返回和正常cleanup owner；其中source提前复用、destination提前复用和invalid FSM已经由
modes 7、12、8取得16-rank有界board observation，wait未知event的mode 9也取得相同的预期错误和clean
completion证据。modes 10--11的raw async serial/window各3 samples × 16 ranks通过结果、返回码、guard和
cleanup gate。receiver未prepare的独立mode 13已实测为host timeout并污染runtime context，production仍不得
发板；raw async window没有可信共同time base，
因此不由correctness推出DTE/CT时间重叠。

transport PMU在Direct DTE四种基础有序交互、event后安全复用和两destination broadcast上增加
16/32/64/256/4096B sweep，共30个full-card case。每次只改变active
payload length，固定4KiB logical slot的未使用后缀保持poison，并继续检查source/receive/compute SPM guard、
DDR guard、terminal和NCC instruction count。DTE channel0/1 transfer/execution与SPM T2/T3 port8六个
split counter逐rank保存raw modulo-2^64 delta，host只报告payload趋势和可能的整数scale，不自行命名单位。
TMNOC当前只有base address而没有decoded只读offset，因此只做header-backed static-negative，不进入上板读取。

该30项已全部完成板端expected、status、guarded SPM readback和cleanup gate。DTE channel0 transfer raw
median在每个mode内随payload严格递增，modes 1--4呈scale 1、modes 5--6呈scale 2；execution raw非单调，
SPM PMU没有enable。故compiler只可消费DTE transfer的profile内相对payload相关性，不能把execution raw或
SPM样本写成latency/cost，也不能由相对scale命名counter单位。modes 7--12不属于这30项PMU sweep，但已
分别通过各自的同步错误或raw async correctness oracle。

### 5.8 Parallel issue、engine pair 与 multi-rank suite

并行测试先证明correctness，再判断是否存在可观测overlap。负overlap样本只说明当前workload/profile没有形成
窗口，不等于硬件不支持并行。

| case类 | 覆盖矩阵 | control与oracle |
| --- | --- | --- |
| single-engine multi-issue | CT/NE/RDMA/WDMA普通1/2/4，TDMA 1/2；worker1/2只做代表routing | 独立地址/pattern、精确instruction count、全部result/guard；documented `D`与tight `D+1`沿用已闭合manual gate，不再加深 |
| 10个cross-engine pair | CT/NE/RDMA/WDMA/TDMA两两组合；每个pair两个issue order；含TDMA只做r2，其它先r2再r4 | 同一payload的显式serial control与window各至少3次；`Ea + Eb - FU` median稳定为正才取得overlap资格 |
| backlog形成 | compute使用足够大的CT/NE shape，movement使用至少64KiB强sentinel；prebuilt packet与wrapper路径分开 | 证明两engine各自count/result不变，避免4KiB短workload在构包间隙自然排空 |
| dependency observation | RAW/WAR/WAW/RAR × exact/partial/adjacent/disjoint/strided-envelope全部有界case | 实际packet/descriptor range、逐段composition/raw result、未选operand guard和completion共同闭合；PMU只决定是否升级并行资格 |
| multi-worker | w0/w1/w2 disjoint单发、两两window和三worker window；同地址只跑显式ordered正向 | matching join、各worker count/result/guard；不由wall time宣称并行 |
| DTE/NCC interaction | producer→DTE、DTE→consumer、disjoint local-first/DTE-first；broadcast每个destination独立pattern | correctness/completion先闭合；NCC与DTE无共同可信time base时overlap保持`unknown` |
| software-pipeline vertical | 双SPM slot的movement[n+1]、compute[n]、writeback[n-1]，含prologue/steady/epilogue、奇偶iteration和single-iteration identity | baseline/optimized使用同一source与package gate；只有结构窗口、完整expected和可信板端对照都通过才进入production |

parallel performance row还必须记录serial/window原始PMU、per-engine execution、global union、blocking、
issue order、repeat ordinal和profile identity。host wall time只用于发现launch异常；没有稳定正overlap或PMU
measurement basis时，不形成cost常数或scheduler capability。

### 5.9 当前板端执行状态与未闭合边界

实现阶段按suite生成共享catalog/oracle和共享device dispatcher；同一ABI/resource class只编译、链接和打包
一次，由typed request选择case，不为每个row重复编译。所有static-negative、host oracle、device compile/link
和no-card package在进入实卡环境前一次性闭合，因此实卡时间主要用于execution/readback，不随case数线性重复
编译。

本地CT/NE/DataMove、layout和descriptor资格默认只在一个known-good tile的worker0运行；worker routing只用
代表case覆盖worker1/2。只有Direct DTE、multi-tile arrival、collective或明确rank-dependent语义才启动多rank，
不能为了“每rank workload相同”把同一个local instruction oracle在所有rank重复16遍。

每个新执行会话只运行一次ordinary Add建立资格；合同外unit 32/37 `VuVLoop`和其它excluded packet不再发板。
截至当前，串行清单中已经实际完成并形成结论的部分如下；未列出的case不因catalog/离线准备完成而视为已上板：

1. NCC→Kcore depth-4 no-local-wait与local-wait均通过marker/result/guard；两者在snapshot前都已自然完成，
   因而没有区分local wait必要性或scope，NCC→Kcore出口继续保留matching completion。
2. 3个standalone DMA strict compact-SPM strided case与36个NCC strided dependency case均通过；
   stride只作用DDR endpoint，SPM endpoint按`transfer_bytes`连续，RAW/WAR/WAW仍保留IR edge和issue order。
3. DDR tile-offset probe完成16 rank、两组actual allocation、RDMA/WDMA、14个相对offset的2688次guarded
   transfer；40GiB sparse probe的六个高位窗口也全部exact。它们闭合当前allocation内的i64相对寻址，
   不形成physical bank/controller/hop结论。
4. 34个expanded pair全部通过strict correctness。10个sustained disjoint cell取得三次重复正FU excess和
   更低window plan，可进入窄profile/shape/schedule overlap capability；6个4KiB RDMA+WDMA
   relation/order cell的FU excess恒为0，只形成issue/wait收益。
5. 清单末尾ordinary Add逐bit exact，clean execution会话正常收口。
6. CT会话资格的ordinary Add heartbeat（CTest 125）以3.10s通过。CTest 106 `ct-capability`以32.77s
   通过，CTest 107 `qualification-regression`以4.94s通过。CTest 108首次执行时case id
   `10000..10054`共55项依次通过，旧op009 f16 tail geometry随后在byte 449出现数值不一致。修正为
   `full=8000, elem=320, unit=64, full_unit=1600`后再次执行CTest 108，`10000..10054`再次全部通过；
   id `10055`最初在host解析阶段报告record/execute mirror失败，runner按first-failure合同停止；根因是
   corrected geometry没有同步进入device decoder白名单。修复白名单后，同一case以
   `full=8000, elem=320, unit=64, full_unit=1600`重新上板，`execute=1`且完整结果
   `mismatches=0`，因此该geometry取得板端exact结果。续跑的id `10056..10186`也全部通过各自板端oracle。
   id `10187`与`10199` bool-tail只因旧host oracle逐字节比较无效高2 bit canary而曾停止；修正为只比较
   logical-valid bit后，两项均在实板取得`execute=1/mismatches=0`。随后`10200..10467`全部通过。
   id `10468` `LogicOp_V_V_not f16 main`按旧bit-pattern oracle最初报告device output byte 0为`0x00`、
   expected为`0x20`；该次mismatch非timeout且lifecycle正常，后续审计已确认它属于oracle语义错误而非
   硬件数值失败。聚合批次随后已执行到最终id `12026`，所有board
   launch/lifecycle正常；末尾ordinary Add heartbeat（CTest 125）以1.58s通过，证明该批没有留下可由
   heartbeat观察到的卡状态污染。初次按REC sample回放曾得到`PASS=590, FAIL=63, MISSING=0`；审计确认
   63项均是host oracle语义错误，不是硬件数值失败：op78--87的value logic对每个数值取truth后执行
   NOT/AND/OR/XOR并写回数值0/1，而非对FP bit-pattern做位运算；op92--94的bitpacked BOOL VuV按
   byte-addressed storage将logical 37-bit unit扩成40-bit physical RHS周期，末尾不足完整物理周期的RHS
   读取为0。修正后在`board-ct-vector-corrected63-v2`重跑id
   `10468..10527,10553,10559,10565`，63项均`execute_result=1, mismatches=0`且process exit 0。
   因此current catalog最终为`PASS=653, FAIL=0, MISSING=0`；旧actual/expected mismatch只作oracle修正
   历史，不再作为unsupported或硬件失败证据。
7. 修复后的DTE独立64元素production baseline在16 ranks exact；modes 1--6 ×
   `16/32/64/256/4096B`共30个full-card case全部通过expected、status、guarded SPM readback和正常cleanup。
   DTE channel0 transfer raw median随payload严格递增，modes 1--4为相对scale 1、modes 5--6为scale 2；
   execution raw非单调；这批既有DTE样本没有启用SPM PMU，因此其中SPM raw不能使用。随后modes 7、8、12
   的16B错误观察均在16 ranks得到预期
   transport error，再以clean success、exact guard和正常cleanup收尾。mode 9也取得同样的16-rank预期
   error/clean-success结果。modes 10--11的64KiB raw async serial control及在`send_async`与
   `wait_done`间插CT的window均完成3 samples × 16 ranks，结果exact，`send_async`、`wait_done`、
   `release`返回码全0，guard exact并正常cleanup。至此modes 1--12均通过各自board oracle；该结果仍不把
   raw async window的correctness提升成DTE/CT时间重叠证据。

此前已经闭合的合同内`VuVLoop unit=64`两个geometry及前后Add、BackwardConv FP16/BF16各3个样本、
40个4352/65536 pair、26个phase/alignment case、54个纯NCC cache pair和5个nonpreferred SPM geometry
同样已有对应行为结论，不重复执行。

一批case只有在完整expected、physical span、padding/guard、terminal、issue count和cleanup全部通过时才记
`board-observed`；calibration与独立held-out均通过后才可记`calibrated`；production source-to-package纵向
再通过后才可记`supported`。原始日志按profile和case row关联保存，不把一次板端输出直接改写成跨profile
compiler常数。

既有机器矩阵覆盖28个导航域和125个叶子：74个`board-positive`、24个`board-observation`、
6个`delegated-positive`、17个`static-negative`及4个`isolated-deferred`。此前以CMake注册的board
calibration/probe、matrix leaf绑定和raw artifact交叉审计得到的“safe/default漏跑为0”，只证明这125个
既有基线叶子，不覆盖本轮新建的pending hardware-behavior inventory，也不能把`board-ready`写成
`board-observed`。memory descriptor 155项已按current 2MiB schema全量重放并由current validator通过，
SPM 112个delegated row也不再存在旧memory schema证据缺口。本轮此前还闭合Pool当前24项、
DataMove extended安全路径、NCC contiguous/strided hazard、wait/producer-consumer、large backlog、双slot、
active-observation、subset-join、depth-4 A/B、DDR tile/sparse及expanded pair；raw N/HWC Reduce已
fail-closed。

新pending inventory把每个可安全观测的未闭合项绑定到真实catalog对象、device request/record、host
oracle、no-card CTest、board CTest及runner batch；不可观测项则绑定到实际执行的typed host rejection，
没有“只有文档记录”的第三种状态。当前新case均尚未上板，queue-full/backpressure、default/local-fence
跨worker scope、subset exclusion、worker placement/progress、SPM conflict proxy、DDR active-rank
contention、collective algorithm/traffic、Direct-DTE raw multidestination/fan-in、ArgMin tie/NaN、
Unpool repeated-overlap、single-engine rate和engine-pair balance仍不能写成硬件事实。physical queue
resident数、SPM/DDR bank identity、absolute per-worker timestamp、physical arbiter policy、8/15源DTE
fan-in、device phase、physical route、cross-card transport及production three-stage producer缺口由各自
typed gate保守关闭。历史op013→op014不进入当前安全队列；native `dims=HW` Concat是错误指令用法，
永久static-negative并从全部测试面删除。准备完成和局部板端证据都不
等于Q37完成，也不提前授权production scheduling变化。

板端统一入口为：

```bash
python3 tools/run_hardware_calibration.py --build-dir <board-build> --list
WAFER_EXECUTE_HARDWARE_TESTS=1 \
  python3 tools/run_hardware_calibration.py --build-dir <board-build> --execute
```

runner先审计已注册的`board;hardware` CTest与manifest完全一致，再做一次串行增量构建；随后逐CTest单进程
执行，首个failure、timeout或skip立即停止，不自动retry/reset/power。每项保存log和JUnit，session级
`session.json`增量记录未执行项与失败位置；每次`--execute`生成新的32位hex session id并只向该轮build/CTest
子进程传递。跨CTest合并的engine/NE archive必须同时匹配该session id、target profile、完整runtime launch contract、device id、
runtime version/device name/PCI/tile count/runtime library SHA；旧轮或其它profile的archive在发板前fail closed。
首尾各执行一次known-good Add heartbeat。
已有证据不需重放时使用可重复的`--step`或`--batch`冻结精确子集。本批全部待板端case的单一机器可审计入口为：

```bash
python3 tools/run_hardware_calibration.py --build-dir <board-build> --list \
  --batch pending-hardware-calibration
WAFER_EXECUTE_HARDWARE_TESTS=1 \
  python3 tools/run_hardware_calibration.py --build-dir <board-build> --execute \
  --batch pending-hardware-calibration
```

该master batch由pending inventory反向绑定15个family的全部210个互异board CTest和1231个pending target
cell，并校验每个CTest恰有runner step。归一化后的pending target参数无重复；同invocation中为漂移、
session或危险case健康检查而执行的既有baseline不计入1231个pending target，例如SPM equivalence的
`8 baseline + 88 held-out`仍作为完整matched invocation执行；
`compiler-optimization-paired`、`collective-characterization`、`collective-traffic-behavior`、
`engine-pipeline-characterization`和`pending-execution-boundaries`只是可重复选择的语义子批次，彼此存在
有意重叠，不应在同一命令中机械叠加。选择master batch后自动加首尾heartbeat；每个group CTest在单一
进程中按冻结顺序执行其matched cells，不把381个engine cell、
44个worker cell、45个active-rank cell或16个SPM sustained cell拆成并发板测；25个Direct-DTE raw
case同样保持逐CTest串行。native `dims=HW` Concat
不属于合法pending inventory，且不存在任何board case或runner step。旧instruction-family、
memory-descriptor和SPM全量重放只保留为`*-full-replay`显式步骤，不进入本批待执行路径。

### 5.10 区分实验执行台账

#### 2026-07-27 板端批次

本轮绑定TX8110、16 tile、runtime version 1300及同一runtime library identity执行。当前合法pending
inventory共有15个family、210个互异board CTest和1231个target cell；本轮执行台账仅覆盖远程扩展前的
179个合法CTest，按互异CTest记账为：

- `PASS=79`；
- ordinary failure `=9`，均在结果/oracle边界干净退出；
- `UNEXECUTED=91`；
- 10次runner invocation的initial known-good Add heartbeat全部通过。

本次远程新增31个CTest、37个target cell尚未进入该轮执行；因此在当前210项清单中，这一检查点共有
79项通过、9项普通失败和122项未执行。

79个通过项由以下互异CTest组成：production reciprocal 1项，AllGather Direct/Ring三种payload 3项，
AllToAll/Permute traffic 11项，queue saturation 30项，worker subset 12项，worker
placement/progress 8项，SPM equivalence/sustained pilot 6项，DDR active-rank contiguous 6项，
DDR conflict-equivalence 2项。它们只形成各自result、guard、count、completion及已声明proxy合同的板端
证据；未满足promotion条件的cost、bank/color、resident和scope结论仍保持Unknown。

9个ordinary failure按根因边界分类如下：

1. production common-factor与resident-fanout-share各1项：前者在winner前的INT8 CT算术基线即出现
   numeric mismatch，后者仅在`Neg(+0)`的signed-zero位上不一致；二者共同暴露“静态可编码”未连接
   profile-scoped numeric admission，不能据此把优化candidate或精确signed-zero语义准入production。
2. ReduceScatter Direct 256B的INT8 Add结果不符；同批AllGather transport 3项通过，因此该项保守归入
   未取得资格的INT8 CT算术域，不归因为DTE transport。
3. single-engine seed在RDMA起始处返回`SEED_FAILED`；host catalog把`Fmt_INT8`误写成值8，而
   version-matched ABI中INT8为0、值8为UINT8。这是测试枚举事实源错误，不是engine rate证据。
4. worker wait-scope 1项因后续repeat在pre-wait snapshot前自然排空而被逐样本oracle判失败；已有至少一个
   repeat在目标pending时完成区分。修复方向是聚合distinguishing repeats，不能把自然排空当硬件失败或
   扩大default wait scope。
5. ArgMin tie 1项在第二个区分输入上重复了前一launch的value/index组合；coherence失败保持有效，但当前
   fixture尚不能区分result writeback stale与input consumption stale，不能放宽tie资格。
6. Unpool index repeated-overlap 1项得到全零writeback；host bounded-subset oracle遗漏drop-all/zero-fill
   baseline，故该case不能承担collision winner区分，也不能由此升级ordinary indexed Unpool。
7. DDR active-rank RDMA-only stride及RDMA+WDMA stride各1项在同一host假定hole处失败；descriptor实际把
   DDR strided endpoint gather到compact SPM，按DDR envelope与SPM compact footprint分域重算后全span
   无差异。这两项是fixture/oracle复用了错误布局codec，不是硬件descriptor错误。

此前遗留的native `dims=HW` Concat执行case不属于合法能力：它是错误指令用法，本应永久
static-negative并从测试面删除，却被错误纳入旧批次并在发射后触发completion timeout。该项不计入上述
旧批次179个合法CTest、9个ordinary failure或1194个target cell。timeout后立即停止全部板测，没有执行terminal
heartbeat，也没有retry、reset或power；只读管理面idle不恢复execution资格。当前catalog、dispatcher、
CTest、inventory和runner均不得存在该指令的任何执行入口；合法source-level HW concat继续由typed
GatherScatter materialize。

旧批次其余91个合法CTest及本次远程新增31个CTest均未执行，必须在新的合格会话中按当前inventory继续，
不能把catalog/no-card准备记成板端
证据。NCC wait-scope剩余项、collective剩余算法、ArgMin NaN、Unpool mask、engine其余组和DDR
WDMA-only stride等均仍按各自保守gate处理。runner继续遵守单进程串行、bounded timeout、普通failure
停止当前invocation并转入host归因、任一timeout或设备异常停止全部板测且不retry/reset/power。

本节其余段落描述仍存在的case资产与promotion合同。NCC侧30个queue saturation、18个wait-scope和
12个subset-scope case已经进入typed request、device dispatcher、pre-wait/boundary/final record及host
oracle；SPM单tile侧复用8个既有baseline control并新增88个base-translation、reciprocal-order和第二
workload schedule模板，组合成48个paired coordinate。每个coordinate用4次same-invocation执行
serial/window双row，共384个pair-only measurement row。DDR rank-one侧新增108个同invocation坐标batch，
每批包含serial/window、双issue order和四次重复，共1728个pair-only measurement row；physical-tile侧
新增384个跨16 rank的RDMA/RDMA同allocation坐标组，每个launch共3072个paired measurement row；
activation要求forward/reverse rank order两次成对launch，因此最小执行6144 row，并由显式selector与旧
tile-offset runner隔离。worker characterization的44个case按14个matched group执行：fixed-total
placement 14、bounded progress 18、逐worker low/high outstanding 12；SPM sustained pilot的16个cell按
4个matched group执行；DDR active-rank的45个case按9个五点matched group执行；Direct-DTE新增25个逐case项：
四源fan-in 1项，以及raw broadcast/scatter/shuffle × fanout 2/4/8/15 × adjacent/interleaved共24项；
ArgMin tie/NaN与Unpool index/mask repeated-overlap各有2个逐case项。对应package/ELF/wafer-run、强oracle、
no-card/host contract、board CTest和runner step的实际执行状态以上述本轮台账为准。

master runner对所有带`pending`标签的case归档source MLIR/meta、最终linked module、manifest、JSON/raw证据
及实际compiler/runtime/objdump/clang路径与SHA-256，不再按family名称漏掉NE、memory或numeric case。
DDR active-rank每次2MiB资源先做完整逐字节oracle，再以同目录临时文件、`fsync`和原子replace保留
request/record、完整resource SHA-256和parsed PMU；只有archive发布成功后才删除已验证的大型临时resource，
避免9组执行和runner复制形成约45GiB峰值。archive发布或验证失败时保留raw；任何情况下都不允许用hash代替
执行时的完整比较。

真正不能由当前ABI观测的边界不伪造board case：active resident数量、
physical SPM bank/color、physical DDR bank/controller、absolute per-worker completion timestamp和physical
arbiter policy均有具体typed host gate；它们没有board CTest，也不会进入runner。只有version-matched
owner-backed只读观测面或真实production producer出现后，才能把对应host blocker替换为board-ready case。

这些实验的artifact边界固定为：消费current profile identity、已验证typed packet/descriptor和owned
SPM/DDR range；产出raw request、boundary/final record、result/guard、PMU/CSR及profile-scoped行为结论。
它们不产出新的IR、ABI、side table或scheduler hint。只有满足各自promotion gate的结果才能进入cost或
completion capability；未执行、自然排空、观测字段缺失或结果相互矛盾都保持现有fail-closed处理。

#### Queue saturation response

- 语义key为`queue-saturation-response`。要区分的不是“总共能否完成`D+1`条”，而是
  `D+1`提交时是否出现可复现的queue-full/backpressure响应；resident count是更强问题，不能继续从
  `task_done`反推。
- 每类engine使用`short/sustained × D-1/D/D+1`六格factorial control。`D-1`是确认阈值信号不会在
  documented boundary之前出现的负控制，`D`是边界控制，`D+1`才是backpressure candidate。
  packet全部预构并释放builder，使用互不重叠的owned range；同一short/sustained pair只改变单条工作量，
  同一`D/D+1` pair只增加最后一条issue。紧邻issue之间只允许`rdcycle`，不读MMIO、不做oracle、不wait；
  最后一条issue后每个worker只读一次control，再进入matching requested wait和safety drain。禁止发送
  `D+2`或更深overflow。
- 每格至少保留三次独立样本和首尾ordinary Add。强oracle包括packet/request echo、目标worker/engine
  instruction delta、每条独立result、双侧guard、final completion、per-issue cycle、blocking delta及
  stable PMU snapshot；`execute_rc`不能单独判成功。
- 只有`sustained-D+1`在全部calibration和held-out样本中出现控制组没有的稳定blocking或last-issue
  threshold信号，且结果/count/guard仍exact，才可记录窄`backpressure-observed`行为。即使该门禁通过，
  也不能得到resident数量或直接令`pipeline window=D`；后者必须另有version-matched、owner-backed
  queue head/tail/occupancy只读语义。没有该观测依据时，本family可以关闭full-response问题，但
  resident count继续`unknown`。
- `queue-active-resident-count-unobservable`不是文档占位：host contract审计current NCC record确实没有
  head/tail/occupancy/resident字段，并要求任何resident request在package准备前抛出typed rejection；
  safe executable alternative明确指向上述30个response case。静态depth、task-done、完成数和host elapsed
  均被列为禁止surrogate，因此该项不生成board CTest。

#### Worker wait scope pending exclusion

- 语义key为`worker-wait-scope-exclusion`。分别让worker0/1/2成为target，并以NE和RDMA两种
  result-producing sustained backlog做held-out；另一个轮转worker先发短marker。使用tight submission并
  推迟逐packet观察。requested wait前的唯一control snapshot必须证明target仍pending，否则该样本无区分力，
  不计入scope结论。
- 每个`target worker × target engine`的A/B/C复用完全相同的packet、地址、issue order和seed，只改变wait
  kind：default `TsmWaitfinish()`、matching `bywork(target)`和local fence。wait刚返回就读取全部worker
  control、target completion marker及boundary result，然后才对所有participant执行matching safety drain和final
  exact oracle。
- `bywork(target)`必须在boundary完成target，作为正控制。只有default或local-fence返回后仍稳定观察到
  target pending，才能证明对应primitive不覆盖target；若target已完成，只能归类为
  `covered-or-naturally-drained`，不得据此扩大scope。若在安全资源和timeout内无法让pre-wait snapshot
  捕获pending，则本family继续`pending`，不通过无限增加workload追求观察。

#### Worker subset join pending exclusion

- 语义key为`worker-subset-join-exclusion`。分别让w0/w1/w2中的一个成为最后提交的NE或RDMA
  sustained backlog，其余worker使用短marker；每个`目标worker × target engine`构造一对只改变join mask
  的请求：control包含该worker，candidate排除该worker。
- join前snapshot必须证明目标worker pending。control在boundary必须完成目标worker；candidate只有在
  mask内worker全部完成、目标worker仍pending时才证明subset exclusion。随后统一safety join并验证三worker
  的instruction count、完整result和guard。两个请求若都在boundary自然排空，则只保留included-worker
  正向结论，mask排除性继续`unknown`。
- 该family只验证disjoint输出和completion scope；跨worker同地址、仲裁公平性或无ordered producer的
  hazard仍不进入正向板测，也不能由subset结果外推。

#### Worker placement、bounded progress与outstanding区分

- 真实worker characterization共有44个case、14个matched group：14个CT/RDMA fixed-total
  单/双/三worker placement，18个NE/RDMA backlog-only、sentinel-only和concurrent progress control，
  以及12个CT/RDMA × worker × low-2/high-6 outstanding区分case。device protocol把issue ordinal与
  physical SPM slot分开；sentinel-only与concurrent使用同一个physical sentinel slot，backlog使用独立slot，
  避免把地址差异误当成progress差异。
- observer boundary必须在target safety drain前保存，并将target pending字段与同一时刻的raw `CONTROL`
  交叉核对。随后matching join的返回码必须为成功，post-join `CONTROL`必须idle；device cycle deadline、
  whole-output tail canary、完整result/guard、逐workerinstruction/blocking和global PMU全部是强gate。
  partial/failed join或超时不能由最终cleanup掩盖。
- partial issue失败保留rc==1的accepted count，同时以包含失败issue在内的全部attempted worker作为保守
  cleanup participant；observation deadline失败同样保留primary status。两者最多进入一次独立bounded
  `CONTROL` cleanup，不调用无界matching wait；cleanup deadline失败即独立标记poison并停批，不二次
  poll/wait。host拒绝任何accepted/pending/mask/final-control/cleanup状态不一致的failure record。
- 每个group三次串行执行，最后matching safety drain并检查terminal/cleanup。
- 这些case只允许形成current profile下的fixed-total placement差异与bounded observer progress，不从
  全局cycle推导每个worker的绝对完成时刻，也不命名硬件仲裁算法。
  `worker-absolute-per-worker-completion-cycle`与`worker-physical-arbiter-policy`两个typed boundary会在host
  准备阶段拒绝对应字段；只有owner-backed timestamp/arbiter surface出现后才能替换为board case。

#### SPM conflict equivalence

- 语义key为`spm-conflict-equivalence`。目标不是继续收集单点offset latency，而是检验一个预先冻结的
  conflict分类能否跨absolute base、tile和held-out workload复现；若不能复现，就明确否定当前profile下
  的compiler bank-coloring输入。
- 当前单tile可执行矩阵固定CT/RDMA和8192-byte relative offset，扫描
  `base mod 256 = 0/64/128/192`、absolute translation `0/0x20000/0x40000`、
  `256/512-byte`两种transfer/compute workload、A→B/B→A reciprocal issue order以及serial/window，
  96个schedule模板组成48个paired coordinate（复用8个既有control，新增88个held-out）。每个coordinate
  执行4次；每次invocation内serial/window共享同一request、payload和output allocation，执行先后各2次，
  共384个measurement row。record回显actual allocation、inner request、CT/RDMA source、result和SPM地址；
  host按absolute base、transfer和issue order分组，先验证逐row完整result、physical span、双侧guard、
  instruction count、completion及whole-output canary，再比较plan/full-execution/CT-blocking/RDMA-blocking的
  `window-minus-serial`方向；性能raw不能替代正确性。
- physical-tile轴使用独立16-rank shared package重放一个显式选择的paired coordinate。每个coordinate固定
  4轮：round 0/2按rank 0→15、round 1/3按15→0逐rank barrier激活，且
  `first_schedule=(rank+round) mod 2`，使每个rank的serial/window-first各2次。record回显round、
  rank order/phase、first schedule和execution ordinal；host要求每轮16个physical coordinate唯一、跨轮
  logical-rank映射稳定，并要求serial/window共享同一allocation。它只检验local-SPM分类是否跨physical tile
  复现，不测试remote SPM或跨tile并发冲突。其余bank-period offset继续作为既有offset control；
  任一issue-order、workload、base或tile方向翻转都不能越过`no-bank-coloring`。
- 当前NCC engine/FU/blocking counter只能作为冲突proxy。sustained probe已经保存、设置、核对并恢复
  owner-backed SPM PMU enable，以high-low-high读取port0/6的T2/T3 raw counter；这些port编号仍没有
  owner-backed physical bank/port mapping，也没有bank-specific attribution，因此不得命名bank。只有
  version-matched资料提供owner-backed bank/port映射与归因合同，或者预先冻结的等价分类在全部base
  translation、held-out tile和held-out workload上保持同一冲突方向，且`plan_cycles`或
  `full_execution`至少有一个预先指定的非零稳定信号，才允许形成窄profile cost feature；全零delta即使
  方向形式上一致也只能记为`consistent-but-zero-signal`。
  任一translation/tile翻转、仅median成立或依赖runtime allocation偶然base，都保持
  `no-bank-coloring`；即使promotion通过，也只影响已验证候选排序，不改变SPM legality、capacity或lifetime。
- 一个更小的sustained activation pilot另提供16个真实cell、4个matched group：
  `4KiB/16KiB × A→B/B→A`，每组在同一invocation内同时执行8192 candidate与4352 control的
  serial/window四行，并做四种执行位置轮转。每行还执行matched WDMA readback，完整owned SPM、gap
  canary、request/address echo、port0/6 T2/T3、pair-only count/completion/lifecycle均为强gate；
  constructor/submit在首次drain前partial accept时，只对已接受traffic做一次有界matching-worker
  safety drain，再恢复PMU enable并返回；正常drain本身失败则直接标记poisoned并停批，不做第二次drain
  或其它retry。cleanup失败不能伪装成普通case失败。四个group当前
  `pending-board`。
  `spm-physical-bank-color-class-unobservable` host gate承认raw port counter存在，但拒绝在没有physical
  bank/port mapping和bank-specific attribution时把offset、base alignment、port delta或aggregate timing
  序列化成physical bank identity。

#### DDR conflict equivalence

- 语义key为`ddr-conflict-equivalence`。既有54个`ddr-bank-pair` row保留为历史raw candidate：它们覆盖
  RDMA/RDMA、WDMA/WDMA、RDMA/WDMA、9个offset和serial/window，但每格独立launch且PMU窗口包含seed和
  readback，不能据此分类或命名DDR bank。
- 新的rank-one pending矩阵把一个coordinate的serial/window、A→B/B→A和四次重复收进同一invocation，
  使每个issue order下两种schedule都各处于第一/第二执行位置两次。
  seed完成后才读取PMU before，目标pair完成后立即读取after，随后才做readback；record回传actual allocation
  base和A/B地址，每行要求目标pair instruction count及after-PMU payload/guard mismatch exact，并按
  repetition和issue order交替serial/window执行位置；整个batch再要求最终archive、总count和terminal
  completion exact。矩阵覆盖RDMA/RDMA与WDMA/WDMA、`256/4096-byte`两种workload、
  translation `0/4096`和既有9个offset；RDMA另覆盖same-allocation/cross-allocation，WDMA cross-allocation
  因rank-one ABI只有一个host-visible output allocation而显式defer，不伪造正向oracle。
- physical-tile held-out复用16-rank cluster package，但使用显式pending selector，不改变旧tile-offset runner。
  每次barrier phase只激活一个rank；activation必须使用偶数且至少两次launch，由launch parity交替
  rank 0→15和15→0，每个coordinate的两次样本及launch parity再交替serial/window执行位置，record回显
  rank order和schedule position。当前矩阵只覆盖RDMA/RDMA同allocation：两个独立allocation分别扫描
  base `0x20000/0x40000`、relative candidate `0/4096/32768`、`256/4096-byte`、双issue order、
  serial/window和两次样本；host绑定真实tile coordinate、actual 64-bit address、完整payload/guard/count和
  pair-only PMU。该路径只形成RDMA同allocation分类的跨tile重复性证据，不把它外推到WDMA或cross-allocation，
  也不恢复controller/channel/hop或物理bank编号。
- promotion要求同一coordinate的paired controls共享actual地址生命周期，reciprocal order、workload、
  allocation/base和已执行的physical-tile held-out上proxy方向不翻转，并且所有correctness oracle通过。
  全零或仅噪声内可交换的delta不构成有区分度的conflict proxy。
  只有RDMA/RDMA同allocation轴具备当前physical-tile held-out；WDMA和cross-allocation轴没有该覆盖时不能
  宣称跨tile promotion。缺少bank-specific PMU或owner-backed地址映射时，结果最多是
  `conflict-equivalence` cost输入；当前compiler继续
  `no-ddr-bank-coloring`，也不从timing反推DDR legality、arena assignment或address mapping。

#### DDR active-rank contention

- 与bank equivalence分开，active-rank probe是可执行的full-card行为实验：
  RDMA-only、WDMA-only、RDMA+WDMA三类方向分别覆盖4KiB contiguous、64KiB contiguous和
  64KiB stride held-out；每个方向/workload在同一matched group内执行1/2/4/8/16 active rank，共45个
  case、9个board CTest。全部16 rank参与同一lifecycle，active rank要求exact output、guard、count、
  device `rdcycle`和PMU，inactive rank要求output/PMU canary不变。每个sample内五个active-count使用
  sample-major顺序，四轮forward/reverse/half-rotation/reverse-half使condition平均位置相同但不冒充完整
  五位置Latin rotation；1/2/4/8 active mask随sample共同平移，且同sample的single-rank baseline必须是
  其它active set的子集。
- promotion只允许比较同一五点组的contention方向，并要求contiguous calibration与stride held-out一致；
  host elapsed不是cost，active-rank趋势也不能命名bank/controller。`ddr-bank-owner-mapping-absent`和
  `ddr-bank-specific-counter-absent`为实际typed host blocker；WDMA cross-allocation held-out因当前只有
  一个host-visible output owner另有显式typed boundary，不以同allocation结果代签。

### 5.11 Collective 与后续性能行为 case

本节前半是collective algorithm case、执行状态和最终证据的事实表；其它hardware-behavior family与typed
blocker以本节末表和第5.10节共同为准。`Direct-DTE`列描述所有current collective共同使用的transport；
`algorithm`列才描述compiler-private schedule。现有名为Direct-DTE的板测不得解释为
Direct AllReduce；current AllReduce只有Ring和ordered Tree两种schedule。

第一批固定16 rank、i8和相同placement/ABI。`B`表示每rank参与的logical collective payload：
AllGather输入`B/16`、输出`B`；ReduceScatter输入`B`、输出`B/16`；AllReduce输入/输出`B`。
sentinel同时编码source rank、目标slice位置和logical lane，modular reduction由CPU独立计算。4KiB AllReduce
复用现有source/payload，不另建重复case。

sentinel由`rank + logical lane + payload size`进入固定64-bit mixing后折叠为i8，不使用线性mod-256
序列，也不声称有限i8域内所有lane byte全局唯一。catalog mutation gate逐AllGather source chunk、
逐reduction source contribution（ReduceScatter按destination segment）及逐ReduceScatter output slice
检查1/32/256B rotation，要求16个source contribution两两不同、16个ReduceScatter destination output
两两不同；对每个reduction source逐一删除并以其余每个source替换，每个destination slice都必须改变。
AllGather另要求16个concat chunk两两不同，且逐source相邻rank swap改变expected。

| case key | collective | algorithm A / B | B | accepted Instr结构oracle | runtime oracle | 状态 / 允许结论 |
| --- | --- | --- | ---: | --- | --- | --- |
| `all-gather-direct-vs-ring-256b` | AllGather | Direct / Ring | 256B | Direct逐rank验证15-peer cyclic round和source slice；Ring逐rank验证固定前后继、`P-1`轮、round→forwarded-slice递推及单一16-rank cycle；完整message tuple跨rank一一匹配 | 每rank按rank-group顺序完整concat；共同lifecycle/status合同见表后 | `pending`；不得影响cost |
| `all-gather-direct-vs-ring-4096b` | AllGather | Direct / Ring | 4096B | 同上 | 同上 | `pending`；不得影响cost |
| `all-gather-direct-vs-ring-65536b` | AllGather | Direct / Ring | 65536B | 同上 | 同上 | `pending`；不得影响cost |
| `reduce-scatter-direct-vs-ring-256b` | ReduceScatter | Direct / Ring | 256B | Direct逐rank验证source-round/destination-slice的15-peer owner exchange；Ring验证`P-1`个chunk round、round→slice递推和单一16-rank cycle；完整message tuple跨rank一一匹配 | 每rank只得到自己的destination slice，全部source贡献exact；共同lifecycle/status合同见表后 | `pending`；不得影响cost |
| `reduce-scatter-direct-vs-ring-4096b` | ReduceScatter | Direct / Ring | 4096B | 同上 | 同上 | `pending`；不得影响cost |
| `reduce-scatter-direct-vs-ring-65536b` | ReduceScatter | Direct / Ring | 65536B | 同上 | 同上 | `pending`；不得影响cost |
| `all-reduce-ring-vs-tree-256b` | AllReduce | Ring / ordered Tree | 256B | Ring验证`2(P-1)`轮两阶段round→slice递推、chunk和单一16-rank cycle；Tree验证15条reduce边、broadcast exact reverse、`slice=child`、`round=child depth`、单root/无环/二叉且rank-group inorder；完整message tuple跨rank匹配 | 16个replicated modular exact output；共同lifecycle/status合同见表后 | `pending`；不得影响cost |
| `all-reduce-ring-vs-tree-4096b` | AllReduce | Ring / ordered Tree | 4096B | 同上；复用原`tree-all-reduce` payload，但Ring必须由actual phase确认，不能由reserved-baseline名字推断 | 同上 | `pending`；不得影响cost |
| `all-reduce-ring-vs-tree-65536b` | AllReduce | Ring / ordered Tree | 65536B | 同上 | 同上 | `pending`；不得影响cost |

accepted Instr sidecar schema v2为每rank保存排序后的完整message tuple：
`direction / peer / communication_id / phase / round / payload_slice / issue_bytes /
constant_loop_multiplicity / executed_bytes`；独立的phase/peer/round/slice集合只作摘要，不承担graph证明。
send与对端recv必须在除direction外全部字段相同且多重集一一抵消。test-only selector还要求16个execution rank
逐rank只含请求的collective phase family；缺rank、Direct/Ring或Ring/Tree混合、混入其它collective family均fail closed。
AllGather输入采用partitioned boundary、输出replicated；ReduceScatter的rank-local contribution输入和destination
output均采用partitioned boundary，不再把rank-distinct数据标成replicated。

共同transport/lifecycle oracle先从每份schema-v6 manifest验证顶层`kind=kernel`、当前已解析的nested
kernel form/entry ABI/ordered phases，并验证16个rank各有唯一内部
`u32[1]`、64B storage/alignment、read-write `transport_status`，entry绑定
`wafer-direct-dte-status-v2`且`host_watchdog_required=true`，并有恰好16个matching
`entry_return` terminal completion。真实board command必须是单次`--all-ranks --board`且带有界
completion timeout；stdout还必须给出按序launch/completion/D2H/cleanup和16条matching terminal。
current BoardRuntime会在发布user output前D2H读取并要求全部16个status为Success，任一错误或不完整readback均失败并
quarantine；但`BoardRuntimeInvocationResult`和`wafer-run` stdout尚不导出逐rank raw
`(resource, status ABI, value)`。因此archive显式保存
`runtime_all_rank_success_enforced=true`、`observed_all_rank_success=false`及该observation gap，
不得写成“16条raw Success已独立观察”；若后续promotion要求可重放raw status，必须先扩展runtime output surface。

2026-07-26 pre-board状态：上述9/9 case均已实际生成两种typed package，accepted Instr tuple sidecar、
normalized manifest、status/watchdog/completion contract、最终ELF结构和双包no-card gate通过；
真实板端9/9仍为`pending`。首轮RS 256B重放
发现Direct lowering把同一input root的15个target subview连续send后才group wait，无法通过现有
one-live-sender/root-isolation gate；修复为逐send matching wait且未放宽gate，随后RS三种payload全部通过。
64KiB AR会在constant-trip structured loop中分tile，sidecar的send/recv bytes因此按constant loop
multiplicity加权；否则仅累计静态callsite会把真实执行量误报为一半。初版线性mod-256 sentinel也在
pre-board mutation审计中因16-rank reduction退化为32B周期而被拒绝，替换为上述hash并重跑9/9后才保留。

每个variant都必须从同一verified post-SPMD structured source生成，经过完整scheduling、SPM/DDR、
Direct-DTE matching、target ABI和package gate。test-only selector只接受实际phase匹配的accepted clone，
找不到就fail closed；它不是公开CLI、IR attr或package字段。A/B normalized manifest、host-visible binding、
source/profile/rank domain必须一致。板端按256B→4KiB→64KiB，pair内A/B、B/A交替至少3次；任一timeout、
untrusted terminal或设备异常立即停批，不retry/reset/power。host process elapsed只保存为诊断；在device
cycle/PMU unit、scope、clear/wrap与workload correlation未闭合前，本表不产生winner或latency常数。

9个collective algorithm case以外，本批其余hardware-behavior family已按实际资产重新审计。下表不再用
`planned`表示已经实现的case，也不把缺少观测面或producer的项伪装成待上板：

| 后续family | 区分的硬件行为 | 最小matched case | activation / stop gate | 当前状态 |
| --- | --- | --- | --- | --- |
| AllToAll / Permute traffic semantics | 全交换fanin/fanout、cycle与send-only/recv-only/self/unmapped-zero-fill角色 | 3个AllToAll payload、3个Permute forward-cycle payload，加reverse、opposite-pairs、disjoint-pairs、sparse-roles和two-epoch chain，共11个structured source→package→ELF→wafer-run case | 全rank exact、status-v2、accepted source graph、terminal/cleanup；单一Direct schedule只测traffic behavior，不伪造算法A/B或device cost | `board-ready/pending-board`（11 case、11 CTest/runner step）；ragged AllToAll与same-physical-buffer alias缺少typed surface，host fail closed |
| DTE raw multidestination / fan-in | broadcast、scatter、shuffle在不同fanout与destination order下的实际payload/mapping，以及current receiver protocol可承载的fanin | raw broadcast/scatter/shuffle × fanout 2/4/8/15 × adjacent/interleaved共24项；另有四源fan-in 1项 | 25项各有独立board CTest/runner step、single-mode process、全payload/guard/nonparticipant/status/cleanup oracle；`dest_num`编码和destination mapping只记录观察，不提前固化ABI或cost | `board-ready/pending-board`（25 case、25 CTest/runner step）；8/15源fan-in受current receiver FSM容量阻断，device phase、physical route与跨卡transport缺可信surface，继续typed fail closed |
| single-engine throughput | CT/NE/RDMA/WDMA/TDMA启动成本、tail和steady slope | raw dispatcher有20个small/steady/movement cell；NE另有`f16 m65/k129/n129` tail真实source→package case，共21 cell | exact result/padding/guards/count、manifest绑定的唯一terminal、ordered lifecycle与device PMU，至少3次；engine组按sample-major正/反/半程轮转。NE tail只在session id和完整board qualification fingerprint一致时与`single/ne/small`、`single/ne/steady-16k`合并，三点device-PMU方向完整才激活rate | `board-ready/pending-board`：raw cells为1个activation-group CTest，NE tail为独立`single-ne-tail` CTest；无新增板端证据 |
| engine-pair stage balance | 10个pair的A-bound/balanced/B-bound、A→B/B→A方向、iteration与same/cross-worker | 360个board-executable cell组成60个完整activation group；每组冻结serial/window、ratio、方向和held-out controls | 每个group单进程按sample-major正/反/半程轮转，full result/guard/count、manifest绑定的唯一terminal、三次重复、per-engine/global-union PMU齐全才可分类；host elapsed不作cost | `board-ready/pending-board`（60 group CTest/runner step）；没有稳定device方向则保持Unknown |
| three-stage software pipeline | RDMA→CT/NE→WDMA的prologue/steady/epilogue、slot数和stage balance | catalog显式枚举109个production three-stage cell及其capacity fallback，但current compiler尚不能产出带认证provenance的accepted-Instr multi-buffer program | 必须由production multi-buffer IR vertical产生；手写16KiB双slot只作既有observation，不能代签，也不允许raw adapter绕过 | `typed-fail-closed`（109 cell、0 board CTest）；真实production producer实现前不进入runner |
| worker placement/arbitration | w0/w1/w2 fixed-total placement、bounded backlog progress与low/high outstanding区分 | fixed-total placement 14、NE/RDMA progress 18、CT/RDMA × worker × low-2/high-6 outstanding 12，共44 case、14 matched group | sentinel/concurrent固定同一physical slot；safety drain前捕获observer boundary并将pending与raw `CONTROL`交叉核对；完整result/guards/routing count/tail canary、逐workerinstruction/blocking、global PMU、join RC/post-join idle、device deadline与cleanup全通过；partial issue/observation failure只做一次attempted-worker bounded cleanup，失败即poison且不retry | `board-ready/pending-board`（14 group CTest/runner step）；absolute per-worker timestamp和physical arbiter policy各有typed host blocker |
| DDR active-rank contention | RDMA-only、WDMA-only、双向在1/2/4/8/16 active ranks的contention slope | 3方向 × 4KiB/64KiB/64KiB-stride × 5 active-count，共45 case、9 matched group；所有16 rank参与lifecycle | 五个active-count按sample-major四轮equal-mean-position轮换，不声称完整Latin rotation；1/2/4/8 rank mask随sample换physical phase，single-rank baseline保持为同sample其它active set的子集；write-only未写区按runtime `0xa5`初始化逐字节检查，active exact/guards/count/device cycle/stable PMU、inactive output/PMU canary及逐rankmatching terminal齐全，contiguous和stride held-out方向一致才可promotion | `board-ready/pending-board`（9 group CTest/runner step）；不由active-rank趋势命名bank/controller |
| SPM conflict pilot | matched candidate/control是否存在可复现非零差异 | `4/16KiB × A→B/B→A`四组；每组含8192 candidate与4352 control的serial/window四cell和四种执行顺序，共16 cell | 同invocation owned-SPM/gap canary、address/count、owner-backed port0/6 T2/T3、matched WDMA readback、completion/lifecycle全部通过；partial accept失败先安全drain再restore，非零方向还需held-out一致 | `board-ready/pending-board`（4 group CTest/runner step）；raw port counter可观测，但physical bank/port mapping和bank-specific attribution仍由typed host gate拒绝 |
| DDR bank/coloring | compiler能否控制稳定physical class | rank-one/cross-tile conflict-equivalence board CTest只测actual-address scoped proxy；bank命名必须另有owner-backed mapping与bank-specific counter | proxy方向翻转或全零则停止；任何equivalence结果都不恢复controller/channel/hop或bank编号 | equivalence为`board-ready/pending-board`；physical bank mapping、bank counter及缺owner的WDMA cross-allocation axis为`typed-fail-closed`，不得伪造6144-row“bank测试已完成” |

## 6. 明确禁测或保守处理

- `depth + 1`不得混入普通calibration；它只允许进入typed tight gate：单engine、单case CTest、有界重复，
  packet builder预先释放，相邻execute之间不插入register MMIO，并要求完整count/output/guard、matching safety
  drain和正常lifecycle。同一software/runtime身份的合格会话只由总runner做首尾heartbeat，不为每个case重复资格化。
  除这个精确`D+1`向量外，任意更深或未证明occupancy/full的overflow仍不执行。
- 不发射SCALAR、DTE或CSR raw packet到`TsmExecute`。
- 不发射native `dims=HW` Concat。该组合是错误指令用法，永久`static-negative`；不得创建packet、
  board CTest、runner step或任何其它执行入口。合法source-level HW concat只允许materialize为typed
  GatherScatter。
- 当前profile不发射TDMA Memset native `Fmt_BOOL` packet；BOOL full-footprint只允许先canonicalize为I8
  byte fill，logical-valid BOOL fill保持fail closed。
- 当前profile不发射历史raw dimension 3/5的N/HWC Reduce；version-matched enum没有该语义，首个隔离
  case未取得completion。八个raw-axis row保留为`isolated-deferred`，不能用数值编码绕过dispatcher gate。
- 不使用大于2的worker id；底层`% 3`会静默别名，production必须先拒绝。
- 不执行跨worker同地址且没有producer completion的case。
- 不使用未由当前SPM arena/reservation证明完整owned range的诊断地址；尤其禁止复用曾导致timeout的
  `0x70000..0x7ffff` 64KiB WDMA地址交换。
- production不执行DTE receiver未准备；独立mode 13已经确认该路径在host deadline内不完成并使runtime
  context变为`poisoned`。错误tile坐标/participant在host
  gate拒绝；source/destination提前复用、invalid FSM和unknown event只允许进入专用有界错误观察，不进入
  普通正向路径。
- 不把历史Atomic Barrier、firmware替换型instruction profiler或power-cycle脚本作为普通probe。
- 不把aggregate PMU升级为per-cycle trace或cycle-accurate model；没有measurement basis的counter保持Unknown。

## 7. Compiler消费规则

- semantic legality只能消费稳定instruction/layout/numeric结果，不能由性能probe反推。
- vendor instruction builder的heap调用保持exact `rt_malloc/rt_free` loader ABI；CRT不得猜测scope并桥接到
  参数不同的allocator。
- memory planning从当前IR的SSA、view/range/effect和completion重算；hardware busytable落实current
  verified same-worker NCC dependency，但不是IR edge、lifetime或跨worker proof。
- 完整Instr IR在SPM planning之前，从typed MemoryEffects和SSA alias/root/view path重算RAW/WAR/WAW
  dependency并保持issue order；pure same-worker NCC链不在conflicting consumer前materialize local fence，
  RAR只在另有resource/control edge时保序。local drain只在NCC→Kcore/Direct DTE、跨worker join、显式
  barrier/structured completion backedge及terminal/host publication等completion-domain boundary物化并
  合并；不得匹配case、opcode、shape或buffer名。
- RDMA/WDMA stride描述DDR endpoint，local SPM operand按连续`transfer_bytes`规划；descriptor envelope、
  compact SPM footprint和cache visibility是三个不同合同，不能由一次mismatch互相归因。
- scheduling只移动dependency DAG中ready的指令。expanded矩阵中10个sustained disjoint cell已经通过
  correctness、三次重复正FU excess和更低window plan的资格门禁，只对匹配的profile、engine pair、
  worker关系、transfer class和address relation开放并行候选；4KiB RDMA+WDMA六个cell只计issue/wait收益，
  不计engine overlap。某pair可重叠不等于alias case可重排，未匹配shape/pair继续保守串行。
- bank/latency/bandwidth在未校准时保持Unknown；支持时也只排序已经通过exact legality的candidate。
- local drain、cross-worker join、DTE completion、multi-tile arrival和host publication保持不同typed边界。
- wait只在NCC→Kcore/Direct DTE、跨worker join、barrier/structured completion backedge、
  terminal/host publication等completion-domain exit的latest-legal位置materialize并合并；same-worker NCC
  的RAW/WAR/WAW消费或地址复用只保持dependency issue order，不自动产生wait。非default worker使用
  matching `bywork`。runtime lowering还要建立issue→poll→boundary consumer的机器顺序；
  当前没有证据允许把default/local fence当作跨worker join，也不能把`bywork(0)`诊断路径写成通用handshake。
- mapped-SPM `0x30400000 + offset`是uncached weak-order alias，只使用有序load/store与fence/sync；DDR
  cache publication/readback才按owned range clean/invalidate。两者不能共享同一个“flush cache”helper。
- `wafer.instr.fill`的BOOL count仍按typed physical bit domain解释；native packet exclusion只影响TX81
  target/CRT mapping，不能把上层BOOL改写成I8语义。I8 byte fill仅对完整`physical_footprint`成立。
- profile-specific数据进入target capability/calibration consumer，不进入上层StableHLO、Shardy或数学IR。
- probe源码、case名、sample count和raw packet不是production协议；通用实现只读取当前IR和typed target profile。
