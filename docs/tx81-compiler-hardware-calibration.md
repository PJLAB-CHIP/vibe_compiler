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
  外推到其它firmware/runtime/SKU，不以reset/power cycle作为测试步骤，不用unsafe queue overflow或缺失
  participant的DTE case探索错误行为。
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
只有设备实际异常时才由用户决定是否重启。

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

### 2.4 Schema 2 hazard与手动饱和边界

parameterized NCC probe在wire中显式携带first/second operand角色；RAW/WAR/WAW/RAR不能从lane名、
engine名或buffer名恢复。host只在同一engine pair的rounds=2 disjoint serial/window对照已经观察到
可重叠后，才允许执行exact/partial/adjacent hazard case。device以实际packet range回传证据，并用逐段
composition golden、未选operand guard和最终copyback共同闭合正确性。

queue `depth+1`不进入普通calibration suite，只能由typed manual-saturation flag启动。该suite固定同一
engine/worker的raw window、无requested wait、精确发射`D+1`条，每次记录worker control/IB和`rdcycle`
包围的execute耗时；boundary允许结果尚未完成，但guard必须始终为零，最终safety drain后instruction count、
完整结果和guard必须精确。只有重复样本均达到`IB >= D`且第`D+1`次调用耗时稳定高于此前调用时，才记为
`backpressure-observed`，否则记为`inconclusive`，不能据此推断nonblocking/full-return语义。当前只完成
schema、device/host实现和no-card验证，尚无板端结论。

### 2.5 DTE/SPM PMU有效样本边界

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
| CT numeric | f16/bf16/f32 opcode、convert、rounding、NaN/Inf/subnormal/signed-zero、tail | 非零elementwise vectors、边界值和held-out tail | 10/11/17 numeric capability | f16 Add `board-observed`；其余`unknown` |
| NE numeric/layout | f16/bf16、accumulation、transpose、C0 tail、padding、K/M/N边界 | 非零identity/diagonal GEMM、完整padded range/canary | 08/10/11/17 | f16 16x16 identity和retained C0=16 `board-observed` |
| RDMA/WDMA descriptor | byte/logical-element stride转换、iteration、inclusive range、tail | contiguous + 1/2/3D stride，非零round-trip和guard | 08/11/14 | contiguous与既有large GEMM `supported`；完整stride matrix待held-out |
| TDMA Memset | element count、byte stride、raw logical iteration、inclusive range和dtype packet encoding | whole/128B×32/64B×64 geometry，I8/F16/BF16 raw与CRT，全range和guard | 10/11/14 | 普通dtype descriptor `calibrated`；I8/F16/BF16 vectors `board-observed` |
| TDMA BOOL fill | native `Fmt_BOOL` completion与bitpacked physical-footprint实现 | native小range timeout隔离；production BOOL→I8 byte fill需独立raw register、全range和guard | 10/11/14 | native `Fmt_BOOL`在当前profile `excluded`；I8 canonicalization待board held-out |
| TDMA movement variants | GatherScatter和其它DataMove的byte count、stride/iteration、range与kind-specific geometry | 每个已准入kind使用能区分错误descriptor的非零pattern、全range和guard | 08/10/11/14 | GatherScatter已有compiler/model路径；本轮板端measurement basis及其它variant仍`unknown` |
| SPM capacity/reservation | allocatable range和保留区 | boundary-positive与verifier negative；不触碰保留区 | 09/11 | 静态hard bound；board边界held-out未闭合 |
| SPM alignment/bank | 256B legality、非1024-bit访问代价、bank/color映射 | disjoint offset sweep，固定长度/engine pair/serial control | 09 placement与06 cost | 256B静态；exact bank mapping `unknown` |
| DDR/cache/coherence | host H2D、Kcore cache、DMA completion和host publication是不同域 | Kcore read前invalidate对照、DMA round-trip、matching drain后D2H | 09/12/14/15 | stale-cache机制`calibrated`；完整direction matrix待闭合 |
| queue capacity | CT/NE/RDMA/WDMA深度6，TDMA深度4以内不丢entry；full行为 | `N=1,D-1,D`，每entry唯一地址/pattern/count；独立typed manual suite才允许`D+1` | 10/11/16 | RDMA/CT深度6内部分`board-observed`；D+1 probe已通过no-card，板端行为`unknown` |
| worker scope | worker0/1/2 routing、default wait和`bywork` scope | CT三worker；matching wait后、safety drain前读CSR/result | 10/11/14/15 | worker0 `board-observed`；worker1/2 wait scope `unknown` |
| cross-engine overlap | 五类engine全部10个pair的可重叠性和共享资源 | disjoint backlog2/4 + serial control；`Ea+Eb-FU`重复正值 | 06/10/11/16 cost/scheduling | RDMA+CT `calibrated`；其余`unknown` |
| address dependency | busytable对RAW/WAR/WAW/RAR及exact/partial/adjacent/stride envelope的处理 | 仅对已证实可重叠pair做composition golden和PMU对照 | 09-11 legality/scheduling | 显式operand/range schema与composition oracle已通过no-card；板端行为`unknown` |
| issue overhead | wrapper构包/heap间隔对短window的影响，prepared issue是否值得materialize | 同packet序列wrapper与prebuilt对照 | 06/14 candidate/lowering | RDMA+CT `calibrated`；不构成新IR语义 |
| local completion | default wait、`bywork`、local fence的范围和visibility | matching wait后立即CSR/Kcore oracle，再做safety drain | 09-11/15 | worker0正向`board-observed`；跨worker`unknown` |
| cross-worker join | 多worker并行、仲裁与地址依赖是否跨worker | disjoint w0/w1/w2，逐workerjoin；同地址不做无序正向 | 10/11/15 | `unknown`；跨worker同地址无completion为`excluded` |
| Direct DTE | source read、destination visibility、participant、channel/FSM、terminal status | 16-rank receiver-first；producer→DTE、DTE→consumer和disjoint顺序 | 13-16 | 四个有序case `board-observed`；sender overlap `unknown` |
| multi-tile arrival | full-card/subgroup barrier的participant与复用合同 | production 16-rank正向；缺participant/错误坐标不测试 | 13/15 | full-card production路径`board-observed`；通用subgroup `unknown` |
| host launch/runtime | kernel/model launch、resource staging/readback、timeout、failure cleanup | 同package schema、exact output、terminal/cleanup | 14-16 | 已有kernel/model与16-rank路径`board-observed/supported`，按owner证据解释 |
| NCC PMU basis | instruction count、engine exec、global union、worker scope、wrap稳定读取 | 单engine等式、pair union、high-low-high和重复样本 | 16与后续Q9 | worker0 engine/union `calibrated` |
| DTE/SPM/TMNOC PMU | counter scope、unit和与workload相关性 | 独立单域workload和held-out payload sweep | 16与后续Q9 | register shape `static`；measurement basis `unknown` |
| SCALAR/CSR ordinary issue | 是否属于`TsmExecute` typed queue | 静态negative，不发送未知packet | 11/14 | 当前ABI `excluded` |

## 4. 当前profile已经闭合的事实

本节记录当前已取得的事实，不把精确cycle提升成跨profile硬件常数。

### 4.1 NCC mode、queue和RDMA/CT overlap

- 3个worker的`serial_mode`只读值均为0。
- worker 0上disjoint RDMA/CT使用64KiB强sentinel workload；issue count `1/2/4/6`均在显式drain和
  WDMA round-trip后完整exact，未执行depth+1。
- prebuilt packet在backlog 2已出现稳定正overlap；one-shot wrapper因构包、heap和释放间隔，在较深backlog
  才形成明显窗口。issue-count-4重复样本的prebuilt overlap显著大于wrapper。
- 这证明当前profile的RDMA/CT queue能够并行，也证明“queue容量”“硬件可并行”和“软件是否及时喂饱queue”
  是三个不同问题；它不证明其它9个pair或任何alias关系。

overlap只按下面的PMU关系解释：

```text
overlap_cycles = rdma_exec + ct_exec - fu_union_exec
```

`FU_EXE_TIME`是active engine时间的union；`*_BLOCKING_TIME`是queue backpressure，不是dependency stall。

### 4.2 单engine correctness与ABI观察

- CT f16 Add使用非零输入、完整fp16 golden、guard和DMA round-trip通过。
- RDMA和WDMA分别以非零pattern、全range round-trip和精确instruction count通过。
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
- 在当前profile的配对样本里，64B×64比whole 4KiB和128B×32用时更长。该结果只是一条
  profile-scoped performance observation，不是stride legality、固定代价或跨profile cost常数。
- 原生TDMA `Fmt_BOOL`的小range安全case在10秒内未完成；执行上下文被隔离，随后只读设备状态已回到
  idle且没有残留进程，因此没有reset或重启。这足以把native packet在当前profile标记为`excluded`，
  不能据此推断其它profile或其它TDMA kind。
- production只对`physical_footprint` BOOL fill定义替代路径：Instr/TargetCall保持bit count和canonical
  bool scalar，TX81 CRT边界把完整physical bytes改写成`Fmt_INT8`、byte count和`0x00/0xff` byte splat，
  继续由TDMA Memset执行。该路径会有意覆盖最后一个byte中的unused bits，故不能用于
  `logical_valid` BOOL fill；其板端held-out通过前不升级为`supported`。

### 4.4 Cache、completion与DTE

- Kcore普通load读取复用的cacheable DDR input可能命中旧cache；对range执行匹配cache invalidate和fence后
  恢复exact。同一payload的NCC DMA round-trip不受该旧load影响。
- 因此host H2D、Kcore DDR load、NCC local drain和host-visible publication是不同visibility合同，不能互换。
- 16-rank production Direct DTE基线、NCC producer经local drain后作为DTE source、DTE receive wait后由NCC
  consumer，以及两个disjoint安全顺序均16/16 exact、terminal success、canary正确。
- 当前Direct DTE helper把真实attach/send/wait/release集中在wait路径，不能用现有ABI证明sender与NCC
  overlap；该维度保持Unknown。

## 5. 当前首先要闭合的case批次

以下顺序按诊断价值和恢复风险排序；任一异常都停止当前批次，不自动reset：

1. **TDMA held-out**：用独立value/length复验I8/F16/BF16 descriptor；只执行会发出`Fmt_INT8` packet的
   production BOOL physical-footprint canonicalization，并核对实际register、完整byte range和guard。native
   `Fmt_BOOL`不再发射；GatherScatter和其它DataMove kind分别保留自己的校准项。
2. **single-engine completion**：CT/NE/RDMA/WDMA/TDMA在worker 0做`N=1,D-1,D`唯一entry；CT在worker
   1/2验证routing与matching wait。
3. **manual saturation**：只有对应single-engine `D`样本完整通过后，才逐engine、逐进程执行typed `D+1`
   case。每个进程有独立外层timeout，任何timeout、状态异常、count/output/guard失败立即停止整个批次；
   不自动重试、reset或power。结果只按2.4节证据分成`backpressure-observed`或`inconclusive`。
4. **全部disjoint pair**：10个engine pair各用serial control、backlog2/4和重复PMU样本；4KiB只做正确性，
   64KiB或由单engine时长选择的payload才用于overlap判断。
5. **dependency relation**：只对已证明能重叠的pair做RAW/WAR/WAW/RAR × exact/partial/adjacent；每个结果
   使用逐元素或逐字节composition golden。
6. **SPM offset sweep**：固定workload，只改变相对offset；建立经验conflict class，不提前命名物理bank。
7. **worker/join/visibility**：三worker disjoint并行，matching wait后且safety drain前验证CSR、Kcore和host
   visibility；跨worker同地址只跑显式ordered正向。
8. **DTE/SPM PMU与multi-tile**：先闭合counter basis，再测有序producer/consumer和full-card arrival；
   错误坐标、缺participant和资源提前复用保持negative，不上板。
9. **numeric/layout held-out**：以f16/bf16为主，补CT/NE的tail、transpose、边界值和special value；模型
   纵向保持模型自身dtype。

## 6. 明确禁测或保守处理

- `depth + 1`只允许进入2.4节定义的typed manual-saturation suite；普通calibration、任意更深overflow、
  缺独立timeout/最终drain/强oracle的变体均不执行。
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
- scheduling只移动dependency DAG中ready的指令。某engine pair可重叠不等于alias case可重排。
- bank/latency/bandwidth在未校准时保持Unknown；支持时也只排序已经通过exact legality的candidate。
- local drain、cross-worker join、DTE completion、multi-tile arrival和host publication保持不同typed边界。
- `wafer.instr.fill`的BOOL count仍按typed physical bit domain解释；native packet exclusion只影响TX81
  target/CRT mapping，不能把上层BOOL改写成I8语义。I8 byte fill仅对完整`physical_footprint`成立。
- profile-specific数据进入target capability/calibration consumer，不进入上层StableHLO、Shardy或数学IR。
- probe源码、case名、sample count和raw packet不是production协议；通用实现只读取当前IR和typed target profile。
