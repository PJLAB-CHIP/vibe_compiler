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

本轮取得显式manual授权以区分“静态depth”和“总提交数”后，CT另执行一次typed tight `D+1` manual gate：仍为单engine、
单case、单样本，七份packet的builder预先释放；七次`TsmExecute`之间只保留`rdcycle`采样，不插入逐issue
register MMIO，只记录planned range供每个entry的oracle关联而不冒充实际register capture，整个window之后
只读一次control，再执行matching wait和完整result/guard oracle。前置Add 1/1 exact且cleanup完成；七次execute
rc均为1，issue-order cycle为
`[3558, 322, 561, 365, 236, 237, 218]`，`worker0.ct instruction_delta=7`，CT/full execution delta均为
553 cycles、blocking delta为0，七份boundary/final result与guard mismatch均为0，window后control为`0x100`；
后置Add同样1/1 exact且cleanup完成。这个向量证明当前profile可接受并完成超过静态depth的多次连续提交，因此
documented depth描述pending queue storage，不是一次完整submission/completion lifetime的总提交上限。
静态实现还表明`TsmExecute`没有software queue-full check；但该短workload在观测前已经排空，没有观察到resident
数量、queue full或backpressure，不能把七次成功写成“七条同时驻留”或饱和语义。typed tight `D+1`只保留为
隔离manual gate；任意更深提交在真实occupancy/full行为闭合前仍不执行。其它engine的exact
documented-depth边界已经闭合，但其`D+1`及所有engine的真实occupancy/full/backpressure仍待校准。

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
| CT numeric | f16/bf16/f32 opcode、convert、rounding、NaN/Inf/subnormal/signed-zero、tail | 非零elementwise vectors、边界值和held-out tail | 10/11/17 numeric capability | finite f16/bf16 elementwise、convert、reduce和select vectors `board-observed`；f32、special value及held-out tail仍`unknown` |
| NE numeric/layout | f16/bf16、accumulation、transpose、C0 tail、padding、K/M/N边界 | 非零identity/diagonal GEMM、完整padded range/canary | 08/10/11/17 | f16 16x16 identity和retained C0=16 `board-observed` |
| RDMA/WDMA descriptor | byte/logical-element stride转换、iteration、inclusive range、tail | contiguous + 1/2/3D stride，非零round-trip和guard | 08/11/14 | contiguous与既有large GEMM `supported`；完整stride matrix待held-out |
| TDMA Memset | element count、byte stride、raw logical iteration、inclusive range和dtype packet encoding | whole/128B×32/64B×64 geometry，I8/F16/BF16 raw与CRT，全range和guard | 10/11/14 | 普通dtype descriptor `calibrated`；I8/F16/BF16 vectors `board-observed` |
| TDMA BOOL fill | native `Fmt_BOOL` completion与bitpacked physical-footprint实现 | native小range timeout隔离；production BOOL→I8 byte fill需独立raw register、全range和guard | 10/11/14 | native `Fmt_BOOL`在当前profile `excluded`；I8 canonicalization待board held-out |
| TDMA movement variants | GatherScatter和其它DataMove的byte count、stride/iteration、range与kind-specific geometry | 每个已准入kind使用能区分错误descriptor的非零pattern、全range和guard | 08/10/11/14 | f16 Pad vector `board-observed`；GatherScatter已有compiler/model路径，其它variant仍`unknown` |
| SPM capacity/reservation | allocatable range和保留区 | boundary-positive与verifier negative；不触碰保留区 | 09/11 | 静态hard bound；board边界held-out未闭合 |
| SPM alignment/bank | 256B legality、非1024-bit访问代价、bank/color映射 | disjoint offset sweep，固定长度/engine pair/serial control | 09 placement与06 cost | 256B静态；exact bank mapping `unknown` |
| DDR/cache/coherence | host H2D、Kcore cache、DMA completion和host publication是不同域 | Kcore read前invalidate对照、DMA round-trip、matching drain后D2H | 09/12/14/15 | stale-cache机制`calibrated`；完整direction matrix待闭合 |
| queue shape与连续提交边界 | register/header中CT/NE/RDMA/WDMA静态depth为6、TDMA为4；该数值描述pending storage，不是完整lifetime总提交上限，active occupancy和full行为不能由形状或总提交数推出 | 普通calibration只跑1/2/4（TDMA 1/2）；隔离manual gate先验证恰好`D`，再经显式manual授权执行typed tight `D+1`，均为单engine/case/sample、前后Add heartbeat和完整count/output/guard；禁止任意更深提交 | 10/11/16 | CT/NE/RDMA/WDMA exact `D=6`与TDMA exact `D=4`的submission/completion/count/output/guard均为当前profile `board-observed`；CT tight `D+1=7`也为`board-observed`，但window后control为`0x100`且blocking为0，只证明总提交可超过depth；其它engine的`D+1`及所有engine的occupancy/full/backpressure仍`unknown` |
| worker scope | worker0/1/2 routing、default wait和`bywork` scope | CT三worker；matching wait后、safety drain前读CSR/result | 10/11/14/15 | CT worker0/1/2 routing与matching `bywork`均`board-observed`；default wait跨worker scope仍`unknown` |
| cross-engine overlap | 五类engine全部10个pair的可重叠性和共享资源 | disjoint backlog2 + serial control；仅在不触及任一engine full-depth时增加backlog4；`Ea+Eb-FU`重复正值 | 06/10/11/16 cost/scheduling | 旧CT+RDMA正overlap降级为`historical/inconclusive`；本轮两个方向r4对照median均为0，其余pair当前workload也未观察到PMU overlap；不外推成硬件不支持并行，compiler全部串行 |
| address dependency | busytable对RAW/WAR/WAW/RAR及exact/partial/adjacent/stride envelope的处理 | 仅对已证实可重叠pair做composition golden和PMU对照 | 09-11 legality/scheduling | 显式operand/range schema与composition oracle已通过no-card；本轮两次RAW选择均在hazard发射前被资格门禁拦截，板端hazard未执行；当前暂不适用 |
| issue overhead | wrapper构包/heap间隔对短window的影响，prepared issue是否值得materialize | 同packet序列wrapper与prebuilt对照 | 06/14 candidate/lowering | 旧RDMA+CT差异为`historical/inconclusive`，本轮未复现稳定正overlap，不构成新IR语义或收益结论 |
| local completion | default wait、`bywork`、local fence的范围和visibility | matching wait后立即CSR/Kcore oracle，再做safety drain | 09-11/15 | worker0/1/2 matching `bywork`正向`board-observed`；default/local-fence跨worker scope仍`unknown` |
| cross-worker join | 多worker并行、仲裁与地址依赖是否跨worker | disjoint w0/w1/w2，逐workerjoin；同地址不做无序正向 | 10/11/15 | 三worker disjoint CT matching join `board-observed`；并行性、仲裁与同地址行为仍`unknown/excluded` |
| Direct DTE | source read、destination visibility、participant、channel/FSM、terminal status | 16-rank receiver-first；producer→DTE、DTE→consumer和disjoint顺序 | 13-16 | 四个有序case `board-observed`；sender overlap `unknown` |
| multi-tile arrival | full-card/subgroup barrier的participant与复用合同 | production 16-rank正向；缺participant/错误坐标不测试 | 13/15 | full-card production路径`board-observed`；通用subgroup `unknown` |
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
- 随后的CT typed tight `D+1`只在明确manual gate中执行：七份builder预先释放，七次`TsmExecute`之间仅做
  `rdcycle`采样，planned range只用于entry与oracle关联、不视为实际register capture，window后才统一读取
  control并进入matching wait/full oracle。前置Add 1/1 exact且cleanup完成；七次rc均为1，issue-order cycle为
  `[3558, 322, 561, 365, 236, 237, 218]`，CT instruction delta为7、CT/full execution delta均为553 cycles、
  blocking delta为0，全部result/guard mismatch为0，window后control为`0x100`；后置Add同样1/1 exact且
  cleanup完成。这证明七次总提交可被接受和完成，静态depth不是完整lifetime总提交上限。
- 该短CT workload在唯一control观测前已排空，所以没有看到七条resident、queue full或backpressure；
  `TsmExecute`静态路径也没有software queue-full check。实际full行为继续为`unknown`，不得用本结果选择
  production outstanding window。其它engine的`D+1`仍待校准，任意更深overflow不执行。

overlap只按下面的PMU关系解释：

```text
pairwise_excess = engine_a_exec + engine_b_exec - fu_union_exec
```

`FU_EXE_TIME`是active engine时间的union；`*_BLOCKING_TIME`是queue backpressure，不是dependency stall。

### 4.2 单engine correctness与ABI观察

- CT f16 Add使用非零输入、完整fp16 golden、guard和DMA round-trip通过。
- instruction-family typed catalog的40个safe case已逐个串行launch并通过完整bit oracle、SPM guard、
  terminal与cleanup：f16/bf16 Neg/Add/Sub/Mul/Max/Min/Pow2/Relu，I8→f16/bf16、bf16→f16、
  f16→bf16/i16 convert，f16 Sum/Max、bf16 Min/Avg reduction，f16/bf16 Bit2FP+MaskMove select，
  f16 NE GEMM、f16 TDMA Pad、f16 TDMA Img2Col、f16 PoolMax、peripheral LUT16与f16 peripheral
  ArgMax/ArgMin。PoolMax使用
  `[1,2,4,64] -> [1,1,2,64]`、无padding、2x2 kernel/stride，两个输出窗口的128个FP16结果逐bit正确，
  256B physical output span和suffix guard均通过。ArgMax在含负数的128元素输入上返回
  `100@index73`；ArgMin在全正128元素输入上返回`0.5@index42`。两者都由CRT等待writeback后把FP16 value
  写到slot `[0:2]`、uint32 index写到`[4:8]`，并保持`[2:4]` poison不变。ArgMin负数对照返回了
  `-30@index0`，没有返回真实最小值`-100@index42`，因此current profile只闭合ArgMin正数普通值domain，
  负数域保持unsupported。该批只证明catalog中的有限普通值与固定geometry，不外推f32、special value、
  held-out tail或其它instruction family。
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
- ordinary FP16 Conv使用非对称`Sx/Sy=2/1`、input `[1,1,3,4]`、HWOI weight `[1,1,4,4]`和output
  `[1,1,2,4]`，8个结果逐bit正确；16B logical result、256B physical span及suffix guard通过。该向量能
  区分旧HWIO channel轴和对称stride误解，并与已修正的compiler verifier/target ABI合同一致。
- 同一Conv geometry以BF16 tight payload独立上板，8个结果、16B logical result、256B physical span及
  guard同样逐bit正确；该证据只新增BF16 format资格，不外推非平凡accumulation rounding或其它geometry。
- NE BF16 `M1K16N16`非平凡累加向量让每个输出具有16个非零K贡献；输入和FP32点积均为精确二进制数，
  expected同时包含非tie round-up/down。32B结果逐bit正确，256B physical span与guard通过，因此排除逐项
  BF16累加和末端截断；该向量仍不证明transpose、batch或logical tail。
- reduction首次运行暴露的是probe ABI错误而非硬件错误：四个case的逻辑结果均为128B，但CT会写满256B
  physical block，后128B是padding。catalog把allowed output span误写成128B，因而准确报告128B guard
  mismatch；将`result_bytes=128`与`output_span=256`分开后，四个reduction及余下case全部通过。结果逻辑域、
  physical write span和suffix guard必须分别建模，不能用逻辑shape缩小硬件写范围。
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
- 16-rank production Direct DTE基线、NCC producer经local drain后作为DTE source、DTE receive wait后由NCC
  consumer，以及两个disjoint安全顺序均16/16 exact、terminal success、canary正确。
- 当前Direct DTE helper把真实attach/send/wait/release集中在wait路径，不能用现有ABI证明sender与NCC
  overlap；该维度保持Unknown。

## 5. 当前首先要闭合的case批次

以下顺序按诊断价值和恢复风险排序；任一异常都停止当前批次，不自动reset：

1. **TDMA held-out**：用独立value/length复验I8/F16/BF16 descriptor；只执行会发出`Fmt_INT8` packet的
   production BOOL physical-footprint canonicalization，并核对实际register、完整byte range和guard。native
   `Fmt_BOOL`不再发射；GatherScatter和其它DataMove kind分别保留自己的校准项。
2. **single-engine completion**：CT/NE/RDMA/WDMA/TDMA在worker 0从`N=1`开始递增，但每个case严格低于
   register/header给出的静态depth；普通calibration使用1/2/4，TDMA只使用1/2。CT在worker 1/2只做低深度
   routing与matching wait。
3. **documented-depth manual**：修正probe后，一次只选择一个engine和一个case，恰好发射该engine文档depth，
   只运行一遍，并在前后各跑一次known-good Add heartbeat。验证连续提交、最终completion、instruction count、
   完整output和guard，不声明并发occupancy；CT/NE/RDMA/WDMA `D=6`与TDMA `D=4`均已按此合同闭合。
4. **typed tight depth-plus-one manual**：只有同engine的`D`向量已通过且取得显式manual授权时，才能由独立typed
   gate发射恰好`D+1`条；单engine、单case、单样本、前后Add heartbeat和完整oracle不变，相邻execute之间
   不做register MMIO。CT `D+1=7`已证明总提交数可以超过pending queue depth，但没有观察到occupancy/full；
   其它engine仍待校准，任意更深提交不执行。
5. **全部disjoint pair（当前workload已闭合）**：10个engine pair的serial/window各3个样本均正确；
   CT+RDMA本轮两个方向r4对照median均为0，旧正overlap降级为historical/inconclusive；其余9个pair当前也
   未观察到PMU overlap。含TDMA的pair只运行r2，r4因会触及TDMA静态depth 4而不运行；负观察不外推成硬件
   永久不能并行。
6. **dependency relation（资格未满足）**：两次RAW exact/partial/adjacent选择都在hazard发射前被
   disjoint资格门禁拦截，当前不执行板端hazard。所有pair保持保守串行；只有未来同方向serial/window对照
   稳定达到median正overlap，才恢复对应composition golden。
7. **SPM offset sweep**：固定workload，只改变相对offset；建立经验conflict class，不提前命名物理bank。
8. **worker/join/visibility**：三worker disjoint并行，matching wait后且safety drain前验证CSR、Kcore和host
   visibility；跨worker同地址只跑显式ordered正向。
9. **DTE/SPM PMU与multi-tile**：先闭合counter basis，再测有序producer/consumer和full-card arrival；
   错误坐标、缺participant和资源提前复用保持negative，不上板。
10. **numeric/layout held-out**：以f16/bf16为主，补CT/NE的tail、transpose、边界值和special value；模型
   纵向保持模型自身dtype。

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
- `wafer.instr.fill`的BOOL count仍按typed physical bit domain解释；native packet exclusion只影响TX81
  target/CRT mapping，不能把上层BOOL改写成I8语义。I8 byte fill仅对完整`physical_footprint`成立。
- profile-specific数据进入target capability/calibration consumer，不进入上层StableHLO、Shardy或数学IR。
- probe源码、case名、sample count和raw packet不是production协议；通用实现只读取当前IR和typed target profile。
