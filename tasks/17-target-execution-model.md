# Wafer Target Execution Model

当前模型是owner-backed target LLVM/TargetCall上的untimed functional-event model。它验证target语义、physical
Tile交互和完整输出，不是accepted IR解释器、runtime ABI替代品或cycle model。Model是独立typed invocation consumer，
必须由current `DeviceExecutable` source vertical重新证明，不能沿用旧执行域
结论或把package结果代签model能力。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  同一次compiler transaction产生的`DeviceExecutable`、与其绑定的owner-backed target LLVM module set、typed program
  invocation、`ProgramDataHandoff`、`TargetTensor`与独立CPU expected；`DeviceExecutable`覆盖single card的all-and-only 16 Tiles并保留
  (card_id, tile_id, launch_slot)。
- Current stage responsibility:
  将每个`TargetTensor`编码一次并绑定到exact `TileEntryArgument`；通过host JIT执行final target LLVM entries并解码closed TargetCall
  registry；在SystemC中按Tile、worker、engine、event和private memory执行functional语义；原子发布完整结果。
- Output IR / files:
  TargetModelResult：target identity、model implementation/evidence、card-scoped completion统计、numeric flags、typed outputs及诊断；
  不产生可被compiler或runtime消费的schedule sidecar。
- Downstream consumer:
  source/model differential、target command qualification、production无卡验证，以及后续独立board correlation。
- User-level driver / named pipeline:
  wafer-compile `search|none`的target-model执行路径和configured model integration tests；model不增加第三种optimization policy。
- Explicit non-goals:
  不重新lower accepted IR；不解释package中的历史格式；不模拟vendor packet/loader；不宣称cycle accuracy、
  bandwidth或board performance；不从symbol、OS thread、ordinal或container position恢复Tile身份。
- Done criteria:
  current target-call registry、memory/effect/event/numeric正负例通过；同源source的16-Tile完整output与CPU expected
  比较；production qualification matrix fresh重放。真实设备相关结论仍由独立board gate签发。
```

## 2. 稳定边界

### 2.1 Owner-backed input

model只接受compiler保留的same-invocation owners：

- `DeviceExecutable`提供program boundary bindings、Tile executable domain和completion/transport contract；
- 与该`DeviceExecutable`绑定的target LLVM owner set提供target conversion真正发布所用的LLVM modules与typed
  `TileEntryArgument`；
- program invocation提供source tensor值，不复制target schema或猜测entry argument；
- independent CPU expected只用于最终差分，不进入compiler IR/package。

model输入边界就是`DeviceExecutable`及与其绑定的invocation-local target LLVM owner set，不定义额外稳定output层。

`prepareTargetModelInvocation`必须在JIT materialization前all-and-only消费每个非output program tensor和每个
`TileEntryArgument`。它按显式program tensor identity、entry argument relation和target layout建立model-private memory：
card级`TargetTensor`、input/output及SharedWorkspace由card共享，普通workspace/status由Tile独占。一个card input只编码和初始化一次，引用它的
16个Tile entry arguments绑定同一base；input physical bytes与argument values由prepared invocation拥有，不alias source
NPY storage。这里的model-private memory只服务functional model，不定义package中的provider allocation identity。

同一物理resource的geometry与每个Tile的access是两件事。Invocation准备与memory registry共用kind/index、dtype/layout/shape、bytes、alignment、
materialization和zero-initialize比较；同一SharedWorkspace允许writer为WriteOnly、reader为ReadOnly或未使用者为None。
这些权限仍逐slot保留，由实际read/write校验消费，不能并成所有Tile都可写。覆盖16-Tile source broadcast/add、1024/1025长度、
真实temporal subview读取与完整独立reference；不同geometry或initialization为拒绝例，越权读写仍失败。

Current program-data/package合同下，parameter/external captured-constant不得经per-Tile invocation重新打开或读取。当前model先从同一
`ProgramDataHandoff`形成typed `ProgramTileInvocation`，再按explicit `TileEntryArgument`建立model-private bytes；它不得读取package
文件或重新解释manifest。后续若让model与package共享materialization owner，必须保持每个selected representation一次转换和
bounded window合同；相同digest仍不能合并不同`ProgramTensor`或不同target representation。

### 2.2 Explicit physical identity

model全链保留三个独立typed fields：

```text
CardId
TileId
LaunchSlotId
```

每个 `TargetCallTileDescriptor`、arguments、dynamic transaction、memory owner、output与diagnostic都携带该关系。
launch slot只负责canonical调用位置；它不等于Tile ID。aggregate target module或host JIT bridge不得用 `pid`、vector
ordinal、entry name或thread ID取代显式binding。

### 2.3 TargetCall frontend

TargetCall frontend是final target LLVM到typed transaction的唯一host桥：

1. 验证完整16-Tile module/slot domain；
2. `begin`把immutable invocation descriptor交给sink；
3. 每个Tile entry调用closed descriptor registry中的target calls；
4. decoder从exact argument positions生成typed payload并校验range/enum/format/worker；
5. Tile return调用 `completeTile(card_id, tile_id, launch_slot)`；
6. 所有Tile成功后调用一次`finish`，sink通过`completeInvocation`验证完整调用并返回结果；
7. 任一失败调用`abort`，不返回partial commands或outputs。

内部 `wafer_target_call_dispatch`只完成JIT call interception。它不是package export、runtime ABI、serialized schema或
用户入口，也不拥有physical-dataflow scheduling语义。

#### 标量整数夹界的主机执行合同

输入为final target LLVM中的标量整数SSA及`llvm.smin/smax/umin/umax`；本层验证完整module后，
由同一LLVM host JIT执行原intrinsic，输出仍是原bitwidth的SSA值，直接用于TargetCall地址/字段或控制流。
生产target-model入口与component测试使用同一个frontend。没有新TargetCall、CRT符号、数值backend或第二解释器。
依据[LLVM整数min/max语义](https://llvm.org/docs/LangRef.html#llvm-smax-intrinsic)和pinned
`Intrinsics.td`，只开放这四个无内存effect、可返回的标量整数intrinsic；signed/unsigned、位宽和截断保持原LLVM语义。
pinned `ArithToLLVM.cpp`将`arith.minsi/maxsi/minui/maxui`直接映射到这四种操作，因此不能将合法的夹界与
任意外部调用一起拒绝。其它intrinsic、vector overload、指针解引用和未知外部调用保持原拒绝边界，失败必须发生在sink `begin`之前。

本项不新增SPM scalar load、不改变completion/search/device code，也不宣称动态词表、完整LM数值或实卡已通过。
完成条件为下表实际执行、直接Simulator组件及canonical完整增量构建/no-op通过。

| 输入等价类 | exact输出 / failure | 直接下游witness |
| --- | --- | --- |
| i8/i16/i32/i64/i128标量min/max；signed/unsigned、相等、零、符号边界、最大值 | 运行时参数保留位型；逐项核对四种运算的结果，不能靠IRBuilder常量折叠 | host JIT→同一TargetCall decoder→记录sink；scalar oracle是位宽语义的有界测试 |
| rank3 F16/BF16 `[2,S,64]`、S=1024/1025/1031、16 Tiles、64行block及tail | 每个实际窗口的clamp地址、计数、Tile身份和调用顺序精确；所有行恰好一次 | final target LLVM→JIT→实际decoded RDMA参数；不代签DMA执行或数值readback |
| vector min/max、其它intrinsic及既有native负例 | 精确unsupported诊断；sink未begin、无partial结果 | 同一host frontend |

### 映射内存标量访问的主机消费者

14号lowering输出SDK mapping call及原生LLVM整数load/store。Host frontend只接受从已注册mapping返回值经GEP得到的
pointer，拒绝未知来源、指针逃逸及非i32/i64访问；DDR目前只读。Invocation-owned clone在JIT之前将这些实际load/store
转为同一TargetCommandSink的typed内存访问，mapping返回模型地址，设备地址不会作为主机pointer被解引用。
DDR读取同时检查原mapping范围和实际ABI allocation，SPM按实际Tile范围检查；写仅形成原子byte effect。
整数clamp/cast仍由LLVM JIT执行，model不复写source算术。

SystemC将mapping acquire及标量访问作为同步Kcore observer，存在冲突pending NCC时拒绝，不补join，也不新增NCC pending。
覆盖i32/i64边界位型、rank3/1024/1025/1031/16 Tile的实际索引→行地址→全输出；另测mapping越界、未完成NCC写入和
错误pointer来源。该合同只签发主机功能；设备延迟、发令成本及cache行为按硬件事实和本轮板测单独验收。

## 3. SystemC functional-event architecture

### 3.1 Process model

SystemC elaboration为每个Tile建立一个SC_THREAD，在线程中调用该Tile的JIT entry。同步target call可以让当前
线程等待model event，同时保留JIT stack；独立Tile线程可继续推进。

model使用event-driven fixed point：只有ready transaction执行，执行后发出data-ready/engine-completion/transport event，
再唤醒依赖线程。没有ready work且未达到card-scoped completion时报告NoProgress，不通过host轮询猜测顺序。

### 3.2 Tile memory与地址

每个Tile拥有隔离的SPM/engine-visible address domain；同一个card-scoped `TargetTensor`必须让引用它的
`TileEntryArgument`共享同一typed base和同一model-private bytes，不能按argument复制storage。Tile-scoped
workspace/status保持独立memory。地址解析只依赖prepared entry argument memory、physical layout和checked range：

- 每次read/write验证alignment、byte span、access mode与overflow；
- view/strided/gather-scatter按target physical encoding求址；
- 同一地址值在不同Tile SPM domain不构成alias；
- NoC/Direct-DTE只有显式typed source/destination Tile和range才能跨domain传输；
- compiler workspace/status不会被误作user output；role相同、name相同或bytes相同都不能创建alias。

### 3.3 Engine、worker与completion

TargetCall registry拥有issue engine、optional NCC worker argument及local instruction completion behavior。model按typed
事实维护per-Tile engine/worker state，不从symbol字符串分组。

- synchronous writeback在call完成时可见；
- ordered pending work在对应completion/join后可复用；
- Direct-DTE prepare/issue返回opaque event，wait只完成exact event；
- `NCCJoin`只等待participant mask指定的local NCC domains；
- Tile entry返回只有在其local drain contract满足时才计为completed；
- card-scoped apply要求16个Tiles与所有transport obligations闭合。

缺失wait、wrong worker/participant、range hazard、event reuse或cross-Tile message mismatch都必须确定失败。

## 4. Functional transaction semantics

`TargetTransactionPayload`是closed variant，覆盖current target movement、compute、conversion、collective completion和
Direct-DTE families。每个family分三层负责：

1. descriptor/decoder：ABI字段位置、宽度、enum与typed payload；
2. plain C++ kernel：数值或byte-level functional semantics；
3. SystemC wrapper：resource availability、event、latency-free ordering和failure propagation。

同一规则适用于RDMA/WDMA、gather-scatter、fill/mask/convert、GEMM、elementwise、reduce、conv/pool/unpool、TDMA、
peripheral和transport。unsupported组合返回typed error，不能落到“近似执行”、host library默认行为或第二解释器。

## 5. Numeric contract

target numeric command、physical tensor、formal arithmetic和backend evidence是四个owner，不存在跨四者共享的profile/registry：

- `TargetOperation`/`TargetCall`拥有rounding、operation、dimension、format、geometry field及exact decoded payload；
- shared physical tensor descriptor只拥有format、layout、static shape与checked count，不携model identity或内建digest；
- formal/model implementation从typed command直接进入family-specific validator/kernel，支持或拒绝都不经过global resolver；
- `WaferTargetNumericBackend`拥有model-facing host numeric dispatch；其中`WaferOneDNNBackend`只实现qualified GEMM/reorder。
  oneDNN qualification只绑定concrete problem、payload、comparator、backend和environment，不记录semantic profile或resolution digest。

compiler/ABI legality不读取formal/target numeric backend support；model support和board correlation也不能反向签发compiler legality。
历史`NumericSemantics` aggregate、model profile、capability pattern和resolved command不是current合同，也不保留
compatibility surface。

### 5.1 Physical codec

source tensors先按对应TargetTensor或external port的target descriptor以及exact `TileEntryArgument`编码。outputs从
相同descriptor解码回source-visible dtype/shape，并按unique output port一次发布；多个Tile output arguments只是同一
model-private memory的checked byte range，不形成
多个结果。byte size相等不能推断layout；padding、blocked layout、bitpacked format和narrow integer都由共享physical codec处理。

source compact encoding与target physical encoding各自提供明确的byte/bit/noncanonical policy；codec不得从process-global model
profile取得这些事实。TargetTensor的非identity value conversion由14号显式materialization action拥有，不通过fake CT command进入
formal model。

codec必须提供bounded window接口：source range、target range、padding和incremental digest都以checked 64-bit arithmetic推进；
禁止为完整tensor建立每element对象数组、完整logical副本加完整physical副本，或按Tile重复编码。formal小tensor路径可以保留
便于验证的局部value表示，但large-data consumer必须进入同一codec语义的bounded lane并与formal golden逐段对照。

### 5.2 Formal lane

formal lane拥有确定的dtype arithmetic、rounding、NaN/Inf/signed-zero、conversion和exception flags。它用于边界语义和
小规模exact/typed tolerance验证，不调用target lowering或执行结果本身作oracle。API按convert、elementwise、GEMM和reduce
family接收typed target operation/format/parameter；实现未覆盖的组合在任何memory write或flags merge前分类拒绝，不以稀疏policy row、
wildcard selector或digest resolution决定控制流。

F32 native reduction的formal数学子集包括sum/max/min。Sum保持原逐步RNE加法；max/min复用LLVM APFloat的IEEE maximum/minimum语义，
分别从负/正无穷初始化，传播canonical NaN、累计signaling-NaN invalid并区分正负零。其它dtype和未支持CT轴仍明确拒绝。
既有managed-reference lane对非NaN F32的sum/max/min使用同一逻辑遍历与identity，包含无穷和正负零；NaN仍在任何写入前拒绝。
该扩展用于主机数学验证，不签发硬件位级或性能资格，也不改变被测Instr。覆盖rank3/4、1024/1025/1031、负值、无穷、零、NaN、
多归约维度、padding和工作预算，并通过真实source到SystemC的输出比较；大模型沿既有显式managed-reference入口执行。

源浮点Div的当前目标实现由11号设计规定为Recip再Mul。模型按actual Instr分别计算与舍入两条指令，
不把它们合并为直接divide；TargetElementwiseOperation及Wafer target-call registry已移除Div，保留独立Recip。

普通Pool的主机子集直接消费`TargetPoolCommand`的NHWC/NCx几何及X/Y kernel/stride，验证N/C和输出窗口范围后，
通过原physical codec读取窗口并调用既有formal maximum/minimum/add。Max/min使用原format；sum当前只覆盖F32，
用于原始低精度AvgPool的F32 opmath，归一化继续执行实际后续指令。Raw native Avg、indexed和非零native padding仍是typed
unsupported；源显式padding与count_include_pad除数不在模型中猜测。所有检查和work budget拒绝先于结果写入；
结果只作为pending write发布，input记录同一pending read供NCC完成与reuse验证。这是主机数学范围，不新增板端资格。

### 5.3 `WaferOneDNNBackend` qualification lane

大规模支持项可以进入qualified oneDNN implementation，但必须：

- 通过固定concrete problem、payload、comparator、backend/environment与validated record准入；
- 使用与formal lane相同的typed input/output codec；
- 对unsupported shape/layout/dtype fail closed；
- 保留command count与implementation provenance；
- 不因性能自动扩大numeric capability。

完整output与独立CPU expected比较。整数/bit pattern采用exact；浮点使用case-owned dtype policy和明确容差，同时单独检查
NaN/Inf分类、shape、bytes与guard。

## 6. Aggregate target module 与Tile执行

Grid/Cluster target lowering可以把16个不同Tile body合成一个低层module；model对此不增加第二协议：

- `DeviceExecutable`的Tile interfaces与其target LLVM owner set仍显式列出16个三元组和每Tile ABI；
- internal dispatch通过verified launch-slot relation选择body；
- `tile_id`与`launch_slot`非恒等时结果必须保持一致；
- module count不改变SC_THREAD count、memory ownership、transaction identity或completion gate。

因此aggregate只是module materialization，不能把MPMD降格成“一个body复制16次”或缩减显式Tile domain。

## 7. Failure与atomicity

所有failure归因到明确stage和physical identity：

- invalid DeviceExecutable/program binding/Tile entry argument在JIT前失败；
- decoder/target transaction错误带card/tile/launch slot与issue ordinal；
- address/alias/hazard在issue时失败；
- deadlock/no-progress带pending event/resource摘要；
- numeric/expected mismatch只在完整execution结果形成后报告；
- 任何execution failure调用sink abort并销毁private state，不返回partial output。

模型不修改source program、target LLVM module或package，也不把失败结果回写compiler candidate search。

## 8. Verification

Unit/integration gate至少覆盖：

- 16-Tile all-and-only准备、duplicate/missing/foreign Tile和non-identity tile/slot mapping；
- dense Tile entry arguments、card-shared memory、Tile-local workspace/status与physical codec roundtrip；
- 每个parameter/constant `TargetTensor`只编码一次、引用同一target representation的Tile共享model-private bytes、bounded codec peak window与
  package materialization逐字节一致；
- descriptor registry的每个payload family、bad width/enum/range/format负例；
- 一Tile一SC_THREAD、independent progress、event wait/wakeup、NoProgress和atomic abort；
- local SPM isolation、shared DDR sharing、cross-Tile Direct-DTE、worker/join与reuse hazard；
- formal numeric、`WaferOneDNNBackend` qualification、完整output differential与environment provenance；
- aggregate/nonaggregate module topology具有相同typed Tile interfaces和functional outputs。

Production source/model gate必须重新执行generic mixed DAG、HF prefill、functional two-step decode和Llama block的current
FP16/BF16输入。通过只证明current target functional semantics与CPU expected一致；current package exact-provider、真实board
correctness和performance仍是独立gate。

Whole-program scale默认只要求compiler/package/no-card闭合；只有model capability与host budget明确覆盖完整大图时，
才运行完整target-model differential。因budget或unsupported target call拒绝必须作为typed model limitation记录，不能把
单block model通过写成完整模型证据。

## 9. 不可越过的结论边界

- `TargetModelResult`不证明cycle、bandwidth、NoC contention或board wall time。
- SystemC event ordering不证明vendor queue实现相同，只证明typed dependency合同自洽。
- target-call成功不证明current loader/provider可执行同一module；exact package execution另行验证。
- model与board相关性必须使用current source/config/payload/ABI和held-out cases，不能读取历史raw重新签发。
- Host qualification在fresh package/no-card完整矩阵通过后只能到`board-ready`；真实matched A/B通过前不得标`done`。

### Packed predicate 与 select movement

输入为actual TargetBit2FPCommand/TargetMaskMoveCommand及其SPM内容，输出为精确read/write effects；SystemC仍负责原有worker及completion。
Bit2FP按logical bit解包为目标F16/BF16/F32的0/1。MaskMove消费相同格式的canonical 0/1 mask，true复制source、false保留原destination；
其它mask值保持typed unsupported，不推测硬件语义。两项按原预算约束工作，physical codec保留padding/guard。
覆盖rank3 1024/1025/1031 Bool常量、广播mask、两种半精度、inline与参数常量的source→package→SystemC完整输出；
直接命令检查false保留、packed tail和不支持的mask编码。本项不增加compiler opcode或修改target ABI。

Native控制流白名单允许i32与F32之间的等宽bitcast，以承接mapped scalar读取和Memset raw value；不由此开放native浮点算术、
pointer reinterpretation或浮点control flow。逐bit检查正负零、普通值、Infinity和NaN payload，经16个Tile的实际host JIT转回同一原始字段。

Managed-reference tensor后端在既有F16/F32之外接入BF16。BF16读取是精确扩宽；写回复用共享`convertTargetScalar`的F32→BF16 nearest-even规则，
不另写舍入算法。Same-shape convert和逐元素域使用F16/BF16/F32，归约继续保持原F32 sum/max/min合同，NaN和其它原不支持域仍typed拒绝。
覆盖rank3 1024/1025/1031、正负零、subnormal、普通值、最大有限数与舍入边界，对照formal完整codeword；全LM继续使用原比较合同。

### Formal GEMM 的独立输出并行

输入为已验证的FormalGemmOperation、同次invocation的不可变logical inputs及原work budget。较大的实际GEMM可按输出元素并行；
每个元素仍依次调用原APFloat FMA evaluator，K顺序、psum加法位置和最终转换完全不变。各worker只写自有结果/error/flags slot，
完成后按原row-major顺序汇总，选择首个错误；任一错误仍不发布部分结果或context flags。使用LLVM既有parallel执行设施，不增加MLIR依赖或数值policy。
覆盖rank3 1024/1025/1031、F16/BF16/F32、transpose、psum及flags，逐codeword对照同一公开scalar evaluator组成的串行oracle；
预算拒绝在执行前保持不变。记录work/wall/RSS，并以整层实际TargetModel保持原门限验证。该主机并行不修改任何target同步或设备指令。
