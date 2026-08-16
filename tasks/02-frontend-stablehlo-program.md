# Wafer Frontend 与 StableHLO Program Directory 设计

状态：2026-08-15按card-level GSPMD、产品frontend入口与program-data ownership边界同步。本文只拥有StableHLO program directory、
metadata/payload和frontend verification合同；`num_partitions`描述card partition，不描述单卡16个Tile。
typed model/state/resource graph与Tile级时空综合属于下游，不是frontend事实。实现状态看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  production只接受framework/exporter生成的StableHLO program directory，其中`functions/forward.mlir`可来自
  pre-exported StableHLO且可包含frontend `mhlo.sharding`。显式module只进入IR-local frontend verifier，
  不是`wafer-compile`的第二种production输入。
- Current stage responsibility:
  parse并verify StableHLO module；校验单entry function与`forward.meta`的shape/dtype/arg-role关系；校验
  parameter/external captured-constant NPY payload；对post-SPMD program校验canonical `forward.meta.distributed_boundary`、
  parameter shard metadata、logical card-partition domain和payload coverage；拒绝graph break、eager fallback、无界
  dynamic shape及不安全路径。当前program-directory入口只接受static ranked boundary；bounded dynamic仅由
  IR-only frontend verifier检查，尚未进入directory schema或production lowering。
- Output IR / files:
  verified StableHLO program directory。它仍由MLIR、`forward.meta`和必要payload共同组成，不是
  `TensorProgram`、`CardModule`、`CardExecutable`、target module或`ExecutablePackage`。
- Downstream consumer:
  Q15 typed compiler driver在transaction-owned snapshot上建立card-partition mesh，调用pinned XLA SPMD helper，
  然后做local compute normalization并发布verified card-local structured tensor program；05 structured
  optimization继续在同一IR上建立verified `TensorProgram` boundary。06随后以target physical topology为独立输入，
  形成`CardModule`并联合搜索spatial placement、temporal tiling、TileRegion/融合与communication。
- User-level driver / named pipeline:
  `wafer-compile-stablehlo --verify-stablehlo-program`只做frontend verification；继续编译只经
  当前`wafer-compile --input-program-dir=... --output-package-dir=... --num-partitions=1`；
  Q59会按真实package destination原位改名并同步全部consumer，不保留旧alias；
  source-to-package optimization policy只为`search|none`，current target identity由compiler固定提供，不是用户选择。
  `wafer-opt`及named MLIR pipelines只处理显式IR，不拥有program-directory I/O。
- Explicit non-goals:
  不定义typed model/state ABI、MPMD member graph、physical endpoint、layout、SPM/DDR allocation、DTE、
  target ABI、manifest或runtime handle；不从parameter/function/file名恢复后端语义。
- Completion gate:
  当前实现门禁是测试owner生成的真实PyTorch/XLA exporter产物及pre-exported text fixtures通过同一program verifier，metadata、
  payload、static boundary和post-SPMD shard负例在进入下游前fail closed；Q15直接消费该verified output，而不是重建第二份
  frontend对象模型。Q60完成门禁另要求产品`wafer.frontend.export_pytorch_program`产出的program与pre-exported portable
  StableHLO进入同一ingestion；installed `wafer-verify-program`只复用该ingestion做advisory validation，compiler transaction仍独立
  snapshot、重新验证并构造`CompilationRequest`。在Q60完成前不得把测试generator写成产品frontend。
```

## 2. 当前 Program Directory 合同

当前production输入是一个single-entry program directory：

```text
program/
  functions/
    forward.mlir
    forward.meta
    forward.bytecode                    # 可选，保留但不作为当前IR事实源
    forward.parameter_shards.json       # 仅post-SPMD program存在
  data/<parameter>                      # pre-SPMD immutable parameter NPY
  constants/<position>                  # exporter-captured constant NPY
  parameter_shards/<parameter>/<file>   # post-SPMD logical shard NPY
  ...                                   # 其它exporter成员按原字节保留
```

`functions/forward.mlir`与`functions/forward.meta`共同拥有function boundary事实；二者任何shape、dtype、
arg/result数量或role不一致都使整个program非法。bytecode和其它非IR成员可以随directory保留，但当前compiler
不从它们恢复语义。某stage没有明确typed rewrite合同时，不得静默丢弃或改写这些成员。

`forward.meta`当前消费的字段是：

- `name`，当前单入口output必须精确为`forward`；
- `input_signature[]`与`output_signature[]`中的shape/dtype；
- 与function arguments一一对应的`input_locations[]`；
- input location的`type_`只接受`parameter`、`constant`或`input_arg`。

`input_arg.position`必须非负、唯一并从0连续；parameter locator必须是program内安全的单路径组件，不能使用
absolute path、`..`或路径分隔符逃逸program root。name/path只负责在当前container内定位payload和形成诊断，
不能成为candidate/schedule、sharding、rank、resource或lowering分支条件。

post-SPMD时同一份canonical metadata还包含`distributed_boundary`；这是function boundary由global tensor变成
per-card-partition local tensor的typed output合同，不是planner sidecar，也不是Tile placement。若未来需要typed mutable state、alias/mutation、
多个entry或program graph，必须先设计可由IR/metadata verifier证明且有下游consumer的最小表示；不能恢复历史
私有model dialect、复合frontend owner或side-table对象图作为前置。

### 2.1 Program data ownership 与 transaction lifetime

parameter/external captured-constant的逻辑身份、source bytes和target物理表示是三个不同边界。frontend verified output对每个payload只建立：

- 显式logical parameter/external captured-constant identity及其function argument关系；
- transaction拥有的`ProgramDataSource`；它必须由private snapshot、pinned content或其它能证明整次读取内容稳定的owner支撑，
  仅持有一个regular-file descriptor、mtime或path不构成immutable证明；
- source内checked half-open `ProgramDataRange`，包括logical dtype、shape和partition/slice descriptor；post-SPMD card slice由SPMD handoff另行形成；
- 内容digest与source provenance，用于证明读取的bytes未变；digest相等不创建alias。current source schema中每个external
  parameter/captured-constant binding拥有自己的`ProgramTensorId`，只有该binding的replication/slice可以共享同一source/range。

path和name仍只用于当前container定位与诊断。verifier必须检查header、element count、payload extent与整数运算，拒绝truncated、
trailing、overflow及source在transaction期间变化；不得先验证路径再在后续stage无owner地重新打开。NPY是当前source adapter，
不是target layout、target-ready program data或runtime binding。

source snapshot只需要隔离IR、metadata和目录结构事实。大payload通过`ProgramDataHandoff`中的owned source与checked range传递，
不能为了snapshot、propagated program、post-SPMD merge或每个Tile binding复制整棵data tree。SPMD只为实际改变bytes的新card
shard建立新的`ProgramDataSource`；replication或contiguous slice引用已有source和新range，不能同时保留original data和
byte-identical shard来暗示sharing。

SPMD helper是外部进程边界，不是一个可继续传裸C++引用的pass。Q58必须让helper消费transaction-owned、content-stable的
all-and-only IR/metadata/ProgramDataRange，并把真正产生的新shard作为`ProgramDataSource` readback接管；helper不能重新打开原source path、整树复制
或整NPY读入host vector。current product compiler只接受`num_partitions=1`，多partition shard只在frontend/helper isolated gate中
证明，不冒充source→CardExecutable完整产品路径。

该边界由Q58原位替换current复制实现。它不定义checkpoint registry、framework adapter、target packing、package schema、
device residency或compute-time weight streaming。

### 2.2 产品 frontend 与外部 StableHLO 边界

唯一backend输入仍是本节定义的一种program directory。framework adapter是directory的producer，而不是把Python/framework
object直连C++ compiler的第二入口。产品adapter只负责capture/export、拒绝graph break/eager fallback/unsupported side
effect、写入metadata与外部数据引用并调用共享verifier；workload corpus、seed、CPU oracle、模型名分支、target topology和
optimization policy都留在adapter之外。

当前真实PyTorch/XLA路径仍由测试generator承载，forward.mlir仍是current IR事实源。Q60会做一次current-only cutover：
program IR authority原位替换为`functions/forward.stablehlo.bc`承载的StableHLO portable serialization，text MLIR只作可选诊断
且不进入program directory；pre-exported input与framework adapter output进入
同一compiler-owned snapshot、deserialize/parser和verifier。不保留text/portable双reader、raw StableHLO旁路或可跨source
mutation复用的“verified path”。

产品adapter和source verifier必须复用同一ingestion实现；advisory verifier只报告当前路径是否通过检查，compiler仍在自己的
transaction中重新打开、拥有并验证全部输入。参数内容的`ProgramDataSource`/`ProgramDataRange`生命周期由Q58负责，
不在Q60重建参数管理层或plugin registry。

## 3. Frontend Verification

### 3.1 IR 与 function boundary

IR-only frontend verifier先parse MLIR并运行MLIR verifier，然后检查graph-break/eager marker与dynamic bound：

- `wafer.import.graph_break`和`wafer.import.eager_fallback`的truthy marker直接拒绝；
- static tensor shape可直接接受；dynamic ranked tensor必须在对应argument/result上有
  `wafer.frontend.dynamic_bounds`；
- bound数组rank与tensor rank相同，dynamic维的bound为正，static维的bound精确等于static dimension；
- unranked或无finite bound的dynamic boundary拒绝。

program-directory verifier在此基础上要求恰有一个可验证entry，并要求所有argument/result都是static ranked
tensor；metadata signatures的数量、shape和normalized dtype与function type逐项一致。当前`forward.meta`虽保留
exporter的`dynamic_dims`字段，但C++ parser尚未建立它与IR bound的双向合同，因此bounded dynamic module只能通过
`--verify-frontend-program`的IR-local gate，不能进入`wafer-compile`。这项限制必须由后续typed schema与真实
lowering consumer解除，不能把IR-only verification写成production支持。

### 3.2 Parameter 与 constant payload

pre-SPMD parameter来自`data/<parameter>`，captured constant来自`constants/<position>`。verifier要求：

- locator指向regular file；
- NPY header可解析且不是Fortran order；
- NPY shape和dtype与对应ranked tensor精确一致；
- payload extent精确等于由checked element-count和element width推导的raw bytes，不接受trailing data；
- unsupported dtype、overflow、truncated或trailing payload fail closed。

`ProgramTensor`边界的多字节element统一使用canonical little-endian storage。NPY `<f2`可直接进入F16 payload，`=f2`
只在little-endian host上与该合同等价，`>f2`必须拒绝；BF16使用NumPy `|V2`承载已经canonicalize的little-endian raw
16-bit encoding，producer必须显式写little-endian bytes，consumer不能按host-native `uint16`重新解释。该规则只固定
public payload bytes，不把NumPy dtype或host endianness提升为target physical layout。

普通tensor constant最终由StableHLO-to-Linalg路径转换成`arith.constant`或其它`ConstantLike`结构。frontend
不引入`wafer.constant`，也不决定weight packing、physical layout、DDR residency或runtime binding。

### 3.3 Post-SPMD parameter shards

post-SPMD marker存在时，每个parameter必须在
`functions/forward.parameter_shards.json`中有唯一记录。该文件由repo-owned helper与frontend verifier同步演进，
不建立独立版本线。root至少包含：

```text
function
num_partitions
parameters[]
```

`num_partitions`只表示logical card-partition count，不表示Tile数量。历史`logical_rank_count`/`rank`字段已删除，
frontend不提供双reader或兼容翻译。

每个parameter record通过`argument_index`绑定function argument，并声明`name`、dtype、global/local shape、
`distribution`和每partition shard。`distribution`只接受：

- `replicated`：每个logical card partition有完整global tensor，replica domain完整，所有payload byte-identical；
- `partitioned`：当前每partition一个slice且`replica_id == 0`，所有slice无重叠并精确覆盖global tensor。

每个shard中的`partition_id`必须唯一且位于`[0, num_partitions)`；
offset/size/stride的tensor rank匹配，stride当前为1，slice不越global shape，NPY shape/dtype与slice一致。
metadata中的`num_partitions`必须等于
logical `wafer.execution.mesh` shape product。没有显式subgroup关系的partial replication拒绝，不能从重复offset或
文件名推测replica group。

parameter shard JSON是post-SPMD payload绑定，不是第二份sharding planner、physical endpoint、memory plan或
runtime manifest。它的consumer是frontend/program verifier和后续card-partition output构造；单卡内部16个
`tile_id`不出现在该schema中。

### 3.4 Post-SPMD distributed boundary

pre-SPMD `forward.meta`不得有`distributed_boundary`。明确post-SPMD marker存在时该object必须存在，反向也成立；
只写marker、只改local function signature或另加sidecar都不能建立Q15→Q16合同。schema当前为：

```text
distributed_boundary:
  num_partitions
  inputs[]   # 精确覆盖input_locations.type_ == input_arg
  outputs[]  # 精确覆盖全部function results
```

input/output record分别以`argument_index`/`result_index`绑定原boundary identity，并包含`distribution`、
`global_shape`、`local_shape`、dtype和`partitions[]`中的
`{partition_id, replica_id, offsets, sizes, strides}`。它不拥有nested版本；helper与frontend verifier随同一源码
revision同步演进。历史`logical_rank_count`/`ranks`/`rank`字段已删除。helper从XLA
`HloSharding` typed API生成这些事实，不解析或反推opaque sharding string。

verifier要求record coverage精确且无重复，dtype/local shape与post-SPMD module和`input/output_signature`逐项一致，
`num_partitions`等于logical execution mesh product。`replicated`要求global/local shape相同、每partition完整覆盖且replica id domain
完整；`partitioned`要求每partition `replica_id == 0`、slice不重叠并完整覆盖global tensor，最大slice shape解释local
module shape；stride当前只能为1。partial replication、single-device、manual/unknown/shard-group等当前无法完整表达
的boundary sharding fail closed，不能从字符串、名字或重复offset恢复语义。

## 4. Importer 与真实 Source Corpus

Wafer后端的稳定入口是上述verified program directory，不是某个framework Python API。import adapter可以支持
PyTorch、JAX或pre-exported StableHLO，但必须收敛到相同output和verifier；framework module name、parameter
name与版本workaround只留在adapter或诊断中。

当前PyTorch路径使用source-built PyTorch/XLA exporter产生StableHLO program directory。真实capture gate必须：

- 不以手写StableHLO emitter替代framework/exporter；
- 对graph break、eager/host fallback和无法导出的side effect fail closed；
- 从exporter metadata/payload取得parameter与constant，不在后端按名字重新配对；
- 保存exporter产生的`mhlo.sharding`，不提前转成target Tile或Wafer私有strategy。

Q5.C的source-backed corpus由
`test/Tools/Inputs/workloads/single-card-vertical.json`和真实capture generator拥有。当前固定case是
linear-residual MLP与tiny Llama decoder block；spec记录source revision、config、seed、dtype、shape和payload/
reference digest。独立NumPy oracle、framework CPU交叉检查与重复export canonical-equivalence只证明source
verification，不证明compiler、runtime或board完成。Q20/Q21必须直接消费这些admitted program，不能换成手写
task/instruction fixture。

Q28另以`test/Tools/Inputs/workloads/llama-2-7b-block.json`固定标准Llama-2 7B单block配置：H=4096、
I=11008、32 heads、head dimension 128、FP16、batch 1、sequence 16。generator直接分块填充最终FP16 parameter
allocation，避免为90M-element projection额外建立全尺寸临时数组；最终`expected.npy`必须由同一parameter/input的
PyTorch eager CPU完整block执行产生，手写NumPy路径只作诊断。scale payload使用显式SplitMix64 counter映射：
global row-major index、固定seed和彼此独立的parameter stream共同形成长周期、FP16-exact值，避免matrix axis短周期重复及
跨projection系统性相关；input、全部parameter和expected在digest及output writing前逐项检查finite。重复export必须
得到canonical-equivalent program和固定digest，既有tiny corpus保持冻结而不随scale算法迁移。该case的shape、seed和
payload算法是corpus参数，不进入frontend output协议。

Q31 numeric characterization没有修改上述冻结case的seed、digest或verification。repository test-input generator可以从一个
固定base case显式生成不同seed的diagnostic variant，但必须记录base case、实际seed、动态input/parameter/expected/program
digest和`verification=false`；variant不能写回workload spec、复用固定digest字段或被`--emit-workload-corpus`当作正式case。
variant仍由同一PyTorch eager block产生expected、由同一真实exporter产生program，并进入同一production compiler pipeline；
因此它只扩充有限tested payload domain，不改变frontend program directory、dtype、shape、sharding或参数绑定协议。Q31的两个
variant seed和原固定seed全部通过后，scale case只把source/model comparator policy收紧为`atol=0.004, rtol=0.002`；该字段
不进入source/config/payload/program digest，也不把variant提升为corpus verification。

旧Q44的TP16/rank-as-Tile资格只作历史背景，不属于current frontend合同。当前GEMM、HuggingFace attention、
KV-cache decode与Llama-2 7B block都从真实framework module和原始dtype tensor导出
`num_partitions=1`的card-local program；source IR不携带物理Tile mesh或卡内TP标记。Q49 `none` baseline与Q51 `search`随后
从同一structured DAG决定16个Tile上的spatial mapping、temporal tiling、TileRegion/融合与通信。不得用手写
StableHLO/MLIR、parameter name或测试fixture把这些卡内决定提前编码进frontend。
同一组tensor先在PyTorch eager CPU执行形成唯一用户级expected；NumPy不得参与expected生成或最终结果比较。
exporter因NPY output格式使用NumPy作payload序列化属于adapter transport，不取得数值参考结果的ownership。
普通case以固定seed的PyTorch random API构造输入；周期pattern、one-hot和手写简化公式只用于失败后的定向debug。
这些case、torch raw tensor读写和capture comparator集中在`test/Board/PyTorch/`，目录布局只是测试实现索引，不进入
frontend schema、compiler driver或sharding协议。

## 5. Sharding Boundary

frontend只保存exporter能解释的`mhlo.sharding`等输入事实。没有用户sharding仍是合法StableHLO program；
frontend不合成`sdy.*`、`wafer.spmd.*`或策略字符串。

Q15把pre-SPMD snapshot交给pinned helper，由helper内部完成Shardy/XLA SPMD并返回local program与parameter
shards。当前无用户sharding的correctness基线允许helper产生replicated结果。IR-local默认Shardy seed实验会留下
helper不能消费的`sdy.constant`/`sdy.reshard`等语义，因此在完整SDY→StableHLO bridge存在前不进入production
driver。

helper输出必须重新走本文件的metadata/payload verifier，包含上述distributed boundary，并与
`ExecutionConfig::numPartitions`建立的logical execution mesh逐项一致；Tile数量只来自target topology，
不得在此处用partition数推导。
Q15随后做StableHLO-to-Linalg/collective normalization，再发布verified card-partition-local structured
tensor program。frontend verifier本身不执行helper、不形成调度单元，也不公开SPMD stop-stage。

## 6. Ownership 与失败语义

当前`CompilationRequest`是move-only C++ value，只拥有source program locator和validated
`ExecutionConfig`。它不持有MLIR operation/context、helper path、output path、pass callback、candidate policy、
target context或writing authority。

source-to-package driver在parse前建立transaction ownership：IR、metadata和目录结构进入私有snapshot，大payload由
content-stable且完成extent/digest校验的`ProgramDataSource`/`ProgramDataRange`持有；后续frontend verify、helper和IR transforms只消费该transaction
拥有的事实。当前整目录复制以及propagated/original/shard多份payload是Q58必须删除的实现差距，不能成为长期隔离机制。
source不得被原地补metadata、topology或shards。Q15最终发布的是重新
parse/verify过的card-partition-local structured tensor program directory；fixed structured optimization完成后，
physical-dataflow synthesis从单卡partition output构造一个`CardModule`，其中all-and-only available Tiles
各有独立`wafer.tile.module`。每个Tile可有不同op、loop和temporal tile shape；Q51唯一search owner联合决定
placement、tiling、TileRegion/融合和显式NoC/DDR movement，exact gates通过后形成`CardExecutable`，再由target与
package阶段发布`ExecutablePackage`。`TensorProgram`是该综合阶段的唯一输入output；已删除的`wafer.group`
formation/selector没有兼容、debug或发布旁路。

这种最小owner边界有意不保留历史讨论中的复合frontend/executable owner和model-interface registry链。若未来
确需跨stage不可重算的owner，必须由真实consumer和lifetime bug证明后再引入，不能把未实现对象写成当前架构。

## 7. 验证

frontend mandatory coverage包括：

- 产品PyTorch/XLA adapter与pre-exported portable StableHLO分别产生同一种program directory并进入同一verifier；
- graph break、fallback、unsupported side effect、portable/text事实源混用与重复export canonical equivalence；
- generic GEMM、mixed DAG、official HuggingFace prefill、functional KV-cache decode及card-local Llama-2 7B block
  由真实exporter进入production pipeline；同一
  PyTorch eager tensor形成expected，runtime按同shape/dtype完整capture回读为`torch.Tensor`且不做精度转换，
  再经`torch.testing`比较；
- graph break/eager fallback、metadata length/shape/dtype mismatch；
- program-directory static boundary；IR-only bounded dynamic及unbounded/invalid bound负例；
- parameter/external captured-constant `ProgramDataSource`/`ProgramDataRange` owner、digest、source mutation、shape/dtype/order/truncation/trailing与unsafe path负例；
  F16 `<f2`、host-compatible `=f2`、BF16 `|V2`
  canonical bytes正例及`>f2`拒绝；
- `input_arg` position唯一连续；
- `forward.parameter_shards.json` replicated/partitioned card-partition coverage、gap/overlap、payload一致性和mesh mismatch；
- distributed input/result identity、global/local shape、dtype、partition/replica domain及data/column真实策略；
- pre-exported StableHLO parse/printer及显式IR-local lowering补充测试。

完成记录必须区分真实exporter gate、program verifier和下游Q15 gate。FileCheck、手写MLIR或CPU oracle单独通过
都不能证明card-local spatial/temporal/fusion/communication scheduling、`CardExecutable`、target modules、`ExecutablePackage`、runtime
或board正确。
