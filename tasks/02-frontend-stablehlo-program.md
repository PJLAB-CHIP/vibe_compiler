# Wafer Frontend 与 StableHLO Program Directory 设计

本文只拥有StableHLO program directory、
metadata/payload和frontend verification合同；`num_partitions`描述card partition，不描述单卡16个Tile。
typed model/state/resource graph与Tile级physical-dataflow planning属于下游，不是frontend事实。实现状态看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  production只接受framework adapter或pre-exported source生成的StableHLO program directory，其中
  `functions/forward.stablehlo.bc`是唯一IR authority。显式text module只进入IR-local frontend verifier，
  不是`wafer-compile`的第二种production输入。
- Current stage responsibility:
  parse并verify外部StableHLO module，在owned module中合法化下述已定义的数学扩展，再做严格source验证；
  校验单entry function与`forward.meta`的shape/dtype/arg-role关系；校验
  parameter/external captured-constant NPY payload；对post-SPMD program校验canonical `forward.meta.distributed_boundary`、
  parameter shard metadata、logical card-partition domain和payload coverage；拒绝graph break、eager fallback、无界
  dynamic shape及不安全路径。当前program-directory入口只接受static ranked boundary；bounded dynamic仅由
  IR-only frontend verifier检查，尚未进入directory schema或production lowering。
- Output IR / files:
  verified current StableHLO module及program directory的metadata/payload；磁盘portable source保持原字节，
  下游消费已合法化的owned module。它仍由IR、`forward.meta`和必要payload共同组成，不是
  `TensorProgram`、top-level TileModule set、`DeviceExecutable`、target module或`ExecutablePackage`。
- Downstream consumer:
  compiler transaction在owned snapshot上建立card-partition mesh，调用pinned XLA SPMD helper，
  然后做local compute normalization并发布verified card-local structured tensor program；05 structured
  optimization继续在同一IR上建立verified `TensorProgram` boundary。06随后以target physical topology为独立输入，
  形成top-level TileModule set并联合搜索spatial placement、temporal tiling、TileRegion/融合与communication。
- User-level driver / named pipeline:
  `wafer-verify-program --program-dir`只做program-directory advisory verification；继续编译只经
  当前`wafer-compile --input-program-dir=... --output-dir=... --num-partitions=1 --optimization-policy=search|none`；
  source-to-package optimization policy只为`search|none`，current target identity由compiler固定提供，不是用户选择。
  `wafer-opt`的`wafer-frontend-verification` named pipeline只处理显式IR，不拥有program-directory I/O。
- Explicit non-goals:
  不定义typed model/state ABI、MPMD member graph、physical endpoint、layout、SPM/DDR allocation、DTE、
  target ABI、manifest或runtime handle；不从parameter/function/file名恢复后端语义。
- Done criteria:
  当前实现门禁是产品adapter生成的真实PyTorch/XLA exporter产物及pre-exported portable fixtures通过同一program ingestion，metadata、
  payload、static boundary和post-SPMD shard负例在进入下游前fail closed；SPMD handoff直接消费该verified output，而不是重建第二份
  frontend对象模型。产品`wafer.frontend.export_pytorch_program`与pre-exported portable StableHLO进入同一ingestion；installed
  `wafer-verify-program`只复用该ingestion做advisory validation，compiler transaction仍独立snapshot、重新验证并构造
  `CompilationRequest`。
```

## 2. 当前 Program Directory 合同

当前production输入是一个single-entry program directory：

```text
program/
  functions/
    forward.stablehlo.bc                # 唯一production IR authority
    forward.meta
    forward.parameter_shards.json       # 仅post-SPMD program存在
  data/<parameter>                      # pre-SPMD immutable parameter NPY
  constants/<position>                  # exporter-captured constant NPY
  parameter_shards/<parameter>/<file>   # post-SPMD logical shard NPY
  ...                                   # 其它exporter成员按原字节保留
```

portable artifact反序列化后的function boundary与`functions/forward.meta`必须一致；任何shape、dtype、arg/result数量或role
不一致都使整个program非法。`forward.mlir`与旧`forward.bytecode`在source directory中直接拒绝，不存在reader fallback；text IR只在
program directory外作为diagnostic/development输入。其它非IR exporter成员按原字节保留。

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

SPMD helper是外部进程边界，不是一个可继续传裸C++引用的pass。Compiler transaction必须让helper消费transaction-owned、content-stable的
all-and-only IR/metadata/ProgramDataRange，并把真正产生的新shard作为`ProgramDataSource` readback接管；helper不能重新打开原source path、整树复制
或整NPY读入host vector。current product compiler只接受`num_partitions=1`，多partition shard只在frontend/helper isolated gate中
证明，不冒充source→DeviceExecutable完整产品路径。

该边界不定义checkpoint registry、framework adapter、target packing、package schema、
device residency或compute-time weight streaming。

### 2.2 产品 frontend 与外部 StableHLO 边界

#### 原始 module 的直接 XLA capture

- Upstream IR / input：eval 的原始 `torch.nn.Module` 与静态 CPU tensor tuple；模块参数及 buffer 为本次不可变状态。
- Current stage responsibility：在独立 module/input 副本上直接执行 XLA lazy forward，不要求先产生 Dynamo/ExportedProgram；
  从实际输出根提取 StableHLO，并由 XLA device-data 的 tensor identity 绑定输入、参数、buffer 与 captured constant。
  复合算子的既有 opmath 合同通过同一 pinned ATen decomposition 保持，不按模型或 shape 选择算术。
- Output IR / files：原 `forward.stablehlo.bc`、同一 metadata schema 和逐 bit 保留 dtype 的 NPY；只发布完整目录。
- Downstream consumer：原 program-directory ingestion、SPMD 与 compiler；原模型 CPU eager 仍独立产生 reference。
- User-level driver / named pipeline：`wafer.frontend.export_pytorch_program`是原始module的唯一抓图入口；
  板测全部模型以及reference/GEMM/MLP工具调用同一入口，不先生成ExportedProgram，也不在失败后改走Dynamo。
  显式SPMD工具保留其XLA sharding设置和输出根导出，不引入第二条Python字节码抓图路径。
- Explicit non-goals：不改模型、参数、依赖版本、runtime、搜索与板端执行；不把 runtime tensor 移到 CPU 计算，
  不接受数据依赖 Python 分支或状态修改，不扩展输入端口 schema。
- Completion criteria：原 module 不受 capture 影响；混合端口、共享参数、buffer、多输出与尾部通过真实导出和 ingestion；
  运行时输入必须 all-and-only 出现在图的输入绑定中。标量转 Python 只允许实际 XLA 子图无 device-data leaf 的常量表达式；
  其余标量读取、CPU fallback、显式graph step及输入/状态 mutation 在目录发布前拒绝；
  动态端口继续由原 portable/source verifier 拒绝，不扩展 directory 的静态边界合同。

依据为 [PyTorch/XLA 2.5 输出根导出 API](https://docs.pytorch.org/xla/release/r2.5/index.html#torch_xla.core.xla_model.get_stablehlo)
及 pinned `xla_model.py`、`stablehlo.py` 的实际 device-data 绑定和 bundle serializer；不用 `save_torch_model_as_stablehlo`
包一层代替直接导出，因为该便利函数仍调用 `torch.export`。

| 输入等价类 | exact 结果 / 失败 | 直接下游 witness |
| --- | --- | --- |
| rank3、S1024/1025/1031，FP16/BF16、i64 输入、多个输出、共享参数及非 persistent buffer | 输入位置与实际 XLA identity 一致，所有 payload bit-exact，原 module/input 不变 | portable ingestion、实际 XLA CPU 执行与独立 PyTorch reference |
| 仅由静态 shape 构造的标量表达式 | actual scalar 子图无外部 leaf 才可读取；原结果和 runtime 输入仍保留 | 原始完整 LM 的 mask 构造及 S16/1024/1025 source |
| runtime-dependent item/CPU transfer、unused/重复端口、状态 mutation、非 tensor 输出 | 明确失败且不发布目录；失败后下一合法导出可用 | 产品 API 负例与后续合法 export |
| Conv+bias、inference BN、GELU/LayerNorm | 保持既有 opmath、approximate 选择和 dtype；无模块名分支 | 既有精度测试与 source→Linalg |
| MaxPool/AvgPool，rank4、1024/1025、FP16/BF16 | AvgPool保持F32累加和原结果dtype；count_include_pad、自定义divisor及adaptive窗口按原调用保留 | 原始module→直接XLA→全部输出，禁止Dynamo下实际执行 |
| 完整单层 LM、i64 ID→全部词表 logits | 原 embedding、decoder、final norm、LM head 均进入图；新输入改变完整输出 | 原始 HF eager、source ingestion；package/no-card/board 分别登记 |
| 共享板测helper、reference/GEMM/MLP工具 | 在禁止`torch.export.export`的环境中调用同一产品入口；参数、buffer及端口仍逐bit正确 | 原始ResNet-18 source与正式compiler；既有helper合同 |

唯一backend输入仍是本节定义的一种program directory。framework adapter是directory的producer，而不是把Python/framework
object直连C++ compiler的第二入口。产品adapter只负责capture/export、拒绝graph break/eager fallback/unsupported side
effect、写入metadata与外部数据引用并调用共享verifier；workload corpus、seed、CPU oracle、模型名分支、target topology和
optimization policy都留在adapter之外。

AvgPool的低精度输入在原始typed调用边界显式保留F32 opmath，再转回原输出dtype。
`adaptive_avg_pool2d`还需在公开调用边界处理：pinned PyTorch的1×1结果先走`mean`，不会经过
`_adaptive_avg_pool2d` dispatch。依据为[PyTorch 2.5 adaptive入口](https://github.com/pytorch/pytorch/blob/v2.5.0/aten/src/ATen/native/AdaptiveAveragePooling.cpp)
及[CPU累加实现](https://github.com/pytorch/pytorch/blob/v2.5.0/aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp)。
这不授权修改普通`mean`、convolution或GEMM的原始算术，也不消除source中显式sum与归一化的舍入边界。

PyTorch eval BatchNorm在导出前通过pinned `torch._decomp`的inference分解显式保留opmath：
FP16/BF16输入和参数在F32计算，输出回到原dtype；F32/F64保持自身精度。只处理typed ATen inference调用，
不按module名/shape分派，不展开training或改变running statistics。已显式的primitive算术保持原样。
`native_batch_norm`、`_native_batch_norm_legit`及functional形式的training参数必须是literal false；
`_native_batch_norm_legit_no_training`复用同一官方分解链。原 module 在 typed ATen 调用边界执行该分解；
显式 ExportedProgram 输入在其图上执行同一官方分解，再交给 XLA。Pre-exported StableHLO由05号按自身dtype合法化，
不从PyTorch合同反推来源。
完成覆盖为FP16/BF16/F32、1024/1025/1031、affine有/无、非平凡mean/variance/scale/bias、原模块参数不变、
分解图全量eager对比，以及portable source→official Linalg的直接下游检查；training和无BN图为不修改分支。
依据是[PyTorch官方分解](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/_decomp/decompositions.py)
及仓内pinned版本的`native_batch_norm_helper`；不在adapter手写第二套BatchNorm数值实现。

当前真实PyTorch/XLA路径由产品`wafer.frontend.export_pytorch_program`承载；program IR authority是
`functions/forward.stablehlo.bc`中的StableHLO portable serialization，text MLIR只作可选诊断
且不进入program directory；pre-exported input与framework adapter output进入
同一compiler-owned snapshot、deserialize/parser和verifier。不保留text/portable双reader、raw StableHLO旁路或可跨source
mutation复用的“verified path”。旧测试generator调用产品export/save policy，只保留case、oracle与corpus职责。

产品adapter和source verifier必须复用同一ingestion实现；advisory verifier只报告当前路径是否通过检查，compiler仍在自己的
transaction中重新打开、拥有并验证全部输入。参数内容的`ProgramDataSource`/`ProgramDataRange`生命周期由本文件2.1节负责；
产品adapter不重建参数管理层或plugin registry。

#### 外部数学扩展的输入合法化

- Upstream IR / input：同一portable artifact反序列化得到的verifier-valid外部StableHLO；包含pinned XLA公开扩展协议编码的数学调用。
- Current stage responsibility：在ingestion拥有的module中核对完整外部合同，将支持的数学扩展转成CHLO，再用pinned官方
  `chlo-legalize-to-stablehlo`分解；之后仍运行原严格source verifier。文件和metadata不改，失败的module整体销毁。
- Output IR / files：只有builtin/func/StableHLO语义的current module，原function signature、参数/常量身份及dtype保持；不输出新文件格式。
- Downstream consumer：metadata/payload校验及SPMD helper；helper的输入必须从该current module序列化，不能重读原始扩展后绕过合法化。
- User-level driver / named pipeline：同一`deserializeStableHLOProgramDirectory`供advisory和生产compiler调用；不增加第二个导出runner或宽松source入口。
- Explicit non-goals：不接受任意external call，不增加设备数学ABI，不根据模型名识别，不从低精度primitive图反推PyTorch opmath，
  不改变GELU的approximate选择、不手写另一套erf多项式。
- Completion criteria：支持项完整合同验证、官方分解、数值/特殊值及真实framework source到下游；未知target/version、effect、alias和附加配置拒绝。

本项支持XLA `mhlo.erf`、`mhlo.version=1`、空`mhlo.attributes`的公开编码：恰好一个静态浮点tensor输入和相同type的结果，
无side effect、called computation、alias、backend config或layout约定，API为原始默认形式。
允许的float为pinned CHLO明确支持的F16/BF16/F32/F64；低精度erf的内部计算采用官方F32分解，结果保持调用本身的dtype。
未知扩展仍被拒绝；名字在此处是外部协议显式的`call_target_name`字段，不用于恢复其它operation或workload语义。
无该扩展的source不运行分解pipeline，保持原IR。

PyTorch的typed `aten.gelu`与`aten.native_layer_norm`在直接XLA forward期间复用pinned `torch._decomp`官方分解，显式保留算子opmath和
完整结果dtype。输入是原始module中的ATen调用，输出直接进入同一XLA lazy图；不再经ExportedProgram，已有primitive计算不重排。
GELU的none/tanh选择、LayerNorm的normalized axes、affine参数及mean/rstd端口按原调用保存，不依据module名称或shape分派。

覆盖矩阵：rank3+、1024/1025/1031、F16/BF16/F32，GELU none/tanh、正负/近零/饱和与非有限值；
LayerNorm单/多normalized axes、affine有/无、三个结果、参数不变及无相关算子的no-op分支。
扩展格式另覆盖F64、错误version/target/shape/effect/alias/config与混合合法/非法调用，失败不发布部分产物。
真实ViT整除/尾部source必须经过同一ingestion和05号下游；source通过不代签package、实卡数值或性能。

数值oracle边界：普通有限GELU输入对照原默认PyTorch eager；非有限输入对照同版本ATen kernel及官方分解。
pinned oneDNN在BF16/F32的GELU none中将正无穷返回NaN，而ATen与官方reference返回正无穷，不能将此后端差异
当作数学扩展的合同。定向CPU测试只在自己的进程内选择ATen检查特殊值，不改变板测runner的默认reference或容差。

算法依据：XLA的[公开扩展编码](https://github.com/openxla/xla/blob/main/xla/mlir_hlo/mhlo/transforms/hlo_legalize_to_stablehlo/hlo_legalize_to_stablehlo.cc)
与[StableHLO/CHLO分解边界](https://openxla.org/stablehlo/spec#dialect-interop)，具体格式和API以pinned源码确认。
相较于放开opaque调用或替换GELU为tanh，本项消解已定义的数学扩展并保留原算子选择；opmath复用
[PyTorch官方reference实现](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/_refs/nn/functional/__init__.py)，不维护第二套数值算法。

### 2.3 低精度算子的内部计算边界

输入为当前`ExportedProgram`中仍带显式bias operand的ATen convolution。FP16/BF16输入、weight与bias先转换为
FP32，convolution和bias加法在FP32完成，整个算子结果再转换回原dtype；输出的StableHLO SSA必须直接表达这些转换。
直接消费者仍是同一portable program ingestion和05号official legalization，用户入口仍为`export_pytorch_program`。
使用pinned `ExportedProgram.run_decompositions`，不修改用户module、输入、state或PyTorch eager reference。

无bias convolution、F32/F64和显式的低精度`conv → add`保留原算子边界；不能在StableHLO阶段从相邻add猜测bias。
这不是所有算术统一升精度，也不保证不同CPU/device累加顺序逐bit相同。Pinned PyTorch/XLA
`BuildConvolutionOverrideableBias`会在低精度卷积后生成独立低精度add，因此提升必须发生在该信息丢失前。
完成条件为带/不带bias、FP16/BF16/F32、参数state与真实规模tail的export/verifier验证，以及原组合case完整PyTorch验收；
具体矩阵与本轮实卡证据由统一board-testing计划拥有。

### 2.4 Safe-softmax 的导出表示

输入是原始module直接XLA抓图时的typed `aten._safe_softmax`调用。复用pinned PyTorch官方decomposition，
保留dim、显式dtype与全屏蔽行归零语义；FP16/BF16使用F32内部opmath，再转回调用的输出dtype。
普通`torch.softmax`仍保持自己的非有限值行为，不把它替换为safe-softmax。输出是普通StableHLO SSA，
由source verifier、SPMD及05号structured入口消费；SDPA内部的展开同时保存在2.6节composite定义中。

安全分解的region capture必须成为composite展开函数的真实operand。Pinned PyTorch/XLA的边界收集按
`getUsedValuesDefinedAbove`遍历region，不能因顶层op没有直接operand而漏掉归约region使用的外部constant。
必要capture保留在portable函数signature中；target ABI只在证明参数无use后删除对应device参数，不删source binding。

覆盖rank3/4、1024/1025/1031、FP16/BF16/F32、显式dim/dtype、普通有限行、部分mask和全负无穷行。
检查官方safe-softmax完整数值、全屏蔽exact zero、原mask和caller state不变；普通softmax的NaN行为单独检查。
导出/识别通过不代签package、no-card或板端完成。分解以仓库pinned PyTorch/XLA源码及真实capture回归为准。

导出器的受控源码身份为pinned PyTorch/XLA revision加仓内`pytorch-xla-composite-region-captures.patch`。
该补丁让composite closure包含owned region引用的外部SSA定义；不改变framework算术或reference。
构建入口幂等应用同一补丁，corpus provenance记录补丁与修改文件的SHA-256；来源检查要求整个checkout的
tracked变化集合及各文件字节均与记录相等，并拒绝其它变化或untracked文件。完成覆盖包含原revision缺补丁、
额外文件、补丁哈希错误、补丁文件之外的修改，以及真实corpus的重复导出和固定reference。

### 2.5 Attention 的低精度导出

输入为原始eval module在直接XLA capture中调用的ATen SDPA，Q/K/V为FP16或BF16。
按用户明确授权，capture期间启用pinned PyTorch的`allow_fp16_bf16_reduction_math_sdp(True)`，
由原SDPA math实现保留输入精度，继续处理scale、causal、mask及head/batch关系；不在已导出的IR上删除转换。
该选项只拥有SDPA内部的自动提升策略，显式F32输入及其它算术保持原dtype；普通softmax继续遵循2.4节的F32 opmath，
普通低精度score的概率在PV前恢复低精度，不把softmax指数与归约一并降精度。
展开定义保留框架对显式F32 additive mask的算术；02号不自行删cast。05号在composite边界应用统一宽状态合同，
Q/K/V保持原storage，mask进入F32 score，概率只在PV输入处窄化。
用`finally`恢复调用方原设置，包括导出失败；独立CPU reference始终在capture之外由原module生成。

输出为同一portable StableHLO program directory，低精度GEMM及其转换进入composite展开定义的SSA，
下游仍为原source verifier、SPMD、05号attention识别和正式compiler；唯一入口为`export_pytorch_program`。
不增加模型/shape特判、依赖版本、精度开关CLI、第二条导出形式或后端dtype fallback；不放宽数值标准，
不改已有portable source、module、参数或reference。硬件累加/psum支持继续由当前target合同约束。

依据为[PyTorch数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-in-scaled-dot-product-attention-sdpa)
与[pinned 2.5 SDPA实现](https://github.com/pytorch/pytorch/blob/v2.5.0/aten/src/ATen/native/transformers/attention.cpp)。
采用已有math实现的显式策略，避免复制其mask/scale分解或在lowering猜测转换来源。

| 覆盖 | exact要求 | 完成条件与直接下游 |
| --- | --- | --- |
| rank3/4、1024/1025/1031，FP16/BF16 SDPA及F32对照 | 两次dot的输入dtype、输出dtype正确；boolean/additive mask、负scale和causal保留；展开定义保留显式F32 mask语义；原输入及框架设置不变 | 真实portable export/source verifier及XLA完整输出对原PyTorch相似度验收 |
| 调用方选项原为开/关，合法导出与异常退出 | capture采用唯一策略，退出恢复原状态；显式F32算术不被降精度 | 产品API回归及失败后合法导出 |
| 原ViT S1024/1025与LLaMA/大GEMM保护 | 原reference和既定门槛不变，所有完整输出参与比较 | 默认8/42正式构包、fresh no-card、串行实卡；变化的保护包做完整数值及匹配性能 |

导出数值通过不代签完整ViT上板；SPM、target和实卡结果分别记录。

### 2.6 Attention composite 导出合同

产品capture在typed SDPA调用处建立本节唯一composite；2.5节定义其框架展开策略，05号定义structured消费合同。
实施和实卡资格见[统一板测计划](plans/board-workload-matrix.md#attention导出展开与实卡验收)，不由本设计代签。

- Upstream IR / input：原始eval PyTorch/HF module及同一份静态typed输入，attention调用处仍可取得causal、位置和有效长度。
- Current stage responsibility：通过现有产品capture保留逻辑attention的`stablehlo.composite`边界；
  QKV projection、RoPE、KV更新和output projection保持各自SSA边界。展开定义来自框架计算图，准确描述其算术与转换。
- Output IR / files：同一portable program directory；composite、必要operand/attribute和展开函数都在StableHLO中，
  不增加旁路语义文件。磁盘metadata/payload仍与真实function signature逐项一致。
- Downstream consumer：source verifier、pinned SPMD和05号StableHLO到structured转换；后者在通用展开前形成现有attention op。
- User-level driver / named pipeline：仍为`wafer.frontend.export_pytorch_program`及原compiler入口，named pipeline复用同一转换。
- Explicit non-goals：不在导出层选择tile、layout、同步或硬件指令；不新增opaque runtime custom-call执行路径，
  不从模型名、参数名、sample mask数值恢复causal，不用展开函数生成测试expected。
- Completion criteria：真实module经portable roundtrip、source verifier和SPMD后仍保留必要语义，直接进入05号attention；
  本节正例继续到package/no-card及实卡，不能由composite打印成功代签。

Q/K/V、scale、必要的additive mask及GQA head关系必须可由输入type、SSA与明确属性解释。
当前composite名为`wafer.scaled_dot_product_attention`，version为0；唯一属性集合为
`is_causal`、`scale`、`has_mask`、`enable_gqa`、`query_start`、`key_start`、`key_valid_end`。
Q/K/V及可选mask在前，框架展开的scalar captures在后。Static shape与typed `CausalBias`给出位置；
upper-left及lower-right遵循原框架描述，descriptor可为module字段或在forward中构造。普通causal位置从0开始。
05号把位置属性变为index SSA；任意运行时mask保留真实tensor输入，不从payload恢复位置/有效长度。
KV有效prefix由原module切片表达；bool mask、padding与任意浮点bias不得混同。
普通causal不再以完整`[S,S]`mask作为产品输入。任意additive mask仍是实际输入，不根据某次三角形payload改写成causal。
导出、SPMD和导入都须保留这些事实；composite的展开语义与属性不得矛盾，未知或不完整合同明确拒绝。

选用[StableHLO composite](https://openxla.org/stablehlo/spec#composite)是因为它保留融合边界并携带等价展开定义；
`custom_call`仅由实现约定语义。本仓pinned StableHLO已有CompositeOp、pinned PyTorch/XLA已有
`StableHLOCompositeBuilder`；直接XLA capture和SPMD传递同一调用，SPMD无法保持已分片composite的shape合同时明确拒绝。
ATen dispatch作用域限定在Tensor API调用内，避免pinned PyTorch的descriptor构造与dispatch mode冲突；
实际Tensor计算、mutation/host逃逸检查和官方分解仍由同一capture负责，不能另建Dynamo入口。
`MultiheadAttention`的Python functional body会在override dispatch期间关闭外层mode。
capture复用pinned原函数code，仅在其私有globals副本中跳过已消费的override入口检查，重新启用同一mode捕获内部SDPA；
不修改框架全局函数、参数或模型数学。FP16/BF16、1024/1025由原module完整输出及structured attention witness共同验证。
composite本身不授权浮点重排；05号拥有attention混合精度及online重排边界，16号拥有独立module reference。

| 输入等价类/分支 | exact要求或typed failure | 直接下游witness |
| --- | --- | --- |
| FP16/BF16，MHA/GQA，S1024/1025/1031及Q/K/V均为`[1,28,4096,128]` | signature/dtype不变；融合边界、head关系、causal与位置完整；普通causal没有dense mask输入 | 实际module→portable→SPMD→attention→package/no-card→实卡 |
| 非方形prefill、单/多token decode、padding、任意additive mask | query/key绝对位置及有效域保持；不能仅用局部Q/K长度推导causal方向 | 05号main/tail与16号原module数值 |
| 缺少位置、错误属性/type、展开函数不闭合、额外score输出 | verifier或转换明确拒绝不满足的融合合同；额外observable use不得删除 | 主机负例及原SSA存活检查 |

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
只写marker、只改local function signature或另加sidecar都不能建立post-SPMD distributed handoff。schema当前为：

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

Source-backed corpus至少包含普通multi-op图、attention/decode和parameter-heavy完整block。每个正式case记录source
revision、config、seed、dtype、shape及payload/reference digest，并由同一framework eager执行产生用户级expected。
重复export必须得到canonical-equivalent program；diagnostic variant显式记录实际seed和`verification=false`，不能改写
正式corpus。具体case、shape、payload生成和历史characterization属于测试资产或archive，不进入frontend协议。

旧TP16/rank-as-Tile资格只作历史背景，不属于current frontend合同。当前GEMM、HuggingFace attention、
KV-cache decode与Llama-2 7B block都从真实framework module和原始dtype tensor导出
`num_partitions=1`的card-local program；source IR不携带物理Tile mesh或卡内TP标记。Physical-dataflow stage随后
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

compiler transaction把pre-SPMD snapshot交给pinned helper，由helper内部完成Shardy/XLA SPMD并返回local program与parameter
shards。当前无用户sharding的correctness基线允许helper产生replicated结果。IR-local默认Shardy seed实验会留下
helper不能消费的`sdy.constant`/`sdy.reshard`等语义，因此在完整SDY→StableHLO bridge存在前不进入production
driver。

helper输出必须重新走本文件的metadata/payload verifier，包含上述distributed boundary，并与
`ExecutionConfig::numPartitions`建立的logical execution mesh逐项一致；Tile数量只来自target topology，
不得在此处用partition数推导。
SPMD handoff随后进入StableHLO-to-Linalg/collective normalization，并发布verified card-partition-local structured
tensor program。frontend verifier本身不执行helper、不形成调度单元，也不公开SPMD stop-stage。

## 6. Ownership 与失败语义

当前`CompilationRequest`是move-only C++ value，只拥有source program locator和validated
`ExecutionConfig`。它不持有MLIR operation/context、helper path、output path、pass callback、candidate policy、
target context或writing authority。

source-to-package driver在parse前建立transaction ownership：IR、metadata和目录结构进入私有snapshot，大payload由
content-stable且完成extent/digest校验的`ProgramDataSource`/`ProgramDataRange`持有；后续frontend verify、helper和IR transforms只消费该transaction
拥有的事实。整目录复制或同时保留propagated/original/byte-identical shard payload不是合法隔离机制。
source不得被原地补metadata、topology或shards。compiler transaction最终发布重新parse/verify过的card-partition-local
structured program state；local compute normalization与fixed structured optimization随后建立`TensorProgram` boundary，之后
physical-dataflow synthesis从单卡partition output构造一个top-level TileModule set，其中all-and-only available Tiles
各有独立`wafer.tile.module`。每个Tile可有不同op、loop和temporal tile shape；physical-dataflow owner选择
placement、tiling和TileRegion membership并立即生成actual IR，随后在current IR上物化NoC/DDR movement，exact gates通过后形成`DeviceExecutable`，再由target与
package阶段发布`ExecutablePackage`。`TensorProgram`是该planning阶段的唯一输入 output；不会保留未物化的 group formation
记录、selector 或兼容/debug/发布旁路。

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

完成记录必须区分真实exporter gate、program verifier和下游SPMD handoff/TensorProgram normalization gate。FileCheck、手写MLIR或CPU oracle单独通过
都不能证明card-local spatial/temporal/fusion/communication scheduling、`DeviceExecutable`、target modules、`ExecutablePackage`、runtime
或board正确。

### SiLU 的 source opmath

直接Torch XLA抓图时，aten.silu复用pinned PyTorch官方decomposition及其opmath wrapper；F16/BF16输入在F32完成sigmoid和multiply，
仅在完整算子结果上窄化一次。该规则按真实aten operator分派，不改原模型，不将模型运算移到host，不把sigmoid结果提前窄化。
覆盖rank3 1024/1025/1031与F16/BF16，检查实际导出StableHLO的算术dtype及CPU/XLA结果；完整单层LM保持原数值合同。
