# Wafer StableHLO 到 Card-Local Structured Tensor IR

本文拥有post-SPMD StableHLO到target-independent structured tensor IR的normalization合同，以及attention语义识别、
单一graph attention op和FlashAttention/FlashDecoding算法选择；同时定义candidate阶段唯一stateful online form的算法/接口合同。
Tile、temporal block、layout、movement、buffer和schedule由06号
physical-dataflow设计负责。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  frontend已验证的static-ranked StableHLO program，以及GSPMD为一个logical card partition产生的local program；
  function boundary、dtype、shape、parameter/constant payload和card-partition execution mesh保持一致。
- Current stage responsibility:
  先把supported StableHLO collectives规整为typed destination-style tensor ops，再通过仓库pinned官方
  StableHLO-to-Linalg conversion把compute、shape/data movement和constant变成Linalg/Tensor/SCF/Arith/Math；
  折叠可由static IR完全证明的SPMD helper residual，将payload中可证明的投影式Tensor读取绑定为显式Linalg inputs；
  最后从current structured SSA证明完整Q/K/V attention，
  归一为一个`wafer.linalg_ext.attention` op并确定`flash_attention`或`flash_decoding`算法。Attention识别完成后，
  对剩余ordinary pure structured Tensor/Linalg connected component运行一次有界access-relation e-graph normalization，组合并消除
  可证明等价的static reshape、transpose、broadcast、concat和structured compute operand/result access graph；
  已形成的attention op是opaque component barrier，本stage不展开attention，也不在后续candidate或Tile阶段再次运行e-graph。
- Output IR / files:
  一个尚未绑定Tile的card-local structured TensorProgram。普通数学语义由op、region、indexing map、iterator、
  DPS ties、type、SSA/control flow和effect表达；matched attention由一个self-contained semantic op表达；
  card-partition collective仍是typed tensor semantics。Pure component已经使用canonical exact access relation收敛，
  不保留e-graph、e-class ID、rewrite history或其它旁路表示。不产生文件、physical plan或runtime metadata。
- Downstream consumer:
  physical-dataflow planning从normalized current graph的固定semantic roots构造Spatial/Region choice和exact demand/coupled
  contribution/merge。Choice闭合后，candidate structural materialization把graph attention直接转换为每个selected Tile上的
  `online_attention`、三个state endpoint和merge/finalize；第13项从这些current ops生成并立即应用parallel/K2 temporal tiling，
  第14项再机械分解为canonical Linalg/Tensor/SCF。后续layout、movement、bufferization、Instr、completion和memory只从该current IR生成或重算。
- User-level driver / named pipeline:
  `wafer-compile`的`none`与`search`在policy分叉前共同运行同一normalization；
  `wafer-lower-stablehlo-to-linalg`及attention normalization leaf pipeline只用于IR replay和focused tests。
- Explicit non-goals:
  不运行GSPMD，不决定card partition；不选择Tile、KV partition count、temporal block、layout、SPM/DDR、NoC/DTE、
  buffer、worker、schedule、launch slot或runtime binding；不把FA/FD做成physical search axis；不从symbol、operand位置、
  shape模板或workload名称恢复attention；不物化候选TileModule set。E-graph不修改scalar arithmetic region，不做算术结合、
  分配、reduction重排、matmul chain reassociation、compute partition或target/layout选择，也不承担compiler correctness所需的legalization。
- Done criteria:
  official conversion后无StableHLO/SDY residual；attention custom/generic form、verifier、standard interfaces与coupled-state
  query闭合；graph matcher、FA/FD分类、算法reference、planning description、online K2 stateful tiling和late Linalg decomposition均有正负例；
  bounded e-graph对支持的ordinary pure component只产生verified canonical Tensor/Linalg IR，搜索预算结束仍提取已证明的等价式，未得到完整有效提取时保持原component且不改变
  legality；on/off以exact relation/scalar-region proof保持program语义和downstream representability，不要求结构choice集合逐项相同；
  `none`与`search`消费同一normalized TensorProgram；selected prefill/decode分别沿06--15主线形成package/no-card；每个candidate只执行一次
  `attention -> online_attention -> tiled online_attention -> Linalg`转换链。
```

## 2. 稳定 TensorProgram 边界

normal form只包含数学和structured program事实：

```text
func + tensor + linalg + scf + arith + math
  + wafer.linalg_ext.collective.*
  + wafer.linalg_ext.attention
```

必须保持：

- static logical shape、dtype、indexing maps和iterator types；
- DestinationStyleOpInterface的inputs/inits/tied results；
- reduction/collective combiner region及其标量dataflow；
- SCF dominance、loop-carried SSA、recursive effect与speculation结论；
- card-partition execution mesh和frontend已验证的collective group/channel语义；
- attention的Q/K/V、scale、optional mask、output destination、indexing relation和selected algorithm。

这一层没有physical `tile_id`、SPM/DDR encoding、transport route、instruction、target call或package字段。
`num_partitions`是GSPMD的card-partition domain，不能解释为单卡Tile数量。FA/FD算法是graph-level current IR事实，
但其block、partition、merge Tile和physical realization不在本层。

## 3. 普通 Compute Normalization

Wafer复用pinned官方StableHLO legalization，而不是维护按op名分发的第二套converter：

| StableHLO语义 | normalized form | 保留事实 |
| --- | --- | --- |
| constant | `arith.constant`或其它ConstantLike | exact type、shape、value |
| pointwise | `linalg.generic`与scalar arith/math | broadcast/indexing、dtype、scalar body |
| reshape/transpose/slice/concat | tensor/view或structured movement | dimension与static slice relation |
| dot / `dot_general` | matmul、batch matmul或structured generic | batch/contracting dims、indexing、accumulator/result type |
| reduce / supported reduce-window | Linalg reduction或pooling-like form | domain、init、combiner和evaluation order |
| softmax/norm/RoPE/MLP近似图 | 普通reduce/pointwise/view DAG | 原始SSA dataflow；只有完整attention match才形成attention op |

一个contraction能否成为Wafer GEMM由current structured semantics和target lowering合同共同决定，不由参数名或shape模板决定。
普通softmax、precomputed scores或不完整Q/K/V关系的图继续保持Linalg DAG；
它们不因外观相似被提升为attention。

### 3.1 Logistic内部精度

输入为verified function内的`stablehlo.logistic`。在official legalization前，FP16/BF16 logistic转换为
`convert(input, f32) → logistic(f32) → convert(original dtype)`，结果shape与所有用户可见dtype保持原样。
直接消费者为pinned official StableHLO-to-Linalg converter：neg/exp/add/div均在F32计算，只在算子输出处回到低精度。
`wafer-promote-stablehlo-logistic`锚定`func.func`，由production与named pipeline共用的legalization builder调用。
不修改F32/F64 logistic或原IR中的显式低精度exp/add/div，不引入算术重排、target选择或e-graph旁路优化。

算法采用[StableHLO logistic定义](https://openxla.org/stablehlo/spec#logistic)；pinned官方
`MapStablehloToScalarOp.h`当前在输入dtype展开，pinned PyTorch的`torch._refs.sigmoid`使用opmath promotion。
这里先明确opmath dtype，再复用官方展开，不维护第二套logistic converter。
完成条件为FP16/BF16/F32/F64、1024/1025/1031、重复执行与显式primitive保留的IR覆盖，official conversion直接输出
F32 scalar body与单次回写转换，以及原PyTorch source完整数值验收。覆盖矩阵和板端状态归统一board-testing计划。

### 3.2 Convolution的精确输入扩宽融合

Official legalization后的generic convolution若两个输入都来自identity pointwise `arith.extf`，且源为相同FP16/BF16、
accumulator为F32，将扩宽移入该convolution的scalar body。允许跨越constant-zero `tensor.pad`，但必须先从current SSA证明
cast与padding；padding重建为源dtype。算术乘加、顺序、result和bias加法保持F32，输入storage保留低精度。
该变换由`wafer-fold-convolution-input-casts`在official conversion后运行，结果直接交给physical planning；这是精确的
scalar扩宽融合，不属于access-relation e-graph的算术等价探索。不能根据shape/name或历史输入猜测F32值可降为FP16。

### 3.3 BatchNorm inference的显式分解

输入为verified static-ranked浮点`stablehlo.batch_norm_inference`及其`feature_index`、epsilon和四个feature向量。
`Transforms/StableHLO`的function pass在official StableHLO-to-Linalg之前，将其按
[StableHLO规范](https://openxla.org/stablehlo/spec#batch_norm_inference)展开为
`(operand - broadcast(mean)) / broadcast(sqrt(variance + epsilon)) * broadcast(scale) + broadcast(offset)`。
feature向量只沿op声明的轴广播；epsilon从其F32 attribute舍入到当前element type，所有算术保持原dtype和上述顺序。
输出仍为verified StableHLO，直接消费者是唯一official converter；production和named pipeline调用同一builder。
不选择layout、tiling、fusion或transport，不处理training/grad或quantized/dynamic inference，不重排已有primitive图。

Pinned XLA `batchnorm_expander.cc`把inference改写为预计算scale/shift的affine形式；该形式会重新结合浮点运算，
本项选择规范中的center/divide/scale/offset顺序，复用pinned StableHLO op builder和official conversion。
PyTorch的opmath由02号frontend在进入StableHLO之前显式表达；本pass不猜测StableHLO来源或扩大其dtype。

| 输入等价类/分支 | exact输出或failure | 直接下游witness |
| --- | --- | --- |
| rank3/4，feature首/中/末轴，FP16/BF16/F32/F64，1024/1025/1031 | 广播map与feature_index相同；epsilon、subtract→sqrt/divide→multiply→add顺序及dtype准确；重复执行幂等 | official Linalg scalar body与完整返回shape，无BatchNorm inference残留 |
| dynamic shape或quantized inference | 匹配前明确拒绝，不能产生部分分解或假定shape | partial conversion失败；不影响其它op |
| training/grad或已经显式的primitive | 不作为inference重写；不修改其算术 | function pass局部保留；后续不支持仍由原边界报告 |
| 原始PyTorch BatchNorm及ResNet整网 | 同一module导出、全部reference及参数保持；不按模型名分派 | source→正式TensorProgram及package/no-card分别登记，实卡资格另签 |

### 3.3.1 BatchNorm primitive链的精确融合

- Upstream IR / input：official legalization与静态清理后的pure、static-ranked Tensor/Linalg SSA；BN已显式分解为原始标量算术、转换和feature广播。
- Current stage responsibility：识别center、variance/epsilon、sqrt/reciprocal、scale/offset的完整数据链；将已匹配的原scalar operation按SSA依赖嵌入一个标准DPS `linalg.generic`。可包含末尾原有精度转换及直接ReLU，feature轴由indexing map证明。
- Output IR / files：标准Linalg generic及原输入/结果type、operand map和scalar body；不新增BN op、属性、runtime接口或旁路语义表。
- Downstream consumer：同一attention/access-relation normalization、SemanticRoot与spatial/temporal tiling、PBQP及Structured-to-Tile。
- User-level driver / named pipeline：production的none/search及`wafer-lower-stablehlo-to-linalg`共用`wafer-fuse-batch-norm-inference` function pass。
- Explicit non-goals：不把BN折叠进卷积权重，不重新结合浮点算术，不改变转换位置、dtype或fastmath；不融合卷积/归约body，不选择物理tile或SPM合法性。未闭合的内部compute fanout保留原图。
- Completion criteria：BN计算形成一个可直接tiling的root；所有标量步骤、读取坐标和舍入保持；channel-only计算在直接下游仍保持channel域，完整ResNet默认search与本轮package/no-card记录实际Region、IR work、wall及RSS。

此识别是受限BN语义链的scalar-region融合，区别于7节只保留compute occurrence的access关系等价探索；不新增普通Access旁路rewrite。
方法对照采用[MLIR Linalg elementwise fusion](https://mlir.llvm.org/docs/Passes/#-linalg-fuse-elementwise-ops)：由当前indexing maps组合输入坐标并克隆原payload。
本项固定匹配完整BN链，先证明内部compute所有uses闭合，再物化一个region，避免通用贪心fusion隐式复制共享producer。
Unit reshape和广播仅用于证明该已匹配链的输入坐标；其它pure图继续由既有e-graph负责。

| 输入等价类/分支 | exact输出或保留 | 直接下游witness |
| --- | --- | --- |
| rank3/4，feature首/中/末轴，1024/1025/1031，FP16/BF16/F32 | 一个BN generic；原scalar DAG、dtype、转换及feature map一致；repeat幂等 | 原图/融合后全量数值与tiling main/tail |
| reciprocal乘法与直接divide；显式broadcast/unit reshape；可选ReLU | 只克隆对应原步骤，不互换表达式；广播读坐标精确 | PBQP→Tile→Instr及actual SPM |
| 内部compute外部use、错误feature轴、不完整链、非pure body | 不隐式复制或扩大匹配，原图保留 | verifier与原直接消费者 |
| 原始Torch XLA ResNet-18 FP16 | 相同source、默认8/42；Region/工作量实际减少，完整输出端口保留 | verified package与16 Tile fresh no-card；实卡数值/性能另签 |

### 3.4 捕获的投影式Tensor读取

- Upstream IR / input：official legalization后的verified tensor-semantics `linalg.generic`，其payload直接包含
  读取外层Tensor SSA的`tensor.extract`；迭代域和source均为static shape。
- Current stage responsibility：在原source→Linalg pipeline中，将已证明等于迭代坐标投影的读取绑定为真正的DPS input与indexing map。
  source各维要么是同一generic的`linalg.index`且extent等于对应迭代extent，要么是extent=1的零坐标。
  原输入中相同source/map可复用；不同map保持不同输入，不能复用可能参与归约更新的init block argument。
  替换读结果为对应input block argument，保留其余标量算术、dtype、属性、输出map和DPS init/result关系。
  常量求值的直接消费者同时接受map中的显式常量坐标，并沿用逐轴边界检查与原有求值预算；unit广播零坐标不再丢失常量求值能力。
- Output IR / files：同一generic上的实际input operands、indexing maps和region block arguments；不建立额外依赖side table。
- Downstream consumer：原canonicalizer、attention/e-graph及StructuredDAG/SemanticRoot；常规投影路径继续进入原tiling、bufferization和lowering。
- User-level driver / named pipeline：同一`wafer-compile`与`wafer-lower-stablehlo-to-linalg`中的`wafer-normalize-linalg-tensor-reads`。
- Explicit non-goals：不猜data-dependent访问、非unit常量轴、未知extent或越界坐标；这些读取保持原SSA，不能被改成虚假affine map。
  不增加scalar算术、数值重排、物理内存或transport choice；不把输入显式化等同于完整dynamic gather已支持。
  当前e-graph语言拒绝含此类payload Tensor读取的component；本项建立其缺失的显式输入合同，不为已准入语言新增旁路等价探索。
- Completion criteria：每个被提升读取都有逐坐标等价证明，实际DPS input成为唯一依赖；typed DAG/semantic root看见全部新边且无重复输入，
  剩余未知读取不被隐藏。整除/非整除正例、原始source直接下游及固定关键回归分别验收。

采用上游[Linalg显式输入与payload](https://mlir.llvm.org/docs/Dialects/Linalg/)的表示，pinned TOSA `TableConverter`同样将
规则索引输入作为DPS operand、数据相关table read保留在payload；相比只在DAG中扫描capture，本项把可证明的读依赖直接写入IR，
供已有tiling和bufferization共同消费。动态gather仍须单独闭合其访问、生命周期与target合同。

| 分支/输入等价类 | exact要求 | 直接下游witness |
| --- | --- | --- |
| rank3、1024/1025/1031，identity/permutation/broadcast、unit零坐标、FP16/BF16和整数 | 每个input/map与原extract坐标逐项相等；dtype、scalar body及输出关系保持 | verifier、constant数值oracle及TilingInterface main/tail |
| 同一source/map多读、已有同map input、不同map、init同时被捕获 | 相同输入只绑定一次；不同map分开；init读不得误连到正在更新的output block argument | DPS ties、SSA uses、实际DAG edge/semantic key |
| source来自另一个structured root、后继同时有动态表读取 | 新增真实producer→consumer边，保留dynamic extract及其index SSA；非affine表不伪装规则输入 | 原StructuredDAG/SemanticRoot及生产完整LM下一typed边界 |
| data-dependent index、非unit常量轴、source在payload内定义、shape不匹配、source或loop extent未知 | 保持原读取，不引入新input或猜测映射 | verifier及不变IR/原有合法化门禁 |
| 原始完整LM、固定FP16 block | 原source/参数保持，source阶段、实际候选、package/no-card、设备数值分别记账 | 统一runner；实卡完成须有本轮真实设备结果 |

### 3.5 按运行时索引读取连续切片

- Upstream IR / input：verified static-ranked `stablehlo.gather`，显式整数indices和source；本轮主纵向是只读参数表的整行读取。
- Current stage responsibility：在official逐元素GatherConversion之前，将可表示的完整切片gather合法化为标准`tensor.gather`；
  保留source的cast、signedness和StableHLO clamp语义，indices按标准op要求组织坐标维。重复indices合法，不推断`unique`。
  不匹配的StableHLO形态仍由原official converter处理，其后在实际不支持边界返回typed结果。
- Output IR / files：显式source、indices、gather维和结果的`tensor.gather`，以及原语义要求的整数变换；不创建模型专用op。
- Downstream consumer：06号structured root/需求分析及selected output tiling；08/10号物化实际局部输出和DDR行读取。
- User-level driver / named pipeline：生产`wafer-compile`与`wafer-lower-stablehlo-to-linalg`共用同一个conversion实现。
- Explicit non-goals：不改变模型数值、ID端口或越界合同，不在host预计算embedding，不引入表排序、去重、跨调用cache或新硬件能力。
- Completion criteria：标准op的parser/verifier、显式依赖和分块消费者闭合，实际source输入推进到Instr/SPM/target及package/no-card；
  完整数值、动态访问和completion覆盖按本节矩阵执行。只有source转换成功不算本项完成。

方法比较：pinned StableHLO的通用转换以`linalg.generic`中的scalar extract表达完整gather，适合通用语义展开，但过早隐藏了
本仓直接消费者需要的行访问结构。标准[`tensor.gather`](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgather-tensorgatherop)
已经表达完整切片及重复索引；它的越界行为未定义，因此转换必须显式保留
[StableHLO clamp](https://openxla.org/stablehlo/spec#gather)。
[IREE gather tiling](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp)
将output/indices切块，source被索引维保持可访问，连续维按输出需求切片；采用这一边界，不引入IREE dialect或旁路kernel。
Pinned Tensor op没有可直接满足本仓的gather tiling/bufferization实现，所需external interface及实际消费者须在同一修改中接入。

| 输入等价类 | 结构分支及typed失败 | exact输出与下游witness |
| --- | --- | --- |
| rank3输出、长度1024/1025/1031、FP16/BF16，i32/i64 indices | 重复、乱序、首末合法ID，原clamp/cast；不设置unique | 同一编译程序更换indices后全部输出逐bit匹配；source与index依赖显式 |
| static完整切片及output分块 | token/连续维切分、4/16 Tile、多块与tail；不支持的索引形态保持typed拒绝 | 结果覆盖恰好一次，源地址由实际indices决定，连续片段大小正确 |
| 外部只读表及局部输出 | 表保留DDR绑定；局部索引/输出真实allocation；未知来源/访问不伪造静态需求 | actual completion、SPM offsets、target执行和fresh package/no-card |
| 原始完整LM、固定FP16 block回归 | 完整source、默认搜索参数和原始reference；主机/板端分别登记 | gather独立数值与完整产品阶段分别验收；性能以实卡测量验收 |

## 4. Attention Semantic Normalization

### 4.1 Match边界

normalization在official StableHLO-to-Linalg之后、physical planning之前运行一次。它从observable value contraction反向证明：

```text
scores  = contraction(Q, K)
scaled  = scores * scale
masked  = scaled + optional_mask
prob    = softmax(masked, normalization iterators)
result  = contraction(prob, V)
```

production matcher的输入是current产品入口经GSPMD和official conversion产生的实际post-Linalg IR，不是手写的理想attention图。
常见PyTorch/HF前向attention在这一边界共享同一QK--softmax--PV骨架，允许的差异收敛为有限结构族：

| 差异位置 | matcher接受的current IR事实 |
| --- | --- |
| QK/PV contraction | named matmul/batch-matmul，或scalar region与maps证明同一contraction的`linalg.generic` |
| transpose/reshape/cast | exact static tensor/view链；role仍由composed indexing relation推出 |
| scale | QK result上的scalar multiply；若scale已由contraction operands的current SSA显式完成，op原样消费这些operands并使用identity scale |
| mask | additive score adjustment，或可精确归一为同element type adjustment的compare/select+broadcast |
| softmax | max reduction、broadcast/subtract、exp、sum reduction、broadcast/divide的SSA等价形式 |
| KV state | concat、insert-slice或其它current exact prefix-append relation，且updated values和function results闭合 |

matcher不要求这些operation使用一种固定文本顺序，也不把每个组合预写成workload pattern。它先组合view/indexing facts，再检查
contraction、softmax和state dataflow；只有无法从current interfaces证明的结构才保持普通Linalg DAG。proof只读取：

- current SSA def-use与function results；
- Linalg contraction payload和DPS relation；
- indexing maps、iterator kinds和static shape；
- scalar region中的scale、mask、maximum、exponential、sum、divide和multiply-accumulate dataflow；
- standard view/subset relation与MemoryEffectOpInterface。

mask是optional shaped operand，其map必须精确表达对score domain的identity、projection或broadcast；boolean/select形式只有在current
SSA能先归一成同score element type的显式score adjustment时才进入op，否则保留原图。normalization不比较`attention`、`decode`、
`q_proj`等名字，不按参数位置或常见Transformer rank识别。

形成attention proof时，mask输入若来自仅yield原输入、全parallel、output map为permutation的纯broadcast/transpose，
按 `producerInputMap ∘ inverse(producerOutputMap) ∘ maskMap` 组合访问关系，直接绑定原mask和composed map。
不复制或修改producer；其它use仍消费原producer，只有dead closure可删除。非纯转发、非可逆output map或非投影输入map
不穿透，保留原mask operand。这属于完整attention语义识别的operand关系证明；不对opaque attention之外的ordinary
pure component新增e-graph旁路。Scale若已证明是DenseElementsAttr splat，直接形成同element type与同APFloat值的scalar
constant，不物化整张score形状的constant再extract。

访问关系采用[MLIR Linalg fusion的indexing-map composition](https://mlir.llvm.org/docs/Dialects/Linalg/)；
pinned `ElementwiseOpFusion.cpp`同样通过producer output map的inverse组合consumer访问。这里保持现有attention
semantic op和下游TilingInterface，不运行普通producer/consumer fusion，也不改变scalar arithmetic。

PyTorch/HF capture只用于建立和维护上述输入覆盖矩阵，不进入matcher控制流。新增capture若仍可由同一SSA/maps/effect关系证明，
扩canonical analysis或现有typed rule；若需要模型名、固定rank或参数位置才能通过，则该形态不进入current attention合同。

输入view链不能无条件穿透到最早的value：projection后的reshape可能显式展开contraction所需的head或归约轴。
Matcher在同一有界只读查询内考察该链上的actual value；shape只提出轴对应候选，`IndexRelation`组合原contraction maps、
透明view及已匹配score/probability路径，证明Q/K、probability/V的共同迭代坐标、score/output坐标及归约词典序一致后才接受。
不能只按相同extent认定轴相等。Proof未成立或超查询工作界限时保持普通图，不能猜测mapping或签发SPM结论。

输出替换锚定已证明的PV contraction result。必要的同序reshape在该位置形成，原有后续transpose、reshape、projection和其它uses
继续消费它；不穿透transpose后再用元素数量相同的reshape代替实际排列。此变换仍由同一attention normalization pass执行，
输入为current structured tensor SSA，输出为现有attention op与标准views，直接消费者为06号spatial/temporal及原graph users。
不改变scalar arithmetic、dtype、state或算法选择合同，也不运行ordinary graph的额外等价探索。
完成覆盖包含1024/1025/1031、投影后展开的输入、输出transpose、同extent不同轴、多use及相反索引顺序的拒绝，
并用真实完整block的source→attention→actual Instr/SPM/package作为产品witness。

matcher先构造全部proof并在首次mutation前验证overlap：

```text
normalizeAttention(function):
  matches = proveAttentionRoots(current structured SSA, observable results)
  reject conflicting ownership before mutation
  classify each match as flash_attention or flash_decoding
  create one wafer.linalg_ext.attention per match
  replace only the proven final result
  erase only newly-dead, memory-effect-free matched operations
  verify function
```

中间softmax或score有额外observable use时，原producer链为这些uses保留；新attention root只替换已证明的final result，新增工作由后续
planning/cost看见。两个matches共享Q/K/V或mask并不冲突；只有它们试图替换同一result或拥有同一effectful operation时才拒绝。

### 4.2 单一 op schema

图匹配与02号composite共同产生本节唯一schema。结构化causal、位置和宽状态遵循4.5节；所有consumer使用同一类型合同。

概念形式：

```text
%result = wafer.linalg_ext.attention
    ins(%query, %key, %value, %scale, %mask?)
    positions(%query_start, %key_start, %key_valid_end)?
    outs(%output)
    {algorithm = flash_attention | flash_decoding,
     indexing_maps = [query, key, value, scale, mask?, output]}
    score { scalar dot/scale/mask arguments -> original score computation -> attention.yield }
```

`output`是destination，不表示额外accumulate语义；source在attention之后的bias/residual仍是独立SSA consumer。
tensor form返回一个与output同type的result；buffer form写入tied destination。op不公开block size、partition count、Tile、layout、
state buffer、merge owner或schedule字段。

Q/K/V/output使用同一floating storage element type；scale保持source scalar floating type，optional mask保持自己的显式floating或i1 type。
低精度storage对应F32 compute type；QK结果、Maximum/Sum和Accumulator使用compute type。
Online-attention conversion/decomposition按current SSA所表达的转换边界使用它们。这些都是op operand/type事实，不形成algorithm或physical candidate轴。

Score路径的scalar arithmetic由attention自有的单block `score` region保存，并在online form中原样保留。
Block参数依次是QK contraction的compute scalar、scale scalar和可选mask scalar；终结于
`wafer.linalg_ext.attention.yield`，yield的floating type定义Maximum/Sum state type。Region必须封闭、无effect，
只含标量运算；不能捕获外部SSA或包含tensor/buffer。已有DPS/Tiling接口继续拥有tensor输入与state，不增加另一套数值policy。
标准`linalg.yield`的parent合同不适用于这个opaque composite op，因此新增attention专用terminator，直接消费者为op verifier和唯一decomposition。

Matcher从已证明的QK→scale/cast→mask/cast→softmax输入SSA克隆原scalar计算，保留每个dtype与转换位置。例如F32 scale结果先
trunc到F16、再执行F16 mask add、最后ext到F32，这三步必须真实存在于region。不能仅凭scale/mask operand type推断舍入位置。
`score` region是当前op的完整语义，不是future-output IR；graph→online和TilingInterface均通过IRMapping克隆同一region。
Decomposition仅将它嵌入实际score tile的Linalg scalar body，移除原来独立重建scale/mask算术的路径。
对照[IREE AttentionOp的owned region](https://iree.dev/reference/mlir-dialects/LinalgExt/#iree_linalg_extattention-linalgextattentionop)，
采用标准MLIR region与SSA保留source计算，不使用precision模式字符串或按模型恢复语义。

覆盖要求：FP16/BF16/F32、mask有无及不同type、scale前后cast、额外score use、region捕获/effect/type不匹配负例，
1024/1025/1031 graph→spatial→tiled online→Linalg检查相同scalar依赖和舍入；完整block以fresh source验证，最终板端数值另行验收。

softmax→PV之间的浮点转换不是transparent view。完整attention识别允许FP32 softmax结果经一次默认舍入的
`arith.truncf`转换为共同的FP16/BF16 Q/K/V/output storage type，再参与PV；转换两侧可有已证明的layout/view。
这是attention算法形成时允许的浮点重排：online form将normalization移至PV之后，未归一化的exponential在PV输入处
转为storage type，不要求复现eager路径对完整normalized probability的逐点舍入。该授权只适用于完整attention root，
不扩展ordinary graph的cast消除，不改变source、reference或验证门限；score region内原有转换仍按原位置克隆。
其它dtype、显式rounding mode、多重窄化/扩展或cast后的算术仍保留原图。已有score/probability额外users继续消费原SSA。

算法沿用[FlashAttention](https://arxiv.org/abs/2205.14135)的分块online normalization，以及
[IREE attention/online_attention](https://iree.dev/reference/mlir-dialects/LinalgExt/#iree_linalg_extonline_attention-linalgextonlineattentionop)
的graph到stateful form边界；不增加另一套kernel、numeric policy字符串或IR。现有attention op及其types、score region
完整决定融合后的计算，唯一decomposition在actual tiling之后消费它。

本项输入为current structured QK→score→FP32 softmax→storage conversion→PV SSA；输出为现有attention op及保留的额外users；
直接下游为06号spatial/temporal、online decomposition、Instr/SPM和ExecutablePackage。用户入口为普通compiler及
`wafer-normalize-attention` named pipeline。Non-goals为任意cast链消除、source改写、硬件同步调整及数值门限放宽。
完成要求是下表覆盖、canonical构建和完整block fresh no-card/实卡验收，不能只凭attention op形成签发成功。

| 输入/分支 | exact结构要求 | 直接下游与数值witness |
| --- | --- | --- |
| FP16/BF16、K2=1024/1025/1031、FP32 softmax后单次窄化 | 形成一个attention；score yield保持F32，Q/K/V/output保持storage dtype，完整maps/归约轴不变 | online main/tail→Instr；完整block source→package/no-card→实卡 |
| normalized probability有额外user | 仅替换PV root；原divide→trunc依赖为额外user保留 | verifier及返回值use-def |
| F32 storage中的窄化再扩展、多重转换、显式rounding mode | 不形成attention，不消除原转换 | verifier及原SSA依赖 |
| 原同dtype attention、带mask/score转换 | 原算法分类、score region和已有接口不变 | 既有normalization/decomposition/attention integration回归 |

普通浮点完整输出按16号合同同时满足cosine≥0.9999和relative L2≤0.01；模型实卡结果拥有本轮资格，主机结构证明不替代数值验证。

current semantic subset是forward scaled dot-product attention和optional additive/broadcast mask。dropout或其它random effect、backward、
sparse/block-sparse attention、runtime paged-cache lookup及未能由下面maps完整证明的variant不进入该op；它们保持原IR或由未来独立
semantic extension处理，不能通过增加字符串mode绕过verifier。

verifier从maps而非固定维度位置推导attention iteration roles：

```text
B   batch/head-like parallel coordinates
M   query/output-row coordinates
K1  Q/K contraction coordinates
K2  key/value normalization coordinates
N   value/output-channel coordinates

Q      : (B, M, K1)
K      : (B, K2, K1)
V      : (B, K2, N)
mask?  : projected/broadcast subset of (B, M, K2)
output : (B, M, N)
scale  : scalar
```

每组可包含多个iterator；current op接受projected-permutation operand maps，省略iterator即表达broadcast/projection。需要非投影
affine relation的图保持普通Linalg，直到同一current合同扩展verifier、tiling和下游consumer。verifier证明：

- Q/K共享同一K1 domain；K/V共享同一K2 domain；Q/output共享M；V/output共享N；
- output投影全部K1/K2 reduction coordinates；
- mask不依赖未声明coordinate且shape与map一致；
- operands/results为ranked shaped values，element types和DPS tie可由当前op完整解释；
- algorithm attr是closed值；`flash_decoding`具有normalization K2 domain且可形成至少两个nonempty pieces；
- effects、regions、result type和shape reification一致。

`indexing_maps`是ODS inherent field并通过generated accessor读取。当前configured pinned MLIR没有
`IndexingMapOpInterface`的header/TableGen定义；attention直接提供与current consumer所需范围一致的typed map/static-range
accessors。未来若整体LLVM升级提供该standard interface，必须在同一current合同变更中切换producer/consumer，不保留双接口。

### 4.3 Operation interfaces

attention op实现：

- `DestinationStyleOpInterface`：output destination及tensor/buffer tie；
- `TilingInterface`：iteration domain、iterator kinds、output tile position及Q/K/V/mask exact slices；graph semantic form只承担
  保持K1/K2完整的output/parallel tiling；
- `MemoryEffectOpInterface`：tensor form pure，buffer form读取inputs并写destination；
- `ReifyRankedShapedTypeOpInterface`：从output map/type重建result shape；
- `WaferCoupledReductionOpInterface`：为planning提供不创建IR的coupled-state描述。

`WaferCoupledReductionOpInterface`只表达source语义，不返回Tile/layout/buffer/schedule对象：

```text
CoupledReductionDescription
  reductionIterators: K2 iterator IDs
  components:
    Maximum     with row indexing map (B, M) and score result element type
    Sum         with row indexing map (B, M) and score result element type
    Accumulator with output indexing map (B, M, N) and compute element type
  initialization: one neutral state per contribution
  merge: all components are consumed by one coupled combine
  finalization: output is produced once after complete K2 coverage
```

`wafer.linalg_ext.attention`只有一个用户可见result，因此不能直接把K2 block当成final-result tile。`WaferCoupledReductionOpInterface`只提供
Spatial/Exact-demand所需的component maps、coupled grouping和init/final owner query；查询不得为取得这些事实materialize scratch IR，也不得
逐component假设它们独立。

### 4.4 Stateful online-attention form

Spatial/Region choice闭合后、实际TileRegion work创建时，每个selected attention occurrence必须被破坏性转换成
`wafer.linalg_ext.online_attention`。它不是第二条算法路径，而是K/V stateful tiling所需的唯一current IR form：

```text
inputs:
  Query, Key, Value, Scale, optional Mask
DPS inits/results:
  Accumulator with output indexing map
  Maximum     with row indexing map
  Sum         with row indexing map
```

该op实现`DestinationStyleOpInterface`、`TilingInterface`和`WaferCoupledReductionOpInterface`。Parallel/output轴和K2都由同一个
`TilingInterface`切分；K2 tile读取当前三个DPS init并返回更新后的三个state，因此pinned `scf::tileUsingSCF`自然形成loop-carried
Accumulator/Maximum/Sum。K1仍是每个QK block内部的contraction reduction，online-attention层要求K1 full extent。三个state是actual SSA
result，不进入planning record、名字约定或future value ID。

接口语义固定为：

- `getTiledImplementation`要求K1 full extent，按selected parallel/K2 offsets/sizes切Q/K/V/mask，并从当前三个DPS destinations切出state tile；
- 每个tiled op返回更新后的Accumulator、Maximum和Sum；不能把三个result当成相互独立的reduction；
- `getResultTilePosition`分别使用output/row/row indexing map给出三个state的exact offsets/sizes；K2不出现在result map中，因此同一state
  slice成为下一次K2 iteration的DPS init；
- Spatial FD contribution使用同一online step，但cross-Tile combine由第12项在selected merge Region中物化为actual coupled SSA，不调用
  pinned generic reduction driver重建future partial tensor。

仓库pinned `PartialReductionOpInterface`没有IREE当前实现依赖的partial-result tile-position方法；其SCF driver还假设partial result rank与
完整iteration rank一致，不能直接表达attention的output/row/row三种state map。Wafer不为此复制新版MLIR接口或升级整套LLVM；串行K2 block用
上述stateful `TilingInterface`，spatial FD merge用actual current IR。未来升级pinned MLIR时只能在同一合同修改中整体切换，不保留双driver。

`online_attention`本身返回未finalize的state，不返回`Accumulator / Sum`后的用户output。FA在唯一owner的K2 recurrence之后finalize一次；FD的
各contribution不finalize，只有selected merge owner在合并全部state后finalize一次。该op不复制`algorithm`、Tile ID、merge Tile、route或
completion字段：FA/FD差异已经由graph op验证并由actual contribution count、parent TileModule和SSA表达。

同一semantic occurrence在一个IR epoch中只能处于一种形式：graph-level `attention`，或candidate structural IR中的
`online_attention`，或decomposition后的Linalg/Tensor/SCF。转换后旧op立即擦除；不保留兼容reader、双lowering或按属性选择的两条实现。
`online_attention`完成spatial/temporal tiling后由一个确定性decomposition pattern展开为QK、scale/mask、online state update和PV；该pattern
不选择tile、Tile、layout、movement、worker或completion。

Decomposition从current maps构造score map`(B, M, K2)`：QK只reduction K1，随后按current op顺序应用scale和optional additive mask；
Maximum/Sum只reduction K2，Accumulator由probability与V的K2 contraction更新。Old Maximum/Sum/Accumulator分别通过
`exp(oldMaximum - newMaximum)`缩放后作为本block的DPS init，因此已有SCF loop自然承载running state。Score/probability复用一个
current tensor destination，shape只含本次actual batch/head、M tile和K2 block，不含K1或N。该变换保持current `math.exp`、scale/mask
顺序和dtype语义；数值选择不属于本stage。

### 4.5 Composite、结构化causal与混合精度attention

本节定义待实施的attention更新合同，顺序与实卡覆盖由
[统一板测计划](plans/board-workload-matrix.md#attention导出展开与实卡验收)拥有。先更新实现，再启用16号新reference；
旧实现能否通过新reference不是施工前置，不修改原健康性能目标。

- Upstream IR / input：02号验证并保留语义的attention composite，或由已有matcher证明的完整structured attention root。
- Current stage responsibility：统一形成现有`wafer.linalg_ext.attention`，验证位置、mask、head映射和数值语义；
  提供准确的Tiling/DPS/coupled-state事实，并在actual tiling之后唯一展开online attention。
- Output IR / files：带明确语义的attention，随后是candidate-owned tiled online_attention及局部Linalg/Tensor/SCF。
- Downstream consumer：06号空间/时间物化、08号layout/bufferization、10号StructuredToTile及11号Instr。
- User-level driver / named pipeline：原compiler和StableHLO-to-Linalg/attention/decomposition named pipelines调用同一实现。
- Explicit non-goals：不新增第二套attention后端、通用能力查询框架、DTE优化或search计费改动；
  不在普通pure graph中旁路e-graph消除cast，不按模型、固定shape或operand名称恢复语义。
- Completion criteria：本节结构矩阵、16号module reference和计划中的全部可执行正例实卡闭合；
  attention取得匹配设备收益，LLaMA block、大GEMM及受影响已通过case守住原性能目标。

#### 精度与state

| 对象 | 目标合同 |
| --- | --- |
| Q/K/V与两次GEMM输入 | 保持FP16/BF16，不能为宽累加物化整份F32 Q/K/V输入buffer |
| QK accumulator及score结果 | F32；score region的首参数表达实际QK结果type，不再强制等于Q的storage type |
| scale/mask score计算、Maximum/Sum与归一化系数 | F32；任意additive mask保持数值，按明确转换进入score计算 |
| PV的probability operand | 在GEMM输入边界窄化为对应FP16/BF16；V保持原storage type |
| PV Accumulator、跨KV block的状态、merge/finalize计算 | F32；coupled-state description、init、DPS iter args及merge必须使用一致type |
| 最终output | 完成归一化后转回原storage type，只发布完整结果 |

该许可属于完整attention的混合精度数值合同；composite展开定义必须准确，不能把不同舍入图宣称为逐位等价。
普通图中的显式cast、额外score/probability users及未被合同覆盖的算术保持原SSA。
graph到online及tile变换使用同一typed事实；不增加字符串precision模式或按case选择的低精度fallback。
F32状态增加的allocation、布局和lifetime先在actual candidate中物化，再由唯一SPM路径验证。
宽状态不授权psum与GEMM destination同址；11号要求的storage不重叠及各自dtype物理范围继续有效，
DPS或copy消除不得绕过该target约束。

#### 位置与局部展开

两个attention op的`positions`为0或3个index SSA operand，依次是当前query首行绝对位置、当前key首行绝对位置、
有效key域的exclusive end；配套typed `position_map`从完整iteration domain投影query及key的序列坐标，
不把GQA的query-head组误当作序列轴。`causal`为typed boolean，要求完整positions与position_map。
Tiling将实际query/K2 offset分别加到前两项，end保持同一有效域；positions是标量输入，其访问map为空。
`zero_fully_masked`保存来源算子的全屏蔽行语义：SDPA为true，普通eager softmax图为false；仅在最终sum为零时决定
返回零或保留原除法特殊值。两字段属于现有op的语义，不是算法或精度选择。
当前实现对局部score或旧maximum逐元素判断负无穷，再选择指数输入，使空贡献为零；
有限值仍执行原来的subtract与exp。该实现会在整个score块生成比较和select，行级整改合同见4.7。

causal可见性使用真实`key_position <= query_position`及有效KV域。位置经空间切分、temporal tiling和tail后仍由
SSA/明确IR字段解释，不能用局部Q/K shape差重新推断。普通prefill、带cache的多token decode、单token decode共用该规则。
单token只有在有效KV前缀全部可见时才能省去三角屏蔽；padding、滑窗和静态cache无效槽仍受原可见域约束。

06号actual循环物化负责跳过对整个query tile都不可见的KV block，包括其读取和state update；全部可见块省去causal屏蔽，
边界块由本节decomposition生成局部屏蔽计算。任意additive mask继续按实际tile读取，不能从sample payload提升成causal。
全屏蔽行保持来源算子的结果语义；finite additive值与负无穷不能未经证明互换。
边界屏蔽由10/11号验证硬件可实现的局部操作及其真实成本，不能预设GPU寄存器mask在本目标上免费。

有效长度和causal分别证明。证明使用current SSA与ValueBounds；正的常量步长SCF循环按实际可达的最后一个IV
收紧边界，不能把exclusive upper误当成最后一个block起点。不能证明时保留原比较。
当前实现的边界causal使用candidate局部F32 `k−q`模板，与夹界后的`query_start−key_start`标量比较；padding仅在需要时使用
一维key坐标。局部坐标范围总和不超过F32精确整数范围，绝对位置仍为index；不生成Kcore逐元素循环、
SPM mapping或完整序列mask。模板是actual DenseElementsAttr，晚期绑定走14号ProgramData；batch/head共享同一模板。
全可见块没有该模板读/比较；select继续覆盖负无穷，不能用AddVS替代会改变NaN/+inf结果的覆盖语义。
原additive常量及scale分别保持AddVS/MulVS，所需state初始化与不支持immediate的select源按真实硬件合同物化。

Score/probability只覆盖当前query tile×KV block；Maximum/Sum/归一化系数保持行级，通过indexing maps表达广播。
不得借通用decomposition将中间值重新扩大到完整迭代域。DPS准确表达新旧state关系，必要复制由实际旧值用途决定。
layout、physical transpose、broadcast指令与copy cleanup分别属于08/10/11号，不能全部归因或堆入本decomposition。

依据为[FlashAttention的online算法](https://arxiv.org/abs/2205.14135)、
[causal mask坐标实现](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/mask.h)及
[前向F32累加/概率窄化实现](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)。
采用其按位置跳块与宽状态原则，实际指令和资源合法性仍以本目标current IR为准。

| 输入等价类/结构分支 | exact输出或typed failure | 直接下游witness |
| --- | --- | --- |
| FP16/BF16，rank≥3，S1024/1025/1031，多Tile与多KV block | QK结果和三项online state为F32；所有中间shape局部化；finalize后仅输出窄type | graph→tiled online→Linalg→Instr/completion/SPM→package及完整实卡数值 |
| causal全可见/全不可见/边界块，query与KV不等长 | 有效元素exact覆盖；不可见整块无读取/计算；跳块前后state接续正确；绝对位置不丢失 | spatial及temporal主块/tail，单/多token decode实卡 |
| MHA/GQA多head；另有Q/K/V均为`[1,28,4096,128]`的MHA | head分配与尾组不漏不重；禁止全局dense score/mask重物化；F32 state真实容量规划 | 正式source→package/no-card及FP16/BF16实卡 |
| mask有无、additive数值、全屏蔽行、额外score use | 保持原mask与observable语义；非法位置/type/捕获/effect明确拒绝 | 合法行为进入实卡矩阵；verifier负例只在主机执行 |
| 旧state仍被使用/可原地更新、共享输入/被写alias | DPS与effect准确；必要copy保留，消除有proof的冗余 | One-Shot bufferization→movement→实际Instr/SPM，不以copy数量为正确性证明 |

### 4.6 当前版本的mask改进合同（待实现）

本节规定下一步实现目标；4.5中的F32坐标模板仍是当前实现，不能把本节写成已经完成的优化。
本轮范围是现有static attention、bool/additive mask、causal、MHA/GQA、单/多token decode及tail。
本轮causal采用加法方案：按位置生成可见处为0、不可见处为`-inf`的局部bias，再加到score上。
这项选择替代此前拟用的causal `0/1 + -inf源 + MaskMove`；普通bool/select及输入自带additive mask仍保留各自语义。
不新增动态长度入口、paged KV、跨调用准备缓存、任意Python mask callback或新的空间搜索策略。
实施顺序和资格状态分别由[板测计划](plans/board-workload-matrix.md#当前版本attention与mask改进方案)和`progress.md`拥有。

Pipeline position：

- Upstream IR / input：已归一的attention、原score region、typed positions/position_map、mask indexing maps；
  actual spatial/temporal candidate给出当前Tile、query/KV区间、head映射和DPS state。
- Current stage responsibility：05号按本节causal加法合同生成局部bias和score操作，保留其它原score region运算；06号物化可达循环/分支。
  08号确定layout和DPS复用，10/11号消费实际常量、Add及其余predicate操作，发出已有目标指令。
- Output IR / files：标准integer/index位置运算、Tensor/Linalg/SCF局部计算，以及实际Tile/Instr常量、allocation、读写和循环。
  不新增graph mask op、块表ABI、shadow schedule或隐含常驻buffer。
- Downstream consumer：既有completion、唯一SPM规划、TargetCall/CRT、ProgramData/package、TargetModel及runtime。
- User-level driver / named pipeline：现有`wafer-compile`的none/search与相应named transformations共用实现。
- Explicit non-goals：不改softmax精度/算术顺序、SDPA与eager的全屏蔽行差异、KV状态接续、厂商清理或timeout；不把模板复用扩展成跨调用缓存。
- Completion criteria：下述语义分支和计划覆盖矩阵闭合；常规causal无逐块坐标比较/模板准备，原合法特殊mask不丢失，实际SPM和直接下游通过。

causal边界的数值合同为`bias = invalid ? -inf : 0; score = score + bias`，执行于原score region之后、row maximum之前。
加法形式参考[PyTorch SDPA参考公式](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)。
特殊值按实际Add语义验收，不宣称与覆盖逐bit等价；不为切换causal实现增加score有限性扫描、运行时检查或备用分支。

#### 可见性、跳块与mask数值分开

对实际非空query区间`[q0,q1)`、key区间`[k0,k1)`和有效key exclusive end `e`，causal条件为`k <= q && k < e`。
所有位置判断使用integer/index，区间端点运算须检查溢出；转i32必须先证明范围，不能把绝对位置转成F32坐标表。

| 当前IR证明 | 本块执行 |
| --- | --- |
| `k0 >= e || k0 >= q1` | 整块跳过：没有该块K/V/mask读、QK/PV或state update，原`m/l/A`直接接续 |
| `k1 <= e && k1 <= q0 + 1` | causal与有效长度无需逐元素屏蔽；其它真实bool/additive mask仍按原语义执行 |
| 其余情况 | 只在当前query×key局部块上处理边界，不产生完整序列score或mask |

可见域是连续区间且上下界可由current SSA证明时，06号直接收紧KV循环上下界，按原KV顺序物化全可见区和边界区。
等长、零偏移、等块长causal中，query块`i`执行key块`0..i-1`及边界块`i`，不访问`i+1..end`。
这改进当前“完整KV循环内if保护”的循环开销；当前已经跳过的不可见GEMM不能重复登记为新增收益。
不同query/key块长、绝对偏移或tail可能产生多个边界块，不硬编码“一块对角线”。非连续特殊mask不套连续上界公式。

| 输入语义 | 本版执行规则 | 整块跳过的依据 |
| --- | --- | --- |
| 无mask、全有效 | 直接执行原score与online更新，无mask准备 | 仅真实空区间 |
| causal及现有有效KV域 | integer区间分类，边界生成局部`0/-inf` bias并加到score | 上述位置/长度证明 |
| padding、window、prefix、分段等特殊bool mask | 现有入口能表达的常量和输入tensor继续按原map消费；常量局部值可折叠，运行时只读当前块 | 必须由current常量/关系证明整块不可见；不通过样本值、模型名猜规则 |
| 任意运行时bool mask | 原`True=keep`规范化为内部`invalid=1`后masked update；仅在已有合法数据依赖范围复用读取 | 不新增CPU全mask扫描或每块额外reduction来强求跳块 |
| 任意additive mask、bias | 保留原Add、cast位置和dtype；标量/合法unit广播使用已有VS/VuV形式 | finite负数或单独的`-inf`常量都不能自动当bool屏蔽证明 |
| decode、GQA与组合mask | 使用现有绝对位置、query-head到KV-head的映射及实际K/V切片；Q=1仅在整个有效前缀可见时省causal | 与prefill同一规则；FD的K2分片与coupled state merge保持原合同 |

新增可识别规则必须来自现有source语义和明确关系证明；当前无法提取的特殊规则仍由其bool/additive输入执行，
不声称已经拥有FlexAttention的任意索引回调接口。不同来源的多个条件按原逻辑组合；additive变换不与覆盖操作混淆。

#### 常量边界模板与一次调用内复用

对每个实际局部块，定义`BQ = q1 - q0`、`BK = k1 - k0`；两者来自candidate已经物化的query/key区间，
可以不相等，本节不预设块长。模板在query/key两轴上的逻辑形状为`BQ×BK`，tail使用实际剩余行列数。
先由实际tiling确定局部块，再生成所需模板；模板不能反过来把候选限制为固定尺寸或正方形。
选定layout所需的物理padding与逻辑形状分开表达，不扩大逻辑可见域。

静态causal边界在编译期用整数计算`invalid[r,c] = (k0+c > q0+r) || (k0+c >= e)`，其中`0 <= r < BQ`、`0 <= c < BK`。
据此直接生成当前局部shape的最终`0/-inf` bias；位置计算使用整数，bias按score dtype存储。
不生成F32 `k−q`表再在每个边界块比较，也不通过`0/1 * -inf`生成bias。
按实际shape、相对位置及有效域去重；只有模式与布局确实相同才能共享。对齐的等长causal各对角块复用一个模式；
tail、错位和不等块长分别处理，不为每个head或query块复制等价常量。
当一个实际循环body覆盖多个静态模式时，按当前SSA区间及循环步长确定模式集合，在ProgramData中保存模板集合，
用标量integer/index选择实际`BQ×BK`切片；只搬入被选模板，不把整份集合展开到SPM，也不复制整个attention body。

causal常量沿既有ProgramData绑定，作为普通Add的数值operand；所有新buffer和读操作有current-IR owner。
选定布局后的bias物理padding取0，有效域仍由实际layout表达。布局变换不能改变逻辑位置的`0/-inf`模式。
其它bool/select消费链仍保留i1 predicate及所需的数值0/1表示；常量Bit2Fp可在该链折叠，不能把F32冒充i1。
MaskMove路径继续要求canonical 0/1 mask；不默认NPY bool已经是硬件packed bits。
布局转换若仍必要，计入一次准备；只有已有typed representation和编码器能表达时才在包生成时预排布。

同一Tile、同一调用、已证明不变的复用域中，将bias模板load和必要转换移到共同循环外。
同Tile多个head可共享这些只读数据，跨Tile各有自己的实际SPM副本；online `m/l/A`仍按各output piece独立初始化。
复用必须证明source不变、destination私有、没有后续写入或逃逸，并保留分支执行与生命周期约束。
扩展共同PhysicalMovementPlacement处理实际load/fill/convert的destination mutation；在相关操作已物化后、completion/SPM前应用同一证明，
不另写attention专用hoist。复用延长的lifetime仍交给唯一actual SPM路径，不用预估容量决定合法性。

causal只需一份bias模板，逻辑数据量为`BQ × BK × bias元素字节数`，不为它另建完整`-inf`源。
实际allocation包含layout/padding；整个attention的SPM峰值由全部actual buffer及lifetime决定，仍走唯一SPM规划。
模板内容可在package中去重，各Tile仍持有自己的local副本。
多个不同模式不要求全部同时常驻。准备位置、实际load次数和lifetime必须从最终IR核对。

当score的DPS/last-use证明允许原地更新、两输入物理遍历匹配时，常规边界的mask应用为一条整块`AddVV`。
全可见块省去causal bias和Add；有其它score观察者时按普通DPS规则保留必要复制。
其它运算实际需要的同值fill继续使用`XorVV + AddVS`；causal改用Add不承担softmax实现的整体重写。

#### 通用指令规则与方案取舍

| 方法 | 本版选择与理由 |
| --- | --- |
| 局部`0/-inf`模板＋复用＋AddVV | 本轮causal主路径；省去causal专用`-inf`源、0/1转换和MaskMove，模板仍需实际SPM驻留 |
| 数值0/1 mask＋源buffer＋MaskMove | 用于仍要求覆盖的普通bool/select消费链，按实际使用范围准备与复用 |
| 实际比较直接输出数值0/1 | 用于仍需计算predicate的消费链；FP16/BF16/F32已有厂商value/BOOL两类比较，不必固定先i1再Bit2Fp |
| 每行或每段fill | 同值连续区域可以用，细三角边界会产生随行数增长的issue；不作为常规大块causal默认方案 |
| 按元素生成坐标并比较 | 仅用于无法静态折叠的已支持条件；避免完整二维坐标展开。INT32目标比较的具体tuple资格单独核实，不宣称硬件不支持 |
| packed bool模板 | 用于已有bool消费链时实计转换和padding；本轮causal直接生成数值bias，不新增packed host mask ABI |
| 任意select改成加法或乘法 | 本轮causal选择不扩展为通用select改写；其它source显式覆盖语义保持 |

通用比较优化保持source `arith.cmp*`的i1语义。在实际`comparison → Bit2Fp`链上选择厂商数值结果指令，
Instr以destination type区分packed BOOL和数值0/1；必要的结果编码沿同一TargetCall/CRT ABI显式传递。
同步修改verifier、effect/physical span、模型、cost及packet发射；有其它bool consumer时保留其正确表示，不为融合增加未经比较的重复工作。
VV/VS/VuV/VuVLoop只使用对应dtype、单位、tail和物理遍历已有证明的组合；`i1`结果限制是当前软件合同，不是比较输入只能为bool。

填充与broadcast分开：已证明uniform的完整或连续区域用fill，底层统一为按原始storage bits的`XorVV + AddVS`；
普通broadcast优先消费合法VS/VuV形式，无法直接消费的映射保留有证明的GatherScatter/copy。
非同值规律不能用两条fill伪造；位置模板由整数常量计算或原predicate产生，不按mask名字选择指令。

causal加法先独立落地；softmax及finalize的行级整改由4.7单独规定，不作为causal加法实现的前置。
原有运算仍受通用数值比较、fill复用和DPS优化覆盖；保留`math.exp`、既定dtype及验收容差。

采用[FlexAttention](https://pytorch.org/blog/flexattention/)区分块可见性与score修改的组织方式，
以及[FlashInfer variants](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/variants.cuh)
分离mask和logits transform的方式；它们的任意callback、paged KV和GPU线程级predicate不直接作为本版本接口或硬件能力。
本目标的SPM mask、CT/TDMA指令和completion成本必须由实际实现验证。

### 4.7 Attention行级计算整改合同

输入为当前attention的局部score及`m/l/A`状态；输出仍是现有Linalg/Tensor/SCF和Tile/Instr。
目标是把只依赖行状态的工作留在行向量上，消除正常输入也会执行的整块判断、填充、覆盖和重复倒数。
`BQ/BK/D`分别取actual candidate的query行数、KV块长和输出列数，包含实际tail；batch/head前缀沿实际indexing maps表达，不预设固定块长。
问题证据和实施步骤在[统一计划](plans/board-workload-matrix.md#attention其余计算的整改记录)，资格状态由`progress.md`拥有。

Pipeline position：

- Upstream IR / input：tiled `online_attention`、FD contribution states、finalize的行sum与accumulator，及来源`zero_fully_masked`语义。
- Current stage responsibility：05号decomposition与merge/finalize物化明确的行SSA；08/10/11号保留其广播依赖和实际计算范围。
- Output IR / files：行级maximum处理、指数缩放和归一化系数，局部score的subtract/exp及输出multiply；使用既有op与DPS/maps。
- Downstream consumer：layout/bufferization、StructuredToTile、Instr、completion/SPM及原package/TargetModel/runtime。
- User-level driver / named pipeline：现有`wafer-compile`和attention相关named transformations共用同一实现。
- Explicit non-goals：不新增NaN/Inf扫描、输入修补、clamp、通用safe算子或运行时回退；不改变原score region、exp算法、dtype、KV顺序或验收容差。
- Completion criteria：下面三项的最终Instr结构及覆盖矩阵闭合；fresh产品数值、实际SPM规划及统一板测计划规定的验收通过。

整改要求：

1. **Softmax指数计算**：按行处理需要的maximum，再让整个`BQ×BK` score直接执行subtract与exp。
   删除为此生成的逐score负无穷比较、Bit2Fp、整块`-inf`源和MaskMove。
   原maximum state与供指数计算的行值用明确SSA区分；old-state缩放和FD merge使用同一行级规则，不能把临时值写回maximum语义状态。
2. **最终输出的零行结果**：来源要求全屏蔽行输出零时，在行归一化系数中表达；删除专用于该判断的`BQ×D`数值mask、零源和MaskMove。
   `zero_fully_masked=false`保持来源的结果语义。合法mask产生的空行按算子合同处理，不扩大成对非法输入的通用防御。
3. **分母倒数**：先在行sum上计算倒数，再广播用于输出multiply；最终Instr的Recip只消费行向量。
   不先将分母铺成`BQ×D`后重复求倒数，不额外生成整块倒数计算；普通broadcast按既有VS/VuV/GS规则实现。

[FlashAttention softmax实现](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/softmax.h)
将maximum处理和最终归一化系数放在行循环中，再用于列元素。本合同采用这种计算范围划分，继续使用本目标既有指数和倒数指令。
普通纯图的broadcast/逐元素等价优化仍由既有e-graph规则拥有，不在attention之外新增同义greedy旁路。
可消除的工作按actual use-def与indexing maps判定，不能根据buffer名字或样本输入推断。

| 输入等价类/结构分支 | exact检查或typed失败 | 直接下游witness |
| --- | --- | --- |
| FP16/BF16，rank≥3，S1024/1025/1031，多Tile、多KV block；无mask、causal全可见/边界 | 指数相关处理只覆盖行状态；score无上述整块比较/fill/MaskMove链，保留原subtract/exp与窄化位置 | tiled online→Linalg→Instr，完整数值reference |
| bool/additive合法mask；中间空块后出现可见块、最终全屏蔽行 | `m/l/A`正确接续；SDPA零输出与eager原语义分别保持；空行条件不扩成完整score/output mask | online recurrence→finalize→TargetModel及完整输出 |
| FA/FD、多contribution、空局部贡献与非空贡献合并 | maximum state保持原值，行缩放及merge的owner、覆盖和dtype正确 | 实际state endpoints/merge→Instr→输出 |
| 多种实际`BQ/BK/D`、矩形块、主块/tail、不同layout | Recip只遍历行向量及其布局padding，不随D重复；输出整块Recip和零行覆盖链为零 | actual Instr工作量/物理span→completion/SPM→package/no-card |
| 旧state仍被使用、共享operand、非法map/type/region | 保留必要copy和observable use；沿现有verifier/typed failure拒绝非法结构 | bufferization/owner/lifetime验证及原拒绝测试 |

指令条数、处理元素数、broadcast字节、实际allocation/SPM峰值分别统计，不能把元素数减少直接写成issue数或设备耗时同比下降。
共享路径的普通Add/compare/select和原合法mask继续回归；输入使用合法有限Q/K/V，既有mask的`-inf`按其真实语义生成。

## 5. Attention Algorithms

### 5.1 共同 online state

以下`scores(S)`表示QK contraction已经应用current scale和optional mask后的block scores。对一个output row和任意非空K2
subset `S`，定义：

```text
m(S) = max(scores(S))
p(S) = exp(scores(S) - m(S))
l(S) = sum(p(S))
a(S) = sum(p(S) * V(S))
state(S) = (m(S), l(S), a(S))
```

两个disjoint states由同一个coupled combine合并：

```text
m = max(m_left, m_right)
left_scale  = exp(m_left  - m)
right_scale = exp(m_right - m)
l = left_scale * l_left + right_scale * l_right
a = left_scale * a_left + right_scale * a_right
state = (m, l, a)
```

完整K2 coverage结束后只执行一次：

```text
output = a / l
```

这些式子定义algorithm dataflow和state ownership。具体arithmetic operation、dtype及source mask/scale语义继续由current IR原样表达；
本任务不引入其它数值策略或search coordinate。

### 5.2 FlashAttention

`flash_attention`表示每个output piece只有一个K2 spatial owner，K2不做spatial reduction partition。Spatial materialization在该owner中创建
一个`online_attention`，随后current-IR temporal tiling从它的接口选择K2 block和parallel tile。实际形式为：

```text
state = initialState(outputPiece)
for k2Block in exact K2 partition selected from current online_attention:
  qTile      = slice(Q, outputPiece, full K1 work)
  kTile      = slice(K, k2Block, full K1 work)
  vTile      = slice(V, k2Block, output N piece)
  maskTile   = optional slice(mask, outputPiece, k2Block)
  scores     = linalg contraction(qTile, kTile)
  scores     = evaluate original score region(scores, scale, maskTile)
  scores     = apply position-based causal / valid-key mask when needed
  blockState = compute (maximum, sum, accumulator) for this block
  state      = combine(state, blockState)
output = finalize(state)
```

原score region中的bool/select与additive mask分别按源操作执行；位置屏蔽的当前实现与本轮改进分别见4.5和4.6。
score和probability scratch最多覆盖当前`M tile × K2 block`及其batch/head coordinates，不允许物化完整score/probability tensor。
state在K2 loop外建立并通过multi-result SCF iter args携带；block scratch按occurrence显式产生。K1 reduction是decomposition后每个score
block内部的contraction reduction，可由后续普通contraction codegen继续分块，但不得与K2 online state混为同一个spatial split角色。

### 5.3 FlashDecoding

`flash_decoding`表示K2先由Spatial choice分成至少两个nonempty spatial contributions；structural materialization在每个selected Tile中
直接创建一个只覆盖本地K2 interval的`online_attention`，每个contribution内部仍运行上节同一个temporal recurrence：

```text
parallel for contribution_i in exact K2 spatial partition:
  state_i = FlashAttentionPartial(Q, K_i, V_i, mask_i)

merged = combineAll(state_0 ... state_P-1)
output = finalize(merged)
```

Exact-demand analysis按每个output-domain piece建立一个coupled `ReductionMergeRequirement`，其中每个contribution恰覆盖一次K2 fiber，merge后只有
selected merge Tile是final owner。Structural materialization消费`mergeTile`，在该TileModule中创建actual coupled merge与finalize；本地
contribution直接接SSA，remote contribution形成三个actual tensor endpoints。物化后merge位置只由parent `TileModule`决定，参与者只由SSA
operands决定，不保存`merge ID -> TileId`或`RegionExecutionId -> operation`映射。

Q可由多个contribution读取；K/V/mask只读取各自exact K2 slice。Movement stage可以为remote state endpoints选择DDR、direct peer或relay，
但三个components属于同一coupled state：merge必须在全部required components ready后执行，不能逐result独立发布。

merge的算法结构由actual coupled state SSA表达。通信tree、route、worker和completion属于后续current-IR physical stage；K2 partition、
contribution Tile和merge Tile属于本次Spatial transformation choice，并在创建actual IR后立即失效，不进入`algorithm` attr。

### 5.4 Algorithm选择

algorithm在graph normalization中确定，不进入physical search domain：

```text
selectAttentionAlgorithm(match, currentSSA):
  if currentSSA proves:
       K and V are exact fresh prefix-appends of past and new state
       the updated K and V are the values consumed by this attention
       both updated states are returned at the function boundary
       the K2 domain admits at least two nonempty pieces
    return flash_decoding
  return flash_attention
```

Mode与tiling坐标严格正交：

| fixed graph mode | Spatial K2 requirement | actual structural form | Temporal K2 |
| --- | --- | --- | --- |
| `flash_attention` | 每个output piece恰好一个nonempty K2 owner | 一个online-attention state chain；无cross-Tile state merge | owner内部可以有任意多个exact K2 blocks |
| `flash_decoding` | 每个output piece至少两个nonempty、无重叠且完整覆盖的K2 contributions，以及唯一merge Tile | 每contribution一个online state chain；remote state endpoints和一个actual coupled merge/finalize | 每个contribution内部仍可有任意多个exact K2 blocks |

因此temporal block数量、query length、online-attention occurrence数量、普通shape大小或TileRegion数量都不能正向证明FD。K2 cardinality只在
SSA cache-state协议已经成立后检查是否至少能形成两个nonempty contributions；它不能独立完成分类。第12项在任何mutation前同时检查
graph `algorithm`与closed Spatial choice：FA收到多个spatial K2 contributions，或FD收到少于两个contributions、coverage hole/overlap、缺失/
重复merge Tile，均返回typed mode/contract failure；不得自动切换algorithm。转换成功后不再复制mode attr，因为差异已由actual contribution
数量、parent TileModule、state endpoints和merge SSA完整表达。

functional decode proof只使用tensor SSA、slice/insert relation和function results。KV cache仍是普通explicit input/output state；
attention op、package和runtime不拥有cache allocation、eviction、serving scheduler或step policy。无法证明decode时选择FA，不按`Q length`、
参数名或模型入口猜测FD。已经标为FD的root若后续无法形成完整physical plan，compile返回对应typed failure，不静默改回FA。

### 5.5 真实规模示例

以下数值只用于说明同一通用合同，不进入matcher、legality或policy：

```text
Q      tensor<2x16x1025x128xf16>
K      tensor<2x16x1031x128xf16>
V      tensor<2x16x1031x64xf16>
Output tensor<2x16x1025x64xf16>

selected FD K2 contributions:
  [0, 256)       -> Tile 0
  [256, 512)     -> Tile 1
  [512, 768)     -> Tile 2
  [768, 1031)    -> Tile 3
selected merge Tile: Tile 2
```

Structural materialization在四个Tile中分别创建覆盖本地K2 interval的`online_attention`，每个产生
`(Accumulator_i, Maximum_i, Sum_i)`。Tile 2的merge Region直接使用本地state；Tile 0/1/3分别通过三个cross-Tile actual endpoints输入。
Merge op位于`wafer.tile.module(tile_id = 2)`，不再保存`tile_id`字段；其12个state operands说明全部四个参与者。第13项若为每个local K2
range选择128的temporal block，则前三个contribution各有两个完整iteration，Tile 3的263长度形成`128 + 128 + 7`，最后7是唯一tail。
第14项只分解这些loop内的online-attention；selected spatial merge和parent Tile不变。

同一shape的FA只有一个K2 spatial owner；该owner本地遍历完整`[0, 1031)`并形成相同的128-block recurrence，不产生cross-Tile state merge。

## 6. 与 Physical-Dataflow Planning 的唯一接缝

### 6.1 Pure work description

attention op不能作为opaque cost box进入actual admission，也不能在winner阶段突然产生hidden temporaries。Spatial/Exact-demand从graph-level
current op执行只读语义查询；结果只描述当前op能够证明的逻辑域与显式Spatial choice，不描述未来operation、SSA或buffer：

```text
query-local result
  semantic root and fixed algorithm
  iterator roles and exact ranges
  candidate output-piece and K2-contribution intervals
  operand-demand indexing relations
  coupled component maps and merge/finalization requirement
  logical work and mandatory simultaneous-state groups
```

Semantic root遵守06号文档的observable SSA path合同，不含operation pointer、block/operation ordinal、Tile ordinal、printed name或
future worker。查询结果不分配action/value/materialization ID，不进入candidate state、IR attr或文件；Spatial choice变化后重算，choice被
structural materialization消费后销毁。

### 6.2 Physical stage映射

| Stage | 只读attention语义查询的投影 |
| --- | --- |
| Spatial | FA要求K2 logical interval-count product为1；FD要求该product大于1；其它parallel/K1 axes仍按通用domain处理 |
| Exact demand | Q/K/V/mask operand demand、per-output final owner、FD coupled contributions与merge requirement |
| Root work | root-local output、K2 contribution/merge requirement以及external support/boundary |
| Region | attention root与外部producer/consumer的Region membership和显式replica；内部算法步骤不成为Region choice |
| Structural materialization | 消费Spatial/Region choice；每个FA owner或FD contribution直接形成三结果`online_attention`，selected merge Tile形成actual coupled merge/finalize和state endpoints；不创建空shell或execution-to-op映射 |
| Temporal | 只从candidate current IR建立query-local domain；普通op及online-attention的parallel/K2轴使用`TilingInterface`，online K2通过三个DPS state形成serial recurrence；K1留给decomposition后的普通contraction codegen |
| Online-attention decomposition | 已tiled `online_attention`确定性展开为QK、scale/mask、Maximum/Sum/Accumulator update和PV的Linalg/Tensor/SCF；不再选择block、Tile或merge owner |
| Layout/view/bufferization | 针对current operand、scratch、running/partial state和final output建立actual layout conversion、view/alias和allocation |
| Movement | 从current producer/use创建Q/K/V fanout或stage以及FD state transfer/relay；不重建或改选current merge/finalize |
| Instr scheduling | TileRegion-to-Instr后从current operation/effect/token重建event/dependence，应用worker/order并fresh构造completion |
| Actual admission | 从同一current Instr重算buffer relation、lifetime、SPM/DDR/transport和target result |

两条policy都必须满足mode约束：`none`的canonical spatial producer对FA保持K2单一logical interval，对FD构造canonical合法非平凡
K2 partition及stable embedding/merge owner；`search`枚举同一spatial domain中的全部合法factor、embedding和per-output merge placements。
每个current candidate由actual gate判定资源合法性；不得用attention work description、state数量或shape公式预测SPM fit。
这只是physical policy差异，不改变graph attention语义或算法，也不允许`none`把FD降回FA。

这里不新增attention-specific layout、movement、buffer、schedule或resource interface。只读语义查询结果从current semantic op
与显式choice重算；它不携带future action/value/materialization identity，相关choice被actual transformation消费后立即失效，
不成为后续stage schema。

### 6.3 Candidate-owned online-attention materialization与decomposition

Spatial/Region choice闭合后立即进入candidate transaction；不先构造physical value、storage、event或schedule的未来图。
每个candidate执行：

```text
spatial/region choice
  -> validate current TensorProgram and recomputable attention semantic query
  -> create one candidate-owned top-level TileModule subtrees
  -> ordinary spatial work becomes actual Linalg/Tensor/SSA
  -> selected attention becomes actual online_attention contributions, state endpoints and merge/finalize
  -> build query-local temporal choices from current operations and immediately tile/fuse
       ordinary/parallel: TilingInterface
       online K2: stateful TilingInterface
  -> decompose tiled online_attention to canonical Linalg/Tensor/SCF
  -> build/apply current-SSA layout assignment, exact views and bufferization
  -> deterministically convert layout-resolved Linalg compute to wafer.tile.gemm/reduce/elementwise
  -> materialize actual movement on current SSA
  -> lower to Instr, then derive worker/order/completion from current Instr
  -> verify no attention or executable Linalg source remains
  -> actual TileModule/Instr -> DeviceExecutable memory/target gate
```

Structural materialization直接消费fixed algorithm、K2 spatial contribution、Tile embedding和`mergeTile`。FA在一个spatial owner中创建
`online_attention`；FD在每个selected contribution TileRegion中创建覆盖本地exact K2 interval的`online_attention`，并在selected merge
TileModule中创建actual coupled merge/finalize。Merge op不携带Tile ID：其parent TileModule是唯一位置事实，其SSA operands是唯一参与者事实。
本地state直接接SSA，remote state创建三个actual structural endpoints；后续movement只消费这些current values。

Temporal tiling不读取规划阶段的 execution identity 或预物化 temporal 状态。Baseline和search分别从自己candidate中的live
`TilingInterface` operation建立query-local complete domain，选择后立即rewrite并丢弃choice。FA的K2在一个Tile内
形成multi-result SCF state recurrence；FD的每个local contribution可在自己的K2 interval上继续形成同样的recurrence。1024 aligned与
1025/1031 tail遵守06号统一main/remainder合同。

Online-attention decomposition只读取已经tiled的current op和三个DPS state，创建actual QK、scale/mask、Maximum/Sum/Accumulator update、PV
以及必要tensor slices，然后擦除该op。它不重跑全图e-graph或generic tile-and-fuse，不根据future inventory重放IR，不clone整个candidate
owner，也不创建worker、Instr、join或wait。无法分解时销毁candidate，不保留online-attention进入layout，也不调用另一builder。

Module-level transformation在首次mutation前验证全部online op的static tile type、roles和maps；成功后用同一rewriter/listener逐op替换，
不创建或改写SCF loop、TileRegion signature、spatial merge/finalize或boundary endpoint。Named pipeline与compiler adapter调用同一kernel；
layout handoff verifier要求graph/online attention均为零。

compute先到Linalg而不是attention emitter直接创建`wafer.tile`，以复用Linalg indexing/verifier和10号通用structured-to-tile lowering；
但该Linalg只存在于candidate top-level TileModule subtrees transaction内部，不是公开stop stage。Rejected/loser subtree整体销毁，final winner不重建。
Movement、buffer和completion不属于Linalg；它们由直接stage读取current SSA/Instr后生成，不由attention prepared builder预建。

进入actual memory/target gate前必须满足：

- `wafer.linalg_ext.attention`和`wafer.linalg_ext.online_attention`在layout入口均为零；
- 可执行Linalg source op为零；
- all-and-only actions、values、buffers、messages和events已在current IR中表达且可由直接stage verifier解释；
- 每个actual allocation都有current typed owner relation，SPM/DDR/transport结果来自该candidate IR；
- source TensorProgram在candidate失败或落选时保持不变。

本设计只增加一个candidate structural阶段的`wafer.linalg_ext.online_attention`，不新增`wafer.tile.attention`、
`wafer.instr.attention`、attention TargetCall或package/runtime algorithm字段。

## 7. Bounded Access-Relation E-Graph Normalization

### 7.1 Pipeline边界

```text
Pipeline position:
- Upstream IR / input:
  attention semantic recognition完成后的verified static-ranked Tensor/Linalg/Arith/Math graph。前序已经运行仓库pinned MLIR
  实际提供的canonicalization、CSE、elementwise/reshape folds；`wafer.linalg_ext.attention`保持未展开。
- Current stage responsibility:
  将剩余ordinary pure connected component及其ordered observable roots编码为一个有界access-relation e-graph；Rust `egg`中的固定dynamic rules在rebuild后的
  新e-class上继续匹配，C++ request-local `IndexRelation`服务只计算和证明relation composition/map reindex，不预构造最终graph
  candidate。提取形成共享DAG，保持compute occurrence、scalar/combiner、dtype和iterator语义，并通过一次`IRRewriter`原子修改全部roots；不复制
  `ModuleOp`、`FuncOp`或component owner。
- Output IR / files:
  dialect集合不变的verified Tensor/Linalg current IR；支持的reshape、transpose、broadcast、concat和structured compute access
  graph已经收敛。Attention op数量、类型、result type、attribute、algorithm和region保持不变。E-graph、e-class、relation ID、
  match、rewrite history和extractor state全部销毁，不产生文件或旁路IR。
- Downstream consumer:
  StructuredDAG、exact-demand和physical-dataflow choice直接消费rewrite后的current TensorProgram。后续stage不读取或重建e-graph。
- User-level driver / named pipeline:
  `wafer-compile`在policy分叉前调用一次func-level normalization pass；`wafer-lower-stablehlo-to-linalg`与focused named pipeline
  调用同一个pass实现。不存在scoped、candidate或post-tiling e-graph入口。
- Explicit non-goals:
  不修改scalar arithmetic region、dtype或reduction combiner；不做算术结合/分配、matmul chain reassociation、compute partition、
  partial reduction、attention变换、layout assignment、bufferization、movement、SPM admission或winner选择；不把MLIR operation
  pointer、e-class ID或提取结果保存到下一stage；不要求后端为本pass新增Linalg形式，不在attention展开后修补其输出。
- Completion criteria:
  C++ importer只导入current原始节点；至少一个真实phase-order case由两条以上egg rules连续创建中间e-node后闭合，不能由
  `build...Alternatives`或candidate recipe代签。支持的每条rewrite由exact relation/type/iterator proof签发；budgeted exploration确定
  且有界，搜索预算结束不丢弃已证明的等价式；multi-root extraction不复制compute/producer occurrence、不破坏fanout/DPS/effect；输出通过verifier
  并由现有StructuredDAG和exact-demand直接消费；真实规模on/off矩阵记录work、wall、RSS、IR变化和下游stage reachability。
```

Attention recognition必须先于本stage。`wafer.linalg_ext.attention`不创建e-node、不参与rule match、不被展开、克隆或替换；普通producer
component可以把attention data operand rewiring到同type、exact等价的新SSA，attention result在下游component中只作为`Input`，任何rule
都不能同时匹配attention两侧。Collective、call、SCF、effectful op和其它unsupported op使用同一barrier规则。

### 7.2 Component、expression与e-class facts

E-graph只是本pass内部的query-local scratch representation，不是新IR stage或future-output plan。每个request处理一个ordinary pure
connected component及其按current source order排列的全部observable roots。Importer对同一current SSA value只建立一个节点，fanout分支共享
该节点；attention、collective、call、SCF、effect或unsupported op切断component。Request直接携带ordered root node数组，不为它们创建
MLIR tuple op、伪type或future result record。Importer只导入current IR中已经存在的原始节点：

```text
Input(valueId, typeId)

Access(relationId, source)

Concat(axis, resultTypeId, orderedInputs)

Compute(computeId,
        kind = Elementwise | Contraction | Reduction,
        resultTypeId,
        iteratorSignatureId,
        operandMapRelationIds,
        dataInputs,
        initInputs)
```

`computeId`标识原始operation的scalar/combiner region、iterator语义和DPS role；所有rewrite产生的新`Compute`必须保留同一
`computeId`，首批不合并、拆分、删除或复制compute。`reshape`、`transpose`、`broadcast`及其passthrough generic统一成为
`Access`，但只有表示真实tensor transform的exact、total、single-valued relation可以导入。Current MLIR value只在IR未修改且同步
request仍存活时映射到query-local input ID；地址不进入observable排序、跨线程任务或cache。

每个e-class analysis至少携带：

- static logical type、shape、rank和dtype；
- immutable `computeId` multiset与DPS data/init role；
- iterator kinds和ordered reduction iterator identity；
- purity、component boundary和fanout facts；
- relation的source/destination type、total、single-valued、injective/bijective、projected-AffineMap可用性和materialization form；
- 当前直接下游是否已经能消费所得Linalg signature/map。

不同result type、compute facts、effect或boundary不能合并；unknown relation和downstream capability不是finite cost，而是rule不应用。
当前StableHLO concat在official conversion前被规范化为`tensor.empty`加ordered `tensor.insert_slice`链；只有static同rank/同dtype、
单一axis、unit stride、ordered non-overlap、完整coverage、base全部覆盖且中间result无额外observable use时，才导入N-ary `Concat`。
用于实现该Concat的`tensor.empty`/`tensor.insert_slice`链在本次request中属于同一个semantic producer：component connectivity、root/use
判断和提取都读取N-ary inputs，不能把链内的raw `insert_slice` user误判成component外观察者。Extract仍生成标准Tensor IR，不新增Wafer
concat op。

Multi-use Access传播不再由egg外C++ rewrite处理。相同的identity/composition/compute rule在共享component的各root e-class中自然生效；
只有全部ordered roots都完成type/relation/materialization preflight后才提交MLIR mutation。某个分支保持原表达不阻止其它分支在egg中探索，
但C++不得先原地改写一部分consumer再继续运行其它root。Extractor只保留原computeId集合及共享SSA DAG，不增加elementwise fusion、
compute合并、复制或删除规则。

### 7.3 `egg`、C ABI与request-local relation service

实现复用pinned [egg](https://github.com/egraphs-good/egg) Rust library，不自研union-find、hash-cons、rebuild、scheduler、e-class analysis或
extractor core。首批固定使用`egg 0.11.0`的`fb6167957beb5dd7c784121459e08ebd1ccb1a00` revision并启用
`deterministic` feature；Upstream source位于`third_party/egg` git submodule，committed `Cargo.lock`固定registry dependencies，
bootstrap准备Cargo directory source，CMake只运行`cargo build --locked --offline`。产品compile invocation不启动Cargo、rustc、
外部optimizer或临时文件。

C++ importer不生成equivalence edge、最终candidate或candidate-specific recipe。它只导出原始tagged e-node、ordered children、typed
records、ordered root node数组和一个同步request期间有效的relation-service ABI。ABI只保留这一种multi-root schema，旧`rootNode`
单值形式同步删除：

```text
getRelationFacts(relationId)
composeRelations(outer, inner)
composeOperandMap(operandMapRelation, accessRelation)
factorCommonConcatAccess(relation, destinationAxis, segmentTypes)
reindexParallelResult(computeSignature, resultRelation)
reparameterizeElementwise(computeSignature, resultRelation)
getMaterializationForm(relation)
```

该service持有现有C++ `IndexRelation`值和request-local memo，只计算relation/type/map/iterator facts，不构造graph、修改IR或参与下游。
Rust只得到整数ID、typed callback table和一个同步调用期间有效的opaque service handle；不得得到或保存MLIR `Operation`、`Value`、
`Region`或`MLIRContext`指针。Callback不得异步执行或跨request保存handle，返回值区分`Exact`、`Unsupported`、`WorkLimit`和
`InternalError`，不得解析diagnostic文本决定控制流。

Rust adapter对relation facts、composition、Compute/Concat validation、result reindex、elementwise reparameterization和concat factor
使用request-local typed memo；key只含上述整数typed fields，value保留完整callback状态和typed结果。Cache hit不重复跨FFI证明，miss才
计入relation-query budget；cache随request销毁，不跨函数、线程或IR mutation保存，也不改变rule可见的合法集合。

Rust固定注册下面第7.4节的dynamic rules。Searcher只匹配e-node/e-class结构；Applier查询e-class facts与relation service，只有
`Exact`且materializable/downstream-representable时才创建RHS e-node并union。Rebuild后的新e-class继续参与所有rules，因此多步
phase-order闭合发生在`egg`内，而不是C++提前计算最终结果。跨ABI buffer由Rust统一分配和释放；panic在Rust入口转成typed
`InternalError`，不得跨C ABI。`egg`、relation service、memo和extractor在component结束后全部销毁。

Dynamic applier对重复调用保存request-local的实际read set：matched canonical e-class的完整节点及type/valid facts，
以及这些节点的直接child e-classes的同类事实、再下一层的type/valid facts；每个节点的children使用当前union-find代表。
当前固定rules只读取前两层节点，更深层仅查询canonical child/type。复用必须比较完整typed记录，不能仅比较节点数量或hash。
保存的是调用前状态；任何root/child增添节点、union/rebuild导致representative变化或analysis改变，都重新执行原applier。
相同read set只省去已经穷尽且不可能新增union的重复展开，不删除Searcher match、不修改预算计数、rule顺序、rebuild或extractor。
Memo和两项实际检查/跳过计数随同一component销毁；新rule若读取更深的图事实，必须扩展同一read-set合同。
该方法复用[egg的单调等价图及rebuilding边界](https://arxiv.org/abs/2004.03082)，保留
[pinned Applier调用合同](https://egraphs-good.github.io/egg/egg/trait.Applier.html)；不引入第二套Datalog引擎或改变搜索语言。

Rust runner把全部roots加入同一个e-graph。每个root使用同一e-class最优选择，随后按selected e-node和children做deterministic hash-cons，
形成一次共享输出DAG；全component hard check按unique DAG node计算computeId、Access、Concat和node数，不按每root树重复计数。这里不采用
`egglog`、外部solver或TENSAT ILP；不求任意multi-output全局ILP最优，只接受由同一e-class选择得到且对原component严格结构下降的共享DAG。

### 7.4 固定dynamic rule集合

#### Identity access

```text
Access(identity, x) => x
```

#### Access composition

```text
Access(R2, Access(R1, x))
  => Access(composeRelations(R2, R1), x)
```

这两条规则统一处理inverse reshape/transpose、chained broadcast和mixed access chain。Composition可以作为暂时不可单独物化的
exact relation继续参与探索；只有最终extraction中的relation必须具有标准Tensor/Linalg materialization form。

#### Generic compute operand absorption

```text
Compute(id, kind, type, iterators,
        [..., M, ...],
        [..., Access(R, x), ...],
        init)
  =>
Compute(id, kind, type, iterators,
        [..., composeOperandMap(M, R), ...],
        [..., x, ...],
        init)
```

该rule只作用于DPS data input，统一覆盖pure elementwise、contraction和reduction。它保持iteration domain、iterator types/order、
scalar/combiner region、result、init和compute occurrence不变。`Access`必须exact、total、single-valued，组合map必须能由current Linalg
表示且已被当前直接下游支持。Broadcast进入reduction不按operation种类禁止：只要上述proof成立，原reduction loop domain保持不变，
multiplicity仍由current iterator domain表达。General row-major reshape若不能恢复成Linalg AffineMap，可以继续参与access composition，
但不能假装成projected map通过本rule消除；它由下一条rule处理。

#### General reshape through compute

```text
Compute(id, kind, resultType, iterators,
        maps,
        [..., Access(Rreshape, x), ...],
        init)
  =>
Access(Rresult,
       Compute(id, kind, reindexedType, reindexedIterators,
               reindexedMaps,
               [..., x, ...],
               reindexedInit))
```

这是一类dynamic egg rule，不按flatten/unflatten rank或op名称展开多个pattern。Applier调用relation service，只在`Rreshape`为exact static
row-major relation、每个reassociation group只含同一种iterator kind、其它operand/init/result可同步重参数化且current下游接受新Linalg signature时
创建RHS。Elementwise和generic reduction可对连续`parallel`轴及连续`reduction`轴分别collapse/expand，后者必须保持row-major
reduction线性次序和总domain。Contraction只有重参数化后仍能恢复为current rank-2 named matmul或rank-3 named batch-matmul
signature时才创建RHS；额外batch iterator或多个K iterator虽然是verifier-valid Linalg，当前target descriptor链尚未证明可直接消费，不能由
e-graph提前生成后再让Tile lowering猜测扁平化。
一个group混合`parallel`与`reduction`时不改，因为单个Linalg iterator不能同时具有两种kind。
Scalar/combiner region保持不变，contraction仍须恢复为当前支持的generic或named signature。`Rresult`可以暂时不可物化并继续参与composition；最终extract时仍必须
成为identity、standard reshape或其它已有标准materialization。Parallel elementwise可以保留由exact result permutation得到的
non-identity result map；direct Tile lowering以其inverse把全部operand maps同步改写到result coordinates，并把Tile result map规范为
identity。不能只改result map或把任意projected map当成可逆。Reduction和contraction继续使用各自standard projected result map合同。
Callback只返回type/relation ID和typed status，不创建或修改MLIR。
Relation store的intern可移动已有记录；callback跨intern只保留稳定ID或值，不保留lookup返回的指针。
回归覆盖同一component内连续reshape重参数化使relation store增长的情形，检查完整输出shape、原算术与init保留。

#### Concat normalization

```text
Concat(axis, [x]) => x

Concat(axis, [x0, Concat(axis, xs), x1])
  => Concat(axis, [x0, xs..., x1])

Concat(axis, [Access(R, x0), Access(R, x1), ...])
  => Access(R, Concat(remappedAxis, [x0, x1, ...]))
```

第三条由`factorCommonConcatAccess`证明每个segment relation相同、destination axis映射到唯一source axis、ordered source segments
仍all-and-only覆盖且relation没有投影concat axis。它概括`concat(transpose(x), transpose(y)) -> transpose(concat(x, y))`，不为
permutation逐案写rule，也不枚举input重排、子集或partial overlap。

#### Restricted compute result reindexing

```text
Access(R, Compute(id, kind, ...))
  => Compute(id, kind, reindexedSignature(R), ...)
```

这不是任意relation穿过result。`reindexParallelResult`只在`R`为bijective、存在exact iteration-domain bijection、只重参数化parallel
result axes、所有operand maps与init/result可同步转换、reduction axes/order/domain不变、compute occurrence不变且当前下游已支持新
signature时返回`Exact`。Broadcast、projection、slice和涉及reduction axis的relation不应用。

当前实现保持原iteration domain不变，以identity作为iteration-domain bijection，只把DPS init/result map从旧result coordinates组合到
新coordinates。未读取的`tensor.empty` init直接按新result type重建；scalar region读取的init由egg RHS显式建立同一个bijective
`Access`，可继续与已有init access做composition/identity消除。Elementwise输出、generic reduction以及能够恢复成既有
`linalg.matmul`/`linalg.batch_matmul`及transpose-input named variant的exact contraction均走同一rule；其它contraction signature由
当前直接下游能力检查返回`Unsupported`，不要求后端新增形式。

对于pure producer chain中的all-parallel elementwise，Applier还可创建同一等价式的第二种RHS：把iteration domain重参数化到新
result coordinates，将每个data operand map物化为显式exact `Access`，并让elementwise自身使用identity maps。该RHS不融合、复制或
交换Compute；禁止用于含`linalg.index`语义的region。它只用于让后续rule看见`Access(R, producer Compute)`并继续做producer result
reindex；如果不能继续消除，新增Access使结构cost不下降，extractor不会选择。Multi-use producer由multi-root component中的共享node保持
一次，所有分支仍只经过egg rule和统一extraction，不再调用egg外all-users rewrite。

首批不注册elementwise fusion、scalar algebra、matmul associativity/distributivity、reduction domain拆分/合并、concat向matmul/reduce
分配或multi-pattern rewrite；前序pinned MLIR已经拥有的fold/fusion继续由其标准实现负责。Attention、collective、call、SCF、effectful
op、general unsupported slice/insert、pad、gather/scatter和unsupported multi-result compute都是component barrier。

### 7.5 Exploration、extraction与MLIR mutation

E-graph只接收前序pinned MLIR folds之后仍有非相邻或rewrite-order冲突的ordinary pure component。Rule以固定semantic顺序注册，但
输出不依赖rule遍历、hash table、地址或并行完成顺序；pinned `egg` deterministic runner与完整semantic tie-break共同保证可观察确定性。

一次pass invocation按ordinary pure connected component各运行一个multi-root egg request；没有egg外fanout rewrite，也不因某个root先成功而
重跑同一component。一个成功request必须使共享输出DAG的unique `Access`数量严格减少，或在`Access`不变时使canonical `Concat`及总node数
按既定结构顺序严格下降；否则保持原component。第二次运行同一pass必须byte-equivalent。

函数返回的static shaped value若不是当前DPS/Tiling producer，可以在同一次调用内临时包一层identity DPS output closure作为合法commit
point；该closure必须在component收集前创建，并在request结束前精确移除或被提取结果消费。它不进入输出IR，也不计入logical
Access-removal统计。禁止在normalization结束后留下closure，让第二次pass才看到新的component。

该request-local scaffold与06号physical-dataflow入口的`closeStructuredProgramOutputs`不是同一对象：后者在本pass完成后由policy
controller显式物化为current structured output producer，服务Spatial materialization的直接输入合同；本pass不拥有或隐藏该legalization。

预算使用确定性work而不是wall-clock控制输出。Request直接限制relation service call、e-node、rewrite match和iteration；这些有限
container与iteration同时给e-class merge、rebuild和extraction建立上界，并分别报告实际计数。ABI node/child/relation记录在C++与Rust
两端做checked length/offset验证，越过request有限记录表示时保持component不变。当前统计包括：

```text
relation query and composition
e-node insertion
e-class merge
rewrite match
application read-set checks and unchanged applications skipped
rebuild work
iteration
extraction work
ABI import/export records and bytes
```

Request budget由调用方通过typed pass options提供；它只限制本次relation query的确定性工作量，不是由某个workload profile固定的
production常量，shape也不进入选择逻辑。达到iteration、e-node或match上限只停止扩展；从已经rebuild且所有analysis有效的
e-graph运行同一个有界extractor，仍按下述完整root/DAG合同验证并原子提交strictly dominating结果。
饱和不是等价证明的前置条件；pinned egg在一次rewrite结束后rebuild，达到搜索上限不撤销已签发的等价关系，见
[egg Runner](https://docs.rs/egg/latest/egg/struct.Runner.html)。默认迭代上限为32，达到饱和即提前结束；
e-node、match和relation query仍受原独立上限约束，不另外运行局部C++改写。
Relation callback的typed work limit、提取验证无法完成或没有strictly dominating结果时销毁request并保持原verified component；
这不是compiler error、unsupported program、physical rejection或candidate feedback。内部错误始终typed失败，不能被预算状态遮蔽。
达到搜索上限与是否改变IR独立统计；未饱和结果不承诺第二次运行不再改进，正式产品仍只运行一次该stage。
标准MLIR pass statistics记录上述work、input/output op和rule application；fresh qualification另用host profile记录wall/RSS。Timing和RSS
只用于诊断，不进入输出选择。

Multi-root extractor先执行hard constraints：

- ordered root数量、每个root type/dtype和observable result relation相同；
- 共享DAG的unique `computeId`集合、compute occurrence、scalar/combiner region、iterator domain/order和DPS init语义完全相同；
- 每个extracted `Access`都有标准Tensor/Linalg materialization form；
- 每个extracted `Compute` signature/map已被当前直接下游支持。

随后只按结构选择strictly dominating expression：

- 显式`Access`数量最少；
- concat assembly层数最少；
- Tensor/Linalg operation数量最少；
- semantic tie-break只依赖canonical expression、source order和typed fields。

只有前三项unique-DAG结构cost至少一项严格下降时才改IR；canonical tie-break只在多个同cost最优表达之间选择，不能单独触发rewrite。

Extractor不读取target、layout、SPM/DDR、movement、instruction或runtime cost，不接受compute复制换transform减少的trade-off。C++收到
extracted shared DAG和ordered roots后，在首次修改前检查全部type/relation/map/iterator和materialization；随后按DAG拓扑顺序把每个unique
Tensor/Linalg node只创建一次，并收集全部root replacement。旧roots在创建期间保持不变；任一root失败只擦除本component本轮插入的op，
全部成功后按current dominance一次替换所有roots，再删除component内新死且effect-free的旧support op并verify`func::FuncOp`。
这里直接rewrite current IR，不clone `ModuleOp`、`FuncOp`或DAG。成功mutation使旧analysis全部失效；不把input/output operation对应关系
发布给下一stage。

### 7.6 与后续pipeline的隔离

稳定运行顺序是：

```text
official StableHLO-to-Linalg
  -> pinned canonicalization / CSE / supported folds
  -> attention semantic recognition
  -> bounded logical e-graph normalization
  -> final TensorProgram legality
  -> StructuredDAG / exact demand / physical-dataflow pipeline
```

E-graph在policy分叉前只运行一次，只选择ordinary logical graph表达。后续attention expansion必须直接产生其owner定义的canonical
actual Linalg；若它产生冗余IR，应修正该emitter或其本地canonicalization，不能再次调用本pass。Tile-and-fuse、layout PBQP、
`PhysicalLayoutRelation`、movement、bufferization、Instr和memory只读取e-graph已经提交并verify的current IR或各自后续mutation结果，
不读取relation service、e-class或extractor，也不反向扩大本pass的accepted Linalg形式。现有共享`IndexRelation`实现、API、测试及所有
下游caller保持不变；下游从改写后的current IR fresh重算自己的relation。

### 7.7 覆盖矩阵

| 输入等价类 | shape/结构 | typed/optimization failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| multi-rule phase ordering | rank 3--6；1024/1025/1031；`Compute(Access(T, Concat(Access(T,a), Access(T,b))))`及长链不同排列 | 任一composition unknown或work limit时整个component不变 | common-access extraction→nested composition→identity elimination由不同egg rules连续产生；C++最终candidate builder为0；result type/relation相同 | StructuredDAG与exact-demand直接消费最终Concat/Compute SSA |
| continuous mixed Access/Compute chain | 1024/1025/1031；inverse reshape→transpose→elementwise→transpose、common-access concat→elementwise→transpose、broadcast→elementwise→transpose、general reshape/transpose→reduction或contraction→result reshape | general reshape的一个reassociation group混合parallel/reduction、broadcast会丢失唯一loop-bound map、concat gap/overlap或contraction signature不被下游接受时只应用仍可证明的子链，其余保持current IR | general reshape through compute由一类egg rule实际应用；分别检查parallel-only与reduction-only flatten/unflatten，后者保持row-major reduction线性次序和总domain；混合kind负例不改；每条链精确检查Access数、最终maps、computeId/scalar/init和第二次运行byte-equivalent | elementwise形成单一StructuredDAG node；reduction/contraction分别直接通过pinned partial-reduction tiler和既有generic/named lowering |
| generic compute operand absorption | elementwise、matmul/batch-matmul、generic contraction/reduction；1/2/15 inputs/uses；aligned/ragged | relation非total/single-valued、map不可表示、DPS init或downstream unsupported时不rewrite | iteration domain、iterator order、`computeId`、scalar/combiner、init、result和compute occurrence完全不变；只减少Access | current structured consumer及已有tiling/lowering在不改后端时成功 |
| restricted compute result reindex | elementwise/contraction/reduction的parallel result transpose/reassociation；1024/1025/1031 | non-bijective、projection、slice、涉及reduction axis或任一map/init不可同步转换时不rewrite | exact iteration bijection；全部operand/result maps同步；reduction axes/domain不变；compute occurrence不变 | StructuredDAG、reduction tiling和现有direct consumer可消费 |
| canonical concat assembly | N-ary/nested concat；common reshape/transpose/broadcast Access；1024/1025 segment及tail | overlap、gap、partial coverage、dynamic、axis被投影或intermediate external use时不恢复/提取 | ordered pieces all-and-only cover；common relation与axis remap exact；extract回标准Tensor IR | exact-demand piece propagation与consumer maps |
| fanout boundary | component的1/2/15 observable roots、chain/diamond；elementwise、reduction和contraction同构及混合uses；DPS-init与barrier use；1024/1025/1031 | 任一root越过barrier、relation/map不可组合、完整maps无法恢复loop bounds或signature不被下游接受时对应e-class保持原表达；整个request仍须原子materialize | projected/general reshape Access只经egg rules从全部可改写分支消除；shared producer在input/output DAG各一次；每个root maps、iterator、scalar/combiner、DPS init和computeId集合精确检查；无`propagateMultiUseProjectedAccesses`；第二次运行byte-equivalent | StructuredDAG edge、partial-reduction tiler、named contraction lowering和producer occurrence一致 |
| attention/collective/effect barriers | ordinary DAG邻接attention、collective、SCF/call和effect | rule不得同时匹配barrier两侧；malformed输入由原verifier失败 | attention op数量、类型、result type、attributes、algorithm和region逐项不变；只允许data operand被exact同type SSA正常rewire | physical planning看到相同attention semantic roots |
| pinned `egg` multi-root relation-service C ABI与ownership | 1/2/15 roots；empty/single/dense records；连续/并行compiler context；malformed root/tag/length/relation/callback result及forced Rust panic | configure/build缺依赖直接失败；callback typed Unsupported/WorkLimit/InternalError不发布partial rewrite | importer只含原始e-nodes和ordered root indices；dynamic Applier实际创建RHS；output hash-cons共享DAG；无MLIR对象跨ABI；handle同步且不逃逸；allocator/deallocator all-and-only；旧single-root字段/caller为0 | named/driver同一pass和adapter |
| deterministic budget与真实规模 | tiny independent e-class oracle；rank3 1024/1025/1031 transpose→compute及共享DAG；fresh PyTorch/HF/LLaMA dense ordinary component | iteration/e-node/match上限后仍可提取；relation/extraction证据不足保持原component，内部错误typed失败，不进入physical legality | 相同budget产生相同IR/diagnostic；一轮上限下Access减少且全部root、compute、dtype、maps保持；无已证明改进则原样；至少一个fresh真实component多rule有效改写 | StructuredDAG/tiling、physical-dataflow与产品package |
| dynamic rule read-set复用 | unchanged root；root不变而child新增等价式；union/rebuild、跨request/并行；1024/1025/1031 mixed chain与原模型 | 同样数量的不同节点不能命中；callback WorkLimit/InternalError仍保留原typed语义 | 实际skip计数、后续新节点仍触发多rule闭合；同budget前后IR及原计数相同；不改变match额度 | named normalization、当前ResNet与LLaMA source及既有direct downstream |

小shape只用于独立e-class congruence和extractor oracle；所有production rewrite family仍须由表中真实规模case覆盖。

## 8. Card-Partition Collective Boundary

supported StableHLO collective先转换为：

- `wafer.linalg_ext.collective.all_gather`；
- `wafer.linalg_ext.collective.all_reduce`；
- `wafer.linalg_ext.collective.reduce_scatter`；
- `wafer.linalg_ext.collective.all_to_all`；
- `wafer.linalg_ext.collective.collective_permute`。

这些op实现DestinationStyleOpInterface、TilingInterface和MemoryEffectOpInterface，并保留input/init/result、axis或split/concat
dimension、channel、source-target pairs以及reduction combiner。StableHLO的`replica_groups`在本层规范化为
`partition_group` / `partition_groups`；其中ID只属于logical card-partition mesh，不是Tile、DTE endpoint、route、SPM buffer或launch slot。

normalization不得把collective直接lower成Direct DTE，也不得把algorithm、Tile group或physical peer写入LinalgExt attrs。
single-card mesh上的singleton collective可在后续materialization中证明为identity；非singleton card-partition collective需要独立
cross-card transport合同，不能借片内16 Tile通信凑出结果。

## 9. Static Residual Cleanup

GSPMD输出可能含由constants和static tensor views完全决定的partition/mask helper。cleanup仅覆盖可精确证明的：

- `stablehlo.partition_id` / `stablehlo.replica_id`形成的static helper；
- static extract-slice、collapse/expand-shape和tensor.extract常量链；
- all-constant integer passthrough、add和compare generic。

证明只读取DenseElementsAttr、static type/offset/shape/stride及常量整数SSA链。dynamic index、越界、未知来源或无法证明的view保持
原IR并由最终legality gate拒绝。cleanup不是runtime shape evaluator，也不能按partition名、symbol或常见mask shape猜结果。
最终输出不得残留SDY或raw StableHLO。

### 9.1 常量按需读取与坐标解释

- Upstream IR / input：official legalization后的verified Linalg/Tensor IR，实际DenseElementsAttr、static view和indexing maps。
- Current stage responsibility：在原`wafer-fold-static-tensor-ops`中按实际坐标读取常量；单个元素查询不能先展开整份输入。
  Slice只读取结果窗口，等元素数量reshape直接复用DenseElementsAttr存储；Generic先由可逆output permutation将结果坐标映射到
  迭代坐标，再使用输入map中的维度或显式常量坐标读取元素；两类坐标都经过逐轴边界检查。
  未使用的init block argument不读取；body、dtype和算术顺序保持。
- Output IR / files：同一SSA位置的精确`arith.constant`，或不适用时保持原操作；不产生旁路数据、cache或新文件格式。
- Downstream consumer：原canonicalizer、attention识别和structured graph normalization；生产与named pipeline使用同一实现。
- User-level driver / named pipeline：原`wafer-compile`及`wafer-lower-stablehlo-to-linalg`。
- Explicit non-goals：不扩展constant scalar算术语言、不改数值、不改现有结果元素/字节/scalar work预算，不改变SPM、搜索或设备合同。
  这不是ordinary graph的relation等价探索，也不新增e-graph前的graph rewrite。
- Completion criteria：读取工作量随实际输出需求与scalar body线性增长，不包含“输出元素数×输入元素数”的隐藏复制；
  每维坐标先检查边界，output map不可逆或输入访问不受支持时不折叠；实际大shape源图通过直接下游且输出逐项精确。

算法采用[pinned MLIR DenseElementsAttr](https://mlir.llvm.org/doxygen/classmlir_1_1DenseElementsAttr.html)的随机访问iterator与
`reshape`，而不是每次查询构造Attribute数组或建立跨operation缓存。pinned `BuiltinAttributes.h/.cpp`确认iterator按index定位存储，
reshape保持元素类型、数量和原始数据；permutation解释复用标准`inversePermutation`，不根据相同shape猜测坐标。

| 输入等价类/分支 | exact结果或失败 | 直接下游witness |
| --- | --- | --- |
| rank3、S1024/1025/1031，非splat/splat、整数add/compare、broadcast输入、identity/permuted输出、init使用/不使用 | 全部输出与独立坐标oracle一致；无完整输入向量逐元素重建 | 原pass后verified constant及正式normalization |
| FP16/BF16常量passthrough、负零/非有限位型、static reshape/slice | dtype及每个元素原始位型保持，窗口只取所需元素；空窗口明确得到空constant | Tensor/Arith verifier及returned constant |
| 大逻辑splat输入、小结果窗口、非单位stride、多轴位置 | 无与完整source元素数成比例的临时展开；所有读取坐标均在原输入范围 | 真实规模slice结果与外层用户 |
| 超既有预算、dynamic index、未知输入、非可逆output map、未知scalar body | 保持原IR且verifier-valid，不放松预算或猜结果 | 原下游消费与负例检查 |
| 原始完整LM S16与S1024/1025、固定FP16 block回归 | source/参数不变，记录实际pass工作与wall/RSS；完整产品与局部阶段分别验收 | 同一生产builder、package/no-card及后续实卡分级登记 |

## 10. Failure 与 Atomicity

- attention near-miss不是错误，保持普通verified Linalg DAG；
- e-graph component不受支持、没有完整strictly dominating extraction或relation/extraction work不足不是错误，保持该component原IR；搜索扩展预算结束仍尝试同一extractor；
- e-graph声称等价但extracted graph无法通过type/relation proof或verifier是compiler error；rewrite transaction必须回滚且不得发布部分结果；
- C ABI schema/tag/length/ownership、relation-service callback contract错误、Rust panic或`egg` internal failure是compiler-internal typed failure；不得fallback自研C++ engine、
  外部进程或另一rewrite路径；缺失pinned Rust build dependency在configure/build时直接失败，不伪装为runtime optimization skip；
- conflicting matches、malformed existing attention op或rewrite后verifier failure终止normalization；
- normalization在首次mutation前收集完整proof，所有create/replace/erase通过同一个`IRRewriter`；不clone Module/Func/DAG；
- algorithm classification缺decode proof只产生FA，不记录失败历史或候选；
- op/interface无法描述selected spatial/temporal work时返回typed unsupported；current语义查询与显式choice矛盾是compiler contract error；
- Selected-attention lowering、wafer.tile conversion或直接stage verifier失败擦除完整新top-level TileModule subtrees并终止该candidate，
  不在materializer内换算法、layout、route或buffer；
- `none`和`search`任一失败都不调用另一policy兜底。

## 11. Verification

attention正例直接采用真实规模的rank-3或更高Q/K/V/output shape，至少一个sequence或其它主要迭代维度不小于1024；
block与partition验证必须包含整除矩阵，例如Q/K/V sequence为`1024`；同时包含非整除矩阵，例如
Q=`2x1025x128`、K=`2x1031x128`、V=`2x1031x64`、output=`2x1025x64`，检查remainder、tail及不均匀pieces。
这些只是覆盖参数，不进入op schema、matcher或algorithm分类。只有逐点穷举的独立reference和最小verifier负例可以缩小
shape；同一normalization、interface或decomposition机制仍必须有整除/非整除真实规模矩阵，并检查maps、owners、state和
selected pieces，而不是只检查op或pass成功。

直接验证至少覆盖：

1. `wafer.linalg_ext.attention` graph form和`wafer.linalg_ext.online_attention`三状态form的roundtrip、parser/printer、dependent dialect及
   verifier正负例；同一occurrence不同时保留两种form；
2. Q/K/V/mask/output maps的rank-independent role inference，覆盖batch/head、multi-axis、broadcast mask、static view和invalid relation；
3. named/generic QK/PV contraction、scale位置、additive/select mask、softmax SSA等价形式、view/cast组合、extra observable
   intermediate、shared inputs、multiple attention roots、precomputed-score near-miss和effectful conflict；
4. functional KV prefix append/return的FD正负例；改变symbol、argument order或model name不改变分类，无法证明时稳定得到FA；
5. 有界独立reference逐block比较FA state update，并逐partition/tree比较FD contribution/merge/finalize；reference可为逐点
   穷举缩小domain，但同一算法另以真实规模shape覆盖tail、多个output pieces和多block/partition；
6. graph attention Tiling、Wafer coupled-state query、online-attention的stateful Tiling以及late decomposition的
   shape/map/init/final owner一致，planning query前后IR byte-identical；
7. 只读语义查询不产生action/value/materialization ID；structural materialization后每个FA owner或FD contribution的
   Accumulator/Maximum/Sum、merge/finalize和external boundary all-and-only存在于current IR，没有empty shell或hidden inventory；
8. candidate transaction中graph attention被破坏性转换一次；第13项从current online-attention直接切parallel/K2并形成necessary tail，
   第14项对每个tiled online-attention恰分解一次；layout入口两种attention op均为零，生成的Linalg/Tensor/SCF随后全部成为existing
   wafer.tile compute，失败注入保持source和parent原样；
9. `none`和`search`从同一normalized TensorProgram分别走自己的policy-specific materializer；baseline直接消费固定规则，
   search才消费explicit structural choice。两者在policy-complete Instr后消费共同actual leaf；不存在attention algorithm axis或whole-program clone；
10. fresh运行current PyTorch产品入口，至少覆盖native SDPA causal prefill、当前HF attention prefill和functional two-step decode；
    保存/检查本轮portable StableHLO与post-Linalg typed witness，再形成accepted Tile dataflow、Instr、Target LLVM、package和fresh
    no-card。KV cache是显式external state ports，第二步由第一步output绑定，不依赖runtime-owned cache policy。手写MLIR只补
    op/matcher unit，不能代签这一项。

局部op/interface测试不能代替第8--10项。真实板端matched A/B属于后续显式board qualification，不属于本任务的host完成声明。

## 12. 实现入口与扩展规则

production内部顺序为：

```text
collective normalization
  -> static residual cleanup
  -> official StableHLO-to-Linalg legalization
  -> canonicalization
  -> attention semantic normalization and algorithm classification
  -> bounded access-relation e-graph normalization
  -> final TensorProgram legality
```

实现可以拆成多个patterns/passes，但长期合同是上述输入、输出和legality。创建Wafer op的pass必须声明dependent dialects。
attention ODS/verifier/interfaces属于IR owner；attention graph proof/classification和一次性ordinary access-relation e-graph pass属于normalization owner；
query-local work description属于Planning；online-attention IR/standard interfaces属于IR owner，其materialization/decomposition和
structured-to-tile conversion属于Transforms/Conversion。依赖保持单向，
不建立IR→Planning反向include。

新增用户可见attention variant先扩graph semantic op/current algorithm enum、typed proof和work description；不得为每种算法新增
`online_attention`变体、字符串implementation registry或parallel lowering path。Graph `attention`与stateful `online_attention`是固定的
相邻IR层，不是两个可选实现。未来真正不同且无法由当前state和verifier表达的算法，必须先说明其独立semantic对象及各physical-stage
consumer，不能仅因某篇实现使用另一个op名就复制接口。

## 13. 参考实现与采用边界

- [MLIR Linalg transformations](https://mlir.llvm.org/docs/Dialects/Linalg/)与
  [standard passes](https://mlir.llvm.org/docs/Passes/)提供indexing-map驱动的elementwise fusion、transpose/broadcast folding、
  tiling和producer-consumer fusion；Wafer先复用pinned版本实际存在的patterns，再将剩余非相邻等价图交给bounded e-graph。
- [egg](https://arxiv.org/abs/2004.03082)提供rebuilding、e-class analysis、conditional/dynamic rewrite和equality saturation的基础；
  Wafer复用其pinned Rust library，C++ request-local relation service只为dynamic Applier签发exact relation结果，不预构造graph candidate，
  也不把e-class建成IR或跨stage协议。
- [egglog](https://github.com/egraphs-good/egglog)是活跃的next-generation equality-saturation/Datalog engine，但首批Wafer
  transform language不需要database execution model，且fanout/DAG extraction仍需本仓独立合同，因此不与`egg`并行接入。
- [TENSAT](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf)及其
  [rules实现](https://github.com/uwplse/tensat/blob/master/src/rewrites.rs)证明tensor DAG equality saturation能缓解rewrite phase ordering，
  并展示shape-checked custom Applier、transpose/elementwise和concat/transpose等规则；其multi-pattern增长、cycle和DAG-aware ILP
  extraction不适合本项有界确定性pass，因此明确排除。Wafer只使用ordered multi-root、per-eclass extraction和deterministic hash-cons
  恢复共享DAG，不引入ILP或全局代价求解。
- [Glenside](https://arxiv.org/abs/2105.09377)展示将pure access pattern与compute分离、再以通用reshape/transpose/compute rules组合
  多步变换的可行性；Wafer复用已有`IndexRelation`与Linalg per-operand indexing map实现更通用的operand access absorption，不新增
  公开access-pattern dialect或逐operation复制transpose规则。
- [FlashAttention](https://arxiv.org/abs/2205.14135)提供block-wise Q/K/V traversal、online state和避免完整attention matrix
  materialization的算法基础；本文采用其forward state结构，不把论文中的GPU线程层级写入Wafer IR。
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)说明work partition与并行组织仍需结合实际执行层；本文把这些选择留给physical planning，
  不把warp/block调度提升为graph op字段。
- [Flash-Decoding](https://pytorch.org/blog/flash-decoding/)明确采用KV sequence split、每split FlashAttention partial及最终state/output
  merge；本文把split交给spatial planning、coupled availability交给exact-demand analysis、payload/combine交给movement与schedule。
- [IREE LinalgExt attention ops](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td)、
  [attention到online state转换](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/TileAttention.cpp)、
  [tiling/partial reduction](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp)和
  [decomposition](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp)
  证明Q/K/V semantic op、三结果online state、K2 tiling及late Linalg decomposition可以分层。Wafer采用这一IR分层；IREE current使用较新的
  partial-reduction接口，Wafer按仓库pinned API改用stateful `TilingInterface` serial K2 loop，不照搬其target配置、Transform-dialect调度或
  backend pipeline。

外部实现只提供算法和MLIR机制参考。Wafer op schema、fixed FA/FD classification、physical plan、resource proof、Tile/Instr lowering、
target/package和完成门禁始终由本仓current合同拥有。
