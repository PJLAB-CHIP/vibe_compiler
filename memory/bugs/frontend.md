# Bug模式：Frontend、数值与输入语义

按主题记录可复用根因；入口见[bug索引](../bugs.md)。规则冲突时以当前编号设计与[AGENTS](../../AGENTS.md)为准。

## 低精度K分块不能沿用窄输出作为partial

- 现象：完整K的GEMM通过原数值容差，分块后出现大量误差；SPM、地址、输入及循环覆盖均正确。
- 根因：GEMM内部使用宽累加并不保证其输出partial也是宽类型。每块先写F16/BF16，再逐块窄加法，会新增原逻辑GEMM没有的舍入。
- 修复模式：在Spatial/Temporal切分前显式建立F32 init/result与原输出处的一次转换；输入保持低精度，GEMM ABI独立传递输入和输出
  format。后续partial、merge、存储及通信消费实际F32 SSA；只把CT add改宽不能恢复已经舍入的GEMM partial。
- 防复发：用抵消输入区分窄partial和宽partial，覆盖整除/尾部、非零init及真实长K下游；原PyTorch容差保持不变。
  `SetPsum`的可选输入、alias和writeback另由显式effect合同决定，不能从支持F32输出推断其融合副作用。

## Frontend不得为了compiler命中改写模型语义

- 现象：fixture手写mask、替换RoPE、预先计算state或重排参数后优化case通过，但真实framework export失败或数值不同。
- 根因：把case构造器当成compiler semantic adapter，形成两个source事实源。
- 修复模式：operation、constant、mask、RoPE、scalar flow、dtype、control flow和function boundary原样进入compiler；允许的
  preprocessing必须是模型本身的普通值计算。
- 防复发：source fidelity gate比较exported graph与官方API语义；mask中的finite值/`-inf`不由compiler注入或删改。

## Temporal reduction变换不建立数值策略

- 现象（历史）：reduction分块曾引入`fastmath`、源顺序、整数overflow flag和typed comparator等额外gate，使同一个IR结构变换
  被误写成numeric policy判断，并让domain与actual emitter接受不同的order。
- 收敛结论：删除独立的reassociation/numeric legality helper及全部temporal调用。temporal domain只决定iterator size/order，
  materializer保留source combiner operation与dtype并构造对应loop-carried state；本层不判断数值可交换性、不选择comparator，也不
  建立numeric search axis。
- 防复发：reduction temporal测试断言all-and-only iterator coverage、main/tail、loop order、DPS init和source arithmetic op/dtype仍在；
  不得以`fastmath`、combiner类别、overflow flag或外部comparator决定temporal candidate是否存在。纯reduction标量输出没有parallel轴时仍必须先构造all-factor=1、单参与Tile的typed
  unpartitioned functional coordinate；这是该root的无parallel轴退化，不是baseline全局Tile数。current placement domain表达不了是baseline implementation gap，不能用来跳过temporal
  split或把source判unsupported。已知「最小tile超SPM」反例必须由最小complete candidate的actual SPM rejection证明；没有该结果时
  保持unknown，不用无关placement失败冒充负例。

## 同storage width不能代签dtype数值转换

- 现象：F16 1.0转换到BF16时直接把`0x3c00`塞进BF16，结果仍是`0x3c00`而非`0x3f80`；因为两者都是16 bit，
  size/count/codec roundtrip均可能通过。
- 根因：代码只比较storage width/category，并调用目标format的raw-value构造器清padding；该操作验证encoding宽度但不执行数值语义。
- 修复模式：source/target format不同就解析current target conversion route并调用formal numeric conversion；rounding参数显式选择
  deterministic nearest-even，缺route、未实现route或缺zero-point等语义参数时fail closed。identity只能是同format的raw copy。
- 防复发：转换测试必须选择同宽但不同encoding的已知值并断言exact target bits；review中看到`RawLogicalValue{target, source.bits}`或
  `makeRawLogicalValue(target, source.bits, ...)`应默认视为bitcast，除非接口明确命名并验证bitcast语义。

## Program source IR authority不能受include顺序或双reader影响

- 现象：source directory同时保留text和generic bytecode时，compiler只读text、测试把bytecode当辅助；不同producer可以提交互相矛盾的IR。
  同批profile header中的未限定`LaunchSlotId`还曾随include顺序解析成target或package type，fresh full build才暴露ODR/API错位。
- 根因：外部format authority、内部TensorProgram text和package typed ID没有按owner显式限定，增量构建掩盖了header重编译事实。
- 修复模式：外部source只读StableHLO portable artifact，retired成员fail closed；post-SPMD text使用不同internal API。跨namespace schema字段
  include其owner并解析成唯一强类型，public header fresh全构建。advisory verifier和compiler复用同一ingestion但不共享verified-path state。
- 防复发：canonical build中的installed product adapter→verifier→compiler→no-card、portable corrupt/retired/mismatch负例和public full rebuild同批执行；
  不能因某个增量target链接成功就跳过完整header consumer构建。

## Ordinary Conv不能把kernel轴当成NCx batch轴

- 根因：原canonical weight将kernel宽排在高前，并复用按首轴独立对齐的NCx；native bare forward实际读取HWOI的一个Cx volume。
- 修复：从current Linalg indexing maps证明weight角色并物化HWOI/Cx；feature/output继续NHWC/NCx，kernel寄存器仍按X/Y打包。
- 防复发：非方形2×3/3×2、O1/O2、I65跨block及1024/1025/1031完整PyTorch见证；Tile/Instr明确拒绝NCx weight。

## 输入format不能决定comparison的输出编码

- 根因：Instr relation的输入是浮点，结果是packed i1；CRT却按输入format选择SDK value/BOOL方法，导致浮点结果写入BOOL allocation。
  下游Bit2Fp会把浮点codeword逐bit解释成谓词，数值误差呈现codeword位模式；设备正常completion不能排除这种错写。
- 修复：relation结果编码由typed op固定为BOOL，CRT始终调用Bool relation方法，format仅描述输入dtype。MaskMove继续消费浮点mask。
- 防复发：检查完整producer→predicate→Bit2Fp→consumer链和actual buffer跨度，并以真实规模正负/特殊值及tail的PyTorch结果验证；
  不根据最终mask现象直接改MaskMove合同，先分别确认产生的编码和消费的编码。

## 扩宽opmath不等于保持低精度卷积的累加顺序

- 根因：消除bias前的低精度回写后，不同F32求和树仍可能跨过FP16舍入中点，后续低精度算子会把该差异放大到默认容差之外。
- 定位：先确认PyTorch实际backend与pinned实现。Slow2d的低精度no-transpose GEMM是四路F32 partial sums，余项进入第0路，最后合并再加bias；某个bias-seeded试算命中expected不能证明它就是reference算法。
- 修复：在已有Tile convolution数值边界物化明确顺序的actual mul/add和可复用scratch，保留上层结构，由同一SPM/completion路径验证。前端逐项展开会放大规划输入，不应为此扩大全局编译预算。
- 防复发：原module、seed、oracle与容差不变；检查整个组合输出、K余项和partial合并，而不只看独立卷积是否在容差内。此方法的适用范围需明确dtype/geometry与reference后端，不外推全域逐bit等价。

## Lowering必须消费真实Linalg payload与目标rank合同

- Linalg body参数不总是`inputs + init`，例如`linalg.map`没有init块参数。使用`getOpOperandsMatchingBBargs`，不能依据op类别猜参数位置。
- 当前unit迭代轴可以把`m+w`这样的访问化为投影；归一后的input map仍须匹配实际operand/result shape，output identity不能跳过检查。
- Tile reduce的logical rank不等于native Instr的可编码rank；native选择必须先满足rank上限，其他已支持形态使用同一既有ordered构造。

## 原始module直接XLA导出的参数别名与算子边界

- CPU→XLA的`Module._apply`可能分别替换多个位置注册的同一Parameter；不能用转移后的参数枚举反推原别名。
  在独立module副本上保存实际注册槽及对象关系，转移后恢复同一对象的共享，再从实际输出图的device-data identity绑定payload。
  非persistent buffer仍是有效source状态，不能只依赖`state_dict()`；BF16 payload保留原始16位存储。
- `native_layer_norm`的CompositeImplicitAutograd分解可能先于Python dispatch，进入其中的training BN再按BN inference处理会误拒绝。
  在原typed LayerNorm调用边界使用pinned官方分解；普通ATen分解仍由同一capture scope承载，不按模型名判断。
- 防复发：共享参数和非persistent buffer的1024/1025/1031、FP16/BF16真实导出，逐bit payload/输出以及修改runtime ID后复用同一图；
  验证原CPU对象不变，数据相关标量、CPU fallback、显式graph step、状态修改和不安全路径失败后不发布目录。

## 常量折叠的预算不能约束隐藏的完整输入复制

- 根因：输出元素/scalar body计数有界，但每个元素查询都把DenseElementsAttr展开为完整Attribute数组；
  broadcast后再逐项compare会变成输出元素数乘输入元素数的工作。Slice复制完整source也会把很小的需求变成大工作。
- 修复模式：用pinned DenseElementsAttr iterator直接索引，仅遍历需要的结果窗口；splat窗口用resizeSplat，
  相同dtype/元素数的reshape复用原始存储。先验证每维坐标；generic从可逆output map反解迭代坐标，不能假定输出就是identity。
- 防复发：真实规模非splat与broadcast、1024/1025/1031及permuted输出逐项核对；FP16/BF16负零、非有限位型、
  大逻辑splat小窗口、非单位stride及空窗口保持精确。未使用init不读，未知body/输入与超原预算保持原IR。

## Linalg payload的投影式捕获读取必须成为真实输入

- 根因：官方gather legalization可在generic payload中读取另一structured producer的Tensor；只遍历外层operands的
  DAG/semantic-root分析会漏掉该真实依赖。只放宽可观测路径检查不能修复tiling和后续物化。
- 修复边界：source坐标可证明为同一generic的迭代投影，且对应extent相等，或unit维的零坐标时，
  绑定为实际DPS input/indexing map并改用input block argument。相同source/map复用；不同map分开。
  捕获init仍读取原Tensor SSA，不能替换为可能已更新的归约accumulator argument。
- 防复发：1024/1025/1031逐bit数据、main/tail实际切片和exact覆盖，验证真实producer→consumer DAG边及semantic key。
  数据相关table读取仍保留原索引、clamp和capture；没有动态访问合同不能伪造affine map或宣称完整gather已闭合。

## SDK标量字段的整数载体不改变运算dtype

- 已确认的软件错误：fill的F32 ABI调用在CRT中被按存储宽度换成INT32；SDK实际packet也因此成为INT32。
  为保留特殊值位型而增加的规则同时污染了成本分类和测试预期，主机字段自洽不代表硬件数值正确。
- 修正：scalar使用目标dtype的原始编码，format独立保留该dtype；显式convert才改变数值类型。
  回归分别核对IR/ABI类型、SDK格式字段和scalar编码，不能仅核对字节宽度。SDK编码验证与实卡数值资格分别记录。
- 边界：该软件缺陷已定位；它是否导致后续TDMA不能由异常先后或清理后的SPM内容推定。
