# Wafer TileRegion IR 与事务物化

本文定义`builtin.module`、`wafer.tile.module`和`wafer.tile.region`的selected actual IR边界，以及从
card-local TensorProgram构造该IR的唯一transformation。Spatial/region/temporal choice归06号设计；compute/movement类型化
lowering归10号设计；Instr、memory和communication分别归11–13。

## 1. 核心边界

TileRegion物化是确定性IR transformation，不是第二个optimizer。Baseline和search使用两个独立policy-owned入口：

- baseline materializer从current TensorProgram、exact demand和固定placement/single-root-region/temporal规则直接构造IR，
  不创建search choice/domain/state；
- search materializer消费已闭合的spatial/region/temporal choice。

两个入口可共享single-op typed builders、relation helpers和conversion kernels，但不共享complete schema、controller或fallback。
它们的输入只能是：

- verifier-valid current TensorProgram和其standard interfaces；
- baseline固定规则在本次调用中得到的局部参数，或search已选择的spatial placement、region membership和
  temporal traversal参数；
- 显式target configuration和本次rewrite可重算的analysis。

它的输出是candidate-owned actual structural TileModule/TileRegion IR。不接收也不创建future physical value、storage object、event、
schedule或completion plan。下游只从输出IR的operation、SSA、type、region、control flow和effect读取事实，并按08、10、11的
固定顺序完成physical realization、execution structure和Instr。

```text
post-attention bounded-normalized TensorProgram + structural choice
  -> candidate-owned TileModule/TileRegion rewrite
  -> compact temporal tile-and-fuse; attention remains one semantic op
  -> selected-attention lowering to actual Linalg/Tensor/SCF
  -> late remainder specialization and local slice/view cleanup
  -> verifier
  -> current-IR layout/view/bufferization
  -> current-IR movement
  -> current Tile execution structure
  -> TileRegion-to-Instr
```

Rejected candidate擦除整个新subtree。Accepted owner原样交给下游，不重建TileRegion、SSA或buffer。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  05号bounded access-relation e-graph normalization完成的card-local TensorProgram；Linalg/Tensor/SCF/Arith/Math、typed
  collective和fixed FA/FD attention完整表达语义，不携带e-class或rewrite history。
  Baseline入口不额外接收search choice；search入口另接收closed spatial/region/temporal choice。
- Current stage responsibility:
  消费spatial/region/temporal choice，在builtin module中生成all-and-only top-level TileModules和non-nested TileRegions、
  先从current producer/use生成canonical traversal loop和compute SSA；compact tile-and-fuse期间attention保持同一个semantic op。
  随后的独立selected-attention lowering才消费固定algorithm与剩余attention choice并生成canonical actual Linalg/Tensor/SCF、
  coupled state和loop-carried SSA。最后只specialize必要tail并清理本次产生的local slice/view。只生成structural IR，不选择或物化
  layout、movement、software pipeline、rotating storage、worker、order或completion。
- Output IR / files:
  verifier-valid structural `builtin.module`、`wafer.tile.module`、`wafer.tile.region`以及实际Linalg/Tensor/SCF或typed Tile compute。
- Downstream consumer:
  current-IR layout/view/function-boundary与region-local bufferization；随后依次是movement/boundary closure、Tile execution structure、
  TileRegion-to-Instr/worker/order/completion与actual SPM/DDR/transport/target gate。
- User-level driver / named pipeline:
  none/search各自的compiler transaction和materializer entry；focused leaf test可共享同一registered single-op pipeline/API。
- Explicit non-goals:
  不选择winner、不在失败后retile/spill/recompute/换route，不分配SPM/DDR offset，不选worker/order/completion，
  不新建shadow plan、side table或attention-specific Tile/Instr op；不调用05号全图e-graph修补candidate展开。
- Completion criteria:
  每个structural choice只生成一份candidate IR；实际SSA与TileRegion表达all-and-only execution/coverage；本stage没有physical layout、
  allocation、movement、pipeline slot或completion事实；compact tile-and-fuse不展开attention，selected-attention lowering后attention为零；
  mutation后analysis失效；failure不留partial subtree；输出可由08的下一stage直接读取。
```

## 3. IR 层级

### 3.1 `builtin.module`

`builtin.module`拥有candidate的topology/mesh、shared DDR declarations和all-and-only top-level TileModules，是本次
candidate transaction的共同owner，不存放candidate list或score。每个physical identity由TileModule上的
`(card_id, tile_id)`表达；builtin module自身不复制该身份。

### 3.2 `wafer.tile.module`

TileModule绑定唯一physical Tile。每个Tile可以有不同operation、loop shape和execution length。SPM root、memref view和
SSA alias不得跨TileModule。

### 3.3 `wafer.tile.region`

TileRegion是一个Tile内selected execution与local storage ownership scope，不是单个loop、hardware Tile、fusion标签或SPM合法性结论。
同一个operation按current IR lowering进度具有下列三种显式形式，不使用`stage`、`version`或其它phase attribute：

| form | shaped boundary | region内部 | 直接consumer |
| --- | --- | --- | --- |
| structural | tensor SSA；scalar/control按原typed合同 | actual Linalg/Tensor/SCF、traversal、tail和loop-carried state；没有physical allocation/movement | layout与region-local bufferization |
| layout-resolved | function boundary已经bufferize；logical TileRegion tensor boundary仍显式；每个实际compute use已有current memref endpoint、view/alias和encoding | actual region-local allocation与layout materialization；没有route、staging或跨boundary transfer | movement/boundary closure |
| physical | shaped region I/O只允许Wafer DDR memref；peer/collective通过region内typed op/effect表达，不把SPM作为region result | actual local/DDR/peer/collective movement、staging、token和effect | execution-structure transformation |

Structural和layout-resolved form不宣称SPM容量合法，也不允许SPM memref root直接作为region operand/result。Layout-resolved tensor
boundary只能通过current SSA和标准bufferization/view operation绑定region内的实际endpoint；这种bridge不是跨region SPM alias许可，
module-stage check只允许它形成同一candidate内的直接TileRegion boundary chain，且下一直接consumer必须是movement
transformation；它不能进入其它pass或accepted IR。该关系不能放入side table、名字或临时attribute。
Movement stage必须消除所有未闭合的tensor boundary并形成physical form；不能先创建DDR donor再替换成peer路线。

Physical TileRegion才是一个Tile内的SPM ownership/lifetime domain。一个physical region可包含：

- consumer-driven coupled traversal；
- 多个独立traversal及其不同temporal shape；
- actual layout conversion、view/alias、scratch、accumulator和staging；
- local/DDR/peer/collective movement以及typed effect/token。

SPM root和shaped alias不跨TileRegion。若选择不同region，所有跨界shaped data必须由actual DDR store/completion/load或
其它已定义boundary IR表达。Region boundary本身不是join、device barrier或launch boundary。Operation verifier只检查三种form共同的
局部type/region关系；各stage verifier分别拒绝本stage不允许的form，不能通过一个phase attr绕过检查。

## 4. Region membership 与coupled traversal

同一structural region只选择共同local-storage scope，不自动证明SPM residency或op fusion；只有physical form和late actual planner
共同闭合后才能声明共享SPM residency。Coupled traversal必须同时满足：

- producer work真实位于consumer temporal traversal内；
- producer tile与consumer operand由direct SSA连接；
- intermediate是current tile/window大小，不是完整local shard；
- 没有独立producer traversal或中间DDR store/load。

Same-region grouping不预先选择producer delivery、storage或nested placement。Producer相对Region只有external、local-once和
explicit-replica三种结构结果；其中local-once不预先决定它位于consumer loop内还是loop外。Materializer先在candidate-owned Region中创建
actual operations及其current SSA use-def，再由SCF tile-and-fuse直接检查该IR并立即rewrite。未融合producer可以在同一Region内
独立执行并由后续bufferization形成local reuse；它不因未融合自动变成DDR。融合后的producer实际位于consumer loop内。
`LocalUseDelivery`、`DirectNestedValue`、`StoredRegionValue`、`ReconstructedRegionValue`和`rewireDirectSSA`不属于终态输入或IR。

Region formation必须覆盖fanout的每个use、reduction partial/merge、effect order和observable output。不相关component不因
“region更大”而合并。只有explicit replica允许产生额外producer execution；spill、storage和cut不在Region choice中，其结果若被
后续stage选择，必须是actual allocation、movement和SSA，不保存`resident=true`或长期lifetime table。

## 5. Structural Materialization Algorithm

1. **只读preflight**：在第一次mutation前检查source op/interface、type/indexing、symbol closure和Tile domain。Baseline在本次
   调用中直接得到固定参数；search检查其closed spatial/region/temporal choice。临时C++对象不创建future SSA/buffer/event。
2. **建立transaction**：controller为本次attempt提供唯一candidate owner。当前路径在source parent下新建TileModule set和all-and-only
   TileModules时，不再clone该owner；只有试行已有isolated owner且caller仍需保留原IR时，controller才clone最近的
   `IsolatedFromAbove` scope。Source保持不变；failure只擦除新subtree。后续tile/fuse与attention lowering不得再clone TileModule owner。
3. **创建TileRegion与traversal**：根据region membership和explicit replica创建non-nested regions及actual roots。Temporal domain只
   保存自由tile参数和dependence-legal loop order；可从current producer/consumer exact relation唯一推导的tile不是独立choice，
   non-unique/unsupported/indeterminate关系保留原自由参数与独立producer。FD可按explicit contribution/merge choice创建body仅含
   terminator的actual target Region shells；它们只表达Region membership，必须由第5步填入actual work且不能进入layout。使用pinned MLIR
   SCF tiling生成canonical `scf.for` nest；
   loop IV和bounded size表达wave与remainder，不递归生成`first / steady / tail`笛卡尔积。
4. **从current IR创建普通compute SSA与fusion**：以actual consumer为tiling root，通过standard SCF producer-fusion control读取同一
   Region内current producer/use、indexing relation、use count和effect。Attention保持同一个semantic op，只允许其接口明确支持的
   output/parallel tiling及无需推测内部use/replica的外部exact edge，不能匹配内部QK/PV/state。相同exact request由scoped CSE共享；多个consumer默认保留一个
   loop外producer，只有explicit replica允许实际复制。Reduction/contraction在standard interface与exact无重叠result tile证明下参与，
   collective、cross-region、overlap和unknown保持barrier。Pinned接口不能表达的exact reshape/insert window只使用窄current-SSA
   adapter，不能另建通用fusion worklist/cache。
5. **独立lower selected attention**：调用05号唯一selected-attention transformation，读取current attention op、固定FA/FD及尚未消费的
   K1/K2、contribution/merge choice，一次性创建actual QK、scale/mask、Maximum/Sum/Accumulator、PV、combine/finalize、tensor slice与
   SCF state。FD需要同时创建多个TileModule/TileRegion中的contribution和selected merge，因此该transformation锚定在candidate
   TileModule set这一最近共同owner，不作为读取root sibling的TileRegion pass。它不重跑generic tile-and-fuse，不接收future
   action/value/materialization ID，不创建Instr/join/wait，成功后attention op为零。
6. **late remainder specialization**：对上述两种transformation产生的ragged loops按内到外只peel最后一个partial iteration，promote
   单次tail loop并运行bounded canonicalization、CSE和DCE；不peel first。`r`个ragged tiled axes最多产生`2^r`个static main/tail
   组合。非`TilingInterface`的stream/copy traversal只调用一个机械main-loop+static-tail helper，不进入structured producer fusion。
7. **验证与handoff**：运行op/interface verifier和card-scoped structural stage check，直接检查traversal coverage、SSA use-def、
   tensor boundary、loop-carried state和source-form elimination。Success后下游只消费该current IR；本stage不顺带调用layout或movement。

## 6. 下游Physical与Execution Stages

Layout stage先一次完成function-boundary与region-local bufferization，再由current value type及`IndexRelation`/
`PhysicalLayoutRelation`解释physical mapping。Analysis不创建buffer。一个layout
conversion只在actual consumer需要时创建；多个use共享同一SSA result时不重复materialization。如果physical map、alias、
effect或lifetime不能证明零copy，保留explicit materialization或返回typed unsupported。该stage只生成layout-resolved form。

Movement不是type cast。每个movement op必须显式拥有source、destination、domain、direction和effect。不从value名、shape、
future value ID或donor scan恢复movement。Cleanup只删除current IR上已证明fully redundant的transfer，不移动region cut、
改route或创建spill/recompute。该stage把layout-resolved form闭合为physical form。

Movement闭合后，execution-structure transformation才可依据current physical TileRegion选择并立即物化Serialized或software-pipelined
结构。Pipelined结果必须显式包含prefix/steady/tail、chunk control、rotating allocation roots、slot SSA选择以及实际movement/compute
occurrence和下游必须闭合的reuse/observation obligation；不能把`ExecutionStructurePlan`、buffer multiplicity或预测lifetime带到
Instr或memory stage。

Current transformation入口只接收同一IR epoch的actual `scf.for`、top-level operation groups、stage assignment和可选的
`memref.alloc` rotation binding。Serialized是verifier-checked byte-equivalent identity；pipelined choice立即调用pinned SCF机械
pipeliner生成prologue/kernel/epilogue。Rotating binding在allocation所属TileRegion内创建全部actual slot roots，并以归一化
`(iv-lower)/step % multiplicity`形成loop-local`arith.select` SSA；caller-owned relation在同一transaction扩展到每个slot。
这些query-local choice和raw handle不越过调用，downstream只看到rewritten SCF/memref/SSA/effect。

## 7. Control flow、event 与completion

Physical TileRegion IR可包含actual loop、branch、token和effect；execution-structure stage闭合后，TileRegion-to-Instr只保留这些
结构，不重新选择pipeline或slot。随后scheduler从
current Instr operation、SSA、range、effect、token和control flow构造一次性dependence graph。应用order/worker choice会修改IR并
使该graph失效。Completion owner随后fresh构造minimum/latest join/wait。

Block order、loop iteration、traversal结束、spill点和TileRegion exit不自动证明completion。只有current effect/token/lifetime和
已证hardware/ABI boundary可以要求join/wait。

## 8. Verifier 与failure

Operation verifier只检查TileRegion自身和local operand/result/region关系：

- parent TileModule和non-nested region形状；
- region argument/result、tensor或DDR boundary type和local effect合同；
- SPM value不作为region I/O、view/layout op的local关系；
- body terminator和control-flow结构。

Structural stage拒绝physical allocation/movement；layout-resolved stage要求每个实际use具有current endpoint且没有route/staging；
physical stage要求所有shaped boundary闭合为DDR或typed communication，并拒绝tensor boundary和跨region SPM alias。
TileModule/Tile coverage、cross-region root/alias、communication totality和lifetime在最近common owner上运行stage check。不在verifier中重建
expected execution、future buffer或event inventory。

所有precondition尽量在第一次mutation前检查。Mutation后失败擦除candidate owner并返回compiler error，不使用
fallback builder或partial result。Unsupported semantics、resource exhaustion、capacity rejection和compiler error保持区分。

## 9. Structured semantic coverage

| semantic family | 从current MLIR读取 | actual IR结果 |
| --- | --- | --- |
| contraction | iterator types、indexing maps、DPS init、combiner和type | typed GEMM/batch/accumulator chain |
| affine-window convolution | Linalg dimension inference、window maps、DPS init、explicit `tensor.pad` | canonical typed convolution与必要movement |
| elementwise/relation/select/convert | scalar region、dtype、broadcast/permutation relation | typed compute或explicit composite |
| reduction | reduction iterators、init、combiner、axis/result mapping | native/composite reduce和loop-carried state |
| share/replica | SSA use-def、exact demand、effect/speculation与显式replica choice | shared SSA producer或consumer-local actual execution |
| attention | fixed FA/FD op、Q/K/V/mask maps、selected block/partition | actual Linalg/Tensor/SCF actions，后降为普通Tile compute |

本stage不修改当前arithmetic op、dtype或数值语义。

## 10. Verification and Done Criteria

覆盖矩阵必须包含：

- rank 3–6的1024与1025/1031，实际经过多Tile、多wave、remainder和tail；
- single-root、multi-root region、independent/coupled traversal、fanout/fanin、reduction partial/merge；
- same-region current SSA edge、single/multi-use producer、reduction/contraction producer barrier，以及actual explicit
  replica；
- 1024整除时每个traversal只有一个shared loop body；1025/1031只含必要的main/remainder static form，不含front peel，
  不随wave trip count复制compute closure；两个ragged tiled axes覆盖四种main/tail static组合，一般`r`轴不超过`2^r`；
- exact unique producer tile从temporal free domain移除但物化IR集合不变；non-unique、unsupported和indeterminate保留自由参数、
  独立producer及Region candidate，proposal顺序开关不改变raw domain；
- 每个source structured op的actual iteration tiles并集等于selected work且除explicit replica外两两不重叠；未融合producer
  位于consumer loop外且每selected execution只物化一次，融合producer只位于对应consumer traversal内；
- compact tile-and-fuse的每个attention occurrence保持op kind、algorithm、type和opaque内部语义；新增occurrence逐一由outer tile、
  necessary tail或explicit replica解释；selected-attention lowering对每个actual occurrence恰运行一次；FA产生一个
  K2-owner的actual coupled-state recurrence；FD target Region shells逐显式choice存在且compact前后只有terminator，selected lowering后
  contributions、merge/finalize和跨Region current tensor boundary all-and-only，进入layout前attention op和未填充attention shell均为零；
- structural candidate owner每attempt只新建或clone一次；tile/fuse和selected-attention lowering均不额外clone TileModule owner；
- exact/partial view、layout-compatible/incompatible、shared conversion、alias和explicit copy；
- function result direct destination、region-local SPM reuse和真正cross-region DDR boundary；因output DPS缺失产生的冗余
  DDR→DDR publication copy为0，必要copy有SSA/alias/effect witness并在movement closure后成为typed movement；Instr
  conversion不接收未分类`memref.copy`或新建copy-only TileRegion；
- local、DDR、peer/relay/collective movement与actual effect/token；
- structural、layout-resolved和physical form的parser/printer、合法transition和wrong-stage rejection；
- Serialized与software-pipelined execution structure、prefix/steady/tail、rotating roots和slot SSA；
- attention prefill/decode的FP16/BF16、aligned/ragged和batch/head/seqlen/head-dim axes；
- parser/printer、local verifier负例、stage check、named/driver parity和`verify-each`。

正例必须断言actual TileRegion数、structured execution coverage、SSA owner、fusion result、producer occurrence、alias/copy、
movement、tail和直接下游Instr可消费性。
不以plan field数、fixture成功或单个小shape作为完成证据。
