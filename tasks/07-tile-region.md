# Wafer TileRegion IR 与事务物化

本文定义`wafer.card.module`、`wafer.tile.module`和`wafer.tile.region`的selected actual IR边界，以及从
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

它的输出是candidate-owned actual structural Card/TileRegion IR。不接收也不创建future physical value、storage object、event、
schedule或completion plan。下游只从输出IR的operation、SSA、type、region、control flow和effect读取事实，并按08、10、11的
固定顺序完成physical realization、execution structure和Instr。

```text
current TensorProgram + structural choice
  -> candidate-owned Card/TileRegion rewrite
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
  normalized card-local TensorProgram；Linalg/Tensor/SCF/Arith/Math、typed collective和fixed FA/FD attention完整表达语义。
  Baseline入口不额外接收search choice；search入口另接收closed spatial/region/temporal choice。
- Current stage responsibility:
  消费spatial/region/temporal choice，在新Card subtree中生成all-and-only TileModules、non-nested TileRegions、
  traversal loops、tail、compute SSA和loop-carried state。只生成structural IR，不选择或物化layout、movement、software pipeline、
  rotating storage、worker、order或completion。
- Output IR / files:
  verifier-valid structural `wafer.card.module`、`wafer.tile.module`、`wafer.tile.region`以及实际Linalg/Tensor/SCF或typed Tile compute。
- Downstream consumer:
  current-IR layout/view/function-boundary与region-local bufferization；随后依次是movement/boundary closure、Tile execution structure、
  TileRegion-to-Instr/worker/order/completion与actual SPM/DDR/transport/target gate。
- User-level driver / named pipeline:
  none/search各自的compiler transaction和materializer entry；focused leaf test可共享同一registered single-op pipeline/API。
- Explicit non-goals:
  不选择winner、不在失败后retile/spill/recompute/换route，不分配SPM/DDR offset，不选worker/order/completion，
  不新建shadow plan、side table或attention-specific Tile/Instr op。
- Completion criteria:
  每个structural choice只生成一份candidate IR；实际SSA与TileRegion表达all-and-only execution/coverage；本stage没有physical layout、
  allocation、movement、pipeline slot或completion事实；mutation后analysis失效；failure不留partial subtree；输出可由08的下一stage直接读取。
```

## 3. IR 层级

### 3.1 `wafer.card.module`

CardModule拥有一个card的observable boundary、shared DDR declaration、all-and-only TileModules和跨Tile communication验证范围。
它是candidate transaction的最小card-scoped owner，不存放candidate list或score。

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
card-scoped stage check只允许它形成同一candidate内的直接TileRegion boundary chain，且下一直接consumer必须是movement
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

Region formation必须覆盖fanout的每个use、reduction partial/merge、effect order和observable output。不相关component不因
“region更大”而合并。Retain/recompute/spill/cut是transformation choice，但其结果必须是actual traversal、SSA、allocation和movement；
不保存`resident=true`或长期lifetime table。

## 5. Structural Materialization Algorithm

1. **只读preflight**：在第一次mutation前检查source op/interface、type/indexing、symbol closure和Tile domain。Baseline在本次
   调用中直接得到固定参数；search检查其closed spatial/region/temporal choice。临时C++对象不创建future SSA/buffer/event。
2. **建立transaction**：在source parent下创建新CardModule和all-and-only TileModules。Source保持不变；failure只擦除
   新subtree。
3. **创建TileRegion与traversal**：根据region membership创建non-nested regions，根据temporal choice创建compact loop、
   exact tail、branch、reduction accumulator和loop-carried state。所有execution all-and-only一次。
4. **创建compute SSA**：从current structured op class、indexing maps、region和DPS/Tiling interfaces生成actual Linalg/Tensor/SCF
   或typed Tile compute。Padding、window和scalar payload只从current op读取。
5. **验证与handoff**：运行op/interface verifier和card-scoped structural stage check，直接检查traversal coverage、SSA use-def、
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
Card/Tile coverage、cross-region root/alias、communication totality和lifetime在最近common owner上运行stage check。不在verifier中重建
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
| share/recompute | SSA use-def、exact demand、effect/speculation | shared SSA producer或consumer-local actual execution |
| attention | fixed FA/FD op、Q/K/V/mask maps、selected block/partition | actual Linalg/Tensor/SCF actions，后降为普通Tile compute |

本stage不修改当前arithmetic op、dtype或数值语义。

## 10. Verification and Done Criteria

覆盖矩阵必须包含：

- rank 3–6的1024与1025/1031，实际经过多Tile、多wave、remainder和tail；
- single-root、multi-root region、independent/coupled traversal、fanout/fanin、reduction partial/merge；
- exact/partial view、layout-compatible/incompatible、shared conversion、alias和explicit copy；
- local、DDR、peer/relay/collective movement与actual effect/token；
- structural、layout-resolved和physical form的parser/printer、合法transition和wrong-stage rejection；
- Serialized与software-pipelined execution structure、prefix/steady/tail、rotating roots和slot SSA；
- attention prefill/decode的FP16/BF16、aligned/ragged和batch/head/seqlen/head-dim axes；
- parser/printer、local verifier负例、stage check、named/driver parity和`verify-each`。

正例必须断言actual TileRegion数、structured execution coverage、SSA owner、alias/copy、movement、tail和直接下游Instr可消费性。
不以plan field数、fixture成功或单个小shape作为完成证据。
