# Production Compiler Optimization Board Campaign 实施计划

状态：pre-board资产已完成，真实板端执行`pending`。本文只组织上板前的同源候选对照资产和执行批次，不复制
`tasks/06-physical-dataflow-synthesis.md`中的candidate语义，也不把板端观测反写成legality。该8+1集合只做
当前production winner qualification，不代表compiler选择空间或硬件行为完备；独立collective矩阵见
`tasks/plans/collective-hardware-characterization.md`，所有最终case状态和证据统一归入硬件校准文档。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q15验证后的source program及typed boundary，Q32 production candidate owner产生并通过
  rank-local/whole-variant exact gates的reserved baseline和默认winner。
- Current stage responsibility:
  通过compiler-private test seam让同一source的reserved baseline与默认production winner
  分别重放完整SPM/DDR、instruction、target ABI和package publication；按会改变真实
  instruction、layout、movement、resource lifetime、issue order或communication的机制组织
  少量区分case，冻结相同payload、CPU expected、最终ELF结构oracle、串行A/B执行顺序和
  bounded timeout。板端首先证明两份package都正确，再保留成对观测；只有可信PMU/timing
  consumer闭合后才校准Q9排序。
- Output artifact / IR:
  两份schema一致、source与host-visible boundary一致、各自由accepted Instr IR独立lower并
  发布的verified package，以及结构签名、完整output/canary/completion结果和原始成对观测。
  candidate frontier、test selection mode、计时样本和结构签名不进入production package协议。
- Downstream consumer:
  Q37 pre-board批次和后续Q9 cost calibration；失败用于定位production winner的正确性、
  target实现或静态排序质量，不改变06/09/12/13的语义合法性。
- User-level driver / named pipeline:
  默认winner继续由wafer-compile source-to-package production pipeline产生；reserved
  baseline只由wafer-compile-test的compiler-private seam产生。两份package都由同一个
  wafer-run board runtime消费，统一由tools/run_hardware_calibration.py串行编排。
- Explicit non-goals:
  不新增公开强制tile/layout/algorithm/ordinal选项，不序列化candidate frontier，不为每个
  canonicalization/verifier/pass单独上板，不用修改source或编译两版compiler伪造A/B，不把
  host wall time、单次样本或untimed model称为hardware speedup；尚未实现的software
  pipeline、worker placement和三slot选择不以手写packet代替production vertical。
- Completion gate:
  当前production优化轴全部映射到new paired-board、existing-board、host-exact或future
  production gate之一；需要A/B的case实际生成结构不同但boundary等价的baseline/winner
  package，no-card双包均通过；board CTest和默认外的显式批次顺序注册完成，初始/终止
  heartbeat、单进程串行、timeout、无retry/reset/power合同保持。真实上板前状态为pending，
  不把资产ready写成board evidence。
```

## 1. 覆盖原则

板测按“下游可观察机制”合并，不按源文件或pass数量展开：

1. tile/reduction split、implementation和fixed Cx/NCx/direct-mapped route共同改变NE、
   GS、DMA、tail与SPM工作集；
2. resident/spill、fusion、physical-version reuse及share/recompute共同改变DDR movement、
   compute work和live range；
3. numeric DAG与target implementation改变opcode、op count、dependency depth和低精度结果；
4. ready-order及后续software pipeline改变真实issue order、completion boundary和engine并行；
5. Direct/Ring/Tree改变all-rank message、hop、event和staging；
6. normalization、alias proof、packing、capacity reject、atomic failure和verifier negative只走
   host exact gate；它们不消耗板卡时间，但会由每条source纵向自然重放。

每个可执行case必须同时具备：

- baseline与winner来自同一source snapshot、target profile、完整runtime launch contract和payload；
- 两份package均独立通过完整late gate，manifest boundary、resource role/type/bytes和expected
  binding相同；
- final ELF结构签名证明目标优化真的造成目标call数量、种类或顺序差异；
- CPU生成的完整expected、write-only canary、transport status和normal cleanup；
- A/B与B/A平衡顺序；正确性失败立即停批，性能观测不能掩盖失败；
- 原始样本只作为observation归档。没有validated PMU/timing profile时，production仍按当前
  static policy或保守baseline，不从本批次自动写入latency常量。

## 2. 上板前实现顺序

### A. Compiler-private paired package seam

- reserved baseline仍是现有frontier中的唯一baseline，不另造source或shadow IR；
- seam只选择已经通过同一whole-variant acceptance的all-baseline tuple，并继续走target
  translation、device link、manifest assembly及readback；
- 正常`wafer-compile`不读取test环境变量，CLI/help/package schema均不变化；
- focused test必须证明同一source的production winner与baseline target结构不同、boundary
  相同，并证明production driver不受test-only选择影响。

### B. P0机制family

- `tile-physical-route`：aligned与odd M/K/N tail两个paired GEMM逐K lane激活（含`k-1`），用
  integer-valued sparse f16输入形成full exact expected；existing single/MN-tiled/K-sharded GEMM继续承担
  更重workload correctness；
- `resident-share-recompute`：fanout-share case要求winner静态RDMA/WDMA callsite减少且三类consumer均保留；
  recompute case要求neg callsite增加、workspace减少并检查三个独立full output；alias/effect/capacity负例留在host；
- `numeric-dag-implementation`：i8 modular common-factor以multiply callsite减少和full modular exact为oracle，
  f32 reciprocal以div→recip和power-of-two exact为oracle；其它f16/bf16 algebraic变体继续走既有host/board
  numeric gate，不由相邻case代签；
- `ready-order`：final linked scheduler body先证明无条件分支和间接跳转，才比较movement-first call sequence；
  输入/expected对两条source路径均敏感，alias/fence/hazard negative留在host；
- `loop-invariant`：当前由host production-frontier positive/negative gate覆盖。公开StableHLO
  source-to-package链尚不能把该rewrite消费的SCF loop带到tensor-program stage，因此不为凑板测
  引入测试旁路；未来真实source producer闭合后再按trip=1与trip>1接入同源双包；
- `collective-algorithm`：现有Direct DTE production vertical在campaign中复跑；Tree all-reduce同源双包只声明
  final target静态prepare callsite与workspace减少、scheduler body不同，并以i8 modular full output和16条
  all-and-only completion/status验证正确性。ordered rank graph由既有host exact gate负责，不能由静态callsite
  反推。Ring all-gather当前公开SPMD boundary只提交reserved baseline，不能拿两个相同package上板冒充A/B。

### C. Q37 software pipeline

software pipeline在Checkpoint B实现前只保留catalog中的future production gate。实现后必须复用
同一个paired-package、结构、numeric和批次合同，覆盖trip `0/1/2/3/4/5`、双slot rotation、
prologue/steady/epilogue、capacity/alias/loop-carried hazard及DTE/fence负例；手写双slot probe
不能代签。

## 3. 验证与批次

1. catalog unit test检查所有current优化轴恰有处置、board项有结构和numeric oracle、future项有
   typed producer前置；
2. 每个paired case先运行双包compile、ELF签名和双包no-card preflight；
3. CTest inventory必须把board case标成`board;hardware;compiler-optimization;paired`并持有同一
   device resource lock；
4. 现有hardware runner在默认校准步骤之外提供显式
   `--batch compiler-optimization-campaign`，按canonical order执行既有Direct vertical和8个paired case；
   `--batch compiler-optimization-paired`只保留8个同源双包子集。二者都放在一次初始heartbeat之后、
   terminal heartbeat之前，不重复环境qualification；
5. 每个optimizer case归档三份source snapshot、两份最终ELF与manifest、结构/观测JSON、raw payload及
   `wafer-compile`、`wafer-compile-test`、`wafer-run`、objdump实际路径和digest，保证失败可重放；
6. 本轮没有配置板卡时只报告`pending board execution`，不能把no-card、ELF或host样本写成板端
   A/B结论。

## 4. 收尾

- 同步06的selection/Q9边界、16的paired board evidence合同和`tasks/progress.md`的Q37状态；
- 测试方式若形成稳定复用规则，写入`memory/general_dev.md`；本轮发现的“单winner板测不能归因
  optimizer”写入`memory/bugs.md`；
- fresh构建、focused unit/lit、catalog、no-card、runner inventory及相关CTest通过后提交；
- 真实板端执行留到用户指定的同一会话串行完成，timeout后停止，不自动retry/reset/power。
