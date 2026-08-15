# K-Sharded GEMM Board Vertical 实施计划

状态：已完成。本文只拆解 full-4096 K-sharded GEMM 从 StableHLO 到真实板端的施工与验证步骤；稳定
SPMD、tile/dataflow、SPM、communication、target、runtime 和 verification 合同仍由编号设计文档拥有。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  deterministic f16 StableHLO `dot_general` program directory，global lhs/rhs/result 均为
  4096x4096；两个contracting operand通过frontend `mhlo.sharding`显式沿K轴分到16个logical rank。
  pinned XLA helper必须产生verified post-SPMD rank-local program：lhs=4096x256、rhs=256x4096、
  result=4096x4096 replicated，并以logical sum all-reduce组合16份partial result。
- Current stage responsibility:
  用production `wafer-compile`重放frontend、SPMD、structured normalization、physical-dataflow、
  M/N traversal tiling、SPM/DDR planning、logical collective lowering、Direct DTE acceptance、target
  conversion和package publication；随后由同一package进入production no-card与qualified board provider。
  upstream SPMD固定16个rank partial；每rank local K=256可继续由现有candidate按资源做内部切分，
  但不能改变rank间all-reduce边界。
- Output artifact / IR:
  schema-v5/status-v2的16-rank cluster Direct DTE verified package、all-and-only typed host resources、
  frozen deterministic raw inputs/CPU expected、静态tiling/SPM/transport证据以及board output/status/lifecycle证据。
  不增加workload、shape或test名相关的compiler/runtime协议字段。
- Downstream consumer:
  large-shape tile/dataflow与Direct DTE持续回归，以及Q22.C可选的workload-level board evidence；本case本身
  不完成Q22.C，也不升级per-op numeric profile。
- User-level driver / named pipeline:
  `wafer-compile --execution-ranks=16 --target-profile=wafer-tx81-single-card
  --launch-abi=tx81-cluster-direct-dte-prepare-main-v1`；no-card和board均由`wafer-run --all-ranks`
  消费同一typed package。局部`wafer-opt`/FileCheck只补结构覆盖。
- Explicit non-goals:
  不新增launch ABI，不走当前不支持Direct DTE的type-6/type-7 model launch，不增加numeric mode或IR carrier，
  不声明physical tile coordinate、packet provenance、性能/timing、通用GEMM数值域或Q22.C profile；
  不在失败后自动retry/reset/power/reboot。
- Completion gate:
  helper/readback证明16个K shard无重叠完整覆盖且output replicated；production selection证明完整4096x4096
  traversal不能把两个2 MiB local input与32 MiB output的完整工作集整体放入SPM，selected program的accepted SPM high-water
  位于[65536, 3080192)，GEMM指令的K chunks无重叠完整覆盖每rank local K=256；all-rank Direct DTE
  issue/wait/binding及status-v2 package闭合；
  production no-card通过；armed hardware gate实际执行未skip，16个rank status均SUCCESS且16份完整32 MiB
  output与冻结expected逐字节相等，cleanup后只读设备资源回到执行前基线。
```

## Case 参数与可观测性

- Global GEMM：`A[4096,4096] x B[4096,4096] -> C[4096,4096]`，dtype为f16。
- SPMD：K轴16分，rank `r`消费`A[:, r*256:(r+1)*256]`和
  `B[r*256:(r+1)*256, :]`；每rank local GEMM为`4096x256 x 256x4096`。
- 输入使用f16可精确表示的有界二进制缩放值：local `A[m,k] = (r + 1 + (m mod 17)) / 4096`，
  `B[k,n] = 1 + (n mod 19)`。17/19与selected M/N tile stride 互素且周期大于对应tile数，使不同tile起点具有
  不同phase；每个rank贡献非零且随rank变化，漏掉、重复或错配任一rank/tile都会改变结果。
- target call的lhs/rhs/result storage format必须都是F16；这里使用的是F16 GEMM指令形态，语义profile规定内部以
  F32 fused accumulator累加后舍入到F16 destination，并不发射F32 operand/result GEMM指令。
- 冻结expected为
  `C[m,n] = ((136 + 16 * (m mod 17)) / 16) * (1 + (n mod 19))`，raw SHA-256为
  `f82ced1cea5d133a8f4640a527025333a80cbf2e49e7a529dc9c7c941e20360f`。
  最大值465.5；输入和每rank最终partial可由f16精确表示。冻结323种`(m mod 17,n mod 19)`组合已经按
  selected ordered-tree的逐步f16 RNE顺序穷举，虽然21种组合在中间节点发生`0.125`舍入，最终root与冻结expected
  仍逐bit一致。因此本case可以使用raw byte exact验证该固定payload和执行顺序，而不把有限样本提升为通用浮点
  GEMM/collective结论。
- 每rank两个2 MiB输入合计已超过可用SPM窗口，32 MiB replicated output也显著超出该窗口；成功package必须
  通过真实M/N traversal tiling、movement和fixed-capacity planning，不能由single-tile fixture冒充。

上述shape和payload只是验证参数。通用协议来自typed function boundary、sharding geometry、structured contraction、
collective、memory/effect/token和package resource关系；compiler/runtime代码不得按这些常量分支。

## Checkpoints

1. **Source / SPMD**：新增Board test fixture；验证global/local boundary、16个K slice、replicated output及
   StableHLO→local matmul+all-reduce handoff。既有PyTorch/XLA `row` strategy test继续证明framework mark_sharding
   走同一production helper边界；full-4096板测fixture直接以StableHLO program directory进入，不增加板测对framework
   importer runtime的依赖。
2. **Tiling / memory / communication**：fresh compile full shape；审计实际指令的lhs/rhs/result均为F16、各K chunk
   无重叠完整覆盖local K=256且没有发射F32 operand/result GEMM；审计terminal all-reduce自身按result tile物化、
   local GEMM producer slice融合、完整M/N traversal和all-and-only output stitch；同时核对local K、SPM high-water、
   per-tile DTE payload/range、all-rank loop/control instance、GEMM completion→send source和recv wait→consumer ordering，
   以及one-shared-ELF publication。
3. **Static / no-card gates**：新增脚本contract tests、configured no-card、相关unit/lit及动态import/manifest readback；
   no-card路径不生成百MiB raw payload。
4. **Board gate**：只读验卡确认online、heartbeat、0% utilization、无进程和资源基线；随后单个armed one-shot进程
   生成raw输入/expected并执行fresh package。outer deadline只终止该进程；失败后不retry/reset/power。
5. **Repeat / cleanup**：首次完整exact后再执行独立fresh invocation；两次均要求16个status SUCCESS、16份output
   exact和完整cleanup。执行后只读SMI与执行前基线比较；异常状态立即停止后续device effect。
6. **收尾**：同步tasks/16、tasks/17、progress和必要memory，顺序执行full host gate与hardware gate，记录
   unsupported/skipped清单和环境identity；计划归档、队列移入Done Index并提交相关改动。

## 2026-07-22 首次执行记录

本轮从同一full-4096 f16 source重新形成schema-v5/status-v2 package。实际post-SPMD IR为每rank
`tensor<4096x256xf16> x tensor<256x4096xf16>` local `linalg.matmul`，其结果进入16-rank sum
`wafer.linalg_ext.collective.all_reduce`；output保持replicated `tensor<4096x4096xf16>`。selected shared ELF为
40512 bytes，反汇编确认每rank按M步长256、N步长512遍历，静态GEMM参数为`M=256,K=256,N=512`，因此每rank
动态执行16×8=128个GEMM tile；每个DTE payload为`0x40000`即262144 bytes。local K始终为256，GEMM、每轮
all-reduce add、DTE wait和tile writeback之间存在显式local fence。

fresh production no-card CTest实际执行并以6.27秒通过。第一次armed命令在任何device effect前被host expected哨兵值错误
拦截；修正`expected[4095,4095]`为258.5后重新执行。真实board invocation在11.82秒内到达trusted completion、D2H和
完整output comparator，但resource 2在byte 0即失败：`expected=0x40, actual=0x46`。按one-shot策略没有继续第二轮、retry、
reset或power。执行后只读`tsm_smi`与前置基线一致：`9248M / 65536M`、0% utilization、无进程，因此这是clean numeric
mismatch，不是已证明的provider poison；Q35继续保持`doing`，hardware raw-exact gate未完成。

已确认的layout事实是：rank-2 GEMM输入和结果均物化为`Cx`；当前logical all-reduce lowering却显式
`getOrMaterialize(..., MemLayout::Tensor)`并把结果记录为Tensor，后续`CommAllReduceOp` lowering也拒绝非Tensor SPM buffer。
所以实际路径包含`Tensor -> Cx`的lhs/rhs materialization和GEMM result的`Cx -> Tensor` materialization，再以Tensor执行
all-reduce。Direct DTE range acceptance本身已能解释Cx，但这不足以让collective自动获得Cx legality。该硬编码确认阻断了
producer→communication→consumer的layout传播；它是下一轮必须隔离和修复的问题，但当前证据尚不能断言它就是byte-0
mismatch的唯一根因。

下一轮先从保留产物做分层定位，不盲目重跑硬件：

1. 对照selected Instr/target-call参数、lhs/rhs `Tensor -> Cx` segment、RHS orientation和GEMM result `Cx -> Tensor`
   segment，区分local GEMM/layout错误与collective错误；必要时分别构造无communication GEMM和已知partial all-reduce隔离case。
2. 将all-reduce设计为same-layout typed operation：只有input/recv/result physical encoding完全相同、elementwise accumulator和
   DTE byte/range均合法时保留Cx/NCx；tail padding/valid lane无法证明时回退Tensor。转换只在真实boundary/consumer要求时出现，
   不能按Q35 shape特判。
3. 收口本轮终审发现但尚未形成完整fresh gate的通用问题：internal all-reduce作为producer时必须拆成独立task或fail closed，
   不能留下原始full collective和tiled clone；Direct DTE loop-instance匹配还需证明IV到tile traversal的结构映射，不能只比较
   loop bounds/step/order；value-less/global memory effect不能被issue→wait buffer isolation静默放过。
4. 上述host正负例、production whole-variant和no-card全部fresh通过后，再只读验卡并执行单次armed复测；首错仍立即停止。

## 2026-07-23 standalone GEMM隔离与CRT修复

以production StableHLO入口新增rank-one无communication f16 GEMM：`256x256 x 256x512`，即full-4096 winner实际发射的
单个GEMM tile shape。lhs为单位阵，rhs为
`1 + ((17*k + 3*n) mod 1024) / 8`，因此CPU oracle直接是`C[m,n] = rhs[m,n]`；所有值均可由f16精确表示，
每个output只有一个非零乘积，完整262144-byte output可以使用raw exact。该case继续经过production
StableHLO normalization、physical selection、SPM/DDR、target conversion、device link、schema-v5 package和
per-rank pointer-block runtime，不手写post-SPMD/Instr IR，也不进入collective。

修复前fresh ELF反汇编显示semantic normal/normal最终调用`SetTransflag(0, 0)`；真实板端3.47秒到达trusted completion、
D2H和cleanup，随后在resource 2 byte 2发生numeric mismatch。current V5.6 Kcore source和SDK GEMM example均要求RHS raw
hardware bit与semantic orientation相反，即normal/normal发`(0, 1)`；按该错误读取Q35旧payload还可重放出首元素
84.375（f16 `0x5546`），其little-endian首字节`0x46`与7月22日板端首错逐bit一致。

CRT现只在raw packet边界映射该事实：v1固定发`(0, 1)`，v2发
`(lhs_orientation, !rhs_orientation)`；Instr/TargetCall/public signature继续携带semantic orientation。更新后的fresh ELF
反汇编已确认调用点为`a1=0, a2=1`，target CRT conformance、focused lit和production no-card通过。同一standalone
case随后在真实板端3.41秒完成，resource 2完整262144 bytes `exact=true`；执行后只读设备仍为
`9248M / 65536M`、0% utilization、无进程。

该结果证明当前shape上的Tensor→Cx输入、GEMM Cx计算、Cx→Tensor结果和WDMA writeback可形成正确完整输出，并将7月22日
首错根因从collective排除。all-reduce强制Tensor仍是独立layout优化缺口，但不再作为该numeric mismatch的解释。下一步以
当前compiler、CRT和Q36 collective实现fresh编译full-4096 package，静态核对新ELF后执行一次16-rank raw-exact board gate。

## 2026-07-23 strided DMA修复与完成证据

rank-one大shape隔离case使用同一production StableHLO入口执行
`4096x256 x 256x4096 -> 4096x4096`，不做SPMD sharding或communication。planner选择
`M=512,K=256,N=512`，形成8×8共64个tile；one-hot lhs让CPU expected直接取对应rhs row。修复前该case完成
trusted terminal、D2H和cleanup，但完整比较在byte 8192，即第二个output row首字节，读到未写回哨兵值。

current Kcore `__set_rdma_config`/`__set_wdma_config`和正常caller共同证明
`ConfigStrideIteration`的inner与三层stride均以logical element计数；BOOL以logical bit计数，wrapper再pack为byte。
Wafer Instr/TargetCall/public CRT ABI继续保持byte-level descriptor，CRT到vendor wrapper的唯一边界把
`inner_bytes`和三层byte stride一起做checked element conversion。旧实现只转换inner，导致f16 strided RDMA/WDMA
的实际byte hop放大两倍。修复后的fresh ELF反汇编确认f16 inner/stride均除以2；同一rank-one大shape在板端3.24秒完成，
32 MiB output逐字节`exact=true`。

full-4096 fresh production no-card随后通过。armed hardware CTest连续执行两轮fresh invocation，耗时19.05秒；
每轮16个rank均达到`entry_return`，resource 2、6、…、62的16份32 MiB output全部逐字节`exact=true`，
expected SHA-256保持
`f82ced1cea5d133a8f4640a527025333a80cbf2e49e7a529dc9c7c941e20360f`。执行后只读设备回到
`9248M / 65536M`、0% utilization、无进程基线，全程未调用retry/reset/power。由此本计划completion gate闭合；
证据只覆盖固定workload/environment，不完成Q22.C、性能或timing。

## 不算完成

- 只编译小shape、只看到`linalg.matmul`/symbol/manifest或只跑no-card。
- full shape成功但没有证明complete M/N traversal、accepted SPM range和all-rank communication ordering。
- 只比较rank 0、抽样输出、摘要hash或容差结果；本case要求16份完整raw exact。
- 使用手写post-SPMD local IR、manifest旁路、runtime GEMM特判或test-only launch ABI。
- timeout、untrusted completion、部分copyback或资源未回基线后继续触卡；或调用reset/power掩盖失败。
- 把单case结果写成Q22.C完成、通用GEMM数值profile、性能/timing或physical coordinate结论。
