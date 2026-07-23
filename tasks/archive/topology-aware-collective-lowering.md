# Topology-Aware Collective Lowering 实施计划

状态：已完成。本文记录 logical collective 到 explicit p2p instruction 的拓扑驱动候选、通信
correctness 修复和静态 cost 闭环；稳定语义分别由 `tasks/04-topology-execution-mesh.md`、
`tasks/06-physical-dataflow-synthesis.md`、`tasks/11-instruction-ir.md`、
`tasks/13-communication.md` 和 `tasks/16-verification-contract.md` 拥有。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  post-SPMD `wafer.linalg_ext.collective.*`、rank-specialized buffer-level collective、
  `wafer.execution.mesh`中的logical rank domain与rank-to-endpoint placement，以及其引用的
  `wafer.target.topology`规则邻接和unavailable endpoint事实；候选输入始终是isolated
  complete-rank clone。
- Current stage responsibility:
  从current typed topology/mesh fresh构造可重算的execution-topology analysis；为每种collective
  枚举固定小集合的合法algorithm/chunk/order参数；Tree参数由有界interval DP派生，并要求中序遍历严格保持
  `rank_group`。每个参数立即展开为真实p2p、local movement/reduction、token、wait和fence IR。memory
  planning和Direct DTE all-rank matching后，从final send IR、canonical source rank和logical peer重算
  payload bytes与minimum-hop link-byte demand，
  交给统一whole-variant Pareto/static policy选择。
- Output artifact / IR:
  不含collective或algorithm attr的memory-planned complete-rank instruction variants；winner只保留
  explicit send/recv/wait/local-work/effect IR和既有Direct DTE binding。topology-derived cost、
  shortest-path结果和候选参数只在当前analysis/rewrite调用栈中存在，不序列化。
- Downstream consumer:
  whole-variant exact-resource Pareto与target static policy、Direct DTE target conversion、package
  publication，以及后续no-card/model/board verification。
- User-level driver / named pipeline:
  production仍只由`wafer-compile`完整pipeline进入；不增加communication算法CLI或要求用户手工拼pass。
- Explicit non-goals:
  不绑定某张卡或固定4x4 rank编号，不把Ring/Tree写回logical collective IR，不猜未经target contract
  证明的实际路由、cycle、带宽重叠或PMU时间，不新增runtime算法选择，不在本任务实现segmented exchange、
  arbitrary weighted graph、cross-card packet protocol或board reset/power流程。
- Completion gate:
  当前五类production collective均完成实现审计和correctness回归；All-Reduce Ring使用真实
  reduce-scatter+all-gather分块算法，standalone Reduce-Scatter具有Direct baseline与分块Ring候选；
  singleton logical collective在产生tile communication前折叠为local identity；规则mesh/torus、explicit
  placement和unavailable endpoint的rank mapping/shortest-hop分析有正负例；同payload候选可因minimum-hop
  demand产生不同whole-card cost并影响统一winner；ordered Tree严格保持`rank_group`中序并用于floating
  collective，没有numeric permission的floating Ring被拒绝；production rank-count=1/16 no-card链路通过且
  accepted IR不出现旁路plan/cost attr。
```

## 边界与选择原则

- logical collective只表达通信语义、rank group、axis/combiner和SSA/effect；算法不是语义。
- `wafer.target.topology`与`wafer.execution.mesh`是当前规则拓扑和placement的唯一事实源。实现不得读取
  target profile名、workload名、symbol spelling或默认rank编号来恢复拓扑。
- 当前IR能精确表达规则card/tile grid、card mesh/torus、unavailable endpoint及all-available/explicit
  placement。任意带权、不规则link graph需要后续typed topology扩展；本任务不以side table或硬编码补猜。
- 在没有确定路由合同时，shortest-path hop是图上可证明的minimum link traversal，不冒充实际方向链路负载、
  拥塞或cycle。selection同时保留payload injected bytes与minimum-hop link-byte demand。
- “选优”只表示在有界、已materialize且通过共同legality/resource gate的候选集合中按当前typed facts选择；
  不声称求解所有collective算法、chunking、routing和overlap的全局最优。

## Checkpoint A：合同与现状审计

- 逐项冻结 all-gather、reduce-scatter、all-reduce、all-to-all、collective-permute 当前消息量、round、
  local work、completion和拓扑参数缺口。
- 将Q35暂时转为依赖本任务；不把其单个GEMM数值失败改写成本任务的通用通信结论。
- 用primary sources核对标准Ring All-Reduce、collective算法的消息大小/round tradeoff和topology-aware
  synthesis原则；来源只约束算法事实，不照搬其它项目IR或实现。

完成条件：编号设计、任务队列和本文对pipeline cut、cost语义与非目标一致。

## Checkpoint B：共享 Execution Topology Analysis

- 从module的唯一execution mesh和其引用topology解析logical rank到typed 4D endpoint；all-available与
  explicit placement必须共用一份实现。
- 从规则邻接和unavailable mask计算可达性及logical peer间shortest-hop distance；overflow、缺失、
  不一致和不可达均fail closed。
- verifier、target lowering和whole-card cost逐步消费同一无状态helper，删除重复rank mapping事实源。
- analysis不写IR attr，不缓存跨clone状态，不返回长期route plan。

完成条件：mesh/torus、显式置换、unavailable detour、非法/断连及确定性测试通过。

## Checkpoint C：Correctness 与标准分块算法

- All-Reduce Ring改为chunked reduce-scatter + all-gather；每个round只传一个chunk，local reduction只作用
  于对应chunk，完整result由全部reduced chunk组成。不能被rank数整分的静态payload要么用typed ragged
  chunk明确实现，要么只拒绝该Ring候选并保留合法Tree baseline。
- standalone Reduce-Scatter保留Direct all-to-owner baseline，并加入`P-1`轮topology-derived Ring候选；
  每轮只传一个result-sized chunk并显式归约，不能形成typed连续chunk时只拒绝Ring clone。
- All-to-All补齐remote insert完成后的最终local visibility fence；当前网络路径保持fixed-size direct
  exchange的最小payload量。`MoveInsertSlice`下层仍为每slot复制完整累计result，该local movement优化移入
  后续独立任务，不能冒充已完成，也不得牺牲non-contiguous axis的显式pack/unpack语义。
- Collective-Permute仅在没有incoming edge时zero-fill；remote recv与local copy分别具有正确producer/
  consumer completion，避免fill与recv并发写同一buffer。
- singleton All-Gather、Reduce-Scatter与All-Reduce在logical-to-tile边界折叠为identity，不要求channel、
  不分配recv buffer，也不生成`Comm*`或DTE op；group size大于一才进入算法候选。
- All-Gather Ring、Reduce-Scatter Direct和保持`rank_group`中序的ordered-Tree All-Reduce保留为已知正确
  baseline，同时补充其non-contiguous/fail-closed负例。

完成条件：message identity、send/recv bytes、local reduction次数、token/wait/fence、singleton identity和
非2次幂rank group均有直接IR测试；All-Reduce标准Ring全卡payload为 `2*(P-1)*B`，Reduce-Scatter Ring
每rank为`(P-1)`条result-sized send。

## Checkpoint D：Topology-Derived Candidate 与 Hop-Aware Cost

- Ring候选参数携带compiler-private有序cycle；Tree候选携带root与parent/children edge；Direct使用semantic
  group index的确定性issue order，不伪装成topology参数。Tree在`rank_group`连续区间上做interval DP，先
  最小化edge shortest-hop总和，再以最大/总root distance及logical rank确定性解平局；中序必须严格等于
  `rank_group`。它不是MST加center，也不使用root 0、XOR/binomial模板。所有参数均由current
  topology/placement确定，不进入dialect。
- 至少保留一个确定性baseline，并加入固定小上界的topology-local候选。任一额外候选失败只丢弃该clone。
- whole-card late analysis按canonical source rank解析每个final `dte_send.peer`，计算
  `sum(payload_bytes * shortest_hops)`；static loop multiplicity和算术overflow沿现有knowledge/reason传播。
- exact Pareto与NoC static priority显式比较minimum-hop demand。没有可证明拓扑时该维度为Unknown并保留
  baseline，不用带宽常数合成伪时间。

完成条件：相同注入bytes、不同rank order/tree edge的候选得到不同hop cost；统一winner在规则拓扑和explicit
placement测试中选择更低hop候选，且不读取算法名或candidate ordinal。floating All-Reduce Auto实际使用
ordered Tree并保持left/local/right leaf次序；显式floating Ring在没有numeric permission时结构化拒绝。

## Checkpoint E：Production Replay 与收尾

- 运行相关unit、lit、production-shaped rank-count=1/16 no-card和现有collective/whole-variant回归；
  检查unsupported/skipped而不是只看汇总通过。
- 搜索accepted IR、package和public CLI，确认不存在route/cost sidecar、算法attr、task编号或硬件特判。
- 同步编号设计、任务队列与可复用memory；记录已实现候选、静态hop语义和仍未实现的small-message/
  irregular-topology扩展。
- 审查diff，提交本任务相关改动；Q35只在本任务gate闭合后恢复推进。

完成条件：本轮fresh验证有可复现命令与实际执行计数，文档/代码/测试一致，相关改动已提交。

## 完成证据

- full-feature build 的 Wafer unit tests 实际执行 431/431，通过；其中包含 execution-topology、
  collective topology、communication alternatives/completion、whole-card hop cost、large-K tiled
  collective 与 Direct DTE all-rank matching。
- full-feature lit 实际发现 225 项，223 项执行通过；2 项 `*-disabled.test` 是 feature-on 配置的
  inverse negative，显式 unsupported 清单中没有 StableHLO/XLA、framework importer、numeric、
  bulk 或 SystemC 主线缺依赖项。
- numeric model 56/56、bulk model 18/18，SystemC integration/event/DTE 组件全部通过。
- host-only CTest 明确排除 `hardware` label；首次 30 项中 29 项通过并暴露 nested CMake
  venv symlink 解析问题，修复后失败的 SystemC configuration gate 独立重放通过。该验证未调用板卡。
- production-shaped rank-count=1/16 runtime no-card、cluster Direct DTE no-card 和 full-4096
  K-sharded GEMM no-card均在同一host-only CTest中执行通过。

这些证据闭合当前规则 topology/mesh、至多16-rank exact Ring/ordered-Tree analysis 和
minimum-hop静态选择。它不声明任意带权图、实际route/拥塞/cycle、浮点cyclic Ring或板端性能。
