# Compiler Search Scalability

状态：Q41等待Q42。本任务只优化whole-variant search的编译时间，不修改Q39 NoC-resident语义，也不处理
Q40的Direct-DTE issue/wait并行。

```text
Pipeline position:
- Upstream artifact / IR:
  verified rank programs、complete-rank current IR、typed candidate domain和ordinary/profile compile request。
- Current stage responsibility:
  量化并优化candidate generation、attempt planning、analysis、late gate、clone/import/lowering和profile
  capture construction，删除不改变candidate domain、winner或artifact的重复工作。
- Output artifact / IR:
  语义不变的accepted whole variant、package/profile companion及稳定compile-time diagnostics。
- Downstream consumer:
  target/package/no-card/runtime、Q9 profiler和model-scale compile workflow。
- User-level driver / named pipeline:
  wafer-compile ordinary/profile production pipeline。
- Explicit non-goals:
  不用shape/op/name matcher跳过搜索，不关闭profile，不改变Q39 legality/profitability或Q40 choice/wait合同。
- Completion gate:
  per-stage wall、peak RSS、candidate/attempt/late-gate/clone/lowering/capture计数完整；search work有显式上界；
  M-sharded K=1024 case在相同Release环境满足时间门禁且winner不变。只运行直接搜索回归，不以板端、
  target model、profile报告或全量suite作为完成条件。
```

复现：16-rank FP16 `A[4096,1024] × B[1024,4096] -> C[4096,4096]`，A/C沿M分片、K不分片、
B replicated。production winner `--profile`运行16分12秒仍未发布package，约28个逻辑CPU持续工作、
RSS约18 GiB，随后人工停止。对照的K-sharded global `4096³` profile约8分钟出包。当前没有阶段计时和
candidate/attempt/capture计数，不能先验断言具体根因。

Q41先补阶段观测，再删除重复clone、analysis、late gate、accepted-artifact lowering和per-capture search。
验证只覆盖小型搜索回归和上述M-sharded K=1024 production winner profile，检查搜索上界、编译时间及winner
digest不变；不追加普通/profile双编译、K-sharded对照、no-card、板端或全量回归。绝对wall-time预算在首轮
阶段数据取得后写入本计划；未写入前Q41不能标记`done`。
