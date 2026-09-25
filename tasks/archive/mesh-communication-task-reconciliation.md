# 通信实现任务的调度收拢

2026-09-22，用户指出`mesh-communication-materialization`仍被当作下一项开发工作，
但没有给出具体剩余实现。本次核对提交、当前源码及已有验收后，移除这个重复的独立排队条目。
本次只修正文档和任务归属，不修改编译器、不重新运行测试，也不签发新的技术验收。

## 记录为什么发生冲突

| 提交或记录 | 实际含义 |
| --- | --- |
| `6c5e764c` | 将通用Ring构造从TX81 transport分离，增加抽象participant/distance接口及直接测试 |
| `cfcd92ba` | 收敛通用collective与frontend边界；当时重新打开架构合同，旧board-ready不能直接沿用 |
| `0feee94e` | 通用ordered exchange与reduction discovery继续收敛，更新算法接口、物化和测试 |
| `40b80b60` | 记录上述复审后通信及Q53的board-ready；包含主机/no-card/SystemC证据，不是实卡全覆盖 |
| `413cef1e` | 进入板端接续，Add/Direct-DTE通过，生产source-to-DTE与通信矩阵继续验证 |
| `d44b677a` | 将真实板测交给独立`board-testing`，原通信项改为queued并保留笼统的“编译器通信收敛余项” |
| 后续板测归档 | Ring AllGather、personalized AllToAll、ReduceScatter、AllReduce按各自实际路径完成指定实卡范围 |

后续progress沿用了queued行；实施计划又保留9月6日的board-ready段落，两者没有按具体剩余产出收拢。
这不能证明需要重新开发通信分层，也不能仅凭源码存在就把全部历史合同标为done。

## 已有实现与证据保留在哪里

- 稳定算法/transport边界由[13号3.3节](../13-communication.md#33-通用算法与tx81-transport边界)拥有。
- 通用算法接口及实现为
  [CollectiveAlgorithms.h](../../include/Wafer/Planning/PhysicalDataflow/CollectiveAlgorithms.h)与
  [CollectiveAlgorithms.cpp](../../lib/Wafer/Planning/PhysicalDataflow/CollectiveAlgorithms.cpp)。
  Ring消费opaque participant和distance oracle，ordered AllToAll消费logical mesh。
- 实际peer/combine物化在
  [DistributedCollectiveMovement.cpp](../../lib/Wafer/Transforms/Tile/DistributedCollectiveMovement.cpp)，
  对应的通用算法、actual collective和主尾块测试已注册于
  [StructuredToTileTest.cpp](../../unittests/Transforms/Tile/StructuredToTileTest.cpp)。
- 主机验收历史保留在[原实施计划](physical-dataflow-host-readiness.md)。其中9月6日的board-ready
  只描述当时版本，不代签当前源码的完整产品矩阵。
- 生产source-to-DTE及1024/1025/1031的实际通信数值、completion和清理证据见
  [板测归档](board-correctness-qualification.md)。AllReduce专项实际为direct贡献交换加Ring AllGather，
  不能将它写成所有Ring/recursive-doubling等可选算法都完成实卡。

## 收拢后的归属

| 工作 | 唯一归属 |
| --- | --- |
| 已实现的通用通信构造与TX81 lowering | 原代码、13号设计和上述历史证据；不再次列为待开发能力 |
| 当前版本受影响的完整通信/模型实卡、数值、guard与性能 | `board-testing`及统一板测计划 |
| 当前版本source/IR/package/oracle/no-card/SystemC完整主机矩阵 | `production-host-readiness`；消费已有通信实现，不再等待这个旧队列项 |
| 以后出现的具体通信实现缺陷 | 根据失败输入、actual IR、直接下游和覆盖缺口归入实际owner；不能仅凭旧标题启动重构 |

独立条目退出不表示未测范围已经通过，也没有删除13号或原矩阵中的正确性要求。
当前调度只由[progress](../progress.md)拥有，本文件不产生新的检查任务或设备批次。
