# RISC-V CPU代码生成实验记录

下文四组实验已撤回，保留方法比较、验收标准和失败取舍，不作为当前实现或继续执行的授权。
当前目标生成合同见[14号设计](../14-target-code-generation.md)，逐次结果见[性能索引](../../docs/board-performance-results.md)。

## RISC-V CPU代码生成优化

输入为已完成Instr/TargetCall验证的target LLVM IR、当前CPU选择和唯一CRT source；输出仍为同一device-link事务的
kernel对象、厂商CRT对象及最终ELF，直接消费者为原ExecutablePackage/no-card/runtime。用户入口和ABI不变。
本层只优化CPU控制程序；硬件指令、布局、dtype、同步和SDK生命周期保持。不开fast-math，不发明packet缓存或MMIO协议。

方法比较采用[LLVM跨模块优化](https://llvm.org/docs/LinkTimeOptimization.html)、目标CPU标量features、代码体积优化和
已知engine的SDK分派消除。跨模块常量传播能减少封装调用，但内联后代码体积、寄存器压力和目标CPU指令选择也会改变。
当前这四组实卡均没有确认净收益，实验修改已全部撤回，kernel/CRT编译和SDK调用保持原实现。
具体配置、失败取舍与证据在板端性能记录，不将已撤回的typed issuer当作current ABI。

性能判断以普通无插桩device elapsed为准，不能用调用数、代码大小或trace开销替代总耗时。
`TsmExecute`读类型分派是已知重复工作，但减少该工作尚不等于缩短关键路径。
当前SDK的`__execute_td`在返回前清零整个packet，简单保留可变packet不能复用配置；
任何后续循环不变准备外提，必须先证明实际builder/issuer字段读写和重入边界，再确定唯一IR owner及下游ABI，不能先引入缓存。

| 覆盖 | exact输出或拒绝 | 直接下游witness |
| --- | --- | --- |
| 跨模块常量传播、原CRT编译、CPU features和优化级别 | 同一Instr/target LLVM输入、唯一符号定义、编译/link错误不发布 | 真实device-link及原子失败oracle |
| 已知engine分派实验：五个engine、三个worker、普通/显式profile | 原SDK相应issuer、worker、packet mutation及返回值提升一致 | 有界host ABI oracle、真实ELF；未采用实验的测试不保留为生产fixture |
| BF16 rank3+ 2048主块与1031尾块、多Tile/多block | 完整output、guard及退出健康；主块不获益的实验无需继续尾块板测 | fresh source→package/no-card；主块三次普通实卡；采用后的最终版本须再闭合尾块实卡 |

候选评估完成须记录主机门禁、fresh no-card、普通实卡和取舍；采用新实现另须canonical/no-op及最终尾块资格。
退化或未确认收益的候选不进入生产。本项不修改search CPU先验，不宣称已测得纯CPU时间或达到硬件极限。
