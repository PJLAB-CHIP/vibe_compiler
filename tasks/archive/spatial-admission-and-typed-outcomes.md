# Spatial admission 与 relation 协调收敛

稳定设计与覆盖矩阵在06号设计§5.1。本轮由用户明确授权处理正式typed路径、错误的仿射闭包结论，以及GEMM/conv两侧的proposal限制。

## 实施顺序

1. 保留domain builder到planning problem及search driver的typed失败，补正式none/search入口和已有domain的choice失败测试。
2. 删除memory中“仿射映射保持两种规则切分”的错误断言，以1025反向索引的可构造反例与正式回归替换。
3. 关系驱动的双向partition协调去掉全parallel consumer与零tensor-read producer门槛；完全重复需求合并，部分重叠拒绝。
4. 从新partition重建reduction groups及merge placement；保留原seed，生成两种传播顺序的合法候选。
5. 按06覆盖矩阵运行直接下游、组件回归与fresh no-card；canonical完整增量构建及无源码变动的no-op后复审、提交。

## 验证边界

无板环境只签本轮host/no-card；本项不要求新的硬件测量，也不代签board-testing。
正式入口unsupported测试必须调用产品driver，不以单独DemandPlanningSession加空assignment代替。
非均匀切分不新增schema；反向仿射回归证明relation能力和proposal表达范围分别在哪里。

## 实现与直接验证

- PlanningProblem保留domain的typed failure；none/search的demand结果都使用同一分类。删除单独的unanalyzable-root布尔旁路，
  canonical coordinate只表示结构选择，不能提前把分析预算耗尽当成unsupported。
- IndexRelation组合实际SSA路径；两种有界传播顺序覆盖有归约consumer和读取tensor的producer。合并重复需求，保留证明独立的轴，
  重建所有merge group/owner；无新边界或不被domain接纳的提案继续尝试其他输入。原seed和raw域不变。
- 原init只绑定merge。实际下游补齐Linalg body/operand接口对应、unit轴投影、native reduce rank合同、1D卷积到2D native视图、
  sparse peer source顺序、稀疏输出端口和同Tile多piece共享DDR root；没有新增join/wait或numerical rewrite。
- 算法对照Shardy双向factor传播；本项为两轮有界proposal而非固定点求解。API按官方MLIR接口文档及pinned Linalg/Tiling源码确认。

| 验证 | 本轮实际覆盖 |
| --- | --- |
| 双向partition | 96组named/generic GEMM/conv ×1024/1025/1031 ×4/16 Tile ×parallel/reduction ×两种顺序；exact coverage、contribution和merge owner |
| 需求与保留边界 | 重复需求、独立reduction轴、多consumer一致/冲突、halo重叠、反向仿射不可表示、无信息输入后继续尝试其他producer |
| 实际物化 | 60组named/generic ×1024/1025/1031 ×parallel/普通merge/独立Tile merge/长K/同Tile多个merge；全部经temporal和verifier |
| 实际memory/target | 12组长K候选完成executable；12组较大partial由actual allocator给出带conflict demand的capacity rejection |
| 同一输入的可编译性 | 上述12个容量负例的原始输入全部经正式search成功；拒绝的是选定coordinate，不是整个输入 |
| 输出边界 | 同Tile多个disjoint output piece共享一个结果root；重叠负例在layout preflight拒绝；无写入Tile保留既有ABI输出端口 |
| typed入口 | 1024/1025 tensor.pack由none/search返回unsupported；rank17静态relation预算不足由两入口返回indeterminate；已建domain的unsupported demand仅关闭当前choice |
| fresh产品 | GEMM tail1031 none；local reduce/local conv/prefill tail1031 none/search；conv-mixed-dag FP16/BF16 search，共9个PyTorch source→package→strict no-card |

容量负例不计为编译成功；no-card不计为设备执行或数值板测。本项不为更丰富的非均匀切分增加schema。

## 完成门禁

本轮canonical完整增量构建通过；完整`check-wafer`通过，包括277个lit、14个组件CTest、public-link/RunBoardIO、
reference/target numeric及17个SystemC测试。最终9个fresh PyTorch no-card全部实际执行成功。

SearchRouting固定样例的8个已尝试结构候选现在全部成功，原先因不完整输出端口而unsupported的7个候选已闭合；
回归保留确切计数8 accepted / 0 unsupported，没有放宽成“至少成功一个”。

实际内存负例仍保留：12个指定大partial坐标带真实allocation conflict demand退出，同一批原始输入全部经正式search找到accepted executable。
本项完成host/no-card边界，不产生新的板端性能或数值验收结论。
