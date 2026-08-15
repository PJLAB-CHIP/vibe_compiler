# Compiler 命名整改计划

状态：`done`。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  active dialect、analysis、transformation、conversion、compiler driver、target/runtime接口和diagnostic。
- Current stage responsibility:
  根据每个对象处理的IR、实际变换和输出，删除错误抽象、拆分混合职责并修正误导名称。
- Output IR / files:
  语义和ABI不变、名称与实现职责一致的active source、IR schema、tests和current docs。
- Downstream consumer:
  后续compiler、runtime和工具开发。
- User-level driver / named pipeline:
  现有`wafer-compile`、`wafer-opt`和named pipelines；本任务不新增driver。
- Explicit non-goals:
  不做关键词机械替换，不改archive，不改变IR语义或用户ABI。
- Done criteria:
  受影响的声明、实现、CMake、测试和current docs同步完成，并通过fresh build与对应测试。
```

## 执行方法

1. 沿active build实际包含的代码检查ODS、public API、pass/pipeline、文件名、diagnostic和current docs。
2. 对每个可疑名称读取实现和consumer，写清它处理的IR、实际变换、输出和ABI影响。
3. 无独立职责的抽象直接删除；混合多个变换的对象先拆分；职责清楚但名称错误的对象再重命名。
4. 每批沿一条完整调用链同步声明、实现、CMake、测试和文档，不保留内部兼容别名。
5. 每批运行fresh build和直接受影响的测试；全部批次完成后再做全仓构建与测试。

命名采用仓库pinned LLVM/MLIR的常见形式：type使用语义名词，function使用说明实际动作的动词短语，conversion说明
源表示和目标表示，optimization说明实际算法。同一概念在ODS、C++、pass、文件和diagnostic中使用同一名称。

## 整改结果

本轮按定义和直接consumer逐项核对，不以关键词替换代替语义判断：

| 范围 | 收敛结果 |
| --- | --- |
| compiler public API | target侧收敛为`PhysicalTileExecutables`、`TargetLLVMModules`、`LinkedTargetModules`和`CompiledProgram`；函数名直接说明compile、link、readback或write package |
| physical-Tile lowering | 原来混合转换、bufferization和memory planning的入口拆成明确的Instr lowering、SPM/DDR assignment与resource verification；文件和测试使用同一职责名 |
| whole-device selection | search、candidate evaluation、resource verification和selection分别命名；不再用含糊的协调、前沿或阶段结束术语代替实际动作 |
| target/runtime | profile数据写入、运行参数构造、target command验证和模型执行分别命名；C/C++符号与diagnostic同步 |
| MLIR source | verifier、target execution facts、instruction verification和StableHLO collective lowering按实际IR层命名；声明、实现、CMake和lit文件同步 |
| tools/tests/docs | Python字段和函数使用file、record、evidence、write、validation等具体对象或动作；current tasks、memory和presentations同步到现有API |

没有为名称建立源码黑名单。现有dependency snapshot v1/v2字段属于已落盘格式，上游StableHLO/PyTorch-XLA类名和环境变量
属于外部API，历史archive文件名属于只读导航；本轮没有通过改写这些字符串制造兼容性变化。所有一手C++/Python抽象、文件名、
diagnostic和current文档已经退出旧术语。

## 验证结果

- `build/q45-fresh`独立配置并完成419个构建步骤；没有复用`build/wafer-dev`生成物。
- fresh `WaferUnitTests`排除Q49/Q52已知长搜索后为629/629通过；长搜索不属于命名整改门禁。
- fresh CTest 1–11为11/11通过，包含lit、dependency snapshot、PyTorch/no-card和配置测试。
- fresh CTest 13–20为8/8通过，包含runtime IO、public-header/link smoke和feature-off link closure。
- source/IR organization、Python compileall、profile report、dependency helpers及受影响Board协议的host-only测试通过。
- `git diff --check`通过；未执行真实板端测试。

`test/Board/wafer_compiler_optimization_campaign_catalog.py`仍包含Q50.S未来迁移的旧source reference；它在Q54前后均未注册到
CMake/CTest，当前自检会因缺少已退役`CandidateRewrites.cpp`而失败。本轮没有伪造新的production实现来满足该字符串检查；
该catalog必须在Q50.S建立actual structured alternatives时按新实现重写或删除，不能作为当前compiler能力证明。
