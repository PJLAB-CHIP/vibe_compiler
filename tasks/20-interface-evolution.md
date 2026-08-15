# Wafer 接口演进与格式兼容边界

状态：2026-08-15完成 active compiler、runtime、tool 和测试中的版本归属收敛。本文件只拥有版本边界、
兼容策略和跨层一致性要求；各 IR、package、runtime、profiler 和 qualification 字段语义仍由原编号文档拥有。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  current frontend program directory、CardExecutable、target modules、runtime/device records、qualification inputs
  以及同一源码树内生成的辅助metadata。
- Current stage responsibility:
  区分仓库内同步演进接口与真实兼容边界；每个可独立部署或持久保存的外围格式只保留一个版本owner；
  删除nested field、算法、hash domain、模型名称和内部工具记录各自建立的版本线，并把必要兼容检查集中在
  parser、loader或ABI入口。
- Output IR / files:
  语义不变的current IR与输出；保留的外围格式具有唯一版本检查，内部表示和辅助记录只存在一种current形态。
- Downstream consumer:
  frontend verifier、compiler pipeline、package writer/reader、RuntimeSession、device record reader、profile report、
  target model和qualification gate直接消费current表示，不在各层传播版本分支。
- User-level driver / named pipeline:
  现有wafer-compile、wafer-run、profile与qualification入口不增加兼容mode或版本选择参数。
- Explicit non-goals:
  不改变IR语义、target能力、数值策略、设备记录布局或package字段含义；不删除第三方/release版本、NPY等
  外部格式版本、许可证版本、硬件规范版本、设备runtime版本检查或历史archive/raw evidence。
- Completion gate:
  active源码中只有本文列出的真实外围边界可以拥有Wafer格式/ABI版本；repo内同步producer/consumer没有旧reader、
  dual path、nested版本字段或带vN的算法/模型/hash-domain名称；相关canonical roundtrip、negative parse、build、
  unit、lit和工具测试通过，文档只描述current格式。
```

## 2. 允许拥有版本的边界

版本是兼容性承诺，不是修改计数器。当前只允许下列边界拥有Wafer版本：

| 边界 | 版本owner | 理由 |
| --- | --- | --- |
| `ExecutablePackage/manifest.json` | manifest顶层字段和runtime parser | compiler输出可由独立runtime进程读取 |
| target runtime ABI | target identity定义和compiler/runtime中央检查 | compiler生成的调用与独立runtime/CRT必须匹配 |
| TX81 profiler record |共享C ABI header中的record header | device writer与host reader独立执行，binary layout必须fail closed |
| Direct-DTE status |共享C ABI header | target CRT写入、host/runtime读取的binary状态合同 |
| 原始profiling/qualification evidence |每种独立文件的顶层字段 | 证据可能脱离生成进程长期保存并由后续工具读取 |

上述边界只能有一个current版本常量和一个集中检查入口。类型、常量和函数名不带当前数值后缀；版本数值只出现在
实际wire/header字段或ABI identity中。旧版本不被支持时直接拒绝，不保留translation path。

设备报告的runtime版本、第三方依赖版本、外部文件格式版本、许可证版本和vendor规范版本是外部事实，继续精确记录，
不计入Wafer内部版本线。

## 3. 不拥有版本的表示

以下对象随同一源码revision同步演进，直接原位修改并同批更新所有消费者：

- C++ API、pass、analysis、非持久化IR、diagnostic和测试helper；
- `forward.meta`中的nested distributed boundary，以及repo-owned SPMD helper生成的parameter shard metadata；
- profile activation引用的plan/site map内部记录及evidence中可由已验证输入重建的版本副本；
- target LLVM内部metadata、loader policy选择、static cost模型名、site correlation/key规则；
- numeric/bulk/model qualification的hash-domain separator、算法名、当前policy/schema字符串和内部dependency snapshot；
- workload fixture、当前corpus和工具CLI中只用于表示“当前实现”的版本后缀。

这些位置使用稳定语义名和strict field verification。结构变化通过同步修改current producer/consumer完成，不能新增
`v2`名称、旧reader、fallback或用户可选compatibility mode。

## 4. 实施约束

1. 先建立每个版本常量的producer、wire位置、consumer和部署/lifetime事实；没有独立消费者的版本删除。
2. nested record不得复制外围版本。已经由parser验证的ABI/format事实不作为普通业务字段继续向下传播。
3. hash domain separator可以保留稳定语义字符串，但不得用`vN`代替字段定义或算法合同；改变语义时修改typed输入与测试。
4. 真正ABI常量采用无数值后缀的current名称；例如C/C++ identifier不写`_V4`，wire identity可保留实际ABI版本。
5. 同批更新代码、canonical fixtures、negative tests和current设计文档；archive和historical raw evidence不重写。

## 5. 当前实现

- frontend distributed boundary和parameter shard metadata不再携带nested版本；current parser对退役字段fail closed。
- profile activation保留唯一格式版本，plan、site map和evidence中的派生记录不复制该版本；device record仍由共享C ABI
  header拥有独立版本。
- target lowering metadata、dependency records、workload corpus、model/policy/hash-domain名称只保留current语义，相关
  producer和consumer同批迁移，不存在兼容reader或双写路径。
- package manifest、target runtime ABI、TX81 profiler record、Direct-DTE status以及独立保存的profiling/qualification
  evidence继续在各自parser或ABI入口集中检查版本。
