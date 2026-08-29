# Wafer 接口演进与格式兼容边界

本文件只拥有active compiler、runtime、tool和测试的兼容边界、
兼容策略和跨层一致性要求；各 IR、package、runtime、profiler 和 qualification 字段语义仍由原编号文档拥有。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  current frontend program directory、DeviceExecutable、target modules、runtime/device records、qualification inputs
  以及同一源码树内生成的辅助metadata。
- Current stage responsibility:
  区分仓库内同步演进接口与真实兼容边界；所有Wafer-owned边界只保留一种无编号current表示；
  删除nested field、算法、hash domain、模型名称、内部工具记录和外围格式各自建立的编号线，并把必要检查集中在
  parser、loader或ABI入口。
- Output IR / files:
  语义不变的current IR与输出；外围格式使用稳定schema/ABI identity和exact field/layout检查，不携带编号。
- Downstream consumer:
  frontend verifier、compiler pipeline、package writer/reader、RuntimeSession、device record reader、profile report、
  target model和qualification gate直接消费current表示，不在各层传播版本分支。
- User-level driver / named pipeline:
  现有wafer-compile、wafer-run、profile与qualification入口不增加兼容mode或版本选择参数。
- Explicit non-goals:
  不改变IR语义、target能力、数值策略、设备记录布局或package字段含义；不删除第三方/release版本、NPY等
  外部格式版本、许可证版本、硬件规范版本、设备runtime版本检查或历史archive/raw evidence。
- Done criteria:
  active源码中没有Wafer-owned `vN`/`VN`名称、编号schema字段或per-symbol版本；repo内同步producer/consumer没有旧reader、
  dual path或nested版本字段；相关canonical roundtrip、negative parse、build、
  unit、lit和工具测试通过，文档只描述current格式。
```

## 2. Current compatibility boundaries

下列边界可以拥有稳定的schema/ABI identity和严格结构检查，但不使用编号名称或编号字段：

| 边界 | current检查入口 | 理由 |
| --- | --- | --- |
| `ExecutablePackage/manifest.json` | schema identity、exact fields和runtime parser | compiler输出可由独立runtime进程读取 |
| target runtime ABI | ABI identity、descriptor registry和compiler/runtime中央检查 | compiler生成的调用与独立runtime/CRT必须匹配 |
| TX81 profiler record |共享C ABI header中的magic、size、offset和guard | device writer与host reader独立执行，binary layout必须fail closed |
| Direct-DTE status |共享C ABI identity、size、alignment和status合同 | target CRT写入、host/runtime读取的binary状态合同 |
| 原始profiling/qualification evidence |schema identity、exact fields和digest binding | 证据可能脱离生成进程长期保存并由后续工具读取 |

上述边界只有一个current identity和一个集中检查入口。类型、常量、函数、wire identity和header都不携带当前数值；
旧输入因schema、field、magic、size、layout、digest或ABI identity不匹配而直接拒绝，不保留translation path。

设备报告的runtime版本、第三方依赖版本、StableHLO portable compatibility target、NPY等外部文件格式版本、许可证版本和
vendor规范版本是外部事实，继续精确记录，
不改写成Wafer名称。

## 3. 不拥有版本的表示

以下对象随同一源码revision同步演进，直接原位修改并同批更新所有消费者：

- C++ API、pass、analysis、非持久化IR、diagnostic和测试helper；
- `forward.meta`中的nested distributed boundary，以及repo-owned SPMD helper生成的parameter shard metadata；
- profile activation引用的plan/site map内部记录及evidence中可由已验证输入重建的版本副本；
- target LLVM内部metadata、loader policy选择、static cost模型名、site correlation/key规则；
- target numeric backend/model qualification的hash-domain separator、算法名、当前policy/schema字符串和内部dependency snapshot；
- workload fixture、当前corpus和工具CLI中只用于表示“当前实现”的版本后缀。

这些位置使用稳定语义名和strict field verification。结构变化通过同步修改current producer/consumer完成，不能新增
编号名称、旧reader、fallback或用户可选compatibility mode。

## 4. 实施约束

1. 先建立每个版本常量的producer、wire位置、consumer和部署/lifetime事实；没有独立消费者的版本删除。
2. nested record不得复制外围版本。已经由parser验证的ABI/format事实不作为普通业务字段继续向下传播。
3. hash domain separator可以保留稳定语义字符串，但不得用`vN`代替字段定义或算法合同；改变语义时修改typed输入与测试。
4. 真正ABI常量采用无数值后缀的current名称；C/C++ identifier和wire identity都不携带Wafer自定义版本号。
5. 同批更新代码、canonical fixtures、negative tests和current设计文档；archive和historical raw evidence不重写。
6. CLI flag、tool action和library result同样只有一种current合同；语义修正时原位替换producer/tests/docs，不保留旧flag alias
   或让build-tree/install-tree走不同入口。

当前接口收敛的实施记录、迁移数量和Board calibration inventory见
`tasks/archive/interface-version-consolidation.md`。这些历史结果不扩展本文件的稳定兼容合同；current注册状态由CMake和
本轮测试结果确认。
