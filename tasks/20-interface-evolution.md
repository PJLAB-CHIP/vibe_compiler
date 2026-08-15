# Wafer 接口演进与格式兼容边界

状态：2026-08-15继续收敛 active compiler、runtime、tool 和测试中的current接口。本文件只拥有兼容边界、
兼容策略和跨层一致性要求；各 IR、package、runtime、profiler 和 qualification 字段语义仍由原编号文档拥有。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  current frontend program directory、CardExecutable、target modules、runtime/device records、qualification inputs
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
- Completion gate:
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

设备报告的runtime版本、第三方依赖版本、外部文件格式版本、许可证版本和vendor规范版本是外部事实，继续精确记录，
不改写成Wafer名称。

## 3. 不拥有版本的表示

以下对象随同一源码revision同步演进，直接原位修改并同批更新所有消费者：

- C++ API、pass、analysis、非持久化IR、diagnostic和测试helper；
- `forward.meta`中的nested distributed boundary，以及repo-owned SPMD helper生成的parameter shard metadata；
- profile activation引用的plan/site map内部记录及evidence中可由已验证输入重建的版本副本；
- target LLVM内部metadata、loader policy选择、static cost模型名、site correlation/key规则；
- numeric/bulk/model qualification的hash-domain separator、算法名、当前policy/schema字符串和内部dependency snapshot；
- workload fixture、当前corpus和工具CLI中只用于表示“当前实现”的版本后缀。

这些位置使用稳定语义名和strict field verification。结构变化通过同步修改current producer/consumer完成，不能新增
编号名称、旧reader、fallback或用户可选compatibility mode。

## 4. 实施约束

1. 先建立每个版本常量的producer、wire位置、consumer和部署/lifetime事实；没有独立消费者的版本删除。
2. nested record不得复制外围版本。已经由parser验证的ABI/format事实不作为普通业务字段继续向下传播。
3. hash domain separator可以保留稳定语义字符串，但不得用`vN`代替字段定义或算法合同；改变语义时修改typed输入与测试。
4. 真正ABI常量采用无数值后缀的current名称；C/C++ identifier和wire identity都不携带Wafer自定义版本号。
5. 同批更新代码、canonical fixtures、negative tests和current设计文档；archive和historical raw evidence不重写。

## 5. 当前实现

- frontend distributed boundary和parameter shard metadata不再携带nested版本；current parser对退役字段fail closed。
- profile activation、plan、site map和evidence不携带编号字段；device record由共享C ABI header中的magic、size、offset和guard
  定义唯一current布局。
- target lowering metadata、dependency records、workload corpus、model/policy/hash-domain名称只保留current语义，相关
  producer和consumer同批迁移，不存在兼容reader或双写路径。
- package manifest、target runtime ABI、TX81 profiler record、Direct-DTE status以及独立保存的profiling/qualification
  evidence在各自parser或ABI入口检查唯一current identity、exact fields、size、layout和digest，不检查Wafer自定义版本号。
- Board calibration保留可复用的硬件观测方法、输入生成、host oracle和结果校验，并迁移到唯一current package、runtime、
  status和qualification接口。可以直接生成current package的raw probe必须通过current no-card入口；依赖尚未闭合的
  global lowering的source case保留current global source和oracle，但不得注册成可执行的no-card或板端结论。
- Board CTest只注册current接口能够实际执行的合同与probe；不恢复旧reader、旧CLI、旧manifest字段、旧SPMD carrier或
  已退役的多版本路径，也不把历史板端输出作为current测试输入。

## 6. Board calibration迁移边界

Board calibration按其真实编译边界迁移，不按旧runner或历史case名称整批保留：

| 类型 | current处理 | 完成证明 |
| --- | --- | --- |
| raw device probe | 保留device source、request/record布局、输入构造、oracle、guard和lifecycle校验；统一生成current package并使用current Tile resource/status合同 | 22个`current-interface` no-card CTest覆盖transport、NCC、barrier、DDR、SPM、worker、CT、NE、datamove、cache和instruction family |
| current global source可直接lower的case | complete-Tile add保留完整f16 output oracle、重复完成和profile Primary/Count/Trace校验；普通与profile路径分别生成current package | 2个额外`current-interface` no-card CTest和1个profile host contract通过 |
| 可直接执行的板端入口 | CMake只登记真实存在的driver与精确参数；统一runner串行调用，不在runner复制case catalog | 10个Board CTest对应10个runner step；无板环境不执行它们 |
| 依赖未闭合global lowering的source case | 保留current global StableHLO source、deterministic input和host oracle；不构造退役SPMD carrier或package | optimizer comparison的11个source case包含full-4096 K-tiled GEMM与M-tiled profile合同；collective algorithm和collective traffic各有独立host source contract并显式记录阻塞条件 |
| 重复旧executor | 先把独有source、shape、dtype、oracle和校验迁入current source contract或current driver，再删除只服务退役接口的执行器 | CMake、inventory和source-organization检查不再引用旧executor |

Inventory中的`host_ctests`、`no_card_ctests`和`board_ctests`只能列CMake真实注册名称。catalog拥有case语义；一个代表性
no-card CTest证明driver能经current接口生成完整package，不等价于该catalog全部case已经取得板端结论。

具体迁移结果：complete-Tile add、complete-Tile barrier、K-tiled GEMM和DDR active-Tile contention使用Tile/current命名；
K-tiled current-source实测因SPM lifetime/capacity约束无法生成package，故只保留受测source/oracle和待lowering runner；M-tiled
profile的package equality、full-output与Primary/Count/Trace要求进入optimizer catalog，并由current global-source runner保留
完整编译、no-card、板端输出和profile校验路径；当前global lowering失败时该runner直接fail closed，不登记成可执行CTest。
两个collective carrier的source与oracle进入current source contract后删除carrier，不保留旁路执行接口。
