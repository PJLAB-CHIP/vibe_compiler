# Component依赖闭合实施计划

状态：`done`。稳定源码owner由18号设计定义，MLIR pass/conversion规则由19号设计与`AGENTS.md`定义；动态顺序只读
`tasks/progress.md`。本项修复源码目录已经迁移、但pass注册、private include、tool link和非C++资产仍保留旧依赖的问题。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  current Wafer source tree、CMake target graph、public/private headers、TableGen pass declarations、registered tests和Q52 staged assets；
  不改变任何compiler IR或package/runtime格式。
- Current stage responsibility:
  使Transforms、Conversion、CodeGen、Driver、Target、Simulator和tools的include/link方向与18号合同一致；拆分pass注册owner；
  删除无consumer资产；让organization checker覆盖本轮暴露的反向依赖、fixture和source-tree artifact。
- Output IR / files:
  IR、ABI和产品输出不变；输出无隐藏反向include、无过宽tool link、无Transforms↔Conversion注册耦合、无孤儿fixture和cache的源码树。
- Downstream consumer:
  Q52 structural materialization/layout/movement接线、wafer-opt、wafer-compile、oneDNN qualification、Simulator和全部本地tests。
- User-level driver / named pipeline:
  不增加或重命名CLI/pass/pipeline；wafer-opt仍注册同一pass集合，production driver仍调用同一实现。
- Explicit non-goals:
  不启动Q52第12–20项，不把`Driver/PhysicalDataflow`、layout cleanup或buffer replacement listener声明为已接入production；
  不改变search合法域、数值语义、SPM/completion、target ABI、package schema或hardware结论。
- Completion criteria:
  下面四批依赖闭合；staged Q52符号仍明确不进入wafer-compile；fresh canonical build、全部registered CTest、public header、
  source/IR/dependency organization、target graph、孤儿资产扫描和二次Ninja no-op通过并提交。
```

## 线性施工

1. **Pass owner闭合**：Transforms与三个Conversion分别生成pass declarations；transform registration不include/link conversion；
   Conversion自己的registration header和link target闭合实现集合；pipeline/tool显式组合二者；CodeGen直接include InstrToLLVM
   conversion API。
2. **隐藏依赖闭合**：CompilationResult builder回Driver；Direct-DTE transport contract回pure Target；Simulator TargetCall descriptor
   从Invocation executable拆成共享contract；program↔target dtype映射移到CodeGen；Package/Simulator/CMake声明真实直接依赖；
   conversion recorder adapter进入Driver，Transforms只保留通用relation replacement listener；Tile dataflow query进入Analysis。
3. **Tool与Linalg owner闭合**：oneDNN qualification移除`WaferCompiler`宽link；structured graph normalization public header回
   `Transforms/Linalg`；Tile layout cleanup使用稳定的`LayoutOptimization`路径和API；Wafer-owned Rust e-graph crate/target不再标为
   ThirdParty，外部`egg`仍由pinned dependency owner管理。
4. **非C++资产与门禁**：删除确认无consumer的Python/LLVM/JSON fixture；lit/CTest禁止source-tree bytecode；清理cache；扩展
   source organization检查，拒绝本轮退役路径、反向include、宽link、Transforms/Conversion聚合、旧private header guard和
   generated artifact。
5. **最终验证**：确认Q52 staged symbols仍只在component archive/unit中、未进入`wafer-compile`；运行完整build/CTest、全部public
   header独立syntax check、source/IR/dependency检查、current文档扫描和二次no-op。

## 覆盖矩阵

| 覆盖类 | 输入/结构 | exact断言 | 直接witness |
| --- | --- | --- | --- |
| pass registration | transform pass、三类conversion pass、wafer-opt、production direct factory | pass spelling与pipeline文本不变；Transforms target不链接Conversion；每个registration只注册自己的集合 | full lit、pipeline unit、wafer-opt smoke |
| hidden include | Package→Driver、Transforms→Driver/Conversion、Target→Frontend、Simulator Memory→Invocation | 退役include为零；CMake direct edges与header edges一致；无target SCC | organization/dependency checker、public header syntax/link |
| tool link | oneDNN source-spec/calibrate/freeze/validate | 无`WaferCompiler`依赖；四种CLI合同与qualification结果不变 | qualification CLI、link-closure、binary dependency query |
| Linalg/e-graph | structured graph normalization、Rust FFI、layout cleanup | API只有一个新路径；旧header/file/target为零；egg仍locked/offline；典型复杂chain统计不变 | Linalg unit、lit、dependency record |
| assets/cache | Python helper、MLIR/LLVM/JSON Inputs、CTest/lit imports | 每个保留资产有current consumer；退役资产为零；测试后source tree无`__pycache__`/pyc | source checker、full CTest后artifact scan |
| Q52 boundary | UnifiedSearch、ActualResultController、StructuredProgram adapter、layout/listener staged mechanics | 符号不进入`wafer-compile`；文档只称staged，不代签Q52 production | `nm` product reachability check、progress Q52为next |

本项只使用现有测试shape，不新增IR算法。受影响IR回归继续包含已有rank>=3、1024/1025/1031、attention、reduction、Tile和tail矩阵。

## 完成证据

- canonical完整build通过；87/87 registered CTest通过，内含255/255 lit且无skip/unsupported。
- 101个`include/Wafer` public header逐个独立syntax check通过；8个public link smoke随CTest通过。
- source/IR/dependency organization通过；592-target CMake图无SCC，`WaferTransforms`无Conversion edge，oneDNN资格工具无
  `WaferCompiler` edge。
- `wafer-compile`不含Q52 staged search/controller/layout/listener符号；相关符号只保留在component archives与unit owner。
- 完整build后的第二次build为Ninja no-op；source tree无Wafer-owned Python bytecode/cache。
