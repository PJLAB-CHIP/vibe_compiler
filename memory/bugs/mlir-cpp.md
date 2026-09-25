# Bug模式：MLIR变换与C++所有权

按主题记录可复用根因；入口见[bug索引](../bugs.md)。规则冲突时以当前编号设计与[AGENTS](../../AGENTS.md)为准。

## 必执行循环不能携带虚构的零次路径

- 现象：长串嵌套循环已完成bufferization和Instr转换，SPM分析仍在ordered successor与`recordUse`中长时间拆分路径。
- 根因：timeline给每个`scf.for`无条件增加optional-body decision，常量上下界已证明非空的循环也保留不存在的bypass；
  pending access经过后续循环时反复相交/相减，虚假分支组合放大工作。
- 修复模式：只读current常量上下界及正step，已证明非空时body继承parent路径；动态/零次边界保留optional，
  真正的loop-local条件分支仍为repeatable，不作为跨iteration的互斥证明。沿用原allocation、effect、completion及packing验证。
- 防复发：同一长段嵌套输入检查exact路径、局部state与actual Instr/completion/SPM，配对覆盖真实分支、动态trip/step、
  零次completion和跨worker冲突。分析变快不等于设备变快，产品需另作fresh资格。

## 可选rewrite必须在依赖扩展后重新检查集合重叠

- 现象：两个原本不共享输入Region的合法exchange在依次合并时，第二次clone解引用已经删除的operation。
- 根因：只按原始通信component判断独立性，遗漏两者后来纳入的同一个纯tensor初始化Region。
- 修复模式：首次改IR前完成所有actual依赖扩展，对重叠集合求并集并重新扩展到不动点；再检查SSA/effect和全部participant，
  每个实际Region只由一个rewrite集合拥有。某Tile不能合并时拒绝关联component整组，不留下部分参与者。
- 防复发：共享初始化与单Tile外部effect成对覆盖，检查关系retarget和下游Instr、SPM、transport，不只检查合并数量。

## 不得从函数参数位置或同型关系猜测output boundary

- 现象：functional tensor program的最后一个真实input与result同型时，被TileModule set lowering当作trailing output参数删除；
  final Tile entry参数减少，但frontend resource binding仍完整，16个Tile统一在TargetABI exact-boundary gate失败。
- 根因：把structured op内部的destination-style语义错误提升成source function ABI，并用
  `numArguments - numResults`恢复角色。
- 修复模式：source函数只按已验证functional arguments/results消费；需要可写destination时，在private scheduling clone
  中显式追加result destinations，记录source argument count，物化后只删除这个精确区间。
- 防复发：覆盖单input同型result、同型尾部普通input、multiple results和no-work Tile；检查final entry arguments/results
  与frontend typed bindings精确双射。

## StringRef视图必须绑定在SmallString最后一次修改之后

- 现象：payload resolver持有指向`SmallString tensorProgram`的StringRef，构造时该path尚未append
  "tensor-program"子目录；后续append触发缓冲重分配，resolver仍指向旧缓冲，stat出transactionRoot而非
  tensor-program目录，constants存在性检查误报MissingPayload且找不到新物化的文件。
- 根因：可修改的SmallString在视图（StringRef成员）建立之后继续append；SmallString内联缓冲与堆缓冲的
  重分配时机不透明，旧缓冲内容残留使错误表现为"路径少一段"而非崩溃。
- 修复模式：指向SmallString/String的StringRef成员必须在最后一次修改之后构造；需要跨阶段复用的路径
  用std::string拥有并只取视图，或用值传递的std::string成员。
- 防复发：新代码里任何`StringRef member`绑定本地SmallString时，先确认绑定后该SmallString不再被append/
  resize；评审时对"resolver持有路径引用"这类长生命周期视图重点检查。

## Move-only结果不能引用先于结果销毁的staging文件

- 现象：compiler成功返回`DeviceExecutable`，但返回后target consumer首次读取parameter就报文件不存在；同时byte-identical
  helper shard虽未进入range identity，重复文件仍随结果存活，规模账本却显示open次数恒定。
- 根因：handoff只保存transaction root下的path，外层scope cleanup在public compile返回时先删除root；source每次range/digest
  再按path打开，真实open/read没有进入只统计establishment的账本；未adopt candidate也没有明确的最后consumer边界。
- 修复模式：让结果类型自己RAII拥有稳定parent下的唯一目录和move-safe read handle；owned bytes成为header/extent/digest最终事实源；
  最低层强制read window并记录actual open/window/bytes/max；最后一次verification后adopt真实source并销毁全部剩余candidate。
- 防复发：直接测试move后删除staging仍能读取且析构清理文件；Card边界断言candidate归零；原地改写fixture明确超过pinned mmap
  threshold；大range断言window数量/bytes、最大window与零新增open；source-to-package规模账本分开验证identity计数和实际I/O work。

## C++17 std::variant比较要求所有alternative同时定义==和!=

- 现象：为6个`TileEntryArgument` payload struct定义friend `operator==`后，`std::variant`比较、`TileEntryArgument`
  aggregate和manifest序列化仍编译失败，报"no match for operator!=（no known conversion）"。
- 根因：C++17的`std::variant operator==/!=`在libstdc++实现中要求每个alternative类型同时有`==`和`!=`；只写`==`并依赖
  C++20的rewritten candidate在C++17不存在。
- 修复模式：closed union的每个payload struct成对定义friend `operator==`和`operator!=`；同类新字段加入时同步两个运算符。
- 防复发：新增进入`std::variant`的payload类型时，先写编译级小测试确认比较完整；不能假设`==`隐含`!=`。

## llvm::ArrayRef模板推导不接受隐式转换

- 现象：`llvm::ArrayRef<int64_t>`与`std::vector<int64_t>`直接`==`比较编译失败，报template argument deduction
  substitution失败。
- 根因：模板实参推导发生在重载决议前，`std::vector`→`ArrayRef`的隐式构造函数不参与推导。
- 修复模式：任一侧显式构造`llvm::ArrayRef<int64_t>(vector)`后再比较；长期语义边界改用typed容器或循环比较。
- 防复发：ArrayRef与STL容器混用时，接口签名优先显式ArrayRef参数，调用侧避免依赖推导转换。

## 结构体持有悬空视图：unique_ptr容器移入owner而不是引用字段

- 现象：`PackageAssembly`先以局部`std::vector<std::unique_ptr<TargetTensorJoin>>`构建`placement`裸指针视图，
  函数返回后manifest打印出垃圾值（diagnostic显示的bytes/offset随机）。
- 根因：视图指针绑定在栈上容器，容器析构后指针悬空；aggregate没有成员所有权。
- 修复模式：把`unique_ptr`容器本身移入`PackageAssembly`作为成员，视图字段只指向成员容器元素；destructor顺序由成员
  声明顺序保证。
- 防复发：任何返回结构体只要含"视图指针"字段，先确认被视图对象本身由同一结构体拥有；禁止栈容器+outlived view模式。

## Range-for解引用临时optional会留下悬空range

- 现象：Release单测在遍历MLIR attribute时崩溃，前面的optional存在性和size断言均通过。
- 根因：generated accessor返回`std::optional<mlir::ArrayAttr>`值；`for (... : *op.getBindings())`把range绑定到临时optional
  的内部对象。当前C++17规则不延长该optional的生命周期，进入循环前引用已失效。
- 修复模式：先保存accessor返回值，检查其存在性，再遍历`*bindings`，使optional覆盖整个循环生命周期。
- 防复发：检查range表达式中返回值的owner，尤其是optional解引用和临时容器的view；Release测试仍须检查全部元素，不能删除循环。

## pinned LLVM版本的API事实

- `llvm::errc`没有`state_not_recoverable`：内部不变量错误用`llvm::errc::operation_not_permitted`；
  `llvm::sys::path::append`最多接受path+3个组件，超过必须分两步append；`llvm::sys::path::join`不存在。
  编译期宏路径烘焙会在install后失效，外部工具/资源一律运行时发现。
- `PresburgerSet`没有默认构造函数（只有space/move构造）：带`PresburgerSet`成员的结构体用
  `std::optional<PresburgerSet>`字段（`IndexSetResult`同款），默认构造的set语义用"absent"表达，
  不用裸成员+聚合初始化碰运气。
- tensor方言接口归属：`tensor.insert_slice`原生实现`DestinationStyleOpInterface`但**不**声明
  `TilingInterface`；`tensor.pad/pack/unpack`的`TilingInterface`走external model，未注册该model的
  context里`isa<TilingInterface>(pad)`直接fatal（"promised by dialect but never implemented"）。
  判定DAG节点（DPS && Tiling）时必须DPS在前短路，不能让Tiling先查；生产与测试都要注册
  `mlir::tensor::registerTilingInterfaceExternalModels`（声明在`TensorTilingInterfaceImpl.h`，需要显式include）。
- pinned的`linalg.generic`汇编要求显式`} -> tensor<...>`结果类型；省略时op解析为零结果，
  报"cannot name an operation with no results"（该报错指被命名的op无结果）。
- `tensor::ExtractSliceOp::getMixedOffsets/getMixedStrides`、`InsertSliceOp::getMixedOffsets`、
  `PadOp::getMixedLowPad`返回`SmallVector<OpFoldResult>`（动态值不保证常量），常量用
  `mlir::getConstantIntValue`解析后走`staticSlice`/`staticInsertSlice`。

## operation count不能作为IR mutation snapshot

- 现象：analysis/query缓存借入一个FuncOp后只记录顶层operation数量；原位修改nested op的attribute、operand或result type时，
  operation数量和root指针均不变，旧relation cache仍被接受。
- 根因：结构元素数量不是IR语义identity；MLIR rewrite可以在不增删operation的情况下改变legality、index relation和lowering。
- 修复模式：需要在不可控mutation边界外fail closed时，使用pinned MLIR的nested `OperationFingerPrint`观察operation identity/
  nesting、attributes/properties、blocks、operands、successors和result types；borrow token只表达共同lifetime，不再维护第二份
  process-global generation。若owner能控制全部mutation，优先由明确的analysis invalidation边界重建query。
- 防复发：invalidation测试必须修改nested semantic attribute、operand或type且保持operation count不变；只测试插入/删除op不能
  证明snapshot完整。

## 扩展 FuncOp 参数必须同步 argument attrs

- 现象：小型unit里函数签名扩展通过，但正常source-to-package的frontend函数带`arg_attrs`时，追加一个scheduling destination参数后
  verifier报告“argument attribute array ... got 3, expected 4”。
- 根因：代码分别调用`setFunctionType`和entry block `addArguments`，绕过了FuncOp对signature、block argument和argument attr
  数组的一体化维护。
- 修复模式：使用pinned MLIR的`FuncOp::insertArgument`追加typed boundary参数及空`DictionaryAttr`，由op API原子更新三者。
- 防复发：任何函数签名扩展除无attr unit外必须经过一个带frontend argument metadata的真实source-to-package gate；本轮
  CHAIN/CROSS/GEMM定向lit即覆盖该路径。

## Definition-only result不能靠伪造use或discard plan闭合

- 现象：`insert_slice`覆盖producer输出的一部分后，canonical spatial仍执行对应shard并需要output storage；该result没有transfer/publication，
  storage把它当成漏use拒绝。若简单允许所有empty-use，又会放过真正漏掉的observable publication。
- 根因：在actual SSA之前预先列出result、carrier和storage，遇到无consumer的future result后只能再发明discard和self-use让计划自洽。
- 修复模式：structural materializer只创建current exact demand和observable/effect closure要求的execution。Pure result物化后没有actual use时，
  由局部DCE/canonicalization删除其compute和allocation；effectful op保留actual effect；observable result必须有actual publication use。不创建
  discard ID、fake self-use或storage edge。
- 防复发：rank-3 1024/1025 multi-piece overwrite检查actual IR无orphan pure compute/allocation；独立负例删除observable publication
  仍必须失败。不得用名字、shape或伪造use判断discard。

## 函数边界type converter不能逐参数重复扫描symbol uses

- pinned One-Shot会为函数的多个参数和结果重复调用functionArgTypeConverterFn；在每次callback中扫描整个module会随ABI宽度放大编译工作。
- 同一次layout/bufferization内按实际FuncOp保留一次边界空间选择。标准FuncOp/CallOp bufferization保留函数身份和callee引用；下一次调用重新查询。
- 用被调用helper与外部entry核对全部参数/结果的空间，并记录查询次数；大型source必须比较相同输入/预算及最终Instr，不能只凭局部计时宣称优化。
