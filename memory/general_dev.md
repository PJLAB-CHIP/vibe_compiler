## 数值验证consumer

- source workload corpus固定typed input/parameter和独立CPU expected；CModel与后续board consumer都比较同一expected，
  但各自独立实现compute、numeric、memory和transport，不能共享结果生成器。
- 不长期维护accepted instruction IR的第二套解释器。它与target CModel复述同一执行语义，却绕过target LLVM/ABI/SystemC，
  增加重复事实源和维护面；compiler correctness继续由verifier、conversion legality和named pipeline gate证明。
- 数值differential的非零payload本身不足以证明覆盖。MLP这类组合case应在独立CPU loop oracle之外增加敏感性检查：逐个
  屏蔽hidden channel、逐层清零bias都必须改变完整expected output。固定payload使用明确整数PRNG和二进制可精确表示缩放，
  避免host distribution差异。
- source invocation装配只共享typed compact tensor、accepted rank slice和已验证payload loading。CModel随后按exact Kernel
  ABI slot编码physical bytes并从slot解码完整output；板端路径也必须独立copyback后比较完整输出。
- target-model mismatch返回非零但不删除已经原子发布的verified package；model CLI的index/tolerance语法错误必须在编译和
  发布前拒绝。

## Wafer compiler local build harness

- 每次提交的验证范围按改动影响面选择，不默认跑全量。普通局部修改先构建受影响target，并运行对应
  unit/lit或脚本自检；只有改动跨多个pipeline边界、公共IR/ODS/interface、ABI/schema、核心调度/
  memory/target基础设施、构建依赖，或任务completion gate明确要求时，才运行全量unit/lit/CTest。
  定向验证已经覆盖改动及其直接consumer时，不为“每次提交”机械追加全量长测试；最终结果要明确
  写出实际验证范围和未运行项。
- 本机普通CPU构建默认使用`cmake --build <build> -j$(nproc)`，不要无依据固定成低并发。只有已观察到
  内存压力、共享机器约束或特定工具不支持并行时才主动降并发，并在进度中说明。板卡执行仍由独立
  resource lock保持单进程串行，不能把CPU构建并发规则套到硬件case。
- 板卡可用窗口用于执行已冻结的probe，不用于现场补catalog、重写oracle或重新做全域审计。离线阶段必须
  先把待执行叶子变成真正可枚举的case：input、expected或bounded observation、physical guard、device
  dispatcher、timeout/cleanup、target/no-card和runner入口均闭合；只有catalog条目或“保守Unknown”不算
  case ready。上板前冻结对应提交和精确执行清单；板端只有实际失败暴露本case缺陷时才回到实现，不能因
  无关的新想法移动本次执行终点。
- 板端版本、ABI、loader symbol和最终ELF反汇编属于环境或相关实现变化时的一次性qualification基线，
  不能默认塞进每轮workload热路径重复执行。qualification按重启后的板测会话复用：software/runtime
  identity未变化且设备持续正常时，只在会话开始确认一次空闲/可用性，后续case直接launch，不重复
  `tsm_smi` inventory、版本/hash、反汇编、manifest/profile qualification或known-good Add heartbeat。
  harness/schema的no-card和host oracle检查属于实现变化后的定向构建验证，也不随每次相同板端invocation
  重跑。普通case的最小执行路径只保留单进程串行、bounded timeout、当前case的完整结果/guard和必要PMU
  读回、正常资源释放；成功后直接继续。只有timeout、device/query error、untrusted terminal、资源未回
  基线、software/runtime identity变化或结果暴露新的低层歧义时，才升级到状态、反汇编、库/firmware
  版本和ABI诊断。恢复后的known-good heartbeat用于重新资格化execution/completion面；有明确风险边界
  的隔离probe仅在对应任务合同明确要求时使用前后heartbeat，不能把它扩成所有case的固定仪式。
- compiler-sensitive硬件probe按profile绑定的证据台账推进：先做no-card/schema gate，再按单engine、
  serial control、disjoint overlap、alias relation和completion/visibility分层上板。每条结论必须区分
  static、single-vector observed、calibrated、supported、unknown和excluded；性能观察不能反向扩大semantic
  legality。packet/register字段只能证明请求和路由，完整非零output、全range readback及双侧guard才是
  correctness oracle。
- 板端结果必须按“已测事实→compiler/runtime消费决策”收口。若descriptor、result、guard、count和completion
  通过，而PMU尚不能区分bank或overlap，应该分别记录“该geometry有界合法”和“当前不做bank coloring/不启用
  overlap”，不能把整项笼统写成Unknown。Unknown只保留给确实未被区分的单一机制边界，并同时给出当前保守
  行为；已经取得bounded observation的case不因缺numeric exact或性能模型退回未测试状态。
- 每个硬件probe在实现前先写明至少两种仍可能成立的行为解释，以及哪个boundary result、raw sink、
  counter relation或guard能把它们区分开；只证明“请求完成”的smoke不能关闭机制问题。板端结果收口时必须
  同时记录“观察事实、排除的解释、尚未排除的解释、当前compiler/runtime决策、下一种区分性case”，不能只
  汇总通过数量。已有case已经具备该区分力时只绑定并执行，不为Unknown重复造同义case。
- 大型硬件校准在上卡前使用机器manifest做叶子级完备性门禁。文档大类只作compiler-consumer导航；
  每个语义叶子必须解析到catalog中的具体case对象或带typed gate及原因的`static-negative`/
  `isolated-deferred`，并绑定calibration/held-out层、独立oracle、physical guard、matching completion、
  资源预算和registered no-card gate。大类状态只能从叶子自动汇总，不能由文件存在、测试注册、总case数或
  手填空`remaining_preparation`变绿。catalog暴露的叶子分组必须全部被manifest记账；同一具体case确需服务
  原生能力和layout能力时，重复引用要有精确白名单与引用次数断言，其余分组只允许一次引用。opcode、layout
  或counter inventory不能因缺case而消失。“有明确fail-closed处置”属于准备完成，“由邻近case外推”不属于。
  shared dispatcher/package的target C build/link和host oracle通过只证明上卡资产可用，不得写成board evidence。
- catalog总数、板端完成数和剩余数按唯一case ID集合求并集；phase/alignment、focused subset或compiler-consumer
  view可以重叠，不能把各view数量直接相加。状态同步至少同时给出唯一总数、已板集合、仅离线集合以及重叠view
  的说明，避免把已执行case重新列成待测，或把一个control row重复计为两项完成。
- 硬件行为未知或没有exact numeric oracle时，优先构造有界`board-observation`，不能直接归入deferred。
  只要owned ABI能表达descriptor/write span，case有prefix/suffix guard、matching completion或最终safety
  drain、外层timeout和正常cleanup，就重复采样并保留raw result、physical span、request echo、execute返回与
  PMU；无seed的随机输出也按这个合同隔离执行。`isolated-deferred`只用于可能永久等待、越过owned range或
  缺少同一runtime session生命周期owner的输入；typed ABI根本没有对应setter/field的组合是
  `static-negative`，不能由相近wrapper或其它family代签。
- cache、slot reuse、resource lifetime等跨phase硬件probe必须由同一个runtime session拥有全部phase，并显式
  证明复用同一allocation/handle；若每个phase分别启动进程且cleanup会卸载program、释放allocation，就只能
  各自形成单invocation visibility case，不能把两次结果解释为stale-before/control-after。无法表达同会话
  所有权时应保留`isolated-deferred`，直到runner合同补齐。
- TX81的`get_spm_memory_mapping(offset)`返回
  `KUIPER_L1SPM_UNCACHE_WEAKORDER_BASE + offset`（current base `0x30400000`），属于uncached
  weak-order mapped-SPM alias。通过该alias的CPU load/store用`fence iorw,iorw`/`sync`建立顺序，不能把
  mapped pointer传给dcache指令。raw `0x0 + offset` cacheable SPM alias和cacheable DDR是不同地址域；
  DDR publication/readback按实际owned cache line clean/invalidate，不能与mapped-SPM共享flush helper。
- 普通NCC instruction/descriptor校准不使用mapped-SPM做seed、guard scan或结果oracle：host payload经
  整槽RDMA进入SPM，NCC结果经整槽WDMA回host后校验result与guard。mapped-SPM只留给真实Kcore数据路径和
  专门的completion/coherence单变量A/B probe；是否需要`TsmWaitfinish`必须由具体方向和scope的板端对照
  决定，不能从alias地址属性或单个数值失败推断。
- local instruction的地址依赖从完整Instr IR重算：以typed MemoryEffects和SSA alias/root/view path建立
  RAW/WAR/WAW edge，pure same-worker NCC链按edge保持issue order并由current verified descriptor域的
  worker busytable落实；链内不为first conflict插`TsmWaitfinish`，RAR只在另有resource/control edge时
  保序。local drain只在NCC→Kcore/Direct DTE、跨worker join、barrier/structured completion backedge、
  terminal/host publication等completion-domain boundary物化并合并。runtime lowering在这些boundary建立
  issue→completion poll→boundary consumer的机器顺序；特定worker wait只用于其已证明scope，不能把一次
  `bywork(0)`诊断硬编码成通用handshake。strided dependency、跨worker同地址和default wait的扩展scope
  未闭合时继续Unknown。
- 板端case一旦timeout立即停止当前批次并隔离该execution context，不在同批次自动重试，也不调用
  reset、power或firmware替换。`tsm_smi` idle、0%利用率、memory baseline和无残留进程只证明管理面表面状态，
  不证明execution/completion面健康。发生异常并由用户恢复后，使用一次新进程的known-good Add heartbeat
  重新确认execution/completion面；heartbeat timeout就停止全部板测并由用户决定恢复方式。健康会话中的
  普通case不重复执行heartbeat。
- 板端诊断地址必须先由当前arena/range validator证明完整半开range的owner、reservation、alignment和
  legality；相邻地址可访问或较小instruction write成功都不能外推更大DMA range。任何未经证明的地址交换
  不进入上板诊断；若因此发生timeout，按poisoned context停批，不能靠cache flush或继续换地址恢复。
- register/header中的NCC queue depth只描述静态storage/register形状，不是可安全连续issue的outstanding上限。
  普通calibration只跑1/2/4，TDMA只跑1/2。恰好documented depth使用独立
  `documented-depth-manual`：packet构造后立即删除`TsmNew` builder，control读取紧随`TsmExecute`，一次只选
  一个engine/case且只跑一遍，前后各做known-good Add heartbeat，核对连续提交、最终completion、
  instruction count、完整output/guard。typed tight `D+1`只能在`D`已通过且取得显式manual授权后使用相同隔离
  边界；packet builder预先释放，相邻execute只做cycle采样，window后统一读control并完成matching wait/full
  oracle。CT/NE/RDMA/WDMA的`D=6`/`D+1=7`与TDMA的`D=4`/`D+1=5`均已board-observed，
  `D+1`证明documented depth不是完整lifetime总提交上限；但control观察时已空闲且blocking为0，不能声明
  并发resident、queue full或backpressure。任意更深overflow不执行。
- `TsmWaitfinish()`、`TsmWaitfinish_bywork(worker)`和multi-tile barrier必须按不同completion scope处理。
  current version-matched静态反汇编中default wait只轮询worker0，local fence直接复用default wait；非default worker使用
  matching `bywork`，跨worker join逐worker显式完成。短workload即使在default wait后结果正确，也可能只是在
  wait观察前自然排空，不能据此外推default会等待其它worker。逐指令wait相对window末尾单次wait的三轮对照
  方向一致地更慢，因此compiler不在same-worker NCC的每条RAW/WAR/WAW或地址复用edge后等待，只在
  latest-legal completion-domain exit放置并合并matching drain；不把单轮cycle写成固定cost。
- CT `VuVLoop`使用显式supported interface gate：`unit_elem_count == 64`且
  `full_elem_count * unit_elem_count == elem_count * full_unit_elem_count`，乘积关系以扩宽或checked
  arithmetic验证。违反合同的raw packet只做host negative，不能以hardware observation、held-out或隔离
  复测名义上板；历史unit 32/37 exact只能保留为out-of-contract hardware observation，不能授权compiler、
  runtime或production。普通`VuV`的geometry按自身独立合同处理，不能与`VuVLoop`互相外推。
- PMU parser必须把“counter可读”和“样本有效”分开：split counter先做稳定读取，再验证enable、scope在window
  内未变化和workload至少触发一个相关delta。enable缺失、scope变化或全部delta为零时样本保持
  `inconclusive`；PMU结论不能替代payload、guard和completion正确性。
- `tasks/progress.md` 是任务队列，不是设计合同。确定下一步时先定位队列项，再读该项指向的编号
  设计文档；不要从旧 progress 叙事、单个工具现状或历史 memory 反推出当前架构边界。若
  `memory/` 与编号设计文档或任务队列冲突，同步修 memory。
- 当实施路线在一个umbrella任务中穿插shared foundation、owner bootstrap、正式producer和下游consumer时，
  每个可独立调度边界都要有queue row和显式`blocked by`；不能把整个umbrella标成单一`next`，再让执行者
  从长计划正文猜状态迁移。需要下游直接消费才能证明完成的producer，拆成implementation row、consumer row和
  integrated completion row，避免隐式循环或提前`done`。
- `docs/`中的hardware/register/reverse-engineering资料只拥有source-backed evidence。production IR、ABI、transport、
  package/runtime policy只由对应编号设计文档拥有；编号合同更新后要搜索supporting docs中的“负责”“主目标”
  “后续自定义ABI”“当前已覆盖/仍待”等规范性或动态措辞，防止形成第二事实源。
- `tasks/plans/`中只有被`tasks/progress.md`当前实施计划索引引用的计划是执行入口；已完成且被新路线
  替代的计划移入`tasks/archive/`并在文件顶部标成historical/superseded，保留审计过程但不得据此恢复
  旧owner或旁路协议。实施计划不能依赖某个agent skill或工具目录才能解释和执行。
- 第三方依赖的固定版本集中在 `cmake/third_party/WaferDependencyVersions.cmake`；不要把 LLVM、StableHLO、
  Shardy、OpenXLA/XLA、PyTorch/XLA、torch-mlir、lit 或 gtest 版本散落到源码里。
- 新numeric/SystemC依赖进入production前分层取证：先枚举checkout/system package，再检查binary architecture和link
  closure，最后只在transaction-local目录用官方candidate源码做configure/build/run probe。每项至少记录exact version/
  commit或digest、license入口、导出的唯一CMake target、thread/TLS状态和最小运行结果；probe通过只形成candidate，不能
  替代统一版本文件、受管bootstrap、上游self-test和项目gate。configure始终显式指定source/build目录和工作目录，避免
  把上游临时文件写入repo root。
- 受管formal numeric依赖入口是`python3 tools/bootstrap_deps.py --numeric-model-deps --numeric-jobs <n>`；它从统一
  版本文件下载、校验并构建SoftFloat/TestFloat/m4/GMP/MPFR，完成conformance后原子发布
  `third_party/numeric-model/numeric-model-deps.json`。CMake启用`WAFER_ENABLE_NUMERIC_MODEL_DEPS=ON`时只消费该
  canonical record，不在配置期下载或fallback到宿主numeric library。
- 受管bulk model依赖入口是`python3 tools/bootstrap_deps.py --bulk-model-deps --bulk-jobs <n>`；它从统一版本文件下载、
  校验并clean-build固定oneDNN，完成API smoke后原子发布`third_party/bulk-model/bulk-model-deps.json`。启用bulk build时
  必须同时设置`WAFER_ENABLE_NUMERIC_MODEL_DEPS=ON`和`WAFER_ENABLE_BULK_MODEL_DEPS=ON`；CMake只消费canonical record
  生成的`WaferBulk::oneDNN`，不在配置期联网或fallback到system oneDNN。
- bulk gate要对称检查feature-on/off：on侧执行`check-wafer`、显式lit unsupported审计、完整CTest、真实
  `wafer-cmodel-qualify-bulk`三阶段和静态oneDNN符号/动态thread-runtime closure；off侧用基础`WaferUnitTests`证明binary
  不含oneDNN/OpenMP/TBB符号或依赖。qualification artifact路径必须是absolute canonical path，producer采用no-replace，
  因此重复测试应使用新的临时目录而不是覆盖旧record。
- 受管SystemC入口是`python3 tools/bootstrap_deps.py --systemc-model-deps --systemc-jobs <n>`；它固定官方3.0.2 archive，
  clean-build静态安装并用独立CMake consumer实际运行两个`SC_THREAD`的delta-event smoke，随后原子发布
  `third_party/systemc-model/systemc-model-deps.json`。启用时同时设置`WAFER_ENABLE_NUMERIC_MODEL_DEPS=ON`和
  `WAFER_ENABLE_SYSTEMC_MODEL=ON`；CMake只从canonical record指定的`SystemCLanguage`目录导入官方
  `SystemC::systemc`，不联网或fallback到宿主package。bootstrap生成的独立consumer保持
  `cmake_minimum_required(VERSION 3.16)`，因为它只验证imported package和C++17 smoke，不能额外要求宿主更新到
  CMake 3.24。plain model core不得包含SystemC header或链接该target。
- SystemC 3.0.2 public process/event headers使用RTTI和异常，而仓库LLVM/MLIR ABI是`-fno-rtti -fno-exceptions`。正式adapter要拆成
  只含plain C++ callback/opaque pointer的bridge TU：bridge可包含SystemC header并独立启用RTTI/异常，但不能包含LLVM/Wafer
  header；LLVM error/model TU保持仓库ABI且不包含SystemC header。不要给同时使用`llvm::ErrorInfo`的model target整体打开RTTI，
  否则会要求LLVM no-RTTI build没有提供的typeinfo。每个实际SystemC test executable只定义一个C-linkage `sc_main`，需要多次
  独立simulation的正负例拆成不同process。
- SystemC `wait()`会保留当前JIT/C++ stack；任何跨wait持有的endpoint/event引用都必须放在追加时引用稳定的owner中，不能指向
  可能被其它rank `push_back`扩容的`std::vector`元素。Direct DTE source具体读取点必须由model profile明确；当前untimed
  profile在typed sender/receiver匹配完成点读取source、原子写destination并完成双方event，不在send call到达时提前snapshot。
  ordinary NCC effect虽然在functional memory中原子commit，ordered-pending期间仍要保存typed read/write byte footprint：
  DTE issue的source不得与pending write重叠，destination不得与pending read/write重叠；matching participant join必须在
  DTE issue之前清除对应hazard，post-issue join不能追认已经提交的传输。strided access可保守扩成bounding interval，
  unknown/overflow保持fail closed。不能用“rank上存在任意pending NCC”替代range conflict，否则disjoint
  elementwise/GEMM会被错误串行化并可能形成全rank假deadlock；也不能先发布peer-ready再等待late join，否则会错误接受
  真实CRT `send_async`已经越过的依赖。
- 用 `python3 tools/bootstrap_deps.py --python` 把固定版本 Python 测试工具安装到
  `third_party/python`。
- 用 `python3 tools/bootstrap_deps.py --importer-sources` shallow fetch 固定版本 PyTorch/XLA、StableHLO、
  Shardy 和 OpenXLA/XLA source submodules 到 `third_party/<name>`；PyTorch/XLA 的 `WORKSPACE`
  `xla_hash` 决定 frontend 要匹配的 XLA 版本，顶层 `third_party/xla` 必须与它一致；不要 full clone
  上游历史作为默认 bootstrap。
- `third_party/pytorch-xla` 是PyTorch/XLA源码事实源；framework importer使用的可运行`torch_xla`必须由该checkout
  编译/安装得到，并让 importer Python 环境通过 `import torch_xla` 和顶层 `import _XLAC`。
  prebuilt `torch_xla` wheel不能作为source-build完成证明。
- `python3 tools/bootstrap_deps.py --importer-python` 只准备 PyTorch/XLA 源码构建需要的 importer
  Python packages；`torch_xla` runtime 必须随后从 `third_party/pytorch-xla` 源码用该 Python 编译/安装。
- `python3 tools/bootstrap_deps.py --importer-bazel`下载受管Bazel 6.5.0到`third_party/tools`，并按统一版本文件
  核对SHA-256 `a40ac69263440761199fcb8da47ad4e3f328cbe79ffbf4ecc14e5ba252857307`；
  `third_party/tools/bazel`是source runtime构建的默认入口，不依赖ambient Bazel/Bazelisk。
- `tools/build_pytorch_xla_runtime.py --jobs 8`是当前`torch_xla`源码构建入口，默认消费
  `third_party/python-importer/bin/python`、`third_party/tools/bazel`和`third_party/pytorch-xla`，并用Bazel override
  固定到本仓库`third_party/xla`/`third_party/llvm-project`及importer Python的`torch` headers/libs。不要resolve
  venv的`bin/python` symlink，否则会越过该venv的`sys.prefix`和site-packages。
- pinned XLA source需要能编译defaulted `noexcept` move的C++17 compiler；helper优先选择`gcc-10`/`g++-10`，
  也接受显式`CC`/`CXX`或`--cc`/`--cxx`，并在重构Bazel workspace前运行probe。compiler选择必须同时传入Bazel
  repository configuration和action environment；只修改shell `PATH`不足以改变已生成的`local_config_cc`。
- editable install完成后从`bazel info bazel-bin`指向的persistent output复制`_XLAC`和
  `_XLAC_cuda_functions`到source editable package，再用同一importer Python验证`torch`、`torch_xla`、顶层
  `_XLAC`和`torch_xla.stablehlo.exported_program_to_stablehlo`。不得从pip临时`build/lib.*`发布extension，也不能
  用只有纯Python package可导入冒充runtime完成；prebuilt `torch_xla` wheel仍不构成source-build证据。
- 当前依赖栈没有pinned LLVM/MLIR prebuilt URL。用`python3 tools/bootstrap_deps.py --llvm-source`
  同步`WaferDependencyVersions.cmake`固定的llvm-project commit，再独立build/install；当前不要使用
  `--llvm`，也不能把它写成可用的prebuilt bootstrap入口。
- 本地bring-up只有在明确接受非固定工具链风险时才可显式override：
  `cmake -S . -B build/wafer-bootstrap -GNinja -DMLIR_DIR=<mlir-cmake-dir> -DLLVM_DIR=<llvm-cmake-dir>
  -DPython3_EXECUTABLE=$PWD/third_party/python/bin/python -DWAFER_ALLOW_UNPINNED_LLVM=ON`。
- 当前统一依赖验证使用上述source-built pinned LLVM/MLIR install。默认build dir使用中性的
  `build/wafer-dev`，不要把阶段名、任务号或某个frontend依赖名写进长期build目录约定：
  `cmake -S . -B build/wafer-dev -GNinja -DMLIR_DIR=<pinned-llvm-install>/lib/cmake/mlir -DLLVM_DIR=<pinned-llvm-install>/lib/cmake/llvm -DWAFER_ENABLE_IMPORTER_DEPS=ON -DWAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS=ON -DWAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON -DWAFER_IMPORTER_PYTHON_EXECUTABLE=$PWD/third_party/python-importer/bin/python -DWAFER_XLA_SPMD_PARTITIONER_HELPER=$PWD/build/xla-spmd-helper/wafer_xla_spmd_partitioner -DWAFER_ENABLE_NUMERIC_MODEL_DEPS=ON -DWAFER_ENABLE_BULK_MODEL_DEPS=ON -DWAFER_ENABLE_SYSTEMC_MODEL=ON`，
  然后跑 `cmake --build build/wafer-dev --target check-wafer -- -j128` 和
  `ctest --test-dir build/wafer-dev --output-on-failure`。
- board-capable配置在同一pinned compiler配置上增加
  `-DWAFER_ENABLE_BOARD_RUNTIME=ON -DWAFER_TX_RUNTIME_ROOT=<tx-sdk-root>`；configure必须在该root内找到public
  `tx_runtime.h`和完整所需symbol的`libhpgr`，否则fail closed。这个开关只构建能力，adapter只在`--board`路径按需加载DSO；
  no-card路径不加载vendor library。board loader必须以同一个open fd完成hash和`/proc/self/fd` load，并按dev/inode核对全部
  dlsym provider；不能分两次按path读取。先用`ctest --test-dir <board-build> -LE hardware --output-on-failure`验证host closure。
  live test还需显式配置默认关闭的`WAFER_ENABLE_BOARD_TEST_EXECUTION=ON`及预期runtime digest/version/PCI/device/tile。
  runtime顶层只有kernel/model两种launch kind；板测case不能通过provider专用ABI选择第三种入口。先执行rank-one kernel
  `WAFER_EXECUTE_HARDWARE_TESTS=1 ctest --test-dir <board-build> -R '^wafer-board-single-op-add$' --output-on-failure`，
  确认未skip并在前后只读核对device memory/process与firmware status。真实16-tile命令只有对应static/fake gate先通过后才串行执行
  `WAFER_EXECUTE_HARDWARE_TESTS=1 ctest --test-dir <board-build> -R '^wafer-board-kernel-grid-add$' --output-on-failure`；Direct DTE再独立执行
  `WAFER_EXECUTE_HARDWARE_TESTS=1 ctest --test-dir <board-build> -R '^wafer-board-cluster-direct-dte$' --output-on-failure`。各项之间只读验卡，
  首个失败或超时即停，不retry/reset/power，也不能用临时runner路径替代已注册CTest。开启importer/SPMD helper的配置还应在未armed
  环境实际执行对应kernel-grid和Direct DTE production no-card gate，它们从production source fresh编译到
  schema-v6 package并进入all-rank no-card consumer，不允许返回77或以fake manifest替代。schema-v6 module不带rank，
  entry不带symbol；rank覆盖只由entry到module引用表达，module通过typed exports定位`prepare`/`main`。
  vendor adapter由`tools/wafer-run` executable拥有，通用`WaferRuntime`只拥有typed provider接口和lifecycle executor；
  不把`tx_runtime` header/library依赖放进compiler或通用runtime library。
- TX device publication必须用当前pinned LLVM installation内的`clang++`消费compiler打印的opaque-pointer LLVM IR；
  不能fallback到ambient旧clang。CMake把resolved absolute Python和LLVM clang++路径写入`wafer-compile`，device link再用
  vendored Xuantie GCC完成CRT/link。CRT和static support用function/data sections、hidden visibility、archive symbol hiding和
  section GC，只发布当前entry closure；final undefined symbols继续经过versioned loader ABI exact allowlist。
- TX board环境分层先证明driver/public runtime版本，再把宿主boot-source Kcore payload与module toolchain对应firmware/ELF闭合，
  并核对driver cached runtime version/status；没有device-RAM dump时不得声称运行中payload byte identity。
  对待执行module的全部UND逐项检查Kcore `__rtmsym_*` export，缺项时在静态qualification阶段排除。随后用一次性
  public-runtime caller记录allocation、H2D、load、resolve、launch、completion、D2H和cleanup每个返回值，并对非平凡输入做
  完整CPU comparison；device异步错误可能只在后续D2H出现，不能只看launch/synchronize或vendor程序退出码。预置sample和fresh
  tutorial build不自动成为known-good。legacy host runtime的`ldd -r`失败只排除该legacy路径，不能反推当前public `libhpgr`
  主线不兼容。
- Board provider只把低层query/device error、deadline或其它无法信任terminal/context的结果标成sticky poisoned；普通
  `NOT_READY`仍是pending。executor只在provider仍明确usable时逆序cleanup；poisoned后
  不再调用copyback、unload、free或provider error-string，CLI flush typed stage/context诊断后直接结束一次性进程。reset、power、
  PCI/driver恢复和retry都不属于invocation cleanup；需要时由用户在进程外显式执行并重新从qualification开始。board成功或失败
  都在显式module/allocation lifecycle和output/diagnostic flush后用`std::_Exit`结束，不触发未经资格化的vendor DSO finalizer。
  `dlopen`成功后的symbol/readback失败也必须leak handle并走相同`std::_Exit`，不能用RAII `dlclose`触发未知fini链。CPU comparator在
  trusted completion、D2H和显式cleanup之后运行；其数值失败不改变provider disposition。只有明确poison、untrusted terminal或
  执行后只读resource/inventory未回到基线时才进入恢复判定，clean numeric mismatch不要求重启。
- 当前V5.6 kernel提交是异步消费host argument bytes；provider必须深拷贝每个argument block并至少持有到成功release，不能把
  invocation-local `SmallVector`/stack地址交给vendor queue。所有kernel ABI在首个TX effect前还必须检查packet不超过`0x7dc`
  （2012）字节。model type-6同步调用没有内部deadline；真实model CTest必须运行在一次性子进程外层deadline内，超时后kill/wait并
  停止后续gate，不在未知context上调用cleanup、reset或power。
- Board allocation admission使用当前free bytes而非total bytes，并对本次全部resource allocation做checked aggregate和显式reserve；
  单个resource小于总容量不能替代aggregate gate。`getContextState()`只能读进程内cached disposition，不得调用TX runtime/device。
- TX `transport:none`多rank multi-launch使用一次owner-backed invocation：完整rank domain先做all-and-only pure preflight，再统一完成
  aggregate admission、resource/module ownership、独立custom stream submit、`txStreamQuery` host deadline、原子copyback和逆序cleanup。
  stream只是command queue，不是tile selector；current V5.6中每条grid1 launch的唯一block落在tile0，所以16份exact结果只能证明
  logical ABI和lifecycle。live workflow先跑rank-one、复核只读卡状态，再跑该smoke；首错即停。
- TX真实多tile Add分两条验证。kernel qualification用一次`txLaunchKernel(grid.x=16, block=1)`和rank-major pointer table，entry按
  `__get_pid(0)`选择不重叠slice；model发射用`txLoadGraph` type-6加载16份tile-specific module，再以type-7 BPM调用一次
  `txLaunchModel`，各tile收到同一56-byte head并读取其后的72-byte input/output/param dyninfo。两条都必须结合静态调度合同、
  full-good logical inventory、逐字节不同于expected的output canary、16个互斥slice和完整CPU exact形成logical tile 0..15执行依据；
  不声明physical tile坐标。BPM是当前V5.6 exact-build恢复ABI，public API无builder；repository-owned typed builder、
  nested allocation lifetime、module identity、artifact export/readback和fake lifecycle必须先于真实板测闭合。
  nested allocation/module identity和fake gate闭合前不得触卡。Direct DTE始终使用独立typed/static/fake/no-card/hardware gate，
  PG selection、inventory、reset或power都不能补placement/readiness/completion语义。
- 完整硬件校准统一用`python3 tools/run_hardware_calibration.py --build-dir <board-build> --list`
  先审计该board-capable配置内注册的`board;hardware` CTest inventory，再以
  `WAFER_EXECUTE_HARDWARE_TESTS=1 python3 tools/run_hardware_calibration.py --build-dir <board-build> --execute`
  执行。入口只做一次增量构建，随后单进程串行、首个失败或skip即停，并保存逐项log、JUnit和增量
  `session.json`；它不retry、reset或power，也不把case构建/no-card通过记作板端证据。新增或删除board CTest时必须同步
  runner manifest和inventory测试。默认full-card Add槽位使用production-artifact profile gate，旧kernel-grid Add只作
  显式smoke，不能在默认批次中重复运行；profile归档必须经`runs/current`和`run_id`身份校验后完整保留
  `evidence.json`、`analysis.json`、`index.html`三文件组。
- TX81 C-Intrinsic Direct DTE不能只调用`direct_sync_init`。vendor生成entry先执行`init_tile_id(logic_id, row_length)`；
  `direct_sync_post`从SPM `0x2f0458`读取row length来计算peer SPM base。当前full-16、offset=0时可用
  `init_tile_id(__get_pid(0), 4)`，其中4来自单卡4×4 target topology；subset cluster的pid不是通用logical tile id，必须显式消费offset映射。
- Direct DTE accepted binding中的remote receiver地址是peer planned SPM offset，不是firmware `DirectDTESendInfo.dst_addr`。
  TX81 CRT sender必须用`get_tile_spm_addr_base(peer, 4, 4) + offset`形成远端映射地址；sender source和receiver FSM继续使用
  本地raw SPM offset，不能统一套`get_spm_memory_mapping`。device-link base loader allowlist和fresh module/Kcore export closure必须包含
  `get_tile_spm_addr_base`，并用CRT/module反汇编同时检查remote add与receiver未映射。
- TX81 Kcore对host分配DDR的scalar store不是自动coherent。firmware进入dynamic module前只invalidate参数表，entry返回不clean
  module写入；Direct DTE status-v2的逻辑`u32`位offset 0，resource以64-byte storage/alignment独占cache line，
  CRT统一写入并对该line执行C908 `dcache.cipa/civa` clean/invalidate序列。
  CRT以`-mcpu=c908`编译，通用kernel LLVM ISA配置保持独立；工具测试要反汇编检查cache opcode，不能只查C源码或`volatile`。
- ready-order处理`wafer.instr.dte_wait`时必须沿token派生in-flight buffer effect：send wait延续source read，recv wait形成destination
  completion write，并沿ViewLike追到storage base。focused unit同时覆盖recv-before-consumer与send-before-overwrite；production
  qualification还要读回最终ELF顺序。反汇编前核对artifact的mtime、shape和digest，避免把并发重编译留下的旧产物当成当前结果。
- Direct DTE的可重叠window必须绑定target ABI revision。V3 lowering对每个sender发射
  `send_prepare -> send_issue_v3`，issue内部只做peer-ready、attach和async submission，exact wait只做
  completion/release；profiler分别给issue与wait建立typed site。V1/V2没有issue symbol，wait内auto-issue只作兼容，
  不能伪造为独立window。包含V3 DTE的手写/compile-only module必须让同一module中的ordinary NCC call也使用V3
  worker参数，禁止混合旧prototype。
- 相同code object可能由vendor runtime复用为同一个module handle。adapter按`handle -> digest + logical owner count`维护所有权：
  同handle同digest只在最后一个logical owner释放时真正unload；同handle不同digest是provider contract violation并立即quarantine。
  query deadline要在每次低层query前后检查，不能只在完整rank轮询结束后检查，否则慢调用会使整体deadline失真。
- `ctest`通过不等于关键program/E2E gate被执行。默认`check-wafer-lit`只运行Dialect、Frontend、
  Pipelines、Spmd和Transforms中的直接IR合同；涉及完整program/source-to-package主链路时，必须点名运行
  对应Tools lit。需要审计unsupported清单时，使用`build/wafer-dev/CMakeCache.txt`中配置的
  `LLVM_EXTERNAL_LIT`执行相应目录或文件的`-sv --show-unsupported`，不要硬编码开发机Python路径。
  structured-program production gate必须执行统一`wafer-compile`到verified structured tensor program；
  `wafer-compile-spmd-partition.test`和`wafer-compile-structured-tensor-program.test`必须在配置了
  `WAFER_XLA_SPMD_PARTITIONER_HELPER`后实际执行，不能只用`ctest passed`宣称完成。
- focused Tools lit可能同时解析`wafer-compile`、`wafer-compile-test`和`wafer-run`。只增量构建其中一个target会让
  同一configured test tree混入旧CLI binary，并以`unknown argument/option`形成假回归；运行前应构建该test的全部直接
  tool依赖，出现CLI参数不识别时先核对各binary mtime和target，而不是修改测试预期。
- 默认`check-wafer-unit`不重放NoC complete-tuple生成和whole-variant production search等分钟级integration
  suites；这些suite仍由同一gtest executable承载，通过`check-wafer-compiler-integration`点名运行。局部改动先跑
  对应suite/filter，任务的完整source/package/no-card证明由该任务自己的直接case承担，不能让默认unit代签。
- model-scale frontier/search workload不进入聚合integration target；局部profitability、numeric和bounded-plan
  合同由小型analysis/planner unit覆盖，真实大输入的编译时间、winner、package和no-card必须由拥有该边界的任务
  直接case证明。不要让一个数分钟的旧gtest同时代签策略正确性、规模上界和source-to-package资格。
- rank-local scheduling若只从typed collective读取`logicalRank`，无collective的verified tensor program可把
  rank frontier视为一个generation class，但不能直接复制某个已截断的分片结果。应按稳定的
  `(source, recipe)`请求组分片，保持partition suppression作用域，按原request ordinal归并并重放全局admission；
  回归逐项比较分片与未分片frontier的metadata及完整module。跨context先用metadata计算bounded whole-variant
  attempt plan，只用MLIR bytecode传输会被消费的module；同一个rank-invariant encoded module可用只读共享存储
  映射到所有rank slot。编译统计至少覆盖generation class/shard/worker、frontier/attempt/late gate/clone/
  lowering/capture计数、各阶段wall和process peak RSS，且只能作invocation-local观测，不能进入selection或artifact。
- optional dependency收口必须保留两个独立build：full-feature配置显式启用StableHLO/Shardy、source-built
  PyTorch/XLA、pinned-XLA helper、numeric、oneDNN和SystemC，要求对应required tests不再因dependency
  unavailable而unsupported；feature-off配置显式关闭这些feature并验证预期unsupported清单及core binary link closure。
  两边都用各自cache中的lit执行`--show-unsupported`并记录完整名单；总数相同或CTest通过不能替代逐项审计。
- 不要并发运行两个会写同一个lit output tree的验证命令，例如同时跑`ctest --test-dir
  build/wafer-dev`和configured lit的`... build/wafer-dev/test`。部分`test/Tools`用固定
  `%t` output 路径，两个 lit 实例会互相清理目录，导致假失败；需要顺序跑。
- Runtime adapter 测试分层：package metadata/exporter 这类 compiler artifact golden 用 lit；no-card
  adapter contract、`fake-tx` test backend call sequence 和 runtime library discovery diagnostics 用 Python
  unittest / ctest；真实板端 launch/completion/error propagation 必须 gated 到有卡环境，不能塞进默认 lit。
- C++ `wafer-run`是verified package的唯一runtime入口：默认`--no-card`只构造typed
  binding/module/launch/completion plan；board feature开启后`--board`按`ResourceId`接收all-and-only raw buffers并调用同一
  preflight后的provider lifecycle。两种模式都验证ABI slot与resource一一对应，并在任何allocation/load side effect前拒绝
  unsupported completion/runtime mode；不能从自由字符串、resource name、文件名或instruction文本恢复语义。
- 16-rank只说明logical rank domain，不等于cluster launch。无跨rank transport的M-sharded replicated-operand
  program仍使用grid/main合同；只有accepted artifact需要cluster prepare/transport时才发布cluster form。board
  case必须显式声明并核对expected launch contract，不能用`rank_count > 1`恢复form或自动追加Direct-DTE
  no-card参数。profile资格应另外证明ordinary/profile production package递归bytes一致，并让同一个profile package
  完整通过Primary→Count→Trace。
- Shardy 不用 standalone Bazel workspace 作为 Wafer dependency 编译验证；`WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON`
  会通过 `cmake/third_party/WaferShardyCMake.cmake` 编译 `wafer-shardy-cmake-gate` / `shardy-sdy-opt`，
  复用同一套固定版本 LLVM/MLIR 和 embedded StableHLO。
- `WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON` 时，`wafer-compile`、`wafer-opt` 和
  `wafer-compile-stablehlo` 会注册 SDY dialect；新增 SDY program gate 要用 `REQUIRES: shardy`，
  避免关闭 Shardy 时让后端 textual tests 硬依赖 `sdy`。
- frontend program verifier 入口是
  `wafer-compile-stablehlo --verify-frontend-program <mlir>`；PyTorch/XLA capture 主链路用
  `wafer-compile-stablehlo --verify-stablehlo-program <program-dir>` 校验 `functions/forward.mlir`、
  `functions/forward.meta` 和 `data/<parameter>`，不要再为同一关系生成 Wafer 私有伴随 JSON。
- production SPMD由pinned XLA helper内部完成Shardy/XLA propagation；`wafer-compile`不能先把
  `sdy.constant`、`sdy.reshard`或其它SDY中间op写给只接StableHLO的helper。graph已有用户`mhlo.sharding`
  时由helper消费；无用户seed时当前采用replicated correctness基线。基于execution mesh自动补split seed的
  named pipeline只作IR-local调试，等有完整SDY→StableHLO bridge后才能进入production。单卡execution rank
  没有默认值，用户必须显式选择1或16；target profile同样没有默认值。
- default-sharding/SPMD chain不能用手写`sdy.sharding`、`wafer.spmd.*` attr、私有JSON或名字约定冒充partitioned
  program。正确主链是：frontend Python 只通过 `torch_xla.distributed.spmd.mark_sharding`
  标记 4096 matmul 图并导出带 `mhlo.sharding` 的 PyTorch/XLA StableHLO program directory；随后由
  `wafer-compile --input-program-dir ... --output-program-dir ... --execution-ranks={1|16} --target-profile=wafer-tx81-single-card-kernel-v1`
  在 Wafer compiler
  层接管 target/mesh，并由pinned helper内部完成Shardy/XLA SPMD partition，再执行local normalization
  和structured tensor program legality。当前typed compiler driver的Q15输出是重新读取并验证过的
  structured tensor program directory；不公开
  program stage selector 或 stop-stage。旧的私有 sharding attr
  emitter、sidecar JSON、单独旧 SPMD verify flag 和 Python post-SPMD
  路线已移除；不要恢复只生成私有 attrs/sidecar、只跑 SDY propagation 冒充完成，或把 Python
  test helper 写成 SPMD / 用户编译入口。
- default-sharding当前测试program入口：
  `test/Tools/Inputs/wafer_pytorch_xla_capture.py --emit-reference-program` 默认生成 4096 reference
  program directory；需要把真实 PyTorch/XLA export program 接到本地 compile gate 时，可以用 `--size <n>` 生成
  小尺寸同构图，避免让 single-tile bring-up 被 4096 工作集容量卡住。
  `test/Tools/Inputs/wafer_pytorch_xla_capture.py --emit-sharded-program --sharding-strategy=<name>` 生成
  pre-partition mark program。post-SPMD partitioned program只能由Wafer-owned SPMD
  partition stage 产生。
- pinned-XLA SPMD partition helper构建入口是`tools/build_xla_spmd_partitioner_helper.py`；它在
  `build/xla-spmd-helper/workspace` 生成围绕 `third_party/xla` 的 Bazel overlay，默认用 clang 构建
  `//xla/wafer_tools:wafer_xla_spmd_partitioner`，产物复制到
  `build/xla-spmd-helper/wafer_xla_spmd_partitioner`。受管环境显式运行
  `tools/build_xla_spmd_partitioner_helper.py --bazel third_party/tools/bazel`，避免ambient Bazel版本进入结果。
  本地把 helper 接进 compiler build / lit：
  `cmake -S . -B build/wafer-dev -DWAFER_XLA_SPMD_PARTITIONER_HELPER=$PWD/build/xla-spmd-helper/wafer_xla_spmd_partitioner`。
  production `wafer-compile` 不接收 helper 路径；driver 从 build-time
  `WAFER_XLA_SPMD_PARTITIONER_HELPER` 解析 helper。`wafer-opt` 只处理显式 MLIR 的IR-local debug/test，
  不拥有 program-directory orchestration。真实SPMD partition和structured-program driver gate位于`test/Tools`，
  必须由其owner点名运行；没有配置helper时通过`REQUIRES: xla-spmd-helper`显示为unsupported。只有启用unit tests的build可用显式
  `WAFER_TEST_XLA_SPMD_PARTITIONER_HELPER`做failure-injection override；production binary忽略该环境变量，
  不能把ambient runtime environment变成helper选择协议。
- PyTorch/XLA StableHLO program directory 的 `data/<parameter>` 由 upstream exporter 用 `np.save` 写入，因此
  partition helper需要解析`.npy` header才能切片输入参数；其输出的rank-local shard payload沿用
  NPY stream，路径为 `parameter_shards/<parameter>/rank_XXXXX.npy`。形状和 dtype 由 NPY header 与
  `forward.parameter_shards.json` 共同校验；不要把 NumPy 文件格式升级成 Wafer package/runtime ABI。
- `forward.parameter_shards.json` schema v3 的 parameter binding 必须显式写
  `distribution = replicated | partitioned`。partitioned slices 做无重叠精确覆盖检查；replicated slices
  必须覆盖完整 global tensor、replica-id domain 完整且 NPY byte-identical。当前 helper 遇到 partial
  replication 直接失败；不要用重复 offset 或 `replica_id = 0` 暗示复制关系。
- PyTorch/XLA transformer / RoPE 这类真实图会把 scalar 或 tensor captured constants 放进
  StableHLO function arguments，并在 metadata 中标成 `input_locations` 的 `type_ = "constant"`、
  payload `constants/<position>`。Wafer frontend verifier 要校验这些 NPY payload；pinned-XLA
  helper 输出 post-SPMD program 时也必须复制 constants 目录，否则统一`wafer-compile` structured-program
  verification会在program-directory边界正确拒绝。
- Wafer-owned Shardy / SPMD 源码放在 `lib/Wafer/Transforms/SPMD/`。只依赖 MLIR / StableHLO /
  Shardy CMake target 的 pass 编进 `WaferTransforms`；需要直接依赖 XLA HLO service /
  `spmd_partitioner` / generated proto / TSL 的入口也放在同一 Wafer 源码目录，但通过
  `tools/build_xla_spmd_partitioner_helper.py` symlink 到 pinned XLA Bazel overlay 编译。不要把这类
  pipeline stage 源码放进 `tools/` 或 `third_party/xla`。
- `test/Spmd`目前只覆盖default input seed和SDY/Shardy program parse/verify，不覆盖
  XLA SPMD partitioner，也不输出 rank-local StableHLO。`test/Frontend` 覆盖 StableHLO/Linalg local
  compute normalization；softmax、RMSNorm、LayerNorm 输入是 fine-grained StableHLO staged graph
  （reduce、broadcast、elementwise、shape ops），不是 `stablehlo.softmax` / `stablehlo.norm`
  或 Wafer 私有 high-level op。`check-wafer` 的大量 lit case 主要来自 Dialect/Transforms/Frontend/
  Pipelines/Integration/Tools，不代表旧 Python post-SPMD helper 仍存在。
- post-SPMD collective 先进入 Wafer LinalgExt-style tensor collective handoff，和 `linalg` 一起进入
  structured scheduling/tiling；`wafer.tile.*` communication只能在`wafer.tile.region` / SPM storage values /
  endpoint resource facts明确后
  materialize。StableHLO collective 直降 `wafer.tile.*` communication 且靠 `unrealized_conversion_cast` 桥 tensor
  和 storage 的 pass/test 已移除；不要在structured source输入侧恢复这种入口。
- linalg extension collective handoff的主线验证入口是同一个`wafer-compile` driver：它必须从真实
  PyTorch/XLA sharded program 产出含 `wafer.linalg_ext.collective.*` 的 `functions/forward.mlir`，并保留
  `forward.parameter_shards.json` 与 rank-local NPY payload。局部 `test/Frontend` fixture 可以覆盖
  `all_reduce` / `reduce_scatter` / `all_to_all` / `collective_permute`，但不能替代这个 program
  handoff gate。
- local compute normalization主线不要恢复本地`wafer-lower-stablehlo-{dot,elementwise,reduce,shape}`或
  `wafer-normalize-constants` 窄子集；这些旧 pass 入口已经删除。`wafer-lower-stablehlo-to-linalg`
  的主线 body 先运行 Wafer collective handoff，再调用当前 StableHLO pin 的官方
  `stablehlo-legalize-to-linalg`；Wafer collective handoff 不能证明时要 `signalPassFailure`，
  不能静默把raw StableHLO留给structured scheduler。
- XLA SPMD partitioner 输出的 rank/mask helper 可能以 residual `stablehlo.partition_id` /
  `stablehlo.replica_id`、静态 tensor view 常量链和 all-constant integer `linalg.generic`
  形式出现。local compute cleanup的职责是在official StableHLO-to-Linalg前后把这类可静态证明的常量
  折掉，确保structured scheduler输入没有raw StableHLO residual；不要把这扩成运行时 shape 计算或 Wafer 私有
  compute lowering。
- `wafer.linalg_ext.collective.*`不是只靠op名字或pass switch的skeleton；五类collective复用
  `DestinationStyleOpInterface`和MLIR `TilingInterface`表达DPS/iteration/tile semantics。不要再用
  Wafer-specific tiling interface重新枚举inputs/outs/results；Wafer-specific collective interface若保留，只暴露
  标准接口没有的rank-group/channel/combiner事实且不返回整份info snapshot。slot-crossing或动态不可证明的
  collective-axis tile应由`TilingInterface`返回failure，等待structured scheduler拆slot-aligned tile或
  tile communication materialization。
- StableHLO `replica_groups` 有多个 row 时不要压成一个 `rank_group`。`wafer.linalg_ext.collective.*`
  现在用互斥的 `rank_group` / `rank_groups` 表达单组或多组 logical ranks；rank-specialized
  tile-region materialization 按当前 logical rank 选择所在 row。这里仍然只保存 logical rank，不保存
  physical endpoint 或 communication algorithm。
- tiling demand和layout realizability只能是从current structured tensor IR重算、随mutation失效的局部analysis；
  不要建立复制DPS/indexing/layout facts的长期Demand/Plan结构，更不能把tile demand、layout assignment或
  materialization cut写成持久IR attr。analysis步骤不生成`wafer.tile.region`；per-rank bundle integrated gate
  要从typed driver产生并重新验证的structured tensor program输入重放这些analysis。
- 依赖一致性检查入口是 `tools/check_deps.py`；默认检查固定版本、importer registration hook、
  public source submodule checkout HEAD、importer Python package pin 和 core/frontend/runtime/test
  tool dependency layering。
- Device-code local link gate 默认不再读取外部 machine-local TX8 deps root。TX8 headers、
  libs、sysroot 和 Xuantie `riscv64-unknown-elf-gcc` 来自 repo-vendored
  `third_party/tx8_deps`；Wafer-owned `wafer_tx81_*`边界由target LLVM编号设计、repo-local public CRT header
  和source/object共同约束；不能从旧`libvr.a`archive或TX81`__*`
  symbol 反推出 compiler target CRT closure。当前 vendored Xuantie toolchain 的可用 64-bit
  double-float multilib 是 `rv64imafdc/lp64d`。
  compiler-generated RISC-V object 在进入 Xuantie GNU ld 2.35 前需要用 vendored
  `riscv64-unknown-elf-objcopy -R .riscv.attributes` 做 metadata normalization；repo-local
  Wafer CRT source/object 和 target object 一起进入 link gate。设备链接不再默认编译或链接
  capture shim；LLVM object 之外的 target CRT symbol 必须来自 repo-local Wafer CRT source/object
  或明确合法的 runtime/loader 外部依赖。`tools/wafer_device_link.py` 能执行
  `.ll -> .o -> kcore .so` 不等于主线 gate 完成；required-symbol 检查必须拒绝未解释的
  `wafer_tx81_*` undefined symbol。
- `tools/check_target_crt_symbols.py`从稳定Target层的typed target-call registry和`WaferAttrs.td` enum
  spelling推导111项production surface，并确认target lowering只消费该registry，再与CRT header/source和编译对象
  `nm`做exact closure；
  `check_target_crt_conformance.py`从instruction verifier、target address lowering和CRT实现交叉证明关系。
  两者都不能解析`tasks/`或supporting Markdown marker作为expected ABI事实源。
- Wafer RDMA/WDMA Instr、TargetCall和public CRT ABI统一使用byte-level inner/stride；TX81
  `ConfigStrideIteration`使用logical element count/stride，BOOL使用logical bit count/stride。转换只发生在repo CRT到
  vendor wrapper的边界，并同时覆盖inner与三层stride；GatherScatter继续使用byte descriptor，不能共用该转换。
- owner-backed target LLVM在host JIT前使用闭集legality：只允许当前producer需要的integer metadata/control-flow和
  direct registered calls，禁止generic intrinsic、global、pointer memory access、inline asm和间接/未知call；不能只拒绝
  target intrinsic后就native retarget。跨rank可yield frontend必须先materialize全部rank，再由每rank process各调用一次
  entry；rank显式记录not-started/running/terminal，sink以fallible prepare-commit和infallible commit分离验证与发布。
- target LLVM call emission 输出给 `mlir-translate --mlir-to-llvmir` 前不能残留任何 Wafer op。target
  topology / execution mesh 在 target LLVM call emission 前是 fact source；lowering 完成后这些 metadata
  应被消费或剥离，并在发现其它 `wafer.*` op 残留时报错。pipeline 测试应把 target/mesh 放进输入，
  防止只检查函数体而漏掉 module-level metadata。
- Wafer IR 文件组织检查入口是 `tools/check_ir_organization.py --root .`；它检查 `WaferOps.td` 只作为
  TableGen 聚合入口、ODS/verifier/test 按 `Tensor`、`Tile`、`Resource`、`Instr`、`Runtime`
  和 `Common` IR 层组织，并检查 `Conversion` 不再被 `WaferTransforms` 直接 owning。
- Wafer transform pass API 的主入口是 `include/Wafer/Transforms/Passes.td` 生成的
  `WaferPasses.h.inc`；新增非可选 pass 应先在 `Passes.td` 声明 argument、summary 和
  dependent dialects，再让实现继承 generated base。pass即使只间接创建某个dialect的type（例如collective
  materialization创建`async.token`）也必须声明该dialect；组合pipeline中的后置pass可能偶然预加载dialect，因此还要有
  只运行该named pipeline的回归。只读 dump pass 结束前要
  `markAllAnalysesPreserved()`。
- Region op 的 verifier 要按 MLIR 阶段拆：boundary / operand / result invariant 放普通
  `verify()`，body argument、terminator 和 region body legality 放 `verifyRegions()`。父 region op
  只解释自己 body 的直接 op，不递归解释子 op 内部 region。
- 长期 op/type 协议优先放 ODS type constraints 和 verifier，不靠手写字符串诊断补类型合法性；
  `!wafer.storage`、ranked tensor boundary 和 async token 这类类型要在 ODS 里约束，并在公开
  dialect header / CMake link 中显式包含对应 MLIR type 依赖。
- `wafer-opt` 需要显式注册要暴露的 MLIR pass families；如果显式IR debug/test依赖 canonicalizer/CSE
  这类标准 pass，注册 `mlir::registerTransformsPasses()` 并链接 `MLIRTransforms`，不要假设
  `MlirOptMain` 会自动注册。
- upstream pass/library的linked、registered、debug-replayable、production-consumed和qualified是不同状态。候选型优化还必须
  区分“产生passing candidate”“进入共同selection”和“成为默认production committed winner”。只有named production pipeline
  或其shared candidate utility真实调用、至少一个通用case发生actual-IR改写、通过等价IR和下游exact gate、进入共同frontier，
  并在对应case由final-IR policy选中且原子提交，才能声明采用；required closure没有独立choice时，其mutation效果必须保留在
  committed winner。generic canonicalizer不能承担required normalization correctness，测试计数也不能代替winner IR/readback。
- 历史 stage-connection 测试和 `tools/check_stage_connection_tests.py` 已删除；后续tile-dataflow/tile/storage
  连接必须由真实frontend/SPMD program chain和当前编号合同恢复，不能重建手写fixture链来冒充主线。
- 任务支持范围按硬件能力、runtime/ABI 证据和当前 IR contract 判断，不能按“当前下游 pass 尚未
  实现”反向裁剪上游语义。若 frontend/SPMD/planner 产出合法且硬件可表达的事实，而 IR/lowering
  还没覆盖，应补 IR contract、verifier 或下游恢复任务；不能把实现缺口写成上游不支持。
- `wafer.target.topology` 的稳定 V0 表示是规则拓扑加例外：`card_grid`、`card_interconnect`、
  `tile_grid` 和 `unavailable_tiles`。不要恢复成展开的 `tile_ids` / `tile_coords` / `links` graph，
  不要把 bad/PG-disabled 分成多套不可用集合，也不要通过 `id_encoding` 把 endpoint 编码规则变成
  IR 合同。单卡 tile 邻接和跨卡 C2C 邻接从规则 grid / mesh-or-torus kind 派生。公开 pass /
  pipeline 选项用 `card-y`、`card-x`、`tile-y`、`tile-x` 表达 grid 规模，不恢复冗余的
  `*-count` spelling。
- `Cx/NCx` layout 规则容易误用，必须按硬件文档的 `get_CxC0` /
  `common_tensor_info_generate_i64` 口径理解：对齐的是 logical last dimension `C`，不是 flatten
  后的任意元素流；INT8/UINT8 full block 是 128，其它 dtype full block 是 64；tail 小于等于半块时
  保留为按 `4/8/16/32/64` 级别对齐的 `C0`，大于半块时 fold 到下一 full block；C alignment 后还要
  计入 256B bank alignment。`Cx` 通常用于 2D，`NCx` 用于 rank > 2，但 `NCx` 的 `N` 只是历史外层
  slice 命名，不等于 semantic batch。full-block 物理顺序是 channel-block major：`Cx` 是
  `[CxBlock][outer][lane]`，`NCx` 是 `[N][CxBlock][HW][lane]`；`aligned_C` 只用于 footprint /
  batch size，不是 logical row stride。不要用 `ceil(C/64)*64`、layout marker 名字或
  `physicalBytes` 单点事实替代完整 physical mapping。
- 判断 `wafer.tile.reshape` 是否需要 instruction movement 时，先保留 StableHLO/tensor reshape
  的 logical 语义：source/result 的 canonical linear element number 对齐，result multi-index
  按新 shape 解释。movement 触发条件是同一 linear element 在 source/result 中的 physical byte
  offset 映射变化，或目标 physical footprint/descriptor 需要 materialized buffer；不是“看见
  reshape”或“看见 cx/ncx”。compact `tensor/ntensor` 可用标准 memref view；`Cx/NCx` reshape
  要先用统一 physical layout calculator 比较 source/result mapping，只有排布变化才发
  `wafer.instr.gather_scatter`。如果统一 helper 还不能表达真实 `C0` tail/fold 和 bank padding，
  先补 helper，不要在 lowering 里临时重写一份局部 layout 解释。
- movement descriptor lowering（`extract_slice/insert_slice/broadcast/transpose`）要从 op
  自身的 logical `IndexRelation` 出发，与两端encoding组合后按affine/reshape/layout piece直接构造
  TDMA三层source/dest stride/iteration descriptor；逐logical element计算offset只保留为独立慢oracle，
  production不能用enumeration/coalesce作为fallback。`insert_slice` 不是只写 slice：它返回 updated dest
  buffer，所以 lowering 必须先把旧
  dest payload copy 到新 result，再把 source slice overlay 到 result。RDMA/WDMA lowering 要消费
  DDR 侧 `memref.subview` / strided memref layout：整块 compact DDR boundary 生成 contiguous
  descriptor，静态 strided tile view 生成三层 byte stride/iteration descriptor；动态 view、负
  stride、bit-packed element 或超过三层的 descriptor 不能靠名字/shape 猜测，必须 structured
  failure 或等上游补显式 boundary facts。candidate tile-view materialization 已覆盖当前 IR 中 explicit static boundary
  `tensor.extract_slice`、direct output `tensor.insert_slice` storeback，以及 candidate output
  tile offsets/sizes candidate evaluation lowering 的 DDR `memref.subview` producer；instruction lowering
  仍不能根据 whole-boundary shape 自己恢复 subview，closed-loop traversal / tile-shape search 归
  candidate-selection。
- SPM/DDR accepted offsets不属于layout本身。同一rank function内的non-nested sibling `wafer.tile.region`通过
  显式operand/result或受支持的view/select/SCF SSA edge传递SPM value，并由一个whole-function timeline/demand set
  联合packing；每条region exit仍独立证明pending set为空。nested/async/parallel scope、raw escape和无法解析的
  provenance失败；没有SPM arena/effect的closed scalar direct callee可穿过live resident，可能执行tile-region的
  callee、external/unresolved或indirect call在缺少arena/resource summary时fail closed。DDR `wafer.ddr.offset`始终是arena-relative fact，没有typed
  arena base binding时target不得把它当absolute address。physical size、alignment和bank span统一从shared
  geometry helper推导；runtime object、physical address和packet字段不得写回planning IR。
- reduction语义恢复不能只看yielded op class。使用`mlir::matchReduction`或等价结构匹配，证明单一combiner
  的operands精确连接reduced value与accumulator。未拆分source reduction保持原合同；candidate把一个reduction
  regroup成多个partial时，generic floating只要求exact single combiner，named floating matmul可按合法K范围切分，
  二者不要求额外fast-math标注。integer split仍只覆盖无overflow flag的modular add和signed min/max；
  unsigned min/max、overflow-qualified add及`maxnum/minnum`保持fail closed。Tree和Ring collective都接受支持的
  floating element type，但仍必须验证rank group、topology、chunk和completion。
- whole-op fast path必须证明整个payload可被删除：passthrough/concat/reduction以及named
  fill/matmul/batch_matmul都要检查exact SSA wiring、允许op集合和effect；只匹配yield、shape或op class会
  静默擦除side effect或改写数值语义。structured materializer/verifier应递归检查nested body dialect/type。
- destination-style tensor仍遵守functional SSA：fill写fresh result而不覆盖旧init；insert_slice在旧dest仍有
  observable use时构造fresh result并延后boundary store。只有旧dest其余use都被证明是unread DPS-init时才可
  direct tile store。`ins + outs` exact SSA必须唯一；不同SSA的physical no-alias由后续typed driver/ABI闭合。
- async completion按path、task identity和engine分别建模：generic async handle的root provenance与task identity分开，
  只有覆盖同一路径的terminal await完成task；local compute/movement的pending SPM effect只在显式
  completion-domain local drain处收口，same-worker NCC链内依赖仍由issue order+busytable落实；DTE send/recv
  只由matching token/wait收口，三者不能互相消费。zero-trip loop、分支join和loop-carried handle没有精确
  proof时拒绝，region/function terminal boundary不得隐式清空pending状态。
- target undefined-symbol gate使用代码拥有的exact allowlist，并检查全部undefined symbols，而不只检查
  `wafer_*`前缀；prefix/substring命中不能替代精确成员关系。allowlist通过只证明loader ABI surface，不证明
  packet、transport、completion或board正确性。
- all-rank target publication直接消费Q16 `ExecutableBundle`。每rank先把returned compiler-managed DDR root
  重定向到append-only output ABI slot，再从剩余DDR alloc重算workspace high-water/alignment并追加唯一i64 arena
  base argument；target pass只有收到显式argument index才把`wafer.ddr.offset`lower成`base + offset`。lowered entry
  必须是与typed `KernelABISlot[]`一一对应的fixed `void(i64...)`。LLVM IR、object/CRT和`.so`只写Q17 transaction
  staging；all-rank entry/RISC-V64 ELF/symbol/digest readback后才发布`TargetArtifactBundle`，不能用单文件atomic
  replace冒充多rank原子性。ABI slot和workspace的最低DDR alignment来自生成memory plan的同一target policy；
  workspace对该policy与全部alloc显式alignment计算checked least common multiple，非正数或int64溢出失败，
  不得用`max`冒充共同整除要求，也不得在artifact层另造更小默认值。
- ABI narrowing必须在compiler verifier/target preflight中完成：地址使用uint64，count/stride/iteration/enum/
  mask等普通字段适配uint32，`Data_Shape`维度适配底层uint16；CRT header/source和compiler call保持同一typed
  signature，不用宽形参加wrapper内部cast隐藏截断。
- candidate provenance必须穿过唯一accepted-artifact handoff：selector在transformation-local clone完成完整
  traversal和legality，rejected clone整体丢弃；debug replay消费同一accepted artifact，不重新运行另一套
  direct lowering。target conversion同样在module clone上运行，full success才替换source；多rank staging由
  外层transaction一次发布，单module成功不等于bundle原子性。
- task-level alternative按稳定passing ordinal消费时，收齐requested ordinal所需数量后即可停止扩展；parallel只允许已提交的
  固定有界batch完成，并按submit order消费。executor应由整个rank-frontier invocation持有，每个worker独占并复用
  `MLIRContext`及相同standalone task parse；不能每个batch重建线程/context，也不能让completion order进入selection。
- parallel worker通过完整candidate gate后，可用invocation-local MLIR文本把actual module移交owner context；owner只做parse、
  verifier和从导入IR fresh recost，不再重跑Tile→Instr→SPM→DDR。该文本是跨context ownership transfer，不是candidate
  identity、缓存、sidecar或可发布artifact。
- 完整output traversal使用compact `scf.for`并显式覆盖static tail；ordered reduction chunk/terminal op仍用
  checked ceil-div/product和4096个host materialization预算。该上限只防止编译时间/内存失控，不能写成硬件容量、
  IR/workload legality或16-tile topology限制。
- candidate hard cap必须分别覆盖source clone、task recipe、semantic scope policy、rank evaluation、rank frontier、
  whole best-first、coordinated correspondence和Pareto retain；parallel evaluation只能消费同一固定预算。当前scope只保留
  root-local closure、complete shared-input closure、terminal cut和conservative partition四类semantic policy；唯一reserved
  conservative spill拥有独立allowance，但执行相同late gates。
- capacity-directed seed必须区分“用于排队的保守inventory”和“可用于拒绝的required-live bound”。当前direct
  rank-2 matmul精确建模Tensor/Cx六个root；exact passthrough transpose+matmul为seed额外计入原始source Tensor，
  共七个root，而early reject只使用后续matmul phase必然同时存活的六个root。seed应先于已知高压full candidate
  消费hard-cap slot，但任何accepted candidate仍须走完整instruction/SPM/DDR gate。其它producer chain无法形成
  安全边界时返回unknown，不从名字或诊断字符串猜测。
- terminal collective的logical tile mapping和production traversal capability必须分开判断。当前只对单输入、单输出、
  shape-preserving all-reduce启用terminal tiled traversal：从collective result tile反向融合producer，并让capacity/geometry
  analysis只为M/N压力看穿到local Linalg root；reduction-range/split仍只读真实yielded Linalg root，不能把SPMD local K重新解释为
  compiler reduction split。其它collective继续`FullTraversalOnly`。all-reduce位于consumer之前时应成为独立task或fail closed，
  不能同时保留原始full collective与tiled clone。shared-input peers要么作为完整SSA-compatible closure整体加入，要么完全不加入；
  不枚举cost-ranked prefix。
- winner只读取final instruction IR、validated SPM/DDR placement、transport/completion和whole-card exact resource vector。
  ordinary Pareto阶段不再计算或保存旧coarse/saturating scalar time；complete Known dimensions上的Pareto先处理
  exact dominance，DDR下降但candidate引入或保留NoC依赖的跨资源tradeoff随后进入Q39 typed point/interval
  profitability gate。
  `EstimatedBenefit`只表示versioned static model清除margin，仍不能称作board-measured time收益。
- rank-local semantic generation和physical derivation是两个不同的correspondence维度：stable ordinal匹配source/recipe/scope，
  artifact kind匹配spill、spill-ready、resident或resident-ready。all-rank tuple必须同时匹配两者；只匹配ordinal会把不同
  physical program拼在一起，即使每个rank单独通过verifier和resource gate也可能破坏collective数值语义。
- rank worker结果跨context移交时，可先从原frontier slot metadata精确重放reserved baseline、bounded Cartesian和coordinated
  correspondence attempt sequence，再只parse完整correspondence attempt引用的owner modules。必须保留slot index/order、
  duplicate key、baseline marker和原attempt budget；worker仍print全部candidate module，malformed metadata保守parse全部并沿用
  原structural failure。这个plan只优化owner import，不能承担legality、cost、identity或winner判断。
- scheduler frontier之后的function-boundary bufferization和physical-memory replanning可能改变movement、offset和issue
  count。应逐alternative独立finalize，只过滤later gate失败的alternative，并从final instruction IR fresh recost；
  一个alternative失败不能拒绝仍有survivor的rank，只有finalized frontier为空才失败。
- structured lifetime path condition必须用sparse sorted decision set；固定宽度bitmask会把超过64个branch/loop的
  合法程序误判unsupported。decision id仍要checked分配，标识域或编译资源耗尽时结构化失败。
- clone/rewire/erase operation后，不得继续使用从旧IR缓存的`mlir::Value`、boundary或root列表；应从当前clone重新
  walk并构造消费集合，避免悬空Value参与provenance/lifetime或后续erase。
- 非splat tensor constant应materialize为typed DDR `memref.global`/`memref.get_global` root并保留payload provenance；
  splat才可用exact typed scalar在SPM local fill。不能用无owner的generic `to_memref`伪造constant storage。
- executable/resource handoff必须来自accepted IR和typed C++ bundle，不从raw instruction文本、文件名或参数名
  重建。当前没有executable dialect或独立resource-view协议；resource/entry/completion事实必须从accepted IR
  use-def、type、effect和offset直接校验后进入bundle，不能成为side table或package旁路。
- tensor-program-to-tile-region的buffer-level collective materialization必须从enclosing typed distributed
  instance/candidate entry取得partition和replica coordinates。局部pass选项只可用于明确的replay测试，
  production driver不得使用default rank 0或CLI option承载rank语义。这个边界仍只产生logical buffer
  schedule；每个send/recv的跨rank静态匹配身份必须由protocol phase和logical payload slice显式进入typed IR，
  structured loop/branch中的动态实例再由control-flow instance区分，不能靠op/scheduler顺序或名字恢复。
  endpoint/channel/FSM由post-memory transport acceptance在exact topology/mesh上处理，
  当前不引入pinned/relocatable runtime remapping。
- Direct DTE public helper的`direct_sync_wait`、`direct_fsm_monitor_receive`与`direct_dte_wait_done`都是无timeout参数的
  blocking wait；后者可报告本地DTE错误。compiler/CRT不能伪造device timeout能力：本地status由target ABI写回，
  timeout必须由launch watchdog观察，跨rank peer failure由runtime completion DAG合成。
- selected instruction handoff固定先由selector在tensor函数clone中完成完整traversal、instruction和candidate-local
  SPM/DDR planning；task commit后要从generation parent清除这些evaluation offset。随后每个rank frontier alternative复用同一份
  function-boundary OneShot Bufferization配置消除tensor signature与`bufferization.to_memref/to_tensor` wrapper，并在final rank
  evaluation clone上重跑whole-rank SPM planning与cost；DDR只在all-rank coordinator的disposable complete tuple上重跑并应用。
  target conversion只消费通过该finalization的accepted artifact，不重新执行task scheduling或tile/instruction
  materialization。所有typed materializer API都显式接收logical rank；IR-local replay也必须显式提供rank。
- structured materializer在DPS `outs`固定后会吸收task内部pure static support producers：`arith.constant`、
  `tensor.empty`、static `tensor.extract_slice` / `tensor.insert_slice`、`tensor.expand_shape` 和
  `tensor.collapse_shape`。这用于避免 XLA/HF 产生的 static `insert_slice` collective input 被错误
  作为外部DDR boundary；不能因此跨side-effect、memref/runtime或raw StableHLO op。
- floating `arith.select` 在 tile-region-to-instr 中不能继续生成 `wafer.instr.elementwise <select>`；
  现在会 lower 成 false-copy `gather_scatter` + `wafer.instr.bit2fp` + `wafer.instr.mask_move`。
  这条序列按 Triton/TX81 的 `bit2fp` / `mask_move` 证据对齐 target wrapper 粒度；integer/select
  泛化仍要等 target wrapper 证据补齐。
- top-level single-result `wafer.linalg_ext.collective.collective_permute` 现在直接 materialize 成
  `wafer.instr.dte_send` / `dte_recv` / `dte_wait`、local copy 或 zero-fill。
- top-level single-result `wafer.linalg_ext.collective.all_to_all` 的 V0 materialization 要求
  `split_count == rank_group.size()`，把每个 split slot 先 extract 成连续 SPM comm buffer，DTE 只收发
  连续 buffer，recv 后再 insert 到 concat result slot；当前没有 ring/blocked schedule selector、
  raw non-unicast DTE 或 cross-card route binding。
- communication schedule不是public pass option或IR attr。production candidate owner从同一tile-region parent
  独立clone并用typed conversion参数物化Direct/Ring all-gather、Direct/Ring reduce-scatter和Ring/Tree
  all-reduce；每个clone都要重新执行instruction、SPM/DDR、verifier与cost gate。IR-local replay可以显式传
  typed参数，但不能恢复用户selector。展开后只保留`wafer.instr.dte_*`/local compute body，不保存algorithm attr。
- execution topology的rank mapping与hop事实只由共享`ExecutionTopologyAnalysis`从current module唯一
  `wafer.target.topology`/`wafer.execution.mesh`重算。isolated task/candidate clone必须同时复制这两个typed fact op；
  不得回退到logical rank编号、target profile名或固定4x4算术。Ring有序cycle和Tree edge/root是rewrite-local
  C++值；whole-card cost从final send peer计算minimum-hop link-byte demand，不把shortest path冒充实际route或timing。
- tile-region-to-instr 的 V0 all-gather lowering 从 `wafer.tile.all_gather` 的 compact `tensor/ntensor`
  local/gather SPM buffer shape 推导唯一 gather axis。`ring` 先把 local chunk 写入本 rank slot，
  插入 `wafer.instr.local_fence` 后沿 ring forward slot view；`direct` 每个 phase 发送 local slot
  给semantic group-index cyclic round中的peer，同时从对应peer接收chunk到contiguous staging并复制到result slot；
  Direct不调用有界topology Ring搜索。
  全部received-slot copy后必须再有final local fence；DTE wait不完成后续movement engine。DTE peer仍是logical rank，
  SPM offset、physical endpoint、DTE id和packet field留给后续planning/ABI边界。
- tile-region-to-instr 的 all-reduce Ring是标准`P-1`轮chunked reduce-scatter加`P-1`轮all-gather，每条message
  只承载`B/P` chunk。reduce阶段wait后显式sum/max/min并fence；gather阶段必须先收进独立recv-buffer chunk，
  joint wait后再copy到accumulator slot并fence，不能让同时进行的send/recv共享一个allocation root。Tree在
  `rank_group`连续区间上做interval DP，求中序严格保持group次序的minimum-total-shortest-hop ordered binary
  tree；每node按left-subtree、local operand、right-subtree累计，再沿reverse tree broadcast。它不是MST加center，
  也不使用root 0、XOR/binomial或固定rank邻接。该ordered Tree可用于floating collective；current cyclic Ring
  reduction只对integer生成，直到production IR携带其leaf permutation的numeric permission。
- tile-region-to-instr 的 V0 reduce-scatter lowering 使用 full input + local slot result 表示：
  structured task materializer不预切当前rank slot，`wafer.tile.reduce_scatter`显式携带scatter `axis`，
  Direct lowering从full input派生per-target slot `memref.subview`并执行all-to-owner；Ring按topology-derived
  cycle执行`P-1`轮result-sized chunk转发与显式local reduction。两者都在wait后用
  `wafer.instr.elementwise`累计；下一轮DTE读取该partial前在NCC→Direct DTE boundary做local drain，
  pure same-worker NCC resident consumer则只保持dependency issue order，不因复用本身插wait。不能形成连续
  非零typed chunk时只拒绝Ring clone，保留Direct baseline。
- singleton logical all-gather/reduce-scatter/all-reduce在tensor-program到tile-region入口折叠为resident identity，
  早于channel、combiner、recv allocation和`wafer.tile.*`通信op；Tile communication IR仍只表示group size大于一
  的真实跨rank协议。
- 通用 compiler target 名称统一为 `wafer`，Wafer IR target attr 的唯一主线 spelling 是
  `#wafer.target<wafer>`。裸的 `tx8` / `tx81` 不能作为 dialect、pipeline、pass、fixture 或可推断字段的
  主线命名；硬件/依赖逆向事实和tasks/14 closed registry中的opaque canonical profile key例外。例如
  `wafer-tx81-single-card-kernel-v1` 只能整体解析为typed `TargetProfileId`，不得拆字符串恢复target、
  revision、ABI或numeric policy。
- 非小修主线任务动实现前必须先写清楚 pipeline contract：upstream artifact / IR、current stage
  responsibility、output artifact / IR、downstream consumer、user-level driver / named pipeline、
  explicit non-goals 和 completion gate。只说明某个 pass / tool / test 的局部功能不够；完成证明
  必须重放已完成上游 program chain，并证明当前 stage 输出会被下游边界直接消费。
- 主链路gate应由独立`wafer-compile` owner-aware program driver重放已完成上游链路，不在Integration
  里手动拼 pass 串。当前 frontend verifier 入口是
  `wafer-compile-stablehlo --verify-stablehlo-program`；production compile入口统一为
  `wafer-compile --input-program-dir ... --output-program-dir ... --execution-ranks={1|16} --target-profile=wafer-tx81-single-card-kernel-v1`。
  typed structured-program boundary从frontend admission推进到重新读取并验证过的structured tensor program directory；
  同一production transaction随后把frontend verifier返回的typed boundary/shard facts和structured module直接交给
  per-rank bundle boundary，不暴露stop-stage。
  `wafer-compile-stablehlo --propagate-stablehlo-sharding`、
  `wafer-compile-stablehlo --partition-stablehlo-program` 已删除，因为 Shardy/SPMD 不属于 frontend
  verifier tool；旧 C ABI compile 入口也已删除。`wafer-opt`和现有named MLIR pipelines只处理显式IR，
  用于IR-local debug/regression，不拥有program-directory I/O，也不构成用户可选stage。当前bundle boundary对
  rank-count 1/16实际创建all-and-only isolated clones，经selector、function bufferization、whole-rank SPM及whole-variant DDR和
  terminal legality后形成move-only `RankExecutable[]`/context-owning `ExecutableBundle`；rank-15 late failure仍在
  同一transaction内，因此不会先发布中间调度checkpoint。无DTE时transport contract为`None`；Q16.T已在完整rank
  domain的post-memory acceptance后形成`DirectDTE`；rank module分拆使sender无法本地重算remote receiver offset，
  因而该cross-rank accepted start必须进入typed binding，不能假设各rank allocation同址。target把async token降成
  CRT返回的opaque i64 event；recv issue先初始化FSM并post ready，send实际attach/send延迟到wait，避免所有rank
  在本地recv ready之前同时阻塞于sender wait。entry status只表达pending/success/local transport error，timeout由
  manifest声明的host watchdog负责，peer failure由runtime合成。
  旧显式 target CRT issue-op、ring collective、SPM/DDR debug path 和 single-tile
  materialization pass 链已删除；不要恢复成用户级 compile flow。当前HF/Llama-style真实program可重放到
  verified structured tensor program staging；HF compute coverage、Direct DTE target/status ABI、runtime、board execution和
  数值correctness仍是后续独立gate。
- ODS op 如果引入 `RecursiveMemoryEffects`、`ReturnLike` 等 interface trait，公开 dialect 头要
  include 对应 C++ interface header，`WaferIR` 也要显式 link 对应 MLIR interface target。
- ODS op 如果直接使用 MLIR `TilingInterface` 这类 upstream op interface，避免让 TableGen 在
  Wafer namespace 下生成未限定的 `SmallVector` / `OpBuilder` / `ArrayRef` 方法声明；可用 interface
  trait 加 `extraClassDeclaration` 写全限定 C++ 签名，或确保公开 dialect 头有明确且局部的别名。
- Dialect 增加 TypeDef 后，base dialect td 需要启用 `useDefaultTypePrinterParser = 1`，否则即使
  `addTypes` 已注册，文本 IR 仍会报 “provides no type parsing hook”。
- `add_mlir_library` 会生成静态库 target 和 `obj.<target>` object target；源文件需要的 compile
  definition 要加到 `obj.<target>`，只加到静态库 target 不会影响实际编译命令。
- 直接用`add_executable`建立但链接LLVM/MLIR库的测试程序也要调用`llvm_update_compile_flags`；否则项目与LLVM的
  RTTI/异常编译选项可能不一致，直到链接或使用`ErrorInfo`等跨库类型时才暴露。这个要求与是否使用
  `add_mlir_library`无关。
- MLIR pass 如果会创建其它 dialect 的 op，必须在 `getDependentDialects` 中显式声明对应 dialect；
  只在 driver registry 里注册还不保证 pass 运行时 context 已加载该 dialect。
- 不要恢复 `wafer-check-softmax-schedule`、`wafer-check-norm-schedule`、
  `wafer-check-linear-residual-schedule` 或 `wafer-check-mlp-schedule` 这类 case-specific
  transformer acceptance pass。StableHLO->Linalg 只证明 structured tensor lowering；softmax/norm/MLP
  的真实完成证明应来自通用structured scheduling、tile/materialization、resource verifier和下游消费。
- 不要恢复 `tools/wafer_package_metadata.py --emit-*` 这类fixed package emitter，也不要把
  `wafer-compile-stablehlo --emit-static-reference-program` 这类 synthetic program emitter 作为 importer
  或package主线。主线typed manifest必须只由accepted executable bundle和verified target modules构造，
  不能从raw IR、单个module、printer text或旁路resource view恢复。
- 近期package wire使用唯一C++ typed model的canonical JSON，不预设另一套schema/registry基础设施。Python只可作
  薄CLI或显式legacy converter；唯一semantic verifier在C++，package不复制instruction schedule。
- Kernel ABI近期由typed slot/resource双射和rank/module/entry digest表达；ELF note或跨进程descriptor等真实
  loader/cache consumer出现后再扩展，不能先建设global identity registry。
- 完整compiler driver不能伪装成纯`OpPassManager` named pipeline。`wafer-compile`拥有source、explicit rank
  clones、target staging和atomic publication；named MLIR pipeline只保留IR-local transform。
- package parsing/semantic verification、pure RuntimeSession preflight和provider execution是三层边界。no-card
  preflight不分配、不加载、不发命令；fake/board provider实际调用必须分别记录failure suppression和cleanup。
- 当前typed package入口是`Wafer/Runtime/PackageManifest.h`：compiler只从Q16 `ExecutableBundle`和Q17
  `TargetArtifactBundle`构造manifest，在私有staging内复制payload、核对digest、canonical serialize/parse readback、
  fsync后no-replace发布。`wafer-run --package-dir <root> --entry-id <id> --no-card`是唯一runtime inspection入口；
  Direct DTE package还必须显式声明兼容environment：`--direct-dte-status-abi wafer-direct-dte-status-v2
  --supports-host-watchdog`，缺失时应fail closed；这些选项只形成preflight facts，不代表provider执行。
  Python adapter只启动该二进制。排查package时先跑`PackageManifestTest.*`和`test/Runtime/wafer-run.test`，不要恢复
  已删除的Python schema/exporter或C++ `HostRuntime` acceptance。
- 纵向source corpus不要用`torch.empty()`、framework默认初始化或提交生成物固定输入。当前最小corpus spec在
  `test/Tools/Inputs/workloads/single-card-vertical-v1.json`：整数序列加二进制可精确表示的f32缩放生成
  input/parameter，独立NumPy实现生成完整CPU expected，再与同payload的framework CPU module按tolerance
  交叉检查；真实PyTorch/XLA export后反读parameter NPY逐元素核对，并用`forward.mlir`、canonical meta和
  typed payload构造canonical program digest做重复export证明。CPU-only入口是
  `wafer_pytorch_xla_capture.py --emit-cpu-reference`，真实admission入口是`--emit-workload-corpus
  --verify-corpus-reproducibility`；两者都只证明corpus/frontend admission，不证明compiler、runtime或board。
- target LLVM module必须与拥有它的`LLVMContext`一起作为move-only artifact跨stage传递；device linker和后续host
  model都直接消费同一份verified module，不能把module隐藏在print/link helper中，也不能为不同消费者重复lowering。
  owner成员声明顺序应确保module先于context析构；模块级schema metadata应在producer返回前typed readback验证。
- 自建MLIR context调用`translateModuleToLLVMIR`时，仅把Builtin/LLVM dialect加入registry仍不够；还必须为该registry
  注册Builtin和LLVM dialect translation interface。production driver与unit-test context都要遵守，否则测试会在
  LLVM IR translation边界失败，而不是在dialect parse/load阶段暴露。
- 多个独立执行consumer需要同一source invocation时，只共享typed compact tensor、accepted rank slice和已验证payload
  loading；不要共享CModel/board compute、numeric或transport实现。target model在共享装配之后，必须再按exact
  Kernel ABI slot的shape/dtype/layout编码target physical bytes，并从slot反向解码output。这样source输入只有一个事实源，
  CModel和board仍是独立实现。
- 需要证明“同一次lowering”时，用factory-only move-owned compilation product同时交付accepted executable和实际生成target
  artifacts的owner-backed LLVM bundle；不要让下游公开构造这个关系，也不要从已发布package或accepted IR重新lower。
- source-backed bulk qualification不能只绑定shape、seed或NPY路径。offline source-spec应嵌入并canonicalize exact target
  physical operand/destination-template bytes，runtime再对command、payload、environment和预期backend output exact-match。
  超过formal budget且无admission时稳定失败；formal fallback只能在checked budget内发生。
- `wafer-compile --target-model`只消费显式`--model-input`和固定source CPU `--model-expected`。已发布的
  F16/BF16/F32 finite output按case显式atol/rtol逐元素比较；整数、布尔和其它非浮点storage raw exact；TF32/F64等
  尚无source-output policy的浮点格式fail closed，NaN/Inf拒绝。比较失败仍保留已经原子发布的verified package供审计。
- target-call decoder closure不能只断言111项都能形成正确variant family。为每个descriptor生成ABI位置互异的sentinel，
  再逐字段比较typed payload中的地址、count、shape/stride、optional parameter、format和static kind；这样字段交换或漏消费
  才会失败。host native frontend的control value也要显式限制为integer/void，LLVM的pointer PHI/select/icmp本身合法，不能
  靠IR verifier替代frontend legality。
- bulk qualification中的`impl_info_str()`和resolved primitive descriptor digest只有进入不可伪造admission、并在每次实际
  primitive创建后与execution evidence比较，才算runtime exact enforcement。只把它们写入final JSON/readback object会留下
  “record看似闭合、执行未消费”的重复事实源；negative应分别篡改implementation和descriptor且保持canonical record可解析。
- 当前SystemC model入口是一driver进程一次initial-elaboration invocation；`sc_start()`运行到quiescent后返回，不声明同进程
  reset/repeat。需要证明source late-rank原子性时，只在test driver中注入terminal rank failure，仍重放同一package publication、
  target-call和SystemC链，并断言稳定stage/rank、无matched model result及已发布package保留。
- 正式受管依赖统一位于`third_party/<lane>`；`build/<configuration>`只保存consumer生成物和canonical snapshot。切换managed
  package root时，CMake的`find_package(... PATHS ... NO_DEFAULT_PATH)`仍可能优先复用已有`<Package>_DIR` cache，因此validator
  读出的config目录必须在`find_package`前以`CACHE ... FORCE`刷新。配置gate应预置一份valid-looking stale package cache，
  证明existing build会选择新record对应package，而不只测试clean configure。
- `tasks/progress.md`只保留当前调度、前置关系、later/external gate和紧凑done索引；逐轮测试数字、实现复盘与历史worklog
  留在编号设计文档、`tasks/archive/`或Git历史中，避免队列再次变成重复事实源。

## 源码模块化与构建门禁

- 同一 production library 内的聚合实现应按稳定语义 family 拆成独立 translation unit；公共 facade 只保留
  legality、选项和 orchestration，共享 helper 放在 owner library 的 private header，不为物理拆文件新增公共 API、
  textual include 聚合或空壳 library target。
- 源码组织 gate 检查 owner、必需 source list、private/public header 边界、legacy aggregate 移除和依赖顺序；不要用
  任意 LOC 阈值、文件数量或任务文档文本代替结构 review。空目录不是 Git 对象，不提交 `.gitkeep` 维持虚假层级。
- 聚合实现拆成多 TU 后，源码型 ABI/conformance checker 必须显式消费完整受控 source set，不能继续只扫描旧 facade；
  同批用强符号集合、registry/count/digest 和正负 vertical 证明没有因 internal linkage 或漏列 CMake source 改变合同。
- 拆分时不能只检查“新函数能链接”。用owner-private header声明集合反向审计每个新TU：真正跨TU的typed helper才保留
  external linkage，未进private header且只在单TU使用的helper收回anonymous namespace；否则会把实现细节扩成静态库符号面，
  private header也容易积累并不存在的协作合同。
- 仓库外部构建的helper若通过生成overlay接入多个source，overlay symlink与外部build `srcs`必须由同一repo-owned source
  清单生成，并对真实外部build做fresh重放。只更新复制入口或只让主文件include其它`.cpp`会绕过本仓库CMake组织gate，
  也不能证明每个translation unit真的独立编译。
- 根 `check-wafer` 应从当前配置实际存在的子 gate 动态组成，避免 feature 组合的 `if/elseif` 漏掉并存 gate；lit 中
  引用的可执行工具必须同时进入 target `DEPENDS`。验证 optional dependency 边界时同时 query feature-on/off 的 Ninja
  graph，并实际运行两个配置的统一入口。

## 内存生命周期analysis的共享边界

- SPM/DDR可共享的是从当前structured IR重算的path condition、operation timeline、query-time
  ViewLike/SelectLike/scf.if/scf.for provenance closure、generic async task completion、live-segment overlap、local
  issue/fence completion和typed static packing；arena、resource limit、descriptor、DTE和accepted offset schema仍由
  各自planner拥有。默认packing先运行受管MiniMalloc fixed-capacity search，使用宽松且确定的全局node fuel，
  不使用短wall-clock timeout；只在`ResourceExhausted`时运行deterministic first-fit安全fallback。完整求解的
  `ProvenInfeasible`才能映射capacity failure，first-fit NoFit不是不可行证明。共享header保持owner library私有，
  不形成跨pass side table或新IR attr。
- compiler-managed allocation `RootRef`、caller-owned/external `ValueOriginRef`和async task identity是三类不同事实。
  async handle的root union只能延长lifetime；不能证明某个task已完成。external origin在最终semantic root上去重，未知
  tracked producer不得退化成external root。
- provenance必须在查询点递归追踪ViewLike、SelectLike、if yield和for init/iter-arg/backedge/result；只在前向遍历时复制
  一次map会漏掉loop fixed-point后来加入的origin。loop发布union时去掉repeatable branch decision，避免把某一iteration
  的分支选择当成整个执行的互斥事实。
- `scf.for` body在timeline中必须有独立的may-zero-trip path。pre-loop issue只在body内fence后仍保留zero-trip pending path；
  body中新issue若在离开body时仍pending则直接拒绝backedge，不能等loop后fence。loop-local if是repeatable decision，
  相反分支不能证明packing互斥；普通lifetime overlap仍按兼容path保守判断。
- generic `async.call` token/value必须由`async.await`完成；只接受direct create/add/await-all group。mutable group alias、
  loop动态task加入captured group、SelectLike distinct task和non-identity-preserving loop recurrence拒绝；if result只完成
  origin存在于对应branch path的task。未await task与unsupported identity flow必须用不同failure class。
- loop body allocation或task一旦通过memref/async handle跨backedge携带，就不是单个静态instance；没有显式
  multi-instance/ping-pong placement时fail closed。non-nested sibling tile-region允许通过显式operand/result共享SPM root，
  并在同一whole-function timeline中规划；nested tile-region仍可能在同一physical arena覆盖outer live buffer，因而当前
  结构化拒绝，同时保守拒绝全部loop-carried async token。
- whole-rank SPM high-water必须从全部accepted allocation的`offset + physicalBytes`相对arena base重算，不能求和或取
  per-region局部peak。provenance closure要沿`wafer.tile.yield -> region result -> sibling operand`延长同一root；否则
  resident handoff会在region出口被错误截断，high-water与reuse结论都会失真。
- DDR module同时含function planning scopes和function外compiler-managed allocation时没有单一timeline；必须拒绝mixed scope，
  不能因发现func.func就静默跳过top-level allocation。
- static memory-space type不是storage provenance。`to_memref`、memory-space cast、ViewLike/control-flow result及
  private helper call只有解析到已有RootRef/ValueOriginRef时才能产生tracked storage；仅DDR func/async入口tensor
  adapter是owner显式external-root例外。private alias helper中所有接触storage-shaped value的op都必须是已知
  alias/control语义，且每个tensor/memref result都须完整解析到静态tracked caller actual；只有无关pure scalar计算可
  独立存在，不能把mixed alias/independent storage result当成no-alias。
- 缺少arena/resource summary的调用按动态执行scope fail closed：DDR module有demand时拒绝external/unresolved
  sync/async及indirect call（defined private pure alias helper除外）；SPM active/async/parallel scope拒绝可能重入
  tile-region的direct/indirect/external call，module有tile-region时external async call全局拒绝。
- packing helper返回以原demand index标识的纯placement结果，不写IR。只有`ProvenInfeasible`可映射
  capacity failure；`ResourceExhausted`才触发first-fit，fallback无解仍保持`ResourceExhausted`，不能升级成
  不可行证明。所有accepted placement先经owner-independent validator复验，owner再完成completion、
  descriptor/range和resource validation并统一提交offset；失败candidate不会留下半份accepted plan。
- 对只接受lifespan/gap的solver精确表达任意pairwise conflict graph时，可对每个connected component
  构造确定性edge-clique activity slots：每个slot必须是原图clique，每条原edge必须被覆盖，non-edge
  不得共slot，并在适配后fail closed复验。这能保留pairwise语义并暴露clique capacity lower bound，但
  triangle-free worst case仍可退化为一edge一slot，不应求解最优clique cover。nonzero arena base用每component
  一个仅覆盖其连续slot区间的fixed prefix表达，避免全局prefix破坏solver的独立分量分解。

## 大规模rank编译与Llama数值纵向

- all-rank lowering的外层并行单位是logical rank：每个worker使用独立、完整注册且关闭内部多线程的MLIR context；结果按rank
  顺序重新导入owner context后再进入跨rank acceptance。不要在共享context上并发运行会安装diagnostic handler的pass。
- 标准Llama-2 7B单block scale corpus由`test/Tools/Inputs/workloads/llama-2-7b-block-v1.json`和对应HF config固定
  H=4096、I=11008、32 heads、FP16、batch 1、sequence 16。最终`expected.npy`必须由同一确定性input/parameter payload的
  PyTorch eager CPU完整block输出产生；手写NumPy仅用于定位误差。production完成入口仍是一次`wafer-compile
  --target-model`的TP16 package与全局output comparison，不把临时corpus目录或生成package写成长期路径。
- scale重放必须从同一次corpus publication取得program data、input、全部parameters、expected和reference metadata，并核对
  同源identity/finite contract；函数MLIR或canonical program digest相同只证明结构等价，不能证明payload代次相同。
  出现数值域失败时先核对完整corpus identity和final typed IR，不从历史目录混配reference，也不靠放宽容差定位。
- TP16 exporter在CPU PJRT上需要显式提供16个logical devices；运行repository workload exporter时设置
  `CPU_NUM_DEVICES=16`，否则XLA会在需要16 devices而只发现默认1 device时结构化拒绝。这只影响本地source
  corpus生成环境，不是compiler、IR、package或runtime协议。
- scale corpus不能使用会沿matrix axis重复的短周期序列，也不能让不同parameter stream保持系统性相关。按global
  row-major counter、固定seed和显式独立stream生成versioned长周期payload，并在digest/artifact publication前检查
  input、全部parameter和expected均finite；public NPY多字节payload统一canonical little-endian。
- target-facing GEMM只接受rank-2 `Cx` plain form或rank-3 single-leading-batch canonical `NCx` form；target call没有
  rank/layout/dimension map，rank >= 4或permuted batch axis必须在ABI前显式canonicalize，否则拒绝。`B=1`的Cx/NCx
  物理等价只能用于解释ABI rank擦除，不能放宽source verifier。
- 大矩阵payload生成应直接分块填充最终FP16 allocation，避免同时保留全尺寸uint64/int32/f32临时数组。大GEMM可走显式
  `managed-reference` oneDNN lane以完成model-reference scale gate，但该environment provenance不能冒充exact qualification
  admission或board-correlated arithmetic。
- scale CModel重放使用Release构建；高并发构建和rank lowering可按机器容量提高`-j`，但这不会并行化当前SystemC保守
  single-issue下的formal APFloat/MPFR命令执行。标准全量gate应作为显式长任务，日常`check-wafer`继续用同合同的小case覆盖
  positive、negative和atomicity，不能把debug构建耗时误判为compiler并发不足。
- scale执行显式使用`--target-model-numeric-policy=managed-reference`；`formal`用于小规模raw-exact，
  `prefer-admitted`只允许exact-record GEMM。managed结果必须同时检查formal command/FMA为零、tensor/bulk command计数、
  environment/implementation evidence和完整PyTorch expected，不能只看命令成功返回。
- scale source/model误差表征使用显式`--model-report-numeric-statistics`，并读取全部rank报告；replicated output的首个rank
  fail-fast diagnostic不代表全卡最坏值。absolute-error quantile适合有限corpus gate，destination-format ULP用于定位舍入差异，
  但near-zero会把很小的绝对误差放大为数千ULP，不能在没有独立语义依据时直接设成硬阈值。
- dtype adapter是scale profile的一部分：F16/BF16到oneDNN F32输入采用精确bit widening，不能为无损转换逐元素构造
  APFloat。当前canonical bulk artifact是SEQ且cache关闭，Release 7B gate仍属慢测；若引入OMP/threadpool或descriptor/
  packed-weight cache，必须更新受管依赖record、environment digest和资格/数值回归，不能继承ambient线程数。
- CModel strided movement保留一份source snapshot和compact descriptor；规则nested stride用span/non-overlap证明，fallback
  线性枚举后排序验证。不要按segment构造独立payload/pending-write对象；所有range、resource、overflow、alignment和
  destination overlap必须在任一write前完成验证，保持source-before-write与命令级原子性。
- 大规模production vertical先用Release fresh baseline和累计CPU profile定位阶段，再对热点做局部wall-time分解；16-rank
  编译的累计CPU百分比不能直接当wall百分比，SystemC command时间也要继续拆成physical unpack、adapter、backend compute和
  finalize。只有实际backend compute成为主要成本时才值得修改oneDNN thread runtime或受管依赖身份。
- 静态layout hot loop应把`memref type -> physical geometry`构造成command/conversion-local可重算calculator，并用已验证
  lexicographic odometer流式遍历；不要为每个element重复反线性化、重建MLIR layout事实或物化等长offset side table。
  性能收口必须同时比较完整package目录、target-model计数/environment和独立expected，不能只依赖wall time或同实现oracle。

## Bounded actual-clone selection

- candidate hard cap必须覆盖生成和保留两端：source clone、每类typed recipe、scope policy、rank evaluation、rank frontier、
  whole tuple和whole Pareto分别限额，唯一conservative baseline用独立allowance先跑全部late gate。只限制frontier top-K却允许
  cap前无界生成不算bounded；budget耗尽不得淘汰已验证baseline。
- 多轴联合覆盖要从actual clone验证：各轴单点先入列，再优先materialize source×recipe、source×policy和all-applicable状态。
  若implementation与route/communication在同一task可共存，必须形成同一complete recipe；两个独立passing单点不能证明联合状态。
- full-shape direct boundary route应从原始task boundary物化。若先构造one-trip complete traversal，identity
  extract/insert slice可能只在lowering后显现，使distinct direct route静默退回Tensor staging；非full tile仍必须依赖exact
  mapped-transfer proof，不能用shape或recipe flag强行直连。
- final selection先保留完整exact Pareto frontier，再由target profile显式static policy在survivors间持续比较当前winner。
  不能找到第一个优于baseline的candidate就返回，否则较早share/fusion会遮住DDR movement更低的recompute联合candidate。
  generation/discovery ordinal只用于确定性与跨rank对应，不是语义winner维度；Unknown、overflow或同一priority class双向tradeoff
  保持保守。
- 昂贵target ABI/LLVM gate可以延迟，但只允许以已经fully target-gated的optimized Pareto frontier作剪枝事实源：
  pre-target candidate先精确模拟正式insertion的Unknown/dominance、equivalent dataflow、stable order和frontier cap；
  would-retain者通过target gate后才能淘汰旧项，target失败不能改变frontier且必须继续后续candidate。reserved baseline先完整
  target-gate，不能用它额外推导可能不具传递性的optimized剪枝。
- ready-order的buffer hazard必须先沿`ViewLikeOpInterface`追到storage base；base memref与cast/subview/reshape view不是独立
  allocation。exact SSA value比较只适合use-def依赖，不能作为memory alias proof。

## Production optimizer同源成对板测

- 验证优化归因时，用同一source snapshot、payload、ExecutionConfig、target profile和host-visible ABI分别发布
  coordinator已接受的reserved baseline与默认production winner。普通对照使用正式CLI的`none`/`production`
  candidate-domain preset；单轴归因使用`production + disable-one`和`none + enable-one`互证，不能靠改source、
  跳correctness pass或编译两个版本构造对照。每个稳定option映射一个语义alternative owner，不映射pass、case或文件；
  canonicalization、verifier、SPM/DDR、completion、transport/resource、ABI和publication始终执行。私有selector只保留给
  公开语义轴无法表达的accepted算法参数/结构资格，不能再次承担通用优化开关。
- optimization configuration作为invocation-local typed value从driver传播到各candidate producer，不写入IR、package或
  selection cost。diagnostic按稳定顺序打印canonical enabled/disabled集合；unknown、duplicate和enable/disable conflict在
  编译/发布前拒绝。ordinary与profile必须消费同一配置并选择字节一致的production package。
- 两份candidate必须各自重放完整SPM/DDR、instruction、transport、ABI、device-link、manifest和readback gate。先核对
  manifest resource role/type/shape/bytes/alignment与launch/completion boundary一致，再从最终linked ELF检查目标call的
  数量、种类、workspace、scheduler-body hash或已证明straight-line body中的顺序差异；没有对应解析器时不外推
  dynamic CFG、loop归属或peer graph。pre-lowering IR和candidate counter不能代替可执行结构。
- 板端正确性对baseline和winner使用同一独立CPU expected，完整比较output、write-only complement canary、
  transport status和lifecycle。
  一包失败即停止该pair；单winner通过只能证明该winner在tested domain可执行，不能证明优化归因或相对收益。
- 同一已资格环境中按A/B、B/A平衡顺序单进程串行，初始/终止heartbeat各一次，所有launch有bounded outer timeout。
  timeout或设备异常立即停批，不自动retry/reset/power。raw sample绑定package、device/runtime/firmware/toolchain identity。
- 成对case的durable archive不能只保存summary JSON：至少保留逐字节相同的source snapshot、两份最终linked ELF和
  manifest、结构/观测JSON、raw payload，以及实际compiler/runtime/objdump路径和digest；否则事后不能重放ELF oracle或
  判断工具身份是否漂移。
- 包含allocation/H2D/load/D2H/cleanup的process wall time受provider、OS和runtime噪声影响，只能记observation。
  profiler的主耗时只使用同一qualified session中未插桩最终production package在TX device stream上的start/end
  event elapsed time；多phase时按phase求和。host submit调用耗时、submit→all-rank trusted-completion envelope和
  poll-gap上界分别保留为diagnostic，并继续要求完整output validation。PMU、entry-local `rdcycle`、count和trace来自
  另行launch的diagnostic capture，只解释最终artifact的engine activity，不能改写主耗时。把证据反馈到compiler
  ranking仍是独立后续任务，需要另行冻结environment/profile、比较对象和稳定性gate；profiler foundation本身
  不构造candidate、baseline、winner或speed verdict。

## 16-tile TSM profiler workflow

- profiler是`wafer-compile --profile`的一项compiler功能，不是第二个可执行文件或runtime mode。普通package必须与
  profile关闭时逐字节一致；companion在临时目录完整形成后最后写`activation.json`，以production manifest SHA-256
  及`plan.json`、`variants.json`、`site-map.json`的逐项SHA-256作为唯一激活边界。读取侧在解析或board effect前先做
  exact key/digest验证，不能扫描目录、文件名或resource name猜身份。compiler在最终原子rename前把整个companion
  directory/regular-file树设为`0777`；no-card和live gate必须检查根及全部后代，不能只检查`runs`和报告叶子。
- 一个companion只有一个`final-artifact`未插桩execution binding，以及count/trace两个内部capture binding。
  runner自动发现并复用普通`wafer-run`的resource、typed expected comparator、output和board参数，不向用户暴露capture选择
  或采样参数。缺external expected的writable resource由Primary按稳定semantic key建立同session exact reference；
  它只证明本轮diagnostic capture与Primary等价。所有launch留在一个qualified、driver-bound session中单进程串行，
  首个timeout、device异常、correctness或协议失败立即停止，不retry/reset/power。
- fixed protocol是一次未插桩最终production package的高分辨率Primary，再各执行一次Count和Trace，共三次launch。
  Primary的TX same-stream event elapsed time是唯一用户级kernel/model主耗时；host submit、host
  launch-to-completion和completion observer resolution分栏显示为diagnostic。不自动warm-up、不默认重复benchmark，
  也不计算median/range。
  Trace header已同时携带该次entry span和aggregate PMU，因此不另建重复事实的summary capture。Count只为固定trace
  buffer提供动态容量预检；trace evidence保留preflight count、`next_sequence`、drop count、raw flags、terminal
  state和record guard，任何overflow/drop/gap/mismatch都fail closed。
- profile CRT必须保持普通completion语义：非trace路径继续调用真实`TsmWaitfinish()`，trace poll同样只在
  `TASK_DONE == 1`时结束并在等待期间采样。只有trace临时enable/restore Direct-DTE PMU；split counter不稳定时显式
  标记raw delta不可用。target测试要验证宏展开后的predicate和fallback，不能只看宏、源码else分支或符号存在。
- `site_id`只在rank内有效，canonical site map来自未插桩final target LLVM；运行时用
  `TARGET_SITE` container sequence建立唯一动态实例，container固定`sub_index=0`，child共享site
  ID/envelope并从1连续递增。NCC command/completion与Direct-DTE aggregate/leaf event挂在实例下而不重复
  造site。NCC event是累计PMU增长所在的
  有界observation window；CT/NE/RDMA/WDMA/TDMA execution delta按vendor producer/parser合同直接是nanoseconds，
  window的begin/end则是Kcore `rdcycle` CPU cycles。sample必须在counter reads前后各取一次cycle；zero delta保留
  `counter-no-change`，same-engine outstanding标`attribution-ambiguous`，不能再把所有后续delta绑给latest site。
  单个engine counter不可用只降级该`(tile, engine)`字段，不传播成全局measurement invalid。
  Direct-DTE event是排除PMU split-read后的实际wait/completion wrapper窗口；aggregate只作container和raw PMU
  归属点，peer-ready、setup/issue、completion wait、cleanup是exclusive叶子phase，有叶子时不得再加aggregate。
  raw DTE PMU只作未校准活动证据。Trace entry、site envelope和operation span的成对`rdcycle`是`Measured`，
  NCC PMU activity window是`Bounded`，counter失效只令engine observation为`Unavailable`；状态不能按整条event
  传播。Direct-DTE板端gate按manifest participant逐tile要求正值source/aggregate/`Measured` phase。
  timeline只能使用Trace自身entry-local `rdcycle`横轴，同tile不同engine可显示
  overlap。Kcore主ledger显式覆盖带`prev/next`的entry-prologue、动态site、between-site-gap和entry-epilogue；
  site envelope减去叶子operation union后得到site-control，不能再留下泛化`unobserved`空白或推断idle。
  所有Trace语义区间的数值只可标proxy或mixed/proxy；PMU sample、event/site bookkeeping、status poll和DTE probe
  作为不计入Primary、也不参与主ledger求和的Trace-only overlay。entry setup/teardown位于entry横轴外单列；
  completion wait等production语义phase虽计入Primary，其Trace cycle仍不能冒充Primary精确成本。
  `statistics_window`保留raw ticks，`tile_clock`仅保留为metadata，不能据此把raw ticks或`rdcycle`换算成ns；
  也不能把Primary device/host时间或其它tile的local cycle混入同一绝对轴。
- local cycle domain没有资格化mapping时，报告仍显示all-and-only 16个逐tile、六engine timeline，但不能声明cross-tile
  order、overlap或global critical path。external expected显式记录`exact`或`relaxed-f16` comparison policy；否则同session
  exact reference只证明repeatability，不能谎称bit-exact semantic correctness。
- profiler timeline逐次展示真实动态调用，不按密度、engine或site折叠。每个实心块保留site instance和精确
  operation rdcycle；NCC PMU active ns另列。site container、engine observation和DTE内部phase是关联证据，
  不能算成额外调用；Program / Sites必须把实际dynamic instance数与activity event数分栏。
- profiler硬件cost reference从accepted final Instr IR fresh发布exact per-rank logical ops/bytes及
  knowledge/reason，并只消费target policy唯一事实源。CT/NE可给peak-throughput理论下界；共享DDR只能先给whole-card
  traffic floor，rank workload对称时才给fair-share启发式；没有SPM bandwidth的TDMA时间必须Unavailable；NoC payload
  serialization只能叫reference而不是collective latency。模型、PMU active ns和Primary wall time分栏，不相加、不回灌
  compiler selection，Unknown/Unsupported/Overflow不能写成零。
- evidence由single-final-artifact analyzer直接消费一个Primary样本，不构造ABBA/BAAB、baseline/winner、speedup或
  signed candidate delta。重复benchmark若以后需要，应作为显式独立workflow，不能重新混入默认profiler。report在
  临时目录中完整生成后原子替换`runs/current`，只公开`evidence.json`、`analysis.json`和
  `index.html`三个`0777`文件；`runs`和current目标目录也必须可穿越。旧目标回收前校验三成员及evidence `run_id`，
  删除失败显式报错。no-card只能证明compiler/ABI/decoder/campaign/report协议；未执行且未确认non-skipped的configured
  live-board gate时，Q9保持进行中。

## Hardware characterization与优化资格分栏

- production winner/baseline成对资格只回答“当前默认选择是否改变了可执行结构、两包是否正确”，不能代表
  compiler全部选择空间或硬件性能表面已覆盖。硬件characterization应从compiler实际可选机制和参数轴反推case，
  已有板端证据做差集去重，并为每个新增case写明正负/matched control、可消费结论和stop gate。
- Direct-DTE是current collective共同的transport合同，Direct/Ring/Tree是compiler-private collective schedule；
  名为Direct-DTE的AllReduce纵向不能记成“Direct AllReduce算法”。算法case必须在accepted Instr IR仍保留
  完整message tuple时做typed确认，不能由baseline/winner名字或最终ELF prepare数量反推。
- accepted Instr sidecar累计通信bytes时要乘所有constant-trip structured loop的执行multiplicity；大payload可能被
  planner分tile而只保留一组静态DTE callsite。只数callsite operand bytes会把真实执行量误报为`1/trip_count`，
  但动态/非constant loop仍应fail closed，不能猜trip。
- collective graph evidence必须保留每条message的direction、peer、communication、phase、round、payload slice、
  issue bytes和loop multiplicity tuple，并从tuple重算摘要和跨rankmatching。彼此独立的peer/round/slice集合会丢失
  关系，不能证明Ring cycle、Tree edge或message identity。
- i8 collective sentinel不能使用rank/lane线性mod-256序列；多rank modular sum会把系数折叠成短周期，
  让destination permutation和tile rotation通过exact compare。使用rank/logical-lane/payload-size counter hash，
  并在不上板的mutation gate中预先证明rank contribution、RS destination、短rotation及source
  missing/duplicate均可被oracle区分。
- Direct-DTE板测证据要区分manifest合同、runtime内部fail-closed enforcement和output可独立观察的raw status。
  current runtime会在发布output前读取并校验16-rank status-v2，但CLI/result未导出逐rank raw值；archive必须显式记录
  observation gap，不能把terminal completion或成功返回改写成“raw Success已观察”。
- case定义、pending/board状态、profile身份、raw evidence和最终compiler消费应在同一硬件校准台账逐row演进；
  实施计划只保存施工依赖。没有可信device measurement basis时，结构和exact execution只形成behavior/correctness
  evidence，不产生性能winner。

## 未上板校准项的机器化准备

- 文档中的`pending`只有同时绑定typed catalog row、真实device carrier/dispatcher、固定输入、完整结果或有界
  observation、guard/count/status/lifecycle oracle、no-card、board CTest和显式runner step时才算可发板；文件名、
  family总数或计划表一行都不能代替这些对象。
- matched characterization的激活单位是完整execution group，不是单个cell。CTest和runner按group注册，group内共享
  package/allocation/measurement basis并包含control、方向/顺序轮换、重复和held-out；机器审计同时核对catalog cell、
  group、CTest与runner四方集合一致。
- 无owner-backed observable surface时不要为了“每行都有板测”伪造surrogate。建立typed preparation rejection，
  明确缺失字段、拒绝的替代推断、最近的安全可执行family和解锁条件；ABI/catalog扩展后host gate必须因stale而失败，
  迫使重新审核。
- 总发板入口从central inventory自动收集全部`pending-board` CTest，保持单进程、resource lock、bounded timeout、
  first-failure stop和no retry/reset/power。CPU增量构建可高并发；硬件launch仍串行。no-card与host gate通过只标记
  pre-board readiness，不升级成`board-observed`。
- 需要跨CTest合并small/steady/tail或control/experiment时，runner必须为本次execute生成不可复用的session id，
  archive还要精确绑定target profile、完整runtime launch contract、device/runtime身份和runtime library digest；只靠work directory
  或case名会把旧轮、旧卡或其它profile的结果混进当前分类。
- matched group按sample-major执行并在sample间轮换condition顺序；涉及physical rank时，baseline和其它condition
  必须使用同phase的nested active set，避免tile差异伪装成contention slope。只做到equal-mean position的四轮
  rotation应明确不是完整Latin rotation，不能把顺序平衡程度写得比实际更强。
- 大型确定性resource先做完整逐字节oracle，再在archive中保存request/record、生成参数和完整输入输出SHA-256；
  JSON使用同目录临时文件、`flush`/`fsync`和原子replace，成功后才删除经过exact-set与路径校验的raw文件。
  验证失败或原子发布失败时保留raw，既限制成功批次磁盘峰值，也不丢失失败归因证据。

## 静态多buffer软件流水

- fixed-slot流水必须从尚无SPM/DDR physical offset的complete-rank IR派生；production顺序中它可以接收已经
  materialize actual worker attrs和minimum joins的typed-worker sibling，但必须原样保留该assignment。随后用
  typed instruction、SSA、MemoryEffects、alias root和completion domain建立DAG，把loop-local static
  allocation变成loop-external普通allocation family，以`scf.for iter_args/yield`表达slot rotation，最后交给
  SCF utility机械生成prologue/steady/epilogue。公共派生API遇到已有SPM/DDR physical offset必须拒绝；否则
  clone会复制物理地址事实，使多个逻辑slot静默重叠。
- slot数由每个root的`lastStage - firstStage + 1` lifetime span和capacity共同决定，不固定双缓冲，也不读取
  header queue depth。队列的D/D+1只能证明有界总提交，不证明resident window；SPM high-water和fixed-capacity
  placement必须在每个actual clone上重算，并检查每个slot获得不同的half-open physical range。
- SCF流水会把原始index改写成`iv + stage displacement`等静态表达式。DDR planner、Target preflight和其它
  downstream consumer应共享一个overflow-safe、fail-closed的静态index range evaluator，支持constant-bounded
  induction variable及受限add/sub/mul/div；不能一个层接受而另一个层仍只认bare IV。
- same-worker pending NCC stream可跨statically non-empty nested loop和tile-region边界传播。loop container、
  loop-local allocation和纯pointer permutation是结构节点；后续same-worker issue可接管有序访问责任，最终由
  一个unconditional participant join收口。conditional region、Direct DTE/Kcore observer、cross-worker冲突、
  root/range Unknown仍必须fail closed，不能为了让planner通过而补逐iteration waitfinish。
- `TargetProfileId`在scheduler入口只代表compiler-shipped、versioned target/ABI合同。候选生成必须离线确定，
  不得读取实卡身份、Q9 profiler、PMU、runtime历史或本地校准缓存；板端结果只能离线验证实现，若要改变静态
  capability，必须通过后续compiler revision评审发布，不能形成per-card schedule。
- fixed-slot source-to-package资格不要把accepted Instr schedule塞进manifest。testing seam从同一次fully accepted
  `ExecutableBundle`派生闭合attestation，至少绑定accepted module digest、placed SPM roots、static loop/root rotation、
  engine×worker issue、DTE token/exact wait和completion；activation最后写入并同时绑定attestation及staged
  schema-v6 manifest bytes。package与相邻qualification目录用双rename no-replace transaction发布，companion失败后
  回滚package；normal production mode不检查也不生成这个testing sibling。需要数值资格时复用同次retained
  TargetLLVMModuleBundle进入SystemC，不重新lower或重编。
- 把root-path storage coalescing扩展到loop body时，`StructuredTimeline`只给出一次静态body顺序，不能代表
  dynamic iteration。最小安全入口应显式建立loop context，只接受direct、无条件、static-positive loop，
  要求root定义在loop外，并让alias access/forwarding同时携带owner和path；另外证明destination copy后只读、
  source snapshot、replacement dominance、同path exact DTE wait及backedge completion。任一证明Unknown就保留
  movement，self-copy也不能在这些门禁前提前删除。
- cheap candidate geometry必须按最终物理instruction实际编码字段分类验证。high-level reduction scope不等于
  最终一定发射CT Reduce；unit-local reduction若最终是CT elementwise，应使用其`uint32_t elem_count`合同，
  只有真实CT Reduce traversal和NE GEMM窄维度才施加各自`uint16_t`字段限制。

## NoC-resident complete-tuple合成

- NoC dataflow的生成单位是correspondence一致的complete-rank actual tuple，不是某个rank的局部rewrite。
  input/parameter、required output、intermediate和已有collective partial等materializer都只消费当前clone，
  按依赖顺序累计改写；任何rank的relation、message、lifetime、SPM、completion或target gate失败都丢弃整组，
  不向frontier部分提交，也不保存role枚举、owner表或shadow schedule。
- worker placement是独立post-Instr candidate维度，不是resident role materializer或metadata的附带字段。
  materialized canonical/unplaced current Instr先从SSA、typed value-associated effects、exact或保守static
  ranges及stable issue order形成all-rank atomic sibling；worker选择直接写入actual issue attrs，旧
  compiler-generated joins删除后从current IR fresh建立minimum joins。Unknown conflict保持同lane，已有
  nonzero assignment不原地重写。fixed-slot只在保留各自worker assignment的siblings上继续派生；每个结果
  fresh重跑SPM/DDR、Direct-DTE、resource和target gate。
- `RankArtifactKind`、buffering kind和worker-placement kind只记录candidate来源及correspondence轴，不能代替
  accepted IR语义。需要证明NoC residency时，从当前Instr IR沿TileRegion边界、透明cast和ViewLike追踪DDR
  movement root：RDMA只能读entry DDR参数，WDMA只能写唯一entry return可达的DDR root；private allocation、
  helper-local/unknown root和无法解释的alias一律fail closed。即使标签为`Spill`，current IR满足该条件也可
  resident；即使标签为`Resident`，仍有private spill/reload也必须拒绝。
- 多机制组合资格必须在同一个pre-target tuple上同时检查actual IR：NoC send/recv、actual typed NCC
  workers、fresh minimum joins及随后派生的worker-preserving fixed-slot loop/root rotation分别存在，并继续
  通过相同completion、resource、TargetCall、package及model
  gates。三个独立passing candidate、metadata拼接或最终aggregate计数都不能证明NoC×fixed-slot×worker组合。
- source vertical的role归因必须由final IR/profile重证。一个compound source可同时证明boundary owner-load、
  collective partial、下游compute family和required output coverage，但只有出现对应round/message与DDR cut时
  才能声称intermediate或output publication materializer实际触发；独立role单测不能被文案合并成“一个case包含
  全部role”。
- communication report区分static issue bytes和dynamic executed bytes。常量loop中的一条静态DTE site必须
  发布`issue_bytes`及完整loop multiplicity，并满足
  `issue_bytes * constant_loop_multiplicity == executed_bytes`；report/profile总流量按executed bytes求和，
  不能因tiling把一份payload拆成多次静态执行就仍断言multiplicity为1。
- NoC equivalence class、communication ID和owner是可见的物理编译结果，必须由rank-major current-IR
  discovery ordinal、program member ordinal及exact global-tile/structured-path/SSA-effect等价确定。
  `llvm::hash_code`在启用ABI breaking checks的构建中可能按进程加盐，只能作同次analysis的预筛，不能参与
  排序、取模或tie-break。回归同时固定具体owner/communication ID，并用两个独立compiler进程比较完整package bytes。

## Complete-rank NoC profitability

- NoC跨资源选择必须相对同一source/config中已通过全部late gate的reserved baseline，从两份current final Instr
  fresh重算；不能复用rank-local估分、artifact label或先前candidate的analysis。
- `ScheduleCostKnowledge::Known`只描述final Instr与typed topology能完整精确计数的work：DDR/SPM bytes、
  compute ops、DTE bytes/messages、endpoint pressure和minimum-hop work。它不证明物理route、arbiter或
  duration；不要因为timing calibration缺失把这些work改成Unknown。
- DDR read/write先在16-rank完整domain求和，再只除以一次整卡带宽。`200 GB/s`只作DDR lower，`150 GB/s`
  是整卡nominal operating point；`128 GB/s`分别作为directional link reference和显式分开的DTE endpoint
  point prior。modeled deterministic shortest path必须标`EstimatedRoute`，不能冒充actual hot-link。
- point model还显式携带message `α`、maximum-route fill hop、SPM和control prior；当前Direct-DTE
  `α=10 us`是versioned policy prior，不是测量，未来由matched board calibration替换。论文只贡献
  `α+nβ`、congestion/dilation、
  work-centric partition和double-buffer resource-envelope等结构，绝对参数不能直接移植。
- 没有qualified multi-buffer时，DDR/NoC/compute按sequential phases计费；只有current-IR fixed-slot、
  exact wait/reuse cut与target capability共同闭合才按steady-state resource maximum。point comparison以
  `candidate.nominal * 1.20 < baseline.nominal`签发normal `EstimatedBenefit`；真实conservative
  `candidate.upper * 1.20 < baseline.lower`才升级`ProvenBenefit`。缺少bound不导致`Indeterminate`，
  必要work仍dynamic/unsupported或算术失败才导致它。
- 新门禁只接管“DDR严格下降且candidate仍依赖NoC执行”的cross-resource tradeoff，包括新增/增加traffic或
  保留已有collective；NoC-free的local resident/recompute等优化继续由现有exact Pareto/static selector处理，
  Unknown DDR不能因此被NoC门禁误伤。

## Direct-DTE/compute matched qualification

- overlap资格从完成SPM/DDR planning与Direct-DTE binding的accepted Instr重算：V3 bound issue、唯一same-block
  exact wait、中间FP16/BF16 CT/NE、operand-specific effect和planned static byte range必须全部闭合。
  fixed-slot identity从同一个complete tuple独立证明；endpoint specialization后不要求issue仍位于原rotating loop。
- digest-bound companion只发布accepted Instr digest、每rank结构窗口计数、DTE token/wait、SPM root和static-loop
  inventory。它用于审计package边界，不复制per-window schedule，也不能回流selection。
- matched no-overlap baseline必须从同一个fully accepted tuple延后本地compute，并重跑transport/resource/accepted-rank/
  target/package gate。逐op binding、cluster launch、transport status ABI、Direct-DTE call inventory、payload和CPU oracle
  与candidate一致；scheduler顺序必须不同。
- board-ready入口先生成两份完整package，检查companion/manifest/target structure与FP16 exact payload，再分别运行
  fresh no-card。真实板端只在显式armed且身份参数完整时按A/B、B/A单进程串行执行；no-card、host elapsed和scheduler
  hash都不能签发hardware overlap或profitability。

## Source-backed 技术汇报构建

- 编译器技术汇报先按production stage建立source map，把representation、analysis、transformation、gate、
  artifact和测试锚点对应起来，再写页面。公开pass注册表只能作为附录索引，不能代替实际driver中的
  transaction顺序。
- IR页面从当前代码、FileCheck或本轮focused运行截取与论点直接相关的8--18行，保留决定语义的op、type、
  SSA、range、layout、token和ABI字段；长dump拆页或进入附录，不为排版删除关键合同。
- 算法、ownership、matching和artifact transaction使用与其语义匹配的DAG、地址几何、frontier或时间线。
  生成式位图可以辅助复杂机制图，但必须先从source冻结精确IR、数字、字段、状态和关系；机制、因果、失败和
  边界直接以中文写入最终位图，精确token同时在页面原生IR/表格中保留。不能依赖事后PPT覆盖去修正主图语义，
  也不能用整页位图代替正文。
- 如果Image2主图为后续overlay预留空白，装配验收必须直接检查最终PPT/PDF中overlay是否真的出现；页面规格里
  只有`figure_overlay_labels`字段不算已渲染。没有可靠坐标化overlay时，应在正式位图中直接写入经source核对的
  中文机制说明，并由页面旁的原生IR/表格保留可复制的精确token，不能把空白callout交付给听众。
- 生成式技术图中的每个op、field、type、状态、数值、边、完成关系和artifact分支都要逐项回查代码、测试或
  当前设计合同；Image2自动补出的寄存器、ABI slot、伪指令、硬件拓扑、候选数和性能曲线一律删除。特别检查
  completion domain、aggregate/PerRank publication和case覆盖边界，避免把相邻case的数字或token混到同一图。
- 同一份页面规格生成PPTX、嵌入Notes、PDF、逐页PNG和contact sheet。交付前同时检查slide/notes数量、
  PPT对象边界、PDF页数、预览分辨率、图片链接、可见文字密度和source map覆盖；100% contact sheet检查后，
  对IR、表格和复杂图页再做原尺寸抽查。

## PyTorch source 板端 tensor 对比

- 普通PyTorch source case用固定seed的`torch.rand`/`torch.randn`构造输入，并用同一module/op和同一组tensor在
  CPU eager直接形成参考结果。周期pattern、one-hot、整数公式或手写等价计算只用于失败后的定向debug，不能
  进入常规完成证据。
- case dtype由输入tensor声明；公共raw tensor读写和comparator按manifest与tensor自身dtype/shape处理，不写死FP16，
  也不把actual或参考结果换算到其它精度。configured-board实例可按板测默认规则选择FP16/BF16，但这只是case参数。
- `wafer-run`只接收输入resource和output capture，不接收PyTorch参考结果做provider raw comparison。capture完成后
  由Python按原shape/dtype解码为`torch.Tensor`，再用`torch.testing.assert_close`完整比较所有声明output；raw
  文件只承担传输，不能成为第二份数值参考。
- Megatron TP source case的parameter角色应由module结构显式提供sharding spec，并按parameter identity校验all-and-only
  覆盖；不要通过parameter name推断column-parallel、row-parallel或replicated语义。完整Transformer block至少核对
  Q/K/V与Gate/Up的column-parallel、O与Down的row-parallel，以及每个row-parallel projection自然形成的AllReduce。
- 板端source case应复用对应qualification workload的实际shape；Q39 distributed GEMM是16-rank `4096³`、local
  K=`256`，Llama block使用已有Llama-2 7B config而不是为缩短编译另造tiny config。缩小shape只适合结构单元测试，
  不能取得该board case的`board-ready`身份。
- PyTorch板端case只有在真实source export、post-SPMD结构检查、production compile、完整package/manifest/bindings、
  board runner注册和fresh no-card全部闭合后才能标`board-ready`；真实设备未执行时仍不能标`done`。

## 编译耗时分层定位

- production compile需要深入定位时显式给`wafer-compile`增加`--compile-timing`；默认不加，避免日常编译承担
  详细instrumentation。该选项只产生本次invocation diagnostic，不进入IR、package、cache key或selection。
- 正常结束会输出Markdown表格，列为`kind / pipeline / item / calls / cumulative wall / cumulative CPU /
  average wall / max wall / failures`并按累计wall降序排列。并行worker的累计wall/CPU描述总work，允许超过
  transaction wall；判断关键路径还要结合active leaf和外层stage wall。
- 每10秒会输出各线程仍active的最内层边界，以及已完成项累计work的Top-N。长case应先用固定短窗口采样确认
  `stage -> search/pass -> lowering pattern -> algorithm`热点，再决定是否需要完整compile；外部timeout前无需等待
  最终表格即可保留可解释证据。短窗口只解释当时执行的phase；优化后必须用同一真实shape跑完整transaction，
  结合外层stage wall重新排序热点，不能把旧phase的局部Top-N继续当成全程结论。
- timing owner按thread散列聚合并最终稳定归并；新增span优先放在stage、candidate、block或可独立行动的analysis
  边界，不给高频递归、逐元素或每次微小eligibility调用加span。首次model-scale诊断同时观察事件数、transaction
  wall及累计线程CPU；累计wall异常大而CPU很小时，先检查observer锁竞争，不能据此判断并行优化退化。
- 日志可直接按`compile-stats stage=`提取串行关键路径，按Markdown table的`cumulative CPU ms`提取并行总work。
  前者用于回答用户等待多久，后者用于选择复用、缓存或剪枝位置；inclusive父子行不能相加。
- 优化必须沿计时证据继续下钻，并保持typed legality：能由static stride/permutation直接构造exact descriptor时
  使用symbolic relation/encoding piece planner，不能证明时structured failure；剪枝只能提前执行已有exact rejection
  或显式修改typed candidate domain。
  不通过延长timeout、缩小真实workload或按shape/op/name matcher掩盖算法复杂度。

## Layout consumer审计

- 不以搜索`Cx`/`NCx`枚举分支作为完整审计。先列出所有会回答logical mapping、physical span、footprint、range、
  alignment、traffic bytes或alias的问题，再逐项确认其事实源分别是`IndexRelation`、physical encoding interface或
  两者的composed access analysis；capability admission可以看layout enum，地址和容量计算不可以。
- 审计时单独区分logical compact payload、physical allocation footprint和engine traffic envelope。collective/network
  message及host tensor文件通常是logical payload；SPM/DDR placement和ABI slot是physical footprint；movement cost取
  已lowered descriptor envelope。三者数值偶然相同不能成为替换依据。
- blocked layout若同时携带非identity MLIR memref layout，必须由显式组合合同解释；当前没有该合同就调用
  `computeWaferPhysicalTensorInfo`或physical encoding interface确认失败关闭。generic reshape/subview不能仅凭element
  count或相同memory space认定为metadata alias。
- encoding新增或修改physical-layout piece时，owner测试要用独立坐标oracle覆盖完整valid domain，检查piece domain
  覆盖/互斥、element start无碰撞、span不越footprint和byte-addressability；production consumer组合piece relation，
  不在每个candidate上重复穷举坐标或重证encoding固有的注入性。metadata view比较两端组合physical relation全域相等，
  descriptor lowering从同一relation取offset、从piece period取符号分段。
- focused回归至少包含Tensor/NTensor/Cx/NCx全pair、F16/BF16/F32/I8、跨CBlock点、C0/tail和per-N padding，另跑
  independent coordinate oracle、memory alignment lit、mapped movement、target/model codec及source-to-instruction链路。
