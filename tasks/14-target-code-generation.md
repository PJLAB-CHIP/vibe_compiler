# Wafer Target Code Generation 与 TargetCall

状态：本文是当前 target conversion、LLVM module、device link、readback 与 TargetCall 的唯一现行合同。单卡编译边界固定覆盖16个
available Tiles；Q49先收口`none` baseline，Q51再把selected `CardExecutable`接入这条边界。现有host/model
验证不能代替Q53的fresh package/no-card和真实板端matched A/B gate。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  Q49 `none` baseline或Q51 `search`选中的`CardExecutable`；其中all-and-only `wafer.tile.module`已投影为16个
  Tile ModuleOp，并完成TileRegion→Instr、fresh completion、SPM/DDR placement、transport与executable verification。
- Current stage responsibility:
  对每个 Tile 做 current target ABI preparation、Instr→Target LLVM conversion、LLVM translation、
  target-call legality、device link、ELF/readback 验证，并原子写入保留显式物理身份的 target modules。
- Output IR / files:
  与同一`CardExecutable`绑定的invocation-local target LLVM owner set及原子发布target-module view；每个Tile
  interface都携带(card_id, tile_id, launch_slot)、entry symbol、typed Kernel ABI slots、module relation、
  target identity、runtime ABI、format与digest。它们是`CardExecutable -> ExecutablePackage`之间的lowering内部表示，
  不是新的稳定output层。
- Downstream consumer:
  `ExecutablePackage` assembly、no-card/runtime validation、host TargetCall frontend、SystemC model、profile instrumentation
  与 board runtime provider。
- User-level driver / named pipeline:
  wafer-compile `search|none`；用户不手工拼接target passes，也不选择内部TargetCall或module materialization。
- Explicit non-goals:
  不重新做physical-dataflow mapping；不从symbol、文件名、vector ordinal、pid或launch position恢复物理身份；
  不把低层module dispatch提升为公开ABI；不提供 target metadata或entry ABI兼容读取路径。
- Completion gate:
  16 个 Tile interfaces all-and-only、物理三元组唯一且关系一致；每个 target module 的 current
  metadata/ABI/exports/digest fresh readback；任一 Tile 失败时无部分 output 可见；Q53 source→package/no-card
  重放通过，并在真实板端gate完成前保持`board-ready`而非`done`。
```

## 2. 稳定对象与身份

### 2.1 CardExecutable 的 Tile entry

`CardExecutable`原子拥有当前卡all-and-only 16个Tile entries；当前实现中的
`TileExecutable`只是单个entry的C++类型索引。每个entry包含：

- `CardId`：当前单卡为 `card_id=0`；
- `TileId`：来自 verified physical topology；
- `LaunchSlotId`：runtime 的 canonical submission order；
- accepted entry symbol 与 typed program bindings；
- `ReturnAfterLocalDrain` entry-local completion；
- `None` 或 `DirectDTE` transport contract；
- 经过 verification 的 Tile-local Instr module。

三个 ID 不互相推导。`launch_slot` 必须唯一、dense、可排序，但不要求等于 `tile_id`。任何 producer、aggregate
materializer、JIT bridge、runtime 或 diagnostic 都必须转发 typed fields，而不是使用容器位置重建它们。

没有单Tile production output，也没有把`num_partitions`当作Tile count的入口；`num_partitions`仍属于
GSPMD的card-level domain。当前实现类`CardExecutable`必须收敛为`CardExecutable`的实现或迁移索引，不能继续定义
一层长期output。

### 2.2 Target LLVM module

每个accepted Tile只翻译一次，结果由`TargetLLVMModule`连同其`LLVMContext`所有。下游target writing、
TargetCall frontend和model必须共享这组owner-backed modules，不得重新lower accepted IR。当前实现类
`TargetLLVMModules`只作为该owner set的代码索引，不是稳定output名称。

current target LLVM module通过typed metadata精确绑定：

- `wafer.target.card_id`；
- `wafer.target.tile_id`；
- `wafer.target.launch_slot`；
- entry symbol；
- target identity、current Kernel Runtime ABI 与 module format；
- dense typed Kernel ABI slot rows。

metadata readback必须与 C++ typed owner逐项相等。缺字段、重复字段、未知字段、错误 target triple、错误 entry
type 或 slot mismatch 均在 writing 前失败。

## 3. Target conversion 责任

### 3.1 ABI preparation

ABI preparation只消费 final accepted Instr IR 和 program boundary bindings，生成 dense、zero-based
`KernelABISlot[]`。每个 slot 显式记录 role、resource index、dtype、physical layout、shape、byte size 与 alignment。

稳定规则：

- program input、parameter、constant、output 对应 card-scoped package resources；
- accepted Tile entry在ABI preparation前精确保留frontend的全部真实arguments和results；CardModule内部使用过的
  scheduling destination已被消费，不能作为额外argument到达本层；
- compiler workspace 与 Direct-DTE status 对应 Tile-scoped resources；
- output 是 caller-owned append-only ABI slot，不通过隐藏返回 buffer 或 symbol 约定发布；
- workspace high-water 与 alignment 从同一个 final physical memory plan重算；
- 地址、count、stride、iteration、enum 和 packet bounds 在 target boundary 窄化，overflow fail closed；
- target call descriptor 是唯一 field-position 与 scalar-width 事实源。

ABI preparation不得改变 selected mapping、temporal tile、fusion、movement、worker、completion 或 placement；
late failure拒绝整个`CardExecutable` candidate，由Q51 candidate set选择其它候选；Q49 baseline则返回明确失败诊断。

### 3.2 Instr 到 TargetCall

Target lowering把 typed Instr 转成 current closed `TargetCallDescriptor` registry中的调用。consumer只能通过
`TargetCallSemantic`、descriptor和typed decoder恢复 transaction；不得解析 symbol spelling。

Direct-DTE begin/send/issue/receive/wait/finish、NCC join以及各 compute/movement family都遵守同一规则：

- descriptor决定参数位置、宽度、result type和issue domain；
- decode context只提供合法 target-domain facts；
- worker/completion behavior来自 typed registry或 current Instr，不由函数名推断；
- unsupported dtype、layout、geometry或字段范围在 conversion/validation失败，不生成 fallback call。

TargetCall/CRT 是 current target ABI，不是 search IR，也不能把 target transaction倒灌到 structured层。

### 3.3 Structure 与 completion

conversion保留 source control-flow、SSA/effect与明确的异步 completion关系。entry 的
`ReturnAfterLocalDrain` 表示：该 Tile entry 返回前，所有本地发起且影响其可观察结果、resource reuse 或
transport status 的工作已由 current Instr/TargetCall completion chain收敛。

它不表示整卡 barrier。card-scoped completion由下游同时观察16个Tile entry和transport obligations；target
conversion不得新增“最终统一等待”来掩盖缺失的 Tile-local completion。

## 4. Module topology 与 writing

### 4.1 保留显式 Tile interfaces

与`CardExecutable`绑定的target writing view包含：

- verified module records；
- exactly 16 个 `VerifiedTargetTileInterface`；
- 每个 interface 的显式 `(card_id, tile_id, launch_slot)`、module ID 与 typed ABI slots；
- card-level ExecutionConfig与RuntimeLaunchContract。

当前实现类`LinkedTargetModules`记录 linker 写出并校验过的 modules，但不能成为`CardExecutable`与
`ExecutablePackage`之间的第二个长期output事实源。

module topology可以按 runtime launch contract使用不同低层表示：Grid/Cluster允许将 16 个不同 Tile body
materialize到一个 aggregate target module，也允许每个 Tile独立 module。无论选择哪一种，长期 output合同始终是
16 个显式 Tile interfaces；module count不等于 Tile count，module path也不拥有 Tile身份。

aggregate materialization只是一项 target-lowering实现：

- 必须保留每个 Tile body的差异；
- dispatch必须使用显式 launch-slot→Tile-interface关系；
- 不允许用 `pid == launch_slot`、`launch_slot == tile_id` 或 source partition编号作为协议；
- host JIT中的 `wafer_target_call_dispatch` 仅是把 final target calls转成 typed transactions 的内部桥，
  不是公开 runtime ABI、package field或兼容入口。

### 4.2 原子发布

writing在私有 staging root中完成 LLVM IR、object、CRT、device link、ELF 和 readback。只有以下条件全部成立
才一次性发布 target root：

1. 16 个 Tile interfaces完整、唯一，三元组与 available topology一致；
2. target identity、runtime ABI、module format与所有 LLVM metadata一致；
3. entry exports按 launch phases闭合，且每个 interface能解析到合法module/export；
4. typed ABI slots与target LLVM entry signature一致；
5. all-and-only linked payload通过format、symbol、undefined allowlist和digest readback；
6. 无未引用module、临时文件或部分输出泄漏。

失败时删除本次 staging transaction；不从已有output directory恢复语义，也不保留旧格式副本。

## 5. Profile-only target writing

profiling以同一次 accepted final output为事实源。instrumented capture module可以是额外内部writing，但：

- ordinary production package只编译一次；
- site identity从 typed target-call ordinal、SSA identity和occurrence派生；
- profile capture不得改变普通 package的mapping、ABI slots、module digest关系或 completion；
- profile instrumentation仍使用显式物理三元组，并校验它与 production manifest逐 Tile一致；
- profile不存在时普通执行不受影响，存在但stale/malformed时fail closed。

## 6. Verification

Host gates至少覆盖：

- typed target LLVM metadata roundtrip与未知/缺失字段拒绝；
- non-identity `tile_id`/`launch_slot` mapping，包含 aggregate与非aggregate writing；
- all-and-only 16 Tile interfaces、duplicate/unavailable Tile、duplicate/missing launch slot负例；
- ABI slot role/layout/size/alignment/signature双射；
- unsupported target call、geometry、dtype、overflow、undefined symbol与digest mismatch负例；
- transaction staging的原子失败；
- 同一组owner-backed target LLVM modules被`ExecutablePackage` assembly、TargetCall frontend和SystemC直接消费，
  无第二次lowering。

Q53完成还必须由current source重新生成generic DAG、HF prefill/decode与Llama package，fresh no-card后达到
`board-ready`；真实设备上 Llama 和一个 prefill/decode代表做同源 matched A/B、exact output/guard并获得可重复改善后
才能标 `done`。历史 target/module通过记录不能代签这一门禁。
