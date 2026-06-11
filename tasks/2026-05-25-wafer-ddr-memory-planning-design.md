# Wafer DDR Memory Planning Design

日期：2026-05-25

状态：设计草案；2026-05-25 边界收口；2026-06-05 对齐 memref-backed Wafer memory attr 合同；
2026-06-11 收敛 R3.2g 为 DDR memory plan acceptance

本文定义 Wafer 编译器中 `#wafer.memory<ddr, layout>` 的资源建模、allocation policy、
verifier 和 lowering 责任。DDR 不是 SPM 文档里的边界注释，也不是另一套 memory-space 语义；
它是和 `#wafer.memory<spm, layout>` 同属 Wafer memory attr 的 addressable storage space，只是容量规模、
生命周期、runtime owner、BO pool、host visibility 和 movement 路径不同。

本文只讨论 device/global DDR 资源。Host CPU memory 在被 runtime import/pin 成 Wafer 可见
buffer object（下文简称 BO）之前，不属于 Wafer DDR address space。一旦它被 driver 表达成
remote/pinned BO，并可通过 Wafer runtime / DTE / memcpy path 访问，它在 Wafer IR 中仍使用
`#wafer.memory<ddr, layout>`，但带有不同 domain/pool allocation policy。

本文只负责 `#wafer.memory<ddr, layout>` 的 demand、allocation policy、resident allocation
contract、capacity/bandwidth/range planning 和 runtime allocation failure diagnostic。它不选择
group boundary、tile shape、physical layout、SPM offset、compute/comm algorithm 或 launch
package format；这些文档只能把 DDR feasibility 作为 cost/legality feedback 使用。

2026-06-11 的任务边界收敛为：R3.2g 不产出“等待后续接受”的候选 DDR plan。R3.2g 消费
R3.2f 之后的 instruction-level IR，并接受或拒绝当前 IR 已经显式表达的 DDR memory plan。
成功时，accepted DDR facts 就是 SSA use-def、`#wafer.memory<ddr, layout>` memref type、
`memref.subview` / strided view、`wafer.instr.*` descriptor、ownership/policy 和可重算
requirement；不额外复制一份 `wafer.ddr.plan` attr 或 side table。失败时返回结构化原因，供
closed-loop candidate driver 改 tile/layout/resource 候选后重跑 R3.2e-g。

## 1. 目标和非目标

目标：

- 统一 `#wafer.memory<spm, layout>` / `#wafer.memory<ddr, layout>` 语义，避免在 layout、SPM、runtime 文档中各自发明
  DDR 口径。
- 明确哪些 DDR buffer 由用户/runtime 绑定，哪些由 compiler/runtime 分配，哪些只是
  launch/package metadata。
- 给 group/layout/SPM planner 一个可重算的 DDR demand / bandwidth / range 输入，而不是只看
  SPM capacity。
- 给 `wafer.tile.load` / `wafer.tile.store`、D2D/P2P、constant load、launch ABI
  一个一致的 source/destination descriptor contract。
- 让 capacity、largest contiguous range、pool、alignment、read-only、host-visible、lifetime、
  alias 和 bandwidth pressure 都能进入 verifier 或 runtime allocation failure diagnostic。

非目标：

- 不把 DDR pool/domain 拆成新的 MLIR memory space。`NPU_NORMAL`、`VISIBLE`、`REMOTE_DRAM`
  这类是 allocation policy，不是 tensor semantic memory space。
- 不在 `wafer.group` 中保存 DDR allocation plan、BO address 或 runtime handle。
- 不重新发明普通 tensor constant、tensor layout 或 memref lowering 语义。
- 不把 host runtime 的某个旧 API 路径写成唯一 ABI。本文只固定 compiler-facing contract。
- 不在 V0 追求全局最优 memory planning；先保证可验证、可诊断、能回到 planner repair。
- 不把 R3.2g 的成功结果推迟给 R3.2h 再 accept；R3.2h 只负责候选枚举、重试或 split，不补
  DDR memory plan 语义。

### 1.1 R3.2g Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.2f memory-planned instruction-level `wafer.tile.region` / `wafer.instr.*` IR。SPM
  `memref.alloc` 已带 `wafer.spm.offset` accepted fact；DDR access 必须已经由
  `#wafer.memory<ddr, layout>` memref、`memref.subview` / strided view 和 RDMA/WDMA/DTE
  descriptor 显式表达。
- Current stage responsibility:
  从当前 IR 重算 DDR access demand、ownership/pool/domain policy、view range、descriptor
  expressibility、capacity、largest-contiguous-range、bandwidth 和 completion/fence demand，并接受或
  拒绝这份显式 DDR memory plan。
- Output artifact / IR:
  same instruction-level IR accepted as DDR-safe，或结构化 failure reason。成功路径不写
  `wafer.ddr.plan`、不写失败候选、不保存 search trace；必要的 requirement summary 必须能从当前 IR
  和目标 policy 重新生成。
- Downstream consumer:
  R3.2h closed-loop candidate driver 用 R3.2g 成功/失败作为候选 legality gate；R3.3/R3.4/R3.5
  只 materialize 已通过 R3.2g 的显式 DDR facts 到 committed tile-region、placed descriptor 和
  runtime allocation/import/package boundary。
- User-level driver / named pipeline:
  R3.2g 提供 `wafer-plan-ddr-memory` 局部 pass；主线验证入口必须提供从 R3.2c/R3.2d/R3.2f
  直接跑到 R3.2g 的 named pipeline，不能要求用户手动拼 pass 作为长期 compile flow。
- Explicit non-goals:
  不重新推 DDR tile subview，不重做 SPM memory planning，不选择 tile shape/layout/group boundary，
  不生成 ABI call、packet 或 runtime BO handle，不用名字或 whole-boundary memref 猜 tile access。
  若 compiler-managed / resident DDR fact 无法由当前 IR/interface 表达，R3.2g 必须结构化失败或先扩
  IR/interface，不能把协议藏到后续阶段。
- Completion gate:
  R3.2f 输出的 tiled matmul/elementwise/storeback 能通过 DDR descriptor/range/ownership/pool/domain/
  capacity/bandwidth acceptance；非法 view、descriptor payload/range、pool/domain、capacity、
  bandwidth 或 missing completion fence 能结构化失败。
```

## 2. 硬件和 Runtime 事实

DDR 设计必须吸收以下已经整理到 `docs/` 的事实。

### 2.1 容量和带宽

`docs/wafer-hardware-instruction-set-and-programming-model.md` 给出的卡级资源：

| 资源 | 已知事实 |
| --- | --- |
| 单卡 DDR 容量 | 64 GB 或 128 GB 两种规格 |
| 单卡 DDR 带宽 | 200 GB/s |
| PCIe | Gen4 x16 |
| RDMA / WDMA | DDR <-> SPM 的 NCC DMA component |
| DTE | tile/chip 间搬运和 DDR2DDR helper，受 DTE/FSM/channel 资源约束 |

这些数值不能只作为性能备注。DDR planner 至少要产出 static memory requirement 和 bandwidth
pressure summary；runtime launch 前必须能根据实际设备容量、free memory、largest contiguous
range 和 BO allocation result 给出失败诊断。

### 2.2 Address Space

已知 address surface：

| 区域 | 已知语义 |
| --- | --- |
| `DDR cached = 0x80000000` | Kcore SoC DDR mapping |
| `DDR weak-order uncached = 0x180000000..0x27fffffff` | uncached / weak-order DDR alias |
| `External DDR = 0x8000000000` | external DDR mapping |
| `DDR_LOWER_BOUND = 0x280000000` | public adapter DDR 合法边界，`addr >= DDR_LOWER_BOUND` |
| `BAR2 visible` | small-BAR 情况最多 32 MiB visible window，visible BO 地址会加 KMD device offset |
| `BAR4/ATU` | 覆盖 MHU、tile window、C2C、DDR controller、VPU、SYS_CTRL、log 等 aperture |

Compiler IR 不应直接硬编码某个 runtime 返回的 physical address，但 placed instruction-level IR /
access descriptor verifier 必须能检查：

- DDR access descriptor 是否位于 target policy 允许的 DDR address domain。
- RDMA source / WDMA destination 的 `#wafer.memory<ddr, *>` endpoint 是否满足 public adapter 或 selected ABI 的
  address lower bound。
- source/destination `end` range 是否落在所属 BO 或 compiler-managed slice 内。
- cached/uncached/external alias 不是新 memory space；它是 runtime/descriptor policy。

### 2.3 BO Domains and Pools

KMD / HPGR runtime 暴露的是 BO domain/pool，而不是一个平坦 device memory：

| 分类 | 已知项 | Compiler 语义 |
| --- | --- | --- |
| domain | `TSM_BO_LOCAL_DRAM` | device-side local DRAM，普通 Wafer DDR tensor / compiler-managed allocation 默认来源 |
| domain | `TSM_BO_REMOTE_DRAM` | host-side SG / pinned BO；import 后可以作为 runtime transfer / mapped BO endpoint |
| pool | `TSM_BO_POOL_NPU_NORMAL` | 普通 device allocation pool，bulk tensor、compiler-managed allocation、resident constant 默认候选 |
| pool | `TSM_BO_POOL_NPU_BIN` | Kcore/NPU binary storage；不是普通 tensor allocation |
| pool | `TSM_BO_POOL_VISIBLE` | BAR2 visible memory，small-BAR 可见窗口有限 |
| pool | `TSM_BO_POOL_VISIBLE_EXTENDED` | VF BAR4 visible/extended 或 large-BAR alias；是否 alias NPU_NORMAL 由设备决定 |
| pool | `TSM_BO_POOL_LOG` | log buffer pool，不作为 tensor allocation |

KMD 源码还暴露了部分固定地址行为：

- `NPU_BIN` 使用 `0x110000000..0x17cffffff`，`0x17d000000..0x17fffffff` 为 Score/AP firmware
  ownership。
- `VISIBLE` 使用 `0x00400000..0x007fffff`，通过 BAR2 visible device base 做地址转换。
- `LOG` 使用 `0x180000000..0x187ffffff`。
- `NPU_NORMAL` 从 AP/PCI config DDR base/size late initialize；small-BAR 情况前 1 GiB 有保留。
- `VISIBLE_EXTENDED` 在 small-BAR / large-BAR 下语义不同，可能是独立 VF BAR4-backed range，
  也可能 alias `NPU_NORMAL` 前部。

因此 `#wafer.memory<ddr, *>` allocation plan 必须记录 pool/domain allocation policy，不能只记录 bytes。

### 2.4 Alignment, Chunking, and Range Facts

当前可用事实：

- BO creation 接受 requested size、physical alignment、domain、pool、flags。
- BO query 返回 physical address、size、device id、domain、pool。
- `tsm_device_memory_info` 区分 total local DRAM、remote SG/pinned usage、NPU binary pool usage、
  NPU normal usage、largest contiguous normal range、visible-extended usage。
- BO allocation path 使用 `SZ_16K` 线索；host address 非 4 KiB aligned 会触发 warning。
- Driver DTE transfer length round 到 8-byte multiple。
- DDR2DDR Direct DTE preferred read/write address alignment 是 256B；这是性能/路径约束，不是所有
  DDR BO 的通用 verifier hard rule。
- RDMA/WDMA descriptor 需要 source/destination address、element count、byte stride、iteration
  和 inclusive `src_end` / `dst_end`。
- NCC parallel mode 的 busytable 会看 DDR range；RDMA/WDMA DDR 端地址和 in-flight DMA range
  overlap 会影响 ready。
- 旧 runtime D2D/P2P 路径有 4 KiB chunking 和多 DTE lane 线索；它属于 host/runtime D2D path，
  不等于 compiler inline Direct DTE 的唯一 lowering。

### 2.5 Local Tile Heap Layout

`docs/tx8-deps-reverse-engineering/tx8-deps-reverse-engineering-reference.md` 中的
`CONFIG_TX81_MEMORY_LAYOUT` 线索说明，Kcore/runtime 侧还存在 local tile DDR heap address
layout：

- default / layout 1/2：`TILE_HEAP_ADDRESS_BASE = 0xE00000000`，tile stride `0x40000000`，
  per-tile heap size `0x2000000`，即 32 MiB。
- layout 6/7/8/9/10/11/12/13 等提供 256 MiB per-tile heap 形态，base 和 stride 随 layout id
  变化。
- layout 14 以 `0x110000000` 为 base，tile stride 为 0，heap size `0x7000000`。

这些 heap 说明 DDR 还有 tile-local/runtime policy 维度。V0 不默认把 local tile heap 当成普通
compiler-managed DDR allocation；它必须作为 target policy / runtime capability 输入。只有 launch/runtime
ABI 明确把某类 tensor temporary 分配到 local tile heap 时，DDR planner 才能把它纳入
pool/domain alternatives，并在 verifier 中检查 per-tile heap size、tile mapping 和 address formula。

## 3. IR 层和边界

DDR 相关事实按 IR 层分布：

| 层 | 表达 | 不表达 |
| --- | --- | --- |
| tensor / linalg / group | tensor shape、dtype、semantic layout、group boundary | DDR pool、physical address、BO handle、runtime allocation |
| `wafer.tile.region` | `#wafer.memory<ddr, layout>` / `#wafer.memory<spm, layout>`、load/store boundary、layout materialization、movement/effect | raw BO address、driver handle、unaccepted allocation trace |
| accepted layout / buffer layer | `memref<..., #wafer.memory<ddr/spm, layout>>` | host malloc pointer、runtime-private pool internals |
| DDR memory plan acceptance | 从 instruction-level IR 重算的 `DdrBufferDemand`、external allocation contract、resident/compiler-managed demand、pool/domain alternatives、lifetime、bandwidth、range/fence legality；输出同一 IR accepted 或结构化失败 | tensor math semantics、SPM offset search、重复 DDR plan attr、失败候选或 search trace |
| placed instruction-level IR | placed memref、access descriptor、compiler-managed base+offset、movement ops | unresolved unplaced Wafer-tagged memref |
| launch / package / runtime | BO allocation/import/query、constant serialization、physical address query/assignment、completion/fence | group formation 或 layout search 的内部 trace |

`wafer.group` 可以依赖 DDR feasibility 结论，但不携带 DDR plan。DDR plan 要么进入更低层 IR
的 explicit allocation / descriptor / memref value，要么作为 launch/package metadata；不能作为
`wafer.group` attr 或 shadow side table 存活。

## 4. DDR Buffer Classes

V0 至少区分以下 buffer class。class 是 demand / owner / verifier 分类，不是 memory space。

| class | owner | 默认 pool/domain | lifetime |
| --- | --- | --- | --- |
| external input | user / runtime allocation | host-visible or local BO allocation policy | launch call 提供；read-only unless explicitly mutable |
| external output | user / runtime allocation | host-visible or local BO allocation policy | launch call 提供；writeback 后 host-visible |
| resident / streaming constant | runtime cache / compiler-selected constant source | `NPU_NORMAL` read-only BO，或 per-use staging/streaming load | executable/module or use lifetime |
| compiler-managed DDR region | compiler plan + runtime BO alloc | `NPU_NORMAL` | launch 内按 interval suballocate，可复用 |
| inter-group tensor | compiler-managed slice 或 explicit BO | `NPU_NORMAL` | producer 到 last consumer |
| DDR staging | runtime/package/communication policy | `NPU_NORMAL` or visible pool if CPU-visible required | bounded by transfer / launch phase |
| control/completion metadata | runtime | `VISIBLE` / `VISIBLE_EXTENDED` | runtime-owned |
| binary/log/firmware | runtime/package | `NPU_BIN` / `LOG` | 不归 tensor DDR allocator |

V0 默认策略：

- Host-visible dynamic input/output 通常是 compact tensor boundary。它们可以由用户直接提供
  host-visible BO，也可以由 runtime 负责 H2D/D2H staging；compiler 只依赖
  `#wafer.memory<ddr, tensor>` access descriptor
  contract，不假设 host 原始指针就是 device DDR。
- Weight / constant 先是 `ConstantLike` tensor value。若需要目标相关排布，由显式 constant
  storage transform / `wafer.tile.load` lowering 直接生成 packed backing data 或 resident DDR
  constant；不要把 weight 特殊写成另一种 physical layout。
- Temporary、inter-group tensor 和 layout/communication staging 优先进入 compiler-managed allocation
  region，由 compiler 在 BO 内 suballocate offset，以便 verifier 能证明 range、lifetime 和 alias。
- `NPU_BIN`、`LOG` 默认不参与普通 tensor allocation。它们可以进入 launch/package resource
  summary，但不作为 generic tensor pool。

## 5. Demand Model

DDR memory plan acceptance 的输入不是裸 size，而是从当前 IR 和 target policy 派生的 demand。下面的
`DdrBufferDemand` 是 pass 内可重算 analysis，不是跨阶段 IR artifact：

```text
DdrBufferDemand {
  id
  kind                  // input, output, constant, temporary, staging, metadata
  owner                 // caller, compiler_constant, compiler_allocation, runtime
  logical_shape
  dtype
  layout_marker
  storage_size
  required_alignment
  preferred_alignment
  wafer_memory_attr     // #wafer.memory<ddr, layout_marker>
  domain_alternatives   // local_dram, remote_dram/imported
  pool_alternatives     // npu_normal, visible, visible_extended, ...
  flags                 // read_only, host_visible, cpu_access, external_visible
  lifetime
  alias_group
  bandwidth_class       // rdma_load, wdma_store, d2d, p2p, constant_prefetch, host_copy
  access_ranges
  tiling_key            // optional: tile shape / slice policy that produced this demand
  constant_slice_key    // optional: source constant + logical slice/chunk + selected layout_marker
}
```

来源：

- function / launch boundary：external input/output allocation demand。
- `wafer.tile.load`：DDR source read range、layout、stride、bandwidth demand；如果 source 是
  `ConstantLike`，它仍产生 read-only constant demand。
- `wafer.tile.store`：DDR destination write range、layout、stride、writeback visibility。
- `arith.constant` / `ConstantLike` lowering：constant size、storage transform、read-only residency。
- inter-group value escape：compiler-managed allocation or explicit output demand。
- `wafer.tile.*` communication ops / DTE op：D2D/P2P/DDR2DDR source/destination demand 和 byte count。
- control/completion op：small visible/control BO demand if runtime ABI requires it。

如果一个 demand 只能通过 name 或示例 case 恢复，说明 IR contract 不够，应该扩 op/type/interface，
不能让 DDR planner 猜。

当前 R3.2g 的最低实现边界应从 `wafer.instr.rdma` / `wafer.instr.wdma` 和后续 DTE/communication
instruction 形态恢复 DDR demand；`wafer.tile.load` / `wafer.tile.store` 只作为更高层 producer。
完整的 `DdrBufferDemand`、pool/domain、lifetime、range 和 bandwidth summary 必须在 R3.2g 内由当前
instruction-level IR 重算。若某个 demand 只能依赖旧 tile op、名字或示例 shape 推导，说明当前
instruction IR/interface 不足，应扩 IR/interface 或结构化失败。

## 6. Allocation Model

DDR allocation 分成 external allocation contract、compiler-managed allocation planning 和 runtime
realization 三个动作。

### 6.1 External Allocation Contract

External input/output 不由 compiler 静态分配。Compiler 生成 allocation contract：

- logical shape、dtype、semantic layout。
- expected `#wafer.memory<ddr, tensor>` compact external layout 或 explicit supported layout。
- byte size、stride、alignment。
- read/write intent。
- host-visible / device-local / imported domain policy。
- whether output may alias input。

Runtime 在 launch 时负责：

- import or allocate BO。
- query physical address、size、pool、domain。
- validate allocation contract。
- perform H2D/D2H staging if the provided host object is not directly Wafer-addressable。

Verifier / runtime validation 失败必须暴露为 launch-time diagnostic，不能 silent fallback 到错误 layout
或错误 pool。

当前 V0 compiler 用 DDR external allocation metadata 作为最小 external demand materialization。
R3.2g 必须从当前 instruction-level IR 的 DDR memref boundary、view chain 和 RDMA/WDMA descriptor
重算 input/output kind、compact tensor byte size、required alignment、read-only、host-visible
policy 和 access range，供后续 launch/runtime package 层 materialize。它不记录 DDR physical address、
BO handle 或 allocation/search trace；pool/domain policy、compiler-managed requirement 和 range
legality 必须在 R3.2g acceptance 中被检查或结构化失败，后续 runtime allocation 只 materialize
已接受的 requirement。

### 6.2 Constant Residency

Frontend 常量先统一到 `arith.constant` 或其它 `ConstantLike` tensor op。Constant storage transform /
load lowering 显式决定：

- fold to immediate / attribute / fill，无 DDR demand。
- raw compact backing data。
- transformed / packed backing data。
- launch-time resident DDR BO。
- per-use streaming load。

Constant value 本身不是 `#wafer.memory<ddr, *>` runtime buffer。只有当 constant 被加载或分配到 device BO 时，
才产生 `#wafer.memory<ddr, layout>` memref / access descriptor / resident constant demand。常量 residency demand 默认
read-only，可以跨 launch cache，但 cache key、eviction 和 sharing 属于 runtime policy。

constant 不是 group external input，也不由 launch caller 提供；但它不能被视为隐式 SPM resident。
每个 tile-region use 都必须经过 `wafer.tile.load` 或等价 load source，最终从 device-addressable
storage 读入 storage。因此 DDR planner 必须统计：

- constant full logical shape、dtype、element count 和 current backing data/resource。
- accepted tile shape、tile slice / access range、consumer indexing map 和 reuse count。
- load result layout marker，以及 raw compact backing data 是否能直接 lower 到该 layout。
- resident DDR BO、per-use staging/streaming load、或 compile-time transformed backing data 的候选。
- read-only lifetime、alignment、storage size、bandwidth class 和 pool/domain alternatives。

#### 6.2.1 Weight / Constant Slicing

Weight 切分由 consumer op 的 tiling relation 推出，不由 DDR planner 单独发明。DDR planner 看到的是
已经在 `wafer.tile.region` 中形成的 constant `wafer.tile.load` uses：

- logical slice：由 accepted tile shape、op indexing map / tiling interface 和 load indices 推导。
- storage chunk：为满足 layout、reuse、capacity 或 bandwidth 而选择的 backing data 粒度。

二者不能混在一起。合法策略是：

- whole resident：把整个 constant 按 selected storage order 放入 read-only DDR BO。适合复用高、
  package/DDR 容量可接受、load ranges 规则的 weight。
- chunk resident：按 tile slice 或多个 slice 的 union/coalesced chunk 生成 read-only backing data。
  适合整块 weight 太大、只有部分 slice 被当前 executable 使用、或 packed whole constant 会造成
  package/DDR 浪费的情况。
- streaming/staging：保持 raw compact backing data，按 use 把需要的 slice 搬到 staging DDR 或直接
  lower 成 raw load + `wafer.tile.materialize_layout`。适合复用低或 resident 失败的情况。

chunk resident 的 key 至少包含 source constant SSA value、logical slice/chunk shape、selected
layout marker、dtype 和 target policy。它只能覆盖已有 load slice 的 union/coalescing，不能改变
compute tiling；如果为了放下 weight 必须改变 K/internal split 或 tile shape，必须返回 group/op
tiling planner，而不是在 DDR planner 内部隐式改变 compute split。

如果原始 constant backing data 与 selected consumer layout 不兼容，合法 repair 只有三类：

- 在 accepted tile/layout 之后做 compile-time constant storage transform，生成与 `wafer.tile.load`
  result layout 兼容的 backing data，并把它作为 read-only constant demand 交给 DDR planner。
- 从 raw compact backing data load 到 compact storage，再插入 `wafer.tile.materialize_layout` 到 consumer
  需要的 layout；这会增加 SPM temp / movement / bandwidth cost。
- 若两种方案都因 SPM、DDR capacity、alignment、largest contiguous range 或 bandwidth 失败，返回
  group planner 调整 tile shape、constant residency policy 或 group boundary。

### 6.3 Compiler-Managed Allocation Region

V0 推荐对 compiler-managed temporary / inter-group / staging 使用 compiler-managed BO：

```text
pool/domain group
  -> one or more compiler-managed BO demand
  -> static suballocation inside each BO
  -> access descriptor = base + offset + size/layout/range
```

这样做的原因：

- Runtime BO allocator 负责找到一个真实 contiguous BO；compiler 只要求 allocation region 大小和 alignment。
- Compiler 可以在 compiler-managed BO 内做 interval allocation，证明互不重叠的 intervals 复用 offset。
- BO 数量少，launch metadata 和 runtime allocation overhead 更低。
- 如果 `largest_contiguous_normal_range` 不够，失败能明确指向 required contiguous region size，而不是一堆小 BO
  的偶然失败。

V0 compiler-managed DDR suballocator 使用 deterministic first-fit：

1. 收集 compiler-owned `DdrBufferDemand`。
2. 按 pool/domain/visibility/read-only 分组。
3. 建 lifetime interval。external-visible 或跨 launch resident 的 demand 不参与普通复用。
4. must-alias 合并；must-not-alias 检查。
5. 按 size、lifetime、alignment、bandwidth-critical 排序。
6. 在 compiler-managed BO address range 内 first-fit offset placement。
7. 计算 peak size 和 required BO alignment。
8. 生成 compiler-managed BO demand，交给 runtime BO allocation。

如果 target runtime 支持 fixed-address import 或低层 allocator 可控 placement，后续可以把 compiler-managed
BO 拆成多个 pool-specific BO。但这不是 V0 必须条件。

R3.2g 的 acceptance 口径是：只接受当前 IR/interface 已经能表达和验证的 compiler-managed DDR
facts。若 inter-group/staging value 只有抽象 `memref.alloc`，但没有可验证 owner、lifetime、
pool/domain 和 suballocation requirement，R3.2g V0 不能假装成功；应返回
`unsupported_compiler_managed_ddr` 或先扩 IR/interface。若 requirement 已可表达，R3.2g 可以接受
“compiler-managed BO demand + slice/range requirement”，实际 BO physical address 和 runtime handle
仍由 R3.5/runtime materialize。

### 6.4 Runtime BO Allocation

Launch/runtime layer 负责实际 BO allocation / import / query：

```text
DDR compiler-managed allocation
DDR external import
DDR access descriptor query
DDR allocation release
```

上面是语义草图，不要求立即固定 op 名。实现可以选择 custom runtime op、`memref` memory space
conversion，或 `wafer.launch` metadata lowering；关键是：

- allocation / import / query / free 是显式 runtime boundary，不是 group attr。
- placed movement op 消费的是带 `#wafer.memory<ddr, layout>` 的 memref 或 access descriptor。
- pool/domain/flags 是 allocation policy，不改变 tensor math semantics。
- runtime allocation failure 是合法的动态失败路径，不能被 compiler 伪装成静态成功。

## 7. Capacity and Legality

DDR feasibility 分两级。

### 7.1 Compile-Time Requirement

Compiler 必须给每个 executable / launch 产出 requirement summary：

```text
DdrRequirement {
  external_input_bytes
  external_output_bytes
  resident_constant_bytes_by_pool
  compiler_managed_bytes_by_pool
  visible_control_bytes
  peak_live_ddr_bytes
  largest_compiler_managed_bo
  required_alignment
  bandwidth_summary
}
```

这不是 IR 语义源，而是 launch/package metadata 和 diagnostic。它必须能从当前 IR、
constant storage transform result、compiler-managed allocation result 和 launch ABI 重新生成。

### 7.2 Runtime Feasibility

Runtime 在具体设备上检查：

- card DDR capacity 是 64 GB 还是 128 GB。
- `TsmMemGetInfo` / device memory info 中 total/free/used。
- NPU normal pool free bytes 和 largest contiguous normal range。
- visible / visible-extended pool availability。
- NPU_BIN / LOG / firmware reservations。
- imported host/remote BO 是否满足 domain、alignment、size、read/write intent。
- compiler-managed BO allocation 是否成功；query 后 size/domain/pool 是否符合 plan。

如果失败，diagnostic 应说明：

- 哪个 pool/domain 不足。
- 是否是 total bytes 不足，还是 largest contiguous range 不足。
- 是 external allocation、constant residency、compiler-managed allocation 还是 visible/control metadata 失败。
- 可选 repair：减少 resident constant、改 streaming load、拆 executable、降低 DDR lifetime、
  禁用某些 overlap/double-buffer、调整 group boundary 或 layout materialization。

## 8. Bandwidth and Scheduling

DDR 不是只看容量。DDR planner / scheduler 至少要把以下事实输入 cost model：

- 单卡 DDR bandwidth 约 200 GB/s。
- RDMA / WDMA 共享 LSU、SPM banks、NoC 和 DDR。
- NCC parallel mode 下 RDMA/WDMA DDR endpoint range overlap 会进入 DMA busytable ready 判断。
- Direct DTE / DDR2DDR 有 DTE/FSM/channel、outstanding/burst 和 alignment preference。
- Host D2D/P2P runtime path 可能有 lane/chunking policy，与 inline Direct DTE 是不同 lowering。

V0 可以采用保守模型：

- 对每个 `#wafer.memory<ddr, *>` access 记录 `[base, end]` 或 symbolic BO slice range。
- 如果两个 in-flight RDMA/WDMA/DTE event 访问同一 BO slice 且 may-overlap，默认认为有 DDR
  range hazard，不能当作免费 overlap。
- 如果 ranges 不重叠，可以并行候选，但仍计入 shared DDR bandwidth pressure。
- 以 200 GB/s 作为 card-level optimistic upper bound；实际 queue blocking 由 PMU 的 RDMA/WDMA/DTE
  execute / blocking time 后续校准。
- 对 DDR2DDR / bulk D2D，256B alignment 作为 preferred alignment 和 cost hint；只有 selected
  ABI 明确要求时才升级为 hard legality。

## 9. Integration with Group / Layout / SPM

### 9.1 Group Planning

Group formation 不能只问 “SPM 放不放得下”。候选 group / tile plan 至少要跑：

```text
layout assignment
  -> SPM demand / allocation
  -> DDR demand summary
  -> movement bandwidth / range hazard estimate
  -> downstream memory planning
```

DDR 的全局 allocation 往往跨 group，因此不要求每个 group 内静态分配最终 BO。但如果一个 group
选择导致：

- external store/load 数量明显增加。
- inter-group tensor 必须落 compiler-managed DDR allocation。
- materialization cut 在 group boundary 重复出现。
- constant / activation residency 超过 compiler-managed allocation 或 bandwidth 预算。
- constant tile shape 导致 packed backing data 过大，或 repeated streaming load 使 DDR bandwidth
  不可接受。
- chunked constant residency 的 chunk 数量、对齐浪费或 package size 过大。

planner 必须能把这些作为 cost / legality feedback。全局 DDR allocation 失败时，可以回到 group
boundary、layout cut、streaming/residency policy 或 executable split，而不是让 runtime 才报一个
无上下文的 malloc 失败。

### 9.2 Layout Materialization

Wafer memory attr 同时携带 address space 和 physical layout marker：

- `#wafer.memory<ddr, tensor>`：常见 host-visible compact boundary。
- `#wafer.memory<ddr, cx/ncx>`：只有 constant storage transform、resident constant、或 explicit device-side
  packed storage 需要时才出现。
- `#wafer.memory<spm, cx/ncx>`：aligned-only compute/movement operand。

Layout materialization 会增加 DDR movement、SPM temp 或 DDR staging。Materialization cut 算法必须
把 DDR allocation pressure 和 repeated boundary conversion cost 作为输入。

### 9.3 SPM Bufferization

SPM allocator 只分配 `#wafer.memory<spm, *>` offset。它仍需要看见 `#wafer.memory<ddr, *>`：

- RDMA source / WDMA destination 的 DDR access descriptor。
- load/store event 对 SPM source/destination lifetime 的影响。
- DDR range hazard 和 bandwidth cost。
- host-visible output boundary 的 drain/wait requirement。

SPM allocation 不分配 DDR BO；DDR memory planner 不分配 SPM offset。二者通过同一套 memory effect、
lifetime event 和 movement op contract 对齐。

## 10. IR Contract Sketch

语法只是草图，真正需要固定的是 verifier/lowering contract。

### 10.1 Memory Space and Buffer Values

```mlir
// External DDR allocation is represented at launch/package resource metadata.
// Tile IR sees a ranked DDR memref boundary value plus memory-space/effect facts.
%arg0 = "builtin.unrealized_conversion_cast"()
    : () -> memref<64x256xf16, #wafer.memory<ddr, tensor>>

%tile = wafer.tile.load %arg0
    : memref<64x256xf16, #wafer.memory<ddr, tensor>>
   -> memref<64x256xf16, #wafer.memory<spm, tensor>>

%out = "builtin.unrealized_conversion_cast"()
    : () -> memref<64x64xf16, #wafer.memory<ddr, tensor>>
wafer.tile.store %out_tile, %out
    : memref<64x64xf16, #wafer.memory<spm, tensor>>
   -> memref<64x64xf16, #wafer.memory<ddr, tensor>>
```

DDR addressability 可以用 placed `memref<..., #wafer.memory<ddr, layout>>`、launch/package metadata
或 explicit access descriptor 表达同一语义。不能出现的是：在 layout 文档叫 external boundary，在 SPM 文档叫 runtime BO，
在 compute 文档又叫另一个 address kind，且 verifier 无法统一检查。

### 10.2 Runtime Boundary

```text
DDR compiler-managed allocation metadata:
- bytes: 268435456
- alignment: 16384
- domain: local_dram
- pool: npu_normal

DDR access descriptor facts:
- base / range / visibility
- pool / domain / residency
- launch allocation key
```

`ddr_domain` / `ddr_pool` 是 allocation policy attrs。它们不替代 `#wafer.memory<ddr, layout>`，也不应出现在
tensor/group 层。

## 11. Verifier

DDR verifier 至少检查：

- `#wafer.memory<ddr, layout>` value 只能由 launch arg、runtime allocation/import、constant load/lowering、
  compiler-managed slice 或合法 access descriptor conversion 产生。
- RDMA source 是 `#wafer.memory<ddr, *>`，destination 是 `#wafer.memory<spm, *>`；WDMA source 是
  `#wafer.memory<spm, *>`，destination 是 `#wafer.memory<ddr, *>`。
- D2D/P2P/DDR2DDR op 的 source/destination domain、byte count、alignment 和 DTE mode 满足
  selected ABI。
- access descriptor range 不越过所属 BO / compiler-managed slice。
- inclusive end address 与 dtype、layout、stride、iteration、bitpack 一致。
- external input/output 的 `#wafer.memory<ddr, tensor>` compact external layout 或 explicit accepted layout 与 launch contract
  一致。
- read-only constant 不被写。
- host-visible output 在 host observe 前有明确 WDMA/DTE/runtime completion fence。
- compiler-managed suballocation 的 may-reuse intervals 不重叠；must-alias / must-not-alias
  relation 与 use-def/effect 一致。
- pool/domain policy 满足 buffer class：普通 tensor allocation 不落 `NPU_BIN` / `LOG`，visible pool
  只用于 CPU-visible/control 需求或显式 policy。
- placed instruction-level IR 中不再有无法 lowering 的 abstract DDR boundary。

## 12. Lowering Responsibility

| 阶段 | 责任 |
| --- | --- |
| layout / compute / comm lowering | 生成 `#wafer.memory<ddr, *>` load/store/comm demand，保持 use-def 和 effect |
| DDR memory plan acceptance | 从 instruction-level IR 接受 external allocation contract、constant residency、compiler-managed requirement/suballocation、range/fence legality 和 requirement summary；失败时给结构化原因，不产出重复 plan attr |
| placement realization | 把 DDR Wafer-tagged memref 降成 placed memref、access descriptor 或 compiler-managed base+offset |
| runtime/package lowering | BO alloc/import/query/free、constant serialization/loading、launch metadata、fence |
| instruction lowering | RDMA/WDMA/DTE descriptor、byte stride、iteration、end range、direction validation |
| verifier | Wafer memory attr、pool/domain、range、alignment、lifetime、host visibility、completion |

Lowering 到 LLVM 时应尽量复用 MLIR `memref` lowering 能表达的部分。目标 runtime handle、
physical address query、BO pool/domain 和 descriptor struct 通过 target-specific conversion
进入 LLVM dialect / runtime call；不要让高层 `storage` 或 package-only metadata 直接漏到 LLVM。

## 13. Failure Feedback and Repairs

DDR planner 失败原因：

- `pool_capacity_overflow`
- `largest_contiguous_range_too_small`
- `visible_window_overflow`
- `alignment_unsatisfied`
- `external_allocation_incompatible`
- `constant_residency_too_large`
- `compiler_managed_peak_too_high`
- `range_overlap_hazard`
- `bandwidth_pressure_too_high`
- `unsupported_domain_or_pool`
- `missing_completion_fence`

repair 建议：

- 把 resident constant 改为 streaming load 或拆分 package。
- 拆 executable / launch。
- 移动 group boundary，缩短 inter-group tensor lifetime。
- 调整 layout materialization cut，减少 repeated DDR writeback / reload。
- 禁用或减少 DDR-side staging / double buffer。
- 允许 output/input alias 或 external output direct allocation。
- 改用 host/runtime D2D path，而不是 inline DTE，或反过来。
- 给用户返回需要更大 DDR / visible pool / contiguous range 的明确诊断。

R3.2g 不应把这些 repair trace 写进 IR。只有当前 IR 已显式表达且通过 acceptance 的 buffer、
descriptor、compiler-managed slice requirement 和 runtime boundary，才能在 R3.3/R3.4/R3.5
进入 committed IR、placed descriptor 或 package metadata。

## 14. Case Sketch

典型 case：一个 group 消费 host input、resident weight，产生两个 output，其中一个给 host，
另一个被下游 group 消费。

1. Launch allocation contract 产生 `%input_ddr` 和 `%out0_ddr` external contract。它们是
   `#wafer.memory<ddr, tensor>`，默认 compact external layout，但实际 BO 由 runtime import/allocate。
2. `%weight_const` 不是 group external input，但它在 tile-region 中仍通过 `wafer.tile.load`
   读入。Op tiling interface 从 consumer tile 推出 weight logical slice。Constant storage transform /
   load lowering 决定它使用 raw compact backing data、whole transformed backing data，还是按 load
   slice / coalesced chunk 生成 transformed backing data。若选择 resident constant，DDR planner
   创建 read-only resident `#wafer.memory<ddr, layout>` demand；若选择 streaming，DDR planner 统计每个 tile slice 的
   read range 和 bandwidth。
3. Layout planner 在 `wafer.tile.region` 中为 compute operand 插入 `wafer.tile.load`，source
   是 `%input_ddr` / `%weight_ddr`，destination 是 `#wafer.memory<spm, *>` memref。
4. Group output 0 通过 `wafer.tile.store` 写 `%out0_ddr`，需要 completion fence 后 host 才能观察。
5. Group output 1 没有 host-visible boundary，DDR planner 可以把它放入 compiler-managed slice，
   下游 group 再从该 slice load。
6. SPM allocation 只分配 tile-local input/psum/output/materialization temp；DDR planner 另外统计
   `%out1` compiler-managed lifetime、constant residency 和 load/store bandwidth。
7. Runtime launch 时分配 compiler-managed BO，import external BO，query physical address；placed
   instruction-level IR 只消费 placed memref、access descriptor 或 base+offset，不再携带抽象 plan。

这个 case 只说明 IR 流向，不是架构边界：

- 是否 resident weight、是否 chunked packed constant、是否 compiler-managed BO 合并，是 planner/runtime
  policy，不改变 compute tiling 语义。
- output 1 是否落 DDR 取决于跨 group lifetime 和 downstream use，不是多输出规则。
- tile shape、layout cut、SPM allocation 和 DDR compiler-managed peak 都是 planner analysis 结果。
- `#wafer.memory<ddr, layout>` 始终是统一 memory attr；pool/domain/visible/cache/import 是 allocation policy。

## 15. Open Questions

这些问题不能写成当前结论：

- HPGR 原生 BO allocation/import/query 的最终 MLIR op 形态。
- large-BAR / small-BAR 下 `VISIBLE_EXTENDED` 和 `NPU_NORMAL` alias 的 runtime capability query。
- local tile heap 32MB / 256MB 是否作为 compiler-managed DDR allocation region，还是仅由 Kcore runtime
  内部使用。
- Direct DTE DDR2DDR 和 host/runtime D2D/P2P path 的选择策略。
- DDR bandwidth cost model 的 PMU 校准方式。
- multi-card C2C / remote DRAM 与 `#wafer.memory<ddr, layout>` access descriptor 的完整 ABI。

当前设计只要求这些能力以 target policy / runtime capability 输入进入 planner，不提前变成
`wafer.group` 或 tensor-level IR 的固定语义。
