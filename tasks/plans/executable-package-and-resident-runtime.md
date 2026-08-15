# ExecutablePackage 数据闭合与设备驻留执行实施计划

设计合同由`tasks/14-target-code-generation.md`、`tasks/15-launch-runtime-package.md`和
`tasks/16-verification-contract.md`拥有，状态只看`tasks/progress.md`。本计划拆解施工顺序，不建立第二份package/runtime
总体设计，也不把尚未实现的目标写成current合同。

状态：Q56 `doing`；Q57 `later`，只在Q53按Q56产生的新current package达到`board-ready`后启动。

## 1. 拆分依据

当前`CardExecutable`已经原子闭合selected整卡执行语义，Q51–Q53继续拥有physical-dataflow search、scalability和
production readiness。runtime/package问题分为两个依赖不同的边界：

1. compiler在`CardExecutable`之后丢失selected physical version、package initializer和allocation/view lifetime，且package复制
   source tree却仍要求caller逐次绑定parameter/constant；这是Q53生成最终package前必须替换的静态序列化问题。Q56只定义
   current transaction payload的consumer seam，source backing owner和去整树复制由后续Q58替换。
2. module、immutable data、device buffer和submission是否跨调用驻留，依赖真实provider/firmware的重复launch、cache
   visibility、terminal completion和failure lifecycle；它不应打断Q51–Q53搜索闭环，也不能凭HPGR存在handle就提前宣称。

因此Q56先闭合静态文件与one-shot consumer，Q57随后只增加device-local lifetime。两项均不改变placement、layout选择、
movement、buffering、event schedule或final Instr；这些事实继续只由06和Q51拥有。

## 2. Q56：ExecutablePackage 数据闭合

```text
Pipeline position:
- Upstream IR / input:
  final verified `CardExecutable`、同一transaction的target writing view，以及current transaction可解析并验证的
  parameter/external captured-constant payload bindings。Q56定义payload consumer seam，但不要求Q58尚未实现的source backing owner。
- Current stage responsibility:
  把final program boundary、physical ABI和payload投影为唯一current package合同；区分调用端口、package-owned immutable
  data与compiler-planned internal storage，并分开package allocation descriptor、package view、logical program tensor和selected
  physical tensor；原位替换writer、loader、one-shot runtime、target model、CLI和fixtures。
- Output IR / files:
  一个move-only `ExecutablePackage`，拥有canonical package root、verified manifest/member views，只含all-and-only referenced
  target modules和digest-bound target-ready data；compiler writer与runtime loader返回同一语义对象，再形成one-shot invocation plan。
- Downstream consumer:
  Q49.P package稳定性检查、Q53 production package/no-card、wafer-run one-shot board、profile collection和Q57 resident loader。
- User-level driver / named pipeline:
  wafer-compile `search|none`写package；wafer-run `--no-card|--board`消费同一contract。
- Explicit non-goals:
  不增加loaded executable、跨调用device residency、mutable state、input/output alias、bounded dynamic ABI、multi-entry package、
  multi-inflight、async copy、cancel、persistent device loop或跨卡执行。
- Done criteria:
  source tree不再作为runtime package成员；module有exact path/digest/size，data image另有alignment且其segment有checked
  range/layout，root无未引用成员；parameter/constant不再是per-invocation caller binding；logical bytes与physical span分别验证；
  每次compile只对每个package-initialized parameter/constant selected target physical version做一次transform，每个package product只写
  一个对应segment projection；external input/output physical tensors不做compile-time transform且不写segment；
  materialization以bounded window增量计算digest，
  不构造按总参数量或Tile数线性增长的host临时对象；全部producer、
  consumer和negative fixtures同步切换且无旧reader。fresh source→package→readback/no-card、target model、fake-provider one-shot
  和完整板端case均准备完成后标`board-ready`；真实板端通过前不标`done`。
```

### Q56 checkpoints

1. **当前事实映射**：列出`ProgramResourceBinding -> KernelABISlot -> PackageResourceRecord -> RuntimeInvocationBinding`
   全链，固定每个字段的唯一owner和被当前manifest丢失的layout/backing/init/lifetime事实。
2. **target physical version接合**：target ABI preparation对16个Tile parameter/external captured-constant slots做一次all-and-only
   join，形成logical binding/view→selected target physical version的typed relation；一个logical view可以因selected physical
   representation不同产生多个显式version，只有同一version identity才能共享target bytes。Q56的consumer可暂时从current
   transaction locator读取payload，Q58再在不改变该target relation的前提下替换为owner-backed source view。
3. **bounded data materialization**：compiler按final physical descriptor只对每个package-initialized parameter/constant selected target
   physical version做一次窗口化转换，
   增量形成target-ready bytes并计算digest；每个ordinary/profile package product各自只投影一个受检segment，可复用同一
   materialization owner或文件块但不得重复transform。不保留整tensor `RawLogicalValue[]`、整模型byte vector或per-Tile
   physical copy。external input/output physical tensors只有descriptor/allocation/view，transform和segment计数为零。profile
   production/count/trace复用同一已验证data materialization，不重复transform。runtime只读取、校验和copy，不解析NPY、不重新shard或pack。
4. **静态schema与owner替换**：以closed typed records分别表达external call ports、logical program tensors、selected physical
   tensors、immutable data、internal storage及allocation/view关系；slot只引用physical tensor或internal buffer，storage
   sharing/subview只由allocation/view identity表达，不从role、name、shape或path恢复。Q56冻结每个initialized physical tensor
   独占allocation/full-span view/initializer/segment，暂不做跨tensor packing；同时把
   compiler-only `VerifiedPackage`与runtime-only `VerifiedPackageManifest`收敛为一个move-only `ExecutablePackage` owner，避免
   通过裸root path重新打开已经验证的成员。
5. **consumer同步**：更新package writer/readback、one-shot planner/executor、profile、TargetModel invocation、wafer-run文件适配层
   和current fixtures；CLI raw vector只保留在文件adapter，不进入核心binding。
6. **验证**：strict JSON/tree正负例、range/overflow/layout/alias/init负例、deterministic package、no-card零provider effect、
   fake-provider allocate/copy/load/submit/cleanup顺序及FP16/BF16 board-ready case；额外记录logical binding/view、selected physical
   version、segment/allocation、transform/write/H2D计数、physical/package bytes、alignment overhead以及Q56 consumer seam到
   target/package阶段的peak RSS和最大live window，并以该seam直接提供的大型完整参数inventory证明它们不按Tile数或profile
   capture数放大。verified source→SPMD→handoff的read/copy/RSS/disk闭环由后置Q58拥有，不作为Q56完成前置。

## 3. Q57：设备驻留的静态执行

```text
Pipeline position:
- Upstream IR / input:
  Q53按Q56合同生成并验证的single-card static `ExecutablePackage`，以及qualified TX device/provider capability。
- Current stage responsibility:
  将静态package装载为完整whole-card device-local executable；显式拥有kernel module set或model graph与immutable data lease，提供受检
  device buffer/view binding和有identity的submission/completion，使多次调用共享装载态而不改变编译语义。
- Output IR / files:
  不产生新IR或新磁盘package；产生move-only owning device、loaded executable、device buffer/view和submission runtime对象，
  one-shot API只组合同一路径。
- Downstream consumer:
  普通重复执行、静态pipeline/recurrent workload adapter以及未来独立的external execution-engine adapter；provider仍只消费完整
  16-Tile launch，不暴露单Tile执行。
- User-level driver / named pipeline:
  runtime load/submit/query-or-wait/close；wafer-run继续提供同步便利入口。
- Explicit non-goals:
  不把request、batch、prefix/page policy、tokenizer或scheduler放入runtime；不引入runtime-owned application state、动态图/JIT、
  multi-entry package、multi-inflight承诺、cancel、persistent Kcore service loop或cross-card execution。
- Done criteria:
  current provider以typed capability声明single context、`max_inflight=1`、无cancel、kernel module set/model graph各自的
  loaded上限；显式submission持有全部buffer/module/internal lease到card terminal；load的unknown accepted subset、
  timeout或不可信completion使device failure domain sticky poisoned且不做破坏性cleanup。host/fake-provider与真实板端证明
  complete module set或model graph load一次执行多次、immutable data只H2D一次、只有已资格化的跨launch producer/consumer
  crossing可直接复用device buffer，workspace在下次read前完整定义或显式初始化，status每轮重新建立lifecycle，
  exact unload/refcount与one-shot parity；provider library的process lifetime不由此child owner暗示finalize。真实板端未通过前不标`done`。
```

### Q57 checkpoints

1. **owner与failure domain**：建立owning Wafer device failure domain替代只复用qualification的非owning session；device poison使全部
   child handle失效，但不暗示vendor provider library可被finalize/dlclose。normal close、poison quarantine与process-owned provider lifetime分开。
2. **loaded executable**：以closed variant原子拥有完整16-Tile kernel module set或model graph、resolved entry、immutable data lease和
   单submission internal slot；provider capability显式声明两种variant的loaded上限，当前model graph不默认允许第二个live graph；
   provider-specific Grid/Cluster/Model细节不进入应用API。
3. **buffer/view与submission**：64-bit checked base/offset/span、alignment、同一owning-device generation下owned或borrowed的
   Wafer `DeviceBuffer`及terminal lease；current TX allocation domain只开放provider `txMalloc`对应的device DDR，foreign pointer、
   userptr、IPC/external provider allocation import、host-pinned或SPM均非本项能力。
   `submit`返回带device generation的显式handle。第二个live submission必须在首个provider effect前返回typed busy/capacity，
   不能先触发provider contract failure再poison；pending/`NOT_READY`是正常query结果。provider phase completion与phase-state release
   只是内部里程碑，public success必须晚于全部phase、16 Tile completion、当轮Direct-DTE status验证及调用者请求的output visibility；
   stale handle/generation必须拒绝，不能查询到后续隐式current submission。
4. **resident identity**：loaded executable、buffer、submission及immutable reuse cache都是owning device/failure-domain generation的
   child；reuse key至少覆盖该generation、canonical package identity、module/data digest、physical descriptor和device qualification。
   poison、device重建或显式reset都会失效旧child，不能仅凭digest与相同qualification跨generation复用；cache/LRU只改变策略，
   不改变binding语义。
5. **资格**：逐项板测module/graph重复launch、每一种对外承诺的跨launch read-after-write/cache visibility crossing、
   workspace define-before-read、Direct-DTE status每轮reset、normal close和timeout quarantine；只有完成的crossing可进入capability，
   不用一个Add case推广一般coherence。多stream、多inflight或persistent loop若有后续需求，另建有板端证据的编号任务。

## 4. 共享约束

- 唯一磁盘产品名保持`ExecutablePackage`；不新增第二种磁盘产品名、泛容器类型或执行配置族。
- 一个package仍对应一个static `CardExecutable`和一个target/configuration；多个specialization由上层分别加载和选择。
- package中的`prepare/main`继续表示device launch phase；host-side load/preparation不得复用这两个名称。
- Q56不拥有跨调用device state：两步程序只能把普通output bytes返回caller，再作为下一次input提交。Q57也不建立
  runtime-owned application state，只允许external device buffer在逐项资格化的producer/consumer crossing上重新绑定。
  真正tied/inout同一对象只有在frontend/IR、alias/effect、DDR、ABI、package、model和runtime同步出现真实consumer时才另立任务；
  不得先放进workspace或native global。
- `max_inflight=1`是current provider capability而不是API identity；public submission始终显式带handle/generation，不能固化为
  可被后续调用覆盖的隐式current submission。未来bounded multi-inflight仍需独立板端任务，但不应迫使核心owner模型重写。
- IREE/PJRT/TileRT仅作为allocation/view、compiled/loaded executable、immutable residency和submission/completion边界参考；
  它们的VM、process-global state、CUDA graph、硬编码device数量和serving policy不进入Wafer合同。
