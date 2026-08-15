# ExecutablePackage 数据闭合与设备驻留执行实施计划

设计合同由`tasks/14-target-code-generation.md`、`tasks/15-launch-runtime-package.md`和
`tasks/16-verification-contract.md`拥有，状态只看`tasks/progress.md`。本计划拆解施工顺序，不建立第二份package/runtime
总体设计，也不把尚未实现的目标写成current合同。

状态：Q56 `doing`；Q57 `later`，只在Q53按Q56产生的新current package达到`board-ready`后启动。

## 1. 拆分依据

当前`CardExecutable`已经原子闭合selected整卡执行语义，Q51–Q53继续拥有physical-dataflow search、scalability和
production readiness。runtime/package问题分为两个依赖不同的边界：

1. compiler在`CardExecutable`之后丢失physical layout、payload backing、初始化和lifetime，且package复制source tree却仍要求
   caller逐次绑定parameter/constant；这是Q53生成最终package前必须替换的静态序列化问题。
2. module、immutable data、device buffer和submission是否跨调用驻留，依赖真实provider/firmware的重复launch、cache
   visibility、terminal completion和failure lifecycle；它不应打断Q51–Q53搜索闭环，也不能凭HPGR存在handle就提前宣称。

因此Q56先闭合静态文件与one-shot consumer，Q57随后只增加device-local lifetime。两项均不改变placement、layout选择、
movement、buffering、event schedule或final Instr；这些事实继续只由06和Q51拥有。

## 2. Q56：ExecutablePackage 数据闭合

```text
Pipeline position:
- Upstream IR / input:
  final verified `CardExecutable`、同一transaction的target writing view，以及frontend已经验证的parameter/constant payload。
- Current stage responsibility:
  把final program boundary、physical ABI和payload投影为唯一current package合同；区分调用端口、package-owned immutable
  data与compiler-planned internal storage，并分开storage root、checked view、logical tensor和physical descriptor；原位替换
  writer、loader、one-shot runtime、target model、CLI和fixtures。
- Output IR / files:
  一个`ExecutablePackage`，只含canonical manifest、all-and-only referenced target modules和digest-bound target-ready data；
  loader形成同一current verified package与one-shot invocation plan。
- Downstream consumer:
  Q49.P package稳定性检查、Q53 production package/no-card、wafer-run one-shot board、profile collection和Q57 resident loader。
- User-level driver / named pipeline:
  wafer-compile `search|none`写package；wafer-run `--no-card|--board`消费同一contract。
- Explicit non-goals:
  不增加loaded executable、跨调用device residency、mutable state、input/output alias、bounded dynamic ABI、multi-entry package、
  multi-inflight、async copy、cancel、persistent device loop或跨卡执行。
- Done criteria:
  source tree不再作为runtime package成员；module有exact path/digest/size，data image另有alignment且其segment有checked
  range/layout，root无未引用成员；parameter/constant不再是per-invocation caller binding；logical bytes与physical span分别验证；全部producer、
  consumer和negative fixtures同步切换且无旧reader。fresh source→package→readback/no-card、target model、fake-provider one-shot
  和完整板端case均准备完成后标`board-ready`；真实板端通过前不标`done`。
```

### Q56 checkpoints

1. **当前事实映射**：列出`ProgramResourceBinding -> KernelABISlot -> PackageResourceRecord -> RuntimeInvocationBinding`
   全链，固定每个字段的唯一owner和被当前manifest丢失的layout/backing/init/lifetime事实。
2. **静态schema替换**：以closed typed records表达external call ports、immutable data、internal storage及root/view关系；slot只引用
   typed binding/view，storage sharing/subview只由root identity表达，不从role、name、shape或path恢复。
3. **data materialization**：compiler按final physical descriptor写target-ready bytes；大payload允许有限文件内的checked ranges，
   文件digest是内容identity的唯一事实源，segment只记录并验证range。runtime只读取、校验和copy，不解析NPY、不重新shard或pack。
4. **consumer同步**：更新package writer/readback、one-shot planner/executor、profile、TargetModel invocation、wafer-run文件适配层
   和current fixtures；CLI raw vector只保留在文件adapter，不进入核心binding。
5. **验证**：strict JSON/tree正负例、range/overflow/layout/alias/init负例、deterministic package、no-card零provider effect、
   fake-provider allocate/copy/load/submit/cleanup顺序及FP16/BF16 board-ready case。

## 3. Q57：设备驻留的静态执行

```text
Pipeline position:
- Upstream IR / input:
  Q53按Q56合同生成并验证的single-card static `ExecutablePackage`，以及qualified TX device/provider capability。
- Current stage responsibility:
  将静态package装载为完整whole-card device-local executable；显式拥有module/graph与immutable data lease，提供受检
  device buffer/view binding和有identity的submission/completion，使多次调用共享装载态而不改变编译语义。
- Output IR / files:
  不产生新IR或新磁盘package；产生move-only owning device、loaded executable、device buffer/view和submission runtime对象，
  one-shot API只组合同一路径。
- Downstream consumer:
  普通重复执行、静态pipeline/recurrent workload adapter，以及未来独立的vLLM/SGLang integration；provider仍只消费完整
  16-Tile launch，不暴露单Tile执行。
- User-level driver / named pipeline:
  runtime load/submit/query-or-wait/close；wafer-run继续提供同步便利入口。
- Explicit non-goals:
  不把request、batch、prefix/page policy、tokenizer或scheduler放入runtime；不引入runtime-owned application state、动态图/JIT、
  multi-entry package、multi-inflight承诺、cancel、persistent Kcore service loop或cross-card execution。
- Done criteria:
  current provider以typed capability声明single context、`max_inflight=1`和无cancel；显式submission持有全部buffer/module/internal
  lease到card terminal；timeout使device failure domain sticky poisoned且不做破坏性cleanup。host/fake-provider与真实板端证明
  complete module set load一次执行多次、immutable data只H2D一次、device output可作为下一次input、workspace/status安全复用、
  exact unload/refcount和one-shot parity；真实板端未通过前不标`done`。
```

### Q57 checkpoints

1. **owner与failure domain**：用owning device替代只复用qualification的非owning session；device poison使全部child handle失效。
2. **loaded executable**：原子拥有完整16-Tile modules/graph、resolved entry、immutable data lease和单submission internal slot；
   provider-specific Grid/Cluster/Model细节不进入应用API。
3. **buffer/view与submission**：64-bit checked base/offset/span、memory domain、alignment、borrow/import ownership和terminal lease；
   `submit`返回显式handle，当前backend拒绝第二个live submission而不是覆盖`current submission`。
4. **resident identity**：immutable reuse key至少覆盖canonical package identity、module/data digest、physical descriptor和device
   qualification；cache/LRU只改变策略，不改变binding语义。
5. **资格**：逐项板测module/graph重复launch、跨launchread-after-write/cache visibility、normal close和timeout quarantine；
   多stream、多inflight或persistent loop若有后续需求，另建有板端证据的编号任务。

## 4. 共享约束

- 唯一磁盘产品名保持`ExecutablePackage`；不新增第二种磁盘产品名、泛容器类型或执行配置族。
- 一个package仍对应一个static `CardExecutable`和一个target/configuration；多个specialization由上层分别加载和选择。
- package中的`prepare/main`继续表示device launch phase；host-side load/preparation不得复用这两个名称。
- application mutable state在Q56/Q57中都通过显式buffer跨调用传递。只有未来frontend/IR、alias/effect、DDR、ABI、package、
  model和runtime同步出现真实inout consumer时，才建立独立任务；不得先放进workspace或native global。
- IREE/PJRT/TileRT仅作为allocation/view、compiled/loaded executable、immutable residency和submission/completion边界参考；
  它们的VM、process-global state、CUDA graph、硬编码device数量和serving policy不进入Wafer合同。
