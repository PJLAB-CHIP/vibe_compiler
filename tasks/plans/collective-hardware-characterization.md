# Collective Hardware Characterization 实施计划

状态：第一批9组case与no-card gate已完成；2026-07-27板端AllGather 256B/4KiB/64KiB三组通过，
ReduceScatter 256B在Direct i8 add暴露未资格化numeric capability，其余五组未执行，因此本campaign仍未
完成。本文只组织case施工和验证；case定义、执行状态、原始
证据及最终compiler消费结论统一写入`docs/tx81-compiler-hardware-calibration.md`。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已验证的post-SPMD structured tensor program，以及Q32从同一语义程序产生的complete-rank
  scheduling frontier；现有raw Direct-DTE证据只作为transport资格，不代替collective算法证据。
- Current stage responsibility:
  通过compiler-private typed seam从已通过相同Instr/SPM/DDR、message matching、target ABI和
  package gate的actual clone中，按accepted Instr IR的DTE protocol phase选择AllGather
  Direct/Ring、ReduceScatter Direct/Ring和AllReduce Ring/Tree；冻结同源A/B、payload、
  完整数值、status/completion oracle和串行执行合同。current package没有独立guard resource，
  因此不声称canary/guard readback。
- Output artifact / IR:
  public boundary一致的成对verified package，以及不进入package协议的test-only
  collective-characterization report；报告逐rank记录实际direction/communication/phase/round/peer/
  payload slice/issue bytes/constant-loop multiplicity/executed bytes的完整message tuple；package、
  manifest/ELF和transport contract evidence由case archive持有。
- Downstream consumer:
  Q37硬件行为台账和后续Q9 cost calibration。板端correctness可收窄capability；只有可信device
  measurement basis、calibration与held-out均稳定时才影响cost ranking。
- User-level driver / named pipeline:
  production source-to-package pipeline继续产生默认winner；wafer-compile-test只在显式
  characterization case中选择已接受的collective alternative。统一hardware runner串行执行。
- Explicit non-goals:
  不新增公开算法CLI/IR attr/package字段，不把Direct-DTE称为Direct collective算法，不用host
  elapsed或ELF callsite数宣称性能，不重复已完成的raw DTE payload/sync/error/broadcast case，
  不把第一批矩阵称为硬件行为完备。
- Completion gate:
  第一批AG Direct/Ring、RS Direct/Ring、AR Ring/Tree在256B、4KiB、64KiB形成9组同源
  A/B case；每个请求由actual Instr phase证明，双包no-card和catalog/runner inventory通过，
  board case保持pending直到真实串行执行。最终状态和证据只进入统一校准文档。
```

## 第一批矩阵

- logical collective payload统一为`256B / 4096B / 65536B`，固定16 rank、i8 exact：
  AllGather每rank输入`B/16`、输出`B`；ReduceScatter每rank输入`B`、输出`B/16`；
  AllReduce每rank输入/输出`B`。
- payload使用rank/logical-lane/payload-size的确定性64-bit mixing后折叠为i8；mutation gate逐AG source
  chunk、reduction source contribution（RS按destination segment）与RS output slice检查1/32/256B
  rotation，并逐reduction source枚举missing及其余source replacement，避免线性mod-256短周期掩盖
  slice/tile错误。
- AllGather对比Direct all-peer与Ring `P-1`轮；ReduceScatter对比Direct owner exchange与Ring
  `P-1`轮；AllReduce对比Ring `2(P-1)`轮与ordered Tree reduce+broadcast。
- 现有`tree-all-reduce`的4KiB source/payload复用为AllReduce 4KiB点，不重复创建同义case。
- 每个variant必须从accepted Instr IR报告完整message tuple并跨rank一一匹配；特别是AllReduce Ring不得用
  可能由Auto回退Tree的baseline名称代签，Ring必须形成单一16-rank cycle，Tree的reduce/broadcast边必须
  exact reverse并保持rank-group inorder。
- 板端按payload从小到大、pair内A/B与B/A交替、至少3个paired repeat；任一timeout、untrusted
  terminal或设备异常立即停止，不retry/reset/power。

## 后续不重复family

第一批完成后，仍需按统一校准文档中的activation gate准备：AllToAll全交换、Permute cycle与稀疏角色、
collective双epoch复用、DTE route/fanin/fanout、single-engine slope、engine-pair stage balance、
RDMA→CT/NE→WDMA三阶段pipeline、worker placement/arbitration和DDR active-rank contention。SPM conflict
只先做matched conflict/control pilot；DDR bank大矩阵在缺少可控physical mapping或bank counter时停止，
保持`no-ddr-bank-coloring`，不以row数量冒充可消费证据。

## 验证

1. C++ focused test证明每种typed请求只接受逐rank实际phase family精确匹配的candidate；缺rank、同family
   混合或其它collective phase混入时fail closed。
2. Python catalog检查9组pair均绑定source、payload、结构/numeric/lifecycle oracle，且existing raw evidence不被
   重复登记。
3. 每个case双包compile并比较normalized manifest；逐条重放message tuple的跨rankmatching和Direct/Ring/Tree
   graph oracle，并归档characterization report、source、package、ELF、transport contract和digest。
4. CTest注册no-card与board case；board label持有同一resource lock，runner只在显式
   `collective-characterization`批次串行执行。
5. 真实板端结果写回统一校准文档同一row；runtime内部status-v2 enforcement与output可独立观察的raw status
   分栏记录，计划文件不保存动态结果。
