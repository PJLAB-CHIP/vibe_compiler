# PyTorch Source Board Verticals 实施计划

状态：Q44 `board-ready`。rank-one、`4096³` K-sharded GEMM/AllReduce和实际Llama-2 7B Megatron TP16
均已闭合真实PyTorch eager/export、production package与fresh no-card；Llama case的通用symbolic movement lowering、
16-rank row-pointer launch ABI及完整production compile也已通过。真实板端tensor capture尚未执行，不得标`done`。本计划
只拆解PyTorch source到board tensor correctness纵向；状态以`tasks/progress.md`为准，frontend和board长期合同
分别由02、16拥有。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  可在CPU eager执行的PyTorch module、由case tensor自身声明dtype的输入与parameter，以及钉死的source-built
  PyTorch/XLA exporter；sharded case还包含由torch_xla SPMD API表达的逻辑mesh/sharding。HuggingFace
  Transformer case消费Llama decoder block config及module显式声明的Megatron TP parameter/activation spec。
- Current stage responsibility:
  用同一份PyTorch tensor先执行eager形成expected，再经真实exporter形成StableHLO program
  directory，交给唯一production compiler形成verified package；runtime按manifest绑定raw输入，完整
  capture每个声明output并解码成torch.Tensor，最终只用torch.testing比较expected与actual。
- Output artifact / IR:
  verified frontend program、rank-local structured tensor program、verified runtime package、PyTorch eager
  expected、runtime output capture和可审计的torch comparator结果。
- Downstream consumer:
  no-card preflight、configured-board correctness、Q40/Q41 optimization qualification及需要高层tensor
  correctness的optimization campaign。
- User-level driver / named pipeline:
  importer Python执行test/Board/PyTorch下的统一runner；编译仍只经wafer-compile，执行仍只经wafer-run。
- Explicit non-goals:
  不以手写MLIR代签PyTorch source coverage；不以NumPy形成expected或最终comparison；不从case/file/
  parameter名恢复语义；不迁移instruction、ABI、layout、DMA、PMU、provider lifecycle或target raw-bit
  calibration的协议参考；no-card不代签真实板端。
- Completion gate:
  rank-one GEMM、16-rank `4096³` K-sharded GEMM/AllReduce和HuggingFace Llama-2 7B decoder block
  Megatron TP16均由
  真实exporter进入production pipeline；每个case的
  expected和capture shape/dtype/bytes一致并由torch.testing全量比较；Q40/Q41及高层optimization tensor
  result复用同一torch raw tensor读写与comparator；完整package、fresh no-card和故障注入证明comparator能拒绝错误后
  到board-ready。三个case的真实board fresh output全部通过后才能done。
```

## 参考结果分层

- 用户可观察的tensor语义以PyTorch eager为唯一expected owner；raw文件只是manifest ABI transport，不能成为
  第二份数值参考。
- runtime capture按expected的shape/dtype原样解码，不做FP32、FP16、整数或其它中间精度转换；默认直接调用
  `torch.testing.assert_close(actual, expected)`。确有目标数值profile时只显式设置comparison tolerance，不能改写tensor dtype。
- 普通case使用固定seed的`torch.rand`/`torch.randn`构造可复现输入；周期pattern、one-hot和手写数值公式只用于
  出现问题后的定向debug，不进入正常PyTorch case。
- target FP16/BF16 fused accumulation、physical codec、padding、canary和transport status仍由各自formal/ABI
  gate拥有。它们可以与PyTorch tensor gate并存，但不能冒充或覆盖用户级结果比较。

## HuggingFace Megatron TP16 case

- source复用仓库已有`meta-llama/Llama-2-7b-hf` config驱动的单个完整block：hidden size `4096`、
  intermediate size `11008`、32 heads、head dim `128`、batch `1`、sequence length `16`，包含RMSNorm、
  Q/K/V projection、RoPE、causal attention、O projection、第二个RMSNorm和SwiGLU MLP；不另造tiny config或
  简化attention/MLP替代。
- 输入和全部parameter由固定seed的Torch random API形成，同一module实例先在CPU eager执行得到完整参考结果；
  exporter使用同一输入和参数，不读取NumPy expected或手写公式。
- 16-rank mesh的输入、最终输出和LayerNorm vector replicated；Q/K/V与Gate/Up weight按输出维column-parallel，
  O与Down weight按输入维row-parallel，对应中间activation显式TP分片，两个row-parallel projection自然形成
  AllReduce。parameter sharding由module结构上的显式spec提供，不能通过parameter name matcher恢复。
- 正常configured-board实例使用已有FP16 HuggingFace config，属于板测默认dtype的case参数；公共raw读写和比较
  仍不固定dtype。Llama case复用Q28已冻结的`atol=0.004, rtol=0.002`全张量比较policy，不能转换actual或
  参考结果dtype。
- fresh production compile transaction为`493.374 s`，peak RSS为`3,015,048 KiB`；发布schema-v6、16-rank
  cluster prepare/main package和一个aggregate ELF。每rank 18个typed launch slot，manifest选择
  `rank-row-pointer-table-v1`，runtime packet只携带16个device row pointer；完整288-resource no-card preflight通过。
  这些只证明board-ready artifact/runtime闭合，不代签真实board output与PyTorch eager比较。

## 实施顺序

1. 在`test/Board/PyTorch/`建立统一source case、torch-only raw tensor读写、output capture comparator和故障注入测试；
   dtype是case参数，不写死在公共协议或文件名中。
2. 保留现有single-tile shape的rank-one GEMM，并让distributed case对齐Q39已有16-rank `4096³` contracting-K
   sharded GEMM，以及实际shape的HuggingFace Llama-2 7B block Megatron TP16；后两者必须在post-SPMD IR自然
   产生AllReduce，不能另写collective MLIR。
3. 让现有高层GEMM/AllReduce、Q40/Q41和optimization campaign的tensor expected/capture复用torch seam；保留
   低层qualification fixture自身的结构/协议检查。
4. 运行source export、增量构建、focused CTest/no-card和静态一致性检查；真实设备不可用时停在
   `board-ready`并明确未执行board gate。

## 不算完成

- 只有手写StableHLO、FileCheck、manifest结构、package生成、runner `--expected`或no-card成功。
- expected由NumPy生成，或runtime只让provider比较raw bytes而没有把capture解码为torch.Tensor再比较。
- 只检查rank 0、抽样元素、digest，或16-rank case没有证明所有声明output均被捕获和比较。
- distributed GEMM缩成不对应Q39的tiny shape；Transformer case使用tiny config、只导出若干GEMM、使用
  简化自定义MLP、依赖parameter name决定TP角色，或没有证明两个
  row-parallel projection形成预期AllReduce。
- 把底层physical/ABI calibration改写成PyTorch语义，或把Q44的测试目录/runner变成新的compiler输入协议。
