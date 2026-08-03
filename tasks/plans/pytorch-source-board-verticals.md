# PyTorch Source Board Verticals 实施计划

状态：Q44 `board-ready`。本计划只拆解PyTorch source到board tensor correctness纵向；状态以
`tasks/progress.md`为准，frontend和board长期合同分别由02、16拥有。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  可在CPU eager执行的PyTorch module、由case tensor自身声明dtype的输入与parameter，以及钉死的source-built
  PyTorch/XLA exporter；sharded case还包含由torch_xla SPMD API表达的逻辑mesh/sharding。
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
  rank-one GEMM和16-rank K-sharded GEMM/AllReduce均由真实exporter进入production pipeline；每个case的
  expected和capture shape/dtype/bytes一致并由torch.testing全量比较；Q40/Q41及高层optimization tensor
  result复用同一torch codec/comparator；完整package、fresh no-card和故障注入证明comparator能拒绝错误后
  到board-ready。rank-one与16-rank真实board fresh output通过后才能done。
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

## 实施顺序

1. 在`test/Board/PyTorch/`建立统一source case、torch-only raw codec、output capture comparator和故障注入测试；
   dtype是case参数，不写死在公共协议或文件名中。
2. 增加小规模rank-one GEMM与16-rank contracting-K sharded GEMM；后者必须在post-SPMD IR自然产生AllReduce，
   不能另写collective MLIR。
3. 让现有高层GEMM/AllReduce、Q40/Q41和optimization campaign的tensor expected/capture复用torch seam；保留
   低层qualification fixture自身的结构/协议检查。
4. 运行source export、增量构建、focused CTest/no-card和静态一致性检查；真实设备不可用时停在
   `board-ready`并明确未执行board gate。

## 不算完成

- 只有手写StableHLO、FileCheck、manifest结构、package生成、runner `--expected`或no-card成功。
- expected由NumPy生成，或runtime只让provider比较raw bytes而没有把capture解码为torch.Tensor再比较。
- 只检查rank 0、抽样元素、digest，或16-rank case没有证明所有声明output均被捕获和比较。
- 把底层physical/ABI calibration改写成PyTorch语义，或把Q44的测试目录/runner变成新的compiler输入协议。
