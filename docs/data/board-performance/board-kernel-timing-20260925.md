# 恢复版本的全矩阵 kernel timing 复验

本轮共 82 项准备配置；80 项完成实卡，共 84 个执行步骤；2 项主机编译未通过。

本轮每个执行步骤只采集一次轻量 timing，同时验证全部输出、guard、正常清理与设备健康。
kernel 为全部 16 Tile 中最长的 main-entry duration；event 为同次 stream event。单位均为 ms。
两者口径不同，单样本不代表普通包性能中位数。两步 decode 分行，第二步使用本轮第一步的实际 KV。

Shape 取自本轮导出的 `source-program/functions/forward.meta`，按 `input_locations` 的原始参数位置还原。
84 个已执行步骤的 metadata SHA256 均与原准备证据一致，输出 shape/dtype 也与数值审计逐项一致；
两项未构包 LM 仅记录 source 声明的输入与输出，不表示有设备输出。JSON 保存每步 shape、dtype 和来源摘要；
CSV 保留 84 个已执行步骤，与下表的实测行同步。

以下都是完整逻辑张量 shape，不是单 Tile 切块或 NCx 物理存储尺寸；NCx 两项保留相同逻辑 shape。
表中 dtype 是计算数据类型，输入 dtype 不同时在 shape 后单独标注（LM IDs 为 int64，部分 mask 为 bool/float32）。
模型已冻结的参数不计入运行时输入列，其主要维度如下：

- GEMM：`A[batch,M,K] × B[1,K,N] → C[batch,M,N]`，B 在 batch 间共享；基础配置为 M=1024/K=256/N=512，
  tail-1025 为 M=1025/K=257/N=513，tail-1031 为 M=1031/K=263/N=519。
  batch-shared-rhs 的 batch=4；其 tail-1025 实际 K=1031。大 GEMM 的 M/K/N 见每行完整 A/B/C。
- 普通 attention / GQA：Q/K/V/O 为 `[batch,heads,sequence,head_dim]`，Q head 与 KV head 分别列出。
  `valid-kv-prefix` 的 KV 存储长 1031、实际使用前 1024；`sliding-window` 的窗口为 64。
  `padding` 的 KV 存储长 33、有效前缀 26；`fully-masked` 的全部 33 个位置均屏蔽。
- 两步 `decode-kv-cache` 包含 Q/K/V/O projection、RoPE 和 KV 更新，hidden=4096、Q/KV heads=32、head_dim=128，
  四个 projection weight 均为 `[4096,4096]`。普通 cache 逐步为 1023→1024、1024→1025；
  long-4096 为 4094→4095、4095→4096。Q2 case 则直接输入 Q/K/V，新增长度 2，cache 为 1024→1026。
- LLaMA block / 单层 LM：batch=1、hidden=4096、Q/KV heads=32、head_dim=128、FFN=11008，均只执行一个 decoder layer。
  block 输入为 hidden states；单层 LM 输入为 token IDs，包含 embedding 和 LM head，vocab=32000。
  已通过的 block 和单层 LM 均为 S16；S1024/1025 保持编译失败。
- ViT：batch=1、hidden=768、heads=12、head_dim=64、MLP=3072，执行一个 encoder block；ResNet18 输入采用 NCHW。
- local-conv、biased-conv 和 conv-mixed-dag：输入/卷积结果为 NCHW，W 为 `[Cout,Cin,Kh,Kw]`，stride=1、padding=1。
  local-conv 的基础/tail-1025/tail-1031 分别使用 3×3、2×3、3×2 kernel，输出 H/W 因此不同。
  conv-mixed-dag 的 FP16 输入宽 1024，BF16 输入宽 1025，不能当作相同 shape 的 dtype 性能对照。
- local-reduce 分别归约输入的 dim=3 和 dim=(2,3)；通信 case 列出整个单卡逻辑输入/输出，16 Tile 参与执行。

| Case | dtype | step | 输入 shape | 输出 shape | kernel（ms） | event（ms） | 结果 |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| single-card-gemm-m4096-k1024-n4096-ncx | bfloat16 | 1 | `A[1,4096,1024]; B[1,1024,4096]` | `C[1,4096,4096]` | 1.834 | 3.135 | 通过 |
| attention-prefill-28-heads-2048-ncx | bfloat16 | 1 | `Q[1,28,2048,128]; K[1,28,2048,128]; V[1,28,2048,128]` | `O[1,28,2048,128]` | 2.447 | 3.562 | 通过 |
| attention-prefill-28-heads-2048 | bfloat16 | 1 | `Q[1,28,2048,128]; K[1,28,2048,128]; V[1,28,2048,128]` | `O[1,28,2048,128]` | 2.675 | 3.766 | 通过 |
| single-card-gemm-4096 | bfloat16 | 1 | `A[1,4096,4096]; B[1,4096,4096]` | `C[1,4096,4096]` | 6.010 | 7.602 | 通过 |
| single-card-gemm-4096 | float16 | 1 | `A[1,4096,4096]; B[1,4096,4096]` | `C[1,4096,4096]` | 6.000 | 7.363 | 通过 |
| single-card-gemm-tail-4097 | float16 | 1 | `A[1,4097,4097]; B[1,4097,4097]` | `C[1,4097,4097]` | 7.366 | 8.881 | 通过 |
| llama-2-7b-block | bfloat16 | 1 | `hidden[1,16,4096]` | `hidden_out[1,16,4096]` | 4.194 | 7.336 | 通过 |
| llama-2-7b-block | float16 | 1 | `hidden[1,16,4096]` | `hidden_out[1,16,4096]` | 4.168 | 7.352 | 通过 |
| all-reduce-sum | float16 | 1 | `lhs[16,1024,1]; rhs[16,1024,1]` | `Y[16,1024,1]` | 0.039 | 0.979 | 通过 |
| all-reduce-sum-tail-1025 | float16 | 1 | `lhs[16,1025,1]; rhs[16,1025,1]` | `Y[16,1025,1]` | 0.039 | 1.023 | 通过 |
| all-reduce-sum-tail-1031 | float16 | 1 | `lhs[16,1031,1]; rhs[16,1031,1]` | `Y[16,1031,1]` | 0.037 | 0.930 | 通过 |
| allgather-add | float16 | 1 | `lhs[16,16,1,1024]; rhs[16,1,1024]` | `Y[16,16,1,1024]` | 0.033 | 1.069 | 通过 |
| allgather-add-tail-1025 | float16 | 1 | `lhs[16,16,1,1025]; rhs[16,1,1025]` | `Y[16,16,1,1025]` | 0.032 | 0.957 | 通过 |
| allgather-add-tail-1031 | float16 | 1 | `lhs[16,16,1,1031]; rhs[16,1,1031]` | `Y[16,16,1,1031]` | 0.034 | 0.926 | 通过 |
| alltoall-transpose | float16 | 1 | `lhs[1024,16,1]; rhs[16,1024,1]` | `Y[1024,16,1]` | 0.381 | 1.288 | 通过 |
| alltoall-transpose-tail-1025 | float16 | 1 | `lhs[1025,16,1]; rhs[16,1025,1]` | `Y[1025,16,1]` | 0.452 | 8.568 | 通过 |
| alltoall-transpose-tail-1031 | float16 | 1 | `lhs[1031,16,1]; rhs[16,1031,1]` | `Y[1031,16,1]` | 0.430 | 1.369 | 通过 |
| attention-causal-decode-two-tokens | bfloat16 | 1 | `Q[1,1,2,64]; K_new[1,1,2,64]; V_new[1,1,2,64]; K_past[1,1,1024,64]; V_past[1,1,1024,64]` | `O[1,1,2,64]; K_updated[1,1,1026,64]; V_updated[1,1,1026,64]` | 0.097 | 2.462 | 通过 |
| attention-causal-decode-two-tokens | float16 | 1 | `Q[1,1,2,64]; K_new[1,1,2,64]; V_new[1,1,2,64]; K_past[1,1,1024,64]; V_past[1,1,1024,64]` | `O[1,1,2,64]; K_updated[1,1,1026,64]; V_updated[1,1,1026,64]` | 0.093 | 1.123 | 通过 |
| attention-decode-kv-cache | bfloat16 | 1 | `hidden[1,1,4096]; K_past[1,32,1023,128]; V_past[1,32,1023,128]; mask[1,1,1,1024]` | `hidden_out[1,1,4096]; K_updated[1,32,1024,128]; V_updated[1,32,1024,128]` | 1.656 | 3.612 | 通过 |
| attention-decode-kv-cache | bfloat16 | 2 | `hidden[1,1,4096]; K_past[1,32,1024,128]; V_past[1,32,1024,128]; mask[1,1,1,1025]` | `hidden_out[1,1,4096]; K_updated[1,32,1025,128]; V_updated[1,32,1025,128]` | 1.704 | 3.742 | 通过 |
| attention-decode-kv-cache | float16 | 1 | `hidden[1,1,4096]; K_past[1,32,1023,128]; V_past[1,32,1023,128]; mask[1,1,1,1024]` | `hidden_out[1,1,4096]; K_updated[1,32,1024,128]; V_updated[1,32,1024,128]` | 1.616 | 4.730 | 通过 |
| attention-decode-kv-cache | float16 | 2 | `hidden[1,1,4096]; K_past[1,32,1024,128]; V_past[1,32,1024,128]; mask[1,1,1,1025]` | `hidden_out[1,1,4096]; K_updated[1,32,1025,128]; V_updated[1,32,1025,128]` | 1.737 | 6.286 | 通过 |
| attention-decode-kv-cache-long-4096 | bfloat16 | 1 | `hidden[1,1,4096]; K_past[1,32,4094,128]; V_past[1,32,4094,128]; mask[1,1,1,4095]` | `hidden_out[1,1,4096]; K_updated[1,32,4095,128]; V_updated[1,32,4095,128]` | 5.127 | 10.034 | 通过 |
| attention-decode-kv-cache-long-4096 | bfloat16 | 2 | `hidden[1,1,4096]; K_past[1,32,4095,128]; V_past[1,32,4095,128]; mask[1,1,1,4096]` | `hidden_out[1,1,4096]; K_updated[1,32,4096,128]; V_updated[1,32,4096,128]` | 5.198 | 8.530 | 通过 |
| attention-decode-kv-cache-long-4096 | float16 | 1 | `hidden[1,1,4096]; K_past[1,32,4094,128]; V_past[1,32,4094,128]; mask[1,1,1,4095]` | `hidden_out[1,1,4096]; K_updated[1,32,4095,128]; V_updated[1,32,4095,128]` | 5.104 | 10.842 | 通过 |
| attention-decode-kv-cache-long-4096 | float16 | 2 | `hidden[1,1,4096]; K_past[1,32,4095,128]; V_past[1,32,4095,128]; mask[1,1,1,4096]` | `hidden_out[1,1,4096]; K_updated[1,32,4096,128]; V_updated[1,32,4096,128]` | 5.132 | 9.058 | 通过 |
| attention-fully-masked | bfloat16 | 1 | `Q[1,1,1024,64]; K[1,1,33,64]; V[1,1,33,64]; mask[1,1,1,33]:bool` | `O[1,1,1024,64]` | 0.106 | 1.104 | 通过 |
| attention-fully-masked | float16 | 1 | `Q[1,1,1024,64]; K[1,1,33,64]; V[1,1,33,64]; mask[1,1,1,33]:bool` | `O[1,1,1024,64]` | 0.118 | 1.057 | 通过 |
| attention-gqa | bfloat16 | 1 | `Q[1,32,1024,128]; K[1,8,1024,128]; V[1,8,1024,128]` | `O[1,32,1024,128]` | 1.047 | 2.147 | 通过 |
| attention-gqa | float16 | 1 | `Q[1,32,1024,128]; K[1,8,1024,128]; V[1,8,1024,128]` | `O[1,32,1024,128]` | 1.040 | 2.154 | 通过 |
| attention-gqa-tail-1025 | bfloat16 | 1 | `Q[1,32,1025,128]; K[1,8,1025,128]; V[1,8,1025,128]` | `O[1,32,1025,128]` | 5.831 | 12.654 | 通过 |
| attention-gqa-tail-1025 | float16 | 1 | `Q[1,32,1025,128]; K[1,8,1025,128]; V[1,8,1025,128]` | `O[1,32,1025,128]` | 5.823 | 7.057 | 通过 |
| attention-noncausal-additive | bfloat16 | 1 | `Q[1,1,1024,64]; K[1,1,33,64]; V[1,1,33,64]; mask[1,1,1024,33]:float32` | `O[1,1,1024,64]` | 0.115 | 1.085 | 通过 |
| attention-noncausal-additive | float16 | 1 | `Q[1,1,1024,64]; K[1,1,33,64]; V[1,1,33,64]; mask[1,1,1024,33]:float32` | `O[1,1,1024,64]` | 0.106 | 1.103 | 通过 |
| attention-noncausal | bfloat16 | 1 | `Q[1,1,1024,64]; K[1,1,33,64]; V[1,1,33,64]` | `O[1,1,1024,64]` | 0.108 | 1.025 | 通过 |
| attention-noncausal | float16 | 1 | `Q[1,1,1024,64]; K[1,1,33,64]; V[1,1,33,64]` | `O[1,1,1024,64]` | 0.103 | 1.069 | 通过 |
| attention-padding | bfloat16 | 1 | `Q[1,1,1024,64]; K[1,1,33,64]; V[1,1,33,64]; mask[1,1,1,33]:bool` | `O[1,1,1024,64]` | 0.116 | 1.098 | 通过 |
| attention-padding | float16 | 1 | `Q[1,1,1024,64]; K[1,1,33,64]; V[1,1,33,64]; mask[1,1,1,33]:bool` | `O[1,1,1024,64]` | 0.115 | 1.093 | 通过 |
| attention-prefill | bfloat16 | 1 | `Q[1,1,1024,64]; K[1,1,1024,64]; V[1,1,1024,64]` | `O[1,1,1024,64]` | 0.258 | 1.296 | 通过 |
| attention-prefill | float16 | 1 | `Q[1,1,1024,64]; K[1,1,1024,64]; V[1,1,1024,64]` | `O[1,1,1024,64]` | 0.256 | 1.295 | 通过 |
| attention-prefill-llama-2-7b | float16 | 1 | `Q[1,32,4096,128]; K[1,32,4096,128]; V[1,32,4096,128]` | `O[1,32,4096,128]` | 8.999 | 10.067 | 通过 |
| attention-prefill-tail-1025 | bfloat16 | 1 | `Q[1,1,1025,64]; K[1,1,1025,64]; V[1,1,1025,64]` | `O[1,1,1025,64]` | 0.261 | 2.799 | 通过 |
| attention-prefill-tail-1025 | float16 | 1 | `Q[1,1,1025,64]; K[1,1,1025,64]; V[1,1,1025,64]` | `O[1,1,1025,64]` | 0.258 | 10.861 | 通过 |
| attention-prefill-tail-1031 | bfloat16 | 1 | `Q[1,1,1031,64]; K[1,1,1031,64]; V[1,1,1031,64]` | `O[1,1,1031,64]` | 0.264 | 1.210 | 通过 |
| attention-prefill-tail-1031 | float16 | 1 | `Q[1,1,1031,64]; K[1,1,1031,64]; V[1,1,1031,64]` | `O[1,1,1031,64]` | 0.264 | 4.199 | 通过 |
| attention-sliding-window | bfloat16 | 1 | `Q[1,1,1025,64]; K[1,1,1031,64]; V[1,1,1031,64]; mask[1,1,1025,1031]:bool` | `O[1,1,1025,64]` | 0.316 | 1.353 | 通过 |
| attention-sliding-window | float16 | 1 | `Q[1,1,1025,64]; K[1,1,1031,64]; V[1,1,1031,64]; mask[1,1,1025,1031]:bool` | `O[1,1,1025,64]` | 0.318 | 7.140 | 通过 |
| attention-valid-kv-prefix | bfloat16 | 1 | `Q[1,1,1025,64]; K[1,1,1031,64]; V[1,1,1031,64]` | `O[1,1,1025,64]` | 0.256 | 1.310 | 通过 |
| attention-valid-kv-prefix | float16 | 1 | `Q[1,1,1025,64]; K[1,1,1031,64]; V[1,1,1031,64]` | `O[1,1,1025,64]` | 0.258 | 1.317 | 通过 |
| batch-shared-rhs-gemm | float16 | 1 | `A[4,1024,1024]; B[1,1024,1024]` | `C[4,1024,1024]` | 0.721 | 1.996 | 通过 |
| batch-shared-rhs-gemm-tail-1025 | float16 | 1 | `A[4,1025,1031]; B[1,1031,1025]` | `C[4,1025,1025]` | 1.412 | 2.786 | 通过 |
| biased-conv | float16 | 1 | `X[1,16,8,1024]; W[24,16,3,3]; bias[24]` | `Y[1,24,8,1024]` | 1.614 | 5.505 | 通过 |
| biased-conv-tail-1025 | float16 | 1 | `X[1,16,8,1025]; W[24,16,3,3]; bias[24]` | `Y[1,24,8,1025]` | 1.660 | 7.135 | 通过 |
| biased-conv-tail-1031 | float16 | 1 | `X[1,16,8,1031]; W[24,16,3,3]; bias[24]` | `Y[1,24,8,1031]` | 1.670 | 10.979 | 通过 |
| conv-mixed-dag | bfloat16 | 1 | `X[1,16,8,1025]; W[24,16,3,3]; bias[24]` | `joined[1,24,8,1025]; sum[1,24]` | 0.517 | 1.715 | 通过 |
| conv-mixed-dag | float16 | 1 | `X[1,16,8,1024]; W[24,16,3,3]; bias[24]` | `joined[1,24,8,1024]; sum[1,24]` | 8.343 | 16.257 | 通过 |
| division | float32 | 1 | `lhs[2,4,1024]; rhs[2,4,1024]` | `Y[2,4,1024]` | 0.019 | 0.925 | 通过 |
| division-tail-1025 | float32 | 1 | `lhs[2,4,1025]; rhs[2,4,1025]` | `Y[2,4,1025]` | 0.019 | 3.334 | 通过 |
| division-tail-1031 | float32 | 1 | `lhs[2,4,1031]; rhs[2,4,1031]` | `Y[2,4,1031]` | 0.018 | 2.940 | 通过 |
| heterogeneous-tiling-dataflow | float16 | 1 | `A[96,64]; B[64,80]; bias[80]` | `mixed[96,80]; sum[96]` | 0.066 | 1.106 | 通过 |
| llama-2-7b-single-layer-lm | bfloat16 | 1 | `input_ids[1,16]:int64` | `logits[1,16,32000]` | 64.410 | 68.385 | 通过 |
| llama-2-7b-single-layer-lm | float16 | 1 | `input_ids[1,16]:int64` | `logits[1,16,32000]` | 63.457 | 67.498 | 通过 |
| local-conv | float16 | 1 | `X[1,16,8,1024]; W[24,16,3,3]` | `Y[1,24,8,1024]` | 0.124 | 3.298 | 通过 |
| local-conv-tail-1025 | float16 | 1 | `X[1,16,8,1025]; W[24,16,2,3]` | `Y[1,24,9,1025]` | 0.155 | 1.168 | 通过 |
| local-conv-tail-1031 | float16 | 1 | `X[1,16,8,1031]; W[24,16,3,2]` | `Y[1,24,8,1032]` | 0.135 | 10.515 | 通过 |
| local-reduce | float16 | 1 | `X[1,24,8,1024]` | `sum_dim3[1,24,8]; sum_dim2_3[1,24]` | 0.034 | 0.914 | 通过 |
| local-reduce-tail-1025 | float16 | 1 | `X[1,24,8,1025]` | `sum_dim3[1,24,8]; sum_dim2_3[1,24]` | 0.035 | 0.884 | 通过 |
| local-reduce-tail-1031 | float16 | 1 | `X[1,24,8,1031]` | `sum_dim3[1,24,8]; sum_dim2_3[1,24]` | 0.033 | 0.915 | 通过 |
| reduce-scatter-sum | float16 | 1 | `rhs[16,1024,1]` | `Y[1,1024,1]` | 0.036 | 0.945 | 通过 |
| reduce-scatter-sum-tail-1025 | float16 | 1 | `rhs[16,1025,1]` | `Y[1,1025,1]` | 0.036 | 5.480 | 通过 |
| reduce-scatter-sum-tail-1031 | float16 | 1 | `rhs[16,1031,1]` | `Y[1,1031,1]` | 0.037 | 0.887 | 通过 |
| resnet18 | float16 | 1 | `image[1,3,224,224]` | `logits[1,1000]` | 10.797 | 19.381 | 通过 |
| sigmoid | float16 | 1 | `X[2,4,1024]` | `Y[2,4,1024]` | 0.032 | 0.941 | 通过 |
| sigmoid-tail-1025 | float16 | 1 | `X[2,4,1025]` | `Y[2,4,1025]` | 0.036 | 0.977 | 通过 |
| sigmoid-tail-1031 | float16 | 1 | `X[2,4,1031]` | `Y[2,4,1031]` | 0.036 | 11.945 | 通过 |
| single-card-gemm | float16 | 1 | `A[1,1024,256]; B[1,256,512]` | `C[1,1024,512]` | 0.067 | 1.053 | 通过 |
| single-card-gemm-m4096-k1024-n4096 | bfloat16 | 1 | `A[1,4096,1024]; B[1,1024,4096]` | `C[1,4096,4096]` | 2.286 | 3.598 | 通过 |
| single-card-gemm-tail-1025 | float16 | 1 | `A[1,1025,257]; B[1,257,513]` | `C[1,1025,513]` | 0.072 | 1.216 | 通过 |
| single-card-gemm-tail-1031 | float16 | 1 | `A[1,1031,263]; B[1,263,519]` | `C[1,1031,519]` | 0.062 | 1.016 | 通过 |
| vit-encoder-block | float16 | 1 | `hidden[1,1024,768]` | `hidden_out[1,1024,768]` | 12.714 | 19.681 | 通过 |
| vit-encoder-block-tail-1025 | float16 | 1 | `hidden[1,1025,768]` | `hidden_out[1,1025,768]` | 12.953 | 23.738 | 通过 |
| attention-prefill-28-heads-4096 | bfloat16 | 1 | `Q[1,28,4096,128]; K[1,28,4096,128]; V[1,28,4096,128]` | `O[1,28,4096,128]` | 8.951 | 11.796 | 通过 |
| attention-prefill-28-heads-4096 | float16 | 1 | `Q[1,28,4096,128]; K[1,28,4096,128]; V[1,28,4096,128]` | `O[1,28,4096,128]` | 8.954 | 11.429 | 通过 |
| llama-2-7b-single-layer-lm-1024 | float16 | — | `input_ids[1,1024]:int64` | `logits[1,1024,32000]` | — | — | 当前 search 8/42 未找到可执行候选；未构包、未上板，无 timing。 |
| llama-2-7b-single-layer-lm-tail-1025 | float16 | — | `input_ids[1,1025]:int64` | `logits[1,1025,32000]` | — | — | 当前 search 8/42 未找到可执行候选；未构包、未上板，无 timing。 |

生产实现恢复基线为 `665452c5`（恢复提交 `9edd2671`），profile紧凑写入为 `139d735a`，可选资源预算为 `9a9aacf6`。
未受修复影响的本轮已准备包保留原编译器身份；重编译配置单独记录新编译器、source 和 package 摘要。
三项 biased-conv 首次构包失败与成功补测均保留；ResNet18两次host失败、最终可选预算版本的fresh产品和首次实卡分别记录。
长 LM S1024/1025 是本轮额外编译复查，原有未通过状态未被取消或计作实卡通过。

详细 shape 来源、16 Tile 时间、数值审计、guard、设备身份和证据路径见[完整 JSON](board-kernel-timing-20260925.json)。
