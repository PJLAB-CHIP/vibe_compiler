# 恢复版本的全矩阵 kernel timing 复验

本轮共 82 项准备配置；80 项完成实卡，共 84 个执行步骤；2 项主机编译未通过。

本轮每个执行步骤只采集一次轻量 timing，同时验证全部输出、guard、正常清理与设备健康。
kernel 为全部 16 Tile 中最长的 main-entry duration；event 为同次 stream event。单位均为 ms。
两者口径不同，单样本不代表普通包性能中位数。两步 decode 分行，第二步使用本轮第一步的实际 KV。

| Case | dtype | step | kernel（ms） | event（ms） | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| single-card-gemm-m4096-k1024-n4096-ncx | bfloat16 | 1 | 1.834 | 3.135 | 通过 |
| attention-prefill-28-heads-2048-ncx | bfloat16 | 1 | 2.447 | 3.562 | 通过 |
| attention-prefill-28-heads-2048 | bfloat16 | 1 | 2.675 | 3.766 | 通过 |
| single-card-gemm-4096 | bfloat16 | 1 | 6.010 | 7.602 | 通过 |
| single-card-gemm-4096 | float16 | 1 | 6.000 | 7.363 | 通过 |
| single-card-gemm-tail-4097 | float16 | 1 | 7.366 | 8.881 | 通过 |
| llama-2-7b-block | bfloat16 | 1 | 4.194 | 7.336 | 通过 |
| llama-2-7b-block | float16 | 1 | 4.168 | 7.352 | 通过 |
| all-reduce-sum | float16 | 1 | 0.039 | 0.979 | 通过 |
| all-reduce-sum-tail-1025 | float16 | 1 | 0.039 | 1.023 | 通过 |
| all-reduce-sum-tail-1031 | float16 | 1 | 0.037 | 0.930 | 通过 |
| allgather-add | float16 | 1 | 0.033 | 1.069 | 通过 |
| allgather-add-tail-1025 | float16 | 1 | 0.032 | 0.957 | 通过 |
| allgather-add-tail-1031 | float16 | 1 | 0.034 | 0.926 | 通过 |
| alltoall-transpose | float16 | 1 | 0.381 | 1.288 | 通过 |
| alltoall-transpose-tail-1025 | float16 | 1 | 0.452 | 8.568 | 通过 |
| alltoall-transpose-tail-1031 | float16 | 1 | 0.430 | 1.369 | 通过 |
| attention-causal-decode-two-tokens | bfloat16 | 1 | 0.097 | 2.462 | 通过 |
| attention-causal-decode-two-tokens | float16 | 1 | 0.093 | 1.123 | 通过 |
| attention-decode-kv-cache | bfloat16 | 1 | 1.656 | 3.612 | 通过 |
| attention-decode-kv-cache | bfloat16 | 2 | 1.704 | 3.742 | 通过 |
| attention-decode-kv-cache | float16 | 1 | 1.616 | 4.730 | 通过 |
| attention-decode-kv-cache | float16 | 2 | 1.737 | 6.286 | 通过 |
| attention-decode-kv-cache-long-4096 | bfloat16 | 1 | 5.127 | 10.034 | 通过 |
| attention-decode-kv-cache-long-4096 | bfloat16 | 2 | 5.198 | 8.530 | 通过 |
| attention-decode-kv-cache-long-4096 | float16 | 1 | 5.104 | 10.842 | 通过 |
| attention-decode-kv-cache-long-4096 | float16 | 2 | 5.132 | 9.058 | 通过 |
| attention-fully-masked | bfloat16 | 1 | 0.106 | 1.104 | 通过 |
| attention-fully-masked | float16 | 1 | 0.118 | 1.057 | 通过 |
| attention-gqa | bfloat16 | 1 | 1.047 | 2.147 | 通过 |
| attention-gqa | float16 | 1 | 1.040 | 2.154 | 通过 |
| attention-gqa-tail-1025 | bfloat16 | 1 | 5.831 | 12.654 | 通过 |
| attention-gqa-tail-1025 | float16 | 1 | 5.823 | 7.057 | 通过 |
| attention-noncausal-additive | bfloat16 | 1 | 0.115 | 1.085 | 通过 |
| attention-noncausal-additive | float16 | 1 | 0.106 | 1.103 | 通过 |
| attention-noncausal | bfloat16 | 1 | 0.108 | 1.025 | 通过 |
| attention-noncausal | float16 | 1 | 0.103 | 1.069 | 通过 |
| attention-padding | bfloat16 | 1 | 0.116 | 1.098 | 通过 |
| attention-padding | float16 | 1 | 0.115 | 1.093 | 通过 |
| attention-prefill | bfloat16 | 1 | 0.258 | 1.296 | 通过 |
| attention-prefill | float16 | 1 | 0.256 | 1.295 | 通过 |
| attention-prefill-llama-2-7b | float16 | 1 | 8.999 | 10.067 | 通过 |
| attention-prefill-tail-1025 | bfloat16 | 1 | 0.261 | 2.799 | 通过 |
| attention-prefill-tail-1025 | float16 | 1 | 0.258 | 10.861 | 通过 |
| attention-prefill-tail-1031 | bfloat16 | 1 | 0.264 | 1.210 | 通过 |
| attention-prefill-tail-1031 | float16 | 1 | 0.264 | 4.199 | 通过 |
| attention-sliding-window | bfloat16 | 1 | 0.316 | 1.353 | 通过 |
| attention-sliding-window | float16 | 1 | 0.318 | 7.140 | 通过 |
| attention-valid-kv-prefix | bfloat16 | 1 | 0.256 | 1.310 | 通过 |
| attention-valid-kv-prefix | float16 | 1 | 0.258 | 1.317 | 通过 |
| batch-shared-rhs-gemm | float16 | 1 | 0.721 | 1.996 | 通过 |
| batch-shared-rhs-gemm-tail-1025 | float16 | 1 | 1.412 | 2.786 | 通过 |
| biased-conv | float16 | 1 | 1.614 | 5.505 | 通过 |
| biased-conv-tail-1025 | float16 | 1 | 1.660 | 7.135 | 通过 |
| biased-conv-tail-1031 | float16 | 1 | 1.670 | 10.979 | 通过 |
| conv-mixed-dag | bfloat16 | 1 | 0.517 | 1.715 | 通过 |
| conv-mixed-dag | float16 | 1 | 8.343 | 16.257 | 通过 |
| division | float32 | 1 | 0.019 | 0.925 | 通过 |
| division-tail-1025 | float32 | 1 | 0.019 | 3.334 | 通过 |
| division-tail-1031 | float32 | 1 | 0.018 | 2.940 | 通过 |
| heterogeneous-tiling-dataflow | float16 | 1 | 0.066 | 1.106 | 通过 |
| llama-2-7b-single-layer-lm | bfloat16 | 1 | 64.410 | 68.385 | 通过 |
| llama-2-7b-single-layer-lm | float16 | 1 | 63.457 | 67.498 | 通过 |
| local-conv | float16 | 1 | 0.124 | 3.298 | 通过 |
| local-conv-tail-1025 | float16 | 1 | 0.155 | 1.168 | 通过 |
| local-conv-tail-1031 | float16 | 1 | 0.135 | 10.515 | 通过 |
| local-reduce | float16 | 1 | 0.034 | 0.914 | 通过 |
| local-reduce-tail-1025 | float16 | 1 | 0.035 | 0.884 | 通过 |
| local-reduce-tail-1031 | float16 | 1 | 0.033 | 0.915 | 通过 |
| reduce-scatter-sum | float16 | 1 | 0.036 | 0.945 | 通过 |
| reduce-scatter-sum-tail-1025 | float16 | 1 | 0.036 | 5.480 | 通过 |
| reduce-scatter-sum-tail-1031 | float16 | 1 | 0.037 | 0.887 | 通过 |
| resnet18 | float16 | 1 | 10.797 | 19.381 | 通过 |
| sigmoid | float16 | 1 | 0.032 | 0.941 | 通过 |
| sigmoid-tail-1025 | float16 | 1 | 0.036 | 0.977 | 通过 |
| sigmoid-tail-1031 | float16 | 1 | 0.036 | 11.945 | 通过 |
| single-card-gemm | float16 | 1 | 0.067 | 1.053 | 通过 |
| single-card-gemm-m4096-k1024-n4096 | bfloat16 | 1 | 2.286 | 3.598 | 通过 |
| single-card-gemm-tail-1025 | float16 | 1 | 0.072 | 1.216 | 通过 |
| single-card-gemm-tail-1031 | float16 | 1 | 0.062 | 1.016 | 通过 |
| vit-encoder-block | float16 | 1 | 12.714 | 19.681 | 通过 |
| vit-encoder-block-tail-1025 | float16 | 1 | 12.953 | 23.738 | 通过 |
| attention-prefill-28-heads-4096 | bfloat16 | 1 | 8.951 | 11.796 | 通过 |
| attention-prefill-28-heads-4096 | float16 | 1 | 8.954 | 11.429 | 通过 |
| llama-2-7b-single-layer-lm-1024 | float16 | — | — | — | 当前 search 8/42 未找到可执行候选；未构包、未上板，无 timing。 |
| llama-2-7b-single-layer-lm-tail-1025 | float16 | — | — | — | 当前 search 8/42 未找到可执行候选；未构包、未上板，无 timing。 |

生产实现恢复基线为 `665452c5`（恢复提交 `9edd2671`），profile紧凑写入为 `139d735a`，可选资源预算为 `9a9aacf6`。
未受修复影响的本轮已准备包保留原编译器身份；重编译配置单独记录新编译器、source 和 package 摘要。
三项 biased-conv 首次构包失败与成功补测均保留；ResNet18两次host失败、最终可选预算版本的fresh产品和首次实卡分别记录。
长 LM S1024/1025 是本轮额外编译复查，原有未通过状态未被取消或计作实卡通过。

详细 16 Tile 时间、数值审计、guard、设备身份和证据路径见[完整 JSON](board-kernel-timing-20260925.json)。
