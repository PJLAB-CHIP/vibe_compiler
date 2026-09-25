# Kernel 本体计时

输入是同一编译结果的轻量 `timing` capture package；输出是每个物理Tile的main entry执行时间，
以及其中最长的duration。单位为微秒。实现合同与覆盖矩阵见[15号设计](../tasks/15-launch-runtime-package.md#kernel-本体轻量计时)。

## 使用

编译时在原有命令中加 `--profile`；对其普通package子目录执行原 `wafer-run` 命令并加 `--kernel-timing`。
no-card也接受该选项，并校验实际timing capture的参数、64字节record和内存计划。板端执行时仍提供原输入、expected、
output及设备资格参数；`--device-timing`可同时保留外层event对照。

PyTorch板测入口在真实板测时默认开启轻量计时，同时记录stream event；沿原路径生成新的输入/reference并验证完整输出。
`--no-card`默认仍验证普通包，明确使用`--kernel-timing`可验证计时capture。
显式`--profile`仍执行完整profile；需要与旧性能基线同口径的无插桩普通包时，使用`--no-kernel-timing --device-timing`。
轻量模式只运行timing capture一次，保留正常完成与清理，不运行完整profile的三次采集。
Profile元数据默认不设固定文件大小或记录数上限，完整调用点清单不会因模型变大而被默认预算拒绝。
嵌入式调用者可通过`PackageParseLimits`显式设置profile资源预算；结构、内容、capture与摘要校验始终执行，
具体边界见[15号package合同](../tasks/15-launch-runtime-package.md#3-current-package-schema)。
普通包没有首尾插桩，也没有record分配。两种event口径在记录中分别标明，不能互作性能回归样本。

正式case集合新增BF16 NCx性能配置：`single-card-gemm-m4096-k1024-n4096-ncx`（seed20260803）及
`attention-prefill-28-heads-2048-ncx`（seed20260922）。两者都用`--optimization-policy search --search-mode standard`
及`--search-width 8 --search-trials 42`。case名绑定外部NCx布局，不需要再手工修改factory或传layout；
其余case的外部布局默认仍为Tensor，显式`--external-layout ncx`继续可用于专项测量。
每次成功板测的`numeric-audit-NN.json`有typed `device_timing`字段：`device_event_ns`、
`kernel_main_entry_us`、`kernel_longest_tile`及16项`kernel_tile_us`；缺失或矛盾时case失败。

输出格式：

```text
kernel_timing_tile: tile_id=0 elapsed_us=...
kernel_timing: kind=vendor-microseconds scope=main-entry aggregation=max-tile-duration longest_tile=... elapsed_us=...
board_timing: kind=tx-stream-events device_elapsed_ns=...
```

起点在读取record配置之后、entry第一项工作之前；终点在原terminal completion之后、record写回之前。
区间包含kernel内部CPU构造与提交指令、搬运、计算、通信和必要等待。外部参数解码、派发、返回通知、记录回传和
独立transport prepare phase不在该区间。只保留两次时钟采样及首尾少量函数指令，不启用site hook、PMU或trace等待替代。
未扣除首尾读钟及函数调用的微小扰动，不称为零开销测量。

## 厂商计时事实

`supported`：SDK `drv/tick.h` 声明 `uint64_t csi_tick_get_us(void)`；`rtm.h` 的导出项为 `{addr, name}`。
当前已审计Kcore firmware SHA256为 `dcdc428b6fc78c6f6e070fa8e68e468942e1cc688315d41a8d4083be096907cf`。
其RTMSymTab项（binary offset `0x340f8`）及相应relative relocation把该名称绑定到`0x13fc0`。
该函数读CLINT timer、减初值、乘1,000,000后除以固件timer频率；生产timing CRT调用此导出，
不猜测`rdcycle`频率、不自行配置timer或cache、不写PMU。

同一firmware的`0x104c8..0x104de`在`process_packet`前后调用这个微秒函数；
上报路径先乘1,000转为纳秒，AP的`npu_manage_mhu_receive_all_done`取各Tile的最大值，
`tx_kernel_launch_triton`日志打印时再除1,000。因此厂商`kcore time`单位是微秒，
范围包含固件的执行请求处理，比生成的entry本体宽，结束通知位于它之外。

`board-observed`：此前同一次2048 BF16 attention三次event为3.546/3.544/3.448ms，
对应厂商kcore为2.493/2.476/2.491ms。这里只用于说明计时范围差异，不作为新计时实现的验收。
本轮实现的两项BF16 NCx实卡各三次通过完整数值、guard及正常清理；最长Tile本体中位数为
GEMM M=N4096/K1024的1.801ms、2048/28-head causal attention的2.461ms。
同次外层event分别为3.158/3.579ms，明细见[性能记录](board-performance-results.md#2026-09-23kernel-本体轻量计时)。

`unknown`：各Tile时钟的共同原点与启动错位没有在当前实现中校准。因此输出最长Tile本地duration，
不计算 `max(end)-min(begin)` 作为全卡跨度；event与entry差值不全部归为launch。
`excluded`：不同engine活动时间求和、从event固定减去空kernel时间、用完整Trace入口跨度代替轻量本体时间。
