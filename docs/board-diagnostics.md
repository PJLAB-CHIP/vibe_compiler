# 板卡运行与Host寄存器诊断

`wafer-board-diagnose`包围一个现有板端命令，输出该次运行的条件、进程结果和驱动/固件日志。
默认关闭高频寄存器采集；只有显式`--capture-registers`才启动采集子进程。工具不创建runtime context，
计算、output/guard验证及厂商清理由原runner负责。设备侧profiler是另一个显式功能，不随本工具自动启用。

## 构建与普通计时

工具由canonical `cmake --build build`构建：入口为`build/bin/wafer-board-diagnose`，
C采集库在`build/share/wafer/board-diagnose/`；产品安装归入Runtime component。
运行和解码均只依赖Python标准库，C采集库不依赖board SDK。下面的设备参数须来自同一已核实设备会话。

```sh
python3 -B build/bin/wafer-board-diagnose run \
  --device /dev/accel/dev-0 --pci-bus-id 0000:3b:00.0 \
  --output-dir /tmp/attention-run \
  -- wafer-run <本次已准备的参数>
```

`--`后的参数原样转发。默认只检查占用、保存日志，不映射寄存器；日志服务收尾等待在kernel事件计时之外。
`run.json`明确记录`host_register_capture: false`。输出目录必须是新目录，避免覆盖已有故障现场。
日志起点使用当前boot的全局journal游标，回读时筛选kernel记录；历史kernel日志轮转为空仍可采集新运行窗口。
`run.json`的`launched`区分启动前失败与runner已经执行。
`--timeout-seconds`默认120秒，只限制等待原runner的时间；超时保留进程交给厂商清理并返回失败，
不发送kill、reset或自动重试。发现本窗口driver/firmware告警同样返回失败，不能用runner返回0覆盖告警。

PyTorch板测入口通过`--board-diagnose-tool build/bin/wafer-board-diagnose`接入，每次repeat分别输出
`diagnostics-NN/`与`measurement-NN.json`。不指定该工具时保持原运行入口，`--capture-registers`也默认false。
该接入支持ordinary及`--profile --kernel-timing`的单次执行；完整`--profile`继续使用自己的收集与报告路径。

## 显式诊断

给上面的`run`命令加`--capture-registers`，或给PyTorch板测入口同时指定工具和该开关，即启动Host采集。
控制器先等采集器ready，再启动原计算。正常计算结束后通知采集器停止；检测到fatal或采集提前结束时报告失败，
保留原始文件。采集默认最长30秒，因此长于此窗口的计算不能签完整寄存器覆盖。

独立只读采集不提交计算，可用于启动前的现场观察：

```sh
sudo python3 -B build/bin/wafer-board-diagnose capture \
  --device /dev/accel/dev-0 --pci-bus-id 0000:3b:00.0 \
  --expected-boot "$(cat /proc/sys/kernel/random/boot_id)" \
  --expected-driver "$(cat /sys/module/tsingmicro/srcversion)" \
  --output /tmp/registers.bin --ready-file /tmp/registers.ready \
  --stop-file /tmp/registers.stop --maximum-ms 1000
```

设备node与PCI地址由调用者提供已资格化的配对。当前厂商的device node是virtual misc，sysfs无PCI parent链接；
工具不能用该链接证明二者关系。实际计算仍由`wafer-run`通过runtime核对device/PCI/库身份。
Live读取核对实际ATU窗口、16 Tile packed XY身份、boot和driver；寄存器定义依据见
[TX81故障定位事实](tx81-tdma-fault-localization.md#直接-fatal-状态的快速采集准备)，不声称支持任意新driver布局。
系统占用检查覆盖`/proc/*/fd`及`fuser`；权限不足记unknown，busy/unknown停止，已核对身份的厂商日志服务单列。

Python控制器通过`ctypes`调用C热循环；`O_RDONLY`及`PROT_READ`映射BAR4。每Tile按顺序读取
stream fatal、worker0 TDMA count、last-command低/高字、count、fatal，并保存单调时钟起止值。
每条56字节；默认1,048,576条ring约56 MiB。热循环不打印或写文件，结束后统一落盘。
首次stream fatal bit12触发后保留最多100ms或半ring的后续记录，以先达到者停止。
其它停止原因包括controller stop、最长时限和全1 aperture读值。非零累计count/last-command本身不拒绝。

## 离线消费与解释

```sh
python3 -B build/bin/wafer-board-diagnose decode /tmp/registers.bin \
  --output /tmp/registers-summary.json
```

解码不访问设备，逐条验证原始记录，只在内存保留统计和有界触发邻域。产物包括：

- `run.json`、`runner.log`、`occupancy.json`、`kernel.log`、`firmware.log`：本次运行及健康判定依据。
- 开启采集时的`capture.bin`、`.identity.json`、`.summary.json`：原始记录、采集配置/映射/工具摘要、触发邻域及采样间隔。
- 每Tile计数差按32位原始字模数计算；MMIO读取宽度不证明计数有效位宽。发生回绕、reset或覆盖不完整时，
  不能把该原始差值当作实际命令总量。

顺序读取不是原子快照，last-command不是已确认fault PC；未采到fatal不证明任意瞬态都没有出现。
只读采集仍可能改变总线时序，带采集与关闭采集的计时必须分别标记。工具的host文件模型验证不代替真实fault采集。
历史raw/log继续作为审计证据留在外部证据目录，不作为新测试输入，也不随源码提交二进制。
