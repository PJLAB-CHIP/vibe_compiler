# TX81 LSU TDMA 故障定位

输入是同一设备执行窗口的 host kernel、AP/Kcore/Score 日志、实际加载库与固件的身份，以及该次实际 ELF、输入和回读。
输出是可核对的故障链、已确认的观测缺口和下一次采集所需字段；直接消费者是板测诊断及厂商问题复现。
本页不定义新的 runtime 清理协议，也不把二进制审计当作板端根因或修复证明。

本次证据见 [2026-09-20 审计记录](data/board-performance/tdma-firmware-localization-20260920.json)。
首次 TDMA 告警、后续执行和最终清理错误分别归档。历史 raw 仅用于离线审计，不作为新测试输入。

## 已确认的告警路径

下列结论属于 **supported：当前二进制静态证据**，绑定记录中的文件摘要。

1. 当前 AP rootfs 中的 `tx_npu.ko` 带符号与 DWARF。`stream_irq_handler` 调用
   `streamint_fatal_err_get`，后者读取 stream interrupt 寄存器块的 `+0xc0`，有效掩码为 `0x1ff03`。
   这是该寄存器块内偏移，不是 host BAR 的绝对地址。
2. `fatal_err_handler` 重定位表的第 12 项是 `fatalerr_lsu_tdma_intr_handle`；第 13 项才是 RDMA。
   因此旧日志中的 `fatal_err=4096` 即 `1<<12`，与 TDMA 路径一致。
3. TDMA handler 上报 `0x0D00C005`，打印 Tile 编号并更新软件计数。
   在该函数完整反汇编中没有故障 PC、TDMA 参数或 NCC 队列快照，也没有 reset 调用。
   `need to reset chip` 是日志文本，不能据此声称当时已经执行芯片复位。
4. 可供厂商定位的符号入口为 `stream_int.c:746` 的 `fatalerr_lsu_tdma_intr_handle`
   和 `npu.c:1871` 的 `stream_irq_handler`。这些位置来自此版本 DWARF，不能跨版本套用。

安装包同时带有旧 `src/refine` 源码和 `src/refine2` 的预编译 core。
旧源码将 `0x0D00C005` 命名为 RDMA，与实际 AP handler 和安装 host driver 的文本不一致。
本次按实际二进制、函数重定位和日志三者确认 TDMA，不能仅凭包内旧 header 改判 engine。

## 已确认的版本与运行事实

| 级别 | 事实 | 不能推出的结论 |
| --- | --- | --- |
| supported：文件对照 | 已安装 `libhpgr.so`、runtime 包内的管理库、AP 动态 rootfs、Kcore/Score bin 与 tufw 均与 5.7.0.0524 对应安装包逐字节相同 | 未读取板上 flash 全量镜像，不能签发所有启动组件完全相同 |
| supported：安装包差异 | driver 包与 runtime 包自带的 `libtsmml.so` 不同；已安装文件匹配 runtime 包 | 不能仅凭两个安装包的库不同就认定混装或根因 |
| board-observed：旧 Q1 | Tile-2、fatal bit 12、TDMA EID 出现在正常完成与清理之前；完整输出仍满足既定数值门槛 | 正确输出不抵消设备错误，也不证明此告警必然永久卡死 |
| board-observed：当前 BF16 4K prefill | 厂商正常退出版本仍在执行窗口出现多个 Tile 的 TDMA；三个输入和回读均为有限值，离线数值门槛满足，guard 通过 | 不能把 NaN/Inf 输入或跳过全局析构当作这一次的已证实解释 |
| board-observed：后续 FP16 | 用户明确要求继续后运行同 workload 的 FP16；固件再次报告 TDMA，随后出现 AP 资源清理超时及 Kcore 关闭失败 | 后续清理失败不能解释先前 TDMA 的初始触发条件 |
| board-observed：重启后首个当前 BF16 4K prefill | 单次执行再次报告 TDMA；只读 PMU 采样取得配置、异常字段和变化中的计数，见下节 | 此次复现不需要本 boot 先连续执行多个 case；采样可能影响时序，不能外推到所有历史故障 |
| unknown | 故障时实际 TDMA packet、源程序位置、raw/命令字段编码及其对应 worker、地址 | 尚不能区分非法命令/地址、依赖等待、状态残留或阈值问题 |

当前 BF16 Q/K/V 各有 14,680,064 个元素，无非有限值；Q/K 范围为 ±0.416015625，V 范围为 ±3.328125。
回读 relative L2 为 0.00194608，所有元素满足既定比较门槛。该检查只排除了这次输入含 NaN/Inf，
没有证明 compiler 生成的地址或硬件依赖合法；带告警的计时仍不作健康性能样本。

## 可用的定位接口及边界

厂商 `collect_logs.sh` 和 `npu_ep_log_tools get_log_now` 用于取日志，不能补出原 handler 未记录的 packet。
当前 `debugfs` 的 `inject_eid` 是错误注入接口，**excluded**：不用于故障现场采集。
`tsm_smi` 的 `DebugSetNpuTimeout` 经管理 API/KIQ 设置另一层运行期限，不是已经证明的 LSU 硬件 timeout 修复入口。
本轮没有修改上述设置、异常掩码、first-catch、power 或 reset。

SDK 的 `pmu_reg.h` 给出了以下 **supported：静态寄存器定义**。NCC PMU Tile-local base 为 `0x590000`；
读取后的保留周期、故障后可访问性及与源程序位置的对应关系尚属 **unknown**。

| 观测项 | 相对 PMU base 的偏移 | 用途 |
| --- | --- | --- |
| first-catch 配置 | `0x1f8` | 保存原值；具体位义仍待确认，不能仅按名称解释或改写 |
| 三个 worker 的 exception raw/stat/mask/cmd ID | worker0 `0x1fc..0x218`，worker1 `0x21c..0x238`，worker2 `0x23c..0x258` | 区分原始异常、掩码与报告状态，保留实际命令编号 |
| TDMA last command info | `0x27c`、`0x280` | 保存原始两字；未确认编码前不能解释为 PC 或源/目的地址 |
| caught-exception-stop | `0x284` | 区分当前异常是否要求 engine 停止 |
| LSU timeout 配置 | `0x28c` | 保存实际配置，比较干净启动与故障窗口；单位和合法调整范围仍待确认 |
| RDMA/WDMA/TDMA exception enable | `0x298`、`0x29c`、`0x2a0` | 保存告警配置；不通过屏蔽告警制造“通过” |
| 每 worker TDMA count/blocking | `0x20/0x38`，另两个 worker 各加 `0x30` | 配合相邻快照区分提交数量和进度；不凭单次值推断队列状态 |
| TDMA execution counter | `0x164`、`0x168` | 保留低/高字与采集顺序，不把跨采样变化误作原子快照 |

厂商 `instr_def.h` 还定义了 TDMA control、src/dst、inclusive end、element/byte count 与 stride/iteration，
以及 CSR TDMA 非法 opcode、超过 3 MiB 地址范围的异常位。它们能支持进一步区分命令错误与 timeout，
但 CSR 的 TDMA 异常字段和 AP 的 fatal bit 12 是不同寄存器域，不能互相代替。

NCC command/CSR 访问使用 Kcore 的 `get_ncc_reg`；当前实现包含专用 `lrd`。
**excluded**：不能把其 `0x01000000 + worker_offset` 地址直接加到 host Tile BAR 映射后读，
这与普通 Tile-local PMU aperture 不是已证明等价的访问路径。

## 重启后首个计算的只读观测（2026-09-20）

证据与原始文件摘要见 [PMU 实测记录](data/board-performance/tdma-pmu-observation-20260920.json)。
新 boot `8da5f00d-b678-4df9-bb23-b93b11a0c54e` 的第一个计算 invocation 是当前 BF16 4K 28-head prefill；
输入、reference 重新生成，source 与已核对的当前 package 相同。设备占用检查仅发现已核实的厂商日志服务。
采集器通过实际 ATU 表只读映射 BAR4，未创建 runtime context 或提交 kernel。

访问验证最初因采集器误把 Tile 身份字段当作线性编号而拒绝；实际逻辑/物理字段均为 `(x << 8) | y`。
修正 host 校验及离线 fixture 后，1,216 项 exact 读取及原拒绝路径通过，真实 16 Tile 基线读取通过后才 launch。
这次前置失败属于采集器校验错误，没有设备计算或寄存器写入。

本次取得 64 帧、每帧 16 Tile × 38 个寄存器，共 38,912 个原始字；帧间隔 10.41–11.70 ms，
单帧依次读取耗时 0.677–0.763 ms。以下均为 **board-observed**，不是原子快照或 fault-handler 入口快照：

| 字段/行为 | 本次实际观测 | 解释边界 |
| --- | --- | --- |
| exception raw | 所有 Tile 的三个 worker bank 均从 0 变为 `0x4000`；高字为 0 | 未确认 PMU raw 位义；不能直接称为 TDMA timeout 位，也不能套用 CSR 或 AP fatal 位表 |
| exception stat/mask | stat 始终 0；mask 始终为低字 `0xffffffff`、高字 `0x00ffffff` | stat 为 0 不能抵消同窗口的 AP/host TDMA 告警 |
| exception command ID | 低字为 `0xf8fc0000` 或 `0xfcfc0000`，高字为 `0xf800` 或 `0xfc00`；同 Tile 三个 bank 相同 | 字段编码、是否共享/广播及对应动态指令未知，不能当作 PC 或三个 worker 均故障的证明 |
| timeout/enable | LSU timeout 始终 `0xffff`，TDMA exception enable 为 1，first-catch 为 1，caught-exception-stop 为 0 | 未修改配置；timeout 单位及适用条件未知 |
| TDMA 进度 | 首次看到 raw 后，所有 Tile 的 worker0 count 继续增加；最终分别为 7,488、8,032 或 4,016，worker1/2 为 0 | 本次未表现为永久停滞；不能据计数恢复具体执行顺序 |
| TDMA last-command | 低字随执行改变，高字为 0 | 不能将最后一个采样值认作最初出错的 packet |
| 退出后的状态 | runner 返回后仍采到非零 raw 和 command ID | 可能是保留状态；下一次 context 是否清除及其因果关系未知，不能直接认定资源泄漏 |

相对 host runner 启动，第 26 帧在 268.181–268.888 ms 仍未见 raw；第 27 帧在 278.641–279.349 ms
已读到上述值；host kernel 在 281.750 ms 记录 `0x0D00C005`。因此取得了 host EID 之前的状态，
但该 raw 的来源、与 TDMA 告警的关系及两帧间最先发生的事件仍未知。
AP 自身时间线上，kernel dispatch 日志后约 0.372 ms 出现 fatal bit 12；AP 与 host 时钟不能直接相减。

Runner 返回 0、回读完整、检查 10,752 guard bytes，随后沿厂商流程卸载和销毁 context；AP 清理记录 `err_code:0`。
离线数值检查全量有限、relative L2 为 0.00194608，仍因设备告警拒绝资格。原始 170.612 ms 只保留审计，
不作健康性能样本；采样对总线时序的影响也未隔离。批次已停，没有第二次 launch、手工 reset 或重试。

### 现场与实际 ELF 的离线对照

在主机模拟器中执行本次 ELF 的 `entry` 及实际 Tile 分派/循环，记录 CRT 调用参数和返回位置；
NPU 计算调用被跳过，DDR 使用独立合成指针，因此这不是数值模拟或板上动态指令追踪。
16 Tile 共得到 110,272 次 TDMA 调用，各 Tile 数量均等于本轮 PMU 最终 count。
全部 87 种不同参数随后进入同 ELF 链接的 SDK GatherScatter/Memset constructor 和 TDMA issuer，
实际 MMIO 写入落在模拟器内：byte/element count、stride/iteration、inclusive end 和 control 均与参数精确相符；
GS 两端 payload 一致、地址包络不相交，GS/Memset 的地址范围均在 3 MiB 内。
这排查了本包这组参数的范围/编码问题，不证明硬件时序、依赖或 timeout 配置正确。

另有一条 **inference：待验证编码线索**：592 个非零 Tile 采样的 last-command 原始字右移 8 位后，
均能匹配本次 ELF 某个实际 TDMA destination；最终原始低字 `0x040000e0` 与最后一条的 destination `0x40000` 一致。
但多个指令复用 destination，counter 与 last-command 的更新时点未知、读取非原子，低 8 bit 和高 6 bit 位义也未确认。
因此这只能用于缩小后续对照范围，尚不能恢复唯一动态命令，更不能认定已抓到最初出错指令。

本包存在 inner 为 4 bytes、iteration 为 65,536 的广播描述符，范围及 SDK 编码检查通过；
没有故障 site、阈值单位和硬件耗时证据时，不能因数值接近 `0xffff` 就认定它触发 timeout，或据此插 join、改分块和阈值。
本轮仅离线审计，没有新增设备执行。完整参数、每 Tile 调用记录与脚本摘要归入同一实测证据。

## 下一次复现需要先补的证据

1. 在实际 TDMA handler 入口、错误状态被更改之前采集一次有界快照最可靠。
   当前发行 handler 没有这项能力；只读 PMU 访问现已验证，但约 10 ms 的采样不能代替 handler 快照。
   先确认 PMU raw/command ID 的确切编码与保留规则，或取得厂商对应 debug 固件/采集支持。
   即使在 host EID 之前采到 raw，也可能晚于出错指令退休，不能声称一定抓到了首条故障指令。
2. 同时保留 Tile、worker、异常 raw/stat/mask、命令 ID、TDMA last-command、timeout/enable、进度计数，
   并保留触发时的实际 packet 参数。参数必须来自真实 issue，不能由预期指令列表猜测。
3. 若现有寄存器不能映射回实际调用位置，再对一个当前包使用有界 issue 记录：关联实际 Tile/worker、
   动态序号、ELF site 及 packet。诊断记录本身的开销需要单列，不以完整 Trace 卡死或普通计时替代它。
4. 干净启动的单个当前失败配置已完成一次带采集复现；后续先离线核对字段与 site/packet 关联，
   依据确定证据缩小局部链，不直接重复同一采样，不重跑矩阵或历史对照包，也不改同步、timeout 或 reset 来试运气。

完成条件是拿到一次可关联实际指令的故障现场，据此修正确定缺陷，再以当前原始 case 及相应连续运行序列验证。
目前完成了安装版本核对、告警路径还原及重启后首个计算的寄存器观测；尚未取得可关联实际指令的故障 packet/PC，
**TDMA 根因与修复仍未完成**。
