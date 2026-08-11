# Firmware Kuiper Hardware SDK Reverse Engineering

This note records a reverse-engineering pass over an external, non-vendored
`firmware_kuiper` SDK snapshot. The audit checkout location is provenance, not
a stable repository path.

The conclusion is that `firmware_kuiper` is a full Kuiper hardware SDK, not a
Triton-only runtime package.  It contains host runtime, driver/firmware
installers, boot firmware, endpoint Linux payloads, system-management APIs,
validation tools, C2C/discovery tooling, DTE/SPM diagnostics, CCL/FlagCX,
multimedia/CV libraries, profiler tooling, and model/kernel sample data.

This is the canonical evidence ledger for the audited SDK/KMD/HPGR snapshot,
not a Wafer IR, ABI, provider-selection, or runtime-policy owner. Those contracts
belong to the numbered design documents linked from this directory's README. The tx8-deps-only
instruction/Kcore evidence remains in `tx8-interface-contract.md` and
`tx8-deps-reverse-engineering-reference.md`; the PyTorch eager layer is handled
separately in `txda-pytorch-runtime-wheel-analysis.md`.

The word `Triton` appears in paths and in `Runtime::IsTriton()`, but in this
SDK that flag is only a compatibility-mode selector for the old `Tsm*` runtime
bridge.  It is not the organizing model for the hardware analysis.

## 1. Evidence Model

I use four evidence levels in this document:

| level | meaning |
| --- | --- |
| Header-confirmed | Public headers define the ABI, structures, constants, or comments. |
| Binary-confirmed | ELF symbols, dynamic dependencies, strings, or targeted disassembly confirm implementation behavior. |
| Data-confirmed | SDK data files, stream maps, JSON metadata, or firmware payload layout confirm behavior. |
| Inference | Plausible hardware/runtime interpretation, but not yet a binding ABI rule without board tests or lower-level source. |

Anything marked "not proven" should not be turned into a compiler/runtime
contract yet.

## 2. Package Map

Top-level SDK layout:

| path | contents | main use |
| --- | --- | --- |
| `firmware/*.run` | Makeself installers for driver, firmware, runtime, validation suite, CCL, FlagCX, multimedia, profiler. | Deployment and board firmware provenance. |
| decrypted driver payload | KMD source, UAPI, tools, headers/libs, services, dynamic firmware. | Driver ioctl, BO, job, DTE, C2C, MHU, firmware-loading evidence. |
| `kuiper/include/tx_runtime.h` | Public HPGR runtime C ABI. | Largest host runtime surface observed in this snapshot; provider ownership remains external to this document. |
| `kuiper/lib/libhpgr.so` | Implements `tx_runtime.h`. | Device, memory, stream/event, model, module, kernel, graph, rank, tile, P2P. |
| `kuiper/lib/libvs_runtime.so` | Legacy/VS `Tsm*` compatibility runtime; depends on `libhpgr.so`. | Bridge semantics and DTE TLV evidence. |
| `kuiper/include/tsmml*.h`, `kuiper/lib/libtsmml.so` | System-management API. | Device/topology/C2C/power/thermal/PCIe/process state. |
| `kuiper/bin` | Runtime tests, validation suite, DTE/copy/discovery/SMI tools, CCL perf, multimedia tests. | Concrete usage and diagnostics. |
| `kuiper/data` | Model metadata, stream configs, kernel `.so`, PG/16-tile validation payloads. | Golden inputs and model/stream structure evidence. |
| `kuiper/include/nccl.h`, `tccl.h`, CCL libs | NCCL/TCCL-compatible collectives using `txStream_t`. | Distributed runtime/collective layer. |
| `kuiper/include/tsmcodec*.h`, `tsmcv.h`, multimedia libs | VPU decoder/encoder, CV/G2D, FFmpeg/GStreamer backend. | Multimedia hardware SDK surface. |
| profiler package | `tsmprof-sys`, `tsmtx`, profiler register/runtime pieces. | Host/device profiling and annotation tooling. |

Important dynamic dependency evidence:

| library/tool | dependency or symbol evidence |
| --- | --- |
| `libhpgr.so` | Standalone HPGR runtime, exports public `tx*` and internal `itx*`/`tx::runtime::*`. |
| `libvs_runtime.so` | NEEDED `libhpgr.so`; maps many `Tsm*` calls to HPGR, but also has direct DTE register helper paths. |
| `libtccl.so`, `libnccl.so` | Public NCCL/TCCL ABI uses `txStream_t`; unresolved `tx*` runtime symbols are expected from process/runtime linkage. |
| `libflagcx.so` | NEEDED `libhpgr.so`; strings show socket/IB/topology/C2C/route/heterogeneous adaptor logic. |
| `libtsmml.so` | Uses `/dev/accel_drv_mgr`, `/dev/mem`, `/sys/bus/pci/devices`, and TSM driver/C2C/topology ioctl strings. |
| `libtsmcodec.so` | Uses `/dev/vdec*`, `/dev/venc*`, `/dev/mem`; has Verisilicon `VC9000D/VC9000E` strings. |
| `libtsmcv.so` | Uses `/dev/g2d*`, `GC820/G2D` strings and SDK allocator/DMA helpers. |

## 3. Firmware, Driver, and Boot Chain

### 3.1 Installers

The `.run` files are Makeself installers.  The package roles are:

| installer | role |
| --- | --- |
| `Tsm_driver_5.6.0.1231_x86_64_installer.run` | Host driver installer. |
| `Tsm_firmware_EVB/REX1008/REX1032_5.6.0.1231_x86_64_installer.run` | Board firmware payloads. |
| `Tsm_runtime_5.6.0.1231_x86_64_installer.run` | HPGR/VS runtime headers, libs, tests, data. |
| `Tsm_validation_suite_5.6.0.1231_x86_64_installer.run` | `tsmvs`, DTE/copy/discovery/SMI diagnostics. |
| `Tsm_ccl`, `Tsm_flagcx` | Collective libraries and network/distributed adaptors. |
| `Tsm_multimedia` | FFmpeg, codec, CV/G2D/VPU SDK. |
| `Tsm_profiler` | Profiler tooling and register/preload components. |

The outer driver package contains `driver_payload.tar.gz.enc` and
`npu_driver_install.sh`.  Despite the suffix, `npu_driver_install.sh` is an
ELF64 x86-64 PIE executable, not a shell script. The encrypted payload was
decrypted and extracted during the audit. It contains KMD source, public UAPI, unit tests, host tools,
systemd services, TSMML headers/libs, dynamic AP/Kcore/Score firmware, and
driver installer logic.  Therefore driver behavior in this document is not only
inferred from user-space tools; several core pieces are UAPI/source-confirmed.

### 3.2 Firmware Package Contents

Each firmware package contains:

```text
firmware/Image
firmware/driver.run
firmware/fip-pcie-bl2.bin
firmware/fip-pcie.bin
firmware/flash_ut.sh
firmware/rootfs.img.gz
firmware/spi_flash_package.bin
firmware/spi_flash_package_crc_file_v5.6.0.1231.bin
firmware/tsm-flash
```

EVB and REX1008 also include `firmware/mcu/runtime.bin`; REX1032 does not in
this package.

Hash comparison shows that `Image`, `driver.run`, `flash_ut.sh`, and
`tsm-flash` are identical across EVB/REX1008/REX1032.  Board-specific payloads
are `fip-pcie-bl2.bin`, `fip-pcie.bin`, `rootfs.img.gz`,
`spi_flash_package.bin`, and the CRC-wrapped SPI flash image.  That means board
type affects early boot firmware, rootfs image, and flash image, while the host
flasher and host driver installer are shared.

`setup_fw_env.sh` installs firmware files into both:

```text
/usr/local/kuiper/driver/firmware
/lib/firmware/accel/flash
```

Unless invoked with `-no_update`, it then finds `*-flash` and
`spi_flash_package_crc_file*.bin` and runs:

```text
tsm-flash -f spi_flash_package_crc_file_v5.6.0.1231.bin
```

### 3.3 `tsm-flash`

`tsm-flash` is a host-side firmware update tool.  Binary evidence shows it:

- opens PCI devices under `/sys/bus/pci/devices`;
- checks vendor/device strings including vendor `0x200c`;
- dynamically opens `libtsmml.so`;
- calls `tsmml_init`, `tsmml_device_get_handle_by_index`, and shutdown paths;
- opens `/dev/tsm/txnpu`;
- uses `mmap` and `ioctl`;
- reads XIP/SPI firmware versions;
- parses firmware headers and CRC32 values;
- supports `-f/--file`, `-a/--addr`, `-l`, `--force`, and device index/bus
  selection;
- triggers upgrade interrupt through `tsm_flash_trigger_upgrade_interrupt`;
- waits for device-ready and update-end states, with a visible three-minute
  timeout string;
- can invoke `driver.run install/uninstall` when firmware is too old for the
  normal update path.

Important update semantics:

| behavior | evidence |
| --- | --- |
| Version gate | Strings include "same version, skip", XIP/SPI version reporting, and PCIE boot force-update warnings. |
| CRC gate | Header CRC and file CRC are checked; `--force` can continue through mismatch paths. |
| Device synchronization | Update waits for device ready and device end states before reporting result. |
| Boot mode awareness | PCIE boot mode requires `--force`; XIP upgrade can be unsupported on some devices. |

### 3.4 Boot Firmware

`fip-pcie-bl2.bin` contains ATF BL2/trusted-boot strings and board bring-up
logic:

- board string `Tsingmicro Kuiper EVB Board`;
- compatible family `tsingmicro, kuiper`;
- secure boot / ROTPK / SM2 / SM3 / RSA-PSS strings;
- PCIe lane count, boot option/mode, SR-IOV capability, link-up polling, DMA
  polling;
- firmware train success/failure;
- reset causes including software, PCIe/power-on, host PCIe reset, watchdog;
- `partial_good` parsing and enable/disable strings;
- tile, PCIe, VPU, GPIO status reporting.

`fip-pcie.bin` contains BL31/PSCI/U-Boot/device-tree logic:

- `kuiper_system_reset`, `kuiper_system_off`, and `kuiper_topology.c`;
- U-Boot environment including `bootcmd=booti 0x81400000 - 0x81000000`,
  SPI flash reads for kernel/rootfs, and `Uncompressing rootfs`;
- DTB fixup for board, DDR capacity, ECC/debug/VPU flags;
- compatible nodes such as `tsingmicro,kuiper-t8`, `kuiper,tsmnpu`,
  `kuiper,pcie-test`, `kuiper,ipc`, `kuiper,dpdkipc`,
  `tsingmicro,kuiper-t8-pcie-ep`;
- device-tree fields such as `tile-index`, `board_power_threshold`,
  `peak_power_threshold`, and average power thresholds.

The packaged `Image` is Linux `5.10.123` for AArch64, built with GCC 10.3.

### 3.5 Endpoint Rootfs Modules

The REX1032 `rootfs.img.gz` decompresses to an ext4 image.  Extracted module
evidence:

| module | role |
| --- | --- |
| `kuiper_flash.ko` | SPI flash update driver; `modinfo` description is "SPI Flash updata driver". |
| `kuiper_remount.ko` | PCIe BAR polling and filesystem switch module. |
| `spi-dw-mmio.ko` | DesignWare SPI MMIO driver. |

`kuiper_flash.ko` strings show SPI read/write/erase/update, CRC verification,
DDR-to-flash, page-aligned write requirements, soft write protect, `mmap`, and
`unlocked_ioctl`.

`kuiper_remount.ko` strings show:

- `/tmp/new_rootfs.img`, `/mnt/rootfs`, `/mnt/rootfs/dev/mem`;
- BAR2/BAR4 base reporting;
- MHU callback registration;
- `ddr_to_flash`;
- filesystem switch and upgrade environment readiness;
- PCIe polling thread;
- rootfs remount and switch-root style behavior.

This indicates the firmware update path is not just a host-side flash writer.
There is an endpoint Linux component that can receive an upgrade/rootfs image
over PCIe/BAR/MHU-mediated control, switch rootfs, and then write SPI flash.

### 3.6 Driver KMD and UAPI

The decrypted driver payload contains the main UAPI at
`src/refine/uapi/tsm_uapi.h` relative to that payload.

It also contains KMD sources under `src/refine`, legacy/low-level NPU sources
under `src/txnpu`, and VPU/G2D sources under `src/txvpu`.

Device-node evidence has two families:

| family | nodes / files | role |
| --- | --- | --- |
| Refined driver | `/dev/accel/dev-N`, `/dev/accel_drv_mgr` | Main KMD device and driver-manager UAPI. |
| Legacy/config path | `/dev/tsm/txnpu`, `/dev/tsm/txnpu_ctrl`, `/dev/npu-config` | Legacy tools, config, and some discovery/flash paths. |

No udev rule is required in the payload evidence; nodes are created by driver
miscdevice paths and associated services/tools.

UAPI constants:

| constant | value |
| --- | ---: |
| `TSM_API_VERSION` | 1 |
| `TSM_DEVICE_MAX_TILE_NUMBER` | 16 |
| `TSM_DEVICE_C2C_PORT_NUMBER` | 8 |
| `TSM_C2C_LINK_NUMBER` | 4 |
| score logs | `16 * 2` |
| Kcore logs | `16` |
| AP logs | `5` |

Top-level ioctl command groups use `_IOWR('T', nr, ...)`:

| nr | command | UAPI struct | purpose |
| ---: | --- | --- | --- |
| 0 | `TSM_BO_CMD` | `tsm_bo_ioctl_args` | BO create/import/query. |
| 1 | `TSM_JOBS_CMD` | `tsm_jobs_ioctl_args` | Compute/DMA/VPU/FW jobs and fences. |
| 2 | `TSM_NPU_CMD` | `tsm_npu_ioctl_args` | Tile memory get/set. |
| 3 | `TSM_C2C_CMD` | `tsm_c2c_ioctl_args` | C2C discovery/link/port/firewall/MAC/link-check BO. |
| 4 | `TSM_LOG_CMD` | `tsm_log_ioctl_args` | AP/Kcore/Score log descriptors and levels. |
| 5 | `TSM_DEVICE_INFO_CMD` | `tsm_device_info_ioctl_args` | Memory/PCIe/NPU/DMA/task info. |
| 6 | `TSM_DRIVER_DEVICES_TOPO_CMD` | `tsm_driver_devices_topo_ioctl_args` | Driver-level mesh/torus topology and discovery status. |
| 7 | `TSM_DRIVER_INFO_CMD` | `tsm_driver_info_ioctl_args` | Driver/vendor/project strings. |
| 8 | `TSM_DRIVER_SET_DTE_CMD` | `tsm_driver_set_dte_trans_ioctl_args` | Driver-level DTE transfer by card and destination XY. |

#### BO and Memory Domains

BO creation has three input forms:

| op | semantics |
| --- | --- |
| `TSM_BO_CREATE` | Allocate a BO with requested size, physical alignment, domain, pool, and flags. |
| `TSM_BO_CREATE_USERPTR` | Import/pin a user CPU pointer for host-side memory. |
| `TSM_BO_CREATE_IMPORT` | Import an existing local-DRAM device address on the same device. |
| `TSM_BO_QUERY` | Query BO physical address, size, device id, domain, and pool. |

BO domains:

| domain | meaning |
| --- | --- |
| `TSM_BO_LOCAL_DRAM` | Device-side memory. |
| `TSM_BO_REMOTE_DRAM` | Host-side memory, including scatter-gather/pinned BO usage. |

BO pools:

| pool | meaning |
| --- | --- |
| `TSM_BO_POOL_NPU_BIN` | Kcore/NPU binary storage. |
| `TSM_BO_POOL_VISIBLE` | BAR2 visible memory. |
| `TSM_BO_POOL_NPU_NORMAL` | Normal device allocation pool. |
| `TSM_BO_POOL_VISIBLE_EXTENDED` | VF BAR4 visible/extended memory. |
| `TSM_BO_POOL_LOG` | Log buffer pool. |

BO flags:

| flag | meaning |
| --- | --- |
| `TSM_BO_CPU_ACCESS` | CPU-visible or host-memory access. |
| `TSM_BO_ONLY_READ` | Read-only allocation/import intent. |

`tsm_device_memory_info` separates total local DRAM, remote SG/pinned BO usage,
NPU binary pool usage, NPU normal pool usage, largest contiguous normal range,
and visible-extended pool usage.  This confirms that runtime memory cannot be
modeled as one flat "device memory" pool.

#### Jobs, DMA, Fences, and Compute Completion

Job types:

| type | meaning |
| --- | --- |
| `TSM_JOB_COMPUTE` | Kernel/model compute packet via KCQ/MHU. |
| `TSM_JOB_DMA` | PCIe DMA copy. |
| `TSM_JOB_VPU` | VPU job. |
| `TSM_JOB_FW` | Firmware upgrade job. |

Job flags:

| flag | meaning |
| --- | --- |
| `TSM_JOB_EXPLICIT_SYNC` | Explicit sync dependencies. |
| `TSM_JOB_IMPLICIT_SYNC` | Implicit sync mode. |
| `TSM_JOB_CROSS_DEVICE` | Cross-device job; DMA code checks visible/cross-device BO state. |
| `TSM_JOB_DEBUG` | Debug job flag. |

`tsm_job_args` includes BO-list count, dependency fence count/array, timeout
in milliseconds, blocking flag, private job args, and output `done_sync_file`.
If timeout is not specified, KMD defaults to 10 seconds.  `tsm_jobs_ioctl_args`
supports batch mode only when all jobs are DMA jobs and have the same direction.

DMA job direction is explicitly only:

| enum | direction |
| --- | --- |
| `TSM_DMA_DEVICE_TO_HOST` | D2H |
| `TSM_DMA_HOST_TO_DEVICE` | H2D |

The DMA implementation has linked-list and non-linked-list modes, writes SAR,
DAR, transfer size, control, LLP, and doorbell registers, and signals a DMA
fence from the DMA scheduler path.

Compute jobs use KCQ and MHU doorbells.  The source currently contains a
critical compatibility note: due to missing AP firmware interrupt support for
compute-done notification, KMD directly signals the compute done fence after
kicking the doorbell. The fence therefore does not by itself prove device-side
model/kernel completion in this snapshot; production mapping belongs to the
numbered runtime and verification designs.

The queue path is more specific than a generic "submit and fence" model:

| path | recovered behavior |
| --- | --- |
| KIQ | A visible local-DRAM, CPU-accessible BO holds 256 packets plus read/write pointers.  KMD writes its base/size to PCI config, copies a packet into the current slot, advances `wptr`, kicks the KIQ MHU doorbell, then polls `packet_response` every 50 ms for `0xCAFEBEEF` success or `0xBEEFDEAD` failure. |
| KCQ compute | `tsm_kcq_job_prepare` only accepts kernel/model packet types.  Submit picks the kernel or model doorbell, kicks MHU, and immediately signals the Linux fence in this driver snapshot. |
| context flush | Multiple device contexts are explicitly not supported yet; the destroy-context KIQ packet carries `ctx_id`, but comments say firmware currently ignores it. |

Kcore firmware strings show that device-side completion machinery does exist:
`process_model_packet`, `process_packet`, `notify_calc_finish`,
`kcore_report_msg_done`, `ringbuffer ready`, and per-block kernel-complete log
strings are present in the dynamic Kcore image.  Therefore the KMD fence issue
is best understood as a host-driver/AP-firmware integration limitation in this
snapshot, not proof that the hardware has no model/kernel completion signal.

MHU evidence:

| doorbell | index |
| --- | ---: |
| compute memory | 0 |
| compute model | 1 |
| compute kernel | 2 |
| KIQ | 3 |
| firmware upgrade | 10 |

- 31 doorbell bits are configured for AP boot/control;
- MHU doorbell kick writes sender channel set bits and currently polls for
  completion instead of using IRQ for that path;
- later MHU access is negotiated through access request/ready registers.

#### Driver Firmware Loading

The refined KMD firmware loader requests firmware through Linux
`request_firmware()` from two locations:

| path prefix | firmware names |
| --- | --- |
| `accel/flash/` | `fip-pcie-bl2.bin`, `fip-pcie.bin`, `Image`, `rootfs.cpio.gz` variants. |
| `accel/dynamic/` | `score0.tufw`, `score1.tufw`, `kcore.tufw`, dynamic rootfs payload. |

The decrypted payload also carries:

```text
firmware/rootfs.cpio.gz
firmware/kcore.tufw
firmware/score0.tufw
firmware/score1.tufw
firmware/kcore_fw.bin
firmware/score0.bin
firmware/score1.bin
```

This shows a split between board flash images and dynamically loaded AP/Kcore
or Score firmware used by the host driver.

#### Driver DTE UAPI and Register Layout

The refined driver DTE source confirms a 16-DTE / 16-tile model:

| constant | value |
| --- | ---: |
| `NPU_KMD_USE_DTES_COUNT` | 16 |
| `NPU_KMD_DTE_CONTROLLER_MAX` | 16 |
| `NPU_MAX_TILES` | 16 |
| per-controller channel records | 4 |
| DTE channel register size | `0x200` |

DTE controller register offsets from the driver source:

```text
0x00401000, 0x00c01000, 0x01401000, 0x01c01000,
0x02401000, 0x02c10000, 0x03401000, 0x03c01000,
0x04401000, 0x04c01000, 0x05401000, 0x05c01000,
0x06401000, 0x06c01000, 0x07401000, 0x07c01000
```

The second-row value `0x02c10000` is written that way in source and should not
be silently "fixed" without hardware confirmation.

Driver tile XY table:

```text
0x0000, 0x0001, 0x0002, 0x0003,
0x0100, 0x0101, 0x0102, 0x0103,
0x0200, 0x0201, 0x0202, 0x0203,
0x0300, 0x0301, 0x0302, 0x0303
```

Declared driver DTE modes:

| mode | value |
| --- | ---: |
| unicast | 0 |
| scatter | 1 |
| broadcast | 2 |
| shuffle | 3 |
| gather | 4 |

This list must be read by layer.  The KMD `dte_mode` register field is only
2 bits wide and the KMD source switch handles unicast/scatter/broadcast/shuffle
only.  `gather = 4` is declared in the driver enum, but this KMD register path
does not encode or dispatch it.  The lower Kcore/direct-DTE software mode list
recovered from `tx8_deps` is wider and includes RDMA, WDMA, DDR2DDR unicast,
and DDR2DDR shuffle.  Treat the driver enum, hardware register field, and
Kcore/direct-DTE mode enum as related but non-identical layers.

Mode bit behavior in the KMD register path:

| mode | register behavior |
| --- | --- |
| unicast | `mode=0`, `sg_flag=1`, stride/iteration cleared. |
| scatter | `mode=1`, `sg_flag=1`, stride/iteration cleared. |
| broadcast | `mode=2`, `sg_flag=0`. |
| shuffle | `mode=3`, `sg_flag=0`, three stride/iteration pairs are written from stream config. |

`out_slice_flag` is always cleared in the KMD path; the source comment says the
first implementation does not support shuffle-to-shuffle.  The generated
`tx8_deps` DTE register headers expose additional bits not used by this KMD
helper, including `mem_bypass`, `dim_flag`, scatter/shuffle exception fields,
read-turbo fields, and an `early_response` bit.

`dte_user_id` bitfields:

| bits | field |
| --- | --- |
| 0..5 | stream id; comments say 0..31 DDR and 32..63 SRAM. |
| 6 | early complete. |
| 7 | target NPU. |
| 8 | switch DDR. |
| 9 | `rv_n`. |
| 10..14 | packet id. |
| 15 | stream transaction. |

`tsm_driver_set_dte_trans_ioctl_args` is the driver-level send-by-XY contract:

| field | meaning |
| --- | --- |
| `card_id` | Source card id sorted by BDF. |
| `dte_index` | Source DTE/tile index. |
| `dstx`, `dsty` | Destination card/tile logic XY coordinate. |
| `len` | Transfer length. |
| `early` | Early response flag. |
| `src_addr`, `dst_addr` | Source and destination addresses. |

The driver rounds transfer length up to an 8-byte multiple.  For local tile
targets it adds `(dsttilexy << 40)` to the destination address.  For non-local
targets it additionally sets bit 39:

```text
remote_dst = dst_addr + (1ULL << 39) + ((uint64_t)dsttilexy << 40)
```

That is a UAPI/source-confirmed cross-card addressing rule for this driver
DTE path.

The KMD convenience ioctl is a synchronous, unicast-oriented path: it selects a
DTE controller/channel, writes source/destination/user-id/mode/length, triggers
`cmd_valid`, then polls status bitfields for `dma_done` or `dma_error` for up
to about 200 ms.  It rounds only to 8 bytes and does not enforce a 64 KiB SPM
address rule.

DTE availability is tied to partial-good state.  The driver reads a SYS_CTRL
tile-good bitmap and a partial-DTE flag.  If partial DTE is set, the maximum
DTE controller count is reduced from 16 to 8.  If the requested DTE/tile is bad
or busy, the driver tries another good unused DTE and logs a bad-tile warning.

#### C2C UAPI

Driver C2C UAPI covers:

- discovery flag/info/NPU info;
- link info set/get;
- per-port info set/get;
- MAC reset and MAC address get/set;
- link-check BO reservation.

`tsm_c2c_port_info` exposes firewall, PCS, PMA, MAC, overall status, RTT
latency in nanoseconds, total bandwidth in Mbps, and mesh card id.  Discovery
state carries mesh/torus topology, local/global card coordinates, host offset,
neighbor IP/card data, down counters, and C2C phase.

This is stronger evidence than the SMI strings alone: C2C is a first-class KMD
UAPI with 8 ports, 4 directional links, mesh/torus topology, firewall state,
latency/bandwidth, and MAC management.

## 4. Hardware Topology and Numbering Spaces

The SDK exposes multiple numbering spaces with different fields and lookup
paths; the snapshot does not prove that they are interchangeable. Typed identity
and joins belong to the numbered topology/runtime designs.

| space | evidence | meaning |
| --- | --- | --- |
| Runtime device id | `txSetDevice`, `txGetDevice`, `txDeviceList.deviceIds`. | Host runtime ordinal. |
| PCIe BDF | `domain:bus:device.function` in `tx_runtime.h` and `tsmml_pcie_info_t`. | Physical PCIe identity. |
| Mesh id | `txDevProp.meshId`, `tsmml_device_t.mesh_id`. | Toolchain/cluster topology identity. |
| Slot/user/soft id | `tsmml_device_t.slot_id/user_id/soft_id`. | Board/infrastructure identifiers. |
| Tile index | 0..15 in runtime tile arrays. | Logical tile index inside a device. |
| Tile physical coordinate | `phyTilex`, `phyTiley`; stream maps also encode 4x4 coordinates. | Physical 4x4 tile placement. |
| C2C direction | east/west/south/north enum values 0..3. | Link direction, not rank or tile id. |
| Rank | `txGet/SetSysPhyRank*`, TCCL/NCCL communicator rank. | Distributed process/device order. |

Header-confirmed limits:

| constant | value |
| --- | ---: |
| `DEVICE_COUNT_MAX` | 32 |
| `NPU_TILE_COUNT_MAX` | 16 |
| `NPU_PG_TILE_COUNT` | 8 |
| `TSMML_MAX_TILE_NUM` | 16 |
| `TSMML_MAX_C2C_NUM` | 8 |
| `TSMML_MAX_LINK_TARGET_NUM` | 4 |
| `TSMML_MAX_DEVICE_NUM` | 32 |
| `TSMML_PG_MAX_DEVICE_NUM` | 16 |

`tx_runtime.h` models PG devices explicitly: `txSetDeviceSelectedTileInfo`
selects exactly 8 functional physical tiles.  `libhpgr.so` disassembly confirms
the implementation copies eight tile records and derives `logicIdStart` from
the first selected tile as `(x << 8) + y`.

The model/stream data is 4x4-tile oriented.  `StreamConfigMap.txt` uses
`TileNum 16`, `GroupXlen 4`, `GroupYlen 4`; tile identifiers include patterns
like:

```text
0, 1, 2, 3,
65536..65539,
131072..131075,
196608..196611
```

That is a separate tile-coordinate encoding in the stream map, likely
`row << 16 | col`.  It is not the same as HPGR's `(x << 8) + y` PG
`logicIdStart` value.

### 4.1 Address Space, BARs, and Reserved Regions

The address-space picture is now source-confirmed at several layers:

| region | recovered semantics |
| --- | --- |
| per-tile SPM | `L1SPM_BASE = 0x0`, size `0x300000` bytes.  KMD tile-memory get/set ioctls are restricted to this 3 MiB SPM window. |
| per-tile register window | KMD spaces tile register groups by `0x800000`.  Important internal offsets include DTE `0x400000`, SCONFIG `0x500000`, tile CRG `0x600000`, MHU blocks around `0x680000..0x691000`, ADDR_MAP `0x6A0000`, broadcast control `0x6C0000`, and tile misc `0x6E0000`. |
| tile identity registers | KMD reads logic id at `0x6A0058`, physical id at `0x6A005C`, and chip id at `0x6A0064`. |
| DTE remote tile encoding | `tx8_deps` uses the same form as KMD: remote tile address adds `(1 << 39) + (tile_id << 40)`. |
| DDR spaces | `tx8_deps` names DDR base `0x80000000`, uncached/weak-order DDR `0x180000000..0x27fffffff`, and external DDR at `0x8000000000`. |

PCIe BAR/ATU behavior is also explicit in KMD:

| BAR / mapping | semantics |
| --- | --- |
| BAR0 | Direct MMIO. |
| BAR2 | Visible window; KMD maps at most 32 MiB as the small BAR case and uses device offset `0x1F6000000` for visible BO addresses. |
| BAR4 | Indirect/MMIO aperture programmed through inbound ATU entries.  Entries cover MHU, four NPU tile windows, C2C ports, DDR controllers, VPU subsystems, PCIe PHY debug, SYS_CTRL, log descriptor, and panic trace regions. |
| large BAR | `is_large_bar` is detected by comparing visible memory size with 32 MiB.  Large-BAR `VISIBLE_EXTENDED` aliases the front of normal NPU memory; small-BAR `VISIBLE_EXTENDED` is a separate VF BAR4-backed range. |

KMD BO pools are not abstract labels; several have fixed address ranges:

| BO pool | recovered address behavior |
| --- | --- |
| `NPU_BIN` | `0x110000000..0x17cffffff` is KMD-managed Kcore binary storage; `0x17d000000..0x17fffffff` is reserved for Score binary/AP firmware ownership. |
| `VISIBLE` | `0x00400000..0x007fffff`, with BO device addresses offset through the BAR2 visible device base. |
| `LOG` | `0x180000000..0x187ffffff`. |
| `NPU_NORMAL` | Late-initialized from AP/PCI config DDR base and size; in the small-BAR case it starts after a 1 GiB front reservation. |
| `VISIBLE_EXTENDED` | Small-BAR case uses `0x00100000..0x3fffffff` through VF BAR4; large-BAR case aliases a front slice of `NPU_NORMAL`. |

Dynamic RISC-V firmware uses the `NPU_BIN` region in a fixed tile-indexed
layout.  Each Kcore image gets a 109 MiB slot at
`0x110000000 + tile_id * 109 MiB`; Score0 and Score1 follow after the 16 Kcore
slots.  KMD startup packets use the tile-good bitmap as `active_bitmap`, set
core id 1 for Kcore and 2/3 for Score0/Score1, and pass the per-core binary
address/run size.

The global stream-mapping table is copied by KMD to `0x17F000000`.  Its layout
is:

```c
struct tsm_stream_mapping_table {
  uint32_t card_num;
  struct card_info cardInfo[32];
};
```

Each card record carries `chipId`, `tile_num`, `good_bitmap`, and 16 tile
records containing tile index plus physical X/Y.

### 4.2 Partial-Good Devices

Partial-good state appears in boot firmware, KMD, HPGR, and data payloads:

| layer | behavior |
| --- | --- |
| boot/SYS_CTRL | BL firmware leaves tile-good state in reserved SYS_CTRL registers.  KMD reads `PG_TILE_INFO_OFFSET` for the good-tile bitmap and a separate NOC/DTE field for partial-DTE state. |
| KMD NPU | Only good tiles are initialized/exposed for tile memory and tile info.  A bitmap of `0` or `0xffff` is treated as all 16 tiles good in the DTE init path. |
| KMD stream table | Driver builds a 32-card global tile mapping table from all devices' NPU info and notifies firmware with a KIQ global-info packet. |
| HPGR | `TxModelMgr::getEngineDevice` accepts only 16-tile or 8-tile device maps.  For PG status it falls back to default `tilex=2`, `tiley=4`, `logic_id_start=0`; otherwise it uses discovered geometry and logic id start. |
| SDK data | `pg_8tile/config_stream.bin` starts with `1,1,8,2,4`; `unicast_16tile_address/config_stream.bin` starts with `1,1,16,4,4`. |

`txPgModel` adds `bin_map_addr/bin_map_length` to the normal model descriptor.
The HPGR PG load path validates each per-tile Kcore firmware image against the
same 109 MiB per-tile slot size, allocates an `NPU_BIN` buffer for
`kcore_fw_num * 109 MiB`, loads each tile firmware image into its slot, and
sends a model-manager command containing both the engine device map and the bin
map memory information.

## 5. Observed HPGR `tx_runtime.h` Interface

`tx_runtime.h` is the public CUDA-like host runtime ABI in this SDK.  It covers
device management, memory, stream/event, model/module/kernel/graph, rank, tile
selection, and P2P.

### 5.1 Handles and Core Structures

Opaque handles:

```c
typedef void *txModule_t;
typedef void *txFunction_t;
typedef void *txStream_t;
typedef void *txEvent_t;
typedef void (*txHostFn_t)(void *userData);
```

Key public structures:

| structure | semantics |
| --- | --- |
| `txDeviceList` | Device count plus up to 32 runtime device ids. |
| `memInfo`, `txMemProp` | Device memory regions: NPU, VPU, Kcore. VPU/Kcore fields are marked future-deleted. |
| `tilePhyInfo` | Tile index plus physical X/Y coordinate. |
| `tileFullInfo` | Tile index, availability flag, physical X/Y. |
| `tileProp` | Tile count, `logicIdStart`, and tile physical info array. |
| `txDevProp` | Device name, mesh id, PCI domain/bus/device/function. |
| `txDeviceProperty` | Device identity, memory layout, and tile topology. |
| `tileTotalInfo` | All 16 tile records. |
| `tileSelectedInfo` | Exactly 8 PG selected tile records. |
| `txModel` | Stream config address/length, param address/length, `ldmem_len`, Kcore firmware count, flexible `kcore_info[]`. |
| `txPgModel` | `txModel` plus `bin_map_addr/bin_map_length` for PG model launch. |
| `txMemcpyBatchParam` | One copy descriptor: `dst`, `src`, `sizeBytes`. |

Error code ranges:

| range | family |
| --- | --- |
| `0x41000001...` | Model/AP-side load/run/tile/config errors. |
| `0x42000001...` | Memory and memcpy errors. |
| `0x43000001...` | Device discovery/property/reset errors. |
| `0x44000001...` | System/logger errors. |
| `0x45000001...` | Module/function/kernel/model/graph errors. |
| `0x46000001...` | Generic invalid value, OOM, invalid handle, not ready, device failed. |

### 5.2 Initialization and Dispatch

`libhpgr.so` implements public `tx*` functions as dispatch-table thunks into
`tx::runtime::*` / `itx*` implementations.  Binary-confirmed startup behavior:

- `std::call_once(itxInit)`;
- raises `RLIMIT_NOFILE`;
- initializes logger and device manager;
- sets default device 0;
- registers signal and `atexit` handlers;
- stores TLS last error;
- returns `0x44000002` on logger init failure and `0x43000003` on device-manager
  init failure.

The dispatch table is about `0x1c0` bytes and covers device, memory,
stream/event, module/kernel/model/graph, rank/tile, P2P, and deprecated Kcore
power functions.

### 5.3 Device, Tile, and Rank APIs

Header and disassembly agree on these semantics:

| API family | semantics |
| --- | --- |
| `txSetDevice`, `txGetDevice`, `txDeviceReset` | Thread-local/current device selection; default device is 0 if never set. |
| `txGetDeviceCount`, `txGetDeviceList` | Enumerate available initialized devices. |
| `txGetDeviceByPCIBusId`, `txGetDevicePCIBusId` | Supports full and shortened PCI BDF formats. |
| `txGetDeviceProperty` | Returns name, mesh id, PCI identity, memory regions, tile properties. |
| `txGetDeviceAllTileInfo` | Returns 16 tile records including availability and physical coordinates. |
| `txSetDeviceSelectedTileInfo` | PG mapping; exactly 8 selected physical tiles. |
| rank APIs | System physical rank size/id are runtime-visible and used by tests. |

`txDeviceSynchronize()` is implemented in HPGR as all-stream synchronization for
the current device.  Do not confuse it with old `TsmDeviceSynchronize()` in
`libvs_runtime.so`, which is a no-op success in this build.

The public `txDeviceReset()` contract applies to the entire selected target
device and all of its cards, returning it to the initial power-on state. The
current `itxDeviceReset` path first synchronizes and destroys all streams, then
releases model, module, and memory-manager state. A deeper reset ioctl was not
proven in this pass, so the public contract remains the conservative boundary:
all existing stream/module/model/allocation handles become invalid, and reset
may itself block while waiting for outstanding streams. It is not a per-kernel
cancellation or invocation cleanup primitive.

### 5.4 Memory and Copy APIs

`txMemcpyKind` values:

| value | direction |
| ---: | --- |
| 0 | Host to host |
| 1 | Host to device |
| 2 | Device to host |
| 3 | Device to device |

Supported behavior:

| API | behavior |
| --- | --- |
| `txMalloc`, `txFree` | Real device allocation/free through `TxMemMgr`. |
| `txMallocHost`, `txFreeHost` | Real host/page-locked allocation/free path. |
| `txMemcpy` | H2D, D2H, D2D supported; H2H rejected. |
| `txMemcpyAsync` | H2D, D2H, D2D queued on stream; H2H rejected. |
| `txMemcpyBatch`, `txMemcpyBatchAsync` | H2D/D2H only; D2D/H2H rejected. |
| `txMemGetInfo` | Calls driver memory-info path for current device. |
| `txSend`, `txRecv` | Runtime P2P API; peer info is documented as `phyTilex << 8 \| phyTiley`. |

Binary-confirmed limits in this build:

- batch count must be 1..16;
- each batch item is capped at `0x100000` bytes;
- common memory errors map to the `0x420000xx` family.

### 5.5 Stream and Event APIs

`libhpgr.so` implements:

- `txStreamCreate`, `txStreamDestroy`, `txStreamSynchronize`, `txStreamQuery`,
  `txStreamWaitEvent`;
- `txEventCreate`, `txEventDestroy`, `txEventRecord`, `txEventQuery`,
  `txEventSynchronize`, `txEventElapsedTime`.

Streams own command queues.  Same-stream memory/model/kernel commands are
ordered by command dependencies and marker/event objects.  This is a runtime
ordering guarantee; it is not proof of SPM bank safety or NE/CT/DTE hardware
parallelism safety.

A stream is not a tile selector. `txLaunchKernel` carries a function, one
argument blob, grid/block dimensions, shared-memory bytes, and a stream, but no
logical or physical tile identifier. `txStreamQuery` returns
`TX_ERROR_NOT_READY` for normal pending work; that value must not be promoted to
a device/context failure. A host deadline can therefore be built from bounded
query polling, but the runtime exposes no cancellation operation: query error
or deadline leaves the invocation quarantined and forbids blocking stream
destroy, module unload, memory free, reset, or power calls in that context.

### 5.6 Module, Kernel, Model, and Graph APIs

Implemented APIs include:

| API | recovered semantics |
| --- | --- |
| `txModuleLoad` | Parses code object, copies device resources, registers module. |
| `txModuleGetFunction` | Looks up named kernel function from module. |
| `txModuleUnload` | Releases module/code-object state. |
| `txLaunchKernel` | Enqueues `NDRangeKernelCommand`. |
| `txLaunchClusterKernel` | Cluster/C-intrinsic style launch. Header constrains cluster dim to `(1,1,1)`, grid x to tile count, one block per tile. |
| `txLaunchKernelGGL`, `txLaunchClusterKernelGGL` | One-shot load/get-function/launch wrappers. |
| `txLaunchModel` | Enqueues a `CusModelCommand` on a stream. |
| `txLaunchModelSync` | Synchronous compiled-model run; validates BPM/stream-config device memory. |
| `txLoadGraph`, `txUnloadGraph` | Graph asset load/unload into module object manager. |

Two ownership/ABI details are material for an all-rank provider:

- `TxModuleObjectMgr::loadCodeObject` deduplicates code objects by a 64-bit MD5
  key and can return the same module handle for repeated identical ELF bytes.
  The matching unload path has no recovered logical reference increment.
  A caller that loads 16 identical rank modules must therefore keep its own
  handle-to-digest logical ownership count and issue the real unload only for
  the final owner; returning one handle for different digests is an
  untrustworthy provider contract violation.
- The current `TxModuleObjectMgr::loadGraph` always reads
  `graphPath/tile0/kcore_fw.so` through `tile15/kcore_fw.so`. Its private
  `graphInfo` record has one shared module name, one shared symbol, and 16
  size/address entries. It constructs a zero-I/O boot parameter whose dynamic
  data is one type-6 `DYNLIB_LOAD` TLV, sends it through the outer type-5
  `MODULE_MODEL_LAUNCH_PACKET`, waits synchronously, and then registers the
  graph. The outer packet is only an envelope: `txLoadGraph` synchronously
  loads the 16 dynamic libraries and does **not** perform one inference. Kcore
  indexes the size/address arrays with its own tile id, so tile N loads the
  corresponding `tileN/kcore_fw.so` and resolves the shared symbol.

- Computation is a later type-7 `DYNLIB_RUN`. The AP broadcasts the same boot
  parameter physical address to every active tile. Current Kcore preserves the
  original boot-parameter pointer, finds the type-6-registered module by name,
  and calls the tile-local entry as `entry(D_BootParamHead *)`. Two legacy host
  builds independently construct the same one-module type-7 payload, while the
  installed V5.6 graph entries read input/output/parameter addresses at the
  expected 56-byte-head plus 72-byte-dyninfo offsets. This recovers an
  exact-build model ABI candidate, not a vendor-supported public builder or a
  direct interpretation of either kernel pointer-table ABI.

Ordinary `txLaunchKernel` is also multi-tile when the grid contains enough
blocks: the current AP/Kcore partitions the total grid over fixed logical tile
ids `0..15` rather than renumbering by active-tile count, and sets the per-call
pid. Consequently 16 independent grid-one launches all run their sole block on
logical tile 0, while one grid-x-16 launch on the current full-good device maps
logical tile `t` to pid `t`. A missing tile loses the corresponding pid instead
of remapping it. `txLaunchClusterKernel` separately
establishes one block per selected tile and broadcasts one
module/function/argument blob. Neither launch can consume 16 independent
per-rank argument blocks without an explicit SPMD publication ABI.

`txLaunchModel` is now more than an opaque research hint: the exact V5.6
type-6/type-7 layouts and device entry call have been recovered. Wafer's
sole current schema-v8 `kind=model` publication/provider carries nested
`entry_abi=tx81-model-bootparam` (the model
BootParam ABI was first introduced in schema-v4; schema-v6 was a historical publication) and owns the typed
builder, nested device-address and module-identity validation, artifact
export/readback, and fake lifecycle gates. A fresh qualified full-good-board
replay also completed two exact type-6/type-7 Add iterations over logical tile
ids `0..15`. This is a logical-execution/result gate, not a physical-coordinate
claim, because the public header exposes neither a supported BPM builder nor a
layout-version guarantee.

Important model strings in HPGR include `bpm_table`, `bpmTableAddr`,
`ModuleLoadPayload`, `tritonLaunchPayload`, `graphTLV`, `DYNLIB_LOAD`,
`DYNLIB_RUN`, `DYNLIB_UNLOAD`, `stream_config_addr`, `kcore_fw_num`,
`tile_good_bitmap`, and `tiles_map`.

Disassembly separates HPGR completion semantics from the KMD compute fence:

| path | completion behavior |
| --- | --- |
| model-manager sync | `TxModelCmdQueue` uses 0x200-byte command/response slots.  `sendCmdSync` obtains a slot, writes the packet, waits until a completion status byte has bit 0 set, copies the 0x200-byte response, then clears the slot. |
| model-manager async | `sendCmdAsync` writes a slot and queues the slot id.  A receive thread, or `startSynchronize` when the thread is inactive, dequeues slot ids, waits for the same completion bit, clears slots, and updates outstanding/completed counters. |
| module launch | `TxModuleObjectMgr::launchModel` sends a type-5 module command that includes a device physical address for `completeSignal`, then polls that signal until it becomes zero before reading the response. |
| stream finish | Stream finish waits on the last queued command's event/completion object rather than trusting the KMD compute fence alone. |

The sync model-manager wait has no recovered timeout in this build; it sleeps in
short intervals while polling the slot completion bit. This proves that HPGR's
command protocol and the KMD compute fence are distinct mechanisms in this
snapshot; it does not make either one the Wafer terminal-completion owner.
Production mapping belongs to the numbered runtime and verification designs.

`txNpuKcPowerOn` and `txNpuKcPowerOff` are deprecated and stub-success in this
build.

## 6. VS / Legacy `Tsm*` Runtime

`libvs_runtime.so` is a compatibility layer over HPGR plus a few direct hardware
helper paths.

### 6.1 Runtime Mode

`TsmInitRuntime(bool)` creates a runtime singleton.  The bool is stored at the
start of the runtime object; `Runtime::IsTriton()` reads it directly.

- `true`: old `Tsm*` calls are bridged to HPGR/`tx_runtime` where supported.
- `false`: many hardware implementation methods return failure `1` or no-op
  success.

This flag is an observed compatibility-mode selector; it does not prove a Wafer
provider/backend selection rule or make Triton the SDK's organizing model.

### 6.2 Bridged vs Stubbed Calls

Bridged or real in this build:

| area | behavior |
| --- | --- |
| Device | `GetDevice*`, `SetDevice`, reset, rank/tile APIs forward to HPGR. |
| Memory | malloc/free/H2D/D2H forward to HPGR. |
| Kernel | `KernelLaunch` and `ClusterKernelLaunch` call `txModuleLoad`, add module, get function, then launch. |
| Model | `Run` resolves address and calls `txLaunchModelSync`. |
| D2D/P2P | Uses VS-generated DTE TLVs and bootparam, not direct `txMemcpy D2D` or `txSend/Recv`. |
| DTEOps | Direct register read/write through `MLCommonBase::read_mem/write_mem`. |

Stub/no-op success or mostly unsupported:

| call | observed behavior |
| --- | --- |
| `SetDeviceOld` | no-op/stub. |
| `DeviceSynchronize` | returns success; use HPGR `txDeviceSynchronize` instead. |
| `InitDevice` | no-op/stub. |
| `Launch`, `LaunchPg`, `AsyncRun` | no real execution in this build. |
| `NpuPowerOn/Off` | no-op/stub. |
| `MemcpyOffsetH2D/D2H` | no-op/stub. |

### 6.3 Dynamic TLV and DTE Tiling

VS `LoadKernel` / `UnloadKernel` are not simple host bookkeeping.  They build
dynamic TLVs, copy them to device memory, set them into an `HrtBootParam`, and
start execution through `Run`.  Observed type ids:

| operation | type |
| --- | ---: |
| dynamic library load | 6 |
| dynamic library run | 7 |
| dynamic library unload | 8 |
| D2D DTE config | 9 |
| P2P send DTE config | `0x0a` |
| P2P recv DTE config | `0x0b` |

Recovered `TileDteCfg` layout:

| offset | field |
| ---: | --- |
| `+0x00` | enable/valid |
| `+0x08` | chunk size, normally `0x1000` |
| `+0x0c` | stride |
| `+0x10` | tail size |
| `+0x18` | loop count |
| `+0x20` | main source |
| `+0x28` | main destination |
| `+0x30` | tail source |
| `+0x38` | tail destination |

Each entry is `0x40` bytes.  D2D uses 16 entries and a `DynTLV_DteCfg` payload
length of `0x410`; the total copied TLV is `0x418`.  HPGR/VS then wraps it in
an outer `0x480` payload with the DTE TLV placed at offset `+0x68`.

D2D tile order:

```text
[0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]
```

D2D uses 16 lanes, 4 KiB chunks, and an aggregate stride of `0x10000`
(`16 * 4 KiB`).

P2P tile subset:

```text
[5,6,9,10]
```

P2P uses 4 lanes, 4 KiB chunks, and aggregate stride `0x4000`.  Peer tile
coordinates are decoded from `peerInfo` as:

```text
peer_x = peerInfo >> 8
peer_y = peerInfo & 0xff
```

`DTEOps` also exposes direct register behavior: DTE channels 0..3 use a base
offset pattern `(channel + 0x2000) << 9`; status bit 0 is done and bit `0x100`
is error.  It includes stream FSM monitor init/receive/deinit helpers.

## 7. TSMML System Management

`tsmml_api_v2.h` and `libtsmml.so` provide a hardware management layer similar
in role to NVML, but with Kuiper-specific tile/C2C/topology fields.

### 7.1 Structures

Important public structures:

| structure | key fields |
| --- | --- |
| `tsmml_device_t` | `index`, `mesh_id`, `slot_id`, `user_id`, `soft_id`, PCIe info, names/versions, `dev_fd`, mutex. |
| `tsmml_pcie_info_t` | domain/bus/device/function, vendor/device id, max/current PCIe gen/width, BDF string, CPU list, switch BDF, NUMA node. |
| `tsmml_device_npu_info_t` | tile count and 16 `tsmml_npu_tile_info_t` records. |
| `tsmml_npu_tile_info_t` | tile index, logic id, physical id, chip id, used flag, good tile flag. |
| `tsmml_device_position_info_s` | local/global position, start id, edge info, good tile start. |
| `tsmml_system_topo_info_s` | topology type 0 mesh / 1 torus, machine coordinate, machine topo row/col, cluster topo row/col. |
| `tsmml_device_link_info_t` | intra-link info, optical/inter-link info, up to 8 C2C status records. |
| `tsmml_device_c2c_status_t` | direction, link state, link speed, firewall set/status. |
| `tsmml_device_memory_info_t` | total/free/used, DDR frequency/temperature/current/voltage. |
| `tsmml_device_temperature_info_t` | top/cpu/vpu/tile/board temperatures and per-tile alarm thresholds. |
| `tsmml_device_clock_info_t` | DDR clock, tile clocks, VPU/GPU/decoder/encoder clocks. |
| `mlDeviceProcessesInfo_t` | per-task memory usage in NPU normal/bin/visible/remote pools. |

C2C directions are explicit:

| value | direction |
| ---: | --- |
| 0 | east |
| 1 | west |
| 2 | south |
| 3 | north |

Product type enum:

| value | product |
| ---: | --- |
| 0 | EVB |
| 1 | REX1032 |
| 2 | REX1008 |

### 7.2 API Families

| family | APIs |
| --- | --- |
| lifecycle | `tsmml_init`, `tsmml_shutdown`. |
| discovery | device count, handle by index, handle by PCI bus id. |
| device info | PCIe, NPU tile, position, utilization, memory, VPU, power, temperature, PCIe transfer bytes, clock, health. |
| C2C | link info, firewall get/set, MAC address get/set, MAC reset get/set. |
| processes | task number and process/task memory data. |
| system | discovery state, driver version, ML version, topology, product type/name. |
| internal | export table and field-value query. |

`libtsmml.so` strings confirm it is the bridge to kernel/driver state.  It uses
device-manager paths, `/dev/mem`, PCI sysfs, TSM driver info ioctls, NPU command
ioctls, task data commands, DTE commands, C2C discovery/topology commands, and
BAR mapping helpers such as `tsmml_bar_addr_map` / `tsmml_get_bar_addr`.

## 8. Validation and System Tools

`kuiper/bin` is a hardware SDK validation suite, not just examples.

### 8.1 HPGR Runtime Tests

| tool | coverage |
| --- | --- |
| `dev_test` | Device count/list/property, all-tile info, PG tile selection, rank size/id. |
| `mem_test` | `txMalloc`, `txFree`, H2D/D2H/D2D `txMemcpy`, offset copy cases. |
| `stream_test` | Stream create/destroy/sync/wait-event, event create/record/destroy, model+kernel in streams. |
| `module_test` | Module load/get-function/unload, kernel launch, cluster kernel launch, model launch, graph load/unload. |
| `module_toolchain_test` | Loads toolchain kernel `.so`, runs `main_kernel_add/gemm/embedding`. |
| `launch_ggl_test` | `txLaunchKernelGGL` and `txLaunchClusterKernelGGL`. |
| `log_test` | Runtime logging. |

`stream_test` includes an event-wait sequence:

```text
EventCreate -> StreamCreate x2 -> txLaunchModel -> txEventRecord
-> txStreamWaitEvent -> txLaunchModel -> StreamSynchronize
```

This is useful for runtime dependency semantics, but it still does not prove
low-level SPM bank safety.

### 8.2 Hardware/System Tools

| tool | hardware surface |
| --- | --- |
| `tsm_smi` | TSMML, C2C link/topology/firewall, tile ids, BAR4 debug, reset/rescan, runtime/library versions. |
| `tsmvs` | Validation suite for NPU/PCIe/DDR/C2C/power/stress/perf/diag. |
| `tsm-dte` | DTE device/tile send/recv and SPM test options. |
| `tsm-copy` | H2D/D2H/update file copy, BO allocation, DMA, firmware update helper flow. |
| `tsm-discovery`, `discovery_host` | Single/cross-node discovery, C2C topology exchange, ioctl state writeback. |
| link scripts | Single/cross-node C2C and link diagnostics, often through `mpirun`. |
| `tsm-flash` | Firmware update/query. |

`tsm_smi` strings expose:

- device link tables with user id, mesh id, soft id, slot id, PCI BDF, connected
  devices, link state, and speed;
- CPU/NUMA/PCIe switch tables;
- C2C firewall get/set/all;
- direction labels east/west/south/north;
- reset flows for REX1008/REX1032 including PCIe remove/rescan and
  `npu_ep_log_svc.service`.

`tsmvs.yaml` defines validation modes:

- `power_test`: `power_level`, data type, sparsity, duration, card id, tile
  position.  Comment notes PG card tile positions `0 front`, `1 middle`,
  `2 behind`, `8 tile`; other cards use `-1`.
- `stress_test`: NPU/PCIe/DDR, C2C commented in the default config.
- `perf_test`: NPU/C2C/PCIe/DDR, PCIe direction and test mode.
- `diag_test`: C2C latency and library/runtime diagnostics.

`tsmvs_ut.sh` runs PCIe perf in both `serial` and `parallel` modes for H2D,
D2H, and bidirectional tests; C2C cases use `mpirun`.

### 8.3 BO, DMA, and Memory Pools

`tsm-copy` and `tsmvs` strings expose driver BO pools and memory domains:

| symbol/string | meaning |
| --- | --- |
| `TSM_BO_LOCAL_DRAM` | Local device DRAM BO. |
| `TSM_BO_REMOTE_DRAM` | Remote DRAM BO. |
| `TSM_BO_POOL_NPU_NORMAL` | General NPU memory pool. |
| `TSM_BO_POOL_NPU_BIN` | NPU binary/code pool. |
| `TSM_BO_POOL_VISIBLE` | Host/device visible pool. |
| `TSM_BO_POOL_VISIBLE_EXTENDED` | Extended visible pool. |
| `TSM_BO_POOL_LOG` | Log pool. |
| `TSM_BO_CREATE_USERPTR`, `TSM_BO_CREATE_IMPORT`, `TSM_BO_QUERY` | BO creation/query/import commands. |

Alignment evidence:

- BO creation paths use `SZ_16K` in validation/copy tools.
- `tsmvs` warns when host addresses are not 4 KiB aligned.
- VS `HostFlush` rounds cached input segments to 4 KiB.
- D2D/P2P DTE TLVs use 4 KiB chunking.

These are confirmed alignment facts.  They are not equivalent to a general
64 KiB SPM user-allocation rule.

### 8.4 Data Payloads

`kuiper/data` contains 180 files in this SDK snapshot.  Important groups:

| group | role |
| --- | --- |
| `triton_data` | Toolchain kernel launch sample and 16-tile model metadata. |
| `vs_data` | VS/NPU/SPM stress data and golden outputs. |
| `graph_data` | Graph model sample. |
| `pg_8tile` | Partial-good 8-tile stream/config payload. |
| `unicast_16tile_address` | 16-tile unicast/address routing payload. |

Observed `model_info.json` facts:

| data set | tile topology | tensor/model facts |
| --- | --- | --- |
| `graph_data` | 16 tiles, 4x4 | input `[4096,1280]`, dtype 5, small param size. |
| `vs_data` | 16 tiles, 4x4 | input `[1,2048]`, output `[1,1000]`, param 4 MiB, imm 512 KiB. |
| `triton_data` | 16 tiles, 4x4 | input/output `[512,64]`, dtype 5, param 0, imm 360448. |

`config_stream.bin` variants:

| data | size / header clue |
| --- | --- |
| common 16-tile stream | 26972 bytes; reused by graph/vs/triton data. |
| `pg_8tile` | 24236 bytes; first words include `1,1,8,2,4`. |
| `unicast_16tile_address` | 45676 bytes; first words include `1,1,16,4,4`. |

`StreamConfigMap.txt` appears in graph/vs data and shows:

- `TileNum 16`;
- `GroupXlen 4`, `GroupYlen 4`;
- `ModelTaskType Inference`;
- 16 `tileStreamCfg` entries;
- 20 `StreamCfg` entries;
- 19 `unicast` streams and one `broadcast` stream;
- stream scope is DDR;
- address offsets are zero in these samples.

The RISC-V `triton_data/kernel_0427.so` exports `add_kernel` and has undefined
symbols such as `__Rdma`, `__AddVV`, `__Wdma`, and `__get_pid`, linking the
sample kernel back to the low-level TX8 instruction/runtime wrappers recovered
from `tx8_deps`.

## 9. CCL and FlagCX

`tccl.h` and `nccl.h` are NCCL-compatible headers modified to include
`tx_runtime.h` and use `txStream_t` instead of CUDA streams.

Important public CCL structures:

| structure | meaning |
| --- | --- |
| `tcclComm_t` / `ncclComm_t` | Opaque communicator handle. |
| `tcclUniqueId` / `ncclUniqueId` | 128-byte communicator unique id. |
| `tcclConfig_t` / `ncclConfig_t` | Communicator config: blocking, cluster size, CTA limits, net name, split sharing, traffic class. |
| `tcclSimInfo_t` / `ncclSimInfo_t` | Group simulation estimated time. |

Datatypes cover signed/unsigned integers, fp16/fp32/fp64, bfloat16, and fp8
e4m3/e5m2.  Reduction operations include sum, product, max, min, average, and
no-op.  Scalar residence is either device-visible memory or host immediate.

Collective and P2P APIs include:

- reduce, broadcast, all-reduce, reduce-scatter, all-gather;
- all-to-all and all-to-all-v;
- gather, scatter;
- send/recv;
- group start/end;
- communicator init/finalize/destroy/abort/query/register/deregister;
- async error query and suspend/resume.

`libtccl.so` / `libnccl.so` export `nccl*`, `tccl*`, `pnccl*`, and `ptccl*`
symbol variants.  They rely on runtime `tx*` symbols such as `txMalloc`,
`txMemcpy`, `txStream*`, `txEvent*`, `txSend`, `txRecv`,
`txLaunchKernelGGL`, and device-query APIs.

`libflagcx.so` depends on `libhpgr.so` and references `txSend`, `txRecv`,
`txMalloc`, `txMemcpy`, `txLaunchModel`, and deprecated Kcore power APIs.
Strings show socket/IB/bootstrap/topology/C2C route logic and heterogeneous
vendor adaptors.  Treat FlagCX as a distributed communication/topology layer,
not as the compiler lowering contract.

## 10. Multimedia, CV, and VPU/G2D

Kuiper SDK also includes a multimedia hardware layer.

### 10.1 Codec

`tsmcodec_common.h` defines:

- codec types: MPEG1/2/4, VC1, H263, H264, HEVC, VP8/9, AVS/AVS+/AVS2, JPEG,
  AV1;
- pixel formats: NV12/NV21/I420/YV12/YUYV/UYVY/P010/I010/YUV444/RGBA/BGRA and
  others;
- memory types: host CPU memory and TSM device memory;
- event types: new frame, sequence, EOS, frame processed, OOM, stream corrupt,
  unsupported stream, buffer overflow, fatal error.

Core buffer ownership structures:

| structure | semantics |
| --- | --- |
| `tsmcodecFramePlane_t` | Device address, stride, allocated length. |
| `tsmcodecFrame_t` | pixel format, color space, width/height, device id, plane count, up to 6 planes, timestamps/private data. |
| `tsmcodecStream_t` | Host/device stream buffer, offset, length, memory type, timestamp/private data. |

Decoder API covers capability query, create/set-params/destroy, EOS, input
stream send, output frame get/release, buffer status, JPEG info, and synchronous
JPEG decode.  Decode parameters include `stride_align`, which must be a power
of two in `[1,2048]`.

Encoder API covers H264/HEVC/JPEG profiles, levels, tiers, entropy mode, B-frame
reference mode, rate control, QP map, ROI, LTR, VUI/SEI, presets, tuning, and
runtime reconfiguration.

Binary strings show `/dev/vdec*`, `/dev/venc*`, Verisilicon
`VC9000D/VC9000E`, and DMA synchronization paths.

### 10.2 CV/G2D

`tsmcv.h` defines a G2D/CV-style API.  `tsmcvImage_t` is the central ABI:

```c
typedef struct tsmcvImage {
  uint32_t width;
  uint32_t height;
  uint64_t dev_mem[6];
  uint32_t stride[6];
  tsmcvPixelFormat_t pixel_fmt;
  tsmcvColorSpace_t color_space;
  tsmcvDepth_t depth;
} tsmcvImage_t;
```

The handle created by `tsmcvCreate()` is tied to the current TSM device.  The
API includes color conversion, resize, crop, rotate, normalize, and related
image operations.  Binary strings show `/dev/g2d*` and `GC820/G2D`.

These multimedia/CV libraries matter for a full SDK inventory and possible
frontend integration, but they do not define the Wafer model compiler runtime.

## 11. Profiler and Common Libraries

The profiler package is a rocprofiler-systems/roctx-like stack with Tsingmicro
renaming:

- `tsmprof-sys`;
- `tsmtx-client`;
- `tsmprofiler-register`;
- `tsmprofiler-sdk-tsmtx`;
- preload/tracing support and Perfetto-style outputs;
- `TSM_PROFILER_EN=1` examples in package docs.

`tsmtx` exposes host annotation concepts: mark, range push/pop/start/stop,
thread naming, pause/resume.  The CLI exposes CPU/host/device/GPU sampling
options, but actual device-side event coverage requires board validation.

`libcommon.so` appears to be common package infrastructure rather than a core
hardware ABI.

## 12. Parallel Mode, SPM, and Bank-Conflict Evidence

This is the part where it is easy to overfit.  The combined
`firmware_kuiper` + `tx8_deps` evidence proves that parallel instruction issue
is a real hardware/software mode, but this snapshot still does not expose the exact
SPM-address-to-bank function or penalty. The authoritative SPM1 design separately gives
8×2048-bit banks and LSB interleaving; that supports only the coarse working phase described
by `tasks/09`, not an exact SDK-derived mapping.

### 12.1 What Is Confirmed

Confirmed parallel-related facts:

| fact | evidence |
| --- | --- |
| `serial_mode` is an instruction field. | `tx8_deps` instruction definitions comment bit 0 as serial mode: `1` means all instructions enter one queue without inter-instruction parallelism; `0` means parallel mode. |
| CT/NE/RDMA/WDMA packets carry SPM range metadata. | `tx8_deps` instruction structs include `src_end`, `dst_end`, or equivalent end-address fields describing each operand's SPM storage range. |
| Packets expose range metadata that could participate in hazard detection. | `tx8_deps` packet fields show source/destination begin/end ranges rather than only base pointers; exact scheduler use still needs lower-level or board proof. |
| `tsmvs -mode serial\|parallel` exists for PCIe perf tests. | `tsmvs_ut.sh`, `tsmvs` strings/disassembly. |
| `parallel` PCIe perf path creates pthread workers; `serial` calls the test directly. | `TsmvsPcieTestV2::TestEntry` disassembly. |
| HPGR stream queues order commands in the same stream. | `libhpgr.so` stream/event command objects. |
| VS D2D/P2P parallelizes bulk copies across tile DTE lanes. | `TileDteCfg` TLV generation. |
| D2D uses 16 lanes, 4 KiB chunks, and aggregate stride `0x10000`. | `libvs_runtime.so` disassembly. |
| P2P uses 4 lanes, 4 KiB chunks, and aggregate stride `0x4000`. | `libvs_runtime.so` disassembly. |
| Direct DTE reserves id 0 away from user direct-DTE allocation. | `tx8_deps` direct-DTE headers say id 0 is reserved for Score to avoid async Kcore/Score sharing the same DTE and hanging. |
| High-performance direct DTE is tied to id 2. | `tx8_deps` direct-DTE allocation only allows high-performance outstanding-register configuration on DTE id 2. |
| DDR2DDR direct DTE has an outstanding/burst constraint. | `read_outstanding * axi_read_burst_length <= 12` for the channel-1 DDR2DDR path. |
| DDR2DDR direct DTE prefers 256-byte aligned read/write addresses. | `tx8_deps` comments identify 256-byte alignment as higher-efficiency for DDR2DDR. |
| Driver DTE transfers round length to an 8-byte multiple. | `src/refine/tsm_dte.c` source. |
| Driver DTE remote destinations set bit 39 and encode destination XY at bits 40+. | `tsm_driver_set_dte_trans_ioctl_args` handling in `src/refine/tsm_dte.c`. |
| Driver DTE has 16 controller slots, 4 channel records per controller, and 4x4 XY tile ids. | `src/refine/tsm_dte.h` and `tsm_dte.c`. |
| BO allocation paths use `SZ_16K`; host addresses warn on non-4 KiB alignment. | `tsmvs` / `tsm-copy` strings. |
| SPM paths exist in diagnostics. | `tsm-dte --spm_test`, `MEM_TYPE_LOCAL_SPM`, `REMOTE_SPM`, `OTHER_SPM`, `MEM_BYPASS_SPM`; `DUMP_SPM_SIZE` max 3 MiB. |
| SPM bank hardware exists below the API surface. | `tx8_deps` exposes SPM PMU/ECC/bank register names and Wafer register docs describe in-flight bank tracking. |

### 12.2 What Is Still Not Proven

Not found as a direct string, symbol, or recovered branch in these static
artifacts:

- `isParallel` / `is_parallel`;
- `strategy.isParallel`;
- `bank conflict`;
- a user-facing 64 KiB SPM alignment ABI;
- an exact address-bit formula for SPM bank selection;
- a complete NE/CT/RDMA/WDMA/DTE parallel-issue legality matrix;
- a hardware register explicitly named as the public bank-conflict policy knob.

The 64 KiB value does appear indirectly as `16 * 4 KiB` in the VS D2D tiling
scheme.  That is an internal aggregate DTE stride, not proof by itself that all
parallel SPM addresses must be 64 KiB aligned.

### 12.3 Explicitly Non-Binding Interpretation

The compiler-side line:

```c
uint32_t alignSize = space_->strategy.isParallel ? 64 * 1024 : 256;
```

is likely a historical allocator/scheduler policy, not a public HPGR ABI. The
following explanation is an inference only and does not define Wafer policy:

- serial/single-engine accesses can use the lower 256-byte alignment already
  visible in TX8 instruction/layout evidence and DDR2DDR efficiency conventions;
- `serial_mode=0` lets the device-side scheduler consider multiple independent
  CT/NE/RDMA/WDMA/DTE operations at once instead of forcing all instructions
  through one serialized queue;
- the packet range-end fields give firmware/hardware enough information to
  reject obvious read/write overlap and some resource hazards;
- SPM bank conflicts are then a second-order physical resource hazard, not a
  normal address-overlap hazard;
- a 64 KiB allocation color is only a historical allocator heuristic; it is not needed to
  recover the documented coarse SPM1 phase and must not become a hard placement rule;
- the VS D2D 16-lane, 4 KiB-per-lane, `0x10000` aggregate stride is consistent
  with this interpretation, but it is not sufficient proof of the physical bank
  function.

That is still an inference. The static SDK evidence does not prove the SPM bank
mapping or establish 64 KiB as a hardware ABI. Allocation and scheduling policy
remain with the numbered memory-planning and verification designs.

### 12.4 Evidence Needed for Scheduling Policy

This snapshot provides the following inputs and gaps; it does not define a
production scheduling rule:

- 256-byte alignment appears in instruction, layout, and DDR2DDR evidence;
- 64 KiB appears as a historical allocation-color heuristic, not a recovered
  physical bank mapping;
- packet ranges, busy tables, stream/event/DTE/FSM/CSR mechanisms are distinct
  observations and cannot by themselves prove a Wafer completion contract;
- a board-test matrix that sweeps low SPM address bits and records
  CT/NE/DTE/SPM PMU counters is still needed to identify actual bank conflicts.

## 13. Coverage and Remaining Gaps

Current static coverage:

| area | coverage | remaining risk |
| --- | --- | --- |
| HPGR `tx_runtime` ABI | High: public header plus disassembly. | Board behavior for failure paths, cross-stream hazards, event timing. |
| Driver KMD/UAPI | High: decrypted payload includes `tsm_uapi.h` and KMD source for BO/job/NPU/C2C/DTE/log/info/topology ioctls. | Needs hardware validation for BAR sizes, SR-IOV branches, MSI/MSI-X, DMA channel limits, and AP firmware interactions. |
| VS compatibility runtime | Medium-high: many vtable paths and TLVs recovered. | Some private structure fields remain inferred. |
| D2D/P2P DTE TLV | Medium-high: layout and tiling recovered. | Formal device-side TLV consumer semantics still need lower firmware/board proof. |
| HPGR model/module completion | High: command-slot completion bit, async receive thread, `completeSignal`, and stream command waits recovered by disassembly. | Timeout/error propagation edge cases still require board behavior. |
| Address space / BO pools | High: KMD source plus `tx8_deps` maps BARs, ATU entries, tile windows, BO pools, firmware slots, and stream table address. | Runtime-visible addresses should still be validated on real small-BAR/large-BAR/SR-IOV systems. |
| PG / bad-tile handling | Medium-high: boot strings, SYS_CTRL offsets, KMD bitmap use, stream table, HPGR fallback, and 8-tile data payloads agree. | Exact production PG SKU variants and board policy require hardware inventory. |
| TSMML topology/C2C | High at public ABI level. | Full per-API mapping from TSMML to UAPI/ioctl still needs enumeration. |
| System tools | Medium-high: strings/disassembly/script evidence. | Runtime outputs require hardware. |
| Firmware/boot/rootfs | Medium-high: package layout, boot strings, rootfs modules, and driver firmware loader recovered. | Endpoint runtime after rootfs switch and SPI image layout still need deeper extraction/board validation. |
| CCL/NCCL/TCCL | High at public ABI level. | Runtime algorithms/topology performance require execution. |
| FlagCX | Medium: symbols and strings. | No public header in this SDK. |
| Multimedia/CV | High at public ABI level. | Hardware limits and performance require board tests. |
| Profiler | Medium: CLI/libs/docs visible. | Device-side event fidelity requires board tests. |
| SPM bank / `serial_mode=0` | Medium for existence of parallel mode, operand range metadata, and likely bank-resource hazard; this SDK remains low confidence for mapping, while the separate authoritative SPM1 design gives an 8×2048-bit LSB-interleaved coarse phase. | Need board sweep or lower RTL/firmware evidence for exact port/stride penalty; 64 KiB remains neither calibrated policy nor hardware ABI. |
| `txdnn` eager op layer | Not covered by this SDK. | `libtxdnn.so` / `txdnn.h` are still absent here. |

Evidence handoff to the numbered designs:

- The snapshot contains distinct HPGR, TSMML/system, validation,
  firmware/driver, CCL/FlagCX, multimedia/CV, and profiler surfaces.
- KMD source in this snapshot signals compute fences after the MHU doorbell
  because AP compute-done interrupt support is absent; this is not terminal
  model completion proof.
- `libvs_runtime.so` contains useful compatibility and DTE TLV evidence, while
  several `Tsm*` sync/launch calls are stubs.
- `serial_mode=0`, packet ranges, and the historical 64 KiB coloring heuristic
  are separate facts with different confidence levels.

Topology/identifier, memory-domain, transport, provider, completion, and
calibration decisions belong respectively to the numbered topology, memory,
communication, runtime, and verification documents. This evidence ledger does
not select HPGR, TSMML, CCL/FlagCX, or any other surface as a Wafer architecture
owner.
