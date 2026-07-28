#include "wafer_tx81_crt.h"

#if defined(WAFER_TX81_PROFILE_TRACE_CRT) && !defined(WAFER_TX81_PROFILE_CRT)
#define WAFER_TX81_PROFILE_CRT 1
#endif

#ifdef WAFER_TX81_PROFILE_CRT
#include "wafer_tx81_profiler.h"
#endif

#ifndef CONFIG_NO_PLATFORM_HOOK_H
#define CONFIG_NO_PLATFORM_HOOK_H 1
#endif
#ifndef USING_RISCV
#define USING_RISCV 1
#endif

#include "instr_adapter.h"

extern int8_t *get_spm_memory_mapping(uint64_t offset);
extern uint64_t get_tile_spm_addr_base(uint32_t tile_id_1d, int32_t tile_x,
                                       int32_t tile_y);

typedef struct {
  uint32_t stride;
  uint32_t iteration;
} WaferDirectDTEStrideIteration;

typedef struct {
  uintptr_t src_addr;
  uintptr_t dst_addr;
  uint32_t length;
  uint8_t remote_fsm_id;
  uint8_t mode;
  uint16_t dst_tile;
  uint16_t tile_this;
  uint16_t rsv;
  WaferDirectDTEStrideIteration stride_iterations[3];
  void *dte_node;
} WaferDirectDTESendInfo;

extern void direct_sync_init(int once_sync_num);
extern void direct_sync_post(uint32_t tile_this, uint32_t tile_other);
extern void direct_sync_wait(uint32_t tile_this, uint32_t tile_other);
extern void *direct_fsm_monitor_init(int fsm_id, uintptr_t addr,
                                     int packet_size, int packet_cnt);
extern int direct_fsm_monitor_deinit(void *fsm_hd);
extern void direct_fsm_monitor_receive(uint16_t tile_this, uint16_t tile_other,
                                       void *fsm_hd);
extern void *direct_dte_attach(uint32_t is_high_performance);
extern int direct_dte_release(void *dte_node);
extern int direct_dte_send_async(WaferDirectDTESendInfo *dte_info);
extern int direct_dte_wait_done(WaferDirectDTESendInfo *dte_info);

enum {
  WAFER_DIRECT_DTE_SEND_EVENT = 0x100,
  WAFER_DIRECT_DTE_RECV_EVENT_BASE = 0x200,
  WAFER_DIRECT_DTE_MAX_RECEIVERS = 4,
  WAFER_TX81_TILE_GRID_X = 4,
  WAFER_TX81_TILE_GRID_Y = 4,
};

typedef struct {
  WaferDirectDTESendInfo info;
  uint32_t is_high_performance;
  bool active;
} WaferDirectDTESenderState;

typedef struct {
  void *handle;
  uint16_t local_tile;
  uint16_t remote_tile;
  bool active;
} WaferDirectDTEReceiverState;

static volatile uint32_t *wafer_direct_dte_status;
static uint32_t wafer_direct_dte_status_value;
static WaferDirectDTESenderState wafer_direct_dte_sender;
static WaferDirectDTEReceiverState
    wafer_direct_dte_receivers[WAFER_DIRECT_DTE_MAX_RECEIVERS];

#ifdef WAFER_TX81_PROFILE_CRT
static void wafer_profile_direct_dte_begin(uint64_t direct_event);
static void wafer_profile_direct_dte_end(void);
static uint32_t wafer_profile_direct_dte_phase_begin(uint8_t kind);
static void wafer_profile_direct_dte_phase_end(uint32_t event_index);
#endif

/*
 * Kernel arguments and txMalloc resources are cacheable device-DDR
 * addresses.  The firmware invalidates the argument table before entering a
 * dynamic module, but it does not clean module writes when the entry returns.
 * Publish the transport status with the same C908 clean-and-invalidate
 * sequence used by the firmware's rt_hw_cpu_dcache_ops(FLUSH) path so a host
 * D2H observes the terminal value rather than the pre-launch poison word.
 */
static void wafer_direct_dte_flush_status(volatile uint32_t *status) {
  enum {
    WAFER_TX81_SUPERVISOR_MODE = 1,
    WAFER_TX81_MACHINE_MODE = 3,
  };
  uintptr_t address =
      (uintptr_t)status &
      ~(uintptr_t)(WAFER_TX81_DIRECT_DTE_STATUS_V2_CACHE_LINE_BYTES - 1);
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  if (mode == WAFER_TX81_MACHINE_MODE)
    __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
  else if (mode == WAFER_TX81_SUPERVISOR_MODE)
    __asm__ volatile("dcache.civa %0" : : "r"(address) : "memory");
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static void wafer_direct_dte_publish_status(uint32_t value) {
  wafer_direct_dte_status_value = value;
  if (!wafer_direct_dte_status)
    return;
  *wafer_direct_dte_status = value;
  wafer_direct_dte_flush_status(wafer_direct_dte_status);
}

static void wafer_direct_dte_set_error(void) {
  wafer_direct_dte_publish_status(WAFER_TX81_DIRECT_DTE_STATUS_TRANSPORT_ERROR);
}

static void wafer_direct_dte_reset_state(uint64_t status_addr) {
  wafer_direct_dte_status = (volatile uint32_t *)(uintptr_t)status_addr;
  wafer_direct_dte_sender.active = false;
  for (uint32_t index = 0; index < WAFER_DIRECT_DTE_MAX_RECEIVERS; ++index)
    wafer_direct_dte_receivers[index].active = false;
  wafer_direct_dte_publish_status(WAFER_TX81_DIRECT_DTE_STATUS_PENDING);
}

void wafer_tx81_direct_dte_begin(uint64_t status_addr, uint32_t rank_count) {
  wafer_direct_dte_reset_state(status_addr);
  direct_sync_init((int)rank_count);
}

void wafer_tx81_direct_dte_begin_after_prepare(uint64_t status_addr,
                                               uint32_t rank_count) {
  (void)rank_count;
  wafer_direct_dte_reset_state(status_addr);
}

uint64_t wafer_tx81_direct_dte_send_prepare(uint64_t src, uint64_t remote_dst,
                                            uint32_t byte_count,
                                            uint32_t local_tile,
                                            uint32_t remote_tile,
                                            uint32_t remote_fsm_id,
                                            uint32_t is_high_performance) {
  if (wafer_direct_dte_sender.active || remote_fsm_id >= 4 ||
      local_tile > UINT16_MAX || remote_tile > UINT16_MAX) {
    wafer_direct_dte_set_error();
    return 0;
  }
  uint64_t remote_spm_base = get_tile_spm_addr_base(
      remote_tile, WAFER_TX81_TILE_GRID_X, WAFER_TX81_TILE_GRID_Y);
  if (remote_dst > UINT64_MAX - remote_spm_base) {
    wafer_direct_dte_set_error();
    return 0;
  }
  WaferDirectDTESendInfo info = {0};
  info.src_addr = (uintptr_t)src;
  info.dst_addr = (uintptr_t)(remote_spm_base + remote_dst);
  info.length = byte_count;
  info.remote_fsm_id = (uint8_t)remote_fsm_id;
  info.mode = 0;
  info.dst_tile = (uint16_t)remote_tile;
  info.tile_this = (uint16_t)local_tile;
  wafer_direct_dte_sender.info = info;
  wafer_direct_dte_sender.is_high_performance = is_high_performance;
  wafer_direct_dte_sender.active = true;
  return WAFER_DIRECT_DTE_SEND_EVENT;
}

uint64_t wafer_tx81_direct_dte_recv_prepare(uint64_t dst, uint32_t byte_count,
                                            uint32_t local_tile,
                                            uint32_t remote_tile,
                                            uint32_t local_fsm_id) {
  if (local_fsm_id >= WAFER_DIRECT_DTE_MAX_RECEIVERS ||
      local_tile > UINT16_MAX || remote_tile > UINT16_MAX ||
      byte_count > INT32_MAX) {
    wafer_direct_dte_set_error();
    return 0;
  }
  WaferDirectDTEReceiverState *receiver =
      &wafer_direct_dte_receivers[local_fsm_id];
  if (receiver->active) {
    wafer_direct_dte_set_error();
    return 0;
  }
  receiver->handle = direct_fsm_monitor_init((int)local_fsm_id, (uintptr_t)dst,
                                             (int)byte_count, 1);
  if (!receiver->handle) {
    wafer_direct_dte_set_error();
    return 0;
  }
  receiver->local_tile = (uint16_t)local_tile;
  receiver->remote_tile = (uint16_t)remote_tile;
  receiver->active = true;
  direct_sync_post(local_tile, remote_tile);
  return WAFER_DIRECT_DTE_RECV_EVENT_BASE + local_fsm_id;
}

void wafer_tx81_direct_dte_wait(uint64_t event) {
#ifdef WAFER_TX81_PROFILE_CRT
  wafer_profile_direct_dte_begin(event);
#endif
  if (event == WAFER_DIRECT_DTE_SEND_EVENT) {
    if (!wafer_direct_dte_sender.active) {
      wafer_direct_dte_set_error();
      goto done;
    }
    WaferDirectDTESendInfo *info = &wafer_direct_dte_sender.info;
    {
#ifdef WAFER_TX81_PROFILE_CRT
      uint32_t phase_event = wafer_profile_direct_dte_phase_begin(
          WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT);
#endif
      direct_sync_wait(info->tile_this, info->dst_tile);
#ifdef WAFER_TX81_PROFILE_CRT
      wafer_profile_direct_dte_phase_end(phase_event);
#endif
    }
    int setup_result = -1;
    {
#ifdef WAFER_TX81_PROFILE_CRT
      uint32_t phase_event = wafer_profile_direct_dte_phase_begin(
          WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE);
#endif
      info->dte_node =
          direct_dte_attach(wafer_direct_dte_sender.is_high_performance);
      if (info->dte_node)
        setup_result = direct_dte_send_async(info);
#ifdef WAFER_TX81_PROFILE_CRT
      wafer_profile_direct_dte_phase_end(phase_event);
#endif
    }
    int completion_result = -1;
    if (setup_result == 0) {
#ifdef WAFER_TX81_PROFILE_CRT
      uint32_t phase_event = wafer_profile_direct_dte_phase_begin(
          WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_COMPLETION_WAIT);
#endif
      completion_result = direct_dte_wait_done(info);
#ifdef WAFER_TX81_PROFILE_CRT
      wafer_profile_direct_dte_phase_end(phase_event);
#endif
    }
    if (setup_result != 0 || completion_result != 0)
      wafer_direct_dte_set_error();
    if (info->dte_node) {
#ifdef WAFER_TX81_PROFILE_CRT
      uint32_t phase_event = wafer_profile_direct_dte_phase_begin(
          WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP);
#endif
      int cleanup_result = direct_dte_release(info->dte_node);
#ifdef WAFER_TX81_PROFILE_CRT
      wafer_profile_direct_dte_phase_end(phase_event);
#endif
      if (cleanup_result != 0)
        wafer_direct_dte_set_error();
    }
    wafer_direct_dte_sender.active = false;
    goto done;
  }

  if (event >= WAFER_DIRECT_DTE_RECV_EVENT_BASE &&
      event <
          WAFER_DIRECT_DTE_RECV_EVENT_BASE + WAFER_DIRECT_DTE_MAX_RECEIVERS) {
    uint32_t fsm_id = (uint32_t)(event - WAFER_DIRECT_DTE_RECV_EVENT_BASE);
    WaferDirectDTEReceiverState *receiver = &wafer_direct_dte_receivers[fsm_id];
    if (!receiver->active) {
      wafer_direct_dte_set_error();
      goto done;
    }
    {
#ifdef WAFER_TX81_PROFILE_CRT
      uint32_t phase_event = wafer_profile_direct_dte_phase_begin(
          WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_COMPLETION_WAIT);
#endif
      direct_fsm_monitor_receive(receiver->local_tile, receiver->remote_tile,
                                 receiver->handle);
#ifdef WAFER_TX81_PROFILE_CRT
      wafer_profile_direct_dte_phase_end(phase_event);
#endif
    }
#ifdef WAFER_TX81_PROFILE_CRT
    uint32_t phase_event = wafer_profile_direct_dte_phase_begin(
        WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP);
#endif
    int cleanup_result = direct_fsm_monitor_deinit(receiver->handle);
#ifdef WAFER_TX81_PROFILE_CRT
    wafer_profile_direct_dte_phase_end(phase_event);
#endif
    if (cleanup_result != 0)
      wafer_direct_dte_set_error();
    receiver->active = false;
    goto done;
  }
  wafer_direct_dte_set_error();

done:
  (void)0;
#ifdef WAFER_TX81_PROFILE_CRT
  wafer_profile_direct_dte_end();
#endif
}

void wafer_tx81_direct_dte_finish(void) {
  if (wafer_direct_dte_sender.active) {
    wafer_direct_dte_set_error();
    return;
  }
  for (uint32_t index = 0; index < WAFER_DIRECT_DTE_MAX_RECEIVERS; ++index)
    if (wafer_direct_dte_receivers[index].active) {
      wafer_direct_dte_set_error();
      return;
    }
  if (wafer_direct_dte_status &&
      wafer_direct_dte_status_value == WAFER_TX81_DIRECT_DTE_STATUS_PENDING)
    wafer_direct_dte_publish_status(WAFER_TX81_DIRECT_DTE_STATUS_SUCCESS);
}

static Data_Format wafer_format(uint32_t format) { return (Data_Format)format; }

static RND_MODE wafer_rounding(uint32_t rounding_mode) {
  return (RND_MODE)rounding_mode;
}

static Data_Shape wafer_shape4(uint32_t n, uint32_t h, uint32_t w, uint32_t c) {
  Data_Shape shape = {(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c};
  return shape;
}

static St_StrideIteration
wafer_stride_iteration(uint32_t stride0, uint32_t stride1, uint32_t stride2,
                       uint32_t iteration0, uint32_t iteration1,
                       uint32_t iteration2) {
  St_StrideIteration si = {stride0,    iteration0, stride1,
                           iteration1, stride2,    iteration2};
  return si;
}

static uint32_t wafer_format_bytes(uint32_t format) {
  switch ((Data_Format)format) {
  case Fmt_INT8:
  case Fmt_UINT8:
  case Fmt_BOOL:
    return 1;
  case Fmt_INT16:
  case Fmt_UINT16:
  case Fmt_FP16:
  case Fmt_BF16:
    return 2;
  case Fmt_INT32:
  case Fmt_UINT32:
  case Fmt_FP32:
  case Fmt_TF32:
    return 4;
  case Fmt_INT64:
  case Fmt_UINT64:
    return 8;
  default:
    return 1;
  }
}

static bool wafer_elem_count_from_bytes(uint32_t bytes, uint32_t format,
                                        uint32_t *elem_count) {
  if ((Data_Format)format == Fmt_BOOL) {
    if (bytes > UINT32_MAX / 8U)
      return false;
    *elem_count = bytes * 8U;
    return true;
  }
  uint32_t elem_bytes = wafer_format_bytes(format);
  if (elem_bytes == 0 || bytes % elem_bytes != 0)
    return false;
  *elem_count = bytes / elem_bytes;
  return true;
}

static bool wafer_memset_span_bytes(uint32_t elem_count, uint32_t format,
                                    uint32_t *span_bytes) {
  if (elem_count == 0)
    return false;
  switch ((Data_Format)format) {
  case Fmt_INT8:
  case Fmt_INT16:
  case Fmt_FP16:
  case Fmt_BF16:
  case Fmt_INT32:
  case Fmt_FP32:
  case Fmt_TF32:
  case Fmt_UINT8:
  case Fmt_UINT16:
  case Fmt_UINT32:
  case Fmt_INT64:
  case Fmt_UINT64:
    break;
  default:
    return false;
  }
  uint32_t elem_bytes = wafer_format_bytes(format);
  if (elem_count > UINT32_MAX / elem_bytes)
    return false;
  *span_bytes = elem_count * elem_bytes;
  return true;
}

static uint64_t wafer_shape_elements(Data_Shape shape) {
  return (uint64_t)shape.n * shape.h * shape.w * shape.c;
}

static int32_t wafer_bilinear_scale(uint32_t src, uint32_t dst) {
  if (dst == 0)
    return 0;
  return (int32_t)(((uint64_t)src << 16) / dst);
}

#ifdef WAFER_TX81_PROFILE_CRT
#include "wafer_tx81_profiler_impl.inc"
#endif

static void wafer_execute_ct(CT_Param *instr) {
#ifdef WAFER_TX81_PROFILE_TRACE_CRT
  (void)wafer_profile_execute_ncc(instr, instr->inter_type,
                                  WAFER_TX81_PROFILER_ENGINE_CT);
#else
  (void)TsmExecute(instr);
#endif
}
static void wafer_execute_ne(TsmNeInstr *instr) {
#ifdef WAFER_TX81_PROFILE_TRACE_CRT
  (void)wafer_profile_execute_ncc(instr, instr->inter_type,
                                  WAFER_TX81_PROFILER_ENGINE_NE);
#else
  (void)TsmExecute(instr);
#endif
}
static void wafer_execute_rdma(TsmRdmaInstr *instr) {
#ifdef WAFER_TX81_PROFILE_TRACE_CRT
  (void)wafer_profile_execute_ncc(instr, instr->inter_type,
                                  WAFER_TX81_PROFILER_ENGINE_RDMA);
#else
  (void)TsmExecute(instr);
#endif
}
static void wafer_execute_wdma(TsmWdmaInstr *instr) {
#ifdef WAFER_TX81_PROFILE_TRACE_CRT
  (void)wafer_profile_execute_ncc(instr, instr->inter_type,
                                  WAFER_TX81_PROFILER_ENGINE_WDMA);
#else
  (void)TsmExecute(instr);
#endif
}
static void wafer_execute_td(TsmDataMoveInstr *instr) {
#ifdef WAFER_TX81_PROFILE_TRACE_CRT
  (void)wafer_profile_execute_ncc(instr, instr->inter_type,
                                  WAFER_TX81_PROFILER_ENGINE_TDMA);
#else
  (void)TsmExecute(instr);
#endif
}

static uint64_t wafer_spm_mapped_addr(uint64_t offset) {
  return (uint64_t)(uintptr_t)get_spm_memory_mapping(offset);
}

static void wafer_store_u32(uint64_t addr, uint32_t value) {
  *(volatile uint32_t *)(uintptr_t)addr = value;
}

static void wafer_store_value(uint64_t addr, uint32_t format, uint64_t value) {
  switch ((Data_Format)format) {
  case Fmt_INT8:
  case Fmt_UINT8:
  case Fmt_BOOL:
    *(volatile uint8_t *)(uintptr_t)addr = (uint8_t)value;
    return;
  case Fmt_INT16:
  case Fmt_UINT16:
  case Fmt_FP16:
  case Fmt_BF16:
    *(volatile uint16_t *)(uintptr_t)addr = (uint16_t)value;
    return;
  case Fmt_INT64:
  case Fmt_UINT64:
    *(volatile uint64_t *)(uintptr_t)addr = value;
    return;
  default:
    wafer_store_u32(addr, (uint32_t)value);
    return;
  }
}

/*
 * get_spm_memory_mapping() returns the uncached weak-order L1SPM alias.
 * Complete mapped-SPM accesses before a following NCC command observes the
 * same storage; cache maintenance is invalid for this address domain.
 */
static void wafer_order_mapped_spm(void) {
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

void wafer_tx81_rdma(uint64_t src, uint64_t dst, uint32_t byte_count,
                     uint32_t inner_bytes, uint32_t stride0, uint32_t stride1,
                     uint32_t stride2, uint32_t iteration0, uint32_t iteration1,
                     uint32_t iteration2, uint32_t format) {
  uint32_t inner_elements = 0;
  uint32_t stride0_elements = 0;
  uint32_t stride1_elements = 0;
  uint32_t stride2_elements = 0;
  (void)byte_count;
  if (!wafer_elem_count_from_bytes(inner_bytes, format, &inner_elements) ||
      !wafer_elem_count_from_bytes(stride0, format, &stride0_elements) ||
      !wafer_elem_count_from_bytes(stride1, format, &stride1_elements) ||
      !wafer_elem_count_from_bytes(stride2, format, &stride2_elements))
    return;
  TsmRdmaInstr instr = {0};
  TsmRdma *rdma = TsmNewRdma();
  rdma->AddSrcDst(&instr, src, dst, wafer_format(format));
  rdma->ConfigStrideIteration(&instr, inner_elements, stride0_elements,
                              iteration0, stride1_elements, iteration1,
                              stride2_elements, iteration2);
  wafer_execute_rdma(&instr);
  TsmDeleteRdma(rdma);
}

void wafer_tx81_wdma(uint64_t src, uint64_t dst, uint32_t byte_count,
                     uint32_t inner_bytes, uint32_t stride0, uint32_t stride1,
                     uint32_t stride2, uint32_t iteration0, uint32_t iteration1,
                     uint32_t iteration2, uint32_t format) {
  uint32_t inner_elements = 0;
  uint32_t stride0_elements = 0;
  uint32_t stride1_elements = 0;
  uint32_t stride2_elements = 0;
  (void)byte_count;
  if (!wafer_elem_count_from_bytes(inner_bytes, format, &inner_elements) ||
      !wafer_elem_count_from_bytes(stride0, format, &stride0_elements) ||
      !wafer_elem_count_from_bytes(stride1, format, &stride1_elements) ||
      !wafer_elem_count_from_bytes(stride2, format, &stride2_elements))
    return;
  TsmWdmaInstr instr = {0};
  TsmWdma *wdma = TsmNewWdma();
  wdma->AddSrcDst(&instr, src, dst, wafer_format(format));
  wdma->ConfigStrideIteration(&instr, inner_elements, stride0_elements,
                              iteration0, stride1_elements, iteration1,
                              stride2_elements, iteration2);
  wafer_execute_wdma(&instr);
  TsmDeleteWdma(wdma);
}

void wafer_tx81_gather_scatter(uint64_t src, uint64_t dst, uint32_t byte_count,
                               uint32_t inner_bytes, uint32_t src_stride0,
                               uint32_t src_stride1, uint32_t src_stride2,
                               uint32_t src_iteration0, uint32_t src_iteration1,
                               uint32_t src_iteration2, uint32_t dst_stride0,
                               uint32_t dst_stride1, uint32_t dst_stride2,
                               uint32_t dst_iteration0, uint32_t dst_iteration1,
                               uint32_t dst_iteration2) {
  (void)byte_count;
  TsmDataMoveInstr instr = {0};
  TsmDataMove *move = TsmNewDataMove();
  St_StrideIteration src_si =
      wafer_stride_iteration(src_stride0, src_stride1, src_stride2,
                             src_iteration0, src_iteration1, src_iteration2);
  St_StrideIteration dst_si =
      wafer_stride_iteration(dst_stride0, dst_stride1, dst_stride2,
                             dst_iteration0, dst_iteration1, dst_iteration2);
  move->GatherScatter(&instr, src, dst, inner_bytes, &src_si, &dst_si);
  wafer_execute_td(&instr);
  TsmDeleteDataMove(move);
}

void wafer_tx81_memset(uint64_t dst, uint32_t value, uint32_t elem_count,
                       uint32_t format) {
  uint32_t packet_value = value;
  uint32_t packet_elem_count = elem_count;
  uint32_t packet_format = format;
  if ((Data_Format)format == Fmt_BOOL) {
    if (elem_count == 0)
      return;
    packet_value = value != 0 ? UINT8_MAX : 0;
    packet_elem_count = elem_count / 8U + (elem_count % 8U != 0);
    packet_format = Fmt_INT8;
  }
  uint32_t span_bytes = 0;
  if (!wafer_memset_span_bytes(packet_elem_count, packet_format, &span_bytes))
    return;
  TsmDataMoveInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  St_StrideIteration si = {
      span_bytes, 1, 0, 1, 0, 1,
  };
  peripheral->Memset(&instr, dst, packet_value, packet_elem_count, &si,
                     wafer_format(packet_format));
  wafer_execute_td(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_bit2fp(uint64_t src, uint64_t dst, uint32_t elem_count,
                       uint32_t format) {
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->Bit2Fp(&instr, src, dst, elem_count, wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_mask_move(uint64_t src, uint32_t mask, uint64_t dst,
                          uint32_t elem_count, uint32_t format) {
  TsmMaskDataMoveInstr instr = {0};
  TsmMaskDataMove *move = TsmNewMaskDataMove();
  move->MaskMove(&instr, src, mask, dst, elem_count, wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeleteMaskDataMove(move);
}

#define WAFER_DEFINE_ARITH_UNARY(SYMBOL, METHOD)                               \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t format) {                                               \
    TsmArithInstr instr = {0};                                                 \
    TsmArith *arith = TsmNewArith();                                           \
    arith->METHOD(&instr, src, dst, elem_count, wafer_format(format));         \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteArith(arith);                                                     \
  }

#define WAFER_DEFINE_ARITH_BINARY(SYMBOL, METHOD)                              \
  void SYMBOL(uint64_t lhs, uint64_t rhs, uint64_t dst, uint32_t elem_count,   \
              uint32_t format) {                                               \
    TsmArithInstr instr = {0};                                                 \
    TsmArith *arith = TsmNewArith();                                           \
    arith->METHOD(&instr, lhs, rhs, dst, elem_count, RND_NEAREST_EVEN,         \
                  wafer_format(format));                                       \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteArith(arith);                                                     \
  }

#define WAFER_DEFINE_RELATION(SYMBOL, METHOD, BOOL_METHOD)                     \
  void SYMBOL(uint64_t lhs, uint64_t rhs, uint64_t dst, uint32_t elem_count,   \
              uint32_t format) {                                               \
    TsmRelationInstr instr = {0};                                              \
    TsmRelation *relation = TsmNewRelation();                                  \
    if (format == Fmt_BOOL)                                                    \
      relation->BOOL_METHOD(&instr, lhs, rhs, dst, elem_count,                 \
                            wafer_format(format));                             \
    else                                                                       \
      relation->METHOD(&instr, lhs, rhs, dst, elem_count,                      \
                       wafer_format(format));                                  \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteRelation(relation);                                               \
  }

#define WAFER_DEFINE_LOGIC_UNARY(SYMBOL, METHOD, BOOL_METHOD)                  \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t format) {                                               \
    TsmLogicInstr instr = {0};                                                 \
    TsmLogic *logic = TsmNewLogic();                                           \
    if (format == Fmt_BOOL)                                                    \
      logic->BOOL_METHOD(&instr, src, dst, elem_count);                        \
    else                                                                       \
      logic->METHOD(&instr, src, dst, elem_count, wafer_format(format));       \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteLogic(logic);                                                     \
  }

#define WAFER_DEFINE_LOGIC_BINARY(SYMBOL, METHOD, BOOL_METHOD)                 \
  void SYMBOL(uint64_t lhs, uint64_t rhs, uint64_t dst, uint32_t elem_count,   \
              uint32_t format) {                                               \
    TsmLogicInstr instr = {0};                                                 \
    TsmLogic *logic = TsmNewLogic();                                           \
    if (format == Fmt_BOOL)                                                    \
      logic->BOOL_METHOD(&instr, lhs, rhs, dst, elem_count);                   \
    else                                                                       \
      logic->METHOD(&instr, lhs, rhs, dst, elem_count, wafer_format(format));  \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteLogic(logic);                                                     \
  }

#define WAFER_DEFINE_TRANS(SYMBOL, METHOD)                                     \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t format) {                                               \
    TsmTranscendentalInstr instr = {0};                                        \
    TsmTranscendental *trans = TsmNewTranscendental();                         \
    trans->METHOD(&instr, src, dst, elem_count, wafer_format(format));         \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteTranscendental(trans);                                            \
  }

#define WAFER_DEFINE_ACTIVATION(SYMBOL, METHOD)                                \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t format) {                                               \
    TsmActivationInstr instr = {0};                                            \
    TsmActivation *activation = TsmNewActivation();                            \
    activation->METHOD(&instr, src, dst, elem_count, wafer_format(format));    \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteActivation(activation);                                           \
  }

WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_abs, AbsVV)
WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_recip, RecipVV)
WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_square, SquareVV)
WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_sqrt, SqrtVV)
WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_rsqrt, RsqrtVV)
WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_neg, NegVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_max, MaxVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_min, MinVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_add, AddVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_sub, SubVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_mul, MulVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_div, DivVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_eq, EqualVV, BoolEqualVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_ne, UnEqualVV, BoolUnEqualVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_ge, GreaterEqualVV,
                      BoolGreaterEqualVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_gt, GreaterVV, BoolGreaterVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_le, LessEqualVV, BoolLessEqualVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_lt, LessThenVV, BoolLessThenVV)
WAFER_DEFINE_LOGIC_UNARY(wafer_tx81_elementwise_logic_not, NotV, BoolNotV)
WAFER_DEFINE_LOGIC_BINARY(wafer_tx81_elementwise_logic_and, AndVV, BoolAndV)
WAFER_DEFINE_LOGIC_BINARY(wafer_tx81_elementwise_logic_or, OrVV, BoolOrV)
WAFER_DEFINE_LOGIC_BINARY(wafer_tx81_elementwise_logic_xor, XorVV, BoolXorV)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_log2, Log2)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_ln, Ln)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_pow2, Pow2)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_exp, Exp)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_exp_lp, Explp)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_sin, Sin)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_cos, Cos)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_tanh, Tanh)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_sigmoid, Sigmoid)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_relu, Relu)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_satrelu, Satrelu)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_leakyrelu, Leakyrelu)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_softplus, Softplus)

#define WAFER_DEFINE_REDUCE(SYMBOL, METHOD)                                    \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t dim, uint32_t n,            \
              uint32_t h, uint32_t w, uint32_t c, uint32_t format) {           \
    TsmReduceInstr instr = {0};                                                \
    TsmReduce *reduce = TsmNewReduce();                                        \
    reduce->METHOD(&instr, src, dst, dim, wafer_shape4(n, h, w, c),            \
                   wafer_format(format));                                      \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteReduce(reduce);                                                   \
  }

WAFER_DEFINE_REDUCE(wafer_tx81_reduce_sum, ReduceSum)
WAFER_DEFINE_REDUCE(wafer_tx81_reduce_max, ReduceMax)
WAFER_DEFINE_REDUCE(wafer_tx81_reduce_min, ReduceMin)
WAFER_DEFINE_REDUCE(wafer_tx81_reduce_avg, ReduceAvg)

#define WAFER_DEFINE_CONVERT_ZP(SYMBOL, METHOD)                                \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t zero_point, uint32_t rounding_mode) {                   \
    (void)rounding_mode;                                                       \
    TsmConvertInstr instr = {0};                                               \
    TsmConvert *convert = TsmNewConvert();                                     \
    convert->METHOD(&instr, src, zero_point, dst, elem_count);                 \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteConvert(convert);                                                 \
  }

#define WAFER_DEFINE_CONVERT_ROUND(SYMBOL, METHOD)                             \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t zero_point, uint32_t rounding_mode) {                   \
    (void)zero_point;                                                          \
    TsmConvertInstr instr = {0};                                               \
    TsmConvert *convert = TsmNewConvert();                                     \
    convert->METHOD(&instr, src, dst, elem_count,                              \
                    wafer_rounding(rounding_mode));                            \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteConvert(convert);                                                 \
  }

#define WAFER_DEFINE_CONVERT_PLAIN(SYMBOL, METHOD)                             \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t zero_point, uint32_t rounding_mode) {                   \
    (void)zero_point;                                                          \
    (void)rounding_mode;                                                       \
    TsmConvertInstr instr = {0};                                               \
    TsmConvert *convert = TsmNewConvert();                                     \
    convert->METHOD(&instr, src, dst, elem_count);                             \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteConvert(convert);                                                 \
  }

WAFER_DEFINE_CONVERT_ZP(wafer_tx81_convert_int8_fp16, INT8_FP16)
WAFER_DEFINE_CONVERT_ZP(wafer_tx81_convert_int8_bf16, INT8_BF16)
WAFER_DEFINE_CONVERT_ZP(wafer_tx81_convert_int8_fp32, INT8_FP32)
WAFER_DEFINE_CONVERT_ZP(wafer_tx81_convert_int8_tf32, INT8_TF32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_int16_fp16, INT16_FP16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int16_bf16, INT16_BF16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int16_fp32, INT16_FP32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int16_tf32, INT16_TF32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int32_fp16, INT32_FP16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int32_bf16, INT32_BF16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int32_fp32, INT32_FP32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int32_tf32, INT32_TF32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_bf16_int8, BF16_INT8)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_bf16_int16, BF16_INT16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_bf16_int32, BF16_INT32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_bf16_fp16, BF16_FP16)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_bf16_fp32, BF16_FP32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_bf16_tf32, BF16_TF32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp16_int8, FP16_INT8)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp16_int16, FP16_INT16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp16_int32, FP16_INT32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp16_bf16, FP16_BF16)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_fp16_fp32, FP16_FP32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_fp16_tf32, FP16_TF32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_int8, FP32_INT8)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_int16, FP32_INT16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_int32, FP32_INT32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_fp16, FP32_FP16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_bf16, FP32_BF16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_tf32, FP32_TF32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_tf32_int8, TF32_INT8)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_tf32_int16, TF32_INT16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_tf32_int32, TF32_INT32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_tf32_fp16, TF32_FP16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_tf32_bf16, TF32_BF16)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_tf32_fp32, TF32_FP32)

void wafer_tx81_gemm(uint64_t lhs, uint64_t rhs, uint64_t dst, uint32_t m,
                     uint32_t k, uint32_t n, uint32_t batch_count,
                     uint32_t format) {
  TsmNeInstr instr = {0};
  TsmGemm *gemm = TsmNewGemm();
  gemm->AddInput(&instr, lhs, rhs, wafer_format(format));
  gemm->ConfigMKN(&instr, m, k, n);
  gemm->ConfigBatch(&instr, batch_count, batch_count);
  /* TX81 encodes the RHS hardware transpose bit opposite to GEMM semantics. */
  gemm->SetTransflag(&instr, 0, 1);
  gemm->SetPsum(&instr, 0, 0, Fmt_UNUSED);
  gemm->SetQuant(&instr, 0, 0, 0, 0);
  gemm->AddBias(&instr, 0, 0);
  gemm->SetNegativeAxisScale(&instr, 0, 0);
  gemm->SetPositiveAxisScale(&instr, 0, 0);
  gemm->DisableRelu(&instr);
  gemm->DisableLeakyRelu(&instr);
  gemm->AddOutput(&instr, dst, wafer_format(format));
  wafer_execute_ne(&instr);
  TsmDeleteGemm(gemm);
}

void wafer_tx81_gemm_oriented_v2(uint64_t lhs, uint64_t rhs, uint64_t dst,
                                 uint32_t m, uint32_t k, uint32_t n,
                                 uint32_t batch_count, uint32_t format,
                                 uint32_t lhs_orientation,
                                 uint32_t rhs_orientation) {
  TsmNeInstr instr = {0};
  TsmGemm *gemm = TsmNewGemm();
  gemm->AddInput(&instr, lhs, rhs, wafer_format(format));
  gemm->ConfigMKN(&instr, m, k, n);
  gemm->ConfigBatch(&instr, batch_count, batch_count);
  /* Keep the public ABI semantic; invert only the raw RHS hardware bit. */
  gemm->SetTransflag(&instr, lhs_orientation, !rhs_orientation);
  gemm->SetPsum(&instr, 0, 0, Fmt_UNUSED);
  gemm->SetQuant(&instr, 0, 0, 0, 0);
  gemm->AddBias(&instr, 0, 0);
  gemm->SetNegativeAxisScale(&instr, 0, 0);
  gemm->SetPositiveAxisScale(&instr, 0, 0);
  gemm->DisableRelu(&instr);
  gemm->DisableLeakyRelu(&instr);
  gemm->AddOutput(&instr, dst, wafer_format(format));
  wafer_execute_ne(&instr);
  TsmDeleteGemm(gemm);
}

#define WAFER_CONFIGURE_CONV(OP, INSTR)                                        \
  do {                                                                         \
    OP->AddInput(INSTR, input,                                                 \
                 wafer_shape4(input_n, input_h, input_w, input_c),             \
                 wafer_format(format));                                        \
    OP->AddWeight(INSTR, weight,                                               \
                  wafer_shape4(weight_n, weight_h, weight_w, weight_c),        \
                  wafer_format(format));                                       \
    OP->AddBias(INSTR, 0, 0);                                                  \
    OP->AddOutput(INSTR, dst,                                                  \
                  wafer_shape4(output_n, output_h, output_w, output_c),        \
                  wafer_format(format));                                       \
    OP->SetOpType(INSTR, (uint8_t)kind);                                       \
    OP->SetNegativeAxisScale(INSTR, 0, 0);                                     \
    OP->SetPositiveAxisScale(INSTR, 0, 0);                                     \
    OP->SetSparse(INSTR, 0, 0);                                                \
    OP->SetPsum(INSTR, 0, 0, Fmt_UNUSED);                                      \
    OP->SetPads(INSTR, pad_top, pad_bottom, pad_left, pad_right);              \
    OP->SetUnPads(INSTR, unpad_top, unpad_bottom, unpad_left, unpad_right);    \
    OP->SetKernelStrides(INSTR, kernel_x, kernel_y, stride_x, stride_y);       \
    OP->SetDilations(INSTR, dilation0, dilation1);                             \
    OP->SetQuant(INSTR, 0, 0, 0, 0);                                           \
    OP->DisableRelu(INSTR);                                                    \
    OP->DisableLeakyRelu(INSTR);                                               \
  } while (0)

void wafer_tx81_conv(uint64_t input, uint64_t weight, uint64_t dst,
                     uint32_t kind, uint32_t input_n, uint32_t input_h,
                     uint32_t input_w, uint32_t input_c, uint32_t weight_n,
                     uint32_t weight_h, uint32_t weight_w, uint32_t weight_c,
                     uint32_t output_n, uint32_t output_h, uint32_t output_w,
                     uint32_t output_c, uint32_t pad_top, uint32_t pad_bottom,
                     uint32_t pad_left, uint32_t pad_right, uint32_t unpad_top,
                     uint32_t unpad_bottom, uint32_t unpad_left,
                     uint32_t unpad_right, uint32_t kernel_x, uint32_t kernel_y,
                     uint32_t stride_x, uint32_t stride_y, uint32_t dilation0,
                     uint32_t dilation1, uint32_t format) {
  TsmNeInstr instr = {0};
  TsmConv *conv = TsmNewConv();
  WAFER_CONFIGURE_CONV(conv, &instr);
  wafer_execute_ne(&instr);
  TsmDeleteConv(conv);
}

void wafer_tx81_depthwise_conv(
    uint64_t input, uint64_t weight, uint64_t dst, uint32_t kind,
    uint32_t input_n, uint32_t input_h, uint32_t input_w, uint32_t input_c,
    uint32_t weight_n, uint32_t weight_h, uint32_t weight_w, uint32_t weight_c,
    uint32_t output_n, uint32_t output_h, uint32_t output_w, uint32_t output_c,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right, uint32_t unpad_top, uint32_t unpad_bottom,
    uint32_t unpad_left, uint32_t unpad_right, uint32_t kernel_x,
    uint32_t kernel_y, uint32_t stride_x, uint32_t stride_y, uint32_t dilation0,
    uint32_t dilation1, uint32_t format) {
  TsmNeInstr instr = {0};
  TsmDepthwiseConv *conv = TsmNewDepthwiseConv();
  WAFER_CONFIGURE_CONV(conv, &instr);
  wafer_execute_ne(&instr);
  TsmDeleteDepthwiseConv(conv);
}

void wafer_tx81_backward_conv(
    uint64_t input, uint64_t weight, uint64_t dst, uint32_t kind,
    uint32_t input_n, uint32_t input_h, uint32_t input_w, uint32_t input_c,
    uint32_t weight_n, uint32_t weight_h, uint32_t weight_w, uint32_t weight_c,
    uint32_t output_n, uint32_t output_h, uint32_t output_w, uint32_t output_c,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right, uint32_t unpad_top, uint32_t unpad_bottom,
    uint32_t unpad_left, uint32_t unpad_right, uint32_t kernel_x,
    uint32_t kernel_y, uint32_t stride_x, uint32_t stride_y, uint32_t dilation0,
    uint32_t dilation1, uint32_t format) {
  TsmNeInstr instr = {0};
  TsmConv *conv = TsmNewConv();
  WAFER_CONFIGURE_CONV(conv, &instr);
  wafer_execute_ne(&instr);
  TsmDeleteConv(conv);
}

#define WAFER_DEFINE_POOL(SYMBOL, METHOD)                                      \
  void SYMBOL(uint64_t input, uint64_t dst, uint32_t kind, uint32_t src_n,     \
              uint32_t src_h, uint32_t src_w, uint32_t src_c, uint32_t dst_n,  \
              uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,                  \
              uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,        \
              uint32_t pad_right, uint32_t kernel_x, uint32_t kernel_y,        \
              uint32_t stride_x, uint32_t stride_y, uint32_t format) {         \
    (void)kind;                                                                \
    (void)dst_n;                                                               \
    (void)dst_h;                                                               \
    (void)dst_w;                                                               \
    (void)dst_c;                                                               \
    TsmPoolInstr instr = {0};                                                  \
    TsmPool *pool = TsmNewPool();                                              \
    pool->METHOD(&instr, input, wafer_shape4(src_n, src_h, src_w, src_c), dst, \
                 wafer_shape4(pad_top, pad_bottom, pad_left, pad_right),       \
                 wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),         \
                 wafer_format(format));                                        \
    wafer_execute_ct(&instr);                                                  \
    TsmDeletePool(pool);                                                       \
  }

#define WAFER_DEFINE_POOL_INDEXED(SYMBOL, METHOD)                              \
  void SYMBOL(uint64_t input, uint64_t value_dst, uint64_t index_dst,          \
              uint32_t kind, uint32_t src_n, uint32_t src_h, uint32_t src_w,   \
              uint32_t src_c, uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,  \
              uint32_t dst_c, uint32_t pad_top, uint32_t pad_bottom,           \
              uint32_t pad_left, uint32_t pad_right, uint32_t kernel_x,        \
              uint32_t kernel_y, uint32_t stride_x, uint32_t stride_y,         \
              uint32_t format) {                                               \
    (void)kind;                                                                \
    (void)dst_n;                                                               \
    (void)dst_h;                                                               \
    (void)dst_w;                                                               \
    (void)dst_c;                                                               \
    TsmPoolInstr instr = {0};                                                  \
    TsmPool *pool = TsmNewPool();                                              \
    pool->METHOD(&instr, input, wafer_shape4(src_n, src_h, src_w, src_c),      \
                 value_dst, index_dst,                                         \
                 wafer_shape4(pad_top, pad_bottom, pad_left, pad_right),       \
                 wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),         \
                 wafer_format(format));                                        \
    wafer_execute_ct(&instr);                                                  \
    TsmDeletePool(pool);                                                       \
  }

WAFER_DEFINE_POOL(wafer_tx81_pool_avg, AvgPool)
WAFER_DEFINE_POOL(wafer_tx81_pool_sum, SumPool)
WAFER_DEFINE_POOL(wafer_tx81_pool_max, MaxPool)
WAFER_DEFINE_POOL(wafer_tx81_pool_min, MinPool)
WAFER_DEFINE_POOL_INDEXED(wafer_tx81_pool_indexedmax, IndexdMaxPool)
WAFER_DEFINE_POOL_INDEXED(wafer_tx81_pool_indexedmin, IndexdMinPool)

void wafer_tx81_unpool_unpool(uint64_t input, uint64_t dst, uint32_t kind,
                              uint32_t index, uint32_t src_n, uint32_t src_h,
                              uint32_t src_w, uint32_t src_c, uint32_t dst_n,
                              uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
                              uint32_t kernel_x, uint32_t kernel_y,
                              uint32_t stride_x, uint32_t stride_y,
                              uint32_t format) {
  (void)kind;
  (void)src_n;
  (void)src_h;
  (void)src_w;
  (void)src_c;
  TsmUnPoolInstr instr = {0};
  TsmUnPool *unpool = TsmNewUnPool();
  unpool->Unpool(&instr, input, index, dst,
                 wafer_shape4(dst_n, dst_h, dst_w, dst_c),
                 wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),
                 wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeleteUnPool(unpool);
}

void wafer_tx81_unpool_avg(uint64_t input, uint64_t dst, uint32_t kind,
                           uint32_t index, uint32_t src_n, uint32_t src_h,
                           uint32_t src_w, uint32_t src_c, uint32_t dst_n,
                           uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
                           uint32_t kernel_x, uint32_t kernel_y,
                           uint32_t stride_x, uint32_t stride_y,
                           uint32_t format) {
  (void)kind;
  (void)index;
  (void)src_n;
  (void)src_h;
  (void)src_w;
  (void)src_c;
  TsmUnPoolInstr instr = {0};
  TsmUnPool *unpool = TsmNewUnPool();
  unpool->UnpoolAvg(&instr, input, dst,
                    wafer_shape4(dst_n, dst_h, dst_w, dst_c),
                    wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),
                    wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeleteUnPool(unpool);
}

void wafer_tx81_unpool_mask(uint64_t input, uint64_t dst, uint32_t kind,
                            uint32_t index, uint32_t src_n, uint32_t src_h,
                            uint32_t src_w, uint32_t src_c, uint32_t dst_n,
                            uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
                            uint32_t kernel_x, uint32_t kernel_y,
                            uint32_t stride_x, uint32_t stride_y,
                            uint32_t format) {
  (void)kind;
  (void)src_n;
  (void)src_h;
  (void)src_w;
  (void)src_c;
  TsmUnPoolInstr instr = {0};
  TsmUnPool *unpool = TsmNewUnPool();
  unpool->UnpoolIdx(&instr, input, index, dst,
                    wafer_shape4(dst_n, dst_h, dst_w, dst_c),
                    wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),
                    wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeleteUnPool(unpool);
}

void wafer_tx81_tdma_pad(uint64_t src, uint64_t dst, uint32_t src_n,
                         uint32_t src_h, uint32_t src_w, uint32_t src_c,
                         uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,
                         uint32_t dst_c, uint32_t pad_top, uint32_t pad_bottom,
                         uint32_t pad_left, uint32_t pad_right,
                         uint32_t format) {
  TsmDataMoveInstr instr = {0};
  TsmDataMove *move = TsmNewDataMove();
  move->Pad(&instr, src, wafer_shape4(src_n, src_h, src_w, src_c), dst,
            wafer_shape4(dst_n, dst_h, dst_w, dst_c),
            wafer_shape4(pad_top, pad_bottom, pad_left, pad_right),
            wafer_format(format));
  wafer_execute_td(&instr);
  TsmDeleteDataMove(move);
}

void wafer_tx81_tdma_img2col(uint64_t src, uint64_t dst, uint32_t src_n,
                             uint32_t src_h, uint32_t src_w, uint32_t src_c,
                             uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,
                             uint32_t dst_c, uint32_t pad_top,
                             uint32_t pad_bottom, uint32_t pad_left,
                             uint32_t pad_right, uint32_t kernel_x,
                             uint32_t kernel_y, uint32_t stride_x,
                             uint32_t stride_y, uint32_t format) {
  Data_Shape src_shape = wafer_shape4(src_n, src_h, src_w, src_c);
  Data_Shape dst_shape = wafer_shape4(dst_n, dst_h, dst_w, dst_c);
  TsmDataMoveInstr instr = {0};
  TsmDataMove *move = TsmNewDataMove();
  move->Img2col(&instr, src, src_shape, dst, dst_shape,
                wafer_shape_elements(src_shape),
                wafer_shape_elements(dst_shape),
                wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),
                wafer_shape4(pad_top, pad_bottom, pad_left, pad_right),
                wafer_format(format));
  wafer_execute_td(&instr);
  TsmDeleteDataMove(move);
}

static void wafer_arg_writeback(uint64_t value_dst, uint64_t index_dst,
                                uint32_t format, TsmPeripheralInstr *instr) {
#ifdef WAFER_TX81_PROFILE_TRACE_CRT
  (void)wafer_profile_wait_local_completion();
#else
  (void)TsmWaitfinish();
#endif
  uint64_t value_addr = wafer_spm_mapped_addr(value_dst);
  uint64_t index_addr = wafer_spm_mapped_addr(index_dst);
  wafer_store_value(value_addr, format, instr->param.wb_data0);
  wafer_store_u32(index_addr, (uint32_t)instr->param.wb_data1);
  wafer_order_mapped_spm();
}

void wafer_tx81_peripheral_argmax(uint64_t src, uint64_t value_dst,
                                  uint64_t index_dst, uint32_t kind,
                                  uint32_t elem_count, uint32_t format,
                                  uint32_t lut_elem_count, uint32_t scale,
                                  uint32_t probability,
                                  uint32_t rounding_mode) {
  (void)kind;
  (void)lut_elem_count;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->ArgMax(&instr, src, elem_count, wafer_format(format));
  wafer_execute_ct(&instr);
  wafer_arg_writeback(value_dst, index_dst, format, &instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_argmin(uint64_t src, uint64_t value_dst,
                                  uint64_t index_dst, uint32_t kind,
                                  uint32_t elem_count, uint32_t format,
                                  uint32_t lut_elem_count, uint32_t scale,
                                  uint32_t probability,
                                  uint32_t rounding_mode) {
  (void)kind;
  (void)lut_elem_count;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->ArgMin(&instr, src, elem_count, wafer_format(format));
  wafer_execute_ct(&instr);
  wafer_arg_writeback(value_dst, index_dst, format, &instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_bilinear(
    uint64_t src, uint64_t dst, uint32_t kind, uint32_t elem_count,
    uint32_t format, uint32_t src_n, uint32_t src_h, uint32_t src_w,
    uint32_t src_c, uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,
    uint32_t dst_c, uint32_t lut_elem_count, uint32_t scale,
    uint32_t probability, uint32_t rounding_mode) {
  (void)kind;
  (void)elem_count;
  (void)lut_elem_count;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->Bilinear(
      &instr, src, dst, wafer_shape4(src_n, src_h, src_w, src_c),
      wafer_shape4(dst_n, dst_h, dst_w, dst_c),
      wafer_bilinear_scale(src_w, dst_w), wafer_bilinear_scale(src_h, dst_h),
      wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_lut16(uint64_t src, uint64_t lut, uint64_t dst,
                                 uint32_t kind, uint32_t elem_count,
                                 uint32_t format, uint32_t lut_elem_count,
                                 uint32_t scale, uint32_t probability,
                                 uint32_t rounding_mode) {
  (void)kind;
  (void)format;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->Lut16(&instr, src, dst, lut, elem_count, lut_elem_count);
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_lut32(uint64_t src, uint64_t lut, uint64_t dst,
                                 uint32_t kind, uint32_t elem_count,
                                 uint32_t format, uint32_t lut_elem_count,
                                 uint32_t scale, uint32_t probability,
                                 uint32_t rounding_mode) {
  (void)kind;
  (void)format;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->Lut32(&instr, src, dst, lut, elem_count, lut_elem_count);
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_rand_gen(uint64_t src0, uint64_t src1, uint64_t dst0,
                                    uint64_t dst1, uint64_t dst2, uint32_t kind,
                                    uint32_t elem_count, uint32_t format,
                                    uint32_t lut_elem_count, uint32_t scale,
                                    uint32_t probability,
                                    uint32_t rounding_mode) {
  (void)kind;
  (void)lut_elem_count;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->RandGen(&instr, src0, src1, dst0, dst1, dst2, elem_count,
                      wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_elem_mask(uint64_t src, uint64_t dst, uint32_t kind,
                                     uint32_t elem_count, uint32_t format,
                                     uint32_t lut_elem_count, uint32_t scale,
                                     uint32_t probability,
                                     uint32_t rounding_mode) {
  (void)kind;
  (void)lut_elem_count;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->ElemMask(&instr, src, scale, dst, elem_count,
                       wafer_format(format), probability,
                       wafer_rounding(rounding_mode));
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

/*
 * TsmExecute publishes NCC work through MMIO while TsmWaitfinish only polls
 * taskstatus.  Order the issue before polling and the observed completion
 * before any following issue or memory access.  This does not widen the
 * default wait's worker scope.
 */
static void wafer_order_local_completion(void) {
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

void wafer_tx81_local_fence(void) {
  wafer_order_local_completion();
#ifdef WAFER_TX81_PROFILE_TRACE_CRT
  (void)wafer_profile_wait_local_completion();
#else
  (void)TsmWaitfinish();
#endif
  wafer_order_local_completion();
}

void wafer_tx81_ncc_join(uint32_t participant_mask) {
  wafer_order_local_completion();
  for (uint32_t worker = 0; worker < WAFER_TX81_NCC_WORKER_COUNT; ++worker) {
    if ((participant_mask & (UINT32_C(1) << worker)) == 0U)
      continue;
#ifdef WAFER_TX81_PROFILE_TRACE_CRT
    (void)wafer_profile_wait_ncc_worker_completion(worker);
#else
    (void)TsmWaitfinish_bywork(worker);
#endif
  }
  wafer_order_local_completion();
}
