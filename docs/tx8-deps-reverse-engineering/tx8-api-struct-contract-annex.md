# TX8 API and Struct Contract Annex

This annex expands concrete API signatures and C struct shapes from the reversed dependency package. It is generated from current `tx8_deps` headers and `libtx8_runtime.so` symbols, then interpreted with the disassembly notes in `tx8-interface-contract.md`.

Scope note: this annex is a generated/static tx8-deps artifact.  It does not
try to list the `firmware_kuiper` KMD UAPI structs or HPGR `tx_runtime.h`
structs; those are summarized in
`firmware-kuiper-runtime-hardware-analysis.md`.  When interpreting fields such
as `serial_mode`, DTE mode, `TileMappingTable`, or `TileDteCfg`, use the layer
notes in `tx8-interface-contract.md`: wrapper/Kcore structs, KMD UAPI structs,
and HPGR runtime structs are related but not identical contracts.

## Host Runtime Exported Signatures

- `TsmAsyncRun(TsmDevice*, unsigned long)`
- `TsmClusterKernelLaunch(TsmDevice*, char const*, unsigned long, unsigned long, Dim3, Dim3, Dim3, void*, unsigned int)`
- `TsmCompile(TsmDevice*, TsmModel&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, CompileOption)`
- `TsmCompileMultiGraph(TsmDevice*, TsmModel&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, CompileOption)`
- `TsmDeInitRuntime()`
- `TsmDeviceFree(unsigned long)`
- `TsmDeviceMalloc(TsmDevice*, unsigned long&, unsigned long)`
- `TsmDeviceSynchronize(TsmDevice*)`
- `TsmGetDeviceList(unsigned int&, std::vector<unsigned int, std::allocator<unsigned int> >&)`
- `TsmGetDeviceNum(unsigned int&)`
- `TsmGetDeviceProperties(unsigned int, TsmDeviceProp*)`
- `TsmGetPhyRankId(unsigned int*, unsigned int*)`
- `TsmGetTileInfo(TsmDevice*, TsmTileTotalInfo&)`
- `TsmGraphCompile(TsmDevice*, TsmModel&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, CompileOption)`
- `TsmHostFlush(TsmDevice*, unsigned long, unsigned char*, unsigned long)`
- `TsmHostH2D(TsmDevice*, unsigned long, unsigned long, int)`
- `TsmInitDevice(TsmDevice*)`
- `TsmInitRuntime(bool)`
- `TsmKernelLaunch(TsmDevice*, char const*, unsigned long, unsigned long, Dim3, Dim3, void*, unsigned int)`
- `TsmLaunch(TsmDevice*, TsmModel&)`
- `TsmLaunchPg(TsmDevice*, TsmModel&)`
- `TsmLoadKernel(TsmDevice*, std::vector<TsmModel*, std::allocator<TsmModel*> >&, char*)`
- `TsmMemGetInfo(unsigned long, unsigned int&, unsigned long&, unsigned long&)`
- `TsmMemcpyD2D(void const*, TsmDevice*, void const*, TsmDevice*, unsigned long)`
- `TsmMemcpyD2H(void const*, unsigned long, unsigned long)`
- `TsmMemcpyH2D(unsigned long, void const*, unsigned long)`
- `TsmMemcpyOffsetD2H(void const*, unsigned long, unsigned long, unsigned long)`
- `TsmMemcpyOffsetH2D(unsigned long, void const*, unsigned long, unsigned long)`
- `TsmModel::TsmModel()`
- `TsmModel::TsmModel(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)`
- `TsmModel::~TsmModel()`
- `TsmNpuPowerOff(TsmDevice*)`
- `TsmNpuPowerOn(TsmDevice*, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > >)`
- `TsmProcessProfData(TsmDevice*, TsmProfAction, unsigned short)`
- `TsmRecv(void*, unsigned long, txcclDataType_t, TsmDevice*, int, txcclComm*, void*)`
- `TsmReleaseDevice(TsmDevice*)`
- `TsmResetDevice(TsmDevice*)`
- `TsmRun(TsmDevice*, unsigned long)`
- `TsmSend(void const*, unsigned long, txcclDataType_t, TsmDevice*, int, txcclComm*, void*)`
- `TsmSetDevice(TsmDevice**, unsigned int, unsigned int)`
- `TsmSetDeviceOld(unsigned int, TsmDevice*)`
- `TsmSetMonitorInfo(TsmDevice*)`
- `TsmSetRankId(unsigned int, unsigned int)`
- `TsmSetRankSize(unsigned int, unsigned int)`
- `TsmSetTerminate(TsmDevice*, void*)`
- `TsmSetTileIdMap(TsmDevice*, FullTileMap_t const&)`
- `TsmSetTileInfo(TsmDevice*, TsmTileSelectedInfo)`
- `TsmUnloadKernel(TsmDevice*, std::vector<TsmModel*, std::allocator<TsmModel*> >&)`

## Kcore Wrapper Public Constructors/Destructors/CSR

- `TsmConv *TsmNewConv();`
- `TsmDepthwiseConv *TsmNewDepthwiseConv();`
- `TsmGemm *TsmNewGemm();`
- `TsmRdma *TsmNewRdma();`
- `TsmWdma *TsmNewWdma();`
- `TsmArith *TsmNewArith();`
- `TsmRelation *TsmNewRelation();`
- `TsmLogic *TsmNewLogic();`
- `TsmTranscendental *TsmNewTranscendental();`
- `TsmActivation *TsmNewActivation();`
- `TsmReduce *TsmNewReduce();`
- `TsmPool *TsmNewPool();`
- `TsmUnPool *TsmNewUnPool();`
- `TsmMaskDataMove *TsmNewMaskDataMove();`
- `TsmConvert *TsmNewConvert();`
- `TsmPeripheral *TsmNewPeripheral();`
- `TsmDataMove *TsmNewDataMove();`
- `void TsmDeleteConv(TsmConv *obj);`
- `void TsmDeleteDepthwiseConv(TsmDepthwiseConv *obj);`
- `void TsmDeleteGemm(TsmGemm *obj);`
- `void TsmDeleteRdma(TsmRdma *obj);`
- `void TsmDeleteWdma(TsmWdma *obj);`
- `void TsmDeleteArith(TsmArith *obj);`
- `void TsmDeleteRelation(TsmRelation *obj);`
- `void TsmDeleteLogic(TsmLogic *obj);`
- `void TsmDeleteTranscendental(TsmTranscendental *obj);`
- `void TsmDeleteActivation(TsmActivation *obj);`
- `void TsmDeleteReduce(TsmReduce *obj);`
- `void TsmDeletePool(TsmPool *obj);`
- `void TsmDeleteUnPool(TsmUnPool *obj);`
- `void TsmDeleteMaskDataMove(TsmMaskDataMove *obj);`
- `void TsmDeleteConvert(TsmConvert *obj);`
- `void TsmDeletePeripheral(TsmPeripheral *obj);`
- `void TsmDeleteDataMove(TsmDataMove *obj);`
- `TsmStream *TsmNewStream();`
- `void TsmDeleteStream(TsmStream *obj);`
- `uint8_t TsmWaitfinish();`
- `uint8_t TsmGetCsrTaskstatus();`
- `uint8_t TsmGetCsrIbcounter();`
- `uint8_t TsmGetCsrTaskstatus_bywork(size_t workerid);`
- `uint8_t TsmWaitfinish_bywork(size_t workerid);`
- `uint64_t TsmExecute(void *instr);`

## DTE and Stream Public Declarations

- `int dte_init();`
- `uint8_t mod_kuiper_dte_check_free_cnt(void);`
- `int mod_kuiper_dte_release(mod_kuiper_dte_node_t *node);`
- `int mod_kuiper_dte_config_src_and_dst(mod_kuiper_dte_node_t *node, uint16_t tile_logic_id, uint64_t src_addr, uint64_t dst_addr, uint32_t data_len, kuiper_dte_shuffle_cfg_t *shuffle_cfg);`
- `int mod_kuiper_dte_trig_send(mod_kuiper_dte_node_t *node);`
- `int mod_kuiper_dte_check_send_status(mod_kuiper_dte_node_t *node);`
- `void mod_kuiper_dte_auto_update_packet_cnt(mod_kuiper_dte_node_t *node);`
- `int mod_kuiper_dte_clear_dma_status(mod_kuiper_dte_node_t *node);`
- `int mod_kuiper_dte_set_pmu_en(uint8_t channel, uint8_t en);`
- `uint8_t mod_kuiper_dte_get_pmu_en(uint8_t channel);`
- `int mod_kuiper_dte_clear_pmu_reg(uint8_t channel);`
- `uint8_t mod_kuiper_dte_pmu_get_work_status(uint8_t channel);`
- `uint64_t mod_kuiper_dte_pmu_get_clk_cnts(void);`
- `uint64_t mod_kuiper_dte_pmu_get_clk_cnts_one_channel(uint8_t channel);`
- `uint32_t mod_kuiper_dte_pmu_get_cmd_success_cnts(uint8_t channel);`
- `uint8_t mod_kuiper_dte_pmu_get_cmd_failed_cnts(uint8_t channel);`
- `uint64_t mod_kuiper_dte_pmu_get_transfer_data_cnts(uint8_t channel);`
- `uint64_t mod_kuiper_dte_pmu_get_unalign_burst_cnts(uint8_t channel);`
- `uint64_t mod_kuiper_dte_pmu_get_idle_clk_cnts(uint8_t channel);`
- `uint64_t mod_kuiper_dte_pmu_get_all_idle_clk_cnts(void);`
- `uint32_t GenPayload(uint64_t* payload, uint64_t tile_id, uint64_t core_id, uint64_t channel_id, uint64_t op_type, uint64_t stream_type, uint64_t stream_id, uint64_t stream_addr);`
- `uint32_t SendMailbox(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id, uint32_t remote, uint64_t* payload);`
- `uint32_t OnlineStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id, uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);`
- `uint32_t OnlineStreamPreload(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id, uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr , uint8_t preload_pkt_cnt);`
- `uint32_t OfflineStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id, uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);`
- `uint32_t WaitStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id, uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);`
- `uint32_t ReqStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id, uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);`
- `uint32_t PushStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id, uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);`
- `uint32_t PopStream(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id, uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);`

## Enums

### `D_DynDataType`

```c
enum D_DynDataType {
    D_PKT_FINAL_TYPE = 0,
    D_CFG_PMU_TYPE,
    D_KCORE_CFG_TYPE,
    D_EXPORT_SPM_TYPE,
    D_DISABLE_CALC_TYPE,
    D_PROF_CFG_TYPE,
    D_DYNLIB_LOAD,
    D_DYNLIB_RUN,
    D_DYNLIB_UNLOAD,
    D_MEMCPY_D2D,
    D_P2P_SEND,
    D_P2P_RECV,
    D_GROUP_DATA_DUMP_TYPE,
    D_DATA_TYPE_MAX,
};
```

### `Data_Format`

```c
typedef enum Data_Format {
    Fmt_INT8   = 0,
    Fmt_INT16  = 1,
    Fmt_FP16   = 2,
    Fmt_BF16   = 3,
    Fmt_INT32  = 4,
    Fmt_FP32   = 5,
    Fmt_TF32   = 6,
    Fmt_BOOL   = 7,    // 1/8 BYTE
    Fmt_UINT8  = 8,
    Fmt_UINT16 = 9,
    Fmt_UINT32 = 10,
    Fmt_INT64  = 11,
    Fmt_UINT64 = 12,
    Fmt_UNUSED,
} Data_Format;
```

### `KrtRetCode`

```c
typedef enum KrtRetCode {
    RET_OK = 0,          // 正常退出
    RET_EXIT,            // 主函数结束标志
    RET_ERR,             // 通用错误
    RET_ERR_NOMEM,       // 内存不足
    RET_NO_VERSION,      // 未找到版本信息
    RET_ERR_VERSION,     // 版本不匹配
    RET_ERR_SYMBOL,      // 符号解析错误
    RET_ERR_MODEL_NAME,  // 模型名称错误
    RET_ERR_DTE,         // DTE传输错误
    RET_ERR_MAILBOX,     // mailbox传输错误
    RET_ERR_RB_MSG,      // AP传输过来的ringbuffer消息错误
    RET_ERR_PROF_CONFIG, // HOST下发的profiling config地址错误
} KrtRetCode;
```

### `OP_FUNC_CGRA`

```c
typedef enum OP_FUNC_CGRA {
    // Arithmetic Operators
    OP_FUNC_CGRATensor_ArithOp_V_V_abs = 0,
    OP_FUNC_CGRATensor_ArithOp_V_V_recip = 1,
    OP_FUNC_CGRATensor_ArithOp_V_V_square = 2,
    OP_FUNC_CGRATensor_ArithOp_V_V_sqrt = 3,
    OP_FUNC_CGRATensor_ArithOp_V_V_rsqrt = 4,
    OP_FUNC_CGRATensor_ArithOp_V_V_neg = 5,
    OP_FUNC_CGRATensor_ArithOp_V_VV_max = 6,
    OP_FUNC_CGRATensor_ArithOp_V_VS_max = 7,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_max = 8,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_max_loop = 9,
    OP_FUNC_CGRATensor_ArithOp_V_VV_min = 10,
    OP_FUNC_CGRATensor_ArithOp_V_VS_min = 11,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_min = 12,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_min_loop = 13,
    OP_FUNC_CGRATensor_ArithOp_V_VV_add = 14,
    OP_FUNC_CGRATensor_ArithOp_V_VS_add = 15,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_add = 16,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_add_loop = 17,
    OP_FUNC_CGRATensor_ArithOp_V_VV_sub = 18,
    OP_FUNC_CGRATensor_ArithOp_V_VS_sub = 19,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_sub = 20,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_sub_loop = 21,
    OP_FUNC_CGRATensor_ArithOp_V_VV_mul = 22,
    OP_FUNC_CGRATensor_ArithOp_V_VS_mul = 23,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_mul = 24,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_mul_loop = 25,
    OP_FUNC_CGRATensor_ArithOp_V_VV_div = 26,
    OP_FUNC_CGRATensor_ArithOp_V_VS_div = 27,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_div = 28,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_div_loop = 29,

    // Relational Operators
    OP_FUNC_CGRATensor_RelaOp_V_VV_eq = 30,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_eq = 31,
    OP_FUNC_CGRATensor_RelaOp_V_VS_eq = 32,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_eq = 33,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_eq = 34,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_eq_loop = 35,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_eq = 36,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_eq_loop = 37,

    OP_FUNC_CGRATensor_RelaOp_V_VV_ne = 38,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_ne = 39,
    OP_FUNC_CGRATensor_RelaOp_V_VS_ne = 40,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_ne = 41,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_ne = 42,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_ne_loop = 43,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_ne = 44,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_ne_loop = 45,

    OP_FUNC_CGRATensor_RelaOp_V_VV_ge = 46,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_ge = 47,
    OP_FUNC_CGRATensor_RelaOp_V_VS_ge = 48,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_ge = 49,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_ge = 50,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_ge_loop = 51,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_ge = 52,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_ge_loop = 53,

    OP_FUNC_CGRATensor_RelaOp_V_VV_gt = 54,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_gt = 55,
    OP_FUNC_CGRATensor_RelaOp_V_VS_gt = 56,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_gt = 57,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_gt = 58,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_gt_loop = 59,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_gt = 60,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_gt_loop = 61,

    OP_FUNC_CGRATensor_RelaOp_V_VV_le = 62,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_le = 63,
    OP_FUNC_CGRATensor_RelaOp_V_VS_le = 64,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_le = 65,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_le = 66,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_le_loop = 67,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_le = 68,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_le_loop = 69,

    OP_FUNC_CGRATensor_RelaOp_V_VV_lt = 70,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_lt = 71,
    OP_FUNC_CGRATensor_RelaOp_V_VS_lt = 72,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_lt = 73,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_lt = 74,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_lt_loop = 75,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_lt = 76,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_lt_loop = 77,

    OP_FUNC_CGRATensor_LogicOp_V_V_not = 78,
    OP_FUNC_CGRATensor_LogicOp_V_VV_and = 79,
    OP_FUNC_CGRATensor_LogicOp_V_VV_or = 80,
    OP_FUNC_CGRATensor_LogicOp_V_VV_xor = 81,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_and = 82,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_or = 83,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_xor = 84,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_and_loop = 85,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_or_loop = 86,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_xor_loop = 87,

    OP_FUNC_CGRATensor_LogicOp_bV_bV_not = 88,
    OP_FUNC_CGRATensor_LogicOp_bV_bVbV_and = 89,
    OP_FUNC_CGRATensor_LogicOp_bV_bVbV_or = 90,
    OP_FUNC_CGRATensor_LogicOp_bV_bVbV_xor = 91,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_and = 92,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_or = 93,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_xor = 94,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_and_loop = 95,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_or_loop = 96,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_xor_loop = 97,

    // Transcendental Operator
    OP_FUNC_CGRATensor_TransOp_V_V_log2 = 98,
    OP_FUNC_CGRATensor_TransOp_V_V_ln = 99,
    OP_FUNC_CGRATensor_TransOp_V_V_pow2 = 100,
    OP_FUNC_CGRATensor_TransOp_V_V_exp = 101,
    OP_FUNC_CGRATensor_TransOp_V_V_exp_lp = 102,
    OP_FUNC_CGRATensor_TransOp_V_V_sin = 103,
    OP_FUNC_CGRATensor_TransOp_V_V_cos = 104,

    // Activation Operator
    OP_FUNC_CGRATensor_ActOp_V_V_tanh = 105,
    OP_FUNC_CGRATensor_ActOp_V_V_sigmoid = 106,
    OP_FUNC_CGRATensor_ActOp_V_V_relu = 107,
    OP_FUNC_CGRATensor_ActOp_V_V_satrelu = 108,
    OP_FUNC_CGRATensor_ActOp_V_V_leakyrelu = 109,
    OP_FUNC_CGRATensor_ActOp_V_V_softplus = 110,

    // Reduce Operator
    OP_FUNC_CGRATensor_ReduceOp_T_T_sum = 111,
    OP_FUNC_CGRATensor_ReduceOp_T_T_avg = 112,
    OP_FUNC_CGRATensor_ReduceOp_T_T_max = 113,
    OP_FUNC_CGRATensor_ReduceOp_T_T_min = 114,

    // Pool Operator
    OP_FUNC_CGRATensor_PoolOp_T_T_avg = 115,
    OP_FUNC_CGRATensor_PoolOp_T_T_sum = 116,
    OP_FUNC_CGRATensor_PoolOp_T_T_max = 117,
    OP_FUNC_CGRATensor_PoolOp_T_T_indexedmax = 118,
    OP_FUNC_CGRATensor_PoolOp_T_T_min = 119,
    OP_FUNC_CGRATensor_PoolOp_T_T_indexedmin = 120,

    // DataMove
    OP_FUNC_CGRATensor_DataMoveOp_T_T_unpool = 121,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_unpool_avg = 122,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_maskunpool = 123,
    // reshape
    OP_FUNC_CGRATensor_DataMoveOp_T_T_mirror = 124,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_transpose = 125,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_rotate90 = 126,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_rotate180 = 127,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_rotate270 = 128,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_nchw2nhwc = 129,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_nhwc2nchw = 130,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_concat = 131,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_pad = 132,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_channelnorm = 133,
    // datamove
    OP_FUNC_CGRATensor_DataMoveOp_V_V_maskmove = 134,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_gatherscatter = 135,
    OP_FUNC_CGRATensor_DataMoveOp_V_V_maskgather = 136,
    OP_FUNC_CGRATensor_DataMoveOp_V_bV_maskgather = 137,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_img2col = 138,

    // Conver Operator
    OP_FUNC_CGRATensor_ConvertOp_V_V_int8_fp16 = 139,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int8_bf16 = 140,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int8_fp32 = 141,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int8_tf32 = 142,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int16_fp16 = 143,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int16_bf16 = 144,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int16_fp32 = 145,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int16_tf32 = 146,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int32_fp16 = 147,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int32_bf16 = 148,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int32_fp32 = 149,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int32_tf32 = 150,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_int8 = 151,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_int16 = 152,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_int32 = 153,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_fp16 = 154,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_fp32 = 155,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_tf32 = 156,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_int8 = 157,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_int16 = 158,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_int32 = 159,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_bf16 = 160,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_fp32 = 161,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_tf32 = 162,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_int8 = 163,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_int16 = 164,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_int32 = 165,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_fp16 = 166,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_bf16 = 167,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_tf32 = 168,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_int8 = 169,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_int16 = 170,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_int32 = 171,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_fp16 = 172,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_bf16 = 173,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_fp32 = 174,

    // Peripheral Operator
    OP_FUNC_CGRATensor_PeriOp_S_V_count = 175,
    OP_FUNC_CGRATensor_PeriOp_S_bV_bitcount = 176,
    OP_FUNC_CGRATensor_PeriOp_V_V_argmax = 177,
    OP_FUNC_CGRATensor_PeriOp_V_V_argmin = 178,
    OP_FUNC_CGRATensor_PeriOp_T_memset = 179,
    OP_FUNC_CGRATensor_PeriOp_V_V_fp32_factorize = 180,
    OP_FUNC_CGRATensor_PeriOp_V_V_bit2fp = 181,
    OP_FUNC_CGRATensor_PeriOp_T_T_bilinear = 182,
    OP_FUNC_CGRATensor_PeriOp_V_V_lut16 = 183,
    OP_FUNC_CGRATensor_PeriOp_V_V_lut32 = 184,
    OP_FUNC_CGRATensor_PeriOp_V_rand_gen = 185,
    OP_FUNC_CGRATensor_PeriOp_V_V_elem_mask = 186,
} OP_FUNC_CGRA;
```

### `OP_INSTR_TYPE`

```c
typedef enum OP_INSTR_TYPE {
    I_CGRA,
    I_NEUR,
    I_RDMA,
    I_WDMA,
    I_TDMA,
    I_SCALAR,
    I_DTE,
    I_CSR,
} OP_INSTR_TYPE;
```

### `OP_INSTR_WORKER`

```c
typedef enum OP_INSTR_WORKER {
    I_WORKER0 = 0x0000,
    I_WORKER1 = 0x0100,
    I_WORKER2 = 0x0200,
} OP_INSTR_WORKER;
```

### `RND_MODE`

```c
typedef enum RND_MODE {
    RND_NEAREST_EVEN,
    RND_ZERO,
    RND_POS_INF,
    RND_NEG_INF,
    RND_STOCHASTIC
} RND_MODE;
```

### `Reduce_Dim`

```c
typedef enum Reduce_Dim {
    Reduce_C = 0,
    Reduce_W = 1,
    Reduce_H = 2,
    Reduce_HW = 4,
} Reduce_Dim;
```

### `Tensor_Fmt`

```c
typedef enum Tensor_Fmt {
    T_GemmM = 0, /*M K*/
    T_ConvA = 1, /*H W C*/
    T_ConvW = 2, /*Kx Ky F C*/
    T_Vec = 3,
    T_ConvNA = 4,
    T_ConvNW = 5,
} Tensor_Fmt;
```

### `kuiper_dte_mode_t`

```c
typedef enum {
	MOD_DTE_STATE_FREE,
	MOD_DTE_STATE_ALLOC,
	MOD_DTE_STATE_CONFIG,
	MOD_DTE_STATE_TRANSFER,
	MOD_DTE_STATE_DONE,
	MOD_DTE_STATE_ERR,
} mod_kuiper_dte_state_t;

typedef enum {
    KUIPER_DTE_MODE_UNICAST     = 0,
    KUIPER_DTE_MODE_SCATTER     = 1,
    KUIPER_DTE_MODE_BROADCAST   = 2,
    KUIPER_DTE_MODE_SHUFFLE     = 3,
    KUIPER_DTE_MODE_RDMA        = 4,
    KUIPER_DTE_MODE_WDMA        = 5,
    KUIPER_DTE_MODE_DDR2DDR_U2U = 6,
    KUIPER_DTE_MODE_DDR2DDR_SHUFFLE = 7
} kuiper_dte_mode_t;
```

## Wrapper Function Pointer Structs

### `TsmConv`

```c
typedef struct TsmConv {
    void (*AddInput)(TsmNeInstr *instr, uint64_t X_addr, Data_Shape shape, Data_Format fmt);
    void (*AddWeight)(TsmNeInstr *instr, uint64_t W_addr, Data_Shape shape, Data_Format fmt);
    void (*AddBias)(TsmNeInstr *instr, uint8_t bias_en, uint64_t bias_addr);
    void (*AddOutput)(TsmNeInstr *instr, uint64_t Out_addr, Data_Shape shape, Data_Format fmt);
    void (*SetOpType)(TsmNeInstr *instr, uint8_t type);
    void (*SetNegativeAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t scale_addr); //- negative axis
    void (*SetPositiveAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t scale_addr); //+ positive axis
    void (*SetSparse)(TsmNeInstr *instr, uint8_t sparse_en, uint64_t sparse_addr);
    void (*SetPsum)(TsmNeInstr *instr, uint8_t psum_en, uint64_t psum_addr, Data_Format fmt);
    void (*SetPads)(TsmNeInstr *instr, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);
    void (*SetUnPads)(TsmNeInstr *instr, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);
    void (*SetKernelStrides)(TsmNeInstr *instr, uint32_t Kx, uint32_t Ky, uint32_t Sx, uint32_t Sy);
    void (*SetDilations)(TsmNeInstr *instr, uint32_t d0, uint32_t d1);
    void (*EnableRelu)(TsmNeInstr *instr);
    void (*EnableLeakyRelu)(TsmNeInstr *instr);
    void (*DisableRelu)(TsmNeInstr *instr);
    void (*DisableLeakyRelu)(TsmNeInstr *instr);
    void (*SetQuant)(TsmNeInstr *instr, uint8_t q0, uint8_t q1, uint8_t zp_pre, uint8_t zp_cur);
    /* data */
} TsmConv;
```

### `TsmDepthwiseConv`

```c
typedef struct TsmDepthwiseConv {
    void (*AddInput)(TsmNeInstr *instr, uint64_t X_addr, Data_Shape shape, Data_Format fmt);
    void (*AddWeight)(TsmNeInstr *instr, uint64_t W_addr, Data_Shape shape, Data_Format fmt);
    void (*AddBias)(TsmNeInstr *instr, uint8_t bias_en, uint64_t bias_addr);
    void (*AddOutput)(TsmNeInstr *instr, uint64_t Out_addr, Data_Shape shape, Data_Format fmt);
    void (*SetOpType)(TsmNeInstr *instr, uint8_t type);
    void (*SetNegativeAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t scale_addr); //- negative axis
    void (*SetPositiveAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t scale_addr); //+ positive axis
    void (*SetSparse)(TsmNeInstr *instr, uint8_t sparse_en, uint64_t sparse_addr);
    void (*SetPsum)(TsmNeInstr *instr, uint8_t psum_en, uint64_t psum_addr, Data_Format fmt);
    void (*SetPads)(TsmNeInstr *instr, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);
    void (*SetUnPads)(TsmNeInstr *instr, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);
    void (*SetKernelStrides)(TsmNeInstr *instr, uint32_t Kx, uint32_t Ky, uint32_t Sx, uint32_t Sy);
    void (*SetDilations)(TsmNeInstr *instr, uint32_t d0, uint32_t d1);
    void (*EnableRelu)(TsmNeInstr *instr);
    void (*EnableLeakyRelu)(TsmNeInstr *instr);
    void (*DisableRelu)(TsmNeInstr *instr);
    void (*DisableLeakyRelu)(TsmNeInstr *instr);
    void (*SetQuant)(TsmNeInstr *instr, uint8_t q0, uint8_t q1, uint8_t zp_pre, uint8_t zp_cur);
    /* data */
} TsmDepthwiseConv;
```

### `TsmGemm`

```c
typedef struct TsmGemm {
    void (*AddInput)(TsmNeInstr *instr, uint64_t L_addr, uint64_t R_addr, Data_Format in_fmt);
    void (*ConfigMKN)(TsmNeInstr *instr, uint32_t M, uint32_t K, uint32_t N);
    void (*ConfigBatch)(TsmNeInstr *instr, uint32_t Left_batch, uint32_t Right_batch);
    void (*AddOutput)(TsmNeInstr *instr, uint64_t Out_addr, Data_Format Out_fmt);
    void (*SetPsum)(TsmNeInstr *instr, uint8_t psum_en, uint64_t psum_addr, Data_Format fmt);
    void (*SetTransflag)(TsmNeInstr *instr, uint8_t L_trans, uint8_t R_trans);
    void (*SetQuant)(TsmNeInstr *instr, uint8_t q0, uint8_t q1, uint8_t zp_left, uint8_t zp_right);
    void (*AddBias)(TsmNeInstr *instr, uint8_t bias_en, uint64_t addr);
    void (*SetNegativeAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t addr);
    void (*SetPositiveAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t addr);
    void (*EnableRelu)(TsmNeInstr *instr);
    void (*EnableLeakyRelu)(TsmNeInstr *instr);
    void (*DisableRelu)(TsmNeInstr *instr);
    void (*DisableLeakyRelu)(TsmNeInstr *instr);

    /* data */
} TsmGemm;
```

### `TsmRdma`

```c
typedef struct TsmRdma {
    void (*AddSrcDst)(TsmRdmaInstr *instr, uint64_t src, uint64_t dst, Data_Format fmt);
    void (*ConfigStrideIteration)(TsmRdmaInstr *instr, uint32_t elem_count, uint32_t stride0, uint32_t iteration0,
                                  uint32_t stride1, uint32_t iteration1, uint32_t stride2, uint32_t iteration2);
    void (*Rdma1d)(TsmRdmaInstr *instr, uint64_t src, uint64_t dst, uint32_t elem_count,
                   uint32_t format); //只有stride0,和iteration0,内层循环, 只复制一次
} TsmRdma;
```

### `TsmWdma`

```c
typedef struct TsmWdma {
    void (*AddSrcDst)(TsmWdmaInstr *instr, uint64_t src, uint64_t dst, Data_Format fmt);
    void (*ConfigStrideIteration)(TsmWdmaInstr *instr, uint32_t elem_count, uint32_t stride0, uint32_t iteration0,
                                  uint32_t stride1, uint32_t iteration1, uint32_t stride2, uint32_t iteration2);
    void (*Wdma1d)(TsmWdmaInstr *instr, uint64_t src, uint64_t dst, uint32_t elem_count,
                   uint32_t format); //只有stride0,和iteration0,内层循环, 只复制一次
} TsmWdma;
```

### `TsmArith`

```c
typedef struct TsmArith {
    void(*AbsVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void(*RecipVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void(*SquareVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void(*SqrtVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void(*RsqrtVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void(*NegVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void (*MaxVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                            uint32_t elem_count, RND_MODE reserved, Data_Format fmt);
    void (*MaxVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                              uint32_t elem_count, RND_MODE reserved, Data_Format fmt);
    void (*MaxVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                              uint32_t elem_count, uint32_t unit_elem_count, RND_MODE reserved, Data_Format fmt);
    void (*MaxVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                                    uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                                    uint32_t full_unit_elem_num, RND_MODE reserved, Data_Format fmt);
    void(*MinVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                              uint32_t elem_count, RND_MODE reserved, Data_Format fmt);
     void(*MinVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                              uint32_t elem_count, RND_MODE reserved, Data_Format fmt);
    void(*MinVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                                uint32_t elem_count, uint32_t unit_elem_count, RND_MODE reserved, Data_Format fmt);
    void(*MinVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                                    uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                                    uint32_t full_unit_elem_num, RND_MODE reserved, Data_Format fmt);
    void (*AddVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  RND_MODE rnd_mode, Data_Format fmt);
    void (*AddVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                  uint32_t elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*AddVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                    uint32_t unit_elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*AddVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                        uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                        uint32_t full_unit_elem_num, RND_MODE rnd_mode, Data_Format fmt);
    void (*SubVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  RND_MODE rnd_mode, Data_Format fmt);
    void (*SubVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                  uint32_t elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*SubVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                    uint32_t unit_elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*SubVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                        uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                        uint32_t full_unit_elem_num, RND_MODE rnd_mode, Data_Format fmt);
    void (*MulVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  RND_MODE rnd_mode, Data_Format fmt);
    void (*MulVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                  uint32_t elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*MulVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  uint32_t unit_elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*MulVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, RND_MODE rnd_mode, Data_Format fmt);
    void (*DivVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  RND_MODE rnd_mode, Data_Format fmt);
    void (*DivVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                  uint32_t elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*DivVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  uint32_t unit_elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*DivVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, RND_MODE rnd_mode, Data_Format fmt);
} TsmArith;
```

### `TsmRelation`

```c
typedef struct TsmRelation {
    void (*EqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*EqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*EqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*EqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*UnEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolUnEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*UnEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolUnEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*UnEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolUnEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*UnEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolUnEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*GreaterEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolGreaterEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*GreaterEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolGreaterEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*GreaterEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolGreaterEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*GreaterEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolGreaterEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*GreaterVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolGreaterVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*GreaterVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolGreaterVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*GreaterVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolGreaterVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*GreaterVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolGreaterVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*LessEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolLessEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*LessEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolLessEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*LessEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolLessEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*LessEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolLessEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*LessThenVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolLessThenVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*LessThenVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolLessThenVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*LessThenVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolLessThenVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*LessThenVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolLessThenVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
} TsmRelation;
```

### `TsmLogic`

```c
typedef struct TsmLogic {
    void (*NotV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*AndVV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*OrVV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*XorVV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*AndVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*OrVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*XorVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*AndVuVLoop)(TsmArithInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*OrVuVLoop)(TsmArithInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*XorVuVLoop)(TsmArithInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*BoolNotV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BoolAndV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BoolOrV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BoolXorV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BoolAndVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count);
    void (*BoolOrVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count);
    void (*BoolXorVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count);
    void (*BoolAndVuVLoop)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count,
                           uint32_t unit_elem_count, uint32_t full_elem_num, uint32_t full_unit_elem_num);
    void (*BoolOrVuVLoop)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count,
                          uint32_t unit_elem_count, uint32_t full_elem_num, uint32_t full_unit_elem_num);
    void (*BoolXorVuVLoop)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count,
                           uint32_t unit_elem_count, uint32_t full_elem_num, uint32_t full_unit_elem_num);
} TsmLogic;
```

### `TsmTranscendental`

```c
typedef struct TsmTranscendental {
    void (*Log2)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Ln)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Pow2)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Exp)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Explp)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Sin)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Cos)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
} TsmTranscendental;
```

### `TsmActivation`

```c
typedef struct TsmActivation {
    void (*Tanh)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Sigmoid)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Relu)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Satrelu)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Leakyrelu)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Softplus)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
} TsmActivation;
```

### `TsmReduce`

```c
typedef struct TsmReduce {
    void (*ReduceSum)(TsmReduceInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t dim, Data_Shape shape,
                      Data_Format fmt);
    void (*ReduceAvg)(TsmReduceInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t dim, Data_Shape shape,
                      Data_Format fmt);
    void (*ReduceMax)(TsmReduceInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t dim, Data_Shape shape,
                      Data_Format fmt);
    void (*ReduceMin)(TsmReduceInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t dim, Data_Shape shape,
                      Data_Format fmt);
} TsmReduce;
```

### `TsmPool`

```c
typedef struct TsmPool {
    void (*MaxPool)(TsmPoolInstr *instr, uint64_t src0, Data_Shape src_shape, uint64_t dst, Data_Shape pad,
                    Data_Shape swr_shape, Data_Format fmt);
    void (*AvgPool)(TsmPoolInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr, Data_Shape pad,
                    Data_Shape swr_shape, Data_Format fmt);
    void (*SumPool)(TsmPoolInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr, Data_Shape pad,
                    Data_Shape swr_shape, Data_Format fmt);
    void (*MinPool)(TsmPoolInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr, Data_Shape pad,
                    Data_Shape swr_shape, Data_Format fmt);
    void (*IndexdMinPool)(TsmPoolInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_arg,
                          uint64_t dst_idx, Data_Shape pad, Data_Shape swr_shape, Data_Format fmt);
    void (*IndexdMaxPool)(TsmPoolInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_arg,
                          uint64_t dst_idx, Data_Shape pad, Data_Shape swr_shape, Data_Format fmt);
} TsmPool;
```

### `TsmUnPool`

```c
typedef struct TsmUnPool {
    void (*Unpool)(TsmUnPoolInstr *instr, uint64_t src0_addr, uint32_t index, uint64_t dst_addr, Data_Shape dst_shape, Data_Shape swr_shape, Data_Format fmt);
    void (*UnpoolAvg)(TsmUnPoolInstr *instr, uint64_t src0_addr, uint64_t dst_addr, Data_Shape dst_shape, Data_Shape swr_shape, Data_Format fmt);
    void (*UnpoolIdx)(TsmUnPoolInstr *instr, uint64_t src0_addr, uint32_t index, uint64_t dst_addr, Data_Shape dst_shape, Data_Shape swr_shape, Data_Format fmt);
} TsmUnPool;
```

### `TsmMaskDataMove`

```c
typedef struct TsmMaskDataMove {
    void (*MaskMove)(TsmMaskDataMoveInstr *instr, uint64_t src0_addr, uint32_t mask, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*MaskGather)(TsmMaskDataMoveInstr *instr, uint64_t src0_addr, uint32_t index, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*MaskGather_bV)(TsmMaskDataMoveInstr *instr, uint64_t src0_addr, uint32_t bitindex, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
} TsmMaskDataMove;
```

### `TsmConvert`

```c
typedef struct TsmConvert {
    void (*INT8_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint32_t zp, uint64_t dst_addr, uint32_t elem_count); // Data_Format fmt is INT8
    void (*INT8_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint32_t zp, uint64_t dst_addr, uint32_t elem_count);
    void (*INT8_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint32_t zp, uint64_t dst_addr, uint32_t elem_count);
    void (*INT8_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint32_t zp, uint64_t dst_addr, uint32_t elem_count);
    void (*INT16_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count); // INT16
    void (*INT16_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*INT16_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*INT16_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);

    void (*INT32_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*INT32_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*INT32_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*INT32_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);

    void (*BF16_INT8)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BF16_INT16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*BF16_INT32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*BF16_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BF16_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BF16_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);

    void (*FP16_INT8)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP16_INT16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP16_INT32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP16_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode); // rnd_mode 0~4
    void (*FP16_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*FP16_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);

    void (*FP32_INT8)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP32_INT16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP32_INT32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP32_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode); // rnd_mode 0~4
    void (*FP32_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP32_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);

    void (*TF32_INT8)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*TF32_INT16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*TF32_INT32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*TF32_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*TF32_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);// rnd_mode 0~4
    void (*TF32_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
} TsmConvert;
```

### `TsmPeripheral`

```c
typedef struct TsmPeripheral {
    void (*Count)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint32_t elem_count, Data_Format fmt);
    void (*Memset)(TsmDataMoveInstr *instr, uint64_t dst_addr, uint32_t value, uint32_t elem_count,
                  St_StrideIteration *si, Data_Format fmt); // si.stride is byte size. but ele_count is only element count
    void (*Bit2Fp)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*ArgMax)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint32_t elem_count, Data_Format fmt);
    void (*ArgMin)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint32_t elem_count, Data_Format fmt);
    void (*Bilinear)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint64_t dst0_addr, Data_Shape src_shape,
                  Data_Shape dst_shape, int32_t scale_w, int32_t scale_h, Data_Format fmt);
    void (*Lut16)(TsmPeripheralInstr *instr, uint64_t src1_addr, uint64_t dst0_addr, uint64_t lut16_addr,
                  uint32_t src_elem_count, uint32_t lut_elem_count);
    void (*Lut32)(TsmPeripheralInstr *instr, uint64_t src1_addr, uint64_t dst0_addr, uint64_t lut32_addr,
                  uint32_t src_elem_count, uint32_t lut_elem_count);
    void (*RandGen)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                  uint64_t dst1_addr, uint64_t dst2_addr, uint32_t src_elem_num, Data_Format fmt);
    void (*Factorize)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint64_t dst1_addr, uint64_t dst2_addr,
                      uint32_t src_elem_num);
    void (*ElemMask)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint32_t scale, uint64_t dst_addr, uint32_t src_elem_num, Data_Format fmt,
                     uint32_t prob, RND_MODE rnd_mode);
} TsmPeripheral;
```

### `TsmDataMove`

```c
typedef struct TsmDataMove {
    void (*Mirror)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                    Data_Shape dst_shape, Data_Format fmt);
    void (*Transpose)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*Rotate90)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                    Data_Shape dst_shape, Data_Format fmt);
    void (*Rotate180)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*Rotate270)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*Nchw2nhwc)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*Nhwc2nchw)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*Concat)(TsmMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape0, uint64_t src1_addr,
                Data_Shape src_shape1, uint64_t dst_addr, Data_Shape dst_shape, uint32_t dims, Data_Format fmt);
    void (*Pad)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                Data_Shape dst_shape, Data_Shape pad, Data_Format fmt);
    void (*Img2col)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                    Data_Shape dst_shape, uint64_t src_elem_num, uint64_t dst_elem_num, Data_Shape swr, Data_Shape pdr,
                    Data_Format fmt);
    void (*TensorNom)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*GatherScatter)(TsmDataMoveInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t size,
                          St_StrideIteration *src_si, St_StrideIteration *dst_si);
} TsmDataMove;
```

### `TsmStream`

```c
typedef struct TsmStream {
    uint32_t (*OnlineStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint32_t (*OfflineStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint32_t (*WaitStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint32_t (*ReqStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint32_t (*PushStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint32_t (*PopStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint8_t (*wait_finish)();
} TsmStream;
```

## Core Instruction and Runtime Structs

### `Data_Shape`

```c
typedef struct Data_Shape {
    uint16_t n;
    uint16_t h;
    uint16_t w;
    uint16_t c;
} Data_Shape;
```

### `St_Elem_Shape`

```c
typedef struct St_Elem_Shape {
    uint32_t elem_count;
    uint32_t unit_elem_count;
    uint32_t full_elem_count;
    uint32_t full_unit_elem_count;
} St_Elem_Shape;
```

### `St_StrideIteration`

```c
typedef struct St_StrideIteration {
    uint32_t stride0;
    uint32_t iteration0;
    uint32_t stride1;
    uint32_t iteration1;
    uint32_t stride2;
    uint32_t iteration2;
} St_StrideIteration;
```

### `InstrParamHead`

```c
typedef struct InstrParamHead {
    uint32_t type;
    uint64_t length;
} InstrParamHead;
```

### `InstrCTParam`

```c
typedef struct InstrCTParam {
    InstrParamHead head;
    TsmArithInstr ct_param;
    uint32_t group_id;
} InstrCTParam;
```

### `InstrNEParam`

```c
typedef struct InstrNEParam {
    InstrParamHead head;
    TsmNeInstr ne_param;
    uint32_t group_id;
} InstrNEParam;
```

### `InstrRDMAParam`

```c
typedef struct InstrRDMAParam {
    InstrParamHead head;
    TsmRdmaInstr rdma_param;
    uint32_t group_id;
} InstrRDMAParam;
```

### `InstrWDMAParam`

```c
typedef struct InstrWDMAParam {
    InstrParamHead head;
    TsmWdmaInstr wdma_param;
    uint32_t group_id;
} InstrWDMAParam;
```

### `InstrTDMAParam`

```c
typedef struct InstrTDMAParam {
    InstrParamHead head;
    TD_Param tdma_param;
    uint32_t group_id;
} InstrTDMAParam;
```

### `InstrInvalidInfo`

```c
typedef struct InstrInvalidInfo {
    volatile uint64_t ne_error_info;
    volatile uint64_t ct_error_info;
    volatile uint64_t td_error_info;
    volatile uint64_t rdma_error_info;
    volatile uint64_t wdma_error_info;
} InstrInvalidInfo;
```

### `Ncc_CT_GR_Ctl_Regs`

```c
typedef struct Ncc_CT_GR_Ctl_Regs {
    uint8_t cmd_valid;   // self clear
    uint8_t rnd_mode;    // 0 :round to nearest even , 1 :round to zero, 2 :round to positive infinity, 3 :round to
                         // negative infinity, 4 :stochastic round
    uint8_t src0_format; // 当CGRATensor_PeriOp_V_V_bit2fp指令，此字段用作dst_format
    uint8_t opcode;      // 详见CGRATensor指令OPcode.v
} Ncc_CT_GR_Ctl_Regs;
```

### `Ncc_CT_GR_Param_Regs`

```c
typedef struct Ncc_CT_GR_Param_Regs {
    uint32_t src0; // spm地址
    uint32_t src1;
    uint32_t dst0;
    uint32_t dst1;
    uint32_t dst2;                 // spm地址
    uint64_t src0_tfr;             // nhwc
    uint64_t dst_tfr;              // nhwc
    uint64_t pdr;                  // TOP BOTTOM,LEFT,RIGHT(分别是上下左右pad的行/列数)
    uint64_t swr;                  // kernel的 Kx(x方向的大小),Ky,Sx(x方向的步进),Sy
    uint64_t elem_count;           // vector运算的元素个数
    uint64_t unit_elem_count;      // vector运算中的短向量的元素个数(最大为64)
    uint64_t int8_scale_val0;      // 双线性插值x方向缩放系数(input_w/output_w)
    uint64_t int8_scale_val1;      // 双线性插值y方向缩放系数(input_h/output_h)
    uint64_t int8_quant;           // abandon
    uint32_t int8_bn_bias;         // abandon
    uint32_t full_elem_count;      // 若干个src_elem_num之和
    uint32_t full_unit_elem_count; // 若干个src_uint_elem_num之和
    uint64_t wb_data0;             // The pointer of Return value. [32] DATA_VALID, [31:0] data,
                                   // 函数只有一个返回值时，返回数据写在此寄存器
    uint64_t wb_data1;             // The pointer of Return value. [32] DATA_VALID, [31:0] data,
                       // 函数有两个返回值时，第二个返回数据写在此寄存器，当只有一个返回值时，此寄存器无效
    uint32_t src0_end; // spm地址(src0结束地址), xxx_end = src/dst + 对应操作数在spm中存储范围
    uint32_t src1_end;
    uint32_t dst0_end;
    uint32_t dst1_end;
    uint32_t dst2_end;
    uint8_t dims; // 000:C 001:W 010:H 011:N 100:HW 101:HWC
} Ncc_CT_GR_Param_Regs;
```

### `CT_Param`

```c
typedef struct CT_Param {
    uint32_t inter_type;
    Ncc_CT_GR_Ctl_Regs ctrl;
    Ncc_CT_GR_Param_Regs param;
} CT_Param;
```

### `Ncc_NE_GR_Ctl_Regs`

```c
typedef struct Ncc_NE_GR_Ctl_Regs {
    uint8_t sparse_en;
    uint8_t cmd_valid;
    uint8_t inpsum_format;
    uint8_t output_format;
    uint8_t input_format;
    uint8_t inpsum_en;
    uint8_t lrelu_en; // either relu or lrelu
    uint8_t relu_en;  // relu_en/lrelu_en/bias_en/scale_en 同时为0时,输出是psum
    uint8_t scale_en;
    uint8_t bias_en;
    uint8_t dilation_conv; // valid as conv backwardconv
    uint8_t type;          // 0:conv 1:depthwise conv 2:backward conv 3:gemm
} Ncc_NE_GR_Ctl_Regs;
```

### `Ncc_NE_GR_Param_Regs`

```c
typedef struct Ncc_NE_GR_Param_Regs {
    uint32_t src_a;   // spm地址(激活/左矩阵)
    uint32_t src_w;   // spm地址(权重/右矩阵)
    uint32_t psum;    // spm地址(输入psum)
    uint32_t bias;    // spm地址(bias)
    uint32_t scale_p; // spm地址(正轴scale)
    uint32_t scale_n; // spm地址(负轴scale)
    uint32_t out;     // spm地址(输出psum)
    uint64_t tfr_0;   // src0 nhwc, [15:0]tensor batch/h/w(范围1~4096);tensor通道数(范围1~16384)
    uint64_t tfr_1;   // conv: out nhwc, 同上tfr_0
    uint64_t pdr;     // pad [15:0]top bottom left right, 分别是上下左右pad的行/列数(范围0~1023)
    uint64_t unpdr;   // unpad [15:0]top bottom left right
    uint64_t swr;     // [15:0]Kx(范围1~255) Ky(范围1~255) Sx(范围1~1023) Sy(范围1~1023)
    uint64_t dilation; // [15:0]空洞卷积的x方向大小(范围1-1023),  [15:0]空洞卷积的y方向大小(范围1-1023)

    uint16_t gemm_lb;   // [15:0]左矩阵batch(范围：1~4096)
    uint16_t gemm_rb;   // [15:0]左矩阵batch(范围：1~4096)
    uint16_t gemm_n;    // 矩阵运算的矩阵大小参数
    uint16_t gemm_m;    // mk*kn---->mn
    uint16_t gemm_k;    // (范围：1~16384)
    uint8_t gemm_l_trs; // 左矩阵转置
    uint8_t gemm_r_trs; // 右矩阵转置
    /*
       Quant formula----A_int8:Left input, B_int8: Right input
            Left input 8bit to 9bit: A_int9 = A_int8 - ZP_A_int8
            Left input 8bit to 9bit: A_int9 = A_int8 - ZP_A_int8
            do conv                : O_int32 = Sum_{A_int9 * B_int9}
            do scale               : O_int16 = Clip_int16(O_int32 >> q1)
            do scale               : O_int9 = Clip_int9((O_int16 * S_int16) >> q2)
            out 9bit to 8bit       : O_int8 = O_int9 + ZP_O_int8
    */
    uint8_t quant_zp_cur;   // 输出零点(0-255).   [39:32]
    uint8_t quant_reserved; //        (0-255). [31:24] conv:unused    gemm:right_zp
    uint8_t quant_zp_pre;   // 输入零点(0-255), [23:16] conv:act_zp    gemm:left_zp(范围：0-255)
    uint8_t quant_q1;       // q1, (范围：0-31),[15:8]
    uint8_t quant_q0;       // q2, (范围：0-31),[7:0]

    uint32_t sparse_index; // spm地址(稀疏化索引)
    uint32_t srca_end;     // xxx_end = src/dst + 对应操作数在spm中存储范围
    uint32_t srcw_end;
    uint32_t psum_end;
    uint32_t bias_end;
    uint32_t scale_p_end;
    uint32_t scale_n_end;
    uint32_t out_end;
    uint32_t sparse_end;
} Ncc_NE_GR_Param_Regs;
```

### `TsmNeInstr`

```c
typedef struct TsmNeInstr {
    uint32_t inter_type;
    Ncc_NE_GR_Ctl_Regs ctrl;
    Ncc_NE_GR_Param_Regs param;
} TsmNeInstr;
```

### `Ncc_DMA_GR_Ctl_Regs`

```c
typedef struct Ncc_DMA_GR_Ctl_Regs {
    uint8_t cmd_valid;
} Ncc_DMA_GR_Ctl_Regs;
```

### `Ncc_DMA_GR_Param_Regs`

```c
typedef struct Ncc_DMA_GR_Param_Regs {
    uint64_t dst; // ddr地址
    uint64_t src; // spm地址
    /*
        for(i = 0; i < itera2; i++)
            for(j = 0; j < itera1; j++)
                for(k = 0; k < itera0; k++)
                    for(l = 0; l < elem_count; l++)
                        dst[l + elem_coun * k + elem_coun * src_itera0 * j + elem_coun * src_itera0 * src_itera1 * i] =
       \ src[l + k * src_stride0 + j * src_stride1 + i * src_stride2];
    */
    uint32_t stride0;    //地址步长
    uint32_t iteration0; // 数据块个数
    uint32_t stride1;
    uint32_t iteration1;
    uint32_t stride2;
    uint32_t iteration2;
    uint32_t elem_count; // 最里面维度单次搬运的元素个数
    uint8_t format;      // 数据类型
    uint64_t src_end;    // src_end = src + ddr中数据存储长度
    uint64_t dst_end;    // dst_end = dst + spm中数据存储长度
} Ncc_DMA_GR_Param_Regs;
```

### `DMA_Param`

```c
typedef struct DMA_Param {
    uint32_t inter_type;
    Ncc_DMA_GR_Ctl_Regs ctrl;
    Ncc_DMA_GR_Param_Regs param;
} DMA_Param;
```

### `Ncc_TDMA_GR_Ctl_Regs`

```c
typedef struct Ncc_TDMA_GR_Ctl_Regs {
    uint8_t cmd_valid;   // [12]
    uint8_t src0_format; // [11:8]
    uint8_t opcode;      //[7:0]
} Ncc_TDMA_GR_Ctl_Regs;
```

### `Ncc_TDMA_GR_Param_Regs`

```c
typedef struct Ncc_TDMA_GR_Param_Regs {
    uint32_t src0;
    uint32_t src1;
    uint32_t dst;
    uint64_t src0_tfr;   // nhwc  c:15~0
    uint64_t dst_tfr;    // nhwc
    uint64_t pdr;        // top bottom left right
    uint64_t swr;        // kx ky sx sy
    uint32_t elem_count; // vector操作的元素个数. memset、gatherscatter指令中代表byte number
    /*
        for(i=0;i<src_itera2;i++)
            for(j=0;j<src_itera1;j++)
                for(k=0;k<src_itera0;k++)
                    for(l=0;l<size;l++)
                        tmp[l+size*k+size*src_itera0*j+size*src_itera0*src_itera1*i]=src[l+k*src_stride0+j*src_stride1+i*src_stride2];

        for(i=0;i<dst_itera2;i++)
            for(j=0;j<dst_itera1;j++)
                for(k=0;k<dst_itera0;k++)
                    for(l=0;l<size;l++)
                        dst[l+k*dst_stride0+j*dst_stride1+i*dst_stride2]=tmp[l+size*k+size*dst_itera0*j+size*dst_itera0*dst_itera1*i];
    */
    uint32_t src_stride0;
    uint32_t src_iteration0;
    uint32_t src_stride1;
    uint32_t src_iteration1;
    uint32_t src_stride2;
    uint32_t src_iteration2;
    uint32_t dst_stride0;    // 地址步长
    uint32_t dst_iteration0; // 数据块个数
    uint32_t dst_stride1;
    uint32_t dst_iteration1;
    uint32_t dst_stride2;
    uint32_t dst_iteration2;
    uint32_t src0_end; // xxx_end = src/dst + 对应操作数在spm中存储范围
    uint32_t src1_end;
    uint32_t dst_end;
    uint8_t dims; //(3b) 000:C 001:W 010:H 011:N 100:HW 101:HWC
} Ncc_TDMA_GR_Param_Regs;
```

### `TD_Param`

```c
typedef struct TD_Param {
    uint32_t inter_type;
    Ncc_TDMA_GR_Ctl_Regs ctrl;
    Ncc_TDMA_GR_Param_Regs param;
} TD_Param;
```

### `Ncc_SCALAR_GR_Ctl_Regs`

```c
typedef struct Ncc_SCALAR_GR_Ctl_Regs {
    uint8_t cmd_valid;
    uint8_t format;
    uint8_t opcode;
} Ncc_SCALAR_GR_Ctl_Regs;
```

### `Ncc_SCALAR_GR_Param_Regs`

```c
typedef struct Ncc_SCALAR_GR_Param_Regs {
    uint32_t srcs; // 立即数
    uint32_t dst;  // 立即数，直接写回RF中
} Ncc_SCALAR_GR_Param_Regs;
```

### `SC_Param`

```c
typedef struct SC_Param {
    Ncc_SCALAR_GR_Ctl_Regs ctrl;
    Ncc_SCALAR_GR_Param_Regs param;
} SC_Param;
```

### `NCC_CSR`

```c
typedef struct NCC_CSR {
    uint64_t ib_status; //[7:0]IB_COUNTER: 指令buffer剩余指令数目, [8]TASK_DONE, 1：task执行结束, 0：task 正在执行,
                        //[63:9]Reserved
    uint64_t exception;      //[7:0]SCALAR_EXCEPTION, [15:8]CT_EXCEPTION, [23:16]NE_EXCEPTION, [31:24]RDMA_EXCEPTION,
                             //[39:32]WDMA_EXCEPTION, [47:40]TDMA_EXCEPTION, [63:48]Reserved
    uint64_t priority;       //[7:0]PRIORITY,当前worker的优先级, [63:8]Reserved
    uint64_t exception_mask; //[47:0]EXCEPTION_MASK, [48]EXCEPTION_UPDATE_ENABLE, [49]EXCEPTION_CLEAR, [63:49]Reserved
    uint64_t serial_mode;    //[0]SERIAL_MODE, 1：串行模式，0：并行模式, [63:1]Reserved
} NCC_CSR;
```

### `EXCEP_SERI`

```c
typedef struct EXCEP_SERI {
    uint64_t exception_mask; //[47:0]EXCEPTION_MASK, [48]EXCEPTION_UPDATE_ENABLE, [49]EXCEPTION_CLEAR, [63:49]Reserved
    uint64_t serial_mode;    //[0]SERIAL_MODE, 1：串行模式，0：并行模式, [63:1]Reserved
} EXCEP_SERI;
```

### `D_BootParamHead`

```c
typedef struct D_BootParamHead {
    uint32_t MaxLen; // BootParamHead + n * BootParamDyninfo, n = inputnum + outputnum + paramnum
    uint32_t LdmemLen;
    uint32_t InputNum;
    uint32_t OutputNum;
    uint32_t ParamNum;
    uint32_t reserved;
    uint64_t CacheMemLen;
    uint64_t CacheMemAddr; // device
    uint32_t Datalen;
    uint32_t reserved1;
    uint64_t DataAddr; // device
} D_BootParamHead;
```

### `D_BootParamDyninfo`

```c
typedef struct D_BootParamDyninfo {
    uint64_t addr; //device
    uint64_t size;
    uint32_t dtype;
    uint32_t dim;
    uint64_t shape[MAX_SHAPE_DIM];
} D_BootParamDyninfo;
```

### `D_DynTLV_Terminate`

```c
typedef struct D_DynTLV_Terminate {
    uint32_t type; //DynDataType
    uint32_t len;
    uint64_t is_final;
} D_DynTLV_Terminate;
```

### `D_KcoreCfgInfo`

```c
typedef struct D_KcoreCfgInfo {
    uint64_t snap_addr[16];
    uint64_t console_addr[16];
    uint64_t spm_dump_addr[16];
    uint64_t spm_dump_size;
    uint32_t log_level;
    uint32_t enable_monitor;
} D_KcoreCfgInfo;
```

### `D_ProfilingConfig`

```c
typedef struct D_ProfilingConfig {
    uint64_t addrs[TILE_NUM];
    uint32_t size;
    uint16_t enable;
    uint16_t prof_type;
    uint64_t barrier_addrs[TILE_NUM];
    uint32_t barrier_size;
    uint64_t instr_addrs[TILE_NUM];
    uint32_t instr_size;
    uint32_t instr_type_mask;
    uint64_t score0_instr_addrs[TILE_NUM];
    uint32_t score0_instr_size;
    uint64_t score1_instr_addrs[TILE_NUM];
    uint32_t score1_instr_size;
    uint64_t param_addrs[TILE_NUM];
    uint32_t param_size;
} D_ProfilingConfig;
```

### `GroupDataInfo`

```c
typedef struct GroupDataInfo {
    uint64_t device_addr;  // 需要dump_data的device端首地址
    uint64_t data_size;    // 需要dump_data的长度
    uint32_t addr_type;    // 需要dump_data地址类型Addr_Type,表示in out param cache
    char     data_name[MAX_GROUP_NAME_NUM]; //dump_data保存成.bin文件的名字
} GroupDataInfo;
```

### `D_GroupDataDumpCfg`

```c
typedef struct D_GroupDataDumpCfg {
    char     data_path[MAX_GROUP_NAME_NUM]; //dump_data存放的相对路径
    uint32_t data_start;    // dump_data启动状态
    uint32_t data_complete; // dump_data完成状态
    uint32_t data_number;   // dump_data的总个数
    GroupDataInfo DataDump[MAX_GROUP_NUM];
} D_GroupDataDumpCfg;
```

### `TileDteCfg`

```c
typedef struct TileDteCfg {
    uint16_t status;                // 该tile是否参与搬运工作
    uint16_t remote_tile_2d;        // 对端tile的phyid
    uint32_t local_tile_2d;         // 本端tile的phyid
    uint32_t element_count;         // 单次搬运cache_line大小，默认4k
    uint32_t stride;                // 步长
    uint32_t left_element_count;    // 搬完cache_line后，剩余的搬运的长度
    uint64_t iteration;             // 搬运cache_line的次数
    uint64_t src_addr;              // 搬运cache_line的源地址 - 物理
    uint64_t dst_addr;              // 搬运cache_line的目的地址 - 物理
    uint64_t left_src_addr;         // 搬运余数的源地址 - 物理
    uint64_t left_dst_addr;         // 搬运余数的目的地址 - 物理
} TileDteCfg;
```

### `D_DteCfgList`

```c
typedef struct D_DteCfgList {
    TileDteCfg tile_dte_cfg[16];
    uint64_t   barrier_addr;
    uint32_t   row_card_num;
    uint32_t   reserved;
} D_DteCfgList;
```

### `TileMappingTable`

```c
struct TileMappingTable {
    uint32_t         card_num;       // 整机的card数量
    struct CardInfo  card_infos[32]; //保存整机32张卡的tile映射关系
} __aligned(8);
```

### `D_DynTLV`

```c
typedef struct D_DynTLV {
    uint32_t type; //DynDataType
    uint32_t len;
} D_DynTLV;
```

### `D_Cfg_Pmu_Info`

```c
typedef struct D_Cfg_Pmu_Info {
    uint32_t tile_bitmap[16];
    uint32_t mac_use_rate;
    uint32_t chip_id;
    uint32_t cycles;
    uint64_t in_ddr;
    uint64_t param_ddr;
    uint64_t out_ddr;
    D_DataShape shape_in;
    D_DataShape shape_param;
    D_DataShape shape_out;
    uint32_t reserved;
} D_Cfg_Pmu_Info;
```

### `D_DynTLV_Cfgpmu`

```c
typedef struct D_DynTLV_Cfgpmu {
    uint32_t type; //DynDataType
    uint32_t len;
    D_Cfg_Pmu_Info cfg_pmu;
} D_DynTLV_Cfgpmu;
```

### `mode_kuiper_dte_dst_config_t`

```c
typedef struct {
	sct_dte_block_stride_iteration_s dim_cfg[3];
} kuiper_dte_shuffle_cfg_t;


typedef struct {
	uint64_t dst_addr;
	sct_dte_block_user_id_s dst_id;
	uint16_t dst_tile;
} mode_kuiper_dte_dst_config_t;
```

### `mod_kuiper_dte_node_t`

```c
typedef struct {
	sct_dte_block_stride_iteration_s dim_cfg[3];
} kuiper_dte_shuffle_cfg_t;


typedef struct {
	uint64_t dst_addr;
	sct_dte_block_user_id_s dst_id;
	uint16_t dst_tile;
} mode_kuiper_dte_dst_config_t;

typedef struct {
	uint32_t alloc_stream_id;
	mod_kuiper_dte_state_t state;
	kuiper_dte_mode_t mode;
	uint8_t dte_index;
	uint64_t src_addr;
	uint32_t data_len;
	uint8_t dst_cnt;
	mode_kuiper_dte_dst_config_t dst_cfg[32]; /*max 32 dest*/
	kuiper_dte_shuffle_cfg_t *src_dim;
	kuiper_dte_shuffle_cfg_t *dest_dim;
} mod_kuiper_dte_node_t;
```
