#ifndef __PLATFORM_FREETOS__
#define __PLATFORM_FREETOS__

#include "lib_log.h"
#include "csi_kernel.h"
#include "stream/stream_rt.h"
#include "mailbox/mod_mailbox.h"
#include "pmu/pmu.h"
#define MAX_SHAPE_DIM      6 //n, h, w, c, x, x

#define TILE_NUM                    16
#define PG_TILE_NUM                 8
/* PG类型魔术字 */
#define PRODUCT_TYPE_PG_VAL         0xabcdefab   
// 全局变量，表示保存所有pg卡的map表的tile的最大值，16张卡，一卡双芯，共256个
#define MAX_PG_TILE_NUM  (16*16)
//Kcore runtime 错误码定义
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

//启动参数
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

typedef struct D_BootParamDyninfo {
    uint64_t addr; //device
    uint64_t size;
    uint32_t dtype;
    uint32_t dim;
    uint64_t shape[MAX_SHAPE_DIM];
} D_BootParamDyninfo;

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

typedef struct D_DynTLV_Terminate {
    uint32_t type; //DynDataType
    uint32_t len;
    uint64_t is_final;
} D_DynTLV_Terminate;

typedef struct D_ExportSpmAddrInfo {
    uint64_t addrs[16];
    uint64_t size;
} D_ExportSpmAddrInfo;

typedef struct D_KcoreCfgInfo {
    uint64_t snap_addr[16];
    uint64_t console_addr[16];
    uint64_t spm_dump_addr[16];
    uint64_t spm_dump_size;
    uint32_t log_level;
    uint32_t enable_monitor;
} D_KcoreCfgInfo;

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

#define MAX_GROUP_NUM 3
#define MAX_GROUP_NAME_NUM 128
enum Addr_Type { ADDR_TYPE_IN = 0, ADDR_TYPE_OUT, ADDR_TYPE_PARAM, ADDR_TYPE_CACHE, ADDR_TYPE_NONE };

typedef struct GroupDataInfo {
    uint64_t device_addr;  // 需要dump_data的device端首地址
    uint64_t data_size;    // 需要dump_data的长度
    uint32_t addr_type;    // 需要dump_data地址类型Addr_Type,表示in out param cache
    char     data_name[MAX_GROUP_NAME_NUM]; //dump_data保存成.bin文件的名字
} GroupDataInfo;

typedef struct D_GroupDataDumpCfg {
    char     data_path[MAX_GROUP_NAME_NUM]; //dump_data存放的相对路径
    uint32_t data_start;    // dump_data启动状态
    uint32_t data_complete; // dump_data完成状态
    uint32_t data_number;   // dump_data的总个数
    GroupDataInfo DataDump[MAX_GROUP_NUM];
} D_GroupDataDumpCfg;

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
typedef struct D_DteCfgList {
    TileDteCfg tile_dte_cfg[16];
    uint64_t   barrier_addr;
    uint32_t   row_card_num;
    uint32_t   reserved;
} D_DteCfgList;

/*
    PG卡用的整机的tile id映射表。FG卡映射关系固定，暂时不需要下面的映射表
*/
/* ============================  映射表 BEGIN ====================== */
struct TileInfo {
    uint32_t tile_id;
    uint32_t phy_tilex;
    uint32_t phy_tiley;
} __aligned(8);

struct CardInfo {
    uint32_t chip_id;
    uint32_t tile_num;
    uint16_t good_bitmap;
    uint16_t pad;
    struct TileInfo tile_info[TILE_NUM];
} __aligned(8);

struct TileMappingTable {
    uint32_t         card_num;       // 整机的card数量
    struct CardInfo  card_infos[32]; //保存整机32张卡的tile映射关系
} __aligned(8);
/* =============================  映射表 END ============================= */

typedef struct D_DynTLV {
    uint32_t type; //DynDataType
    uint32_t len;
} D_DynTLV;

typedef struct D_DataShape {
    uint16_t n;
    uint16_t h;
    uint16_t w;
    uint16_t c;
} D_DataShape;

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

typedef struct D_DynTLV_Cfgpmu {
    uint32_t type; //DynDataType
    uint32_t len;
    D_Cfg_Pmu_Info cfg_pmu;
} D_DynTLV_Cfgpmu;

#pragma pack(4)
enum CONSOLE_TYPE {
    REG_READ,
    REG_WRITE,
    BARRIER,
    BOOT_PARAM,
    BACK_TRACE,
    EXPORT_SPM,
    DEBUG_MSG,
    CONTINUE_DUMP,
    EXPORT_GROUP,
    INVALID
};
/*
spm导出状态机：
    初始状态
    文件落盘完成                        --  该状态由Host-App_report设置，kcore检查到这个状态后，发起下一个spm导出请求
    spm已写入ddr管道                    --  断点完成，kcore设置这个状态，host检查到这个状态后，发起D2H，进行文件落盘
    全部处理完成                        --  全部处理完成后，kcore告知host-app_report线程解除阻塞，同时monitor-task-export_spm也解除阻塞
*/
enum DUMP_SPM_STATUS {
    INIT,
    HOST_SAVED,
    WRITE_DONE,
    ALL_DONE
};
#define MAX_FUNC_NAME_SIZE 64
typedef struct SpmInfo {
    uint32_t    status;
    char        func_name[MAX_FUNC_NAME_SIZE];
    uint32_t    func_line;
} SpmInfo;

#define MAX_STATUS_BUFF_SIZE 4
#define MAX_REQ_BUFF_SIZE    252
#define MAX_RSP_BUFF_SIZE    768
#define GDB_SPM_TOTAL_LEN    (MAX_STATUS_BUFF_SIZE + MAX_REQ_BUFF_SIZE + MAX_RSP_BUFF_SIZE)
typedef struct DebugMsgInfo {
    uint8_t   status;   // 0-请求  1-应答
    uint8_t   rsv[3];
    uint8_t   buff[MAX_REQ_BUFF_SIZE];
    uint8_t   result[MAX_RSP_BUFF_SIZE];
} DebugMsgInfo;

typedef struct GroupInfo {
    uint32_t status;
} GroupInfo;
typedef struct D_TileConsoleInfo { // 用于响应console输入的命令行，该数据结构要与host侧保持同步
    uint32_t version;
    uint32_t magic_num;
    uint32_t type;  // CONSOLE_TYPE
    uint64_t addr;
    uint32_t value;
    SpmInfo  spm_info;
    DebugMsgInfo debug_msg;
    GroupInfo group_info;
} D_TileConsoleInfo;

#define MAX_BOOT_PARAM_SIZE (1024 * 1024)
#define MAX_SNAP_LOG_SIZE (1024 * 1024)
typedef struct D_TileSnapInfo {  // 用于收集device运行过程中的信息，该数据结构要与host侧保持同步
    uint32_t magic_num;
    char     boot_param[MAX_BOOT_PARAM_SIZE];
    char     log_buff[MAX_SNAP_LOG_SIZE];
    uint32_t log_cursor;
} D_TileSnapInfo;

typedef void (*MonitorRespFunc)(void);
typedef struct MonitorRespFuncMap {
    uint32_t type;   // CONSOLE_TYPE
    MonitorRespFunc func;
} MonitorRespFuncMap;

typedef KrtRetCode (*ProcessTlvFunc)(D_BootParamHead* head);
typedef struct ProcessTlvFuncMap {
    uint32_t type;   // TLV类型 D_DynDataType
    ProcessTlvFunc func;
} ProcessTlvFuncMap;

int dte_memcpy_d2d(TileDteCfg tile_dte_cfg, uint16_t tile_id);
int dte_memcpy_p2p(TileDteCfg tile_dte_cfg);
void dte_memcpy(uint64_t src_ddr_ptr, uint64_t dst_ddr_ptr, uint32_t data_len);

k_sem_handle_t *__SEM__GROUP_HOST_SAVED();
k_sem_handle_t *__SEM__GROUP_WRITE_DONE();

typedef struct Global_Kcore_Config {
    uint64_t snap_addr;                 // kcore快照信息地址
    uint64_t console_addr;              // kcore响应host请求的地址
    uint64_t profiling_addr;            // profiling写入的地址
    uint32_t profiling_size;            // profiling写入的地址大小
    uint32_t profiling_offset;          // profiling写指针位置
    uint64_t spm_dump_addr;             // 写SPM信息的DDR地址
    uint64_t spm_dump_size;             // 写SPM信息的DDR长度
    uint32_t log_level;                 // kcore日志级别
    uint32_t enable_monitor;            // 开启monitor-task
    uint32_t disable_calc;              // 关闭计算
    uint64_t groupdata_dump_addr;       // groupdata的dump地址
    uint64_t barrier_addr;              // 写barrier性能信息的DDR地址
    uint64_t barrier_size;              // 写barrier性能信息的DDR长度
    uint64_t instr_addr;                // 写instr性能信息的DDR地址
    uint32_t instr_size;                // 写instr性能信息的DDR长度
    uint32_t instr_type_mask;           // 需要探测指令类型的掩码
    uint64_t score0_instr_addr;          // score0 instr性能信息的DDR地址
    uint32_t score0_instr_size;          // score0 instr性能信息的DDR长度
    uint64_t score1_instr_addr;          // score1 instr性能信息的DDR地址
    uint32_t score1_instr_size;          // score1 instr性能信息的DDR长度
    uint64_t param_addr;                 // 写instr参数信息的DDR地址
    uint32_t param_size;                 // 写instr参数信息的DDR长度
} Global_Kcore_Config;
Global_Kcore_Config* get_kcore_config();

// SPM数据动态导出功能相关 Begin
k_sem_handle_t* __SEM__SPM_WRITE_REQ();
k_sem_handle_t* __SEM__SPM_WRITE_DONE();
k_sem_handle_t* __SEM__SPM_HOST_SAVED();
char* __BREAK_POINT_FUNC_NAME__();
uint32_t __BREAK_POINT_FUNC_LINE__();
void set_break_point_info(const char *func_name, uint32_t func_line);
#define __BREAK_POINT_DUMP_SPM__ {    \
    Global_Kcore_Config* kcore_config = get_kcore_config(); \
    if (kcore_config->spm_dump_addr != 0) {    \
        csi_kernel_sem_wait((k_sem_handle_t)(*__SEM__SPM_WRITE_REQ()), -1); \
        TsmWdmaInstr wdma_param = {I_WDMA, {0, }, {0, }};   \
        TsmWdma* wdma = (TsmWdma *)getTsmOpPointer()->wdma_pointer; \
        uint64_t spm_addr = 0;  \
        wdma->Wdma1d(&wdma_param, spm_addr, kcore_config->spm_dump_addr, kcore_config->spm_dump_size, Fmt_INT8);  \
        TsmExecute(&wdma_param);    \
        __LOG__(KCORE_LOG_DEBUG, "spm info has been writeen to 0x%lx, size is %llu\n", kcore_config->spm_dump_addr, kcore_config->spm_dump_size); \
        set_break_point_info(__func__, __LINE__); \
        csi_kernel_sem_post(*__SEM__SPM_WRITE_DONE());  \
        csi_kernel_sem_wait((k_sem_handle_t)(*__SEM__SPM_HOST_SAVED()), -1); \
    }   \
}
// SPM数据动态导出功能相关 End

// TsmRun执行结果 Begin
enum KCORE_RESULT_CODE {
    KCORE_SUCCESS = 0x0,
    KCORE_FAILED = 0x1,
    KCORE_TIMEOUT = 0x2
};
enum TILE_EXCEPTION_TYPE {
    TILE_SUCCESS = 0x0,
    TILE_ERROR_FUNC = 0x1,
    TILE_ERROR_MSG = 0x2
};
#define KCORE_RUN_MAGIC_NUM 0xFBEA
#define KCORE_FUNC_NAME_MAX 23
typedef struct TsmRunResult {
    uint16_t magic_id;
    uint8_t ret_code;
    uint16_t tile_bit_map;
    uint8_t reserved[3];
} TsmRunResult;
typedef struct KcoreErrInfo {
    uint32_t ret_code : 8;
    uint32_t barrier_cnt : 24;
    uint32_t func_line;
    char func_name[KCORE_FUNC_NAME_MAX + 1];
} KcoreErrInfo;
void init_kcore_result();

// TsmRun执行结果 End
void set_kcore_err_info(uint16_t ret_code, const char *buff, uint32_t line_number);
TsmRunResult get_run_result();
#define __KCORE_ERROR__ {  \
    set_kcore_err_info(TILE_ERROR_FUNC, __func__, __LINE__);  \
}

// torus模式：TORUS_EW代表东西向、TORUS_NS代表南北向、TORUS_EW_NS代表东西南北向同时开启
#define TORUS_EW 0
#define TORUS_NS 1
#define TORUS_EW_NS 2


/**
 * @brief 启用Torus网络,torus只支持右边和下边边缘的tile启用
 *
 * 该函数根据传入的参数torus_mode来设置Torus网络的启用状态。
 *
 * @param torus_mode 要启用的Torus网络模式，可以是TORUS_EW、TORUS_NS或TORUS_EW_NS。
 * @return int 函数执行成功返回0，否则返回-1。
 */
int enable_torus(uint32_t torus_mode);
/**
 * @brief 禁用Torus网络, torus只支持右边和下边边缘的tile禁用
 *
 * 该函数根据传入的参数`torus_mode`来设置Torus网络的禁用状态。
 *
 * @param torus_mode 要禁用的Torus网络模式，可以是TORUS_EW、TORUS_NS或TORUS_EW_NS。
 * @return int 函数执行成功返回0，否则返回-1。
 */
int disable_torus(uint32_t torus_mode);
/**
 * @brief 检查Torus网络状态
 *
 * 该函数检查Torus网络的状态，并根据检查结果记录相应的日志信息。
 *
 * @return true 如果Torus网络已打开，返回true
 * @return false 如果Torus网络未打开，返回false
 */
bool check_torus_status();


// Kcore开关相关 Begin
enum KCORE_SWITCH_TYPE {
    TYPE_DISABLE_CALC = 0x1,
    TYPE_DISABLE_ALLREDUCE = 0x2,
    TYPE_DISABLE_ALLGATHER = 0x4
};
void init_kcore_switch_info(void *args);

// Kcore开关相关 End
#define DEBUG_MSG_REQ 0xab
#define DEBUG_MSG_RSP 0xcd
void __DEBUG_FUNC__();


int init_tile_id(uint32_t tile_id, uint32_t row_length); // 通用的初始化函数
bool getTsmTlvInfo(D_BootParamHead *head, uint32_t type, void* data);
void hrt_barrier();
uint32_t get_tile_id_in_chip();
uint32_t get_tile_id_2d(uint16_t g_tile_id_1d, uint16_t *tile_x, uint16_t *tile_y);
/*=================================triton=================================*/
uint32_t __get_pid(uint32_t dim);
uint32_t __grid_dim_x();
uint32_t __grid_dim_y();
uint32_t __grid_dim_z();
void init_atomic_barrier_task();
void atomic_barrier_in();
void atomic_barrier_out();
#define TsmGetPid __get_pid
typedef struct MemoryInfo {
    uint64_t total_system;    // 系统总内存
    uint64_t used_system;     // 当前系统使用内存
    uint64_t max_used_system; // 系统历史最大使用内存
} MemoryInfo;
void print_mem_info();
#pragma pack()

#ifdef __cplusplus
    extern "C" {
#endif

#ifdef __cplusplus
}
#endif


#endif
