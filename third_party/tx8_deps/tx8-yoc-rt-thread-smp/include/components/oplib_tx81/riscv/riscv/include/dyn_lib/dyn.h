#ifndef TX81_DYN_H
#define TX81_DYN_H
#include <stdint.h>
#include <string.h>
#include "rtdef.h"
#include <rtthread.h>
#include "dlfcn.h"
#include "dlmodule.h"
#include "dlelf.h"
#include "tx81_spm.h"
#include "riscv.h"

#define TILE_NUM 16
#define MAX_GREAH_DEBUG_NUM 50
#define ENABLE_GRAPH_DEBUG  // 开启graph_debug
typedef struct DynModule
{
    char module_name[128];
    char module_symbol[128]; //typedef void (*entry_func_t)(void *);
    uint32_t module_size[TILE_NUM];
    uint64_t module_addr[TILE_NUM];//device_addr
} DynModule;

typedef struct DynMods
{
    uint16_t module_num;
    struct DynModule modules[0];
} DynMods; // host共用结构，传过来这个首地址, TLV的V

typedef struct DynFunc
{
    char module_name[128];
    // char* module_symbol;
    uint64_t func_addr;
    struct rt_dlmodule *module;
} DynFunc;

typedef struct DynFuncNode {
    rt_list_t list;
    uint32_t model_id;
    DynFunc func_hd;
} DynFuncNode;

typedef enum RTT_SUBTHREAD_STATS {
    MODS_IDLE = 0,
    MODS_READY,
    MODS_RUNNING,
    MODS_EXIT
} E_RTT_SUBTHREAD_STATS;

#ifdef ENABLE_GRAPH_DEBUG
typedef enum E_GRAPH_STATUS {
    GRAPH_LOAD_DOING = 0,
    GRAPH_LOAD_SUCCESS,
    GRAPH_LOAD_FALED_STEP_1,  // dlmodule_load_custom failed
    GRAPH_LOAD_FALED_STEP_2,  // dlsym failed
    GRAPH_LOAD_FALED_STEP_3,  // strcpy failed
    GRAPH_UNLOAD_SUCCESS,
    GRAPH_RUN_DOING,
    GRAPH_RUN_FAILED_STEP_1,  // entry_function is NULL
    GRAPH_RUN_SUCCESS,
} E_GRAPH_STATUS;

#pragma pack(push, 8)
typedef struct GraphInfo {
    char module_name[128];
    uint32_t graph_status;
} GraphInfo;

typedef struct GraphDebug {
    uint64_t model_num;
    GraphInfo graph_infos[MAX_GREAH_DEBUG_NUM];
} GraphDebug;
#pragma pack(pop)
#endif

rt_uint8_t* custom_load(const char* filename);
rt_err_t custom_unload(rt_uint8_t *param);
rt_err_t dlmodule_destroy_custom(struct rt_dlmodule* module);

void dynlib_init(void);
KrtRetCode dynlib_load_module(uint64_t dyn_addr, uint16_t tileid);
KrtRetCode dynlib_unload_module(char* name_str);
void dynlib_update_module(uint64_t dyn_addr, uint16_t tileid);
KrtRetCode dynlib_run_module(char* name_str, void* arg, uint32_t disable_calc);
rt_err_t dynlib_unload_all_modules(void);

#endif
