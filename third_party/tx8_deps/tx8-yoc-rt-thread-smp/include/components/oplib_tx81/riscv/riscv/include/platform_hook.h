#ifndef __PLATFORM_HOOK_H__
#define __PLATFORM_HOOK_H__
#include <rtthread.h>
#define TSM_FREE(size)   rt_free(size)
#define TSM_MALLOC(size) rt_malloc(size)

#ifdef PLATFORM_HOOK_DIALECT
#include "platform_hook_dialect.h"
#endif

#endif


