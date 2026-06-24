#ifndef _KCORE_BARECTF_TRACE_H
#define _KCORE_BARECTF_TRACE_H

#include "barectf_trace_config.h"
#include "barectf.h"

struct ringbuf {
    uint8_t packets[RINGBUF_SZ][PACKET_SZ];
};

struct ring_index {
    uint32_t producer_index;
    uint32_t consumer_index;
};

typedef struct {
    /* ringbuf col index  */
    uint16_t col_index;
} mod_trace_config_t;

/**
 * Returns the barectf context to be used with tracing functions.
 */
struct barectf_default_ctx *tracing_get_barectf_ctx(void);

#define log_trace_str barectf_default_trace_str
#define log_trace_one_int barectf_default_trace_str_one_int
#define log_trace_two_int barectf_default_trace_str_two_int
#define log_trace_three_int barectf_default_trace_str_three_int

/**
 * Initializes the platform.
 */
int tracing_init(void);

/**
 * Finalizes the platform.
 */
void tracing_finish(void);

#endif /* _KCORE_BARECTF_TRACE_H */

