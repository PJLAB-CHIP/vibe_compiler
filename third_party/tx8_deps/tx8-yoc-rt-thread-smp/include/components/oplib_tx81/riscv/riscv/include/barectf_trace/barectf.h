#ifndef _BARECTF_H
#define _BARECTF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct barectf_ctx;

uint32_t barectf_packet_size(const void *vctx);
int barectf_packet_is_full(const void *vctx);
int barectf_packet_is_empty(const void *vctx);
uint32_t barectf_packet_events_discarded(const void *vctx);
uint32_t barectf_discarded_event_records_count(const void * const vctx);
uint32_t barectf_packet_sequence_number(const void * const vctx);
uint8_t *barectf_packet_buf(const void *vctx);
uint8_t *barectf_packet_buf_addr(const void * const vctx);
void barectf_packet_set_buf(void *vctx, uint8_t *buf, uint32_t buf_size);
uint32_t barectf_packet_buf_size(const void *vctx);
int barectf_packet_is_open(const void *vctx);
int barectf_is_in_tracing_section(const void *vctx);
volatile const int *barectf_is_in_tracing_section_ptr(const void *vctx);
int barectf_is_tracing_enabled(const void *vctx);
void barectf_enable_tracing(void *vctx, int enable);

/* barectf platform callbacks */
struct barectf_platform_callbacks {
    /* Clock source callbacks */
    uint64_t (*default_clock_get_value)(void *);

    /* Is the back end full? */
    int (*is_backend_full)(void *);

    /* Open packet */
    void (*open_packet)(void *);

    /* Close packet */
    void (*close_packet)(void *);
};

/* Common barectf context */
struct barectf_ctx {
    /* Platform callbacks */
    struct barectf_platform_callbacks cbs;

    /* Platform data (passed to callbacks) */
    void *data;

    /* Output buffer (will contain a CTF binary packet) */
    uint8_t *buf;

    /* Packet's total size (bits) */
    uint32_t packet_size;

    /* Packet's content size (bits) */
    uint32_t content_size;

    /* Current position from beginning of packet (bits) */
    uint32_t at;

    /* Size of packet header + context fields (content offset) */
    uint32_t off_content;

    /* Discarded event records counter snapshot */
    uint32_t events_discarded;

    /* Packet's sequence number */
    uint32_t sequence_number;

    /* Current packet is open? */
    int packet_is_open;

    /* In tracing code? */
    volatile int in_tracing_section;

    /* Tracing is enabled? */
    volatile int is_tracing_enabled;

    /* Use current/last event record timestamp when opening/closing packets */
    int use_cur_last_event_ts;
};

/* Context for data stream type `default` */
struct barectf_default_ctx {
    /* Parent */
    struct barectf_ctx parent;

    /* Config-specific members follow */
    uint32_t off_ph_magic;
    uint32_t off_ph_stream_id;
    uint32_t off_pc_packet_size;
    uint32_t off_pc_content_size;
    uint32_t off_pc_timestamp_begin;
    uint32_t off_pc_timestamp_end;
    uint32_t off_pc_events_discarded;
    uint64_t cur_last_event_ts;
};

/* Initialize context */
void barectf_init(void *vctx,
    uint8_t *buf, uint32_t buf_size,
    const struct barectf_platform_callbacks cbs, void *data);

/* Open packet for data stream type `default` */
void barectf_default_open_packet(
    struct barectf_default_ctx *sctx);

/* Close packet for data stream type `default` */
void barectf_default_close_packet(struct barectf_default_ctx *sctx);

/* Trace (data stream type `default`, event record type `str`) */
void barectf_default_trace_str(const char *p_the_string);

/* Trace (data stream type `default`, event record type `str_one_int`) */
void barectf_default_trace_str_one_int(const char *p_str, int32_t p_a);

/* Trace (data stream type `default`, event record type `str_three_int`) */
void barectf_default_trace_str_three_int(const char *p_str, int32_t p_a, int32_t p_b, int32_t p_c);

/* Trace (data stream type `default`, event record type `str_two_int`) */
void barectf_default_trace_str_two_int(const char *p_str, int32_t p_a, int32_t p_b);

#ifdef __cplusplus
}
#endif

#endif /* _BARECTF_H */
