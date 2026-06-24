#ifndef _BARECTF_TRACE_CONFIG_H
#define _BARECTF_TRACE_CONFIG_H

/* barectf kuiper platform parameters */
#define CORES_ROWS      16
#define CORES_COLS      3
#define CORES_COUNT     (CORES_ROWS * CORES_COLS)

#define RINGBUF_TOTAL_SIZE  0x100000
#define ARM_RISCV_SH_BASE   0x180000000
#define RING_INDEX_OFFSET   (RINGBUF_TOTAL_SIZE * 63)

/* packet size (must be a power of two) */
#define PACKET_SZ       0x1000
/* ring buffer size (at least 2) */
#define RINGBUF_SZ      0x100

/* backend check timeout (cycles) */
#ifndef BACKEND_CHECK_TIMEOUT
#define BACKEND_CHECK_TIMEOUT   (10000000ULL)
#endif

/* consumer poll delay (µs) */
#ifndef CONSUMER_POLL_DELAY
#define CONSUMER_POLL_DELAY (5000)
#endif

#endif /* _BARECTF_TRACE_CONFIG_H */


