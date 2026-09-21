/* Read-only TX81 register capture. Register provenance:
 * docs/tx81-tdma-fault-localization.md. */
#define _FILE_OFFSET_BITS 64
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
  int fd;
  uint8_t *pmu[16], *stream[16], *ids[16];
} Reader;
typedef struct {
  uint64_t sequence, begin_ns, end_ns;
  uint32_t tile, fatal_before, count_before, last_low, last_high, count_after,
      fatal_after, reserved;
} Record;
typedef struct {
  char magic[8];
  uint64_t start_ns, end_ns, total, kept, trigger_sequence, trigger_ns, reason;
} Header;
_Static_assert(sizeof(Record) == 56, "record layout");
_Static_assert(sizeof(Header) == 64, "header layout");
static uint64_t now_ns(void) {
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t))
    abort();
  return (uint64_t)t.tv_sec * UINT64_C(1000000000) + t.tv_nsec;
}
static uint32_t rd(const uint8_t *page, unsigned offset) {
  return *(const volatile uint32_t *)(page + offset);
}
void fast_close(void *opaque) {
  Reader *r = opaque;
  if (!r)
    return;
  for (unsigned i = 0; i < 16; ++i) {
    if (r->pmu[i])
      munmap((void *)r->pmu[i], 4096);
    if (r->stream[i])
      munmap((void *)r->stream[i], 4096);
    if (r->ids[i])
      munmap((void *)r->ids[i], 4096);
  }
  if (r->fd >= 0)
    close(r->fd);
  free(r);
}
static int map_page(int fd, uint64_t size, uint64_t base, uint8_t **out) {
  if (base % 4096 || size < 4096 || base > size - 4096) {
    errno = EINVAL;
    return -1;
  }
  void *p = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, (off_t)base);
  if (p == MAP_FAILED)
    return -1;
  *out = p;
  return 0;
}
void *fast_open(const char *path, const uint64_t *bases, char *error) {
  Reader *r = calloc(1, sizeof(*r));
  if (!r) {
    snprintf(error, 256, "allocation failed");
    return NULL;
  }
  r->fd = open(path, O_RDONLY | O_CLOEXEC);
  struct stat st;
  if (r->fd < 0 || fstat(r->fd, &st))
    goto fail;
  for (unsigned i = 0; i < 16; ++i) {
    if (bases[i] > UINT64_MAX - 0x6a1000) {
      errno = EINVAL;
      goto fail;
    }
    if (map_page(r->fd, st.st_size, bases[i] + 0x590000, &r->pmu[i]) ||
        map_page(r->fd, st.st_size, bases[i] + 0x610000, &r->stream[i]) ||
        map_page(r->fd, st.st_size, bases[i] + 0x6a0000, &r->ids[i]))
      goto fail;
    uint32_t xy = ((i / 4) << 8) | (i % 4);
    if (rd(r->ids[i], 0x58) != xy || rd(r->ids[i], 0x5c) != xy) {
      snprintf(error, 256, "Tile %u packed XY identity mismatch", i);
      fast_close(r);
      return NULL;
    }
  }
  return r;
fail:
  snprintf(error, 256, "%s", strerror(errno));
  fast_close(r);
  return NULL;
}
static Record sample(Reader *r, unsigned tile, uint64_t sequence) {
  Record x = {0};
  x.sequence = sequence;
  x.tile = tile;
  x.begin_ns = now_ns();
  x.fatal_before = rd(r->stream[tile], 0xc0);
  x.count_before = rd(r->pmu[tile], 0x20);
  x.last_low = rd(r->pmu[tile], 0x27c);
  x.last_high = rd(r->pmu[tile], 0x280);
  x.count_after = rd(r->pmu[tile], 0x20);
  x.fatal_after = rd(r->stream[tile], 0xc0);
  x.end_ns = now_ns();
  return x;
}
int fast_capture(void *opaque, const char *output, const char *ready,
                 const char *stop, unsigned capacity, unsigned maximum_ms,
                 unsigned post_ms, char *error) {
  Reader *r = opaque;
  if (!r || capacity < 32 || capacity > 1048576 || maximum_ms < 1 ||
      maximum_ms > 30000 || post_ms > 500 || !ready || !stop) {
    snprintf(error, 256, "invalid bounded capture configuration");
    return -1;
  }
  Record *ring = calloc(capacity, sizeof(*ring));
  if (!ring) {
    snprintf(error, 256, "ring allocation failed");
    return -1;
  }
  for (unsigned tile = 0; tile < 16; ++tile) {
    Record x = sample(r, tile, tile);
    /* Counts and last-command values persist across healthy invocations. */
    if ((x.fatal_before | x.fatal_after) & 0x1ff03) {
      snprintf(error, 256, "Tile %u baseline is not clean; launch prohibited",
               tile);
      free(ring);
      return -2;
    }
  }
  int fd = open(output, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (fd < 0) {
    snprintf(error, 256, "%s", strerror(errno));
    free(ring);
    return -1;
  }
  FILE *f = fdopen(fd, "wb");
  if (!f) {
    snprintf(error, 256, "%s", strerror(errno));
    close(fd);
    free(ring);
    return -1;
  }
  Header h = {.magic = {'T', 'D', 'M', 'A', 'R', 'I', 'N', 'G'},
              .trigger_sequence = UINT64_MAX};
  h.start_ns = now_ns();
  int ready_fd = open(ready, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (ready_fd < 0) {
    snprintf(error, 256, "ready file: %s", strerror(errno));
    fclose(f);
    free(ring);
    return -1;
  }
  const char message[] = "baseline-clean\n";
  int ready_failed =
      write(ready_fd, message, sizeof(message) - 1) != sizeof(message) - 1;
  if (close(ready_fd))
    ready_failed = 1;
  if (ready_failed) {
    snprintf(error, 256, "ready file write failed");
    fclose(f);
    free(ring);
    return -1;
  }
  for (;;) {
    Record x = sample(r, h.total % 16, h.total);
    ring[h.total % capacity] = x;
    ++h.total;
    if (x.fatal_before == UINT32_MAX || x.fatal_after == UINT32_MAX ||
        (x.count_before == UINT32_MAX && x.count_after == UINT32_MAX &&
         x.last_low == UINT32_MAX && x.last_high == UINT32_MAX)) {
      h.reason = 4;
      break;
    }
    if (((x.fatal_before | x.fatal_after) & 0x1000) &&
        h.trigger_sequence == UINT64_MAX) {
      h.trigger_sequence = x.sequence;
      h.trigger_ns = x.end_ns;
    }
    if (h.trigger_sequence != UINT64_MAX &&
        h.total - h.trigger_sequence >= capacity / 2) {
      h.reason = 5;
      break;
    }
    if (h.trigger_sequence != UINT64_MAX &&
        x.end_ns - h.trigger_ns >= (uint64_t)post_ms * 1000000) {
      h.reason = 3;
      break;
    }
    if (x.end_ns - h.start_ns >= (uint64_t)maximum_ms * 1000000) {
      h.reason = 2;
      break;
    }
    if (!(h.total % 256) && !access(stop, F_OK)) {
      h.reason = 1;
      break;
    }
  }
  h.end_ns = now_ns();
  h.kept = h.total < capacity ? h.total : capacity;
  int failed = fwrite(&h, sizeof(h), 1, f) != 1;
  for (uint64_t seq = h.total - h.kept; seq < h.total && !failed; ++seq)
    failed = fwrite(&ring[seq % capacity], sizeof(Record), 1, f) != 1;
  if (fflush(f))
    failed = 1;
  if (fsync(fd))
    failed = 1;
  if (fclose(f))
    failed = 1;
  free(ring);
  if (failed) {
    snprintf(error, 256, "capture output write failed");
    return -1;
  }
  if (h.reason == 4) {
    snprintf(error, 256, "all-ones aperture; stopped");
    return -3;
  }
  return 0;
}
