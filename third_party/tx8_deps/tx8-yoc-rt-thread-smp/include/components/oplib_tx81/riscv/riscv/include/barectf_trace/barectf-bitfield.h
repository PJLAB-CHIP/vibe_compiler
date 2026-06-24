#ifndef _BARECTF_BITFIELD_H
#define _BARECTF_BITFIELD_H

#include <stdint.h>
#include <limits.h>

#ifdef __cplusplus
# define _CAST_PTR(_type, _value) \
    ((_type)((void *)(_value)))
#else
# define _CAST_PTR(_type, _value)   ((void *) (_value))
#endif

/* We can't shift a int from 32 bit, >> 32 and << 32 on int is undefined */
#define _bt_piecewise_rshift(_vtype, _v, _shift) \
do {                                    \
    unsigned long ___shift = (_shift);              \
    unsigned long sb = (___shift) / (sizeof(_v) * CHAR_BIT - 1);    \
    unsigned long final = (___shift) % (sizeof(_v) * CHAR_BIT - 1); \
                                    \
    for (; sb; sb--)                        \
        _v >>= sizeof(_v) * CHAR_BIT - 1;           \
    _v >>= final;                           \
} while (0)

#define _bt_bitfield_write_le(_ptr, type, _start, _length, _vtype, _v)  \
do {                                    \
    _vtype __v = (_v);                      \
    type *__ptr = _CAST_PTR(type *, _ptr);              \
    unsigned long __start = (_start), __length = (_length);     \
    type mask, cmask;                       \
    unsigned long ts = sizeof(type) * CHAR_BIT; /* type size */ \
    unsigned long start_unit, end_unit, this_unit;          \
    unsigned long end, cshift; /* cshift is "complement shift" */   \
                                    \
    if (!__length)                          \
        break;                          \
                                    \
    end = __start + __length;                   \
    start_unit = __start / ts;                  \
    end_unit = (end + (ts - 1)) / ts;               \
                                    \
    /* Trim v high bits */                      \
    if (__length < sizeof(__v) * CHAR_BIT)              \
        __v &= ~((~(_vtype) 0) << __length);            \
                                    \
    /* We can now append v with a simple "or", shift it piece-wise */ \
    this_unit = start_unit;                     \
    if (start_unit == end_unit - 1) {               \
        mask = ~((~(type) 0) << (__start % ts));        \
        if (end % ts)                       \
            mask |= (~(type) 0) << (end % ts);      \
        cmask = (type) __v << (__start % ts);           \
        cmask &= ~mask;                     \
        __ptr[this_unit] &= mask;               \
        __ptr[this_unit] |= cmask;              \
        break;                          \
    }                               \
    if (__start % ts) {                     \
        cshift = __start % ts;                  \
        mask = ~((~(type) 0) << cshift);            \
        cmask = (type) __v << cshift;               \
        cmask &= ~mask;                     \
        __ptr[this_unit] &= mask;               \
        __ptr[this_unit] |= cmask;              \
        _bt_piecewise_rshift(_vtype, __v, ts - cshift);     \
        __start += ts - cshift;                 \
        this_unit++;                        \
    }                               \
    for (; this_unit < end_unit - 1; this_unit++) {         \
        __ptr[this_unit] = (type) __v;              \
        _bt_piecewise_rshift(_vtype, __v, ts);          \
        __start += ts;                      \
    }                               \
    if (end % ts) {                         \
        mask = (~(type) 0) << (end % ts);           \
        cmask = (type) __v;                 \
        cmask &= ~mask;                     \
        __ptr[this_unit] &= mask;               \
        __ptr[this_unit] |= cmask;              \
    } else                              \
        __ptr[this_unit] = (type) __v;              \
} while (0)

#define bt_bitfield_write_le(ptr, _start, _length, _vtype, _v) \
    _bt_bitfield_write_le(ptr, uint8_t, _start, _length, _vtype, _v)

#endif /* _BARECTF_BITFIELD_H */
