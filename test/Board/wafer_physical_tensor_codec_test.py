#!/usr/bin/env python3
"""Check the independent board-calibration physical tensor codec."""

from __future__ import annotations

import struct

import wafer_physical_tensor_codec as codec


def main() -> int:
    cx = codec.physical_layout((3, 1000), "Cx", 2)
    assert cx.c_block == 64
    assert cx.full_blocks == 16
    assert cx.tail_width == 0
    assert cx.aligned_c == 1024
    assert cx.physical_bytes == 6144
    assert codec.physical_element_offset(cx, (1, 64)) * 2 == 512

    ncx = codec.physical_layout((2, 3, 1000), "NCx", 2)
    assert ncx.full_blocks == 16
    assert ncx.batch_elements == 3072
    assert ncx.physical_bytes == 12288
    assert codec.physical_element_offset(ncx, (1, 2, 64)) * 2 == 6784

    cx_tail = codec.physical_layout((3, 129), "Cx", 2)
    ncx_tail = codec.physical_layout((1, 3, 129), "NCx", 2)
    assert cx_tail.full_blocks == 2
    assert cx_tail.tail_width == 4
    assert cx_tail.physical_bytes == ncx_tail.physical_bytes
    for row in range(3):
        for channel in range(129):
            assert codec.physical_element_offset(
                cx_tail, (row, channel)
            ) == codec.physical_element_offset(
                ncx_tail, (0, row, channel)
            )

    shape = (2, 3, 65)
    values = tuple(
        struct.pack("<H", index) for index in range(2 * 3 * 65)
    )
    storage = codec.pack_scalar_bytes(
        shape, "NCx", 2, values, padding=0xA7
    )
    assert codec.unpack_scalar_bytes(shape, "NCx", 2, storage) == values
    ncx_layout = codec.physical_layout(shape, "NCx", 2)
    logical_offsets = {
        codec.physical_element_offset(ncx_layout, coordinate)
        for coordinate in codec.coordinates(shape)
    }
    for element in range(len(storage) // 2):
        if element not in logical_offsets:
            assert storage[element * 2 : element * 2 + 2] == b"\xa7\xa7"

    compact = codec.pack_scalar_bytes(
        (2, 3, 4), "Tensor", 2, values[:24], padding=0xA7
    )
    assert compact == b"".join(values[:24])
    print(
        "wafer_physical_tensor_codec_test: "
        f"cx_bytes={cx.physical_bytes} ncx_bytes={ncx.physical_bytes} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
