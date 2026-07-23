#!/usr/bin/env python3
"""Independent host codec for board-calibration Tensor/Cx/NCx payloads."""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Iterable, Iterator, Sequence


LAYOUTS = {"Tensor", "NTensor", "Cx", "NCx"}


@dataclasses.dataclass(frozen=True)
class PhysicalLayout:
    shape: tuple[int, ...]
    layout: str
    element_bytes: int
    c_block: int
    full_blocks: int
    tail_width: int
    aligned_c: int
    outer_elements: int
    hw_elements: int
    batch_elements: int
    physical_elements: int

    @property
    def physical_bytes(self) -> int:
        return self.physical_elements * self.element_bytes


def _product(values: Sequence[int]) -> int:
    return math.prod(values, start=1)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment if value else 0


def _tail_width(remainder: int) -> int:
    for width in (4, 8, 16, 32, 64):
        if remainder <= width:
            return width
    raise ValueError(f"unsupported channel remainder {remainder}")


def physical_layout(
    shape: Sequence[int],
    layout: str,
    element_bytes: int,
) -> PhysicalLayout:
    logical_shape = tuple(shape)
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout}")
    if element_bytes not in (1, 2, 4, 8):
        raise ValueError(f"invalid scalar width {element_bytes}")
    if any(dimension <= 0 for dimension in logical_shape):
        raise ValueError("board calibration requires positive static shapes")
    if layout in ("Cx", "NCx") and not logical_shape:
        raise ValueError(f"{layout} requires a channel dimension")

    if layout in ("Tensor", "NTensor"):
        elements = _product(logical_shape)
        return PhysicalLayout(
            logical_shape,
            layout,
            element_bytes,
            0,
            0,
            0,
            0,
            elements,
            elements,
            elements,
            elements,
        )

    logical_c = logical_shape[-1]
    c_block = 128 if element_bytes == 1 else 64
    retain_threshold = 64 if element_bytes == 1 else 32
    full_blocks, remainder = divmod(logical_c, c_block)
    if remainder == 0:
        tail_width = 0
    elif remainder <= retain_threshold:
        tail_width = _tail_width(remainder)
    else:
        full_blocks += 1
        tail_width = 0
    aligned_c = full_blocks * c_block + tail_width
    bank_elements = 256 // element_bytes

    if layout == "Cx":
        outer_elements = _product(logical_shape[:-1])
        batch_elements = _align_up(
            outer_elements * aligned_c, bank_elements
        )
        return PhysicalLayout(
            logical_shape,
            layout,
            element_bytes,
            c_block,
            full_blocks,
            tail_width,
            aligned_c,
            outer_elements,
            outer_elements,
            batch_elements,
            batch_elements,
        )

    if len(logical_shape) < 2:
        raise ValueError("NCx requires leading N and trailing C dimensions")
    hw_elements = _product(logical_shape[1:-1])
    outer_elements = logical_shape[0] * hw_elements
    batch_elements = _align_up(
        hw_elements * aligned_c, bank_elements
    )
    return PhysicalLayout(
        logical_shape,
        layout,
        element_bytes,
        c_block,
        full_blocks,
        tail_width,
        aligned_c,
        outer_elements,
        hw_elements,
        batch_elements,
        logical_shape[0] * batch_elements,
    )


def _row_major_index(
    shape: Sequence[int], indices: Sequence[int]
) -> int:
    linear = 0
    for dimension, index in zip(shape, indices, strict=True):
        if index < 0 or index >= dimension:
            raise ValueError(f"index {tuple(indices)} is outside {tuple(shape)}")
        linear = linear * dimension + index
    return linear


def physical_element_offset(
    layout: PhysicalLayout, indices: Sequence[int]
) -> int:
    coordinate = tuple(indices)
    if len(coordinate) != len(layout.shape):
        raise ValueError("coordinate rank does not match tensor rank")
    if layout.layout in ("Tensor", "NTensor"):
        return _row_major_index(layout.shape, coordinate)

    logical_c = coordinate[-1]
    if logical_c < 0 or logical_c >= layout.shape[-1]:
        raise ValueError("channel coordinate is outside the tensor")
    full_c = layout.full_blocks * layout.c_block
    is_tail = logical_c >= full_c
    channel_offset = (
        logical_c - full_c if is_tail else logical_c % layout.c_block
    )

    if layout.layout == "Cx":
        outer = _row_major_index(layout.shape[:-1], coordinate[:-1])
        if is_tail:
            return (
                layout.full_blocks
                * layout.outer_elements
                * layout.c_block
                + outer * layout.tail_width
                + channel_offset
            )
        block = logical_c // layout.c_block
        return (
            block * layout.outer_elements * layout.c_block
            + outer * layout.c_block
            + channel_offset
        )

    n = coordinate[0]
    if n < 0 or n >= layout.shape[0]:
        raise ValueError("batch coordinate is outside the tensor")
    hw = _row_major_index(layout.shape[1:-1], coordinate[1:-1])
    batch_base = n * layout.batch_elements
    if is_tail:
        return (
            batch_base
            + layout.full_blocks * layout.hw_elements * layout.c_block
            + hw * layout.tail_width
            + channel_offset
        )
    block = logical_c // layout.c_block
    return (
        batch_base
        + block * layout.hw_elements * layout.c_block
        + hw * layout.c_block
        + channel_offset
    )


def coordinates(shape: Sequence[int]) -> Iterator[tuple[int, ...]]:
    return itertools.product(*(range(dimension) for dimension in shape))


def pack_scalar_bytes(
    shape: Sequence[int],
    layout_name: str,
    element_bytes: int,
    logical_values: Iterable[bytes],
    *,
    padding: int = 0xA7,
) -> bytes:
    layout = physical_layout(shape, layout_name, element_bytes)
    values = tuple(logical_values)
    if len(values) != _product(layout.shape):
        raise ValueError("logical scalar count does not match shape")
    if any(len(value) != element_bytes for value in values):
        raise ValueError("logical scalar width does not match element width")
    storage = bytearray([padding] * layout.physical_bytes)
    for coordinate, value in zip(
        coordinates(layout.shape), values, strict=True
    ):
        begin = physical_element_offset(layout, coordinate) * element_bytes
        storage[begin : begin + element_bytes] = value
    return bytes(storage)


def unpack_scalar_bytes(
    shape: Sequence[int],
    layout_name: str,
    element_bytes: int,
    storage: bytes,
) -> tuple[bytes, ...]:
    layout = physical_layout(shape, layout_name, element_bytes)
    if len(storage) != layout.physical_bytes:
        raise ValueError("physical storage size does not match layout")
    values: list[bytes] = []
    for coordinate in coordinates(layout.shape):
        begin = physical_element_offset(layout, coordinate) * element_bytes
        values.append(storage[begin : begin + element_bytes])
    return tuple(values)
