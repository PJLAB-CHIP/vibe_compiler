target triple = "riscv64-unknown-unknown-elf"

declare i32 @__get_pid(i32)
declare i32 @init_tile_id(i32, i32)
declare void @direct_sync_init(i32)
declare void @wafer_tx81_complete_tile_barrier_probe(i32, i64, i64, i64, i64)

@wafer_barrier_slots_per_tile = external hidden constant i64
@wafer_barrier_input_slot = external hidden constant i64
@wafer_barrier_output_slot = external hidden constant i64
@wafer_barrier_status_slot = external hidden constant i64

define void @__wafer_kernel_prepare(ptr %tile_major_slots) {
entry:
  %pid = call i32 @__get_pid(i32 0)
  %unused = call i32 @init_tile_id(i32 %pid, i32 4)
  call void @direct_sync_init(i32 16)
  ret void
}

define void @main(ptr %tile_major_slots) {
entry:
  %pid = call i32 @__get_pid(i32 0)
  %pid64 = zext i32 %pid to i64
  %slots.per.tile_id = load i64, ptr @wafer_barrier_slots_per_tile, align 8
  %input.ordinal = load i64, ptr @wafer_barrier_input_slot, align 8
  %output.ordinal = load i64, ptr @wafer_barrier_output_slot, align 8
  %status.ordinal = load i64, ptr @wafer_barrier_status_slot, align 8
  %row = mul i64 %pid64, %slots.per.tile_id
  %input.index = add i64 %row, %input.ordinal
  %output.index = add i64 %row, %output.ordinal
  %status.index = add i64 %row, %status.ordinal
  %input.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %input.index
  %output.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %output.index
  %status.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %status.index
  %input = load i64, ptr %input.slot, align 8
  %output = load i64, ptr %output.slot, align 8
  %status = load i64, ptr %status.slot, align 8
  %slots = ptrtoint ptr %tile_major_slots to i64
  call void @wafer_tx81_complete_tile_barrier_probe(i32 %pid, i64 %input, i64 %output, i64 %status, i64 %slots)
  ret void
}
