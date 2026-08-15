target triple = "riscv64-unknown-unknown-elf"

declare i32 @__get_pid(i32)
declare i32 @init_tile_id(i32, i32)
declare void @direct_sync_init(i32)
declare void @wafer_tx81_spm_cross_tile_conflict_probe(
    i32, i64, i64, i64, i64, i64)

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
  %row = mul i64 %pid64, 5
  %request.index = add i64 %row, 0
  %payload.index = add i64 %row, 1
  %serial.index = add i64 %row, 2
  %window.index = add i64 %row, 3
  %status.index = add i64 %row, 4
  %request.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %request.index
  %payload.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %payload.index
  %serial.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %serial.index
  %window.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %window.index
  %status.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %status.index
  %request = load i64, ptr %request.slot, align 8
  %payload = load i64, ptr %payload.slot, align 8
  %serial = load i64, ptr %serial.slot, align 8
  %window = load i64, ptr %window.slot, align 8
  %status = load i64, ptr %status.slot, align 8
  call void @wafer_tx81_spm_cross_tile_conflict_probe(
      i32 %pid, i64 %request, i64 %payload, i64 %serial, i64 %window,
      i64 %status)
  ret void
}
