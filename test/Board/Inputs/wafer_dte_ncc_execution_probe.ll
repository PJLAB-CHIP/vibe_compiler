target triple = "riscv64-unknown-unknown-elf"

declare i32 @__get_pid(i32)
declare i32 @init_tile_id(i32, i32)
declare void @direct_sync_init(i32)
declare void @wafer_tx81_dte_ncc_execution_probe(i32, i64, i64, i64)

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
  %row = mul i64 %pid64, 3
  %input.index = add i64 %row, 0
  %output.index = add i64 %row, 1
  %status.index = add i64 %row, 2
  %input.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %input.index
  %output.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %output.index
  %status.slot = getelementptr inbounds i64, ptr %tile_major_slots, i64 %status.index
  %input.base = load i64, ptr %input.slot, align 8
  %output.base = load i64, ptr %output.slot, align 8
  %status = load i64, ptr %status.slot, align 8
  %resource.offset = mul i64 %pid64, 264448
  %input = add i64 %input.base, %resource.offset
  %output = add i64 %output.base, %resource.offset
  call void @wafer_tx81_dte_ncc_execution_probe(i32 %pid, i64 %input, i64 %output, i64 %status)
  ret void
}
