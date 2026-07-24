target triple = "riscv64-unknown-unknown-elf"

declare i32 @__get_pid(i32)
declare i32 @init_tile_id(i32, i32)
declare void @direct_sync_init(i32)
declare void @wafer_tx81_ddr_tile_offset_probe(i32, i64, i64, i64, i64, i64)

define void @__wafer_cluster_prepare(ptr %rank_major_slots) {
entry:
  %pid = call i32 @__get_pid(i32 0)
  %unused = call i32 @init_tile_id(i32 %pid, i32 4)
  call void @direct_sync_init(i32 16)
  ret void
}

define void @main(ptr %rank_major_slots) {
entry:
  %pid = call i32 @__get_pid(i32 0)
  %pid64 = zext i32 %pid to i64
  %row = mul i64 %pid64, 5
  %input0.index = add i64 %row, 0
  %input1.index = add i64 %row, 1
  %output0.index = add i64 %row, 2
  %output1.index = add i64 %row, 3
  %status.index = add i64 %row, 4
  %input0.slot = getelementptr inbounds i64, ptr %rank_major_slots, i64 %input0.index
  %input1.slot = getelementptr inbounds i64, ptr %rank_major_slots, i64 %input1.index
  %output0.slot = getelementptr inbounds i64, ptr %rank_major_slots, i64 %output0.index
  %output1.slot = getelementptr inbounds i64, ptr %rank_major_slots, i64 %output1.index
  %status.slot = getelementptr inbounds i64, ptr %rank_major_slots, i64 %status.index
  %input0 = load i64, ptr %input0.slot, align 8
  %input1 = load i64, ptr %input1.slot, align 8
  %output0 = load i64, ptr %output0.slot, align 8
  %output1 = load i64, ptr %output1.slot, align 8
  %status = load i64, ptr %status.slot, align 8
  call void @wafer_tx81_ddr_tile_offset_probe(i32 %pid, i64 %input0, i64 %input1, i64 %output0, i64 %output1, i64 %status)
  ret void
}
