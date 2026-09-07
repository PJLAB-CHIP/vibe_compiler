target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_board_probe(i64)
declare i32 @__get_pid(i32)

@wafer_pmu_slots_per_tile = external hidden constant i64

define void @entry(ptr %tile_major_slots) {
entry:
  %pid = call i32 @__get_pid(i32 0)
  %is_tile_zero = icmp eq i32 %pid, 0
  br i1 %is_tile_zero, label %run, label %done

run:
  %pid64 = zext i32 %pid to i64
  %slots_per_tile = load i64, ptr @wafer_pmu_slots_per_tile, align 8
  %row = mul i64 %pid64, %slots_per_tile
  %tile_slots = getelementptr inbounds i64, ptr %tile_major_slots, i64 %row
  %output_slot = getelementptr inbounds i64, ptr %tile_slots, i64 2
  %output = load i64, ptr %output_slot, align 8
  call void @wafer_tx81_board_probe(i64 %output)
  br label %done

done:
  ret void
}
