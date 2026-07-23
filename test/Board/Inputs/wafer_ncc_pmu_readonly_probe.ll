target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_board_probe(i64)

define void @main(ptr %slots) {
entry:
  %output_slot = getelementptr inbounds i64, ptr %slots, i64 2
  %output = load i64, ptr %output_slot, align 8
  call void @wafer_tx81_board_probe(i64 %output)
  ret void
}
