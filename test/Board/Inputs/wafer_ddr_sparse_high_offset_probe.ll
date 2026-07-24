target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_ddr_sparse_high_offset_probe(i64, i64, i64)

define void @main(ptr %slots) {
entry:
  %input.slot = getelementptr inbounds i64, ptr %slots, i64 0
  %output.slot = getelementptr inbounds i64, ptr %slots, i64 1
  %workspace.slot = getelementptr inbounds i64, ptr %slots, i64 2
  %input = load i64, ptr %input.slot, align 8
  %output = load i64, ptr %output.slot, align 8
  %workspace = load i64, ptr %workspace.slot, align 8
  call void @wafer_tx81_ddr_sparse_high_offset_probe(
      i64 %input, i64 %output, i64 %workspace)
  ret void
}
