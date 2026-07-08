target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_missing()

define void @kernel_entry(i64 %input0, i64 %output0) {
entry:
  call void @wafer_tx81_missing()
  ret void
}
