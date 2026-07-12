target triple = "riscv64-unknown-unknown-elf"

declare void @unexpected_loader_hook()

define void @kernel_entry(i64 %input0, i64 %output0) {
entry:
  call void @unexpected_loader_hook()
  ret void
}
