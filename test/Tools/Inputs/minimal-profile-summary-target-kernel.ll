target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_profile_entry_begin_from_config(i64)
declare void @wafer_tx81_profile_entry_end()

define void @main() {
entry:
  call void @wafer_tx81_profile_entry_begin_from_config(i64 4096)
  call void @wafer_tx81_profile_entry_end()
  ret void
}
