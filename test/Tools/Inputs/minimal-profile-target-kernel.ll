target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_profile_entry_begin_from_config(i64)
declare void @wafer_tx81_profile_entry_end()
declare void @wafer_tx81_profile_site_begin(i32)
declare void @wafer_tx81_profile_site_end(i32)

define void @main() {
entry:
  call void @wafer_tx81_profile_entry_begin_from_config(i64 4096)
  call void @wafer_tx81_profile_site_begin(i32 0)
  call void @wafer_tx81_profile_site_end(i32 0)
  call void @wafer_tx81_profile_entry_end()
  ret void
}
