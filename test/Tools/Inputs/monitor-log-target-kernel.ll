; Loader ABI witness; no tensor computation or device execution in this test.
target triple = "riscv64-unknown-unknown-elf"

declare void @monitor_write_log(ptr, ptr, i32, ptr, ...)
declare void @rcs_ep_log(ptr, ptr, i32, i32, ptr, ...)

define void @kernel_entry(ptr %format) {
  call void (ptr, ptr, i32, ptr, ...) @monitor_write_log(ptr %format, ptr %format, i32 0, ptr %format)
  call void (ptr, ptr, i32, i32, ptr, ...) @rcs_ep_log(ptr %format, ptr %format, i32 0, i32 5, ptr %format)
  ret void
}
