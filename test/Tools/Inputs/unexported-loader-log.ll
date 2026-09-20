; Loader-only negative: SDK declarations do not establish firmware exports.
target triple = "riscv64-unknown-unknown-elf"

declare i32 @tx8_kernel_printf(ptr, ...)
declare i32 @tx8_kernel_vprintf(ptr, ptr)
declare void @tsm_ep_log(ptr, ptr, i32, i32, ptr, ...)

define void @kernel_entry(ptr %format, ptr %args) {
  %a = call i32 (ptr, ...) @tx8_kernel_printf(ptr %format)
  %b = call i32 @tx8_kernel_vprintf(ptr %format, ptr %args)
  call void (ptr, ptr, i32, i32, ptr, ...) @tsm_ep_log(ptr %format, ptr %format, i32 0, i32 3, ptr %format)
  ret void
}
