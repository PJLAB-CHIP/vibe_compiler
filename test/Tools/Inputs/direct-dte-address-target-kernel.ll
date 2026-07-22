target triple = "riscv64-unknown-unknown-elf"

declare i64 @wafer_tx81_direct_dte_send_prepare(i64, i64, i32, i32, i32,
                                                 i32, i32)

define void @kernel_entry() {
entry:
  %event = call i64 @wafer_tx81_direct_dte_send_prepare(
      i64 65536, i64 65792, i32 16, i32 0, i32 1, i32 0, i32 0)
  ret void
}
