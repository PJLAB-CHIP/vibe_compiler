target triple = "riscv64-unknown-unknown-elf"

declare i64 @wafer_tx81_direct_dte_multisend_prepare(i64, i32, i32, i32, i32,
                                                      i32)
declare void @wafer_tx81_direct_dte_multisend_add_destination(i64, i64, i32,
                                                              i32)
declare void @wafer_tx81_direct_dte_send_issue(i64)
declare void @wafer_tx81_direct_dte_wait(i64)

define void @kernel_entry() {
entry:
  %event = call i64 @wafer_tx81_direct_dte_multisend_prepare(
      i64 65536, i32 256, i32 0, i32 2, i32 2, i32 0)
  call void @wafer_tx81_direct_dte_multisend_add_destination(
      i64 %event, i64 65792, i32 1, i32 0)
  call void @wafer_tx81_direct_dte_multisend_add_destination(
      i64 %event, i64 66048, i32 2, i32 1)
  call void @wafer_tx81_direct_dte_send_issue(i64 %event)
  call void @wafer_tx81_direct_dte_wait(i64 %event)
  ret void
}
