target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_rdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

define void @kernel_entry(i64 %input, i64 %output) {
entry:
  call void @wafer_tx81_rdma(i64 %input, i64 %output,
                             i32 64, i32 64,
                             i32 0, i32 0, i32 0,
                             i32 1, i32 1, i32 1,
                             i32 5, i32 0)
  ret void
}
