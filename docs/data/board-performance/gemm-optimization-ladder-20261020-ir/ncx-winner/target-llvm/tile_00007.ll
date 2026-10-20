; ModuleID = 'wafer.target.card.0.tile.00007.launch.00007'
source_filename = "wafer_target_physical_tile"
target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_direct_dte_begin_after_prepare(i64, i32)

declare void @wafer_tx81_direct_dte_finish()

declare i64 @wafer_tx81_direct_dte_recv_prepare(i64, i32, i32, i32, i32)

declare void @wafer_tx81_direct_dte_send_issue(i64)

declare i64 @wafer_tx81_direct_dte_send_prepare(i64, i64, i32, i32, i32, i32, i32)

declare void @wafer_tx81_direct_dte_wait(i64)

declare void @wafer_tx81_gather_scatter(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_gemm(i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_memset(i64, i32, i32, i32, i32)

declare void @wafer_tx81_ncc_join(i32)

declare void @wafer_tx81_rdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_wdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

define void @entry(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4) {
  call void @wafer_tx81_direct_dte_begin_after_prepare(i64 %4, i32 16)
  br label %6

6:                                                ; preds = %22, %5
  %7 = phi i64 [ %30, %22 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, 4096
  br i1 %8, label %9, label %31

9:                                                ; preds = %6
  call void @wafer_tx81_memset(i64 65536, i32 0, i32 262144, i32 5, i32 0)
  br label %10

10:                                               ; preds = %14, %9
  %11 = phi i64 [ %21, %14 ], [ 0, %9 ]
  %12 = phi i64 [ %12, %14 ], [ 65536, %9 ]
  %13 = icmp slt i64 %11, 768
  br i1 %13, label %14, label %22

14:                                               ; preds = %10
  %15 = udiv i64 %11, 64
  %16 = mul i64 %15, 524288
  %17 = add i64 %16, 229376
  %18 = add i64 %1, %17
  call void @wafer_tx81_rdma(i64 %18, i64 2686976, i32 131072, i32 32768, i32 524288, i32 0, i32 0, i32 4, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %19 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 2162688, i32 524288, i32 7, i32 3, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %19)
  call void @wafer_tx81_gemm(i64 2686976, i64 2162688, i64 1114112, i64 %12, i32 256, i32 256, i32 1024, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1114112, i64 %12, i32 1048576, i32 1048576, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %20 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 2162688, i64 2162688, i32 524288, i32 7, i32 6, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %20)
  call void @wafer_tx81_direct_dte_wait(i64 %20)
  %21 = add i64 %11, 256
  br label %10

22:                                               ; preds = %10
  %23 = add i64 %1, 6520832
  call void @wafer_tx81_rdma(i64 %23, i64 2686976, i32 131072, i32 32768, i32 524288, i32 0, i32 0, i32 4, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %24 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 1114112, i32 524288, i32 7, i32 3, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %24)
  call void @wafer_tx81_gemm(i64 2686976, i64 1114112, i64 1638400, i64 %12, i32 256, i32 256, i32 1024, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1638400, i64 2162688, i32 524288, i32 524288, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %25 = udiv i64 %7, 64
  %26 = mul i64 %25, 524288
  %27 = add i64 %26, 229376
  %28 = add i64 %2, %27
  call void @wafer_tx81_wdma(i64 2162688, i64 %28, i32 524288, i32 32768, i32 524288, i32 0, i32 0, i32 16, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %29 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 1114112, i64 1114112, i32 524288, i32 7, i32 6, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %29)
  call void @wafer_tx81_direct_dte_wait(i64 %29)
  %30 = add i64 %7, 1024
  br label %6

31:                                               ; preds = %6
  call void @wafer_tx81_direct_dte_finish()
  ret void
}

!llvm.module.flags = !{!0}
!wafer.target.card_id = !{!1}
!wafer.target.tile_id = !{!2}
!wafer.target.launch_slot = !{!2}
!wafer.target.entry = !{!3}
!wafer.target.identity = !{!4}
!wafer.target.abi = !{!5}
!wafer.target.module_format = !{!6}
!wafer.target.entry_arguments = !{!7, !8, !9, !10, !11}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i64 0}
!2 = !{i64 7}
!3 = !{!"entry"}
!4 = !{!"wafer-tx81-single-card"}
!5 = !{!"wafer-tx81-kernel"}
!6 = !{!"elf-riscv64"}
!7 = !{i64 0, !"external-input", i64 0, !"", !"bf16", !"ncx", i64 8388608, i64 256, !"read-only", i64 1, i64 1024, i64 4096, i64 0}
!8 = !{i64 1, !"external-input", i64 1, !"", !"bf16", !"ncx", i64 8388608, i64 256, !"read-only", i64 1, i64 4096, i64 1024, i64 0}
!9 = !{i64 2, !"external-output", i64 0, !"", !"bf16", !"ncx", i64 33554432, i64 256, !"write-only", i64 1, i64 4096, i64 4096, i64 0}
!10 = !{i64 3, !"workspace", i64 0, !"default_ddr_arena", !"u8", !"tensor", i64 256, i64 256, !"read-write", i64 256, i64 0}
!11 = !{i64 4, !"transport-status", i64 0, !"direct_dte_status", !"u32", !"tensor", i64 64, i64 64, !"read-write", i64 1, i64 0}

