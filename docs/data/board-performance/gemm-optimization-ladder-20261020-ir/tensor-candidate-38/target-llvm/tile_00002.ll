; ModuleID = 'wafer.target.card.0.tile.00002.launch.00002'
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
  %6 = add i64 %2, 4096
  %7 = add i64 %0, 4096
  call void @wafer_tx81_rdma(i64 %7, i64 65536, i32 2097152, i32 2048, i32 8192, i32 0, i32 0, i32 1024, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %8 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 65536, i64 65536, i32 2097152, i32 2, i32 6, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %8)
  call void @wafer_tx81_direct_dte_wait(i64 %8)
  %9 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 65536, i64 65536, i32 2097152, i32 2, i32 10, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %9)
  call void @wafer_tx81_direct_dte_wait(i64 %9)
  %10 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 65536, i64 65536, i32 2097152, i32 2, i32 14, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %10)
  call void @wafer_tx81_direct_dte_wait(i64 %10)
  br label %11

11:                                               ; preds = %46, %5
  %12 = phi i64 [ %47, %46 ], [ 0, %5 ]
  %13 = icmp slt i64 %12, 1024
  br i1 %13, label %14, label %48

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %33, %14
  %16 = phi i64 [ %45, %33 ], [ 0, %14 ]
  %17 = icmp slt i64 %16, 1024
  br i1 %17, label %18, label %46

18:                                               ; preds = %15
  call void @wafer_tx81_memset(i64 2162688, i32 0, i32 65536, i32 5, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  br label %19

19:                                               ; preds = %23, %18
  %20 = phi i64 [ %32, %23 ], [ 0, %18 ]
  %21 = phi i64 [ %21, %23 ], [ 2162688, %18 ]
  %22 = icmp slt i64 %20, 768
  br i1 %22, label %23, label %33

23:                                               ; preds = %19
  %24 = add i64 %16, 2048
  %25 = add i64 %16, 2048
  %26 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 2686976, i32 131072, i32 2, i32 1, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %26)
  call void @wafer_tx81_gather_scatter(i64 2686976, i64 2818048, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  %27 = mul i64 %20, 2048
  %28 = mul i64 %16, 2
  %29 = add i64 %27, %28
  %30 = add i64 65536, %29
  call void @wafer_tx81_gather_scatter(i64 %30, i64 2949120, i32 131072, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 2818048, i64 2949120, i64 2424832, i64 %21, i32 256, i32 256, i32 256, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2424832, i64 %21, i32 262144, i32 262144, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %31 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 2686976, i64 2555904, i32 131072, i32 2, i32 3, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %31)
  call void @wafer_tx81_direct_dte_wait(i64 %31)
  %32 = add i64 %20, 256
  br label %19

33:                                               ; preds = %19
  %34 = add i64 %16, 2048
  %35 = add i64 %16, 2048
  %36 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 2424832, i32 131072, i32 2, i32 1, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %36)
  call void @wafer_tx81_gather_scatter(i64 2424832, i64 2555904, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  %37 = mul i64 %16, 2
  %38 = add i64 %37, 1572864
  %39 = add i64 65536, %38
  call void @wafer_tx81_gather_scatter(i64 %39, i64 2686976, i32 131072, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 2555904, i64 2686976, i64 2818048, i64 %21, i32 256, i32 256, i32 256, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2818048, i64 2949120, i32 131072, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 0)
  %40 = mul i64 %12, 8192
  %41 = add i64 %6, %40
  %42 = mul i64 %16, 2
  %43 = add i64 %41, %42
  call void @wafer_tx81_wdma(i64 2949120, i64 %43, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %44 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 2424832, i64 2818048, i32 131072, i32 2, i32 3, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %44)
  call void @wafer_tx81_direct_dte_wait(i64 %44)
  %45 = add i64 %16, 256
  br label %15

46:                                               ; preds = %15
  %47 = add i64 %12, 256
  br label %11

48:                                               ; preds = %11
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
!2 = !{i64 2}
!3 = !{!"entry"}
!4 = !{!"wafer-tx81-single-card"}
!5 = !{!"wafer-tx81-kernel"}
!6 = !{!"elf-riscv64"}
!7 = !{i64 0, !"external-input", i64 0, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 1024, i64 4096, i64 0}
!8 = !{i64 1, !"external-input", i64 1, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 4096, i64 1024, i64 0}
!9 = !{i64 2, !"external-output", i64 0, !"", !"bf16", !"tensor", i64 33554432, i64 256, !"write-only", i64 1, i64 4096, i64 4096, i64 0}
!10 = !{i64 3, !"workspace", i64 0, !"default_ddr_arena", !"u8", !"tensor", i64 256, i64 256, !"read-write", i64 256, i64 0}
!11 = !{i64 4, !"transport-status", i64 0, !"direct_dte_status", !"u32", !"tensor", i64 64, i64 64, !"read-write", i64 1, i64 0}

