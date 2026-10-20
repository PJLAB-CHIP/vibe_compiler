; ModuleID = 'wafer.target.card.0.tile.00005.launch.00005'
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

declare void @wafer_tx81_wdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

define void @entry(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4) {
  call void @wafer_tx81_direct_dte_begin_after_prepare(i64 %4, i32 16)
  %6 = add i64 %2, 8390656
  %7 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 65536, i32 2097152, i32 5, i32 1, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %7)
  br label %8

8:                                                ; preds = %45, %5
  %9 = phi i64 [ %46, %45 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 1024
  br i1 %10, label %11, label %47

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %31, %11
  %13 = phi i64 [ %44, %31 ], [ 0, %11 ]
  %14 = icmp slt i64 %13, 1024
  br i1 %14, label %15, label %45

15:                                               ; preds = %12
  call void @wafer_tx81_memset(i64 2162688, i32 0, i32 65536, i32 5, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  br label %16

16:                                               ; preds = %20, %15
  %17 = phi i64 [ %30, %20 ], [ 0, %15 ]
  %18 = phi i64 [ %18, %20 ], [ 2162688, %15 ]
  %19 = icmp slt i64 %17, 768
  br i1 %19, label %20, label %31

20:                                               ; preds = %16
  %21 = add i64 %9, 1024
  %22 = add i64 %13, 1024
  %23 = add i64 %13, 1024
  %24 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 2686976, i32 131072, i32 5, i32 4, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %24)
  call void @wafer_tx81_gather_scatter(i64 2686976, i64 2818048, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  %25 = mul i64 %17, 2048
  %26 = mul i64 %13, 2
  %27 = add i64 %25, %26
  %28 = add i64 65536, %27
  call void @wafer_tx81_gather_scatter(i64 %28, i64 2949120, i32 131072, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 2818048, i64 2949120, i64 2424832, i64 %18, i32 256, i32 256, i32 256, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2424832, i64 %18, i32 262144, i32 262144, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %29 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 2686976, i64 2686976, i32 131072, i32 5, i32 6, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %29)
  call void @wafer_tx81_direct_dte_wait(i64 %29)
  %30 = add i64 %17, 256
  br label %16

31:                                               ; preds = %16
  %32 = add i64 %9, 1024
  %33 = add i64 %13, 1024
  %34 = add i64 %13, 1024
  %35 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 2424832, i32 131072, i32 5, i32 4, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %35)
  call void @wafer_tx81_gather_scatter(i64 2424832, i64 2555904, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  %36 = mul i64 %13, 2
  %37 = add i64 %36, 1572864
  %38 = add i64 65536, %37
  call void @wafer_tx81_gather_scatter(i64 %38, i64 2686976, i32 131072, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 2555904, i64 2686976, i64 2818048, i64 %18, i32 256, i32 256, i32 256, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2818048, i64 2949120, i32 131072, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 0)
  %39 = mul i64 %9, 8192
  %40 = add i64 %6, %39
  %41 = mul i64 %13, 2
  %42 = add i64 %40, %41
  call void @wafer_tx81_wdma(i64 2949120, i64 %42, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %43 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 2424832, i64 2424832, i32 131072, i32 5, i32 6, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %43)
  call void @wafer_tx81_direct_dte_wait(i64 %43)
  %44 = add i64 %13, 256
  br label %12

45:                                               ; preds = %12
  %46 = add i64 %9, 256
  br label %8

47:                                               ; preds = %8
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
!2 = !{i64 5}
!3 = !{!"entry"}
!4 = !{!"wafer-tx81-single-card"}
!5 = !{!"wafer-tx81-kernel"}
!6 = !{!"elf-riscv64"}
!7 = !{i64 0, !"external-input", i64 0, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 1024, i64 4096, i64 0}
!8 = !{i64 1, !"external-input", i64 1, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 4096, i64 1024, i64 0}
!9 = !{i64 2, !"external-output", i64 0, !"", !"bf16", !"tensor", i64 33554432, i64 256, !"write-only", i64 1, i64 4096, i64 4096, i64 0}
!10 = !{i64 3, !"workspace", i64 0, !"default_ddr_arena", !"u8", !"tensor", i64 256, i64 256, !"read-write", i64 256, i64 0}
!11 = !{i64 4, !"transport-status", i64 0, !"direct_dte_status", !"u32", !"tensor", i64 64, i64 64, !"read-write", i64 1, i64 0}

