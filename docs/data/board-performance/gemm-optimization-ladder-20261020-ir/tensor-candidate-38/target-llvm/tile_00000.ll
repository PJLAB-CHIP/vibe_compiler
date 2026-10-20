; ModuleID = 'wafer.target.card.0.tile.00000.launch.00000'
source_filename = "wafer_target_physical_tile"
target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_direct_dte_begin_after_prepare(i64, i32)

declare void @wafer_tx81_direct_dte_finish()

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
  call void @wafer_tx81_rdma(i64 %0, i64 65536, i32 2097152, i32 2048, i32 8192, i32 0, i32 0, i32 1024, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %6 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 65536, i64 65536, i32 2097152, i32 0, i32 4, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %6)
  call void @wafer_tx81_direct_dte_wait(i64 %6)
  %7 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 65536, i64 65536, i32 2097152, i32 0, i32 8, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %7)
  call void @wafer_tx81_direct_dte_wait(i64 %7)
  %8 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 65536, i64 65536, i32 2097152, i32 0, i32 12, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %8)
  call void @wafer_tx81_direct_dte_wait(i64 %8)
  br label %9

9:                                                ; preds = %45, %5
  %10 = phi i64 [ %46, %45 ], [ 0, %5 ]
  %11 = icmp slt i64 %10, 1024
  br i1 %11, label %12, label %47

12:                                               ; preds = %9
  br label %13

13:                                               ; preds = %32, %12
  %14 = phi i64 [ %44, %32 ], [ 0, %12 ]
  %15 = icmp slt i64 %14, 1024
  br i1 %15, label %16, label %45

16:                                               ; preds = %13
  call void @wafer_tx81_memset(i64 2162688, i32 0, i32 65536, i32 5, i32 0)
  br label %17

17:                                               ; preds = %21, %16
  %18 = phi i64 [ %31, %21 ], [ 0, %16 ]
  %19 = phi i64 [ %19, %21 ], [ 2162688, %16 ]
  %20 = icmp slt i64 %18, 768
  br i1 %20, label %21, label %32

21:                                               ; preds = %17
  %22 = mul i64 %10, 2048
  %23 = add i64 %1, %22
  %24 = mul i64 %18, 2
  %25 = add i64 %23, %24
  call void @wafer_tx81_rdma(i64 %25, i64 2686976, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2686976, i64 2818048, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  %26 = mul i64 %18, 2048
  %27 = mul i64 %14, 2
  %28 = add i64 %26, %27
  %29 = add i64 65536, %28
  call void @wafer_tx81_gather_scatter(i64 %29, i64 2949120, i32 131072, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 2818048, i64 2949120, i64 2424832, i64 %19, i32 256, i32 256, i32 256, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2424832, i64 %19, i32 262144, i32 262144, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %30 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 2686976, i64 2686976, i32 131072, i32 0, i32 1, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %30)
  call void @wafer_tx81_direct_dte_wait(i64 %30)
  %31 = add i64 %18, 256
  br label %17

32:                                               ; preds = %17
  %33 = mul i64 %10, 2048
  %34 = add i64 %1, %33
  %35 = add i64 %34, 1536
  call void @wafer_tx81_rdma(i64 %35, i64 2424832, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2424832, i64 2555904, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  %36 = mul i64 %14, 2
  %37 = add i64 %36, 1572864
  %38 = add i64 65536, %37
  call void @wafer_tx81_gather_scatter(i64 %38, i64 2686976, i32 131072, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 2555904, i64 2686976, i64 2818048, i64 %19, i32 256, i32 256, i32 256, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2818048, i64 2949120, i32 131072, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 0)
  %39 = mul i64 %10, 8192
  %40 = add i64 %2, %39
  %41 = mul i64 %14, 2
  %42 = add i64 %40, %41
  call void @wafer_tx81_wdma(i64 2949120, i64 %42, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %43 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 2424832, i64 2424832, i32 131072, i32 0, i32 1, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %43)
  call void @wafer_tx81_direct_dte_wait(i64 %43)
  %44 = add i64 %14, 256
  br label %13

45:                                               ; preds = %13
  %46 = add i64 %10, 256
  br label %9

47:                                               ; preds = %9
  call void @wafer_tx81_direct_dte_finish()
  ret void
}

!llvm.module.flags = !{!0}
!wafer.target.card_id = !{!1}
!wafer.target.tile_id = !{!1}
!wafer.target.launch_slot = !{!1}
!wafer.target.entry = !{!2}
!wafer.target.identity = !{!3}
!wafer.target.abi = !{!4}
!wafer.target.module_format = !{!5}
!wafer.target.entry_arguments = !{!6, !7, !8, !9, !10}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i64 0}
!2 = !{!"entry"}
!3 = !{!"wafer-tx81-single-card"}
!4 = !{!"wafer-tx81-kernel"}
!5 = !{!"elf-riscv64"}
!6 = !{i64 0, !"external-input", i64 0, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 1024, i64 4096, i64 0}
!7 = !{i64 1, !"external-input", i64 1, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 4096, i64 1024, i64 0}
!8 = !{i64 2, !"external-output", i64 0, !"", !"bf16", !"tensor", i64 33554432, i64 256, !"write-only", i64 1, i64 4096, i64 4096, i64 0}
!9 = !{i64 3, !"workspace", i64 0, !"default_ddr_arena", !"u8", !"tensor", i64 256, i64 256, !"read-write", i64 256, i64 0}
!10 = !{i64 4, !"transport-status", i64 0, !"direct_dte_status", !"u32", !"tensor", i64 64, i64 64, !"read-write", i64 1, i64 0}

