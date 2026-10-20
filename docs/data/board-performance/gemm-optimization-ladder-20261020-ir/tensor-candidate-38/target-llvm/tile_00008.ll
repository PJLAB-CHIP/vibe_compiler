; ModuleID = 'wafer.target.card.0.tile.00008.launch.00008'
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
  %6 = add i64 %2, 16777216
  %7 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 65536, i32 2097152, i32 8, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %7)
  br label %8

8:                                                ; preds = %46, %5
  %9 = phi i64 [ %47, %46 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 1024
  br i1 %10, label %11, label %48

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %32, %11
  %13 = phi i64 [ %45, %32 ], [ 0, %11 ]
  %14 = icmp slt i64 %13, 1024
  br i1 %14, label %15, label %46

15:                                               ; preds = %12
  call void @wafer_tx81_memset(i64 2162688, i32 0, i32 65536, i32 5, i32 0)
  br label %16

16:                                               ; preds = %20, %15
  %17 = phi i64 [ %31, %20 ], [ 0, %15 ]
  %18 = phi i64 [ %18, %20 ], [ 2162688, %15 ]
  %19 = icmp slt i64 %17, 768
  br i1 %19, label %20, label %32

20:                                               ; preds = %16
  %21 = add i64 %9, 2048
  %22 = mul i64 %21, 2048
  %23 = add i64 %1, %22
  %24 = mul i64 %17, 2
  %25 = add i64 %23, %24
  call void @wafer_tx81_rdma(i64 %25, i64 2686976, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2686976, i64 2818048, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  %26 = mul i64 %17, 2048
  %27 = mul i64 %13, 2
  %28 = add i64 %26, %27
  %29 = add i64 65536, %28
  call void @wafer_tx81_gather_scatter(i64 %29, i64 2949120, i32 131072, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 2818048, i64 2949120, i64 2424832, i64 %18, i32 256, i32 256, i32 256, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2424832, i64 %18, i32 262144, i32 262144, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %30 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 2686976, i64 2686976, i32 131072, i32 8, i32 9, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %30)
  call void @wafer_tx81_direct_dte_wait(i64 %30)
  %31 = add i64 %17, 256
  br label %16

32:                                               ; preds = %16
  %33 = add i64 %9, 2048
  %34 = mul i64 %33, 2048
  %35 = add i64 %1, %34
  %36 = add i64 %35, 1536
  call void @wafer_tx81_rdma(i64 %36, i64 2424832, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2424832, i64 2555904, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  %37 = mul i64 %13, 2
  %38 = add i64 %37, 1572864
  %39 = add i64 65536, %38
  call void @wafer_tx81_gather_scatter(i64 %39, i64 2686976, i32 131072, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 2555904, i64 2686976, i64 2818048, i64 %18, i32 256, i32 256, i32 256, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2818048, i64 2949120, i32 131072, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 0)
  %40 = mul i64 %9, 8192
  %41 = add i64 %6, %40
  %42 = mul i64 %13, 2
  %43 = add i64 %41, %42
  call void @wafer_tx81_wdma(i64 2949120, i64 %43, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %44 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 2424832, i64 2424832, i32 131072, i32 8, i32 9, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %44)
  call void @wafer_tx81_direct_dte_wait(i64 %44)
  %45 = add i64 %13, 256
  br label %12

46:                                               ; preds = %12
  %47 = add i64 %9, 256
  br label %8

48:                                               ; preds = %8
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
!2 = !{i64 8}
!3 = !{!"entry"}
!4 = !{!"wafer-tx81-single-card"}
!5 = !{!"wafer-tx81-kernel"}
!6 = !{!"elf-riscv64"}
!7 = !{i64 0, !"external-input", i64 0, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 1024, i64 4096, i64 0}
!8 = !{i64 1, !"external-input", i64 1, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 4096, i64 1024, i64 0}
!9 = !{i64 2, !"external-output", i64 0, !"", !"bf16", !"tensor", i64 33554432, i64 256, !"write-only", i64 1, i64 4096, i64 4096, i64 0}
!10 = !{i64 3, !"workspace", i64 0, !"default_ddr_arena", !"u8", !"tensor", i64 256, i64 256, !"read-write", i64 256, i64 0}
!11 = !{i64 4, !"transport-status", i64 0, !"direct_dte_status", !"u32", !"tensor", i64 64, i64 64, !"read-write", i64 1, i64 0}

