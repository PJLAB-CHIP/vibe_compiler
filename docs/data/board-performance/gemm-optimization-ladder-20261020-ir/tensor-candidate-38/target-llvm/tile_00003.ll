; ModuleID = 'wafer.target.card.0.tile.00003.launch.00003'
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
  %6 = add i64 %2, 6144
  %7 = add i64 %0, 6144
  call void @wafer_tx81_rdma(i64 %7, i64 65536, i32 2097152, i32 2048, i32 8192, i32 0, i32 0, i32 1024, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %8 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 65536, i64 65536, i32 2097152, i32 3, i32 7, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %8)
  call void @wafer_tx81_direct_dte_wait(i64 %8)
  %9 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 65536, i64 65536, i32 2097152, i32 3, i32 11, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %9)
  call void @wafer_tx81_direct_dte_wait(i64 %9)
  %10 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 65536, i64 65536, i32 2097152, i32 3, i32 15, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %10)
  br label %11

11:                                               ; preds = %44, %5
  %12 = phi i64 [ %45, %44 ], [ 0, %5 ]
  %13 = icmp slt i64 %12, 1024
  br i1 %13, label %14, label %46

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %32, %14
  %16 = phi i64 [ %43, %32 ], [ 0, %14 ]
  %17 = icmp slt i64 %16, 1024
  br i1 %17, label %18, label %44

18:                                               ; preds = %15
  call void @wafer_tx81_memset(i64 2162688, i32 0, i32 65536, i32 5, i32 0)
  br label %19

19:                                               ; preds = %23, %18
  %20 = phi i64 [ %31, %23 ], [ 0, %18 ]
  %21 = phi i64 [ %21, %23 ], [ 2162688, %18 ]
  %22 = icmp slt i64 %20, 768
  br i1 %22, label %23, label %32

23:                                               ; preds = %19
  %24 = add i64 %16, 3072
  %25 = add i64 %16, 3072
  call void @wafer_tx81_ncc_join(i32 1)
  %26 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 2555904, i32 131072, i32 3, i32 2, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %26)
  call void @wafer_tx81_gather_scatter(i64 2555904, i64 2424832, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  %27 = mul i64 %20, 2048
  %28 = mul i64 %16, 2
  %29 = add i64 %27, %28
  %30 = add i64 65536, %29
  call void @wafer_tx81_gather_scatter(i64 %30, i64 2818048, i32 131072, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 2424832, i64 2818048, i64 2555904, i64 %21, i32 256, i32 256, i32 256, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2555904, i64 %21, i32 262144, i32 262144, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %31 = add i64 %20, 256
  br label %19

32:                                               ; preds = %19
  %33 = add i64 %16, 3072
  %34 = add i64 %16, 3072
  call void @wafer_tx81_ncc_join(i32 1)
  %35 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 2818048, i32 131072, i32 3, i32 2, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %35)
  call void @wafer_tx81_gather_scatter(i64 2818048, i64 2424832, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  %36 = mul i64 %16, 2
  %37 = add i64 %36, 1572864
  %38 = add i64 65536, %37
  call void @wafer_tx81_gather_scatter(i64 %38, i64 2555904, i32 131072, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 2424832, i64 2555904, i64 2686976, i64 %21, i32 256, i32 256, i32 256, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 2686976, i64 2818048, i32 131072, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 0)
  %39 = mul i64 %12, 8192
  %40 = add i64 %6, %39
  %41 = mul i64 %16, 2
  %42 = add i64 %40, %41
  call void @wafer_tx81_wdma(i64 2818048, i64 %42, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %43 = add i64 %16, 256
  br label %15

44:                                               ; preds = %15
  %45 = add i64 %12, 256
  br label %11

46:                                               ; preds = %11
  call void @wafer_tx81_direct_dte_wait(i64 %10)
  call void @wafer_tx81_ncc_join(i32 1)
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
!2 = !{i64 3}
!3 = !{!"entry"}
!4 = !{!"wafer-tx81-single-card"}
!5 = !{!"wafer-tx81-kernel"}
!6 = !{!"elf-riscv64"}
!7 = !{i64 0, !"external-input", i64 0, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 1024, i64 4096, i64 0}
!8 = !{i64 1, !"external-input", i64 1, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 4096, i64 1024, i64 0}
!9 = !{i64 2, !"external-output", i64 0, !"", !"bf16", !"tensor", i64 33554432, i64 256, !"write-only", i64 1, i64 4096, i64 4096, i64 0}
!10 = !{i64 3, !"workspace", i64 0, !"default_ddr_arena", !"u8", !"tensor", i64 256, i64 256, !"read-write", i64 256, i64 0}
!11 = !{i64 4, !"transport-status", i64 0, !"direct_dte_status", !"u32", !"tensor", i64 64, i64 64, !"read-write", i64 1, i64 0}

