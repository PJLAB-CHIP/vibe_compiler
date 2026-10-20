; ModuleID = 'wafer.target.card.0.tile.00012.launch.00012'
source_filename = "wafer_target_physical_tile"
target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_direct_dte_begin_after_prepare(i64, i32)

declare void @wafer_tx81_direct_dte_finish()

declare i64 @wafer_tx81_direct_dte_recv_prepare(i64, i32, i32, i32, i32)

declare void @wafer_tx81_direct_dte_wait(i64)

declare void @wafer_tx81_gather_scatter(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_gemm(i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_memset(i64, i32, i32, i32, i32)

declare void @wafer_tx81_ncc_join(i32)

declare void @wafer_tx81_rdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_wdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

define void @entry(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4) {
  call void @wafer_tx81_direct_dte_begin_after_prepare(i64 %4, i32 16)
  %6 = add i64 %2, 25165824
  %7 = add i64 %1, 6291456
  call void @wafer_tx81_rdma(i64 %7, i64 65536, i32 524288, i32 524288, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 3, i32 0)
  br label %8

8:                                                ; preds = %21, %5
  %9 = phi i64 [ %25, %21 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 4096
  br i1 %10, label %11, label %26

11:                                               ; preds = %8
  call void @wafer_tx81_memset(i64 589824, i32 0, i32 131072, i32 5, i32 0)
  br label %12

12:                                               ; preds = %16, %11
  %13 = phi i64 [ %20, %16 ], [ 0, %11 ]
  %14 = phi i64 [ %14, %16 ], [ 589824, %11 ]
  %15 = icmp slt i64 %13, 896
  br i1 %15, label %16, label %21

16:                                               ; preds = %12
  %17 = mul i64 %13, 2
  %18 = add i64 65536, %17
  call void @wafer_tx81_gather_scatter(i64 %18, i64 1900544, i32 65536, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 2, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 2, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %19 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 1114112, i32 131072, i32 12, i32 13, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %19)
  call void @wafer_tx81_gather_scatter(i64 1114112, i64 1769472, i32 131072, i32 128, i32 1024, i32 128, i32 0, i32 128, i32 8, i32 1, i32 128, i32 16384, i32 0, i32 128, i32 8, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 1900544, i64 1769472, i64 1245184, i64 %14, i32 256, i32 128, i32 512, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1245184, i64 %14, i32 524288, i32 524288, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %20 = add i64 %13, 128
  br label %12

21:                                               ; preds = %12
  call void @wafer_tx81_gather_scatter(i64 67328, i64 1114112, i32 65536, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 2, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 2, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %22 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 1769472, i32 131072, i32 12, i32 13, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %22)
  call void @wafer_tx81_gather_scatter(i64 1769472, i64 1441792, i32 131072, i32 128, i32 1024, i32 128, i32 0, i32 128, i32 8, i32 1, i32 128, i32 16384, i32 0, i32 128, i32 8, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 1114112, i64 1441792, i64 1179648, i64 %14, i32 256, i32 128, i32 512, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1179648, i64 1572864, i32 262144, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 8, i32 1, i32 1024, i32 128, i32 0, i32 256, i32 8, i32 1, i32 0)
  %23 = mul i64 %9, 2
  %24 = add i64 %6, %23
  call void @wafer_tx81_wdma(i64 1572864, i64 %24, i32 262144, i32 1024, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %25 = add i64 %9, 512
  br label %8

26:                                               ; preds = %8
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
!2 = !{i64 12}
!3 = !{!"entry"}
!4 = !{!"wafer-tx81-single-card"}
!5 = !{!"wafer-tx81-kernel"}
!6 = !{!"elf-riscv64"}
!7 = !{i64 0, !"external-input", i64 0, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 1024, i64 4096, i64 0}
!8 = !{i64 1, !"external-input", i64 1, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 4096, i64 1024, i64 0}
!9 = !{i64 2, !"external-output", i64 0, !"", !"bf16", !"tensor", i64 33554432, i64 256, !"write-only", i64 1, i64 4096, i64 4096, i64 0}
!10 = !{i64 3, !"workspace", i64 0, !"default_ddr_arena", !"u8", !"tensor", i64 256, i64 256, !"read-write", i64 256, i64 0}
!11 = !{i64 4, !"transport-status", i64 0, !"direct_dte_status", !"u32", !"tensor", i64 64, i64 64, !"read-write", i64 1, i64 0}

