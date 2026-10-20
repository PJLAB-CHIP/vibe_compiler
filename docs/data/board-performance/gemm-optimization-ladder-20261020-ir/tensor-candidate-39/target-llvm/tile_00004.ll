; ModuleID = 'wafer.target.card.0.tile.00004.launch.00004'
source_filename = "wafer_target_physical_tile"
target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_gather_scatter(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_gemm(i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_memset(i64, i32, i32, i32, i32)

declare void @wafer_tx81_ncc_join(i32)

declare void @wafer_tx81_rdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_wdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

define void @entry(i64 %0, i64 %1, i64 %2, i64 %3) {
  %5 = add i64 %2, 8388608
  %6 = add i64 %1, 2097152
  call void @wafer_tx81_rdma(i64 %6, i64 65536, i32 524288, i32 524288, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 3, i32 0)
  br label %7

7:                                                ; preds = %23, %4
  %8 = phi i64 [ %29, %23 ], [ 0, %4 ]
  %9 = icmp slt i64 %8, 4096
  br i1 %9, label %10, label %30

10:                                               ; preds = %7
  call void @wafer_tx81_memset(i64 589824, i32 0, i32 131072, i32 5, i32 0)
  br label %11

11:                                               ; preds = %15, %10
  %12 = phi i64 [ %22, %15 ], [ 0, %10 ]
  %13 = phi i64 [ %13, %15 ], [ 589824, %10 ]
  %14 = icmp slt i64 %12, 896
  br i1 %14, label %15, label %23

15:                                               ; preds = %11
  %16 = mul i64 %12, 8192
  %17 = add i64 %0, %16
  %18 = mul i64 %8, 2
  %19 = add i64 %17, %18
  call void @wafer_tx81_rdma(i64 %19, i64 1638400, i32 131072, i32 1024, i32 8192, i32 0, i32 0, i32 128, i32 1, i32 1, i32 3, i32 0)
  %20 = mul i64 %12, 2
  %21 = add i64 65536, %20
  call void @wafer_tx81_gather_scatter(i64 %21, i64 1900544, i32 65536, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 2, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 2, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1638400, i64 1769472, i32 131072, i32 128, i32 1024, i32 128, i32 0, i32 128, i32 8, i32 1, i32 128, i32 16384, i32 0, i32 128, i32 8, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 1900544, i64 1769472, i64 1114112, i64 %13, i32 256, i32 128, i32 512, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1114112, i64 %13, i32 524288, i32 524288, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %22 = add i64 %12, 128
  br label %11

23:                                               ; preds = %11
  %24 = add i64 %0, 7340032
  %25 = mul i64 %8, 2
  %26 = add i64 %24, %25
  call void @wafer_tx81_rdma(i64 %26, i64 1638400, i32 131072, i32 1024, i32 8192, i32 0, i32 0, i32 128, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 67328, i64 1507328, i32 65536, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 2, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 2, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1638400, i64 1376256, i32 131072, i32 128, i32 1024, i32 128, i32 0, i32 128, i32 8, i32 1, i32 128, i32 16384, i32 0, i32 128, i32 8, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 1507328, i64 1376256, i64 1114112, i64 %13, i32 256, i32 128, i32 512, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1114112, i64 1572864, i32 262144, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 8, i32 1, i32 1024, i32 128, i32 0, i32 256, i32 8, i32 1, i32 0)
  %27 = mul i64 %8, 2
  %28 = add i64 %5, %27
  call void @wafer_tx81_wdma(i64 1572864, i64 %28, i32 262144, i32 1024, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %29 = add i64 %8, 512
  br label %7

30:                                               ; preds = %7
  call void @wafer_tx81_ncc_join(i32 1)
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
!wafer.target.entry_arguments = !{!7, !8, !9, !10}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i64 0}
!2 = !{i64 4}
!3 = !{!"entry"}
!4 = !{!"wafer-tx81-single-card"}
!5 = !{!"wafer-tx81-kernel"}
!6 = !{!"elf-riscv64"}
!7 = !{i64 0, !"external-input", i64 0, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 1024, i64 4096, i64 0}
!8 = !{i64 1, !"external-input", i64 1, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 4096, i64 1024, i64 0}
!9 = !{i64 2, !"external-output", i64 0, !"", !"bf16", !"tensor", i64 33554432, i64 256, !"write-only", i64 1, i64 4096, i64 4096, i64 0}
!10 = !{i64 3, !"workspace", i64 0, !"default_ddr_arena", !"u8", !"tensor", i64 256, i64 256, !"read-write", i64 256, i64 0}

