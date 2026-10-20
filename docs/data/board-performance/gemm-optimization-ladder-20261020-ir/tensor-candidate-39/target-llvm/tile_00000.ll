; ModuleID = 'wafer.target.card.0.tile.00000.launch.00000'
source_filename = "wafer_target_physical_tile"
target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_gather_scatter(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_gemm(i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_memset(i64, i32, i32, i32, i32)

declare void @wafer_tx81_ncc_join(i32)

declare void @wafer_tx81_rdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_wdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

define void @entry(i64 %0, i64 %1, i64 %2, i64 %3) {
  call void @wafer_tx81_rdma(i64 %1, i64 65536, i32 524288, i32 524288, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 3, i32 0)
  br label %5

5:                                                ; preds = %21, %4
  %6 = phi i64 [ %27, %21 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 4096
  br i1 %7, label %8, label %28

8:                                                ; preds = %5
  call void @wafer_tx81_memset(i64 589824, i32 0, i32 131072, i32 5, i32 0)
  br label %9

9:                                                ; preds = %13, %8
  %10 = phi i64 [ %20, %13 ], [ 0, %8 ]
  %11 = phi i64 [ %11, %13 ], [ 589824, %8 ]
  %12 = icmp slt i64 %10, 896
  br i1 %12, label %13, label %21

13:                                               ; preds = %9
  %14 = mul i64 %10, 8192
  %15 = add i64 %0, %14
  %16 = mul i64 %6, 2
  %17 = add i64 %15, %16
  call void @wafer_tx81_rdma(i64 %17, i64 1638400, i32 131072, i32 1024, i32 8192, i32 0, i32 0, i32 128, i32 1, i32 1, i32 3, i32 0)
  %18 = mul i64 %10, 2
  %19 = add i64 65536, %18
  call void @wafer_tx81_gather_scatter(i64 %19, i64 1900544, i32 65536, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 2, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 2, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1638400, i64 1769472, i32 131072, i32 128, i32 1024, i32 128, i32 0, i32 128, i32 8, i32 1, i32 128, i32 16384, i32 0, i32 128, i32 8, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 1900544, i64 1769472, i64 1114112, i64 %11, i32 256, i32 128, i32 512, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1114112, i64 %11, i32 524288, i32 524288, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %20 = add i64 %10, 128
  br label %9

21:                                               ; preds = %9
  %22 = add i64 %0, 7340032
  %23 = mul i64 %6, 2
  %24 = add i64 %22, %23
  call void @wafer_tx81_rdma(i64 %24, i64 1638400, i32 131072, i32 1024, i32 8192, i32 0, i32 0, i32 128, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 67328, i64 1507328, i32 65536, i32 128, i32 2048, i32 128, i32 0, i32 256, i32 2, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 2, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1638400, i64 1376256, i32 131072, i32 128, i32 1024, i32 128, i32 0, i32 128, i32 8, i32 1, i32 128, i32 16384, i32 0, i32 128, i32 8, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 1507328, i64 1376256, i64 1114112, i64 %11, i32 256, i32 128, i32 512, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1114112, i64 1572864, i32 262144, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 8, i32 1, i32 1024, i32 128, i32 0, i32 256, i32 8, i32 1, i32 0)
  %25 = mul i64 %6, 2
  %26 = add i64 %2, %25
  call void @wafer_tx81_wdma(i64 1572864, i64 %26, i32 262144, i32 1024, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %27 = add i64 %6, 512
  br label %5

28:                                               ; preds = %5
  call void @wafer_tx81_ncc_join(i32 1)
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
!wafer.target.entry_arguments = !{!6, !7, !8, !9}

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

