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
  br label %6

6:                                                ; preds = %23, %4
  %7 = phi i64 [ %31, %23 ], [ 0, %4 ]
  %8 = icmp slt i64 %7, 4096
  br i1 %8, label %9, label %32

9:                                                ; preds = %6
  call void @wafer_tx81_memset(i64 65536, i32 0, i32 131072, i32 5, i32 0)
  br label %10

10:                                               ; preds = %14, %9
  %11 = phi i64 [ %22, %14 ], [ 0, %9 ]
  %12 = phi i64 [ %12, %14 ], [ 65536, %9 ]
  %13 = icmp slt i64 %11, 896
  br i1 %13, label %14, label %23

14:                                               ; preds = %10
  %15 = add i64 %1, 2097152
  %16 = mul i64 %11, 2
  %17 = add i64 %15, %16
  call void @wafer_tx81_rdma(i64 %17, i64 589824, i32 65536, i32 256, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %18 = mul i64 %11, 8192
  %19 = add i64 %0, %18
  %20 = mul i64 %7, 2
  %21 = add i64 %19, %20
  call void @wafer_tx81_rdma(i64 %21, i64 1114112, i32 131072, i32 1024, i32 8192, i32 0, i32 0, i32 128, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 589824, i64 1376256, i32 65536, i32 128, i32 256, i32 128, i32 0, i32 256, i32 2, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 2, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1114112, i64 1245184, i32 131072, i32 128, i32 1024, i32 128, i32 0, i32 128, i32 8, i32 1, i32 128, i32 16384, i32 0, i32 128, i32 8, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 1376256, i64 1245184, i64 589824, i64 %12, i32 256, i32 128, i32 512, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 589824, i64 %12, i32 524288, i32 524288, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %22 = add i64 %11, 128
  br label %10

23:                                               ; preds = %10
  %24 = add i64 %1, 2097152
  %25 = add i64 %24, 1792
  call void @wafer_tx81_rdma(i64 %25, i64 1179648, i32 65536, i32 256, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %26 = add i64 %0, 7340032
  %27 = mul i64 %7, 2
  %28 = add i64 %26, %27
  call void @wafer_tx81_rdma(i64 %28, i64 1048576, i32 131072, i32 1024, i32 8192, i32 0, i32 0, i32 128, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1179648, i64 720896, i32 65536, i32 128, i32 256, i32 128, i32 0, i32 256, i32 2, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 2, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 1048576, i64 589824, i32 131072, i32 128, i32 1024, i32 128, i32 0, i32 128, i32 8, i32 1, i32 128, i32 16384, i32 0, i32 128, i32 8, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 720896, i64 589824, i64 786432, i64 %12, i32 256, i32 128, i32 512, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 786432, i64 1048576, i32 262144, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 8, i32 1, i32 1024, i32 128, i32 0, i32 256, i32 8, i32 1, i32 0)
  %29 = mul i64 %7, 2
  %30 = add i64 %5, %29
  call void @wafer_tx81_wdma(i64 1048576, i64 %30, i32 262144, i32 1024, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %31 = add i64 %7, 512
  br label %6

32:                                               ; preds = %6
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

