; ModuleID = 'wafer.target.card.0.tile.00003.launch.00003'
source_filename = "wafer_target_physical_tile"
target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_gather_scatter(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_gemm(i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_memset(i64, i32, i32, i32, i32)

declare void @wafer_tx81_ncc_join(i32)

declare void @wafer_tx81_rdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @wafer_tx81_wdma(i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

define void @entry(i64 %0, i64 %1, i64 %2, i64 %3) {
  %5 = add i64 %2, 6144
  br label %6

6:                                                ; preds = %42, %4
  %7 = phi i64 [ %43, %42 ], [ 0, %4 ]
  %8 = icmp slt i64 %7, 1024
  br i1 %8, label %9, label %44

9:                                                ; preds = %6
  br label %10

10:                                               ; preds = %29, %9
  %11 = phi i64 [ %41, %29 ], [ 0, %9 ]
  %12 = icmp slt i64 %11, 1024
  br i1 %12, label %13, label %42

13:                                               ; preds = %10
  call void @wafer_tx81_memset(i64 65536, i32 0, i32 65536, i32 5, i32 0)
  br label %14

14:                                               ; preds = %18, %13
  %15 = phi i64 [ %28, %18 ], [ 0, %13 ]
  %16 = phi i64 [ %16, %18 ], [ 65536, %13 ]
  %17 = icmp slt i64 %15, 768
  br i1 %17, label %18, label %29

18:                                               ; preds = %14
  %19 = mul i64 %7, 2048
  %20 = add i64 %1, %19
  %21 = mul i64 %15, 2
  %22 = add i64 %20, %21
  call void @wafer_tx81_rdma(i64 %22, i64 327680, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %23 = add i64 %11, 3072
  %24 = mul i64 %15, 8192
  %25 = add i64 %0, %24
  %26 = mul i64 %23, 2
  %27 = add i64 %25, %26
  call void @wafer_tx81_rdma(i64 %27, i64 589824, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 327680, i64 720896, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 589824, i64 851968, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 720896, i64 851968, i64 327680, i64 %16, i32 256, i32 256, i32 256, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 327680, i64 %16, i32 262144, i32 262144, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %28 = add i64 %15, 256
  br label %14

29:                                               ; preds = %14
  %30 = mul i64 %7, 2048
  %31 = add i64 %1, %30
  %32 = add i64 %31, 1536
  call void @wafer_tx81_rdma(i64 %32, i64 589824, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %33 = add i64 %11, 3072
  %34 = add i64 %0, 6291456
  %35 = mul i64 %33, 2
  %36 = add i64 %34, %35
  call void @wafer_tx81_rdma(i64 %36, i64 720896, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 589824, i64 327680, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 458752, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 327680, i64 458752, i64 589824, i64 %16, i32 256, i32 256, i32 256, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 589824, i64 720896, i32 131072, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 0)
  %37 = mul i64 %7, 8192
  %38 = add i64 %5, %37
  %39 = mul i64 %11, 2
  %40 = add i64 %38, %39
  call void @wafer_tx81_wdma(i64 720896, i64 %40, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %41 = add i64 %11, 256
  br label %10

42:                                               ; preds = %10
  %43 = add i64 %7, 256
  br label %6

44:                                               ; preds = %6
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
!2 = !{i64 3}
!3 = !{!"entry"}
!4 = !{!"wafer-tx81-single-card"}
!5 = !{!"wafer-tx81-kernel"}
!6 = !{!"elf-riscv64"}
!7 = !{i64 0, !"external-input", i64 0, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 1024, i64 4096, i64 0}
!8 = !{i64 1, !"external-input", i64 1, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 4096, i64 1024, i64 0}
!9 = !{i64 2, !"external-output", i64 0, !"", !"bf16", !"tensor", i64 33554432, i64 256, !"write-only", i64 1, i64 4096, i64 4096, i64 0}
!10 = !{i64 3, !"workspace", i64 0, !"default_ddr_arena", !"u8", !"tensor", i64 256, i64 256, !"read-write", i64 256, i64 0}

