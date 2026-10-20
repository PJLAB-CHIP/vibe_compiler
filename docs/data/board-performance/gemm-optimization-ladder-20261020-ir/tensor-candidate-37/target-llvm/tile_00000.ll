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
  br label %5

5:                                                ; preds = %39, %4
  %6 = phi i64 [ %40, %39 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 1024
  br i1 %7, label %8, label %41

8:                                                ; preds = %5
  br label %9

9:                                                ; preds = %27, %8
  %10 = phi i64 [ %38, %27 ], [ 0, %8 ]
  %11 = icmp slt i64 %10, 1024
  br i1 %11, label %12, label %39

12:                                               ; preds = %9
  call void @wafer_tx81_memset(i64 65536, i32 0, i32 65536, i32 5, i32 0)
  br label %13

13:                                               ; preds = %17, %12
  %14 = phi i64 [ %26, %17 ], [ 0, %12 ]
  %15 = phi i64 [ %15, %17 ], [ 65536, %12 ]
  %16 = icmp slt i64 %14, 768
  br i1 %16, label %17, label %27

17:                                               ; preds = %13
  %18 = mul i64 %6, 2048
  %19 = add i64 %1, %18
  %20 = mul i64 %14, 2
  %21 = add i64 %19, %20
  call void @wafer_tx81_rdma(i64 %21, i64 327680, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %22 = mul i64 %14, 8192
  %23 = add i64 %0, %22
  %24 = mul i64 %10, 2
  %25 = add i64 %23, %24
  call void @wafer_tx81_rdma(i64 %25, i64 589824, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 327680, i64 720896, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 589824, i64 851968, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 720896, i64 851968, i64 327680, i64 %15, i32 256, i32 256, i32 256, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 327680, i64 %15, i32 262144, i32 262144, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %26 = add i64 %14, 256
  br label %13

27:                                               ; preds = %13
  %28 = mul i64 %6, 2048
  %29 = add i64 %1, %28
  %30 = add i64 %29, 1536
  call void @wafer_tx81_rdma(i64 %30, i64 589824, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %31 = add i64 %0, 6291456
  %32 = mul i64 %10, 2
  %33 = add i64 %31, %32
  call void @wafer_tx81_rdma(i64 %33, i64 720896, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 589824, i64 327680, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 458752, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 327680, i64 458752, i64 589824, i64 %15, i32 256, i32 256, i32 256, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 589824, i64 720896, i32 131072, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 0)
  %34 = mul i64 %6, 8192
  %35 = add i64 %2, %34
  %36 = mul i64 %10, 2
  %37 = add i64 %35, %36
  call void @wafer_tx81_wdma(i64 720896, i64 %37, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %38 = add i64 %10, 256
  br label %9

39:                                               ; preds = %9
  %40 = add i64 %6, 256
  br label %5

41:                                               ; preds = %5
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

