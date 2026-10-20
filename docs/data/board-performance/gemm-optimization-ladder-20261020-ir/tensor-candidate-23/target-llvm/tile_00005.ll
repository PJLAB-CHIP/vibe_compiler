; ModuleID = 'wafer.target.card.0.tile.00005.launch.00005'
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
  %6 = add i64 %2, 8390656
  br label %7

7:                                                ; preds = %40, %5
  %8 = phi i64 [ %41, %40 ], [ 0, %5 ]
  %9 = icmp slt i64 %8, 1024
  br i1 %9, label %10, label %42

10:                                               ; preds = %7
  br label %11

11:                                               ; preds = %28, %10
  %12 = phi i64 [ %39, %28 ], [ 0, %10 ]
  %13 = icmp slt i64 %12, 1024
  br i1 %13, label %14, label %40

14:                                               ; preds = %11
  call void @wafer_tx81_memset(i64 65536, i32 0, i32 65536, i32 5, i32 0)
  br label %15

15:                                               ; preds = %19, %14
  %16 = phi i64 [ %27, %19 ], [ 0, %14 ]
  %17 = phi i64 [ %17, %19 ], [ 65536, %14 ]
  %18 = icmp slt i64 %16, 768
  br i1 %18, label %19, label %28

19:                                               ; preds = %15
  %20 = add i64 %8, 1024
  %21 = mul i64 %20, 2048
  %22 = add i64 %1, %21
  %23 = mul i64 %16, 2
  %24 = add i64 %22, %23
  call void @wafer_tx81_rdma(i64 %24, i64 327680, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %25 = add i64 %12, 1024
  call void @wafer_tx81_gather_scatter(i64 327680, i64 720896, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %26 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 589824, i32 131072, i32 5, i32 1, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %26)
  call void @wafer_tx81_gather_scatter(i64 589824, i64 851968, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 720896, i64 851968, i64 327680, i64 %17, i32 256, i32 256, i32 256, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 327680, i64 %17, i32 262144, i32 262144, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %27 = add i64 %16, 256
  br label %15

28:                                               ; preds = %15
  %29 = add i64 %8, 1024
  %30 = mul i64 %29, 2048
  %31 = add i64 %1, %30
  %32 = add i64 %31, 1536
  call void @wafer_tx81_rdma(i64 %32, i64 589824, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %33 = add i64 %12, 1024
  call void @wafer_tx81_gather_scatter(i64 589824, i64 327680, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %34 = call i64 @wafer_tx81_direct_dte_recv_prepare(i64 720896, i32 131072, i32 5, i32 1, i32 0)
  call void @wafer_tx81_direct_dte_wait(i64 %34)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 458752, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 327680, i64 458752, i64 589824, i64 %17, i32 256, i32 256, i32 256, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 589824, i64 720896, i32 131072, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 0)
  %35 = mul i64 %8, 8192
  %36 = add i64 %6, %35
  %37 = mul i64 %12, 2
  %38 = add i64 %36, %37
  call void @wafer_tx81_wdma(i64 720896, i64 %38, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %39 = add i64 %12, 256
  br label %11

40:                                               ; preds = %11
  %41 = add i64 %8, 256
  br label %7

42:                                               ; preds = %7
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

