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
  br label %6

6:                                                ; preds = %46, %5
  %7 = phi i64 [ %47, %46 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, 1024
  br i1 %8, label %9, label %48

9:                                                ; preds = %6
  br label %10

10:                                               ; preds = %31, %9
  %11 = phi i64 [ %45, %31 ], [ 0, %9 ]
  %12 = icmp slt i64 %11, 1024
  br i1 %12, label %13, label %46

13:                                               ; preds = %10
  call void @wafer_tx81_memset(i64 65536, i32 0, i32 65536, i32 5, i32 0)
  br label %14

14:                                               ; preds = %18, %13
  %15 = phi i64 [ %30, %18 ], [ 0, %13 ]
  %16 = phi i64 [ %16, %18 ], [ 65536, %13 ]
  %17 = icmp slt i64 %15, 768
  br i1 %17, label %18, label %31

18:                                               ; preds = %14
  %19 = mul i64 %7, 2048
  %20 = add i64 %1, %19
  %21 = mul i64 %15, 2
  %22 = add i64 %20, %21
  call void @wafer_tx81_rdma(i64 %22, i64 720896, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %23 = mul i64 %15, 8192
  %24 = add i64 %0, %23
  %25 = mul i64 %11, 2
  %26 = add i64 %24, %25
  call void @wafer_tx81_rdma(i64 %26, i64 327680, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 458752, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 327680, i64 589824, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 458752, i64 589824, i64 720896, i64 %16, i32 256, i32 256, i32 256, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 %16, i32 262144, i32 262144, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %27 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 589824, i32 131072, i32 0, i32 4, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %27)
  call void @wafer_tx81_direct_dte_wait(i64 %27)
  %28 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 589824, i32 131072, i32 0, i32 8, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %28)
  call void @wafer_tx81_direct_dte_wait(i64 %28)
  %29 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 589824, i32 131072, i32 0, i32 12, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %29)
  call void @wafer_tx81_direct_dte_wait(i64 %29)
  %30 = add i64 %15, 256
  br label %14

31:                                               ; preds = %14
  %32 = mul i64 %7, 2048
  %33 = add i64 %1, %32
  %34 = add i64 %33, 1536
  call void @wafer_tx81_rdma(i64 %34, i64 720896, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %35 = add i64 %0, 6291456
  %36 = mul i64 %11, 2
  %37 = add i64 %35, %36
  call void @wafer_tx81_rdma(i64 %37, i64 327680, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 458752, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 327680, i64 589824, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 458752, i64 589824, i64 720896, i64 %16, i32 256, i32 256, i32 256, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 851968, i32 131072, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 0)
  %38 = mul i64 %7, 8192
  %39 = add i64 %2, %38
  %40 = mul i64 %11, 2
  %41 = add i64 %39, %40
  call void @wafer_tx81_wdma(i64 851968, i64 %41, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %42 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 720896, i32 131072, i32 0, i32 4, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %42)
  call void @wafer_tx81_direct_dte_wait(i64 %42)
  %43 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 720896, i32 131072, i32 0, i32 8, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %43)
  call void @wafer_tx81_direct_dte_wait(i64 %43)
  %44 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 720896, i32 131072, i32 0, i32 12, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %44)
  call void @wafer_tx81_direct_dte_wait(i64 %44)
  %45 = add i64 %11, 256
  br label %10

46:                                               ; preds = %10
  %47 = add i64 %7, 256
  br label %6

48:                                               ; preds = %6
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

