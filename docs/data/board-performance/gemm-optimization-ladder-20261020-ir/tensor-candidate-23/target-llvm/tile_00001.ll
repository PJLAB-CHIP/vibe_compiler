; ModuleID = 'wafer.target.card.0.tile.00001.launch.00001'
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
  %6 = add i64 %2, 2048
  br label %7

7:                                                ; preds = %49, %5
  %8 = phi i64 [ %50, %49 ], [ 0, %5 ]
  %9 = icmp slt i64 %8, 1024
  br i1 %9, label %10, label %51

10:                                               ; preds = %7
  br label %11

11:                                               ; preds = %33, %10
  %12 = phi i64 [ %48, %33 ], [ 0, %10 ]
  %13 = icmp slt i64 %12, 1024
  br i1 %13, label %14, label %49

14:                                               ; preds = %11
  call void @wafer_tx81_memset(i64 65536, i32 0, i32 65536, i32 5, i32 0)
  br label %15

15:                                               ; preds = %19, %14
  %16 = phi i64 [ %32, %19 ], [ 0, %14 ]
  %17 = phi i64 [ %17, %19 ], [ 65536, %14 ]
  %18 = icmp slt i64 %16, 768
  br i1 %18, label %19, label %33

19:                                               ; preds = %15
  %20 = mul i64 %8, 2048
  %21 = add i64 %1, %20
  %22 = mul i64 %16, 2
  %23 = add i64 %21, %22
  call void @wafer_tx81_rdma(i64 %23, i64 720896, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %24 = add i64 %12, 1024
  %25 = mul i64 %16, 8192
  %26 = add i64 %0, %25
  %27 = mul i64 %24, 2
  %28 = add i64 %26, %27
  call void @wafer_tx81_rdma(i64 %28, i64 327680, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 458752, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 327680, i64 589824, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 458752, i64 589824, i64 720896, i64 %17, i32 256, i32 256, i32 256, i32 1, i32 3, i32 5, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 %17, i32 262144, i32 262144, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %29 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 589824, i32 131072, i32 1, i32 5, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %29)
  call void @wafer_tx81_direct_dte_wait(i64 %29)
  %30 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 589824, i32 131072, i32 1, i32 9, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %30)
  call void @wafer_tx81_direct_dte_wait(i64 %30)
  %31 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 589824, i32 131072, i32 1, i32 13, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %31)
  call void @wafer_tx81_direct_dte_wait(i64 %31)
  %32 = add i64 %16, 256
  br label %15

33:                                               ; preds = %15
  %34 = mul i64 %8, 2048
  %35 = add i64 %1, %34
  %36 = add i64 %35, 1536
  call void @wafer_tx81_rdma(i64 %36, i64 720896, i32 131072, i32 512, i32 2048, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  %37 = add i64 %12, 1024
  %38 = add i64 %0, 6291456
  %39 = mul i64 %37, 2
  %40 = add i64 %38, %39
  call void @wafer_tx81_rdma(i64 %40, i64 327680, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 458752, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gather_scatter(i64 327680, i64 589824, i32 131072, i32 128, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 0)
  call void @wafer_tx81_gemm(i64 458752, i64 589824, i64 720896, i64 %17, i32 256, i32 256, i32 256, i32 1, i32 3, i32 3, i32 5, i32 0)
  call void @wafer_tx81_gather_scatter(i64 720896, i64 851968, i32 131072, i32 128, i32 128, i32 32768, i32 0, i32 256, i32 4, i32 1, i32 512, i32 128, i32 0, i32 256, i32 4, i32 1, i32 0)
  %41 = mul i64 %8, 8192
  %42 = add i64 %6, %41
  %43 = mul i64 %12, 2
  %44 = add i64 %42, %43
  call void @wafer_tx81_wdma(i64 851968, i64 %44, i32 131072, i32 512, i32 8192, i32 0, i32 0, i32 256, i32 1, i32 1, i32 3, i32 0)
  call void @wafer_tx81_ncc_join(i32 1)
  %45 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 720896, i32 131072, i32 1, i32 5, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %45)
  call void @wafer_tx81_direct_dte_wait(i64 %45)
  %46 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 720896, i32 131072, i32 1, i32 9, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %46)
  call void @wafer_tx81_direct_dte_wait(i64 %46)
  %47 = call i64 @wafer_tx81_direct_dte_send_prepare(i64 327680, i64 720896, i32 131072, i32 1, i32 13, i32 0, i32 0)
  call void @wafer_tx81_direct_dte_send_issue(i64 %47)
  call void @wafer_tx81_direct_dte_wait(i64 %47)
  %48 = add i64 %12, 256
  br label %11

49:                                               ; preds = %11
  %50 = add i64 %8, 256
  br label %7

51:                                               ; preds = %7
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
!2 = !{i64 1}
!3 = !{!"entry"}
!4 = !{!"wafer-tx81-single-card"}
!5 = !{!"wafer-tx81-kernel"}
!6 = !{!"elf-riscv64"}
!7 = !{i64 0, !"external-input", i64 0, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 1024, i64 4096, i64 0}
!8 = !{i64 1, !"external-input", i64 1, !"", !"bf16", !"tensor", i64 8388608, i64 256, !"read-only", i64 1, i64 4096, i64 1024, i64 0}
!9 = !{i64 2, !"external-output", i64 0, !"", !"bf16", !"tensor", i64 33554432, i64 256, !"write-only", i64 1, i64 4096, i64 4096, i64 0}
!10 = !{i64 3, !"workspace", i64 0, !"default_ddr_arena", !"u8", !"tensor", i64 256, i64 256, !"read-write", i64 256, i64 0}
!11 = !{i64 4, !"transport-status", i64 0, !"direct_dte_status", !"u32", !"tensor", i64 64, i64 64, !"read-write", i64 1, i64 0}

