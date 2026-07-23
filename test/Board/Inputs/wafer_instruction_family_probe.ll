target triple = "riscv64-unknown-unknown-elf"

declare void @wafer_tx81_instruction_family_probe(i64, i64, i64)

define void @main(ptr %slots) {
entry:
  %request_slot = getelementptr inbounds i64, ptr %slots, i64 0
  %request = load i64, ptr %request_slot, align 8
  %payload_slot = getelementptr inbounds i64, ptr %slots, i64 1
  %payload = load i64, ptr %payload_slot, align 8
  %output_slot = getelementptr inbounds i64, ptr %slots, i64 2
  %output = load i64, ptr %output_slot, align 8
  call void @wafer_tx81_instruction_family_probe(
      i64 %request, i64 %payload, i64 %output)
  ret void
}
