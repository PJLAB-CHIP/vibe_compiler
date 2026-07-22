target triple = "riscv64-unknown-unknown-elf"

declare i32 @__get_pid(i32)

define void @kernel_entry(ptr %slots) {
entry:
  %pid = call i32 @__get_pid(i32 0)
  %selected = getelementptr inbounds i64, ptr %slots, i32 %pid
  %address = load i64, ptr %selected, align 8
  ret void
}
