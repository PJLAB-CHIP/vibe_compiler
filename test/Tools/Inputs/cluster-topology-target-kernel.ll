target triple = "riscv64-unknown-unknown-elf"

declare i32 @__get_pid(i32)
declare i32 @init_tile_id(i32, i32)

define void @kernel_entry(ptr %slots) {
entry:
  %pid = call i32 @__get_pid(i32 0)
  %status = call i32 @init_tile_id(i32 %pid, i32 4)
  ret void
}
