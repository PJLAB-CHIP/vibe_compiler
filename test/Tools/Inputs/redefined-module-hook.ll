; Scalar negative fixture: reject ownership conflicts at the ELF ABI boundary.
target triple = "riscv64-unknown-unknown-elf"

define void @module_init() {
  ret void
}
