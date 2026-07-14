target triple = "riscv64-unknown-unknown-elf"

@data_entry = global i64 0

define internal i32 @local_entry() #0 {
entry:
  ret i32 1
}

define i32 @kernel_entry(i64 %input0, i64 %output0) {
entry:
  %value = load volatile i64, ptr @data_entry
  %loaded = trunc i64 %value to i32
  %local = call i32 @local_entry()
  %result = add i32 %loaded, %local
  ret i32 %result
}

attributes #0 = { noinline optnone }
