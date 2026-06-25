# Wafer CRT

This directory is the default repo-local root for Wafer CRT device libraries
used by `tools/wafer_device_link.py`.

Expected layout:

```text
third_party/wafer_crt/
  lib/
    libvr.a
```

`libvr.a` is not part of the TX8 dependency bundle.  The vendored copy in this
directory is the default Wafer CRT archive used by device-code execute-mode
linking.  It is debug-stripped so the vendored Xuantie GNU ld 2.35 does not see
LLVM-generated RISC-V debug relocations that it cannot decode.  An explicit
`--wafer-crt-lib-dir` override is still available for local debug setups that
need a different CRT build.
