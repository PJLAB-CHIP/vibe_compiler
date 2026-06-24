# Wafer CRT

This directory is the default repo-local root for Wafer CRT device libraries
used by `tools/wafer_device_link.py`.

Expected layout:

```text
third_party/wafer_crt/
  lib/
    libvr.a
```

`libvr.a` is not part of the TX8 dependency bundle.  Device-code execute-mode
linking intentionally fails until the Wafer CRT library is provided here or an
explicit `--wafer-crt-lib-dir` override is passed for a local debug setup.
