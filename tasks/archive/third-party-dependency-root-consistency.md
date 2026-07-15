# Third-Party Dependency Root 一致性实施计划

状态：historical/completed（2026-07-15）。本计划对应Q13.W，只修复正式SystemC feature-on构建使用一次性`build/`
依赖root的workflow偏差；numeric、bulk及其它第三方依赖保持现有正确布局。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  固定的SystemC 3.0.2版本/archive digest、tools/bootstrap_deps.py producer、third_party默认依赖root和已完成Q22.S模型实现。
- Current stage responsibility:
  在third_party/systemc-model重新生成并验证canonical dependency tree，令正式feature-on build只消费该root。
- Output artifact / IR:
  third_party/systemc-model下被Git忽略的source/build/install/conformance tree与canonical record，以及消费它的build-local snapshot；
  不产生或修改compiler IR/package。
- Downstream consumer:
  SystemC::systemc imported target、WaferSystemCBridge、WaferSystemCModel、component tests和wafer-compile target-model mode。
- User-level driver / named pipeline:
  tools/bootstrap_deps.py --systemc-model-deps；WAFER_ENABLE_SYSTEMC_MODEL=ON的CMake配置与check/CTest入口。
- Explicit non-goals:
  不vendor生成物进Git，不移动Wafer-owned model源码，不重写SystemC/numeric/event语义，不改变其它依赖root。
- Completion gate:
  默认root bootstrap和record validator通过；feature-on cache/readback指向third_party/systemc-model；build、SystemC component、
  完整CTest和dependency CMake gates通过；旧build/wafer-systemc-deps不再被引用并被清理。
```

## 完成记录

- 默认bootstrap在`third_party/systemc-model`重新下载、校验、构建并静态安装SystemC 3.0.2，独立consumer的两个
  `SC_THREAD` delta-event smoke通过，canonical record由validator readback通过。
- existing feature-on与独立SystemC build cache均切换到该root；generated snapshot、`SystemCLanguage_DIR`、root和record
  不再引用旧目录。发现并修复config-mode `find_package`复用旧`SystemCLanguage_DIR` cache的问题。
- 配置gate新增valid-looking stale package cache正例，连同四项fail-closed负例通过；五个SystemC executable通过。
- feature-on build与CTest 22/22通过，lit 252项为250 pass/2个预期unsupported；feature-off build与CTest 12/12通过，
  lit为249 pass/3个预期unsupported。旧`build/wafer-systemc-deps`已无引用并清理。

本任务只收敛依赖目录和CMake discovery，不改变Q22已发布的IR、ABI、numeric、transaction或event语义。
