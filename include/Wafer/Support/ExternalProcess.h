//===- ExternalProcess.h - Bounded compiler subprocesses -------*- C++ -*-===//

#ifndef WAFER_SUPPORT_EXTERNALPROCESS_H
#define WAFER_SUPPORT_EXTERNALPROCESS_H

namespace wafer::support {

/// External helper/link invocations are part of one compiler transaction and
/// must not wait forever when a dependency hangs. The product driver already
/// uses the same upper bound for source compilation; keeping it here gives
/// library callers the same fail-closed behavior.
inline constexpr unsigned kExternalProcessTimeoutSeconds = 1800;

} // namespace wafer::support

#endif // WAFER_SUPPORT_EXTERNALPROCESS_H
