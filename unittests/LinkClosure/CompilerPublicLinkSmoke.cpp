#include "Wafer/Compiler/Compilation.h"

#include "llvm/Support/Error.h"

int main() {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Kernel);
  if (config)
    return 0;
  llvm::consumeError(config.takeError());
  return 1;
}
