#include "Wafer/Target/Core/NCCCompletion.h"
#include "Wafer/Target/Core/TargetSchedulingCapability.h"
#include "Wafer/Target/Layout/PhysicalLayout.h"

#include "llvm/Support/Error.h"

int main() {
  static_assert(wafer::kTargetNCCWorkerCount == 3);
  if (wafer::stringifyPhysicalTensorLayout(
          wafer::PhysicalTensorLayout::Tensor) != "tensor")
    return 1;
  auto registry = wafer::getTargetSchedulingCapabilityRegistry();
  if (registry)
    return 0;
  llvm::consumeError(registry.takeError());
  return 1;
}
