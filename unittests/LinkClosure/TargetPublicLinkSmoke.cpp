#include "Wafer/Target/Core/NCCCompletion.h"
#include "Wafer/Target/Layout/PhysicalLayout.h"

int main() {
  static_assert(wafer::kTargetNCCWorkerCount == 3);
  if (wafer::stringifyPhysicalTensorLayout(
          wafer::PhysicalTensorLayout::Tensor) != "tensor")
    return 1;
  return 0;
}
