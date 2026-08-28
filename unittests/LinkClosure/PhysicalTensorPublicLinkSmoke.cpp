#include "Wafer/Target/PhysicalTensor/PhysicalLayout.h"

int main() {
  if (wafer::stringifyPhysicalTensorLayout(
          wafer::PhysicalTensorLayout::Tensor) != "tensor")
    return 1;
  return 0;
}
