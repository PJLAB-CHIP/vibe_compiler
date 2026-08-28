#include "Wafer/Simulator/Memory/TargetModelMemory.h"

int main() {
  return wafer::model::stringifyTargetModelMemoryErrorCode(
             wafer::model::TargetModelMemoryErrorCode::InvalidInvocation) ==
                 "invalid-invocation"
             ? 0
             : 1;
}
