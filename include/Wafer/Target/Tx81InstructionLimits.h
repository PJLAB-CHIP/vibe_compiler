//===- Tx81InstructionLimits.h - TX81 instruction limits -------*- C++ -*-===//

#ifndef WAFER_TARGET_TX81INSTRUCTIONLIMITS_H
#define WAFER_TARGET_TX81INSTRUCTIONLIMITS_H

#include <cstdint>

namespace wafer {

/// Semantic bounds of fields encoded by TX81 instruction registers.
struct Tx81InstructionLimits {
  static constexpr uint32_t dataShapeOuterMax = UINT32_C(4096);
  static constexpr uint32_t dataShapeChannelMax = UINT32_C(16384);
  static constexpr uint32_t paddingMax = UINT32_C(1023);
  static constexpr uint32_t kernelMax = UINT32_C(255);
  static constexpr uint32_t strideMax = UINT32_C(1023);
  static constexpr uint32_t dilationMax = UINT32_C(1023);
  static constexpr uint32_t gemmKMax = UINT32_C(16384);
  static constexpr uint32_t gemmBatchMax = UINT32_C(4096);
};

} // namespace wafer

#endif // WAFER_TARGET_TX81INSTRUCTIONLIMITS_H
