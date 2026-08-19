//===- WaferDialect.h - Wafer dialect declarations -------------*- C++ -*-===//

#ifndef WAFER_IR_WAFERDIALECT_H
#define WAFER_IR_WAFERDIALECT_H

#include "Wafer/Target/Layout/PhysicalLayout.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "Wafer/IR/WaferEnums.h.inc"
#include "Wafer/IR/WaferPhysicalEncodingInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "Wafer/IR/WaferAttrs.h.inc"

#include "Wafer/IR/WaferOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Wafer/IR/WaferTypes.h.inc"

#include "Wafer/IR/WaferInterfaces.h"

namespace wafer {

/// Reject attributes that are neither declared by the operation schema nor
/// explicitly owned by another dialect. This keeps instrumentation metadata
/// composable while preventing an unnamespaced spelling or an undeclared
/// `wafer.*` attribute from becoming a second semantic IR schema.
mlir::LogicalResult
verifyNoSchemaFreeSemanticAttributes(mlir::Operation *operation);

} // namespace wafer

namespace mlir::OpTrait::wafer {

template <typename ConcreteType>
class NoSchemaFreeSemanticAttrs
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                     NoSchemaFreeSemanticAttrs> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *operation) {
    return ::wafer::verifyNoSchemaFreeSemanticAttributes(operation);
  }
};

} // namespace mlir::OpTrait::wafer

#define GET_OP_CLASSES
#include "Wafer/IR/WaferOps.h.inc"

namespace wafer {

inline constexpr char kWaferSPMOffsetAttrName[] = "wafer.spm.offset";
inline constexpr char kWaferDDROffsetAttrName[] = "wafer.ddr.offset";
inline constexpr int64_t kWaferSPMBankLineBytes = 256;

/// Typed parameter contract shared by instruction verification and consumers
/// of an accepted convert instruction.
enum class InstrConvertParameterKind { None, RoundingMode, ZeroPoint };

std::pair<mlir::Type, mlir::Type>
getInstrConvertTypePair(mlir::MLIRContext *context, InstrConvertKind kind);
InstrConvertParameterKind getInstrConvertParameterKind(InstrConvertKind kind);

/// Maps the target ABI reduce dimension code to logical tensor dimensions.
/// An empty result means the code is not valid for the given input rank.
llvm::SmallVector<int64_t, 3> getInstrReduceLogicalDims(int64_t targetDim,
                                                        int64_t inputRank);

struct WaferPhysicalTensorInfo {
  mlir::RankedTensorType logicalTensorType;
  MemorySpace memorySpace;
  MemLayout layout;
  int64_t compactBytes = -1;
  int64_t physicalBytes = -1;
  int64_t physicalElements = -1;
  int64_t elementBytes = -1;
  int64_t cBlock = 0;
  int64_t cxBlocks = 0;
  int64_t c0 = 0;
  int64_t alignedC = -1;
  int64_t tailC = 0;
  int64_t outerElements = -1;
  int64_t hwElements = -1;
  int64_t batchElements = -1;
  int64_t bankAlignElements = -1;
  bool bitPackedElement = false;
};

/// Reusable static logical-coordinate to physical-byte calculator.  Creation
/// validates the memref geometry once and precomputes byte/blocked-layout
/// strides; repeated lowering and codec traversals can then reuse the same
/// physical mapping without rebuilding MLIR layout state per element.
class WaferStaticPhysicalOffsetCalculator {
public:
  static std::optional<WaferStaticPhysicalOffsetCalculator>
  create(mlir::MemRefType type);

  const WaferPhysicalTensorInfo &getInfo() const { return info; }

  /// Checked entry point for an arbitrary logical coordinate.
  std::optional<int64_t>
  getByteOffset(llvm::ArrayRef<int64_t> logicalIndices) const;

  /// Fast entry point for a coordinate already proven in-bounds by a static
  /// lexicographic traversal or verified op index relation.
  int64_t
  getByteOffsetForValidIndices(llvm::ArrayRef<int64_t> logicalIndices) const {
    std::optional<int64_t> bitOffset = calculator.getBitOffset(logicalIndices);
    assert(bitOffset && *bitOffset >= 0 && *bitOffset % 8 == 0);
    return *bitOffset / 8;
  }

private:
  WaferStaticPhysicalOffsetCalculator(
      WaferPhysicalTensorInfo info,
      StaticPhysicalTensorOffsetCalculator calculator)
      : info(std::move(info)), calculator(std::move(calculator)) {}

  WaferPhysicalTensorInfo info;
  StaticPhysicalTensorOffsetCalculator calculator;
};

bool isWaferMemRefType(mlir::Type type);
bool isWaferSPMMemRefType(mlir::Type type);
bool isWaferDDRMemRefType(mlir::Type type);
MemoryAttr getWaferMemoryAttr(mlir::MemRefType type);
std::optional<WaferPhysicalTensorInfo>
computeWaferPhysicalTensorInfo(mlir::MemRefType type);

/// Combine the physical encoding's natural alignment with caller-owned target,
/// arena, ABI, or allocation requirements using checked least-common-multiple
/// arithmetic. This is the common alignment projection of the physical
/// encoding contract; callers must not replace it with max() or divisibility
/// assumptions.
mlir::FailureOr<int64_t> computeWaferRequiredAlignmentBytes(
    mlir::MemRefType type, llvm::ArrayRef<int64_t> additionalRequirements = {});

std::optional<int64_t>
computeWaferPhysicalElementByteOffset(mlir::MemRefType type,
                                      llvm::ArrayRef<int64_t> logicalIndices);
std::optional<int64_t>
computeWaferPhysicalElementByteOffset(mlir::MemRefType type,
                                      const WaferPhysicalTensorInfo &info,
                                      llvm::ArrayRef<int64_t> logicalIndices);

/// Returns the physical bit ordinal of one logical element relative to the
/// memref view base. Byte-addressable elements reuse the byte-offset helper;
/// bitpacked i1 layouts return an element bit ordinal whose within-byte order
/// is selected separately by the consuming codec/profile. Cx/NCx use the same
/// encoding-owned block/tail/bank ordering as byte-addressable elements, with
/// lane ordinals interpreted as bit offsets.
std::optional<int64_t>
computeWaferPhysicalElementBitOffset(mlir::MemRefType type,
                                     llvm::ArrayRef<int64_t> logicalIndices);

} // namespace wafer

#endif // WAFER_IR_WAFERDIALECT_H
