//===- PhysicalRepresentation.h - Selected value layouts ----*- C++ -*-===//

#pragma once

#include "Wafer/Planning/Search/CoupledRegion.h"

#include <algorithm>
#include <tuple>

namespace wafer::compiler::detail {

enum class PhysicalValueRole : uint8_t { Operand, Result };

struct PhysicalRepresentationChoice {
  TileId tile{0};
  StructuredDAGNodeID node = 0;
  PhysicalValueRole role = PhysicalValueRole::Operand;
  unsigned index = 0;
  MemLayout layout = MemLayout::Tensor;

  friend bool operator==(const PhysicalRepresentationChoice &lhs,
                         const PhysicalRepresentationChoice &rhs) {
    return lhs.tile == rhs.tile && lhs.node == rhs.node &&
           lhs.role == rhs.role && lhs.index == rhs.index &&
           lhs.layout == rhs.layout;
  }
  friend bool operator<(const PhysicalRepresentationChoice &lhs,
                        const PhysicalRepresentationChoice &rhs) {
    return std::tuple(lhs.tile.getValue(), lhs.node,
                      static_cast<uint8_t>(lhs.role), lhs.index,
                      static_cast<uint32_t>(lhs.layout)) <
           std::tuple(rhs.tile.getValue(), rhs.node,
                      static_cast<uint8_t>(rhs.role), rhs.index,
                      static_cast<uint32_t>(rhs.layout));
  }
};

struct CardPhysicalRepresentationAssignment {
  llvm::SmallVector<PhysicalRepresentationChoice, 32> values;

  friend bool operator==(const CardPhysicalRepresentationAssignment &lhs,
                         const CardPhysicalRepresentationAssignment &rhs) {
    return lhs.values == rhs.values;
  }
  friend bool operator<(const CardPhysicalRepresentationAssignment &lhs,
                        const CardPhysicalRepresentationAssignment &rhs) {
    return std::lexicographical_compare(lhs.values.begin(), lhs.values.end(),
                                        rhs.values.begin(), rhs.values.end());
  }
};

/// Lazy Cartesian domain for the primary physical version of every shaped
/// structured operand/result in each selected node/Tile shard. Legal layouts
/// come from the current physical encoding interface for the actual temporal
/// leaf shape; no local cost or preferred layout is selected here.
class CardPhysicalRepresentationDomain {
public:
  static mlir::FailureOr<CardPhysicalRepresentationDomain>
  create(const CardProgramAnalysis &program,
         const analysis::LogicalShardTrial &trial,
         const CoupledRegionDomain &coupledDomain,
         const CoupledRegionAssignment &coupledAssignment,
         const CardTemporalDomain &temporalDomain,
         const CardTemporalAssignment &temporalAssignment,
         std::string *failureReason = nullptr);

  CardPhysicalRepresentationAssignment getFirstAssignment() const;
  mlir::FailureOr<std::optional<CardPhysicalRepresentationAssignment>>
  getNextAssignment(
      const CardPhysicalRepresentationAssignment &assignment) const;
  bool contains(const CardPhysicalRepresentationAssignment &assignment) const;

private:
  struct ValueDomain {
    TileId tile{0};
    StructuredDAGNodeID node = 0;
    PhysicalValueRole role = PhysicalValueRole::Operand;
    unsigned index = 0;
    llvm::SmallVector<MemLayout, 4> layouts;
  };

  explicit CardPhysicalRepresentationDomain(
      llvm::SmallVector<ValueDomain, 32> values)
      : values(std::move(values)) {}

  llvm::SmallVector<ValueDomain, 32> values;
};

} // namespace wafer::compiler::detail
