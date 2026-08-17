//===- ExactDemand.cpp - Policy-free logical placement demand -------------===//

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"

namespace wafer::analysis {

IREpoch IREpoch::mint() {
  // Allocation identity is owned by this value and its copies. It provides a
  // borrow-lifetime token without process-global mutable compiler state.
  return IREpoch(std::make_shared<const Token>());
}

ExactDemandStatus mapIndexRelationStatus(IndexRelationStatus status) {
  switch (status) {
  case IndexRelationStatus::Unsupported:
    return ExactDemandStatus::UnsupportedSemanticRelation;
  case IndexRelationStatus::Exact:
  case IndexRelationStatus::SoundBound:
  case IndexRelationStatus::Invalid:
  case IndexRelationStatus::ResourceExhausted:
    // An exact relation is not a failure (the caller continues the query).
    // SoundBound is an over-approximation that cannot form an exact proof,
    // and Invalid/ResourceExhausted are machinery failures; all of them stop
    // the owning legalization path as indeterminate instead of deleting a
    // placement trial.
    return ExactDemandStatus::IndeterminateFailure;
  }
  return ExactDemandStatus::IndeterminateFailure;
}

} // namespace wafer::analysis
