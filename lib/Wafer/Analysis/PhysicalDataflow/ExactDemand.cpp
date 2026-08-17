//===- ExactDemand.cpp - Policy-free logical placement demand -------------===//

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"

#include <atomic>

namespace wafer::analysis {
namespace {

std::atomic<uint64_t> &getProcessGeneration() {
  static std::atomic<uint64_t> generation{1};
  return generation;
}

} // namespace

IREpoch IREpoch::mint() {
  // The process counter provides token uniqueness only; it is not an
  // invalidation mechanism and never enters a cache key or legality
  // decision.
  return IREpoch(getProcessGeneration().fetch_add(1, std::memory_order_relaxed) +
                 1);
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
