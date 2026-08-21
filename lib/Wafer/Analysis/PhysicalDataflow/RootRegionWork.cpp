//===- RootRegionWork.cpp - Derived single-root Tile work -------------===//

#include "Wafer/Analysis/PhysicalDataflow/RootRegionWork.h"

namespace wafer::analysis {

const RootRegionWork *getRootRegionWork(const RootRegionWorkOutcome &outcome) {
  return std::get_if<RootRegionWork>(&outcome);
}

} // namespace wafer::analysis
