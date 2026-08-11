//===- DistributedSupport.cpp - Distributed validation support --------===//

#include "ProgramInternal.h"

#include "Wafer/IR/WaferDialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

using namespace mlir;

namespace wafer::frontend::program_detail {

namespace {

bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

} // namespace

bool hasSpmdParameterShardings(ModuleOp module) {
  if (module->getAttr("mhlo.spmd_parameters_shardings"))
    return true;

  bool found = false;
  module.walk([&](Operation *op) {
    if (op->getAttr("mhlo.spmd_parameters_shardings")) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

wafer::frontend::ProgramDistributionKind
getVerifiedDistributionKind(llvm::StringRef distribution) {
  return distribution == "partitioned"
             ? wafer::frontend::ProgramDistributionKind::Partitioned
             : wafer::frontend::ProgramDistributionKind::Replicated;
}

FailureOr<int64_t>
getSingleExecutionMeshPartitionCount(ModuleOp module,
                                     llvm::raw_ostream &diagnostics) {
  SmallVector<wafer::ExecutionMeshOp, 2> meshes;
  for (wafer::ExecutionMeshOp mesh : module.getOps<wafer::ExecutionMeshOp>())
    meshes.push_back(mesh);
  bool nestedMesh = false;
  module.walk([&](wafer::ExecutionMeshOp mesh) {
    if (mesh->getParentOp() != module.getOperation()) {
      nestedMesh = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (nestedMesh) {
    rejectProgramDirectory(
        "post-SPMD execution mesh must be a direct module member", diagnostics);
    return failure();
  }
  if (meshes.size() != 1) {
    rejectProgramDirectory(
        "post-SPMD metadata requires exactly one wafer.execution.mesh",
        diagnostics);
    return failure();
  }

  int64_t partitionCount = 1;
  for (int64_t dim : meshes.front().getShapeAttr().asArrayRef()) {
    int64_t next = 0;
    if (dim <= 0 || !checkedMul(partitionCount, dim, next)) {
      rejectProgramDirectory("execution mesh partition count is invalid",
                             diagnostics);
      return failure();
    }
    partitionCount = next;
  }
  return partitionCount;
}

} // namespace wafer::frontend::program_detail
