//===- StaticLoopDomain.h - Structured iteration domains ---------------===//

#ifndef WAFER_ANALYSIS_CONTROLFLOW_STATICLOOPDOMAIN_H
#define WAFER_ANALYSIS_CONTROLFLOW_STATICLOOPDOMAIN_H

#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>

namespace wafer::analysis {

/// A read-only view of one current loop. Handles belong to the queried IR
/// epoch.
struct StaticLoopDomain {
  mlir::scf::ForOp loop;
  int64_t lower, upper, step;
  auto bounds() const { return std::tie(lower, upper, step); }
};

inline std::optional<StaticLoopDomain>
getStaticLoopDomain(mlir::Operation *operation) {
  auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(operation);
  if (!loop)
    return std::nullopt;
  auto lower = mlir::getConstantIntValue(loop.getLowerBound());
  auto upper = mlir::getConstantIntValue(loop.getUpperBound());
  auto step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *lower < 0 || *upper <= *lower ||
      *step <= 0 || !loop.getInductionVar().getType().isIndex())
    return std::nullopt;
  return StaticLoopDomain{loop, *lower, *upper, *step};
}

/// Repeated execution is accepted only through nonempty static loops and
/// otherwise transparent single-execution regions, up to the current function.
inline std::optional<llvm::SmallVector<StaticLoopDomain, 4>>
getEnclosingStaticLoopDomains(mlir::Operation *operation) {
  llvm::SmallVector<StaticLoopDomain, 4> loops;
  for (auto *parent = operation->getParentOp(); parent;
       operation = parent, parent = parent->getParentOp()) {
    if (mlir::isa<mlir::func::FuncOp>(parent)) {
      std::reverse(loops.begin(), loops.end());
      return loops;
    }
    if (auto loop = getStaticLoopDomain(parent)) {
      loops.push_back(*loop);
      continue;
    }
    auto flow = getSingleExecutionRegionFlow(parent);
    if (!flow || operation->getParentRegion() != flow->region)
      return std::nullopt;
  }
  return std::nullopt;
}

inline bool haveSameStaticLoopDomains(mlir::Operation *lhs,
                                      mlir::Operation *rhs) {
  auto a = getEnclosingStaticLoopDomains(lhs);
  auto b = getEnclosingStaticLoopDomains(rhs);
  return a && b && a->size() == b->size() &&
         llvm::all_of(llvm::zip_equal(*a, *b), [](const auto &pair) {
           return std::get<0>(pair).bounds() == std::get<1>(pair).bounds();
         });
}

} // namespace wafer::analysis
#endif
