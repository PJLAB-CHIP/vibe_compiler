//===- Passes.cpp - Wafer transform pass registration --------------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace wafer {

std::unique_ptr<mlir::Pass> createFunctionBoundaryBufferizationPass() {
  mlir::bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.allowReturnAllocsFromLoops = true;
  options.inferFunctionResultLayout = true;
  options.bufferAlignment = 64;
  options.defaultMemorySpaceFn =
      [](mlir::TensorType tensorType) -> std::optional<mlir::Attribute> {
    return MemoryAttr::get(tensorType.getContext(), MemorySpace::DDR,
                           MemLayout::Tensor);
  };
  options.unknownTypeConverterFn =
      [](mlir::Value value, mlir::Attribute memorySpace,
         const mlir::bufferization::BufferizationOptions &options)
      -> mlir::BaseMemRefType {
    (void)options;
    return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(
        mlir::cast<mlir::TensorType>(value.getType()), memorySpace);
  };
  options.functionArgTypeConverterFn =
      [](mlir::TensorType tensorType, mlir::Attribute memorySpace,
         mlir::func::FuncOp funcOp,
         const mlir::bufferization::BufferizationOptions &options)
      -> mlir::BaseMemRefType {
    (void)memorySpace;
    (void)funcOp;
    (void)options;
    if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(tensorType)) {
      return mlir::MemRefType::get(ranked.getShape(), ranked.getElementType(),
                                   mlir::MemRefLayoutAttrInterface{},
                                   MemoryAttr::get(ranked.getContext(),
                                                   MemorySpace::DDR,
                                                   MemLayout::Tensor));
    }
    return mlir::UnrankedMemRefType::get(
        tensorType.getElementType(),
        MemoryAttr::get(tensorType.getContext(), MemorySpace::DDR,
                        MemLayout::Tensor));
  };
  return mlir::bufferization::createOneShotBufferizePass(options);
}

#define GEN_PASS_REGISTRATION
#include "Wafer/Transforms/WaferPasses.h.inc"

void registerWaferTransformPasses() {
  static bool registered = [] {
    registerWaferTransformsPasses();
#ifdef WAFER_ENABLE_SHARDY
    mlir::registerPass([] { return createApplyDefaultSpmdShardingPass(); });
#endif
    return true;
  }();
  (void)registered;
}

} // namespace wafer
