//===- BufferizeInstrFunctionBoundaries.cpp - Wafer buffer contract -----===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

#include <optional>

namespace wafer {

#define GEN_PASS_DEF_BUFFERIZEINSTRFUNCTIONBOUNDARIESPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"

namespace {

static mlir::bufferization::OneShotBufferizationOptions
getInstrFunctionBoundaryBufferizationOptions() {
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
         const mlir::bufferization::BufferizationOptions &)
      -> mlir::BaseMemRefType {
    return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(
        mlir::cast<mlir::TensorType>(value.getType()), memorySpace);
  };
  options.functionArgTypeConverterFn =
      [](mlir::TensorType tensorType, mlir::Attribute,
         mlir::func::FuncOp,
         const mlir::bufferization::BufferizationOptions &)
      -> mlir::BaseMemRefType {
    mlir::Attribute memorySpace = MemoryAttr::get(
        tensorType.getContext(), MemorySpace::DDR, MemLayout::Tensor);
    if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(tensorType))
      return mlir::MemRefType::get(ranked.getShape(), ranked.getElementType(),
                                   mlir::MemRefLayoutAttrInterface{},
                                   memorySpace);
    return mlir::UnrankedMemRefType::get(tensorType.getElementType(),
                                         memorySpace);
  };
  return options;
}

struct BufferizeInstrFunctionBoundariesPass
    : public impl::BufferizeInstrFunctionBoundariesPassBase<
          BufferizeInstrFunctionBoundariesPass> {
  using impl::BufferizeInstrFunctionBoundariesPassBase<
      BufferizeInstrFunctionBoundariesPass>::
      BufferizeInstrFunctionBoundariesPassBase;

  void runOnOperation() final {
    mlir::bufferization::OneShotBufferizationOptions options =
        getInstrFunctionBoundaryBufferizationOptions();
    if (mlir::failed(mlir::bufferization::runOneShotModuleBufferize(
            getOperation(), options)))
      signalPassFailure();
  }
};

} // namespace
} // namespace wafer
