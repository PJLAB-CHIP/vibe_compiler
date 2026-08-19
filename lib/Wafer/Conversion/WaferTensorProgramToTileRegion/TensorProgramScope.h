//===- TensorProgramScope.h - Private tensor program boundary -*- C++ -*-===//
#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace wafer::tensor_program_to_tile_region {

/// A single-block functional TensorProgram body. It carries no scheduling
/// destination convention and is valid before CardModule construction.
class TensorProgramBody {
public:
  explicit TensorProgramBody(mlir::func::FuncOp function)
      : function(function) {}

  mlir::func::FuncOp getFunction() { return function; }
  mlir::Block &getBody() { return function.getBody().front(); }
  mlir::Location getLoc() { return function.getLoc(); }
  mlir::MLIRContext *getContext() { return function.getContext(); }

private:
  mlir::func::FuncOp function;
};

/// A verified private tensor-program scheduling scope. Its entry arguments are
/// the unchanged source inputs followed by compiler-created scheduling
/// destinations; func.return yields one root per destination. The appended
/// destinations never cross the CardModule conversion boundary.
class TensorProgramScope : public TensorProgramBody {
public:
  TensorProgramScope(mlir::func::FuncOp function,
                     unsigned functionalArgumentCount)
      : TensorProgramBody(function),
        functionalArgumentCount(functionalArgumentCount) {}

  mlir::func::ReturnOp getReturn() {
    return mlir::cast<mlir::func::ReturnOp>(getBody().getTerminator());
  }
  unsigned getOutputCount() { return getFunction().getNumResults(); }
  unsigned getInputCount() { return functionalArgumentCount; }
  /// Extra entry arguments after inputs and scheduling destinations, one per
  /// consumer-side boundary supply. They never reach a sibling root.
  unsigned getBoundaryArgumentCount() {
    return getFunction().getNumArguments() - getInputCount() -
           getOutputCount();
  }
  mlir::ValueRange getInputs() {
    return mlir::ValueRange(getFunction().getArguments())
        .take_front(getInputCount());
  }
  mlir::ValueRange getOutputs() {
    return mlir::ValueRange(getFunction().getArguments())
        .slice(getInputCount(), getOutputCount());
  }
  mlir::ValueRange getBoundaryArguments() {
    return mlir::ValueRange(getFunction().getArguments())
        .take_back(getBoundaryArgumentCount());
  }
  mlir::TypeRange getResultTypes() { return getFunction().getResultTypes(); }

private:
  unsigned functionalArgumentCount;
};

} // namespace wafer::tensor_program_to_tile_region
