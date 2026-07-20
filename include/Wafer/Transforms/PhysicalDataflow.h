//===- PhysicalDataflow.h - MLIR-native physical dataflow ------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_PHYSICALDATAFLOW_H
#define WAFER_TRANSFORMS_PHYSICALDATAFLOW_H

namespace mlir {
class DialectRegistry;
class Operation;
namespace func {
class FuncOp;
}
} // namespace mlir

namespace wafer {

/// Register Wafer-owned target implementation external models for supported
/// upstream structured operations.
void registerTargetImplementationExternalModels(
    mlir::DialectRegistry &registry);

/// Clone a pure tensor producer at each compatible consumer after its first
/// use.  The duplicated producer remains ordinary structured IR and is tiled
/// against the consumer by the existing complete-candidate materializer.
unsigned materializeConsumerLocalTensorRecomputation(mlir::func::FuncOp task);

/// Hoist side-effect-free, speculatable operations whose operands are loop
/// invariant.  This delegates legality and movement to LoopLikeOpInterface;
/// callers must rerun lifetime and placement on the mutated clone.
unsigned hoistStaticLoopInvariantOperations(mlir::func::FuncOp function);

/// Apply one modularly exact integer reassociation to each eligible linalg
/// scalar body.  Operations carrying no-wrap promises are deliberately not
/// rewritten because a changed poison boundary is not a modular equivalence.
unsigned reassociateIntegerElementwiseExpressions(mlir::func::FuncOp task);

/// Balance an eligible four-leaf modular integer addition chain into an
/// explicit SSA tree.
unsigned balanceIntegerElementwiseReductionTrees(mlir::func::FuncOp task);

/// Distribute modular integer multiplication over addition in eligible
/// linalg scalar bodies.
unsigned distributeIntegerElementwiseExpressions(mlir::func::FuncOp task);

/// Factor a common modular integer multiplicand from eligible linalg scalar
/// bodies.
unsigned factorIntegerElementwiseExpressions(mlir::func::FuncOp task);

/// Reorder independent instruction runs with a deterministic movement-first
/// ready policy. SSA dependencies, value-associated read/write hazards, and
/// explicit local fences remain ordering edges in the rewritten IR; no
/// schedule side table is produced.
unsigned scheduleIndependentInstructionsByReadyOrder(mlir::Operation *scope);

} // namespace wafer

#endif // WAFER_TRANSFORMS_PHYSICALDATAFLOW_H
