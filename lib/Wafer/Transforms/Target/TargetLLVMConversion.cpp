//===- Target LLVM lowering implementation -------------------------------===//

#include "Target/LowerInstrToTargetLLVMInternal.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/Target/PhysicalTopology.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Target/TargetFormat.h"
#include "Wafer/Transforms/TargetConversion.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace wafer::target_llvm_detail {

namespace {
static void eraseUnusedPrivateConstantGlobals(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::memref::GetGlobalOp, 4> deadGets;
  moduleOp.walk([&](mlir::memref::GetGlobalOp getGlobal) {
    if (getGlobal.getResult().use_empty())
      deadGets.push_back(getGlobal);
  });
  for (mlir::memref::GetGlobalOp getGlobal : deadGets)
    getGlobal->erase();

  llvm::SmallVector<mlir::memref::GlobalOp, 4> deadGlobals;
  for (mlir::memref::GlobalOp global :
       moduleOp.getOps<mlir::memref::GlobalOp>()) {
    if (!global.getConstantInitValue() ||
        mlir::SymbolTable::getSymbolVisibility(global) !=
            mlir::SymbolTable::Visibility::Private ||
        !mlir::SymbolTable::symbolKnownUseEmpty(global, moduleOp))
      continue;
    deadGlobals.push_back(global);
  }
  for (mlir::memref::GlobalOp global : deadGlobals)
    global->erase();
}

static mlir::LogicalResult
declareCallees(mlir::ModuleOp moduleOp,
               const llvm::StringMap<CalleeSignature> &callees) {
  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());

  llvm::SmallVector<llvm::StringRef, 32> sortedNames;
  for (const auto &entry : callees)
    sortedNames.push_back(entry.getKey());
  llvm::sort(sortedNames);

  for (llvm::StringRef name : sortedNames) {
    if (mlir::Operation *existing = moduleOp.lookupSymbol(name)) {
      auto llvmFunc = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(existing);
      auto found = callees.find(name);
      if (!llvmFunc || found == callees.end() ||
          llvmFunc.getFunctionType() != found->second.type)
        return existing->emitError()
               << "target_llvm_symbol_collision: existing @" << name
               << " does not match the target CRT declaration";
      continue;
    }
    auto found = callees.find(name);
    assert(found != callees.end() && "callee name must have a signature");
    const CalleeSignature &signature = found->second;
    builder.create<mlir::LLVM::LLVMFuncOp>(moduleOp.getLoc(), name,
                                           signature.type);
  }
  return mlir::success();
}

static mlir::LogicalResult
collectDirectCalleeSignatures(mlir::ModuleOp moduleOp,
                              llvm::StringMap<CalleeSignature> &callees) {
  mlir::WalkResult result = moduleOp.walk([&](mlir::LLVM::CallOp call) {
    mlir::FlatSymbolRefAttr callee = call.getCalleeAttr();
    if (!callee)
      return mlir::WalkResult::advance();
    mlir::LLVM::LLVMFunctionType type = call.getCalleeFunctionType();
    auto [entry, inserted] =
        callees.try_emplace(callee.getValue(), CalleeSignature{type});
    if (inserted || entry->second.type == type)
      return mlir::WalkResult::advance();
    call.emitError() << "target_llvm_symbol_collision: direct calls to @"
                     << callee.getValue()
                     << " have incompatible target signatures";
    return mlir::WalkResult::interrupt();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

static mlir::FailureOr<int64_t>
resolveDirectDTEContractParticipantCount(const PhysicalTopology &topology,
                                         mlir::ModuleOp diagnosticModule,
                                         PhysicalCardId physicalCardId) {
  std::optional<llvm::ArrayRef<PhysicalTileId>> available =
      topology.getAvailableTileIds(physicalCardId);
  if (!available || available->empty() ||
      available->size() > std::numeric_limits<uint32_t>::max())
    return diagnosticModule.emitError()
           << "target_abi_narrowing: Direct DTE participant count must fit "
              "a positive uint32_t";
  return static_cast<int64_t>(available->size());
}
} // namespace

mlir::LogicalResult
lowerModuleInPlace(mlir::ModuleOp moduleOp, bool transportPreparedBeforeEntry,
                   int64_t defaultDDRArenaArgumentIndex, int64_t physicalCardId,
                   int64_t physicalTileId, int64_t transportStatusArgumentIndex,
                   int64_t profileRecordArgumentIndex) {
  if (mlir::failed(flattenTileRegions(moduleOp)))
    return mlir::failure();

  bool hasDirectDTEOps = false;
  moduleOp.walk([&](mlir::Operation *op) {
    if (mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(op))
      hasDirectDTEOps = true;
  });
  const bool hasDirectDTEContract = transportStatusArgumentIndex >= 0;
  if (hasDirectDTEOps && !hasDirectDTEContract)
    return moduleOp.emitError()
           << "unsupported_target_transport: Direct DTE requires a "
              "launch-observable status argument";

  std::optional<PhysicalTopology> physicalTopology;
  if (hasDirectDTEContract || hasDirectDTEOps) {
    std::string reason;
    mlir::FailureOr<PhysicalTopology> resolved =
        PhysicalTopology::create(moduleOp, &reason);
    if (mlir::failed(resolved))
      return moduleOp.emitError()
             << "unsupported_target_transport: Direct DTE requires a valid "
                "physical topology: "
             << reason;
    physicalTopology.emplace(std::move(*resolved));
  }

  std::optional<int64_t> dteParticipantCount;
  if (hasDirectDTEContract) {
    mlir::FailureOr<int64_t> resolved =
        resolveDirectDTEContractParticipantCount(
            *physicalTopology, moduleOp, PhysicalCardId(physicalCardId));
    if (mlir::failed(resolved))
      return mlir::failure();
    dteParticipantCount = *resolved;
  }

  std::optional<DirectDTEEndpointDomain> dteDomain;
  if (hasDirectDTEOps) {
    mlir::FailureOr<DirectDTEEndpointDomain> resolved =
        resolveDirectDTEEndpointDomain(*physicalTopology, moduleOp,
                                       PhysicalCardId(physicalCardId),
                                       PhysicalTileId(physicalTileId));
    if (mlir::failed(resolved))
      return mlir::failure();
    dteDomain = std::move(*resolved);
    if (static_cast<int64_t>(dteDomain->availableTileIds.size()) !=
        *dteParticipantCount)
      return moduleOp.emitError()
             << "unsupported_target_transport: Direct DTE endpoint domain "
                "does not match the physical Tile participant count";
  }

  analysis::DirectCallGraphAnalysis callGraph(moduleOp.getOperation());
  if (mlir::failed(validateDirectCallsForTarget(
          callGraph, defaultDDRArenaArgumentIndex,
          transportStatusArgumentIndex, profileRecordArgumentIndex)))
    return mlir::failure();
  std::string dteEntrySymbol;
  if (hasDirectDTEContract) {
    mlir::FailureOr<mlir::func::FuncOp> entry =
        findUniqueRootFunction(callGraph);
    if (mlir::failed(entry))
      return mlir::failure();
    if (transportStatusArgumentIndex >=
            static_cast<int64_t>((*entry).getNumArguments()) ||
        !(*entry)
             .getArgument(transportStatusArgumentIndex)
             .getType()
             .isInteger(64))
      return (*entry).emitError()
             << "target_abi_mismatch: Direct DTE status argument index does "
                "not identify an entry i64 argument";
    dteEntrySymbol = (*entry).getSymName().str();
  }
  if (profileRecordArgumentIndex >= 0) {
    mlir::FailureOr<mlir::func::FuncOp> entry =
        findUniqueRootFunction(callGraph);
    if (mlir::failed(entry))
      return mlir::failure();
    if (profileRecordArgumentIndex !=
            static_cast<int64_t>((*entry).getNumArguments()) - 1 ||
        !(*entry)
             .getArgument(profileRecordArgumentIndex)
             .getType()
             .isInteger(64))
      return (*entry).emitError()
             << "target_abi_mismatch: profiler record argument index must "
                "identify the final entry i64 argument";
  }
  if (mlir::failed(lowerSCFToControlFlow(moduleOp)))
    return mlir::failure();
  llvm::DenseMap<mlir::Operation *, AliasSummary> aliasSummaries;
  if (mlir::failed(
          analyzeDDRAliasContracts(moduleOp, callGraph, aliasSummaries)))
    return mlir::failure();
  dropRootAliasResults(moduleOp, callGraph);
  eraseTargetMetadata(moduleOp);
  // Tensor bufferization may temporarily outline a splat constant as a
  // private memref.global.  If tile lowering folded every use to an immediate
  // or local fill, the symbol is no longer a runtime resource and must not
  // leak into the closed target dialect.  A still-used global remains illegal
  // and therefore cannot become an implicit constant-address ABI channel.
  eraseUnusedPrivateConstantGlobals(moduleOp);

  mlir::LLVMTypeConverter converter(moduleOp.getContext());
  converter.addConversion(
      [&](mlir::MemRefType type) -> std::optional<mlir::Type> {
        if (!isWaferMemRefType(type))
          return mlir::Type();
        return mlir::IntegerType::get(moduleOp.getContext(), 64);
      });
  converter.addConversion([&](mlir::async::TokenType) -> mlir::Type {
    return mlir::IntegerType::get(moduleOp.getContext(), 64);
  });

  llvm::StringMap<CalleeSignature> usedCallees;
  mlir::RewritePatternSet patterns(moduleOp.getContext());
  populateTargetLLVMStructureConversionPatterns(converter, patterns,
                                                defaultDDRArenaArgumentIndex);
  populateTargetInstructionConversionPatterns(
      converter, patterns, dteDomain ? &*dteDomain : nullptr);
  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);

  mlir::ConversionTarget target(*moduleOp.getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                           mlir::cf::ControlFlowDialect,
                           mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                           WaferDialect>();
  if (mlir::failed(
          mlir::applyFullConversion(moduleOp, target, std::move(patterns))))
    return moduleOp.emitError()
           << "target_llvm_lowering_failure: full target LLVM conversion "
              "failed";

  if (mlir::failed(collectDirectCalleeSignatures(moduleOp, usedCallees)))
    return mlir::failure();

  TargetCallBuiltin dteBeginBuiltin =
      transportPreparedBeforeEntry
          ? TargetCallBuiltin::DirectDTEBeginAfterPrepare
          : TargetCallBuiltin::DirectDTEBegin;
  if (hasDirectDTEContract &&
      mlir::failed(injectDirectDTEStatusLifecycle(
          moduleOp, dteEntrySymbol, transportStatusArgumentIndex,
          *dteParticipantCount, dteBeginBuiltin, usedCallees)))
    return mlir::failure();

  if (mlir::failed(declareCallees(moduleOp, usedCallees)))
    return mlir::failure();

  mlir::Operation *remainingNonLLVMOp = nullptr;
  moduleOp.walk([&](mlir::Operation *op) {
    if (op == moduleOp || mlir::isa<mlir::LLVM::LLVMFuncOp>(op) ||
        (op->getDialect() &&
         op->getDialect()->getNamespace() ==
             mlir::LLVM::LLVMDialect::getDialectNamespace()))
      return mlir::WalkResult::advance();
    remainingNonLLVMOp = op;
    return mlir::WalkResult::interrupt();
  });
  if (!remainingNonLLVMOp)
    return mlir::success();

  remainingNonLLVMOp->emitError()
      << "target_llvm_lowering_failure: non-LLVM operation remains after "
         "lowering";
  return mlir::failure();
}

} // namespace wafer::target_llvm_detail
