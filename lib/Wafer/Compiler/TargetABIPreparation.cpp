//===- TargetABIPreparation.cpp - Per-rank Kernel ABI preparation -------===//

#include "TargetArtifactInternal.h"

#include "AcceptedCallClosure.h"

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>

namespace wafer::compiler::detail {

PreparedTargetRank::PreparedTargetRank(const ExecutionConfig &executionConfig,
                                       bool transportPreparedBeforeEntry)
    : targetProfile(executionConfig.getTargetProfileId()),
      transportPreparedBeforeEntry(transportPreparedBeforeEntry),
      targetIdentity(getTargetProfileRecord(targetProfile).targetIdentity),
      kernelRuntimeABI(getTargetProfileRecord(targetProfile).kernelRuntimeABI),
      moduleFormat(getTargetProfileRecord(targetProfile).moduleFormat.str()) {}

namespace {

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

std::optional<int64_t> combineAlignmentRequirements(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0)
    return std::nullopt;
  int64_t scaled = lhs / std::gcd(lhs, rhs);
  if (scaled > std::numeric_limits<int64_t>::max() / rhs)
    return std::nullopt;
  return scaled * rhs;
}

KernelABISlotRole getKernelRole(ProgramResourceRole role) {
  switch (role) {
  case ProgramResourceRole::UserInput:
    return KernelABISlotRole::UserInput;
  case ProgramResourceRole::Parameter:
    return KernelABISlotRole::Parameter;
  case ProgramResourceRole::Constant:
    return KernelABISlotRole::Constant;
  case ProgramResourceRole::Output:
    return KernelABISlotRole::Output;
  }
  llvm_unreachable("unknown program resource role");
}

mlir::FailureOr<WaferPhysicalTensorInfo>
getPhysicalInfo(mlir::Type type, mlir::Operation *anchor) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType || !isWaferDDRMemRefType(memrefType)) {
    anchor->emitError()
        << "target_abi_mismatch: kernel resource is not a Wafer DDR memref";
    return mlir::failure();
  }
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes < 0) {
    anchor->emitError()
        << "target_abi_mismatch: cannot derive static resource byte size";
    return mlir::failure();
  }
  return *info;
}

mlir::Value resolveOutputAllocation(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (value) {
    if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = blockArgument.getOwner();
      auto tileRegion =
          owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
                : TileRegionOp{};
      if (!tileRegion || owner != &tileRegion.getBody().front() ||
          blockArgument.getArgNumber() >= tileRegion.getInputs().size())
        return value;
      value = tileRegion.getInputs()[blockArgument.getArgNumber()];
      continue;
    }

    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      return value;
    if (mlir::isa<mlir::memref::AllocOp>(definition))
      return value;
    if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(definition)) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      auto yield = mlir::dyn_cast<TileYieldOp>(
          tileRegion.getBody().front().getTerminator());
      if (!result || !yield ||
          result.getResultNumber() >= yield.getNumOperands())
        return value;
      value = yield.getOperand(result.getResultNumber());
      continue;
    }
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(definition)) {
      value = view.getViewSource();
      continue;
    }
    return value;
  }
  return value;
}

} // namespace

mlir::FailureOr<PreparedTargetRank>
prepareTargetABI(const RankExecutable &rankExecutable,
                 const ExecutionConfig &executionConfig,
                 bool transportPreparedBeforeEntry,
                 ProfileCaptureKind profileCapture) {
  PreparedTargetRank prepared(executionConfig, transportPreparedBeforeEntry);
  prepared.module = rankExecutable.getModule().clone();
  prepared.logicalRank = rankExecutable.getLogicalRank();
  prepared.profileCapture = profileCapture;
  const int64_t defaultDDRAlignment =
      getDefaultWaferTargetPolicy().memory.ddrAlignmentBytes;

  llvm::Expected<AcceptedCallClosure> closure = analyzeAcceptedCallClosure(
      *prepared.module, rankExecutable.getEntrySymbol());
  if (!closure) {
    prepared.module->emitError()
        << "target_abi_mismatch: " << llvm::toString(closure.takeError());
    return mlir::failure();
  }
  mlir::func::FuncOp function = closure->entry;

  unsigned originalArgumentCount = function.getNumArguments();
  unsigned resultCount = function.getFunctionType().getNumResults();
  std::vector<const RankProgramBinding *> argumentBindings(
      originalArgumentCount, nullptr);
  std::vector<const RankProgramBinding *> outputBindings(resultCount, nullptr);
  for (const RankProgramBinding &binding :
       rankExecutable.getProgramBindings()) {
    std::vector<const RankProgramBinding *> &domain =
        binding.role == ProgramResourceRole::Output ? outputBindings
                                                    : argumentBindings;
    if (binding.index < 0 ||
        binding.index >= static_cast<int64_t>(domain.size()) ||
        domain[binding.index]) {
      function.emitError()
          << "target_abi_mismatch: resource bindings do not form an exact "
             "function boundary";
      return mlir::failure();
    }
    domain[binding.index] = &binding;
  }
  if (llvm::is_contained(argumentBindings, nullptr) ||
      llvm::is_contained(outputBindings, nullptr)) {
    function.emitError()
        << "target_abi_mismatch: resource bindings do not cover every "
           "argument and result";
    return mlir::failure();
  }

  prepared.slots.reserve(originalArgumentCount + resultCount + 1);
  auto appendSlot = [&](const RankProgramBinding &binding, mlir::Type type,
                        KernelABISlotRole role) -> mlir::LogicalResult {
    mlir::FailureOr<WaferPhysicalTensorInfo> physical =
        getPhysicalInfo(type, function);
    if (mlir::failed(physical))
      return mlir::failure();
    prepared.slots.push_back({static_cast<int64_t>(prepared.slots.size()), role,
                              binding.index, binding.name, binding.dtype,
                              physical->layout, binding.localShape,
                              physical->physicalBytes, defaultDDRAlignment});
    return mlir::success();
  };

  for (unsigned index = 0; index < originalArgumentCount; ++index)
    if (mlir::failed(appendSlot(*argumentBindings[index],
                                function.getArgument(index).getType(),
                                getKernelRole(argumentBindings[index]->role))))
      return mlir::failure();

  llvm::SmallVector<mlir::func::ReturnOp, 2> returns;
  function.walk([&](mlir::func::ReturnOp returnOp) {
    if (returnOp->getParentOfType<mlir::func::FuncOp>() == function)
      returns.push_back(returnOp);
  });
  if (returns.size() != 1 || returns.front().getNumOperands() != resultCount) {
    function.emitError()
        << "target_abi_mismatch: current output binding requires exactly one "
           "complete entry return";
    return mlir::failure();
  }

  for (unsigned index = 0; index < resultCount; ++index) {
    mlir::Value root =
        resolveOutputAllocation(returns.front().getOperand(index));
    auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
    if (!allocation || !isWaferDDRMemRefType(allocation.getType()) ||
        !allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName)) {
      returns.front().emitError()
          << "target_abi_mismatch: output #" << index
          << " is not backed by one accepted compiler-managed DDR root";
      return mlir::failure();
    }
    mlir::Type resultType = function.getFunctionType().getResult(index);
    if (allocation.getType() != resultType) {
      allocation.emitError()
          << "target_abi_mismatch: output root type differs from result type";
      return mlir::failure();
    }
    unsigned outputArgumentIndex = function.getNumArguments();
    function.insertArgument(outputArgumentIndex, resultType,
                            mlir::DictionaryAttr{}, function.getLoc());
    mlir::BlockArgument outputArgument =
        function.getArgument(outputArgumentIndex);
    allocation.getResult().replaceAllUsesWith(outputArgument);
    allocation.erase();
    if (mlir::failed(appendSlot(*outputBindings[index], resultType,
                                KernelABISlotRole::Output)))
      return mlir::failure();
  }

  int64_t arenaBytes = 0;
  int64_t arenaAlignment = defaultDDRAlignment;
  mlir::LogicalResult arenaValid = mlir::success();
  function.walk([&](mlir::memref::AllocOp allocation) {
    if (mlir::failed(arenaValid) || !isWaferDDRMemRefType(allocation.getType()))
      return;
    auto offset =
        allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName);
    std::optional<WaferPhysicalTensorInfo> info =
        computeWaferPhysicalTensorInfo(allocation.getType());
    int64_t end = 0;
    if (!offset || !info || info->physicalBytes < 0 ||
        !checkedAdd(offset.getOffset(), info->physicalBytes, end)) {
      arenaValid = allocation.emitError()
                   << "target_abi_mismatch: cannot derive default DDR arena "
                      "high-water bytes";
      return;
    }
    arenaBytes = std::max(arenaBytes, end);
    if (auto alignment = allocation.getAlignmentAttr()) {
      std::optional<int64_t> combined =
          combineAlignmentRequirements(arenaAlignment, alignment.getInt());
      if (!combined) {
        arenaValid = allocation.emitError()
                     << "target_abi_mismatch: combined default DDR arena "
                        "alignment is invalid or exceeds int64";
        return;
      }
      arenaAlignment = *combined;
    }
  });
  if (mlir::failed(arenaValid))
    return mlir::failure();

  if (arenaBytes > 0) {
    prepared.defaultDDRArenaArgumentIndex = function.getNumArguments();
    function.insertArgument(prepared.defaultDDRArenaArgumentIndex,
                            mlir::IntegerType::get(function.getContext(), 64),
                            mlir::DictionaryAttr{}, function.getLoc());
    prepared.slots.push_back({static_cast<int64_t>(prepared.slots.size()),
                              KernelABISlotRole::Workspace,
                              0,
                              "default_ddr_arena",
                              "u8",
                              MemLayout::Tensor,
                              {arenaBytes},
                              arenaBytes,
                              arenaAlignment});
  }

  if (rankExecutable.getTransportContract() == TransportContract::DirectDTE) {
    prepared.transportStatusArgumentIndex = function.getNumArguments();
    function.insertArgument(prepared.transportStatusArgumentIndex,
                            mlir::IntegerType::get(function.getContext(), 64),
                            mlir::DictionaryAttr{}, function.getLoc());
    prepared.slots.push_back(
        {static_cast<int64_t>(prepared.slots.size()),
         KernelABISlotRole::TransportStatus,
         0,
         "direct_dte_status",
         "u32",
         MemLayout::Tensor,
         {1},
         WAFER_TX81_DIRECT_DTE_STATUS_V2_STORAGE_BYTES,
         WAFER_TX81_DIRECT_DTE_STATUS_V2_STORAGE_ALIGNMENT});
  }

  if (profileCapture != ProfileCaptureKind::None) {
    if (executionConfig.getRankCount() != WAFER_TX81_PROFILER_TILE_COUNT ||
        executionConfig.getRuntimeLaunchKind() == RuntimeLaunchKind::Model) {
      function.emitError()
          << "target_abi_mismatch: profiler capture requires the complete "
             "16-rank pointer-table launch domain";
      return mlir::failure();
    }
    const uint64_t recordBytes = getProfileCaptureRecordBytes(profileCapture);
    if (recordBytes >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      function.emitError()
          << "target_abi_mismatch: profiler record bytes exceed int64";
      return mlir::failure();
    }
    prepared.profileRecordArgumentIndex = function.getNumArguments();
    function.insertArgument(prepared.profileRecordArgumentIndex,
                            mlir::IntegerType::get(function.getContext(), 64),
                            mlir::DictionaryAttr{}, function.getLoc());
    prepared.slots.push_back({static_cast<int64_t>(prepared.slots.size()),
                              KernelABISlotRole::Workspace,
                              1,
                              "tx81_profiler_record",
                              "u8",
                              MemLayout::Tensor,
                              {static_cast<int64_t>(recordBytes)},
                              static_cast<int64_t>(recordBytes),
                              WAFER_TX81_PROFILER_BUFFER_ALIGNMENT});
  }

  if (mlir::failed(mlir::verify(*prepared.module)))
    return mlir::failure();
  return prepared;
}

} // namespace wafer::compiler::detail
