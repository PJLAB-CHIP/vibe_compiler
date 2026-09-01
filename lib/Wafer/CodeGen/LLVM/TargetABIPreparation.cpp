//===- TargetABIPreparation.cpp - Tile ABI preparation --------===//

#include "Wafer/CodeGen/DeviceExecutableInternal.h"
#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"
#include "Wafer/CodeGen/ProgramElementTypeConversion.h"

#include "Wafer/Analysis/Module/ExecutableCallClosure.h"

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Target/TargetMemory.h"

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
#include <optional>

namespace wafer::compiler::detail {

PreparedTile::PreparedTile(const ExecutionConfig &executionConfig,
                           bool transportPreparedBeforeEntry)
    : transportPreparedBeforeEntry(transportPreparedBeforeEntry),
      targetIdentity(executionConfig.getTargetIdentityId()),
      kernelRuntimeABI(KernelRuntimeABIId::waferTx81Kernel()),
      moduleFormat(kCurrentTargetModuleFormat.str()) {}

namespace {

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

mlir::FailureOr<int64_t> getRequiredPhysicalAlignment(
    mlir::MemRefType type, llvm::ArrayRef<int64_t> requirements,
    mlir::Operation *anchor, llvm::StringRef failureReason) {
  mlir::FailureOr<int64_t> combined =
      computeWaferRequiredAlignmentBytes(type, requirements);
  if (mlir::failed(combined)) {
    anchor->emitError() << "target_abi_mismatch: " << failureReason;
    return mlir::failure();
  }
  return *combined;
}

TileEntryArgumentKind getTileEntryArgumentKind(ProgramResourceRole role) {
  switch (role) {
  case ProgramResourceRole::UserInput:
    return TileEntryArgumentKind::ExternalInput;
  case ProgramResourceRole::Parameter:
  case ProgramResourceRole::Constant:
    return TileEntryArgumentKind::TargetTensor;
  case ProgramResourceRole::Output:
    return TileEntryArgumentKind::ExternalOutput;
  }
  llvm_unreachable("unknown program resource role");
}

TileEntryArgumentAccess getTileEntryArgumentAccess(TileEntryArgumentKind kind) {
  switch (kind) {
  case TileEntryArgumentKind::ExternalInput:
  case TileEntryArgumentKind::TargetTensor:
    return TileEntryArgumentAccess::ReadOnly;
  case TileEntryArgumentKind::ExternalOutput:
    return TileEntryArgumentAccess::WriteOnly;
  case TileEntryArgumentKind::Workspace:
  case TileEntryArgumentKind::SharedWorkspace:
  case TileEntryArgumentKind::ProfileRecord:
  case TileEntryArgumentKind::TransportStatus:
    return TileEntryArgumentAccess::ReadWrite;
  }
  llvm_unreachable("unknown tile entry argument kind");
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

mlir::FailureOr<LogicalFormat>
getPhysicalFormat(const WaferPhysicalTensorInfo &physical,
                  mlir::Operation *anchor) {
  mlir::Type type = physical.logicalTensorType.getElementType();
  if (mlir::isa<mlir::Float16Type>(type))
    return LogicalFormat::F16;
  if (mlir::isa<mlir::BFloat16Type>(type))
    return LogicalFormat::BF16;
  if (mlir::isa<mlir::Float32Type>(type))
    return LogicalFormat::F32;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    const bool isUnsigned = integer.isUnsigned();
    switch (integer.getWidth()) {
    case 1:
      return LogicalFormat::Bool;
    case 8:
      return isUnsigned ? LogicalFormat::U8 : LogicalFormat::I8;
    case 16:
      return isUnsigned ? LogicalFormat::U16 : LogicalFormat::I16;
    case 32:
      return isUnsigned ? LogicalFormat::U32 : LogicalFormat::I32;
    case 64:
      return isUnsigned ? LogicalFormat::U64 : LogicalFormat::I64;
    default:
      break;
    }
  }
  anchor->emitError()
      << "target_abi_mismatch: physical tensor has no target format";
  return mlir::failure();
}

mlir::Value resolveOutputAllocation(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (value) {
    if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Value entry =
          analysis::getSingleExecutionRegionEntryOperand(blockArgument);
      if (!entry)
        return value;
      value = entry;
      continue;
    }

    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      return value;
    if (mlir::isa<mlir::memref::AllocOp>(definition))
      return value;
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
      if (mlir::Value exit =
              analysis::getSingleExecutionRegionExitOperand(result)) {
        value = exit;
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

struct ProgramResourceBoundary {
  std::vector<const ProgramResourceBinding *> argumentBindings;
  std::vector<DDRBindingAttr> ddrBindings;
  std::vector<const ProgramResourceBinding *> outputBindings;
};

mlir::FailureOr<ProgramResourceBoundary> buildProgramResourceBoundary(
    mlir::func::FuncOp function,
    llvm::ArrayRef<ProgramResourceBinding> programBindings,
    bool emitDiagnostics) {
  const unsigned argumentCount = function.getNumArguments();
  ProgramResourceBoundary boundary;
  boundary.argumentBindings.assign(argumentCount, nullptr);
  boundary.ddrBindings.resize(argumentCount);
  for (unsigned index = 0; index < argumentCount; ++index)
    boundary.ddrBindings[index] = function.getArgAttrOfType<DDRBindingAttr>(
        index, kWaferDDRBindingAttrName);
  boundary.outputBindings.assign(function.getFunctionType().getNumResults(),
                                 nullptr);

  for (const ProgramResourceBinding &binding : programBindings) {
    std::vector<const ProgramResourceBinding *> &domain =
        binding.role == ProgramResourceRole::Output ? boundary.outputBindings
                                                    : boundary.argumentBindings;
    if (binding.index < 0 ||
        binding.index >= static_cast<int64_t>(domain.size()) ||
        domain[binding.index] ||
        (&domain == &boundary.argumentBindings &&
         boundary.ddrBindings[binding.index])) {
      if (emitDiagnostics)
        function.emitError()
            << "target_abi_mismatch: resource bindings do not form an exact "
               "function boundary: index="
            << binding.index << " argument_count=" << argumentCount
            << " duplicate_program_binding="
            << static_cast<bool>(binding.index >= 0 &&
                                 binding.index <
                                     static_cast<int64_t>(domain.size()) &&
                                 domain[binding.index])
            << " has_ddr_binding="
            << static_cast<bool>(
                   &domain == &boundary.argumentBindings &&
                   binding.index >= 0 &&
                   binding.index <
                       static_cast<int64_t>(boundary.ddrBindings.size()) &&
                   boundary.ddrBindings[binding.index]);
      return mlir::failure();
    }
    domain[binding.index] = &binding;
  }
  if (llvm::any_of(llvm::seq<unsigned>(0, argumentCount),
                   [&](unsigned index) {
                     return !boundary.argumentBindings[index] &&
                            !boundary.ddrBindings[index];
                   }) ||
      llvm::is_contained(boundary.outputBindings, nullptr)) {
    if (emitDiagnostics) {
      auto diagnostic = function.emitError()
                        << "target_abi_mismatch: resource bindings do not "
                           "cover every argument and result: "
                           "uncovered_arguments=[";
      bool first = true;
      for (unsigned index = 0; index < argumentCount; ++index) {
        if (boundary.argumentBindings[index] || boundary.ddrBindings[index])
          continue;
        if (!first)
          diagnostic << ',';
        first = false;
        diagnostic << index;
      }
      diagnostic << "] argument_count=" << argumentCount;
    }
    return mlir::failure();
  }
  return boundary;
}

} // namespace

mlir::LogicalResult verifyProgramResourceBoundary(
    mlir::func::FuncOp function,
    llvm::ArrayRef<ProgramResourceBinding> programBindings) {
  return mlir::succeeded(
             buildProgramResourceBoundary(function, programBindings,
                                          /*emitDiagnostics=*/false))
             ? mlir::success()
             : mlir::failure();
}

mlir::FailureOr<PreparedTile>
prepareTargetABI(const TileExecutable &tileExecutable,
                 const ExecutionConfig &executionConfig,
                 bool transportPreparedBeforeEntry,
                 ProfileCaptureKind profileCapture) {
  PreparedTile prepared(executionConfig, transportPreparedBeforeEntry);
  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::TargetABIModuleClone);
  prepared.module = tileExecutable.getModule().clone();
  if (tileExecutable.getCardId() != CardId(0) ||
      tileExecutable.getTileId().getValue() < 0 ||
      tileExecutable.getTileId().getValue() >= executionConfig.getTileCount() ||
      tileExecutable.getLaunchSlotId().getValue() < 0 ||
      tileExecutable.getLaunchSlotId().getValue() >=
          executionConfig.getTileCount()) {
    prepared.module->emitError(
        "target_abi_mismatch: Tile identity is outside the "
        "single-card execution domain");
    return mlir::failure();
  }
  prepared.cardId = tileExecutable.getCardId();
  prepared.tileId = tileExecutable.getTileId();
  prepared.launchSlotId = tileExecutable.getLaunchSlotId();
  prepared.profileCapture = profileCapture;
  const int64_t defaultDDRAlignment = getTargetMemoryPolicy().ddrAlignmentBytes;

  llvm::Expected<ExecutableCallClosure> closure = analyzeExecutableCallClosure(
      *prepared.module, tileExecutable.getEntrySymbol());
  if (!closure) {
    prepared.module->emitError()
        << "target_abi_mismatch: " << llvm::toString(closure.takeError());
    return mlir::failure();
  }
  mlir::func::FuncOp function = closure->entry;

  mlir::FailureOr<ProgramResourceBoundary> boundary =
      buildProgramResourceBoundary(function,
                                   tileExecutable.getProgramBindings(),
                                   /*emitDiagnostics=*/true);
  if (mlir::failed(boundary))
    return mlir::failure();
  const unsigned originalArgumentCount = function.getNumArguments();
  const unsigned resultCount = function.getFunctionType().getNumResults();
  const auto &argumentBindings = boundary->argumentBindings;
  const auto &ddrBindings = boundary->ddrBindings;
  const auto &outputBindings = boundary->outputBindings;

  prepared.slots.reserve(originalArgumentCount + resultCount + 3);
  auto appendSlot = [&](const ProgramResourceBinding &binding, mlir::Type type,
                        TileEntryArgumentKind kind) -> mlir::LogicalResult {
    mlir::FailureOr<WaferPhysicalTensorInfo> physical =
        getPhysicalInfo(type, function);
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
    mlir::FailureOr<int64_t> alignment =
        memrefType ? getRequiredPhysicalAlignment(
                         memrefType, {defaultDDRAlignment}, function,
                         "cannot combine target and physical encoding DDR "
                         "alignment")
                   : mlir::FailureOr<int64_t>(mlir::failure());
    if (mlir::failed(physical) || mlir::failed(alignment))
      return mlir::failure();
    mlir::FailureOr<LogicalFormat> format =
        getPhysicalFormat(*physical, function);
    if (mlir::failed(format))
      return mlir::failure();
    prepared.slots.push_back({static_cast<int64_t>(prepared.slots.size()), kind,
                              binding.index, binding.name, *format,
                              physical->layout, binding.localShape,
                              physical->physicalBytes, *alignment,
                              getTileEntryArgumentAccess(kind)});
    if (kind == TileEntryArgumentKind::TargetTensor) {
      std::optional<LogicalFormat> sourceFormat =
          getTargetLogicalFormat(binding.dtype);
      if (!sourceFormat) {
        function.emitError()
            << "target_abi_mismatch: TargetTensor source element type has no "
               "target logical format";
        return mlir::failure();
      }
      // The current compiler only creates an implicit action when no numeric
      // conversion is needed. A future dtype-changing selection must carry
      // its rounding/zero-point parameter into this boundary explicitly.
      llvm::Expected<TargetTensorMaterializationAction> materialization =
          TargetTensorMaterializationAction::create(*sourceFormat, *format,
                                                    /*parameter=*/std::nullopt);
      if (!materialization) {
        function.emitError()
            << "target_abi_mismatch: TargetTensor materialization is not "
               "explicit: "
            << llvm::toString(materialization.takeError());
        return mlir::failure();
      }
      prepared.slots.back().targetTensorMaterialization =
          std::move(*materialization);
    }
    return mlir::success();
  };

  for (unsigned index = 0; index < originalArgumentCount; ++index) {
    if (const ProgramResourceBinding *binding = argumentBindings[index]) {
      if (mlir::failed(appendSlot(*binding,
                                  function.getArgument(index).getType(),
                                  getTileEntryArgumentKind(binding->role))))
        return mlir::failure();
      continue;
    }
    DDRBindingAttr binding = ddrBindings[index];
    auto declaration =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
            function, binding.getResource());
    auto resource = declaration ? declaration->getAttrOfType<DDRResourceAttr>(
                                      kWaferDDRResourceAttrName)
                                : DDRResourceAttr{};
    auto memrefType =
        mlir::dyn_cast<mlir::MemRefType>(function.getArgument(index).getType());
    if (!declaration || !resource || !memrefType ||
        declaration.getType() != memrefType || resource.getResourceId() < 0 ||
        resource.getResourceId() != binding.getResourceId()) {
      function.emitError(
          "target_abi_mismatch: card DDR argument has no matching resource "
          "declaration");
      return mlir::failure();
    }
    mlir::FailureOr<WaferPhysicalTensorInfo> physical =
        getPhysicalInfo(memrefType, function);
    mlir::FailureOr<int64_t> alignment = getRequiredPhysicalAlignment(
        memrefType, {defaultDDRAlignment}, function,
        "card DDR boundary alignment is invalid");
    mlir::FailureOr<LogicalFormat> format =
        mlir::succeeded(physical)
            ? getPhysicalFormat(*physical, function)
            : mlir::FailureOr<LogicalFormat>(mlir::failure());
    if (mlir::failed(physical) || mlir::failed(alignment) ||
        mlir::failed(format))
      return mlir::failure();
    TileEntryArgumentAccess access = binding.getAccess() == DDRAccess::None
                                         ? TileEntryArgumentAccess::None
                                     : binding.getAccess() == DDRAccess::Read
                                         ? TileEntryArgumentAccess::ReadOnly
                                     : binding.getAccess() == DDRAccess::Write
                                         ? TileEntryArgumentAccess::WriteOnly
                                         : TileEntryArgumentAccess::ReadWrite;
    prepared.slots.push_back(
        {static_cast<int64_t>(prepared.slots.size()),
         TileEntryArgumentKind::SharedWorkspace, resource.getResourceId(),
         binding.getResource().getValue().str(), *format, physical->layout,
         std::vector<int64_t>(memrefType.getShape().begin(),
                              memrefType.getShape().end()),
         physical->physicalBytes, *alignment, access});
  }

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

  llvm::SmallPtrSet<mlir::Operation *, 4> outputAllocations;
  llvm::SmallVector<mlir::memref::AllocOp, 4> orderedOutputAllocations;
  orderedOutputAllocations.reserve(resultCount);
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
    if (!outputAllocations.insert(allocation.getOperation()).second) {
      returns.front().emitError()
          << "target_abi_mismatch: output results require distinct "
             "compiler-managed DDR roots";
      return mlir::failure();
    }
    orderedOutputAllocations.push_back(allocation);
    if (mlir::failed(appendSlot(*outputBindings[index], resultType,
                                TileEntryArgumentKind::ExternalOutput)))
      return mlir::failure();
  }

  int64_t arenaBytes = 0;
  int64_t arenaAlignment = defaultDDRAlignment;
  mlir::LogicalResult arenaValid = mlir::success();
  function.walk([&](mlir::memref::AllocOp allocation) {
    if (mlir::failed(arenaValid) || !isWaferDDRMemRefType(allocation.getType()))
      return;
    if (outputAllocations.contains(allocation.getOperation()))
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
    llvm::SmallVector<int64_t, 2> alignmentRequirements = {arenaAlignment};
    if (auto alignment = allocation.getAlignmentAttr())
      alignmentRequirements.push_back(alignment.getInt());
    mlir::FailureOr<int64_t> physicalAlignment = getRequiredPhysicalAlignment(
        allocation.getType(), alignmentRequirements, allocation,
        "combined default DDR arena alignment is invalid or exceeds int64");
    if (mlir::failed(physicalAlignment)) {
      arenaValid = mlir::failure();
      return;
    }
    arenaAlignment = *physicalAlignment;
  });
  if (mlir::failed(arenaValid))
    return mlir::failure();

  std::optional<uint64_t> profileRecordBytes;
  if (profileCapture != ProfileCaptureKind::None) {
    if (executionConfig.getTileCount() != WAFER_TX81_PROFILER_TILE_COUNT) {
      function.emitError()
          << "target_abi_mismatch: profiler capture requires the complete "
             "16-Tile pointer-table launch domain";
      return mlir::failure();
    }
    profileRecordBytes = getProfileCaptureRecordBytes(profileCapture);
    if (*profileRecordBytes >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      function.emitError()
          << "target_abi_mismatch: profiler record bytes exceed int64";
      return mlir::failure();
    }
  }

  for (unsigned index = 0; index < resultCount; ++index) {
    const unsigned outputArgumentIndex = originalArgumentCount + index;
    function.insertArgument(outputArgumentIndex,
                            function.getFunctionType().getResult(index),
                            mlir::DictionaryAttr{}, function.getLoc());
  }
  for (auto [index, allocation] : llvm::enumerate(orderedOutputAllocations)) {
    mlir::BlockArgument outputArgument =
        function.getArgument(originalArgumentCount + index);
    allocation.getResult().replaceAllUsesWith(outputArgument);
    allocation.erase();
  }

  {
    const int64_t effectiveArenaBytes =
        std::max(arenaBytes, defaultDDRAlignment);
    prepared.defaultDDRArenaArgumentIndex = function.getNumArguments();
    function.insertArgument(prepared.defaultDDRArenaArgumentIndex,
                            mlir::IntegerType::get(function.getContext(), 64),
                            mlir::DictionaryAttr{}, function.getLoc());
    prepared.slots.push_back(
        {static_cast<int64_t>(prepared.slots.size()),
         TileEntryArgumentKind::Workspace,
         0,
         "default_ddr_arena",
         LogicalFormat::U8,
         MemLayout::Tensor,
         {effectiveArenaBytes},
         effectiveArenaBytes,
         arenaAlignment,
         getTileEntryArgumentAccess(TileEntryArgumentKind::Workspace)});
  }

  if (tileExecutable.getTransportContract() == TransportContract::DirectDTE) {
    prepared.transportStatusArgumentIndex = function.getNumArguments();
    function.insertArgument(prepared.transportStatusArgumentIndex,
                            mlir::IntegerType::get(function.getContext(), 64),
                            mlir::DictionaryAttr{}, function.getLoc());
    prepared.slots.push_back(
        {static_cast<int64_t>(prepared.slots.size()),
         TileEntryArgumentKind::TransportStatus,
         0,
         "direct_dte_status",
         LogicalFormat::U32,
         MemLayout::Tensor,
         {1},
         WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_BYTES,
         WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_ALIGNMENT,
         getTileEntryArgumentAccess(TileEntryArgumentKind::TransportStatus)});
  }

  if (profileRecordBytes) {
    prepared.profileRecordArgumentIndex = function.getNumArguments();
    function.insertArgument(prepared.profileRecordArgumentIndex,
                            mlir::IntegerType::get(function.getContext(), 64),
                            mlir::DictionaryAttr{}, function.getLoc());
    prepared.slots.push_back(
        {static_cast<int64_t>(prepared.slots.size()),
         TileEntryArgumentKind::ProfileRecord,
         0,
         "tx81_profiler_record",
         LogicalFormat::U8,
         MemLayout::Tensor,
         {static_cast<int64_t>(*profileRecordBytes)},
         static_cast<int64_t>(*profileRecordBytes),
         WAFER_TX81_PROFILER_BUFFER_ALIGNMENT,
         getTileEntryArgumentAccess(TileEntryArgumentKind::ProfileRecord)});
  }

  for (unsigned index = 0; index < originalArgumentCount; ++index)
    if (ddrBindings[index])
      function.removeArgAttr(index, kWaferDDRBindingAttrName);
  llvm::SmallVector<mlir::memref::GlobalOp, 4> ddrDeclarations;
  for (mlir::memref::GlobalOp global :
       prepared.module->getOps<mlir::memref::GlobalOp>())
    if (global->hasAttr(kWaferDDRResourceAttrName))
      ddrDeclarations.push_back(global);
  for (mlir::memref::GlobalOp global : ddrDeclarations)
    global.erase();

  if (mlir::failed(mlir::verify(*prepared.module)))
    return mlir::failure();
  return prepared;
}

} // namespace wafer::compiler::detail
