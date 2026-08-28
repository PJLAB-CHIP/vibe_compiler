//===- Target LLVM lowering implementation -------------------------------===//

#include "Wafer/Conversion/InstrToLLVM/LowerInstrToTargetLLVMInternal.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/IR/Topology/TargetTopology.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Target/TargetFormat.h"
#include "Wafer/Target/TargetMemory.h"
#include "Wafer/Conversion/InstrToLLVM/InstrToLLVM.h"

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

namespace wafer::target_llvm_detail {

mlir::FailureOr<DirectDTEEndpointDomain>
resolveDirectDTEEndpointDomain(const TargetTopology &topology,
                               mlir::ModuleOp diagnosticModule, CardId cardId,
                               TileId tileId) {
  std::optional<llvm::ArrayRef<TileId>> available =
      topology.getAvailableTileIds(cardId);
  if (!available || !llvm::is_contained(*available, tileId))
    return diagnosticModule.emitError()
           << "unsupported_target_transport: current Tile is not "
              "available in the selected card topology";
  DirectDTEEndpointDomain domain;
  domain.cardId = cardId;
  domain.tileId = tileId;
  domain.availableTileIds.assign(available->begin(), available->end());
  return domain;
}
mlir::FailureOr<mlir::Value>
FunctionLowering::lowerDTESend(InstrDTESendOp op,
                               const DirectDTEEndpointDomain &domain) {
  DirectDTEBindingAttr binding =
      op.getBinding().value_or(DirectDTEBindingAttr());
  if (!binding)
    return op.emitError()
           << "unsupported_target_transport: Direct DTE send is missing "
              "an accepted physical binding";
  if (binding.getAllocationProfile() != DTEAllocationProfile::Normal ||
      binding.getCompletionProfile() !=
          DTECompletionProfile::SenderWaitReceiverFSM)
    return op.emitError()
           << "unsupported_target_transport: Direct DTE binding profile "
              "is not supported by the target CRT";
  if (op.getBytesAttr().getInt() <= 0 ||
      op.getBytesAttr().getInt() > std::numeric_limits<int32_t>::max())
    return op.emitError()
           << "target_abi_narrowing: Direct DTE byte count must fit a "
              "positive int32_t packet size";
  int64_t peer = op.getPeerAttr().getInt();
  const TileId peerTileId(peer);
  if (peer < 0 || !llvm::is_contained(domain.availableTileIds, peerTileId) ||
      peerTileId == domain.tileId)
    return op.emitError()
           << "unsupported_target_transport: Direct DTE peer is outside "
              "the accepted Tile domain";
  mlir::FailureOr<mlir::Value> source =
      materializeAddress(op, op.getBuffer(), "direct DTE send source");
  if (mlir::failed(source))
    return mlir::failure();

  mlir::Value remoteDestination;
  mlir::Value remoteReceiverFsm =
      constantI32(op.getLoc(), binding.getReceiverFsmId());
  switch (binding.getRemoteAddressMode()) {
  case DTERemoteAddressMode::Absolute: {
    int64_t remoteEnd = 0;
    const TargetMemoryPolicy targetMemory = getTargetMemoryPolicy();
    if (!checkedAdd(binding.getRemoteReceiverAddress(),
                    op.getBytesAttr().getInt(), remoteEnd) ||
        binding.getRemoteReceiverAddress() < targetMemory.spmBase ||
        remoteEnd > targetMemory.spmLimit)
      return op.emitError()
             << "unsupported_target_transport: Direct DTE absolute remote "
                "receiver range is outside target SPM";
    remoteDestination =
        constantI64(op.getLoc(), binding.getRemoteReceiverAddress());
    break;
  }
  case DTERemoteAddressMode::SourceRelative:
    remoteDestination = builder.create<mlir::LLVM::AddOp>(
        op.getLoc(), *source,
        constantI64(op.getLoc(), binding.getRemoteReceiverAddress()));
    break;
  case DTERemoteAddressMode::SelectorTable: {
    mlir::Value selector = op.getBindingSelector();
    if (!selector)
      return op.emitError(
          "unsupported_target_transport: Direct DTE selector-table binding "
          "requires one converted i64 selector");
    auto convertedSelector = convertedValues.find(selector);
    if (convertedSelector == convertedValues.end())
      return op.emitError(
          "unsupported_target_transport: Direct DTE selector-table binding "
          "requires one converted i64 selector");
    llvm::ArrayRef<int64_t> table = binding.getRouteBindings().asArrayRef();
    if (table.empty() || table.size() % 3 != 0)
      return op.emitError(
          "unsupported_target_transport: Direct DTE route binding "
          "table is malformed");
    const TargetMemoryPolicy targetMemory = getTargetMemoryPolicy();
    auto setFallback = [&](size_t index) -> mlir::LogicalResult {
      int64_t remoteEnd = 0;
      if (!checkedAdd(table[index + 1], op.getBytesAttr().getInt(),
                      remoteEnd) ||
          table[index + 1] < targetMemory.spmBase ||
          remoteEnd > targetMemory.spmLimit || table[index + 2] < 0 ||
          table[index + 2] > 3)
        return op.emitError(
            "unsupported_target_transport: Direct DTE route binding "
            "entry is outside the target SPM/FSM domain");
      remoteDestination = constantI64(op.getLoc(), table[index + 1]);
      remoteReceiverFsm = constantI32(op.getLoc(), table[index + 2]);
      return mlir::success();
    };
    size_t fallback = table.size() - 3;
    if (mlir::failed(setFallback(fallback)))
      return mlir::failure();
    mlir::Value routeSelector = builder.create<mlir::LLVM::URemOp>(
        op.getLoc(), convertedSelector->second,
        constantI64(op.getLoc(), table.size() / 3));
    for (size_t index = fallback; index != 0;) {
      index -= 3;
      int64_t remoteEnd = 0;
      if (!checkedAdd(table[index + 1], op.getBytesAttr().getInt(),
                      remoteEnd) ||
          table[index + 1] < targetMemory.spmBase ||
          remoteEnd > targetMemory.spmLimit || table[index + 2] < 0 ||
          table[index + 2] > 3)
        return op.emitError(
            "unsupported_target_transport: Direct DTE route binding "
            "entry is outside the target SPM/FSM domain");
      mlir::Value selected = builder.create<mlir::LLVM::ICmpOp>(
          op.getLoc(), mlir::LLVM::ICmpPredicate::eq, routeSelector,
          constantI64(op.getLoc(), table[index]));
      remoteDestination = builder.create<mlir::LLVM::SelectOp>(
          op.getLoc(), selected, constantI64(op.getLoc(), table[index + 1]),
          remoteDestination);
      remoteReceiverFsm = builder.create<mlir::LLVM::SelectOp>(
          op.getLoc(), selected, constantI32(op.getLoc(), table[index + 2]),
          remoteReceiverFsm);
    }
    break;
  }
  }

  llvm::SmallVector<mlir::Value, 8> args;
  args.push_back(*source);
  args.push_back(remoteDestination);
  appendI32(op.getLoc(), args, op.getBytesAttr().getInt());
  appendI32(op.getLoc(), args, domain.tileId.getValue());
  appendI32(op.getLoc(), args, peerTileId.getValue());
  args.push_back(remoteReceiverFsm);
  appendI32(op.getLoc(), args, /*isHighPerformance=*/0);
  mlir::Value event = emitI64Call(
      op.getLoc(),
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTESendPrepare), args);
  emitCall(op.getLoc(),
           getTargetCallDescriptor(TargetCallBuiltin::DirectDTESendIssue),
           mlir::ValueRange(event));
  return event;
}

mlir::FailureOr<mlir::Value>
FunctionLowering::lowerDTERecv(InstrDTERecvOp op,
                               const DirectDTEEndpointDomain &domain) {
  DirectDTEBindingAttr binding =
      op.getBinding().value_or(DirectDTEBindingAttr());
  if (!binding)
    return op.emitError()
           << "unsupported_target_transport: Direct DTE receive is missing "
              "an accepted physical binding";
  if (binding.getAllocationProfile() != DTEAllocationProfile::Normal ||
      binding.getCompletionProfile() !=
          DTECompletionProfile::SenderWaitReceiverFSM)
    return op.emitError()
           << "unsupported_target_transport: Direct DTE binding profile "
              "is not supported by the target CRT";
  if (op.getBytesAttr().getInt() <= 0 ||
      op.getBytesAttr().getInt() > std::numeric_limits<int32_t>::max())
    return op.emitError()
           << "target_abi_narrowing: Direct DTE byte count must fit a "
              "positive int32_t packet size";
  int64_t peer = op.getPeerAttr().getInt();
  const TileId peerTileId(peer);
  if (peer < 0 || !llvm::is_contained(domain.availableTileIds, peerTileId) ||
      peerTileId == domain.tileId)
    return op.emitError()
           << "unsupported_target_transport: Direct DTE peer is outside "
              "the accepted Tile domain";
  mlir::FailureOr<mlir::Value> destination =
      materializeAddress(op, op.getBuffer(), "direct DTE receive destination");
  if (mlir::failed(destination))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 8> args;
  args.push_back(*destination);
  appendI32(op.getLoc(), args, op.getBytesAttr().getInt());
  appendI32(op.getLoc(), args, domain.tileId.getValue());
  appendI32(op.getLoc(), args, peerTileId.getValue());
  appendI32(op.getLoc(), args, binding.getReceiverFsmId());
  return emitI64Call(
      op.getLoc(),
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTERecvPrepare), args);
}

mlir::LogicalResult
FunctionLowering::lowerDTEWait(InstrDTEWaitOp op,
                               llvm::ArrayRef<mlir::Value> events) {
  if (events.size() != op.getTokens().size())
    return op.emitError()
           << "target_llvm_lowering_failure: Direct DTE wait event count "
              "does not match its token count";
  for (mlir::Value event : events)
    emitCall(op.getLoc(),
             getTargetCallDescriptor(TargetCallBuiltin::DirectDTEWait),
             mlir::ValueRange(event));
  return mlir::success();
}

static void registerTargetCallee(mlir::MLIRContext *context,
                                 llvm::StringMap<CalleeSignature> &callees,
                                 const TargetCallDescriptor &descriptor) {
  if (descriptor.result != TargetCallResultType::Void)
    llvm_unreachable("target lifecycle call must return void");
  llvm::SmallVector<mlir::Type, 4> arguments;
  for (TargetCallScalarType scalar : descriptor.arguments)
    arguments.push_back(mlir::IntegerType::get(
        context, scalar == TargetCallScalarType::I64 ? 64 : 32));
  auto type = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(context), arguments,
      /*isVarArg=*/false);
  auto [it, inserted] =
      callees.try_emplace(descriptor.symbol, CalleeSignature{type});
  if (!inserted && it->second.type != type)
    llvm_unreachable("target transport lifecycle symbol type mismatch");
}

mlir::LogicalResult injectDirectDTEStatusLifecycle(
    mlir::ModuleOp moduleOp, llvm::StringRef entrySymbol,
    int64_t statusArgumentIndex, int64_t participantCount,
    TargetCallBuiltin beginBuiltin,
    llvm::StringMap<CalleeSignature> &usedCallees) {
  auto entry = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(entrySymbol);
  if (!entry || entry.isDeclaration() || entry.getBody().empty() ||
      statusArgumentIndex < 0 ||
      statusArgumentIndex >=
          static_cast<int64_t>(entry.getBody().front().getNumArguments()))
    return moduleOp.emitError()
           << "unsupported_target_transport: Direct DTE status entry "
              "argument is missing after LLVM conversion";
  mlir::Value status = entry.getBody().front().getArgument(
      static_cast<unsigned>(statusArgumentIndex));
  if (!status.getType().isInteger(64))
    return entry.emitError()
           << "target_abi_mismatch: Direct DTE status argument must be i64";

  mlir::OpBuilder builder(moduleOp.getContext());
  mlir::Type i32Type = mlir::IntegerType::get(moduleOp.getContext(), 32);
  const TargetCallDescriptor &begin = getTargetCallDescriptor(beginBuiltin);
  const TargetCallDescriptor &finish =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTEFinish);
  registerTargetCallee(moduleOp.getContext(), usedCallees, begin);
  registerTargetCallee(moduleOp.getContext(), usedCallees, finish);

  builder.setInsertionPointToStart(&entry.getBody().front());
  mlir::Value count = builder.create<mlir::LLVM::ConstantOp>(
      entry.getLoc(), i32Type, participantCount);
  builder.create<mlir::LLVM::CallOp>(
      entry.getLoc(), mlir::TypeRange(),
      mlir::FlatSymbolRefAttr::get(moduleOp.getContext(), begin.symbol),
      mlir::ValueRange{status, count});

  llvm::SmallVector<mlir::LLVM::ReturnOp, 4> returns;
  entry.walk(
      [&](mlir::LLVM::ReturnOp returnOp) { returns.push_back(returnOp); });
  if (returns.empty())
    return entry.emitError()
           << "unsupported_target_transport: Direct DTE entry has no return";
  for (mlir::LLVM::ReturnOp returnOp : returns) {
    builder.setInsertionPoint(returnOp);
    builder.create<mlir::LLVM::CallOp>(
        returnOp.getLoc(), mlir::TypeRange(),
        mlir::FlatSymbolRefAttr::get(moduleOp.getContext(), finish.symbol),
        mlir::ValueRange());
  }
  return mlir::success();
}

} // namespace wafer::target_llvm_detail
