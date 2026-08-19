//===- TargetCall.cpp - Typed target call ABI registry ------------------===//

#include "Wafer/Target/Core/TargetCall.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <utility>

namespace wafer {
namespace {

using Scalar = TargetCallScalarType;
using Result = TargetCallResultType;

static std::vector<Scalar> signature(unsigned i64Count, unsigned i32Count) {
  std::vector<Scalar> result(i64Count, Scalar::I64);
  result.insert(result.end(), i32Count, Scalar::I32);
  return result;
}

static TargetNCCCompletionBehavior
getTargetCallCompletionBehavior(const TargetCallSemantic &semantic,
                                TargetCallTSMEngine engine) {
  if (engine == TargetCallTSMEngine::DirectDTE)
    return TargetNCCCompletionBehavior::None;
  if (const auto *peripheral =
          std::get_if<TargetPeripheralOperation>(&semantic)) {
    if (*peripheral == TargetPeripheralOperation::ArgMaximum ||
        *peripheral == TargetPeripheralOperation::ArgMinimum)
      return TargetNCCCompletionBehavior::SynchronousWriteback;
  }
  return TargetNCCCompletionBehavior::OrderedAsynchronousIssue;
}

static std::vector<TargetCallDescriptor> buildDescriptors() {
  std::vector<TargetCallDescriptor> result;
  result.reserve(112);

  auto add = [&](llvm::StringRef stem, Result callResult,
                 std::vector<Scalar> arguments, TargetCallSemantic semantic) {
    std::optional<TargetCallIssueDomain> issueDomain;
    if (std::optional<TargetCallTSMEngine> engine =
            getTargetCallTSMEngine(semantic)) {
      std::optional<size_t> workerArgument;
      if (*engine != TargetCallTSMEngine::DirectDTE) {
        workerArgument = arguments.size();
        arguments.push_back(Scalar::I32);
      }
      issueDomain = TargetCallIssueDomain{
          *engine, workerArgument,
          getTargetCallCompletionBehavior(semantic, *engine)};
    }
    result.push_back({("wafer_tx81_" + stem).str(), callResult,
                      std::move(arguments), semantic, issueDomain});
  };
  auto addVoid = [&](llvm::StringRef stem, std::vector<Scalar> arguments,
                     TargetCallSemantic semantic) {
    add(stem, Result::Void, std::move(arguments), semantic);
  };

  // Synchronization and Direct-DTE lifecycle calls are shared by every
  // Tile execution.
  addVoid("ncc_join", {Scalar::I32}, TargetCallBuiltin::NCCJoin);

  addVoid("direct_dte_begin", {Scalar::I64, Scalar::I32},
          TargetCallBuiltin::DirectDTEBegin);
  addVoid("direct_dte_begin_after_prepare", {Scalar::I64, Scalar::I32},
          TargetCallBuiltin::DirectDTEBeginAfterPrepare);
  add("direct_dte_send_prepare", Result::I64, signature(2, 5),
      TargetCallBuiltin::DirectDTESendPrepare);
  add("direct_dte_recv_prepare", Result::I64, signature(1, 4),
      TargetCallBuiltin::DirectDTERecvPrepare);
  addVoid("direct_dte_wait", {Scalar::I64}, TargetCallBuiltin::DirectDTEWait);
  addVoid("direct_dte_finish", {}, TargetCallBuiltin::DirectDTEFinish);

  auto addEnumSelectedCalls = [&] {
    for (NumericElementwiseOperation operation :
         getNumericElementwiseOperations())
      addVoid(("elementwise_" +
               stringifyNumericElementwiseOperation(operation))
                  .str(),
              signature(getNumericElementwiseArity(operation) == 1 ? 2 : 3, 2),
              operation);

    for (NumericReduceOperation operation : getNumericReduceOperations())
      addVoid(("reduce_" + stringifyNumericReduceOperation(operation)).str(),
              signature(2, 6), operation);

    for (const TargetConvertRoute &route : getTargetConvertRoutes()) {
      TargetConvertOperation operation =
          llvm::cantFail(TargetConvertOperation::create(route.opcode));
      addVoid(("convert_" + route.canonicalSpelling).str(),
              signature(2, 3), operation);
    }

    for (TargetConvolutionOperation operation :
         getTargetConvolutionOperations())
      addVoid(stringifyTargetConvolutionOperation(operation).str(),
              signature(3, 28), operation);

    for (TargetPoolingOperation operation : getTargetPoolingOperations()) {
      bool indexed = operation == TargetPoolingOperation::IndexedMaximum ||
                     operation == TargetPoolingOperation::IndexedMinimum;
      addVoid(
          ("pool_" + stringifyTargetPoolingOperation(operation)).str(),
          signature(indexed ? 3 : 2, 18), operation);
    }

    for (TargetUnpoolingOperation operation : getTargetUnpoolingOperations())
      addVoid(
          ("unpool_" + stringifyTargetUnpoolingOperation(operation)).str(),
          signature(2, 15), operation);

    auto addPeripheral = [&](TargetPeripheralOperation kind, unsigned i64Count,
                             unsigned i32Count) {
      addVoid(
          ("peripheral_" + stringifyTargetPeripheralOperation(kind)).str(),
          signature(i64Count, i32Count), kind);
    };
    addPeripheral(TargetPeripheralOperation::ArgMaximum, 3, 7);
    addPeripheral(TargetPeripheralOperation::ArgMinimum, 3, 7);
    addPeripheral(TargetPeripheralOperation::Bilinear, 2, 15);
    addPeripheral(TargetPeripheralOperation::LookupTable16, 3, 7);
    addPeripheral(TargetPeripheralOperation::LookupTable32, 3, 7);
    addPeripheral(TargetPeripheralOperation::Random, 5, 7);
    addPeripheral(TargetPeripheralOperation::ElementMask, 2, 7);
  };
  // The current ordinary instruction ABI carries an explicit trailing worker.
  addVoid("rdma", signature(2, 9), TargetCallBuiltin::RDMA);
  addVoid("wdma", signature(2, 9), TargetCallBuiltin::WDMA);
  addVoid("gather_scatter", signature(2, 14),
          TargetCallBuiltin::GatherScatter);
  addVoid("memset", signature(1, 3), TargetCallBuiltin::Memset);
  addVoid("bit2fp", signature(2, 2), TargetCallBuiltin::Bit2FP);
  addVoid("mask_move",
          {Scalar::I64, Scalar::I32, Scalar::I64, Scalar::I32, Scalar::I32},
          TargetCallBuiltin::MaskMove);
  addVoid("gemm", signature(3, 5), TargetCallBuiltin::Gemm);
  addVoid("gemm_oriented", signature(3, 7), TargetCallBuiltin::GemmOriented);
  addVoid("tdma_pad", signature(2, 13), TargetCallBuiltin::TDMAPad);
  addVoid("tdma_img2col", signature(2, 17), TargetCallBuiltin::TDMAImg2Col);

  addEnumSelectedCalls();
  addVoid("direct_dte_send_issue", {Scalar::I64},
          TargetCallBuiltin::DirectDTESendIssue);

  assert(result.size() == 112 && "target-call registry must stay closed");
  assert(
      llvm::all_of(result,
                   [&](const TargetCallDescriptor &descriptor) {
                     const std::optional<TargetCallTSMEngine> semanticEngine =
                         getTargetCallTSMEngine(descriptor.semantic);
                     if (!semanticEngine)
                       return !descriptor.issueDomain;
                     if (!descriptor.issueDomain ||
                         descriptor.issueDomain->engine != *semanticEngine)
                       return false;
                     if (*semanticEngine == TargetCallTSMEngine::DirectDTE)
                       return !descriptor.issueDomain->nccWorkerArgument &&
                              descriptor.issueDomain->completionBehavior ==
                                  TargetNCCCompletionBehavior::None;
                     const bool argumentWorker =
                         descriptor.issueDomain->nccWorkerArgument &&
                         *descriptor.issueDomain->nccWorkerArgument + 1 ==
                             descriptor.arguments.size() &&
                         descriptor.arguments.back() == Scalar::I32;
                     return argumentWorker &&
                            descriptor.issueDomain->completionBehavior ==
                                getTargetCallCompletionBehavior(
                                    descriptor.semantic, *semanticEngine);
                   }) &&
      "target-call issue/completion metadata must cover every engine command");
  assert(llvm::all_of(
             result,
             [&](const TargetCallDescriptor &descriptor) {
               return llvm::count_if(
                          result, [&](const TargetCallDescriptor &candidate) {
                            return candidate.symbol == descriptor.symbol;
                          }) == 1;
             }) &&
         "target-call symbols must be unique");
  return result;
}

template <typename SemanticT>
static const TargetCallDescriptor &getDescriptor(SemanticT semantic) {
  const TargetCallDescriptor *descriptor =
      findTargetCallDescriptor(TargetCallSemantic(semantic));
  if (!descriptor)
    llvm::report_fatal_error("target call has no ABI descriptor");
  return *descriptor;
}

} // namespace

llvm::ArrayRef<TargetCallDescriptor> getTargetCallDescriptors() {
  static const std::vector<TargetCallDescriptor> descriptors =
      buildDescriptors();
  return descriptors;
}

const TargetCallDescriptor *findTargetCallDescriptor(llvm::StringRef symbol) {
  for (const TargetCallDescriptor &descriptor : getTargetCallDescriptors())
    if (descriptor.symbol == symbol)
      return &descriptor;
  return nullptr;
}

const TargetCallDescriptor *
findTargetCallDescriptor(const TargetCallSemantic &semantic) {
  for (const TargetCallDescriptor &descriptor : getTargetCallDescriptors())
    if (descriptor.semantic == semantic)
      return &descriptor;
  return nullptr;
}

std::optional<TargetCallTSMEngine>
getTargetCallTSMEngine(const TargetCallSemantic &semantic) {
  if (const auto *builtin = std::get_if<TargetCallBuiltin>(&semantic)) {
    switch (*builtin) {
    case TargetCallBuiltin::RDMA:
      return TargetCallTSMEngine::RDMA;
    case TargetCallBuiltin::WDMA:
      return TargetCallTSMEngine::WDMA;
    case TargetCallBuiltin::GatherScatter:
    case TargetCallBuiltin::Memset:
    case TargetCallBuiltin::TDMAPad:
    case TargetCallBuiltin::TDMAImg2Col:
      return TargetCallTSMEngine::TDMA;
    case TargetCallBuiltin::Gemm:
    case TargetCallBuiltin::GemmOriented:
      return TargetCallTSMEngine::NE;
    case TargetCallBuiltin::Bit2FP:
    case TargetCallBuiltin::MaskMove:
      return TargetCallTSMEngine::CT;
    case TargetCallBuiltin::DirectDTESendIssue:
    case TargetCallBuiltin::DirectDTEWait:
      return TargetCallTSMEngine::DirectDTE;
    case TargetCallBuiltin::NCCJoin:
    case TargetCallBuiltin::DirectDTEBegin:
    case TargetCallBuiltin::DirectDTEBeginAfterPrepare:
    case TargetCallBuiltin::DirectDTESendPrepare:
    case TargetCallBuiltin::DirectDTERecvPrepare:
    case TargetCallBuiltin::DirectDTEFinish:
      return std::nullopt;
    }
    llvm_unreachable("unknown target-call builtin");
  }
  if (std::holds_alternative<TargetConvolutionOperation>(semantic))
    return TargetCallTSMEngine::NE;
  if (std::holds_alternative<NumericElementwiseOperation>(semantic) ||
      std::holds_alternative<NumericReduceOperation>(semantic) ||
      std::holds_alternative<TargetConvertOperation>(semantic) ||
      std::holds_alternative<TargetPoolingOperation>(semantic) ||
      std::holds_alternative<TargetUnpoolingOperation>(semantic) ||
      std::holds_alternative<TargetPeripheralOperation>(semantic))
    return TargetCallTSMEngine::CT;
  llvm_unreachable("unknown target-call semantic");
}

std::optional<TargetCallTSMEngine>
getTargetCallTSMEngine(const TargetCallDescriptor &descriptor) {
  if (!descriptor.issueDomain)
    return std::nullopt;
  return descriptor.issueDomain->engine;
}

llvm::StringRef stringifyTargetCallTSMEngine(TargetCallTSMEngine engine) {
  switch (engine) {
  case TargetCallTSMEngine::CT:
    return "CT";
  case TargetCallTSMEngine::NE:
    return "NE";
  case TargetCallTSMEngine::RDMA:
    return "RDMA";
  case TargetCallTSMEngine::WDMA:
    return "WDMA";
  case TargetCallTSMEngine::TDMA:
    return "TDMA";
  case TargetCallTSMEngine::DirectDTE:
    return "DIRECT_DTE";
  }
  llvm_unreachable("unknown target-call TSM engine");
}

const TargetCallDescriptor &getTargetCallDescriptor(TargetCallBuiltin call) {
  return getDescriptor(call);
}

const TargetCallDescriptor &
getTargetCallDescriptor(NumericElementwiseOperation kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &
getTargetCallDescriptor(NumericReduceOperation kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &
getTargetCallDescriptor(TargetConvertOperation kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &
getTargetCallDescriptor(TargetConvolutionOperation kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &
getTargetCallDescriptor(TargetPoolingOperation kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &
getTargetCallDescriptor(TargetUnpoolingOperation kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &
getTargetCallDescriptor(TargetPeripheralOperation kind) {
  return getDescriptor(kind);
}

} // namespace wafer
