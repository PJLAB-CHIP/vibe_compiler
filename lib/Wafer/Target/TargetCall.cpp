//===- TargetCall.cpp - Typed target call ABI registry ------------------===//

#include "Wafer/Target/TargetCall.h"

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

static bool isUnaryElementwise(InstrElementwiseKind kind) {
  switch (kind) {
  case InstrElementwiseKind::Abs:
  case InstrElementwiseKind::Recip:
  case InstrElementwiseKind::Square:
  case InstrElementwiseKind::Sqrt:
  case InstrElementwiseKind::Rsqrt:
  case InstrElementwiseKind::Neg:
  case InstrElementwiseKind::LogicNot:
  case InstrElementwiseKind::Log2:
  case InstrElementwiseKind::Ln:
  case InstrElementwiseKind::Pow2:
  case InstrElementwiseKind::Exp:
  case InstrElementwiseKind::ExpLp:
  case InstrElementwiseKind::Sin:
  case InstrElementwiseKind::Cos:
  case InstrElementwiseKind::Tanh:
  case InstrElementwiseKind::Sigmoid:
  case InstrElementwiseKind::Relu:
  case InstrElementwiseKind::SatRelu:
  case InstrElementwiseKind::LeakyRelu:
  case InstrElementwiseKind::Softplus:
    return true;
  case InstrElementwiseKind::Max:
  case InstrElementwiseKind::Min:
  case InstrElementwiseKind::Add:
  case InstrElementwiseKind::Sub:
  case InstrElementwiseKind::Mul:
  case InstrElementwiseKind::Div:
  case InstrElementwiseKind::Eq:
  case InstrElementwiseKind::Ne:
  case InstrElementwiseKind::Ge:
  case InstrElementwiseKind::Gt:
  case InstrElementwiseKind::Le:
  case InstrElementwiseKind::Lt:
  case InstrElementwiseKind::LogicAnd:
  case InstrElementwiseKind::LogicOr:
  case InstrElementwiseKind::LogicXor:
    return false;
  }
  llvm_unreachable("unknown elementwise kind");
}

static llvm::StringRef convStem(InstrConvKind kind) {
  switch (kind) {
  case InstrConvKind::Conv:
    return "conv";
  case InstrConvKind::Depthwise:
    return "depthwise_conv";
  case InstrConvKind::BackwardConv:
    return "backward_conv";
  }
  llvm_unreachable("unknown convolution kind");
}

static LocalInstructionCompletion
getTargetCallCompletionBehavior(const TargetCallSemantic &semantic,
                                TargetCallTSMEngine engine) {
  if (engine == TargetCallTSMEngine::DirectDTE)
    return LocalInstructionCompletion::None;
  if (const auto *peripheral =
          std::get_if<InstrPeripheralKind>(&semantic)) {
    if (*peripheral == InstrPeripheralKind::ArgMax ||
        *peripheral == InstrPeripheralKind::ArgMin)
      return LocalInstructionCompletion::SynchronousWriteback;
  }
  return LocalInstructionCompletion::OrderedPending;
}

static std::vector<TargetCallDescriptor> buildDescriptors() {
  std::vector<TargetCallDescriptor> result;
  result.reserve(112);

  auto add = [&](llvm::StringRef stem, Result callResult,
                 std::vector<Scalar> arguments, TargetCallSemantic semantic) {
    std::optional<TargetCallIssueDomain> issueDomain;
    if (std::optional<TargetCallTSMEngine> engine =
            getTargetCallTSMEngine(semantic)) {
      std::optional<NCCWorker> worker;
      if (*engine != TargetCallTSMEngine::DirectDTE)
        worker = NCCWorker::Worker0;
      issueDomain = TargetCallIssueDomain{
          *engine, worker, getTargetCallCompletionBehavior(semantic, *engine)};
    }
    result.push_back({("wafer_tx81_" + stem).str(), callResult,
                      std::move(arguments), semantic, issueDomain});
  };
  auto addVoid = [&](llvm::StringRef stem, std::vector<Scalar> arguments,
                     TargetCallSemantic semantic) {
    add(stem, Result::Void, std::move(arguments), semantic);
  };

  addVoid("rdma", signature(2, 9), TargetCallBuiltin::RDMA);
  addVoid("wdma", signature(2, 9), TargetCallBuiltin::WDMA);
  addVoid("gather_scatter", signature(2, 14), TargetCallBuiltin::GatherScatter);
  addVoid("memset", signature(1, 3), TargetCallBuiltin::Memset);
  addVoid("bit2fp", signature(2, 2), TargetCallBuiltin::Bit2FP);
  addVoid("mask_move",
          {Scalar::I64, Scalar::I32, Scalar::I64, Scalar::I32, Scalar::I32},
          TargetCallBuiltin::MaskMove);
  addVoid("gemm", signature(3, 5), TargetCallBuiltin::Gemm);
  addVoid("gemm_oriented_v2", signature(3, 7),
          TargetCallBuiltin::GemmOrientedV2);
  addVoid("tdma_pad", signature(2, 13), TargetCallBuiltin::TDMAPad);
  addVoid("tdma_img2col", signature(2, 17), TargetCallBuiltin::TDMAImg2Col);
  addVoid("local_fence", {}, TargetCallBuiltin::LocalFence);
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

  for (uint32_t value = 0; value <= getMaxEnumValForInstrElementwiseKind();
       ++value) {
    std::optional<InstrElementwiseKind> kind =
        symbolizeInstrElementwiseKind(value);
    if (!kind)
      continue;
    std::string stem = ("elementwise_" + stringifyEnum(*kind)).str();
    addVoid(stem, signature(isUnaryElementwise(*kind) ? 2 : 3, 2), *kind);
  }

  for (uint32_t value = 0; value <= getMaxEnumValForInstrReduceKind();
       ++value) {
    std::optional<InstrReduceKind> kind = symbolizeInstrReduceKind(value);
    if (kind)
      addVoid(("reduce_" + stringifyEnum(*kind)).str(), signature(2, 6), *kind);
  }

  for (uint32_t value = 0; value <= getMaxEnumValForInstrConvertKind();
       ++value) {
    std::optional<InstrConvertKind> kind = symbolizeInstrConvertKind(value);
    if (kind)
      addVoid(("convert_" + stringifyEnum(*kind)).str(), signature(2, 3),
              *kind);
  }

  for (uint32_t value = 0; value <= getMaxEnumValForInstrConvKind(); ++value) {
    std::optional<InstrConvKind> kind = symbolizeInstrConvKind(value);
    if (kind)
      addVoid(convStem(*kind), signature(3, 28), *kind);
  }

  for (uint32_t value = 0; value <= getMaxEnumValForInstrPoolKind(); ++value) {
    std::optional<InstrPoolKind> kind = symbolizeInstrPoolKind(value);
    if (kind) {
      bool indexed = *kind == InstrPoolKind::IndexedMax ||
                     *kind == InstrPoolKind::IndexedMin;
      addVoid(("pool_" + stringifyEnum(*kind)).str(),
              signature(indexed ? 3 : 2, 18), *kind);
    }
  }

  for (uint32_t value = 0; value <= getMaxEnumValForInstrUnpoolKind();
       ++value) {
    std::optional<InstrUnpoolKind> kind = symbolizeInstrUnpoolKind(value);
    if (kind)
      addVoid(("unpool_" + stringifyEnum(*kind)).str(), signature(2, 15),
              *kind);
  }

  auto addPeripheral = [&](InstrPeripheralKind kind, unsigned i64Count,
                           unsigned i32Count) {
    addVoid(("peripheral_" + stringifyEnum(kind)).str(),
            signature(i64Count, i32Count), kind);
  };
  addPeripheral(InstrPeripheralKind::ArgMax, 3, 7);
  addPeripheral(InstrPeripheralKind::ArgMin, 3, 7);
  addPeripheral(InstrPeripheralKind::Bilinear, 2, 15);
  addPeripheral(InstrPeripheralKind::Lut16, 3, 7);
  addPeripheral(InstrPeripheralKind::Lut32, 3, 7);
  addPeripheral(InstrPeripheralKind::RandGen, 5, 7);
  addPeripheral(InstrPeripheralKind::ElemMask, 2, 7);

  assert(result.size() == 112 && "target-call registry must stay closed");
  assert(llvm::all_of(
             result,
             [&](const TargetCallDescriptor &descriptor) {
               const std::optional<TargetCallTSMEngine> semanticEngine =
                   getTargetCallTSMEngine(descriptor.semantic);
               if (!semanticEngine)
                 return !descriptor.issueDomain;
               if (!descriptor.issueDomain ||
                   descriptor.issueDomain->engine != *semanticEngine)
                 return false;
               if (*semanticEngine == TargetCallTSMEngine::DirectDTE)
                 return !descriptor.issueDomain->nccWorker &&
                        descriptor.issueDomain->completionBehavior ==
                            LocalInstructionCompletion::None;
               return descriptor.issueDomain->nccWorker.has_value() &&
                      descriptor.issueDomain->completionBehavior ==
                          getTargetCallCompletionBehavior(descriptor.semantic,
                                                          *semanticEngine);
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
    llvm::report_fatal_error("target call has no registered ABI descriptor");
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
    case TargetCallBuiltin::GemmOrientedV2:
      return TargetCallTSMEngine::NE;
    case TargetCallBuiltin::Bit2FP:
    case TargetCallBuiltin::MaskMove:
      return TargetCallTSMEngine::CT;
    case TargetCallBuiltin::DirectDTEWait:
      return TargetCallTSMEngine::DirectDTE;
    case TargetCallBuiltin::LocalFence:
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
  if (std::holds_alternative<InstrConvKind>(semantic))
    return TargetCallTSMEngine::NE;
  if (std::holds_alternative<InstrElementwiseKind>(semantic) ||
      std::holds_alternative<InstrReduceKind>(semantic) ||
      std::holds_alternative<InstrConvertKind>(semantic) ||
      std::holds_alternative<InstrPoolKind>(semantic) ||
      std::holds_alternative<InstrUnpoolKind>(semantic) ||
      std::holds_alternative<InstrPeripheralKind>(semantic))
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

const TargetCallDescriptor &getTargetCallDescriptor(InstrElementwiseKind kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &getTargetCallDescriptor(InstrReduceKind kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &getTargetCallDescriptor(InstrConvertKind kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &getTargetCallDescriptor(InstrConvKind kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &getTargetCallDescriptor(InstrPoolKind kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &getTargetCallDescriptor(InstrUnpoolKind kind) {
  return getDescriptor(kind);
}

const TargetCallDescriptor &getTargetCallDescriptor(InstrPeripheralKind kind) {
  return getDescriptor(kind);
}

} // namespace wafer
