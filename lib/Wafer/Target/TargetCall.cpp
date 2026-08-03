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
  if (const auto *peripheral = std::get_if<InstrPeripheralKind>(&semantic)) {
    if (*peripheral == InstrPeripheralKind::ArgMax ||
        *peripheral == InstrPeripheralKind::ArgMin)
      return LocalInstructionCompletion::SynchronousWriteback;
  }
  return LocalInstructionCompletion::OrderedPending;
}

static std::vector<TargetCallDescriptor> buildDescriptors() {
  std::vector<TargetCallDescriptor> result;
  result.reserve(217);

  auto add = [&](llvm::StringRef stem, Result callResult,
                 std::vector<Scalar> arguments, TargetCallSemantic semantic,
                 TargetCallProfileAvailability availability, bool workerAware) {
    std::optional<TargetCallIssueDomain> issueDomain;
    if (std::optional<TargetCallTSMEngine> engine =
            getTargetCallTSMEngine(semantic)) {
      std::optional<NCCWorker> fixedWorker;
      std::optional<size_t> workerArgument;
      if (*engine != TargetCallTSMEngine::DirectDTE) {
        if (workerAware) {
          workerArgument = arguments.size();
          arguments.push_back(Scalar::I32);
        } else {
          fixedWorker = NCCWorker::Worker0;
        }
      }
      issueDomain = TargetCallIssueDomain{
          *engine, fixedWorker, workerArgument,
          getTargetCallCompletionBehavior(semantic, *engine)};
    }
    result.push_back({("wafer_tx81_" + stem).str(), callResult,
                      std::move(arguments), semantic, issueDomain,
                      availability});
  };
  auto addVoid = [&](llvm::StringRef stem, std::vector<Scalar> arguments,
                     TargetCallSemantic semantic,
                     TargetCallProfileAvailability availability,
                     bool workerAware = false) {
    add(stem, Result::Void, std::move(arguments), semantic, availability,
        workerAware);
  };

  constexpr auto oldOrdinary = TargetCallProfileAvailability::V1AndV2;
  constexpr auto v2Only = TargetCallProfileAvailability::V2;
  constexpr auto allProfiles = TargetCallProfileAvailability::All;
  constexpr auto v3Only = TargetCallProfileAvailability::V3;

  // Keep the original 112 descriptors first, in their exact published order.
  addVoid("rdma", signature(2, 9), TargetCallBuiltin::RDMA, oldOrdinary);
  addVoid("wdma", signature(2, 9), TargetCallBuiltin::WDMA, oldOrdinary);
  addVoid("gather_scatter", signature(2, 14), TargetCallBuiltin::GatherScatter,
          oldOrdinary);
  addVoid("memset", signature(1, 3), TargetCallBuiltin::Memset, oldOrdinary);
  addVoid("bit2fp", signature(2, 2), TargetCallBuiltin::Bit2FP, oldOrdinary);
  addVoid("mask_move",
          {Scalar::I64, Scalar::I32, Scalar::I64, Scalar::I32, Scalar::I32},
          TargetCallBuiltin::MaskMove, oldOrdinary);
  addVoid("gemm", signature(3, 5), TargetCallBuiltin::Gemm, oldOrdinary);
  addVoid("gemm_oriented_v2", signature(3, 7),
          TargetCallBuiltin::GemmOrientedV2, v2Only);
  addVoid("tdma_pad", signature(2, 13), TargetCallBuiltin::TDMAPad,
          oldOrdinary);
  addVoid("tdma_img2col", signature(2, 17), TargetCallBuiltin::TDMAImg2Col,
          oldOrdinary);
  addVoid("local_fence", {}, TargetCallBuiltin::LocalFence, allProfiles);
  addVoid("ncc_join", {Scalar::I32}, TargetCallBuiltin::NCCJoin, allProfiles);

  addVoid("direct_dte_begin", {Scalar::I64, Scalar::I32},
          TargetCallBuiltin::DirectDTEBegin, allProfiles);
  addVoid("direct_dte_begin_after_prepare", {Scalar::I64, Scalar::I32},
          TargetCallBuiltin::DirectDTEBeginAfterPrepare, allProfiles);
  add("direct_dte_send_prepare", Result::I64, signature(2, 5),
      TargetCallBuiltin::DirectDTESendPrepare, allProfiles, false);
  add("direct_dte_recv_prepare", Result::I64, signature(1, 4),
      TargetCallBuiltin::DirectDTERecvPrepare, allProfiles, false);
  addVoid("direct_dte_wait", {Scalar::I64}, TargetCallBuiltin::DirectDTEWait,
          allProfiles);
  addVoid("direct_dte_finish", {}, TargetCallBuiltin::DirectDTEFinish,
          allProfiles);

  auto addEnumSelectedCalls = [&](llvm::StringRef suffix,
                                  TargetCallProfileAvailability availability,
                                  bool workerAware) {
    for (uint32_t value = 0; value <= getMaxEnumValForInstrElementwiseKind();
         ++value) {
      std::optional<InstrElementwiseKind> kind =
          symbolizeInstrElementwiseKind(value);
      if (!kind)
        continue;
      std::string stem = ("elementwise_" + stringifyEnum(*kind) + suffix).str();
      addVoid(stem, signature(isUnaryElementwise(*kind) ? 2 : 3, 2), *kind,
              availability, workerAware);
    }

    for (uint32_t value = 0; value <= getMaxEnumValForInstrReduceKind();
         ++value) {
      std::optional<InstrReduceKind> kind = symbolizeInstrReduceKind(value);
      if (kind)
        addVoid(("reduce_" + stringifyEnum(*kind) + suffix).str(),
                signature(2, 6), *kind, availability, workerAware);
    }

    for (uint32_t value = 0; value <= getMaxEnumValForInstrConvertKind();
         ++value) {
      std::optional<InstrConvertKind> kind = symbolizeInstrConvertKind(value);
      if (kind)
        addVoid(("convert_" + stringifyEnum(*kind) + suffix).str(),
                signature(2, 3), *kind, availability, workerAware);
    }

    for (uint32_t value = 0; value <= getMaxEnumValForInstrConvKind();
         ++value) {
      std::optional<InstrConvKind> kind = symbolizeInstrConvKind(value);
      if (kind)
        addVoid((convStem(*kind) + suffix).str(), signature(3, 28), *kind,
                availability, workerAware);
    }

    for (uint32_t value = 0; value <= getMaxEnumValForInstrPoolKind();
         ++value) {
      std::optional<InstrPoolKind> kind = symbolizeInstrPoolKind(value);
      if (kind) {
        bool indexed = *kind == InstrPoolKind::IndexedMax ||
                       *kind == InstrPoolKind::IndexedMin;
        addVoid(("pool_" + stringifyEnum(*kind) + suffix).str(),
                signature(indexed ? 3 : 2, 18), *kind, availability,
                workerAware);
      }
    }

    for (uint32_t value = 0; value <= getMaxEnumValForInstrUnpoolKind();
         ++value) {
      std::optional<InstrUnpoolKind> kind = symbolizeInstrUnpoolKind(value);
      if (kind)
        addVoid(("unpool_" + stringifyEnum(*kind) + suffix).str(),
                signature(2, 15), *kind, availability, workerAware);
    }

    auto addPeripheral = [&](InstrPeripheralKind kind, unsigned i64Count,
                             unsigned i32Count) {
      addVoid(("peripheral_" + stringifyEnum(kind) + suffix).str(),
              signature(i64Count, i32Count), kind, availability, workerAware);
    };
    addPeripheral(InstrPeripheralKind::ArgMax, 3, 7);
    addPeripheral(InstrPeripheralKind::ArgMin, 3, 7);
    addPeripheral(InstrPeripheralKind::Bilinear, 2, 15);
    addPeripheral(InstrPeripheralKind::Lut16, 3, 7);
    addPeripheral(InstrPeripheralKind::Lut32, 3, 7);
    addPeripheral(InstrPeripheralKind::RandGen, 5, 7);
    addPeripheral(InstrPeripheralKind::ElemMask, 2, 7);
  };
  addEnumSelectedCalls("", oldOrdinary, false);

  assert(result.size() == 112 &&
         "the original target-call registry prefix must stay closed");

  // V3 ordinary calls use coexistable symbols and carry an explicit trailing
  // worker. Shared synchronization and DTE lifecycle calls above keep their
  // existing symbols; explicit DTE issue is a V3-only ABI addition below.
  addVoid("rdma_v3", signature(2, 9), TargetCallBuiltin::RDMA, v3Only, true);
  addVoid("wdma_v3", signature(2, 9), TargetCallBuiltin::WDMA, v3Only, true);
  addVoid("gather_scatter_v3", signature(2, 14),
          TargetCallBuiltin::GatherScatter, v3Only, true);
  addVoid("memset_v3", signature(1, 3), TargetCallBuiltin::Memset, v3Only,
          true);
  addVoid("bit2fp_v3", signature(2, 2), TargetCallBuiltin::Bit2FP, v3Only,
          true);
  addVoid("mask_move_v3",
          {Scalar::I64, Scalar::I32, Scalar::I64, Scalar::I32, Scalar::I32},
          TargetCallBuiltin::MaskMove, v3Only, true);
  addVoid("gemm_v3", signature(3, 5), TargetCallBuiltin::Gemm, v3Only, true);
  addVoid("gemm_oriented_v3", signature(3, 7),
          TargetCallBuiltin::GemmOrientedV2, v3Only, true);
  addVoid("tdma_pad_v3", signature(2, 13), TargetCallBuiltin::TDMAPad, v3Only,
          true);
  addVoid("tdma_img2col_v3", signature(2, 17), TargetCallBuiltin::TDMAImg2Col,
          v3Only, true);

  addEnumSelectedCalls("_v3", v3Only, true);
  addVoid("direct_dte_send_issue_v3", {Scalar::I64},
          TargetCallBuiltin::DirectDTESendIssue, v3Only);

  assert(result.size() == 217 &&
         "versioned target-call registry must stay closed");
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
                       return !descriptor.issueDomain->fixedNCCWorker &&
                              !descriptor.issueDomain->nccWorkerArgument &&
                              descriptor.issueDomain->completionBehavior ==
                                  LocalInstructionCompletion::None;
                     const bool fixedWorker =
                         descriptor.issueDomain->fixedNCCWorker ==
                             NCCWorker::Worker0 &&
                         !descriptor.issueDomain->nccWorkerArgument;
                     const bool argumentWorker =
                         !descriptor.issueDomain->fixedNCCWorker &&
                         descriptor.issueDomain->nccWorkerArgument &&
                         *descriptor.issueDomain->nccWorkerArgument + 1 ==
                             descriptor.arguments.size() &&
                         descriptor.arguments.back() == Scalar::I32;
                     return (fixedWorker || argumentWorker) &&
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
static const TargetCallDescriptor &getDescriptor(SemanticT semantic,
                                                 TargetProfileId profile) {
  const TargetCallDescriptor *descriptor =
      findTargetCallDescriptor(TargetCallSemantic(semantic), profile);
  if (!descriptor)
    llvm::report_fatal_error("target call has no ABI descriptor for profile");
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

bool isTargetCallAvailableForProfile(const TargetCallDescriptor &descriptor,
                                     TargetProfileId targetProfile) {
  uint8_t profileBit = 0;
  if (targetProfile == TargetProfileId::waferTx81SingleCardKernelV1())
    profileBit = static_cast<uint8_t>(TargetCallProfileAvailability::V1);
  else if (targetProfile == TargetProfileId::waferTx81SingleCardKernelV2())
    profileBit = static_cast<uint8_t>(TargetCallProfileAvailability::V2);
  else if (targetProfile == TargetProfileId::waferTx81SingleCardKernelV3())
    profileBit = static_cast<uint8_t>(TargetCallProfileAvailability::V3);
  else
    llvm_unreachable("closed target profile is not registered");

  return (static_cast<uint8_t>(descriptor.availability) & profileBit) != 0;
}

const TargetCallDescriptor *
findTargetCallDescriptor(llvm::StringRef symbol,
                         TargetProfileId targetProfile) {
  const TargetCallDescriptor *descriptor = findTargetCallDescriptor(symbol);
  if (!descriptor ||
      !isTargetCallAvailableForProfile(*descriptor, targetProfile))
    return nullptr;
  return descriptor;
}

const TargetCallDescriptor *
findTargetCallDescriptor(const TargetCallSemantic &semantic,
                         TargetProfileId targetProfile) {
  for (const TargetCallDescriptor &descriptor : getTargetCallDescriptors())
    if (descriptor.semantic == semantic &&
        isTargetCallAvailableForProfile(descriptor, targetProfile))
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
    case TargetCallBuiltin::DirectDTESendIssue:
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

const TargetCallDescriptor &
getTargetCallDescriptor(TargetCallBuiltin call, TargetProfileId targetProfile) {
  return getDescriptor(call, targetProfile);
}

const TargetCallDescriptor &
getTargetCallDescriptor(InstrElementwiseKind kind,
                        TargetProfileId targetProfile) {
  return getDescriptor(kind, targetProfile);
}

const TargetCallDescriptor &
getTargetCallDescriptor(InstrReduceKind kind, TargetProfileId targetProfile) {
  return getDescriptor(kind, targetProfile);
}

bool isTargetReduceFormatTupleAvailable(TargetProfileId targetProfile,
                                        InstrReduceKind kind,
                                        LogicalFormat format) {
  // ABI-only target-profile revisions inherit the same qualified command
  // format rows through the format-compatibility identity.  Keep this exact
  // operation tuple separate from the generic CT x format registry: the
  // latter proves only that a Data_Format field can be encoded.
  if (getTargetProfileRecord(targetProfile).formatCompatibilityProfile !=
      TargetProfileId::waferTx81SingleCardKernelV1())
    return false;
  switch (kind) {
  case InstrReduceKind::Sum:
  case InstrReduceKind::Max:
    return format == LogicalFormat::F16;
  case InstrReduceKind::Min:
  case InstrReduceKind::Avg:
    return format == LogicalFormat::BF16;
  }
  llvm_unreachable("unknown instruction reduce kind");
}

const TargetCallDescriptor &
getTargetCallDescriptor(InstrConvertKind kind, TargetProfileId targetProfile) {
  return getDescriptor(kind, targetProfile);
}

const TargetCallDescriptor &
getTargetCallDescriptor(InstrConvKind kind, TargetProfileId targetProfile) {
  return getDescriptor(kind, targetProfile);
}

const TargetCallDescriptor &
getTargetCallDescriptor(InstrPoolKind kind, TargetProfileId targetProfile) {
  return getDescriptor(kind, targetProfile);
}

const TargetCallDescriptor &
getTargetCallDescriptor(InstrUnpoolKind kind, TargetProfileId targetProfile) {
  return getDescriptor(kind, targetProfile);
}

const TargetCallDescriptor &
getTargetCallDescriptor(InstrPeripheralKind kind,
                        TargetProfileId targetProfile) {
  return getDescriptor(kind, targetProfile);
}

} // namespace wafer
