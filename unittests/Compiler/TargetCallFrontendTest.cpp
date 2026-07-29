//===- TargetCallFrontendTest.cpp - Host target-call frontend tests -----===//

#include "Wafer/Compiler/TargetCallFrontend.h"
#include "Wafer/InitAll.h"
#include "Wafer/Target/TargetCall.h"

#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"
#include "../../lib/Wafer/Compiler/TargetArtifactInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

wafer::frontend::ProgramRankSlice singleRankSlice() {
  wafer::frontend::ProgramRankSlice slice;
  slice.logicalRank = 0;
  slice.replicaId = 0;
  slice.offsets = {0};
  slice.sizes = {8};
  slice.strides = {1};
  return slice;
}

wafer::frontend::ProgramBoundaryBinding boundary(int64_t index) {
  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape = {8};
  binding.localShape = {8};
  binding.dtype = "f32";
  binding.rankSlices.push_back(singleRankSlice());
  return binding;
}

wafer::frontend::ProgramBoundaryBinding partitionedBoundary(int64_t index) {
  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Partitioned;
  binding.globalShape = {64};
  binding.localShape = {4};
  binding.dtype = "f32";
  for (int64_t rank = 0; rank < 16; ++rank) {
    wafer::frontend::ProgramRankSlice slice;
    slice.logicalRank = rank;
    slice.replicaId = 0;
    slice.offsets = {rank * 4};
    slice.sizes = {4};
    slice.strides = {1};
    binding.rankSlices.push_back(std::move(slice));
  }
  return binding;
}

std::shared_ptr<mlir::MLIRContext> createCompilerContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

llvm::Expected<wafer::compiler::TargetLLVMModuleBundle>
buildElementwiseTargetBundle(std::string &diagnosticText,
                             wafer::TargetProfileId targetProfile) {
  auto context = createCompilerContext();

  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>, policy = "explicit", shape = array<i64: 1>, topology = @default}
  func.func @main(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>) -> tensor<8xf32> {
    %out = tensor.empty() : tensor<8xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%out : tensor<8xf32>) {
      ^bb0(%a: f32, %b: f32, %old: f32):
        %value = arith.addf %a, %b : f32
        linalg.yield %value : f32
    } -> tensor<8xf32>
    return %sum : tensor<8xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!tensorProgram)
    return llvm::createStringError("failed to parse target-call test module");

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0), boundary(1)};
  program.distributedOutputs = {boundary(0)};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, targetProfile, wafer::RuntimeLaunchKind::Kernel);
  if (!config)
    return config.takeError();
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto executable = wafer::compiler::detail::buildExecutableBundle(
      context, *tensorProgram, std::move(program), *config, diagnostics,
      std::nullopt);
  if (!executable)
    return executable.takeError();
  tensorProgram = nullptr;
  return wafer::compiler::compileExecutableBundleToTargetLLVMModules(
      *executable, diagnostics);
}

llvm::Expected<wafer::compiler::TargetLLVMModuleBundle>
buildElementwiseTargetBundle(std::string &diagnosticText) {
  return buildElementwiseTargetBundle(
      diagnosticText, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
}

llvm::Expected<wafer::compiler::TargetLLVMModuleBundle>
buildDirectDTETargetBundle(std::string &diagnosticText) {
  auto context = createCompilerContext();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64>, policy = "all_available", shape = array<i64: 16>, topology = @default}
  func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
    %out = tensor.empty() : tensor<4xf32>
    %permuted = wafer.linalg_ext.collective.collective_permute
        ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>)
        {source_target_pairs = array<i64: 0, 1, 1, 0, 2, 3, 3, 2,
                                          4, 5, 5, 4, 6, 7, 7, 6,
                                          8, 9, 9, 8, 10, 11, 11, 10,
                                          12, 13, 13, 12, 14, 15, 15, 14>,
         channel_id = 91 : i64} -> tensor<4xf32>
    return %permuted : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!tensorProgram)
    return llvm::createStringError("failed to parse Direct-DTE test module");

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 1;
  program.distributedInputs = {partitionedBoundary(0)};
  program.distributedOutputs = {partitionedBoundary(0)};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Kernel);
  if (!config)
    return config.takeError();
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto executable = wafer::compiler::detail::buildExecutableBundle(
      context, *tensorProgram, std::move(program), *config, diagnostics,
      std::nullopt);
  if (!executable)
    return executable.takeError();
  tensorProgram = nullptr;
  return wafer::compiler::compileExecutableBundleToTargetLLVMModules(
      *executable, diagnostics);
}

class RecordingSink final : public wafer::compiler::TargetTransactionSink {
public:
  llvm::Error begin(const wafer::compiler::TargetCallInvocationDescriptor
                        &descriptor) override {
    began = true;
    invocationRankCount = descriptor.ranks.size();
    rankDescriptors = descriptor.ranks;
    return llvm::Error::success();
  }

  llvm::Expected<uint64_t>
  issue(const wafer::compiler::TargetTransaction &transaction) override {
    if (reenterExecutable && !attemptedReentry) {
      attemptedReentry = true;
      llvm::Error error =
          reenterExecutable->executeRank(transaction.logicalRank);
      if (error)
        reentryDiagnostic = llvm::toString(std::move(error));
    }
    if (failAtIssue && transactions.size() == *failAtIssue) {
      failedLogicalRank = transaction.logicalRank;
      return llvm::createStringError("injected transaction sink failure");
    }
    transactions.push_back(transaction);
    return nextEvent++;
  }

  llvm::Error terminal(int64_t logicalRank) override {
    terminalRanks.push_back(logicalRank);
    return llvm::Error::success();
  }

  llvm::Error prepareCommit() override {
    if (failPrepareCommit)
      return llvm::createStringError("injected prepare-commit failure");
    return llvm::Error::success();
  }

  void commit() override { committed = true; }

  void abort(llvm::StringRef diagnostic) override {
    aborted = true;
    abortDiagnostic = diagnostic.str();
    transactions.clear();
    terminalRanks.clear();
  }

  bool began = false;
  bool committed = false;
  bool aborted = false;
  size_t invocationRankCount = 0;
  uint64_t nextEvent = 0x2000;
  std::optional<size_t> failAtIssue;
  bool failPrepareCommit = false;
  wafer::compiler::TargetCallExecutable *reenterExecutable = nullptr;
  bool attemptedReentry = false;
  std::optional<int64_t> failedLogicalRank;
  std::string reentryDiagnostic;
  std::string abortDiagnostic;
  std::vector<wafer::compiler::TargetCallRankDescriptor> rankDescriptors;
  std::vector<wafer::compiler::TargetTransaction> transactions;
  std::vector<int64_t> terminalRanks;
};

uint64_t supportedF32Code(wafer::TargetFormatEngine engine) {
  const wafer::TargetFormatEncodingRecord *record =
      wafer::findTargetFormatEncoding(
          wafer::TargetProfileId::waferTx81SingleCardKernelV1(), engine,
          wafer::LogicalFormat::F32);
  assert(record && record->isSupported() && record->dataFormatCode);
  return *record->dataFormatCode;
}

wafer::TargetProfileId
profileForDescriptor(const wafer::TargetCallDescriptor &descriptor) {
  if (wafer::isTargetCallAvailableForProfile(
          descriptor, wafer::TargetProfileId::waferTx81SingleCardKernelV1()))
    return wafer::TargetProfileId::waferTx81SingleCardKernelV1();
  if (wafer::isTargetCallAvailableForProfile(
          descriptor, wafer::TargetProfileId::waferTx81SingleCardKernelV2()))
    return wafer::TargetProfileId::waferTx81SingleCardKernelV2();
  return wafer::TargetProfileId::waferTx81SingleCardKernelV3();
}

std::vector<uint64_t>
makeDecodableArguments(const wafer::TargetCallDescriptor &descriptor) {
  std::vector<uint64_t> arguments;
  arguments.reserve(descriptor.arguments.size());
  for (auto [index, scalar] : llvm::enumerate(descriptor.arguments))
    arguments.push_back(scalar == wafer::TargetCallScalarType::I64
                            ? 0x100000 + index * 0x1000
                            : index + 1);
  if (descriptor.issueDomain &&
      descriptor.issueDomain->nccWorkerArgument.has_value())
    arguments[*descriptor.issueDomain->nccWorkerArgument] = 0;

  if (const auto *builtin =
          std::get_if<wafer::TargetCallBuiltin>(&descriptor.semantic)) {
    switch (*builtin) {
    case wafer::TargetCallBuiltin::RDMA:
      arguments[10] = supportedF32Code(wafer::TargetFormatEngine::RDMA);
      break;
    case wafer::TargetCallBuiltin::WDMA:
      arguments[10] = supportedF32Code(wafer::TargetFormatEngine::WDMA);
      break;
    case wafer::TargetCallBuiltin::Memset:
      arguments[3] = supportedF32Code(wafer::TargetFormatEngine::TDMA);
      break;
    case wafer::TargetCallBuiltin::Bit2FP:
      arguments[3] = supportedF32Code(wafer::TargetFormatEngine::CT);
      break;
    case wafer::TargetCallBuiltin::MaskMove:
      arguments[4] = supportedF32Code(wafer::TargetFormatEngine::CT);
      break;
    case wafer::TargetCallBuiltin::Gemm:
      arguments[7] = supportedF32Code(wafer::TargetFormatEngine::NE);
      break;
    case wafer::TargetCallBuiltin::GemmOrientedV2:
      arguments[7] = supportedF32Code(wafer::TargetFormatEngine::NE);
      arguments[8] = 1;
      arguments[9] = 0;
      break;
    case wafer::TargetCallBuiltin::TDMAPad:
      arguments[14] = supportedF32Code(wafer::TargetFormatEngine::TDMA);
      break;
    case wafer::TargetCallBuiltin::TDMAImg2Col:
      arguments[18] = supportedF32Code(wafer::TargetFormatEngine::TDMA);
      break;
    case wafer::TargetCallBuiltin::DirectDTEBegin:
    case wafer::TargetCallBuiltin::DirectDTEBeginAfterPrepare:
      arguments[1] = 16;
      break;
    case wafer::TargetCallBuiltin::DirectDTESendPrepare:
      arguments[6] = 1;
      break;
    case wafer::TargetCallBuiltin::GatherScatter:
    case wafer::TargetCallBuiltin::LocalFence:
    case wafer::TargetCallBuiltin::NCCJoin:
    case wafer::TargetCallBuiltin::DirectDTESendIssue:
    case wafer::TargetCallBuiltin::DirectDTERecvPrepare:
    case wafer::TargetCallBuiltin::DirectDTEWait:
    case wafer::TargetCallBuiltin::DirectDTEFinish:
      break;
    }
    return arguments;
  }
  if (std::holds_alternative<wafer::InstrElementwiseKind>(
          descriptor.semantic)) {
    const size_t payloadSize =
        arguments.size() -
        (descriptor.issueDomain &&
                 descriptor.issueDomain->nccWorkerArgument.has_value()
             ? 1
             : 0);
    arguments[payloadSize - 1] =
        supportedF32Code(wafer::TargetFormatEngine::CT);
    return arguments;
  }
  if (std::holds_alternative<wafer::InstrReduceKind>(descriptor.semantic)) {
    arguments[7] = supportedF32Code(wafer::TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *kind =
          std::get_if<wafer::InstrConvKind>(&descriptor.semantic)) {
    arguments[3] = static_cast<uint32_t>(*kind);
    arguments[30] = supportedF32Code(wafer::TargetFormatEngine::NE);
    return arguments;
  }
  if (const auto *kind =
          std::get_if<wafer::InstrPoolKind>(&descriptor.semantic)) {
    bool indexed = *kind == wafer::InstrPoolKind::IndexedMax ||
                   *kind == wafer::InstrPoolKind::IndexedMin;
    size_t firstField = indexed ? 3 : 2;
    arguments[firstField] = static_cast<uint32_t>(*kind);
    arguments[firstField + 17] =
        supportedF32Code(wafer::TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *kind =
          std::get_if<wafer::InstrUnpoolKind>(&descriptor.semantic)) {
    arguments[2] = static_cast<uint32_t>(*kind);
    arguments[16] = supportedF32Code(wafer::TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *kind =
          std::get_if<wafer::InstrPeripheralKind>(&descriptor.semantic)) {
    size_t addressCount = 0;
    switch (*kind) {
    case wafer::InstrPeripheralKind::ArgMax:
    case wafer::InstrPeripheralKind::ArgMin:
    case wafer::InstrPeripheralKind::Lut16:
    case wafer::InstrPeripheralKind::Lut32:
      addressCount = 3;
      break;
    case wafer::InstrPeripheralKind::Bilinear:
    case wafer::InstrPeripheralKind::ElemMask:
      addressCount = 2;
      break;
    case wafer::InstrPeripheralKind::RandGen:
      addressCount = 5;
      break;
    case wafer::InstrPeripheralKind::Count:
    case wafer::InstrPeripheralKind::Factorize:
      llvm_unreachable("unregistered peripheral kind");
    }
    arguments[addressCount] = static_cast<uint32_t>(*kind);
    arguments[addressCount + 2] =
        supportedF32Code(wafer::TargetFormatEngine::CT);
  }
  return arguments;
}

template <size_t N>
std::array<uint32_t, N> expectedArray(llvm::ArrayRef<uint64_t> arguments,
                                      size_t start) {
  std::array<uint32_t, N> result{};
  for (size_t index = 0; index < N; ++index)
    result[index] = static_cast<uint32_t>(arguments[start + index]);
  return result;
}

void expectPayloadFields(
    const wafer::TargetCallDescriptor &descriptor,
    llvm::ArrayRef<uint64_t> arguments,
    const wafer::compiler::TargetTransactionPayload &payload) {
  const wafer::TargetCallSemantic &semantic = descriptor.semantic;
  SCOPED_TRACE(descriptor.symbol);
  auto u32 = [&](size_t index) {
    return static_cast<uint32_t>(arguments[index]);
  };
  auto expectFormat = [](wafer::LogicalFormat format) {
    EXPECT_EQ(format, wafer::LogicalFormat::F32);
  };

  if (const auto *builtin = std::get_if<wafer::TargetCallBuiltin>(&semantic)) {
    switch (*builtin) {
    case wafer::TargetCallBuiltin::RDMA:
    case wafer::TargetCallBuiltin::WDMA: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::compiler::TargetStridedDMATransaction>(
              payload));
      const auto &value =
          std::get<wafer::compiler::TargetStridedDMATransaction>(payload);
      EXPECT_EQ(value.direction,
                *builtin == wafer::TargetCallBuiltin::RDMA
                    ? wafer::compiler::TargetDMADirection::Read
                    : wafer::compiler::TargetDMADirection::Write);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.byteCount, u32(2));
      EXPECT_EQ(value.innerBytes, u32(3));
      EXPECT_EQ(value.strides, expectedArray<3>(arguments, 4));
      EXPECT_EQ(value.iterations, expectedArray<3>(arguments, 7));
      expectFormat(value.format);
      return;
    }
    case wafer::TargetCallBuiltin::GatherScatter: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::compiler::TargetGatherScatterTransaction>(payload));
      const auto &value =
          std::get<wafer::compiler::TargetGatherScatterTransaction>(payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.byteCount, u32(2));
      EXPECT_EQ(value.innerBytes, u32(3));
      EXPECT_EQ(value.sourceStrides, expectedArray<3>(arguments, 4));
      EXPECT_EQ(value.sourceIterations, expectedArray<3>(arguments, 7));
      EXPECT_EQ(value.destinationStrides, expectedArray<3>(arguments, 10));
      EXPECT_EQ(value.destinationIterations, expectedArray<3>(arguments, 13));
      return;
    }
    case wafer::TargetCallBuiltin::Memset: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::compiler::TargetMemsetTransaction>(
              payload));
      const auto &value =
          std::get<wafer::compiler::TargetMemsetTransaction>(payload);
      EXPECT_EQ(value.destination, arguments[0]);
      EXPECT_EQ(value.value, u32(1));
      EXPECT_EQ(value.elementCount, u32(2));
      expectFormat(value.format);
      return;
    }
    case wafer::TargetCallBuiltin::Bit2FP: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::compiler::TargetBit2FPTransaction>(
              payload));
      const auto &value =
          std::get<wafer::compiler::TargetBit2FPTransaction>(payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.elementCount, u32(2));
      expectFormat(value.format);
      return;
    }
    case wafer::TargetCallBuiltin::MaskMove: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::compiler::TargetMaskMoveTransaction>(
              payload));
      const auto &value =
          std::get<wafer::compiler::TargetMaskMoveTransaction>(payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.mask, u32(1));
      EXPECT_EQ(value.destination, arguments[2]);
      EXPECT_EQ(value.elementCount, u32(3));
      expectFormat(value.format);
      return;
    }
    case wafer::TargetCallBuiltin::Gemm:
    case wafer::TargetCallBuiltin::GemmOrientedV2: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::compiler::TargetGemmTransaction>(
              payload));
      const auto &value =
          std::get<wafer::compiler::TargetGemmTransaction>(payload);
      EXPECT_EQ(value.lhs, arguments[0]);
      EXPECT_EQ(value.rhs, arguments[1]);
      EXPECT_EQ(value.destination, arguments[2]);
      EXPECT_EQ(value.m, u32(3));
      EXPECT_EQ(value.k, u32(4));
      EXPECT_EQ(value.n, u32(5));
      EXPECT_EQ(value.batchCount, u32(6));
      expectFormat(value.format);
      EXPECT_EQ(value.lhsOrientation,
                *builtin == wafer::TargetCallBuiltin::GemmOrientedV2
                    ? wafer::GemmOrientation::Transpose
                    : wafer::GemmOrientation::Normal);
      EXPECT_EQ(value.rhsOrientation, wafer::GemmOrientation::Normal);
      return;
    }
    case wafer::TargetCallBuiltin::TDMAPad:
    case wafer::TargetCallBuiltin::TDMAImg2Col: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::compiler::TargetTDMATransformTransaction>(payload));
      const auto &value =
          std::get<wafer::compiler::TargetTDMATransformTransaction>(payload);
      const bool imageToColumn =
          *builtin == wafer::TargetCallBuiltin::TDMAImg2Col;
      EXPECT_EQ(value.kind,
                imageToColumn
                    ? wafer::compiler::TargetTDMATransformKind::ImageToColumn
                    : wafer::compiler::TargetTDMATransformKind::Pad);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.sourceShape, expectedArray<4>(arguments, 2));
      EXPECT_EQ(value.destinationShape, expectedArray<4>(arguments, 6));
      EXPECT_EQ(value.pads, expectedArray<4>(arguments, 10));
      if (imageToColumn) {
        ASSERT_TRUE(value.kernelStrides.has_value());
        EXPECT_EQ(*value.kernelStrides, expectedArray<4>(arguments, 14));
      } else {
        EXPECT_FALSE(value.kernelStrides.has_value());
      }
      expectFormat(value.format);
      return;
    }
    case wafer::TargetCallBuiltin::LocalFence:
    case wafer::TargetCallBuiltin::NCCJoin: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::compiler::TargetNCCJoinTransaction>(
              payload));
      const auto &join =
          std::get<wafer::compiler::TargetNCCJoinTransaction>(payload);
      EXPECT_EQ(join.participantMask,
                *builtin == wafer::TargetCallBuiltin::LocalFence ? uint32_t{1}
                                                                 : u32(0));
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTEBegin:
    case wafer::TargetCallBuiltin::DirectDTEBeginAfterPrepare: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::compiler::TargetDirectDTEBeginTransaction>(payload));
      const auto &value =
          std::get<wafer::compiler::TargetDirectDTEBeginTransaction>(payload);
      EXPECT_EQ(value.statusAddress, arguments[0]);
      EXPECT_EQ(value.rankCount, u32(1));
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTESendPrepare: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::compiler::TargetDirectDTESendTransaction>(payload));
      const auto &value =
          std::get<wafer::compiler::TargetDirectDTESendTransaction>(payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.remoteDestination, arguments[1]);
      EXPECT_EQ(value.byteCount, u32(2));
      EXPECT_EQ(value.localTile, u32(3));
      EXPECT_EQ(value.remoteTile, u32(4));
      EXPECT_EQ(value.remoteFSM, u32(5));
      EXPECT_EQ(value.highPerformance, u32(6) != 0);
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTESendIssue: {
      ASSERT_TRUE(
          std::holds_alternative<
              wafer::compiler::TargetDirectDTESendIssueTransaction>(payload));
      EXPECT_EQ(std::get<wafer::compiler::TargetDirectDTESendIssueTransaction>(
                    payload)
                    .event,
                arguments[0]);
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTERecvPrepare: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::compiler::TargetDirectDTEReceiveTransaction>(payload));
      const auto &value =
          std::get<wafer::compiler::TargetDirectDTEReceiveTransaction>(payload);
      EXPECT_EQ(value.destination, arguments[0]);
      EXPECT_EQ(value.byteCount, u32(1));
      EXPECT_EQ(value.localTile, u32(2));
      EXPECT_EQ(value.remoteTile, u32(3));
      EXPECT_EQ(value.localFSM, u32(4));
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTEWait: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::compiler::TargetDirectDTEWaitTransaction>(payload));
      EXPECT_EQ(
          std::get<wafer::compiler::TargetDirectDTEWaitTransaction>(payload)
              .event,
          arguments[0]);
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTEFinish:
      EXPECT_TRUE(std::holds_alternative<
                  wafer::compiler::TargetDirectDTEFinishTransaction>(payload));
      return;
    }
  }

  if (const auto *kind = std::get_if<wafer::InstrElementwiseKind>(&semantic)) {
    ASSERT_TRUE(
        std::holds_alternative<wafer::compiler::TargetElementwiseTransaction>(
            payload));
    const auto &value =
        std::get<wafer::compiler::TargetElementwiseTransaction>(payload);
    const size_t payloadSize =
        arguments.size() -
        (descriptor.issueDomain &&
                 descriptor.issueDomain->nccWorkerArgument.has_value()
             ? 1
             : 0);
    const bool unary = payloadSize == 4;
    EXPECT_EQ(value.kind, *kind);
    EXPECT_EQ(value.lhs, arguments[0]);
    if (unary)
      EXPECT_FALSE(value.rhs.has_value());
    else {
      ASSERT_TRUE(value.rhs.has_value());
      EXPECT_EQ(*value.rhs, arguments[1]);
    }
    EXPECT_EQ(value.destination, arguments[unary ? 1 : 2]);
    EXPECT_EQ(value.elementCount, u32(unary ? 2 : 3));
    expectFormat(value.format);
    return;
  }
  if (const auto *kind = std::get_if<wafer::InstrReduceKind>(&semantic)) {
    ASSERT_TRUE(
        std::holds_alternative<wafer::compiler::TargetReduceTransaction>(
            payload));
    const auto &value =
        std::get<wafer::compiler::TargetReduceTransaction>(payload);
    EXPECT_EQ(value.kind, *kind);
    EXPECT_EQ(value.source, arguments[0]);
    EXPECT_EQ(value.destination, arguments[1]);
    EXPECT_EQ(value.dimension, u32(2));
    EXPECT_EQ(value.nhwc, expectedArray<4>(arguments, 3));
    expectFormat(value.format);
    return;
  }
  if (const auto *kind = std::get_if<wafer::InstrConvertKind>(&semantic)) {
    ASSERT_TRUE(
        std::holds_alternative<wafer::compiler::TargetConvertTransaction>(
            payload));
    const auto &value =
        std::get<wafer::compiler::TargetConvertTransaction>(payload);
    EXPECT_EQ(value.kind, *kind);
    EXPECT_EQ(value.source, arguments[0]);
    EXPECT_EQ(value.destination, arguments[1]);
    EXPECT_EQ(value.elementCount, u32(2));
    switch (wafer::getInstrConvertParameterKind(*kind)) {
    case wafer::InstrConvertParameterKind::ZeroPoint:
      ASSERT_TRUE(value.zeroPoint.has_value());
      EXPECT_EQ(*value.zeroPoint, u32(3));
      EXPECT_FALSE(value.roundingMode.has_value());
      break;
    case wafer::InstrConvertParameterKind::RoundingMode:
      EXPECT_FALSE(value.zeroPoint.has_value());
      ASSERT_TRUE(value.roundingMode.has_value());
      EXPECT_EQ(*value.roundingMode, u32(4));
      break;
    case wafer::InstrConvertParameterKind::None:
      EXPECT_FALSE(value.zeroPoint.has_value());
      EXPECT_FALSE(value.roundingMode.has_value());
      break;
    }
    return;
  }
  if (const auto *kind = std::get_if<wafer::InstrConvKind>(&semantic)) {
    ASSERT_TRUE(std::holds_alternative<wafer::compiler::TargetConvTransaction>(
        payload));
    const auto &value =
        std::get<wafer::compiler::TargetConvTransaction>(payload);
    EXPECT_EQ(value.kind, *kind);
    EXPECT_EQ(value.input, arguments[0]);
    EXPECT_EQ(value.weight, arguments[1]);
    EXPECT_EQ(value.destination, arguments[2]);
    EXPECT_EQ(value.inputShape, expectedArray<4>(arguments, 4));
    EXPECT_EQ(value.weightShape, expectedArray<4>(arguments, 8));
    EXPECT_EQ(value.outputShape, expectedArray<4>(arguments, 12));
    EXPECT_EQ(value.pads, expectedArray<4>(arguments, 16));
    EXPECT_EQ(value.unpads, expectedArray<4>(arguments, 20));
    EXPECT_EQ(value.kernelStrides, expectedArray<4>(arguments, 24));
    EXPECT_EQ(value.dilations, expectedArray<2>(arguments, 28));
    expectFormat(value.format);
    return;
  }
  if (const auto *kind = std::get_if<wafer::InstrPoolKind>(&semantic)) {
    ASSERT_TRUE(std::holds_alternative<wafer::compiler::TargetPoolTransaction>(
        payload));
    const auto &value =
        std::get<wafer::compiler::TargetPoolTransaction>(payload);
    const bool indexed = *kind == wafer::InstrPoolKind::IndexedMax ||
                         *kind == wafer::InstrPoolKind::IndexedMin;
    const size_t first = indexed ? 3 : 2;
    EXPECT_EQ(value.kind, *kind);
    EXPECT_EQ(value.input, arguments[0]);
    EXPECT_EQ(value.valueDestination, arguments[1]);
    if (indexed) {
      ASSERT_TRUE(value.indexDestination.has_value());
      EXPECT_EQ(*value.indexDestination, arguments[2]);
    } else {
      EXPECT_FALSE(value.indexDestination.has_value());
    }
    EXPECT_EQ(value.sourceShape, expectedArray<4>(arguments, first + 1));
    EXPECT_EQ(value.destinationShape, expectedArray<4>(arguments, first + 5));
    EXPECT_EQ(value.pads, expectedArray<4>(arguments, first + 9));
    EXPECT_EQ(value.kernelStrides, expectedArray<4>(arguments, first + 13));
    expectFormat(value.format);
    return;
  }
  if (const auto *kind = std::get_if<wafer::InstrUnpoolKind>(&semantic)) {
    ASSERT_TRUE(
        std::holds_alternative<wafer::compiler::TargetUnpoolTransaction>(
            payload));
    const auto &value =
        std::get<wafer::compiler::TargetUnpoolTransaction>(payload);
    EXPECT_EQ(value.kind, *kind);
    EXPECT_EQ(value.input, arguments[0]);
    EXPECT_EQ(value.destination, arguments[1]);
    if (*kind == wafer::InstrUnpoolKind::Avg)
      EXPECT_FALSE(value.indexAddress.has_value());
    else {
      ASSERT_TRUE(value.indexAddress.has_value());
      EXPECT_EQ(*value.indexAddress, u32(3));
    }
    EXPECT_EQ(value.sourceShape, expectedArray<4>(arguments, 4));
    EXPECT_EQ(value.destinationShape, expectedArray<4>(arguments, 8));
    EXPECT_EQ(value.kernelStrides, expectedArray<4>(arguments, 12));
    expectFormat(value.format);
    return;
  }
  if (const auto *kind = std::get_if<wafer::InstrPeripheralKind>(&semantic)) {
    switch (*kind) {
    case wafer::InstrPeripheralKind::ArgMax:
    case wafer::InstrPeripheralKind::ArgMin: {
      ASSERT_TRUE(
          std::holds_alternative<
              wafer::compiler::TargetPeripheralArgExtremaTransaction>(payload));
      const auto &value =
          std::get<wafer::compiler::TargetPeripheralArgExtremaTransaction>(
              payload);
      EXPECT_EQ(value.kind, *kind);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.valueDestination, arguments[1]);
      EXPECT_EQ(value.indexDestination, arguments[2]);
      EXPECT_EQ(value.elementCount, u32(4));
      expectFormat(value.format);
      return;
    }
    case wafer::InstrPeripheralKind::Bilinear: {
      ASSERT_TRUE(
          std::holds_alternative<
              wafer::compiler::TargetPeripheralBilinearTransaction>(payload));
      const auto &value =
          std::get<wafer::compiler::TargetPeripheralBilinearTransaction>(
              payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.elementCount, u32(3));
      expectFormat(value.format);
      EXPECT_EQ(value.sourceShape, expectedArray<4>(arguments, 5));
      EXPECT_EQ(value.destinationShape, expectedArray<4>(arguments, 9));
      return;
    }
    case wafer::InstrPeripheralKind::Lut16:
    case wafer::InstrPeripheralKind::Lut32: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::compiler::TargetPeripheralLUTTransaction>(payload));
      const auto &value =
          std::get<wafer::compiler::TargetPeripheralLUTTransaction>(payload);
      EXPECT_EQ(value.kind, *kind);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.table, arguments[1]);
      EXPECT_EQ(value.destination, arguments[2]);
      EXPECT_EQ(value.elementCount, u32(4));
      expectFormat(value.format);
      EXPECT_EQ(value.tableElementCount, u32(6));
      return;
    }
    case wafer::InstrPeripheralKind::RandGen: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::compiler::TargetPeripheralRandomTransaction>(payload));
      const auto &value =
          std::get<wafer::compiler::TargetPeripheralRandomTransaction>(payload);
      EXPECT_EQ(value.sources,
                (std::array<uint64_t, 2>{arguments[0], arguments[1]}));
      EXPECT_EQ(
          value.destinations,
          (std::array<uint64_t, 3>{arguments[2], arguments[3], arguments[4]}));
      EXPECT_EQ(value.elementCount, u32(6));
      expectFormat(value.format);
      return;
    }
    case wafer::InstrPeripheralKind::ElemMask: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::compiler::TargetPeripheralElementMaskTransaction>(
          payload));
      const auto &value =
          std::get<wafer::compiler::TargetPeripheralElementMaskTransaction>(
              payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.elementCount, u32(3));
      expectFormat(value.format);
      EXPECT_EQ(value.scale, u32(6));
      EXPECT_EQ(value.probability, u32(7));
      EXPECT_EQ(value.roundingMode, u32(8));
      return;
    }
    case wafer::InstrPeripheralKind::Count:
    case wafer::InstrPeripheralKind::Factorize:
      FAIL() << "unregistered peripheral semantic";
      return;
    }
  }
  FAIL() << "unknown target-call semantic";
}

TEST(TargetCallRegistryTest, ExactlyCoversTypedTargetCallSurface) {
  llvm::ArrayRef<wafer::TargetCallDescriptor> descriptors =
      wafer::getTargetCallDescriptors();
  ASSERT_EQ(descriptors.size(), 217u);
  llvm::DenseSet<llvm::StringRef> symbols;
  size_t issueDomainCount = 0;
  size_t nccIssueDomainCount = 0;
  size_t fixedNCCIssueDomainCount = 0;
  size_t argumentNCCIssueDomainCount = 0;
  size_t synchronousWritebackCount = 0;
  size_t directDTEIssueDomainCount = 0;
  for (const wafer::TargetCallDescriptor &descriptor : descriptors) {
    EXPECT_TRUE(llvm::StringRef(descriptor.symbol).starts_with("wafer_tx81_"));
    EXPECT_TRUE(symbols.insert(descriptor.symbol).second);
    EXPECT_EQ(wafer::findTargetCallDescriptor(descriptor.symbol), &descriptor);
    const wafer::TargetProfileId profile = profileForDescriptor(descriptor);
    EXPECT_EQ(wafer::findTargetCallDescriptor(descriptor.semantic, profile),
              &descriptor);
    if (!descriptor.issueDomain) {
      EXPECT_FALSE(wafer::getTargetCallTSMEngine(descriptor));
      continue;
    }
    ++issueDomainCount;
    EXPECT_EQ(wafer::getTargetCallTSMEngine(descriptor),
              descriptor.issueDomain->engine);
    if (descriptor.issueDomain->engine ==
        wafer::TargetCallTSMEngine::DirectDTE) {
      ++directDTEIssueDomainCount;
      EXPECT_FALSE(descriptor.issueDomain->fixedNCCWorker);
      EXPECT_FALSE(descriptor.issueDomain->nccWorkerArgument);
      EXPECT_EQ(descriptor.issueDomain->completionBehavior,
                wafer::LocalInstructionCompletion::None);
    } else {
      ++nccIssueDomainCount;
      EXPECT_NE(descriptor.issueDomain->fixedNCCWorker.has_value(),
                descriptor.issueDomain->nccWorkerArgument.has_value());
      if (descriptor.issueDomain->fixedNCCWorker) {
        ++fixedNCCIssueDomainCount;
        EXPECT_EQ(*descriptor.issueDomain->fixedNCCWorker,
                  wafer::NCCWorker::Worker0);
      } else {
        ++argumentNCCIssueDomainCount;
        EXPECT_EQ(*descriptor.issueDomain->nccWorkerArgument + 1,
                  descriptor.arguments.size());
        EXPECT_EQ(descriptor.arguments.back(),
                  wafer::TargetCallScalarType::I32);
      }
      const auto *peripheral =
          std::get_if<wafer::InstrPeripheralKind>(&descriptor.semantic);
      const bool synchronous =
          peripheral && (*peripheral == wafer::InstrPeripheralKind::ArgMax ||
                         *peripheral == wafer::InstrPeripheralKind::ArgMin);
      EXPECT_EQ(descriptor.issueDomain->completionBehavior,
                synchronous
                    ? wafer::LocalInstructionCompletion::SynchronousWriteback
                    : wafer::LocalInstructionCompletion::OrderedPending);
      synchronousWritebackCount += synchronous;
    }
  }
  EXPECT_EQ(issueDomainCount, 210u);
  EXPECT_EQ(nccIssueDomainCount, 208u);
  EXPECT_EQ(fixedNCCIssueDomainCount, 104u);
  EXPECT_EQ(argumentNCCIssueDomainCount, 104u);
  EXPECT_EQ(synchronousWritebackCount, 4u);
  EXPECT_EQ(directDTEIssueDomainCount, 2u);
  EXPECT_EQ(wafer::findTargetCallDescriptor("wafer_tx81_unknown"), nullptr);

  const auto &join = wafer::getTargetCallDescriptor(
      wafer::TargetCallBuiltin::NCCJoin,
      wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  EXPECT_EQ(join.result, wafer::TargetCallResultType::Void);
  ASSERT_EQ(join.arguments.size(), 1u);
  EXPECT_EQ(join.arguments.front(), wafer::TargetCallScalarType::I32);
  EXPECT_FALSE(join.issueDomain);
  const auto &send = wafer::getTargetCallDescriptor(
      wafer::TargetCallBuiltin::DirectDTESendPrepare,
      wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  EXPECT_EQ(send.result, wafer::TargetCallResultType::I64);
  EXPECT_EQ(send.arguments.size(), 7u);
  EXPECT_EQ(send.arguments[0], wafer::TargetCallScalarType::I64);
  EXPECT_EQ(send.arguments[2], wafer::TargetCallScalarType::I32);
  const auto v1 = wafer::TargetProfileId::waferTx81SingleCardKernelV1();
  const auto v3 = wafer::TargetProfileId::waferTx81SingleCardKernelV3();
  EXPECT_EQ(wafer::getTargetCallDescriptor(wafer::InstrElementwiseKind::Abs, v1)
                .arguments.size(),
            4u);
  EXPECT_EQ(wafer::getTargetCallDescriptor(wafer::InstrElementwiseKind::Add, v1)
                .arguments.size(),
            5u);
  EXPECT_EQ(wafer::getTargetCallDescriptor(wafer::InstrConvKind::Conv, v1)
                .arguments.size(),
            31u);
  EXPECT_EQ(
      wafer::getTargetCallDescriptor(wafer::InstrPeripheralKind::Bilinear, v1)
          .arguments.size(),
      17u);
  EXPECT_EQ(wafer::getTargetCallDescriptor(wafer::InstrElementwiseKind::Add, v3)
                .symbol,
            "wafer_tx81_elementwise_add_v3");
  EXPECT_EQ(wafer::getTargetCallDescriptor(
                wafer::TargetCallBuiltin::GemmOrientedV2, v3)
                .symbol,
            "wafer_tx81_gemm_oriented_v3");
}

TEST(TargetCallRegistryTest, FreezesLegacyAndSeparatesV3Availability) {
  const auto v1 = wafer::TargetProfileId::waferTx81SingleCardKernelV1();
  const auto v2 = wafer::TargetProfileId::waferTx81SingleCardKernelV2();
  const auto v3 = wafer::TargetProfileId::waferTx81SingleCardKernelV3();

  const auto &legacyRDMA =
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::RDMA, v1);
  EXPECT_EQ(legacyRDMA.symbol, "wafer_tx81_rdma");
  EXPECT_EQ(legacyRDMA.arguments.size(), 11u);
  EXPECT_EQ(legacyRDMA.issueDomain->fixedNCCWorker, wafer::NCCWorker::Worker0);
  EXPECT_FALSE(legacyRDMA.issueDomain->nccWorkerArgument);
  EXPECT_EQ(wafer::findTargetCallDescriptor(legacyRDMA.symbol, v2),
            &legacyRDMA);
  EXPECT_EQ(wafer::findTargetCallDescriptor(legacyRDMA.symbol, v3), nullptr);

  const auto &workerRDMA =
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::RDMA, v3);
  EXPECT_EQ(workerRDMA.symbol, "wafer_tx81_rdma_v3");
  EXPECT_EQ(workerRDMA.arguments.size(), 12u);
  EXPECT_FALSE(workerRDMA.issueDomain->fixedNCCWorker);
  ASSERT_TRUE(workerRDMA.issueDomain->nccWorkerArgument);
  EXPECT_EQ(*workerRDMA.issueDomain->nccWorkerArgument, 11u);
  EXPECT_EQ(wafer::findTargetCallDescriptor(workerRDMA.symbol, v1), nullptr);
  EXPECT_EQ(wafer::findTargetCallDescriptor(workerRDMA.symbol, v2), nullptr);

  EXPECT_EQ(
      wafer::findTargetCallDescriptor(
          wafer::TargetCallSemantic(wafer::TargetCallBuiltin::GemmOrientedV2),
          v1),
      nullptr);
  EXPECT_EQ(wafer::getTargetCallDescriptor(
                wafer::TargetCallBuiltin::GemmOrientedV2, v2)
                .symbol,
            "wafer_tx81_gemm_oriented_v2");
  EXPECT_EQ(wafer::getTargetCallDescriptor(
                wafer::TargetCallBuiltin::GemmOrientedV2, v3)
                .symbol,
            "wafer_tx81_gemm_oriented_v3");

  EXPECT_EQ(wafer::findTargetCallDescriptor(
                wafer::TargetCallSemantic(
                    wafer::TargetCallBuiltin::DirectDTESendIssue),
                v1),
            nullptr);
  EXPECT_EQ(wafer::findTargetCallDescriptor(
                wafer::TargetCallSemantic(
                    wafer::TargetCallBuiltin::DirectDTESendIssue),
                v2),
            nullptr);
  EXPECT_EQ(wafer::getTargetCallDescriptor(
                wafer::TargetCallBuiltin::DirectDTESendIssue, v3)
                .symbol,
            "wafer_tx81_direct_dte_send_issue_v3");

  const auto &sharedFence =
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::LocalFence, v1);
  EXPECT_EQ(wafer::findTargetCallDescriptor(sharedFence.symbol, v2),
            &sharedFence);
  EXPECT_EQ(wafer::findTargetCallDescriptor(sharedFence.symbol, v3),
            &sharedFence);

  std::vector<uint64_t> arguments = makeDecodableArguments(workerRDMA);
  auto wrongProfile =
      wafer::decodeTargetCallPayload(workerRDMA, {v1, 1}, arguments);
  ASSERT_FALSE(static_cast<bool>(wrongProfile));
  EXPECT_NE(llvm::toString(wrongProfile.takeError())
                .find("not available for the invocation profile"),
            std::string::npos);
}

TEST(TargetCallRegistryTest, EveryDescriptorDecodesEveryABIField) {
  size_t decoded = 0;
  for (const wafer::TargetCallDescriptor &descriptor :
       wafer::getTargetCallDescriptors()) {
    std::vector<uint64_t> arguments = makeDecodableArguments(descriptor);
    wafer::TargetCallDecodeContext context{profileForDescriptor(descriptor),
                                           16};
    auto payload =
        wafer::decodeTargetCallPayload(descriptor, context, arguments);
    ASSERT_TRUE(static_cast<bool>(payload))
        << descriptor.symbol << ": " << llvm::toString(payload.takeError());
    expectPayloadFields(descriptor, arguments, *payload);
    ++decoded;
  }
  EXPECT_EQ(decoded, 217u);
}

TEST(TargetCallRegistryTest, DecodesExplicitWorkerOneAndTwo) {
  const wafer::TargetCallDescriptor &descriptor =
      wafer::getTargetCallDescriptor(
          wafer::InstrElementwiseKind::Add,
          wafer::TargetProfileId::waferTx81SingleCardKernelV3());
  ASSERT_TRUE(descriptor.issueDomain.has_value());
  ASSERT_TRUE(descriptor.issueDomain->nccWorkerArgument.has_value());
  for (wafer::NCCWorker expected :
       {wafer::NCCWorker::Worker1, wafer::NCCWorker::Worker2}) {
    std::vector<uint64_t> arguments = makeDecodableArguments(descriptor);
    arguments[*descriptor.issueDomain->nccWorkerArgument] =
        static_cast<uint32_t>(expected);
    llvm::Expected<std::optional<wafer::NCCWorker>> worker =
        wafer::decodeTargetCallNCCWorker(descriptor, arguments);
    ASSERT_TRUE(static_cast<bool>(worker))
        << llvm::toString(worker.takeError());
    ASSERT_TRUE(worker->has_value());
    EXPECT_EQ(**worker, expected);
    auto payload = wafer::decodeTargetCallPayload(
        descriptor, {wafer::TargetProfileId::waferTx81SingleCardKernelV3(), 1},
        arguments);
    ASSERT_TRUE(static_cast<bool>(payload))
        << llvm::toString(payload.takeError());
  }
}

TEST(TargetCallRegistryTest, RejectsOutOfRangeExplicitWorker) {
  const wafer::TargetCallDescriptor &descriptor =
      wafer::getTargetCallDescriptor(
          wafer::TargetCallBuiltin::RDMA,
          wafer::TargetProfileId::waferTx81SingleCardKernelV3());
  std::vector<uint64_t> arguments = makeDecodableArguments(descriptor);
  arguments[*descriptor.issueDomain->nccWorkerArgument] =
      wafer::kNCCWorkerCount;
  llvm::Expected<std::optional<wafer::NCCWorker>> worker =
      wafer::decodeTargetCallNCCWorker(descriptor, arguments);
  ASSERT_FALSE(static_cast<bool>(worker));
  EXPECT_NE(llvm::toString(worker.takeError()).find("worker domain"),
            std::string::npos);
}

TEST(TargetCallRegistryTest, NCCJoinDecoderRejectsInvalidParticipantMasks) {
  const wafer::TargetCallDescriptor &descriptor =
      wafer::getTargetCallDescriptor(
          wafer::TargetCallBuiltin::NCCJoin,
          wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  wafer::TargetCallDecodeContext context{
      wafer::TargetProfileId::waferTx81SingleCardKernelV1(), 1};
  auto empty = wafer::decodeTargetCallPayload(descriptor, context, {0});
  ASSERT_FALSE(static_cast<bool>(empty));
  EXPECT_NE(llvm::toString(empty.takeError()).find("participant mask"),
            std::string::npos);
  const uint64_t outside = uint64_t{1}
                           << static_cast<uint32_t>(wafer::kNCCWorkerCount);
  auto outOfRange =
      wafer::decodeTargetCallPayload(descriptor, context, {outside});
  ASSERT_FALSE(static_cast<bool>(outOfRange));
  EXPECT_NE(llvm::toString(outOfRange.takeError()).find("worker domain"),
            std::string::npos);
}

TEST(TargetCallFrontendTest, ExecutesProductionTargetLLVMThroughTypedSink) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  ASSERT_EQ(bundle->getModules().size(), 1u);
  std::string originalTriple =
      bundle->getModules().front().getTargetTriple().str();
  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0, std::vector<uint64_t>(
              bundle->getModules().front().getKernelABISlots().size())}};
  for (size_t index = 0; index < arguments[0].slots.size(); ++index)
    arguments[0].slots[index] = 0x100000 + index * 0x1000;

  RecordingSink sink;
  auto executable =
      wafer::compiler::prepareTargetCallFrontend(*bundle, arguments);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  if (llvm::Error error = executable->begin(sink))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error = executable->executeRank(0))
    FAIL() << llvm::toString(std::move(error));
  auto result = executable->commit();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(sink.began);
  EXPECT_TRUE(sink.committed);
  EXPECT_FALSE(sink.aborted);
  EXPECT_EQ(sink.invocationRankCount, 1u);
  ASSERT_EQ(sink.rankDescriptors.size(), 1u);
  EXPECT_EQ(sink.rankDescriptors[0].logicalRank, 0);
  ASSERT_EQ(sink.rankDescriptors[0].kernelABISlots.size(),
            bundle->getModules().front().getKernelABISlots().size());
  ASSERT_FALSE(sink.rankDescriptors[0].kernelABISlots.empty());
  EXPECT_EQ(sink.rankDescriptors[0].slotValues, arguments[0].slots);
  EXPECT_EQ(sink.rankDescriptors[0].kernelABISlots.front().dtype,
            bundle->getModules().front().getKernelABISlots().front().dtype);
  EXPECT_EQ(sink.terminalRanks, std::vector<int64_t>({0}));
  EXPECT_EQ(result->completedRankCount, 1);
  EXPECT_EQ(result->issuedTransactionCount, sink.transactions.size());
  ASSERT_FALSE(sink.transactions.empty());
  bool sawAdd = false;
  for (size_t index = 0; index < sink.transactions.size(); ++index) {
    const wafer::compiler::TargetTransaction &transaction =
        sink.transactions[index];
    EXPECT_EQ(transaction.logicalRank, 0);
    EXPECT_EQ(transaction.issueOrdinal, index);
    if (transaction.nccIssueDomain) {
      EXPECT_NE(transaction.nccIssueDomain->engine,
                wafer::TargetCallTSMEngine::DirectDTE);
      EXPECT_EQ(transaction.nccIssueDomain->worker, wafer::NCCWorker::Worker0);
      EXPECT_EQ(transaction.nccIssueDomain->completionBehavior,
                wafer::LocalInstructionCompletion::OrderedPending);
    }
    if (const auto *elementwise =
            std::get_if<wafer::compiler::TargetElementwiseTransaction>(
                &transaction.payload)) {
      sawAdd = true;
      ASSERT_TRUE(transaction.nccIssueDomain.has_value());
      EXPECT_EQ(transaction.nccIssueDomain->engine,
                wafer::TargetCallTSMEngine::CT);
      EXPECT_EQ(elementwise->kind, wafer::InstrElementwiseKind::Add);
      EXPECT_TRUE(elementwise->rhs.has_value());
      EXPECT_EQ(elementwise->elementCount, 8u);
      EXPECT_EQ(elementwise->format, wafer::LogicalFormat::F32);
    }
  }
  EXPECT_TRUE(sawAdd);
  EXPECT_EQ(bundle->getModules().front().getTargetTriple(), originalTriple);
}

TEST(TargetCallFrontendTest, CarriesExplicitWorkerOneAndTwoIntoTransactions) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(
      diagnostics, wafer::TargetProfileId::waferTx81SingleCardKernelV3());
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  llvm::Module &module =
      const_cast<llvm::Module &>(bundle->getModules().front().getModule());
  size_t rewritten = 0;
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
        llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
        const wafer::TargetCallDescriptor *descriptor =
            callee ? wafer::findTargetCallDescriptor(callee->getName())
                   : nullptr;
        if (!descriptor || !descriptor->issueDomain ||
            !descriptor->issueDomain->nccWorkerArgument)
          continue;
        wafer::NCCWorker worker =
            descriptor->issueDomain->engine == wafer::TargetCallTSMEngine::CT
                ? wafer::NCCWorker::Worker2
                : wafer::NCCWorker::Worker1;
        call->setArgOperand(
            *descriptor->issueDomain->nccWorkerArgument,
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(module.getContext()),
                                   static_cast<uint32_t>(worker)));
        ++rewritten;
      }
  ASSERT_GT(rewritten, 0u);

  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0, std::vector<uint64_t>(
              bundle->getModules().front().getKernelABISlots().size(),
              UINT64_C(0x100000))}};
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  bool sawWorker1 = false;
  bool sawWorker2 = false;
  for (const wafer::compiler::TargetTransaction &transaction :
       sink.transactions) {
    if (!transaction.nccIssueDomain)
      continue;
    sawWorker1 |=
        transaction.nccIssueDomain->worker == wafer::NCCWorker::Worker1;
    sawWorker2 |=
        transaction.nccIssueDomain->worker == wafer::NCCWorker::Worker2;
  }
  EXPECT_TRUE(sawWorker1);
  EXPECT_TRUE(sawWorker2);
}

TEST(TargetCallFrontendTest,
     CarriesSynchronousWritebackBehaviorIntoDynamicTransaction) {
  std::string diagnostics;
  auto bundle = buildDirectDTETargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  ASSERT_EQ(bundle->getModules().size(), 16u);

  llvm::Module &module =
      const_cast<llvm::Module &>(bundle->getModules().front().getModule());
  llvm::Function *entry =
      module.getFunction(bundle->getModules().front().getEntrySymbol());
  ASSERT_NE(entry, nullptr);
  llvm::IRBuilder<> builder(entry->getEntryBlock().getTerminator());
  for (wafer::InstrPeripheralKind kind : {wafer::InstrPeripheralKind::ArgMax,
                                          wafer::InstrPeripheralKind::ArgMin}) {
    const wafer::TargetCallDescriptor &descriptor =
        wafer::getTargetCallDescriptor(
            kind, bundle->getModules().front().getTargetProfileId());
    llvm::SmallVector<llvm::Type *, 10> argumentTypes;
    llvm::SmallVector<llvm::Value *, 10> callArguments;
    std::vector<uint64_t> values = makeDecodableArguments(descriptor);
    ASSERT_EQ(values.size(), descriptor.arguments.size());
    for (auto [type, value] : llvm::zip_equal(descriptor.arguments, values)) {
      if (type == wafer::TargetCallScalarType::I64) {
        argumentTypes.push_back(builder.getInt64Ty());
        callArguments.push_back(builder.getInt64(value));
      } else {
        argumentTypes.push_back(builder.getInt32Ty());
        callArguments.push_back(builder.getInt32(static_cast<uint32_t>(value)));
      }
    }
    llvm::FunctionCallee callee = module.getOrInsertFunction(
        descriptor.symbol,
        llvm::FunctionType::get(builder.getVoidTy(), argumentTypes,
                                /*isVarArg=*/false));
    builder.CreateCall(callee, callArguments);
  }

  std::vector<wafer::compiler::TargetCallRankArguments> arguments;
  for (const wafer::compiler::TargetLLVMModule &rankModule :
       bundle->getModules())
    arguments.push_back(
        {rankModule.getLogicalRank(),
         std::vector<uint64_t>(
             rankModule.getKernelABISlots().size(),
             UINT64_C(0x100000) +
                 static_cast<uint64_t>(rankModule.getLogicalRank()) *
                     UINT64_C(0x10000))});
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  bool sawArgMax = false;
  bool sawArgMin = false;
  size_t synchronousArgExtremaCount = 0;
  for (const wafer::compiler::TargetTransaction &transaction :
       sink.transactions) {
    const auto *payload =
        std::get_if<wafer::compiler::TargetPeripheralArgExtremaTransaction>(
            &transaction.payload);
    if (!payload)
      continue;
    sawArgMax |= payload->kind == wafer::InstrPeripheralKind::ArgMax;
    sawArgMin |= payload->kind == wafer::InstrPeripheralKind::ArgMin;
    ASSERT_TRUE(transaction.nccIssueDomain.has_value());
    EXPECT_EQ(transaction.nccIssueDomain->engine,
              wafer::TargetCallTSMEngine::CT);
    EXPECT_EQ(transaction.nccIssueDomain->worker, wafer::NCCWorker::Worker0);
    EXPECT_EQ(transaction.nccIssueDomain->completionBehavior,
              wafer::LocalInstructionCompletion::SynchronousWriteback);
    ++synchronousArgExtremaCount;
  }
  EXPECT_TRUE(sawArgMax);
  EXPECT_TRUE(sawArgMin);
  EXPECT_EQ(synchronousArgExtremaCount, 2u);
}

TEST(TargetCallFrontendTest, SinkFailureAbortsWithoutPartialResult) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  sink.failAtIssue = 0;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError())
                .find("injected transaction sink failure"),
            std::string::npos);
  EXPECT_TRUE(sink.began);
  EXPECT_TRUE(sink.aborted);
  EXPECT_FALSE(sink.committed);
  EXPECT_TRUE(sink.transactions.empty());
  EXPECT_TRUE(sink.terminalRanks.empty());
}

TEST(TargetCallFrontendTest, ExecutesAllRanksWithExplicitDTEOpaqueEvents) {
  std::string diagnostics;
  auto bundle = buildDirectDTETargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  ASSERT_EQ(bundle->getModules().size(), 16u);
  std::vector<wafer::compiler::TargetCallRankArguments> arguments;
  for (const wafer::compiler::TargetLLVMModule &module : bundle->getModules())
    arguments.push_back(
        {module.getLogicalRank(),
         std::vector<uint64_t>(module.getKernelABISlots().size(),
                               0x100000 + module.getLogicalRank() * 0x10000)});

  RecordingSink sink;
  auto executable =
      wafer::compiler::prepareTargetCallFrontend(*bundle, arguments);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  if (llvm::Error error = executable->begin(sink))
    FAIL() << llvm::toString(std::move(error));
  for (int64_t rank = 15; rank >= 0; --rank)
    if (llvm::Error error = executable->executeRank(rank))
      FAIL() << llvm::toString(std::move(error));
  auto result = executable->commit();
  ASSERT_TRUE(static_cast<bool>(result))
      << diagnostics << llvm::toString(result.takeError());
  EXPECT_EQ(result->completedRankCount, 16);
  EXPECT_EQ(sink.invocationRankCount, 16u);
  ASSERT_EQ(sink.terminalRanks.size(), 16u);
  for (size_t index = 0; index < sink.terminalRanks.size(); ++index)
    EXPECT_EQ(sink.terminalRanks[index], 15 - static_cast<int64_t>(index));

  llvm::DenseSet<uint64_t> producedEvents;
  llvm::DenseSet<int64_t> beginRanks;
  llvm::DenseSet<int64_t> finishRanks;
  size_t waitCount = 0;
  for (size_t index = 0; index < sink.transactions.size(); ++index) {
    const wafer::compiler::TargetTransaction &transaction =
        sink.transactions[index];
    if (std::holds_alternative<wafer::compiler::TargetDirectDTESendTransaction>(
            transaction.payload) ||
        std::holds_alternative<
            wafer::compiler::TargetDirectDTEReceiveTransaction>(
            transaction.payload)) {
      EXPECT_FALSE(transaction.nccIssueDomain);
      producedEvents.insert(0x2000 + index);
    }
    if (const auto *issue =
            std::get_if<wafer::compiler::TargetDirectDTESendIssueTransaction>(
                &transaction.payload)) {
      EXPECT_FALSE(transaction.nccIssueDomain);
      EXPECT_TRUE(producedEvents.contains(issue->event));
    }
    if (std::holds_alternative<
            wafer::compiler::TargetDirectDTEBeginTransaction>(
            transaction.payload))
      beginRanks.insert(transaction.logicalRank);
    if (std::holds_alternative<
            wafer::compiler::TargetDirectDTEFinishTransaction>(
            transaction.payload))
      finishRanks.insert(transaction.logicalRank);
    if (const auto *wait =
            std::get_if<wafer::compiler::TargetDirectDTEWaitTransaction>(
                &transaction.payload)) {
      ++waitCount;
      EXPECT_FALSE(transaction.nccIssueDomain);
      EXPECT_TRUE(producedEvents.contains(wait->event));
    }
  }
  EXPECT_EQ(beginRanks.size(), 16u);
  EXPECT_EQ(finishRanks.size(), 16u);
  EXPECT_GT(waitCount, 0u);
}

TEST(TargetCallFrontendTest,
     ClusterAggregationConsumesProductionDirectDTERankModules) {
  std::string diagnostics;
  auto bundle = buildDirectDTETargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  auto aggregate =
      wafer::compiler::detail::buildKernelAggregateTargetModule(*bundle);
  ASSERT_TRUE(static_cast<bool>(aggregate))
      << llvm::toString(aggregate.takeError());

  size_t rankBodyCount = 0;
  size_t afterPrepareCalls = 0;
  size_t directSyncInitCalls = 0;
  size_t initTileIdCalls = 0;
  for (const llvm::Function &function : aggregate->module->functions()) {
    if (function.getName().starts_with("__wafer_kernel_rank_") &&
        function.getName().ends_with("_main_body"))
      ++rankBodyCount;
    for (const llvm::BasicBlock &block : function)
      for (const llvm::Instruction &instruction : block) {
        const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        if (!call || !call->getCalledFunction())
          continue;
        if (call->getCalledFunction()->getName() ==
            "wafer_tx81_direct_dte_begin_after_prepare")
          ++afterPrepareCalls;
        if (call->getCalledFunction()->getName() == "direct_sync_init")
          ++directSyncInitCalls;
        if (call->getCalledFunction()->getName() == "init_tile_id")
          ++initTileIdCalls;
      }
  }
  EXPECT_EQ(rankBodyCount, 16u);
  EXPECT_EQ(afterPrepareCalls, 16u);
  EXPECT_EQ(directSyncInitCalls, 1u);
  EXPECT_EQ(initTileIdCalls, 1u);
}

TEST(TargetCallFrontendTest, LateRankFailureAbortsTheWholeInvocation) {
  std::string diagnostics;
  auto bundle = buildDirectDTETargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  std::vector<wafer::compiler::TargetCallRankArguments> arguments;
  for (const wafer::compiler::TargetLLVMModule &module : bundle->getModules())
    arguments.push_back(
        {module.getLogicalRank(),
         std::vector<uint64_t>(module.getKernelABISlots().size(), 0x100000)});
  RecordingSink sink;
  sink.failAtIssue = 20;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError())
                .find("injected transaction sink failure"),
            std::string::npos);
  ASSERT_TRUE(sink.failedLogicalRank.has_value());
  EXPECT_GT(*sink.failedLogicalRank, 0);
  EXPECT_TRUE(sink.aborted);
  EXPECT_FALSE(sink.committed);
  EXPECT_TRUE(sink.transactions.empty());
  EXPECT_TRUE(sink.terminalRanks.empty());
}

TEST(TargetCallFrontendTest, NativeIllegalInlineAssemblyFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  llvm::Module &module =
      const_cast<llvm::Module &>(bundle->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  llvm::IRBuilder<> builder(&entry->getEntryBlock().front());
  llvm::FunctionType *type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(module.getContext()), /*isVarArg=*/false);
  llvm::InlineAsm *assembly =
      llvm::InlineAsm::get(type, "", "", /*hasSideEffects=*/true);
  llvm::CallInst *call = builder.CreateCall(assembly);
  auto restore = llvm::make_scope_exit([&] { call->eraseFromParent(); });

  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("inline assembly"),
            std::string::npos);
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallFrontendTest, NativeTrapFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  llvm::Module &module =
      const_cast<llvm::Module &>(bundle->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  llvm::IRBuilder<> builder(&entry->getEntryBlock().front());
  llvm::Function *trap =
      llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::trap);
  builder.CreateCall(trap);

  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("unsupported intrinsic"),
            std::string::npos);
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallFrontendTest, NativeAddressDereferenceFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  llvm::Module &module =
      const_cast<llvm::Module &>(bundle->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  llvm::IRBuilder<> builder(&entry->getEntryBlock().front());
  llvm::Type *i64 = llvm::Type::getInt64Ty(module.getContext());
  llvm::Value *syntheticAddress = builder.getInt64(0x100000);
  llvm::Value *pointer = builder.CreateIntToPtr(
      syntheticAddress, llvm::PointerType::get(module.getContext(), 0));
  (void)builder.CreateLoad(i64, pointer);

  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  std::string error = llvm::toString(result.takeError());
  EXPECT_TRUE(error.find("inttoptr") != std::string::npos ||
              error.find("load") != std::string::npos)
      << error;
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallFrontendTest, NativePointerSelectFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  llvm::Module &module =
      const_cast<llvm::Module &>(bundle->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  llvm::Type *pointer = llvm::PointerType::get(module.getContext(), 0);
  auto *nullPointer =
      llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointer));
  (void)llvm::SelectInst::Create(
      llvm::ConstantInt::getTrue(module.getContext()), nullPointer, nullPointer,
      "pointer-control", &entry->getEntryBlock().front());

  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("select"),
            std::string::npos);
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallFrontendTest, NativePointerCompareFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  llvm::Module &module =
      const_cast<llvm::Module &>(bundle->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  auto *pointer = llvm::PointerType::get(module.getContext(), 0);
  auto *nullPointer = llvm::ConstantPointerNull::get(pointer);
  (void)new llvm::ICmpInst(&entry->getEntryBlock().front(),
                           llvm::ICmpInst::ICMP_EQ, nullPointer, nullPointer,
                           "pointer-control");

  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("icmp"), std::string::npos);
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallFrontendTest, NativePointerPhiFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  llvm::Module &module =
      const_cast<llvm::Module &>(bundle->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  llvm::BasicBlock *loop =
      llvm::BasicBlock::Create(module.getContext(), "pointer-control", entry);
  llvm::IRBuilder<> builder(loop);
  auto *pointer = llvm::PointerType::get(module.getContext(), 0);
  llvm::PHINode *phi = builder.CreatePHI(pointer, 1);
  builder.CreateBr(loop);
  phi->addIncoming(llvm::ConstantPointerNull::get(pointer), loop);

  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("phi"), std::string::npos);
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallFrontendTest, WrongTargetCallSignatureFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  llvm::Module &module =
      const_cast<llvm::Module &>(bundle->getModules().front().getModule());
  llvm::Function *original = module.getFunction("wafer_tx81_elementwise_add");
  ASSERT_NE(original, nullptr);
  llvm::SmallVector<llvm::Instruction *, 2> calls;
  for (llvm::User *user : original->users())
    calls.push_back(llvm::cast<llvm::Instruction>(user));
  for (llvm::Instruction *call : calls)
    call->eraseFromParent();
  original->eraseFromParent();
  llvm::FunctionType *wrongType = llvm::FunctionType::get(
      llvm::Type::getVoidTy(module.getContext()),
      {llvm::Type::getInt64Ty(module.getContext())}, /*isVarArg=*/false);
  llvm::Function *wrong =
      llvm::Function::Create(wrongType, llvm::GlobalValue::ExternalLinkage,
                             "wafer_tx81_elementwise_add", module);
  (void)wrong;

  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  std::string error = llvm::toString(result.takeError());
  EXPECT_NE(error.find("argument count"), std::string::npos) << error;
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallFrontendTest, SlotMismatchFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {{0, {}}};
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCallFrontend(*bundle, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("slot count"),
            std::string::npos);
  EXPECT_FALSE(sink.began);
  EXPECT_FALSE(sink.aborted);
}

TEST(TargetCallFrontendTest, RejectsRankReentryAndAbortsInvocation) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  auto executable =
      wafer::compiler::prepareTargetCallFrontend(*bundle, arguments);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  sink.reenterExecutable = &*executable;
  ASSERT_FALSE(static_cast<bool>(executable->begin(sink)));
  llvm::Error executionError = executable->executeRank(0);
  ASSERT_TRUE(static_cast<bool>(executionError));
  EXPECT_NE(llvm::toString(std::move(executionError)).find("already running"),
            std::string::npos);
  EXPECT_TRUE(sink.attemptedReentry);
  EXPECT_NE(sink.reentryDiagnostic.find("already running"), std::string::npos);
  EXPECT_TRUE(sink.aborted);
  EXPECT_FALSE(sink.committed);
}

TEST(TargetCallFrontendTest, PrepareCommitFailureCannotPublishResult) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  sink.failPrepareCommit = true;
  auto executable =
      wafer::compiler::prepareTargetCallFrontend(*bundle, arguments);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  ASSERT_FALSE(static_cast<bool>(executable->begin(sink)));
  ASSERT_FALSE(static_cast<bool>(executable->executeRank(0)));
  auto result = executable->commit();
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("prepare-commit failure"),
            std::string::npos);
  EXPECT_TRUE(sink.aborted);
  EXPECT_FALSE(sink.committed);
}

TEST(TargetCallFrontendTest, RunningDestructionAbortsPrivateSinkState) {
  std::string diagnostics;
  auto bundle = buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  std::vector<wafer::compiler::TargetCallRankArguments> arguments = {
      {0,
       std::vector<uint64_t>(
           bundle->getModules().front().getKernelABISlots().size(), 0x100000)}};
  RecordingSink sink;
  {
    auto executable =
        wafer::compiler::prepareTargetCallFrontend(*bundle, arguments);
    ASSERT_TRUE(static_cast<bool>(executable))
        << llvm::toString(executable.takeError());
    ASSERT_FALSE(static_cast<bool>(executable->begin(sink)));
  }
  EXPECT_TRUE(sink.aborted);
  EXPECT_FALSE(sink.committed);
  EXPECT_NE(sink.abortDiagnostic.find("destroyed before commit"),
            std::string::npos);
}

} // namespace
