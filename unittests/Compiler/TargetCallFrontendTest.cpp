//===- TargetCallFrontendTest.cpp - Host target-call frontend tests -----===//

#include "Wafer/Compiler/TargetCallFrontend.h"
#include "Wafer/InitAll.h"
#include "Wafer/Target/TargetCall.h"

#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

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
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
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
buildElementwiseTargetBundle(std::string &diagnosticText) {
  auto context = createCompilerContext();

  auto grouped = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>, policy = "explicit", shape = array<i64: 1>, topology = @default}
  func.func @main(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>) -> tensor<8xf32> {
    %out = tensor.empty() : tensor<8xf32>
    %group = wafer.group ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%out : tensor<8xf32>) {
    ^bb0(%lhs_arg: tensor<8xf32>, %rhs_arg: tensor<8xf32>,
         %out_arg: tensor<8xf32>):
      %sum = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%lhs_arg, %rhs_arg : tensor<8xf32>, tensor<8xf32>)
          outs(%out_arg : tensor<8xf32>) {
        ^bb0(%a: f32, %b: f32, %old: f32):
          %value = arith.addf %a, %b : f32
          linalg.yield %value : f32
        } -> tensor<8xf32>
      wafer.group.yield %sum : tensor<8xf32>
    } : tensor<8xf32>
    return %group : tensor<8xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!grouped)
    return llvm::createStringError("failed to parse target-call test module");

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0), boundary(1)};
  program.distributedOutputs = {boundary(0)};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  if (!config)
    return config.takeError();
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto executable = wafer::compiler::detail::buildExecutableBundle(
      context, *grouped, std::move(program), *config, diagnostics,
      std::nullopt);
  if (!executable)
    return executable.takeError();
  grouped = nullptr;
  return wafer::compiler::compileExecutableBundleToTargetLLVMModules(
      *executable, diagnostics);
}

llvm::Expected<wafer::compiler::TargetLLVMModuleBundle>
buildDirectDTETargetBundle(std::string &diagnosticText) {
  auto context = createCompilerContext();
  auto grouped = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64>, policy = "all_available", shape = array<i64: 16>, topology = @default}
  func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
    %out = tensor.empty() : tensor<4xf32>
    %group = wafer.group ins(%input : tensor<4xf32>)
        outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
      %permuted = wafer.linalg_ext.collective.collective_permute
          ins(%arg0 : tensor<4xf32>) outs(%arg1 : tensor<4xf32>)
          {source_target_pairs = array<i64: 0, 1, 1, 0, 2, 3, 3, 2,
                                            4, 5, 5, 4, 6, 7, 7, 6,
                                            8, 9, 9, 8, 10, 11, 11, 10,
                                            12, 13, 13, 12, 14, 15, 15, 14>,
           channel_id = 91 : i64} -> tensor<4xf32>
      wafer.group.yield %permuted : tensor<4xf32>
    } : tensor<4xf32>
    return %group : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!grouped)
    return llvm::createStringError("failed to parse Direct-DTE test module");

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 1;
  program.distributedInputs = {partitionedBoundary(0)};
  program.distributedOutputs = {partitionedBoundary(0)};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  if (!config)
    return config.takeError();
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto executable = wafer::compiler::detail::buildExecutableBundle(
      context, *grouped, std::move(program), *config, diagnostics,
      std::nullopt);
  if (!executable)
    return executable.takeError();
  grouped = nullptr;
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

std::vector<uint64_t>
makeDecodableArguments(const wafer::TargetCallDescriptor &descriptor) {
  std::vector<uint64_t> arguments;
  arguments.reserve(descriptor.arguments.size());
  for (auto [index, scalar] : llvm::enumerate(descriptor.arguments))
    arguments.push_back(scalar == wafer::TargetCallScalarType::I64
                            ? 0x100000 + index * 0x1000
                            : index + 1);

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
    case wafer::TargetCallBuiltin::TDMAPad:
      arguments[14] = supportedF32Code(wafer::TargetFormatEngine::TDMA);
      break;
    case wafer::TargetCallBuiltin::TDMAImg2Col:
      arguments[18] = supportedF32Code(wafer::TargetFormatEngine::TDMA);
      break;
    case wafer::TargetCallBuiltin::DirectDTEBegin:
      arguments[1] = 16;
      break;
    case wafer::TargetCallBuiltin::DirectDTESendPrepare:
      arguments[6] = 1;
      break;
    case wafer::TargetCallBuiltin::GatherScatter:
    case wafer::TargetCallBuiltin::LocalFence:
    case wafer::TargetCallBuiltin::DirectDTERecvPrepare:
    case wafer::TargetCallBuiltin::DirectDTEWait:
    case wafer::TargetCallBuiltin::DirectDTEFinish:
      break;
    }
    return arguments;
  }
  if (std::holds_alternative<wafer::InstrElementwiseKind>(
          descriptor.semantic)) {
    arguments.back() = supportedF32Code(wafer::TargetFormatEngine::CT);
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

bool payloadMatchesSemantic(
    const wafer::TargetCallSemantic &semantic,
    const wafer::compiler::TargetTransactionPayload &payload) {
  if (const auto *builtin = std::get_if<wafer::TargetCallBuiltin>(&semantic)) {
    switch (*builtin) {
    case wafer::TargetCallBuiltin::RDMA:
    case wafer::TargetCallBuiltin::WDMA:
      return std::holds_alternative<
          wafer::compiler::TargetStridedDMATransaction>(payload);
    case wafer::TargetCallBuiltin::GatherScatter:
      return std::holds_alternative<
          wafer::compiler::TargetGatherScatterTransaction>(payload);
    case wafer::TargetCallBuiltin::Memset:
      return std::holds_alternative<wafer::compiler::TargetMemsetTransaction>(
          payload);
    case wafer::TargetCallBuiltin::Bit2FP:
      return std::holds_alternative<wafer::compiler::TargetBit2FPTransaction>(
          payload);
    case wafer::TargetCallBuiltin::MaskMove:
      return std::holds_alternative<wafer::compiler::TargetMaskMoveTransaction>(
          payload);
    case wafer::TargetCallBuiltin::Gemm:
      return std::holds_alternative<wafer::compiler::TargetGemmTransaction>(
          payload);
    case wafer::TargetCallBuiltin::TDMAPad:
    case wafer::TargetCallBuiltin::TDMAImg2Col:
      return std::holds_alternative<
          wafer::compiler::TargetTDMATransformTransaction>(payload);
    case wafer::TargetCallBuiltin::LocalFence:
      return std::holds_alternative<
          wafer::compiler::TargetLocalFenceTransaction>(payload);
    case wafer::TargetCallBuiltin::DirectDTEBegin:
      return std::holds_alternative<
          wafer::compiler::TargetDirectDTEBeginTransaction>(payload);
    case wafer::TargetCallBuiltin::DirectDTESendPrepare:
      return std::holds_alternative<
          wafer::compiler::TargetDirectDTESendTransaction>(payload);
    case wafer::TargetCallBuiltin::DirectDTERecvPrepare:
      return std::holds_alternative<
          wafer::compiler::TargetDirectDTEReceiveTransaction>(payload);
    case wafer::TargetCallBuiltin::DirectDTEWait:
      return std::holds_alternative<
          wafer::compiler::TargetDirectDTEWaitTransaction>(payload);
    case wafer::TargetCallBuiltin::DirectDTEFinish:
      return std::holds_alternative<
          wafer::compiler::TargetDirectDTEFinishTransaction>(payload);
    }
  }
  if (std::holds_alternative<wafer::InstrElementwiseKind>(semantic))
    return std::holds_alternative<
        wafer::compiler::TargetElementwiseTransaction>(payload);
  if (std::holds_alternative<wafer::InstrReduceKind>(semantic))
    return std::holds_alternative<wafer::compiler::TargetReduceTransaction>(
        payload);
  if (std::holds_alternative<wafer::InstrConvertKind>(semantic))
    return std::holds_alternative<wafer::compiler::TargetConvertTransaction>(
        payload);
  if (std::holds_alternative<wafer::InstrConvKind>(semantic))
    return std::holds_alternative<wafer::compiler::TargetConvTransaction>(
        payload);
  if (std::holds_alternative<wafer::InstrPoolKind>(semantic))
    return std::holds_alternative<wafer::compiler::TargetPoolTransaction>(
        payload);
  if (std::holds_alternative<wafer::InstrUnpoolKind>(semantic))
    return std::holds_alternative<wafer::compiler::TargetUnpoolTransaction>(
        payload);
  if (const auto *kind = std::get_if<wafer::InstrPeripheralKind>(&semantic)) {
    switch (*kind) {
    case wafer::InstrPeripheralKind::ArgMax:
    case wafer::InstrPeripheralKind::ArgMin:
      return std::holds_alternative<
          wafer::compiler::TargetPeripheralArgExtremaTransaction>(payload);
    case wafer::InstrPeripheralKind::Bilinear:
      return std::holds_alternative<
          wafer::compiler::TargetPeripheralBilinearTransaction>(payload);
    case wafer::InstrPeripheralKind::Lut16:
    case wafer::InstrPeripheralKind::Lut32:
      return std::holds_alternative<
          wafer::compiler::TargetPeripheralLUTTransaction>(payload);
    case wafer::InstrPeripheralKind::RandGen:
      return std::holds_alternative<
          wafer::compiler::TargetPeripheralRandomTransaction>(payload);
    case wafer::InstrPeripheralKind::ElemMask:
      return std::holds_alternative<
          wafer::compiler::TargetPeripheralElementMaskTransaction>(payload);
    case wafer::InstrPeripheralKind::Count:
    case wafer::InstrPeripheralKind::Factorize:
      return false;
    }
  }
  return false;
}

TEST(TargetCallRegistryTest, ExactlyCoversTypedTargetCallSurface) {
  llvm::ArrayRef<wafer::TargetCallDescriptor> descriptors =
      wafer::getTargetCallDescriptors();
  ASSERT_EQ(descriptors.size(), 109u);
  llvm::DenseSet<llvm::StringRef> symbols;
  for (const wafer::TargetCallDescriptor &descriptor : descriptors) {
    EXPECT_TRUE(llvm::StringRef(descriptor.symbol).starts_with("wafer_tx81_"));
    EXPECT_TRUE(symbols.insert(descriptor.symbol).second);
    EXPECT_EQ(wafer::findTargetCallDescriptor(descriptor.symbol), &descriptor);
    EXPECT_EQ(wafer::findTargetCallDescriptor(descriptor.semantic),
              &descriptor);
  }
  EXPECT_EQ(wafer::findTargetCallDescriptor("wafer_tx81_unknown"), nullptr);

  const auto &send = wafer::getTargetCallDescriptor(
      wafer::TargetCallBuiltin::DirectDTESendPrepare);
  EXPECT_EQ(send.result, wafer::TargetCallResultType::I64);
  EXPECT_EQ(send.arguments.size(), 7u);
  EXPECT_EQ(send.arguments[0], wafer::TargetCallScalarType::I64);
  EXPECT_EQ(send.arguments[2], wafer::TargetCallScalarType::I32);
  EXPECT_EQ(wafer::getTargetCallDescriptor(wafer::InstrElementwiseKind::Abs)
                .arguments.size(),
            4u);
  EXPECT_EQ(wafer::getTargetCallDescriptor(wafer::InstrElementwiseKind::Add)
                .arguments.size(),
            5u);
  EXPECT_EQ(wafer::getTargetCallDescriptor(wafer::InstrConvKind::Conv)
                .arguments.size(),
            31u);
  EXPECT_EQ(wafer::getTargetCallDescriptor(wafer::InstrPeripheralKind::Bilinear)
                .arguments.size(),
            17u);
}

TEST(TargetCallRegistryTest, EveryDescriptorDecodesOneTypedPayload) {
  wafer::TargetCallDecodeContext context{
      wafer::TargetProfileId::waferTx81SingleCardKernelV1(), 16};
  size_t decoded = 0;
  for (const wafer::TargetCallDescriptor &descriptor :
       wafer::getTargetCallDescriptors()) {
    std::vector<uint64_t> arguments = makeDecodableArguments(descriptor);
    auto payload =
        wafer::decodeTargetCallPayload(descriptor, context, arguments);
    ASSERT_TRUE(static_cast<bool>(payload))
        << descriptor.symbol << ": " << llvm::toString(payload.takeError());
    EXPECT_TRUE(payloadMatchesSemantic(descriptor.semantic, *payload))
        << descriptor.symbol;
    ++decoded;
  }
  EXPECT_EQ(decoded, 109u);
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
    if (const auto *elementwise =
            std::get_if<wafer::compiler::TargetElementwiseTransaction>(
                &transaction.payload)) {
      sawAdd = true;
      EXPECT_EQ(elementwise->kind, wafer::InstrElementwiseKind::Add);
      EXPECT_TRUE(elementwise->rhs.has_value());
      EXPECT_EQ(elementwise->elementCount, 8u);
      EXPECT_EQ(elementwise->format, wafer::LogicalFormat::F32);
    }
  }
  EXPECT_TRUE(sawAdd);
  EXPECT_EQ(bundle->getModules().front().getTargetTriple(), originalTriple);
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
            transaction.payload))
      producedEvents.insert(0x2000 + index);
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
      EXPECT_TRUE(producedEvents.contains(wait->event));
    }
  }
  EXPECT_EQ(beginRanks.size(), 16u);
  EXPECT_EQ(finishRanks.size(), 16u);
  EXPECT_GT(waitCount, 0u);
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
