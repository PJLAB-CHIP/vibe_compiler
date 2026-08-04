//===- SystemCTargetModelFixedSlotIntegrationTest.cpp -------------------===//

#include "Wafer/Model/SystemCTargetModel.h"

#include "Wafer/Compiler/ExecutableBundleInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Target/PhysicalTensorCodec.h"

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

constexpr int64_t kElementCount = 8388608;
constexpr uint64_t kTensorBytes =
    static_cast<uint64_t>(kElementCount) * sizeof(uint16_t);
constexpr uint64_t kABISlotBase = UINT64_C(0x10000000);
constexpr uint64_t kABISlotStride = UINT64_C(0x01000000);

frontend::ProgramBoundaryBinding replicatedBoundary(int64_t index) {
  frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = frontend::ProgramDistributionKind::Replicated;
  binding.globalShape = {kElementCount};
  binding.localShape = {kElementCount};
  binding.dtype = "f16";
  frontend::ProgramRankSlice slice;
  slice.logicalRank = 0;
  slice.replicaId = 0;
  slice.offsets = {0};
  slice.sizes = {kElementCount};
  slice.strides = {1};
  binding.rankSlices.push_back(std::move(slice));
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
  registerAllDialects(registry);
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

llvm::Expected<ExecutableBundle>
buildFixedSlotExecutableForModelTest(std::string &diagnosticText) {
  auto context = createCompilerContext();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%lhs: tensor<8388608xf16>, %rhs: tensor<8388608xf16>)
      -> tensor<8388608xf16> {
    %out = tensor.empty() : tensor<8388608xf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%lhs, %rhs : tensor<8388608xf16>, tensor<8388608xf16>)
      outs(%out : tensor<8388608xf16>) {
    ^bb0(%lhs_value: f16, %rhs_value: f16, %unused: f16):
      %value = arith.addf %lhs_value, %rhs_value : f16
      linalg.yield %value : f16
    } -> tensor<8388608xf16>
    return %sum : tensor<8388608xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!tensorProgram)
    return llvm::createStringError(
        "failed to parse fixed-slot model test source");

  frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {replicatedBoundary(0), replicatedBoundary(1)};
  program.distributedOutputs = {replicatedBoundary(0)};
  llvm::Expected<ExecutionConfig> config = ExecutionConfig::createForSingleCard(
      1, RuntimeLaunchKind::Kernel);
  if (!config)
    return config.takeError();

  llvm::raw_string_ostream diagnostics(diagnosticText);
  return wafer::compiler::detail::buildExecutableBundle(
      context, *tensorProgram, std::move(program), *config, diagnostics,
      std::nullopt,
      wafer::compiler::detail::WholeVariantSelectionMode::
          QualifyStaticFixedSlot);
}

size_t countCallsTo(const llvm::Module &module, llvm::StringRef symbol) {
  size_t count = 0;
  for (const llvm::Function &function : module)
    for (const llvm::BasicBlock &block : function)
      for (const llvm::Instruction &instruction : block) {
        const auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
        const llvm::Function *callee =
            call ? call->getCalledFunction() : nullptr;
        count += callee && callee->getName() == symbol;
      }
  return count;
}

class RecordingTargetSink final : public TargetTransactionSink {
public:
  llvm::Error begin(const TargetCallInvocationDescriptor &descriptor) override {
    invocation = descriptor;
    return llvm::Error::success();
  }

  llvm::Expected<uint64_t>
  issue(const TargetTransaction &transaction) override {
    transactions.push_back(transaction);
    return nextEvent++;
  }

  llvm::Error terminal(int64_t logicalRank) override {
    terminalRanks.push_back(logicalRank);
    return llvm::Error::success();
  }

  llvm::Error prepareCommit() override { return llvm::Error::success(); }
  void commit() override { committed = true; }
  void abort(llvm::StringRef diagnostic) override {
    aborted = diagnostic.str();
  }

  std::optional<TargetCallInvocationDescriptor> invocation;
  std::vector<TargetTransaction> transactions;
  std::vector<int64_t> terminalRanks;
  uint64_t nextEvent = 1;
  bool committed = false;
  std::string aborted;
};

struct ABIRange {
  KernelABISlotRole role;
  int64_t resourceIndex;
  uint64_t begin;
  uint64_t end;
};

const ABIRange *findContainingRange(llvm::ArrayRef<ABIRange> ranges,
                                    uint64_t address, uint64_t byteCount) {
  for (const ABIRange &range : ranges) {
    if (address < range.begin || address > range.end)
      continue;
    if (byteCount <= range.end - address)
      return &range;
  }
  return nullptr;
}

TEST(SystemCTargetModelFixedSlotIntegrationTest,
     ExecutesRotatingSlotsWithOnlyTerminalWorkerJoin) {
  std::string diagnosticText;
  llvm::Expected<ExecutableBundle> executable =
      buildFixedSlotExecutableForModelTest(diagnosticText);
  ASSERT_TRUE(static_cast<bool>(executable))
      << diagnosticText << llvm::toString(executable.takeError());
  ASSERT_EQ(executable->getRankExecutables().size(), 1u);
  EXPECT_EQ(executable->getExecutionConfig().getTargetIdentityId(),
            TargetIdentityId::waferTx81SingleCard());

  mlir::ModuleOp selected =
      executable->getRankExecutables().front().getModule();
  selected.walk([&](InstrRDMAOp operation) {
    operation.setWorkerAttr(
        NCCWorkerAttr::get(selected.getContext(), NCCWorker::Worker1));
  });
  selected.walk([&](InstrElementwiseOp operation) {
    operation.setWorkerAttr(
        NCCWorkerAttr::get(selected.getContext(), NCCWorker::Worker2));
  });
  selected.walk([&](InstrWDMAOp operation) {
    operation.setWorkerAttr(
        NCCWorkerAttr::get(selected.getContext(), NCCWorker::Worker1));
  });
  selected.walk([&](SyncNCCJoinOp operation) {
    operation.setParticipantsAttr(
        mlir::DenseI64ArrayAttr::get(selected.getContext(), {1, 2}));
  });
  llvm::SmallVector<mlir::scf::ForOp, 2> loops;
  llvm::SmallVector<SyncNCCJoinOp, 2> joins;
  unsigned nestedSPMAllocationCount = 0;
  unsigned externalSPMAllocationCount = 0;
  bool missingSPMOffset = false;
  std::set<int64_t> spmOffsets;
  selected.walk([&](mlir::scf::ForOp loop) { loops.push_back(loop); });
  selected.walk([&](SyncNCCJoinOp join) { joins.push_back(join); });
  selected.walk([&](mlir::memref::AllocOp allocation) {
    if (!isWaferSPMMemRefType(allocation.getType()))
      return;
    if (allocation->getParentOfType<mlir::scf::ForOp>()) {
      ++nestedSPMAllocationCount;
      return;
    }
    ++externalSPMAllocationCount;
    auto offset =
        allocation->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
    if (!offset) {
      missingSPMOffset = true;
      return;
    }
    spmOffsets.insert(offset.getOffset());
  });

  ASSERT_EQ(loops.size(), 1u);
  unsigned steadyRDMACount = 0;
  unsigned steadyAddCount = 0;
  unsigned steadyWDMACount = 0;
  unsigned steadyJoinCount = 0;
  loops.front().walk([&](InstrRDMAOp) { ++steadyRDMACount; });
  loops.front().walk([&](InstrElementwiseOp operation) {
    steadyAddCount += operation.getKind() == InstrElementwiseKind::Add;
  });
  loops.front().walk([&](InstrWDMAOp) { ++steadyWDMACount; });
  loops.front().walk([&](SyncNCCJoinOp) { ++steadyJoinCount; });
  EXPECT_EQ(steadyRDMACount, 2u);
  EXPECT_EQ(steadyAddCount, 1u);
  EXPECT_EQ(steadyWDMACount, 1u);
  EXPECT_EQ(steadyJoinCount, 0u);
  ASSERT_EQ(joins.size(), 1u);
  EXPECT_FALSE(joins.front()->getParentOfType<mlir::scf::ForOp>());
  ASSERT_EQ(joins.front().getParticipants().size(), 2u);
  EXPECT_EQ(joins.front().getParticipants()[0], 1);
  EXPECT_EQ(joins.front().getParticipants()[1], 2);
  EXPECT_EQ(nestedSPMAllocationCount, 0u);
  EXPECT_EQ(externalSPMAllocationCount, 6u);
  EXPECT_FALSE(missingSPMOffset);
  EXPECT_EQ(spmOffsets.size(), 6u);

  llvm::raw_string_ostream diagnostics(diagnosticText);
  llvm::Expected<TargetLLVMModuleBundle> targetBundle =
      compileExecutableBundleToTargetLLVMModules(*executable, diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetBundle))
      << diagnosticText << llvm::toString(targetBundle.takeError());
  ASSERT_EQ(targetBundle->getModules().size(), 1u);
  const TargetLLVMModule &targetModule = targetBundle->getModules().front();
  EXPECT_EQ(targetModule.getTargetIdentityId(),
            executable->getExecutionConfig().getTargetIdentityId());

  const llvm::Module &llvmModule = targetModule.getModule();
  const size_t targetAddCallCount = countCallsTo(
      llvmModule, getTargetCallDescriptor(InstrElementwiseKind::Add)
                      .symbol);
  EXPECT_EQ(targetAddCallCount, 3u);
  EXPECT_EQ(countCallsTo(llvmModule,
                         getTargetCallDescriptor(TargetCallBuiltin::RDMA)
                             .symbol),
            targetAddCallCount * 2);
  EXPECT_EQ(countCallsTo(llvmModule,
                         getTargetCallDescriptor(TargetCallBuiltin::WDMA)
                             .symbol),
            targetAddCallCount);
  EXPECT_EQ(countCallsTo(llvmModule,
                         getTargetCallDescriptor(TargetCallBuiltin::NCCJoin)
                             .symbol),
            1u);

  TargetCallRankArguments rankArguments{0, {}};
  std::vector<ABIRange> abiRanges;
  std::vector<TargetModelInputBinding> inputBindings;
  const std::array<uint64_t, 2> inputBits{UINT64_C(0x3c00), UINT64_C(0x4000)};
  for (const KernelABISlot &slot : targetModule.getKernelABISlots()) {
    ASSERT_GE(slot.ordinal, 0);
    ASSERT_EQ(slot.byteSize, static_cast<int64_t>(kTensorBytes));
    const uint64_t base =
        kABISlotBase + static_cast<uint64_t>(slot.ordinal) * kABISlotStride;
    rankArguments.slots.push_back(base);
    abiRanges.push_back({slot.role, slot.resourceIndex, base,
                         base + static_cast<uint64_t>(slot.byteSize)});
    if (slot.role != KernelABISlotRole::UserInput)
      continue;
    ASSERT_GE(slot.resourceIndex, 0);
    ASSERT_LT(static_cast<size_t>(slot.resourceIndex), inputBits.size());
    std::vector<RawLogicalValue> values(
        static_cast<size_t>(kElementCount),
        RawLogicalValue{LogicalFormat::F16,
                        inputBits[static_cast<size_t>(slot.resourceIndex)]});
    NumericTensorKey key = llvm::cantFail(NumericTensorKey::create(
        LogicalFormat::F16, MemLayout::Tensor, {kElementCount}));
    llvm::Expected<std::vector<uint8_t>> bytes =
        packPhysicalTensorLogicalValues(key, values, UINT8_C(0));
    ASSERT_TRUE(static_cast<bool>(bytes)) << llvm::toString(bytes.takeError());
    ASSERT_EQ(bytes->size(), kTensorBytes);
    inputBindings.push_back({0, slot.ordinal, std::move(*bytes)});
  }
  ASSERT_EQ(inputBindings.size(), 2u);
  std::vector<TargetCallRankArguments> arguments{rankArguments};

  RecordingTargetSink recording;
  llvm::Expected<TargetCallExecutionResult> decoded =
      executeTargetCallFrontend(*targetBundle, arguments, recording);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  EXPECT_TRUE(recording.committed);
  EXPECT_TRUE(recording.aborted.empty());
  ASSERT_TRUE(recording.invocation.has_value());
  EXPECT_EQ(recording.invocation->targetIdentity,
            TargetIdentityId::waferTx81SingleCard());
  EXPECT_EQ(recording.terminalRanks, std::vector<int64_t>({0}));
  EXPECT_EQ(decoded->issuedTransactionCount, recording.transactions.size());

  std::array<std::set<uint64_t>, 2> inputSPMSlots;
  std::set<uint64_t> outputSPMSlots;
  std::vector<TargetElementwiseTransaction> adds;
  std::vector<char> issueKinds;
  size_t readCount = 0;
  size_t writeCount = 0;
  size_t joinCount = 0;
  size_t firstWrite = recording.transactions.size();
  size_t lastRead = 0;
  for (auto [index, transaction] : llvm::enumerate(recording.transactions)) {
    EXPECT_EQ(transaction.logicalRank, 0);
    EXPECT_EQ(transaction.issueOrdinal, index);
    if (const auto *dma =
            std::get_if<TargetStridedDMATransaction>(&transaction.payload)) {
      ASSERT_TRUE(transaction.nccIssueDomain.has_value());
      EXPECT_EQ(transaction.nccIssueDomain->worker, NCCWorker::Worker1);
      EXPECT_EQ(transaction.nccIssueDomain->completionBehavior,
                LocalInstructionCompletion::OrderedPending);
      if (dma->direction == TargetDMADirection::Read) {
        ++readCount;
        lastRead = index;
        issueKinds.push_back('R');
        const ABIRange *source =
            findContainingRange(abiRanges, dma->source, dma->byteCount);
        ASSERT_NE(source, nullptr);
        ASSERT_EQ(source->role, KernelABISlotRole::UserInput);
        ASSERT_GE(source->resourceIndex, 0);
        ASSERT_LT(static_cast<size_t>(source->resourceIndex),
                  inputSPMSlots.size());
        inputSPMSlots[static_cast<size_t>(source->resourceIndex)].insert(
            dma->destination);
      } else {
        ++writeCount;
        firstWrite = std::min(firstWrite, static_cast<size_t>(index));
        issueKinds.push_back('W');
        const ABIRange *destination =
            findContainingRange(abiRanges, dma->destination, dma->byteCount);
        ASSERT_NE(destination, nullptr);
        ASSERT_EQ(destination->role, KernelABISlotRole::Output);
        outputSPMSlots.insert(dma->source);
      }
      continue;
    }
    if (const auto *add =
            std::get_if<TargetElementwiseTransaction>(&transaction.payload)) {
      ASSERT_TRUE(transaction.nccIssueDomain.has_value());
      EXPECT_EQ(transaction.nccIssueDomain->engine, TargetCallTSMEngine::CT);
      EXPECT_EQ(transaction.nccIssueDomain->worker, NCCWorker::Worker2);
      EXPECT_EQ(transaction.nccIssueDomain->completionBehavior,
                LocalInstructionCompletion::OrderedPending);
      EXPECT_EQ(add->kind, InstrElementwiseKind::Add);
      ASSERT_TRUE(add->rhs.has_value());
      adds.push_back(*add);
      issueKinds.push_back('A');
      continue;
    }
    if (const auto *join =
            std::get_if<TargetNCCJoinTransaction>(&transaction.payload)) {
      ++joinCount;
      EXPECT_FALSE(transaction.nccIssueDomain.has_value());
      EXPECT_EQ(join->participantMask,
                (uint32_t{1} << static_cast<uint32_t>(NCCWorker::Worker1)) |
                    (uint32_t{1} << static_cast<uint32_t>(NCCWorker::Worker2)));
      EXPECT_EQ(index + 1, recording.transactions.size());
      issueKinds.push_back('J');
      continue;
    }
    FAIL() << "fixed-slot target stream contains an unexpected transaction";
  }

  ASSERT_GE(adds.size(), 3u);
  EXPECT_EQ(readCount, adds.size() * 2);
  EXPECT_EQ(writeCount, adds.size());
  EXPECT_EQ(joinCount, 1u);
  EXPECT_LT(firstWrite, lastRead);
  ASSERT_EQ(issueKinds.back(), 'J');
  EXPECT_EQ(inputSPMSlots[0].size(), 2u);
  EXPECT_EQ(inputSPMSlots[1].size(), 2u);
  EXPECT_EQ(outputSPMSlots.size(), 2u);
  std::set<uint64_t> allDynamicSPMSlots = inputSPMSlots[0];
  allDynamicSPMSlots.insert(inputSPMSlots[1].begin(), inputSPMSlots[1].end());
  allDynamicSPMSlots.insert(outputSPMSlots.begin(), outputSPMSlots.end());
  EXPECT_EQ(allDynamicSPMSlots.size(), 6u);
  for (const TargetElementwiseTransaction &add : adds) {
    EXPECT_NE(inputSPMSlots[0].find(add.lhs), inputSPMSlots[0].end());
    ASSERT_TRUE(add.rhs.has_value());
    EXPECT_NE(inputSPMSlots[1].find(*add.rhs), inputSPMSlots[1].end());
    EXPECT_NE(outputSPMSlots.find(add.destination), outputSPMSlots.end());
  }

  llvm::Expected<TargetCallExecutable> frontend =
      prepareTargetCallFrontend(*targetBundle, arguments);
  ASSERT_TRUE(static_cast<bool>(frontend))
      << llvm::toString(frontend.takeError());
  llvm::Expected<TargetModelResult> result = executeSystemCTargetModel(
      std::move(*frontend), inputBindings,
      TargetModelKernelBudget::create(
          FormalNumericWorkBudget::create(
              /*maximumScalarEvaluations=*/kElementCount,
              /*maximumFusedMultiplyAdds=*/kElementCount),
          /*maximumMovementBytes=*/kTensorBytes,
          /*maximumMovementSegments=*/kElementCount));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->completedRankCount, 1);
  EXPECT_EQ(result->issuedTransactionCount, recording.transactions.size());
  EXPECT_EQ(result->formalNumericCommandCount, adds.size());
  EXPECT_GT(result->finalDeltaCount, 0u);
  EXPECT_EQ(result->schedulerIdentity, "untimed-delta-worker-aware-ncc-v2");
  EXPECT_FALSE(result->numericFlags.invalid);
  EXPECT_FALSE(result->numericFlags.divByZero);
  EXPECT_FALSE(result->numericFlags.overflow);
  EXPECT_FALSE(result->numericFlags.underflow);
  ASSERT_EQ(result->outputs.size(), 1u);

  NumericTensorKey outputKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F16, MemLayout::Tensor, {kElementCount}));
  llvm::Expected<std::vector<RawLogicalValue>> output =
      unpackPhysicalTensorLogicalValues(outputKey,
                                        result->outputs.front().bytes);
  ASSERT_TRUE(static_cast<bool>(output)) << llvm::toString(output.takeError());
  ASSERT_EQ(output->size(), static_cast<size_t>(kElementCount));
  for (const RawLogicalValue &value : *output) {
    EXPECT_EQ(value.format, LogicalFormat::F16);
    EXPECT_EQ(value.bits, UINT64_C(0x4200));
  }
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
