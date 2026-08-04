#include "../../lib/Wafer/Compiler/NoCPartialDataflow.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/DirectDTETransport.h"
#include "../../lib/Wafer/Compiler/NoCResidentDataflow.h"
#include "../../lib/Wafer/Compiler/WholeVariantResourceAcceptance.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/Internal.h"
#include "Wafer/IR/Common/OpVerifierUtils.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

namespace {

using OwnedModules = llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 2>;

static void replaceAll(std::string &text, llvm::StringRef from,
                       llvm::StringRef to) {
  size_t position = 0;
  while ((position = text.find(from.str(), position)) != std::string::npos) {
    text.replace(position, from.size(), to.str());
    position += to.size();
  }
}

static unsigned countRDMA(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](wafer::InstrRDMAOp) { ++count; });
  return count;
}

static unsigned countWDMA(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](wafer::InstrWDMAOp) { ++count; });
  return count;
}

static unsigned countSends(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](wafer::InstrDTESendOp) { ++count; });
  return count;
}

static unsigned countRecvs(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](wafer::InstrDTERecvOp) { ++count; });
  return count;
}

static unsigned countWaits(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](wafer::InstrDTEWaitOp) { ++count; });
  return count;
}

static unsigned countPrivateDDRAllocations(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](mlir::memref::AllocOp allocation) {
    if (wafer::detail::hasWaferMemorySpace(allocation.getType(),
                                           wafer::MemorySpace::DDR))
      ++count;
  });
  return count;
}

static std::string snapshot(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

class NoCPartialDataflowTest : public ::testing::Test {
protected:
  NoCPartialDataflowTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  parseRank(int64_t rank, int64_t elements = 4,
            wafer::tile_region_to_instr::AllReduceSchedule schedule =
                wafer::tile_region_to_instr::AllReduceSchedule::Tree,
            int64_t rankCount = 2) {
    std::string source = R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: __GRID__>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"],
       shape = array<i64: __RANK_COUNT__>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(
      %output: memref<__ELEMENTS__xf32,
                      #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %output : memref<__ELEMENTS__xf32,
                         #wafer.memory<ddr, tensor>>)
        -> (memref<__ELEMENTS__xf32,
                   #wafer.memory<ddr, tensor>>) {
    ^bb0(%dest: memref<__ELEMENTS__xf32,
                       #wafer.memory<ddr, tensor>>):
      %seed = memref.alloc()
          : memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>
      %partial = memref.alloc()
          : memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <mul> %seed, %seed into %partial
          : memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>,
            memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>
        into memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>
      %spill = memref.alloc()
          : memref<__ELEMENTS__xf32, #wafer.memory<ddr, tensor>>
      wafer.instr.wdma %partial to %spill
          {byte_count = __BYTES__ : i64, inner_bytes = __BYTES__ : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>
         to memref<__ELEMENTS__xf32, #wafer.memory<ddr, tensor>>
      %reloaded = memref.alloc()
          : memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %spill to %reloaded
          {byte_count = __BYTES__ : i64, inner_bytes = __BYTES__ : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<__ELEMENTS__xf32, #wafer.memory<ddr, tensor>>
         to memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>
      %reduced = wafer.tile.all_reduce #wafer.reduce_kind<sum>
          %reloaded using %recv
          {local_rank = __RANK__ : i64,
           group_size = __RANK_COUNT__ : i64,
           rank_group = array<i64: __RANK_GROUP__>,
           bytes = __BYTES__ : i64,
           communication_id = 41 : i64}
          : (memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>,
             memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>)
         -> memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %reduced to %dest
          {byte_count = __BYTES__ : i64, inner_bytes = __BYTES__ : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<__ELEMENTS__xf32, #wafer.memory<spm, tensor>>
         to memref<__ELEMENTS__xf32, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %dest
          : memref<__ELEMENTS__xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir";
    replaceAll(source, "__ELEMENTS__", std::to_string(elements));
    replaceAll(source, "__BYTES__", std::to_string(elements * 4));
    replaceAll(source, "__RANK__", std::to_string(rank));
    replaceAll(source, "__RANK_COUNT__", std::to_string(rankCount));
    replaceAll(source, "__GRID__", rankCount == 16 ? "4, 4" : "1, 2");
    std::string rankGroup;
    llvm::raw_string_ostream groupStream(rankGroup);
    for (int64_t member = 0; member < rankCount; ++member) {
      if (member)
        groupStream << ", ";
      groupStream << member;
    }
    groupStream.flush();
    replaceAll(source, "__RANK_GROUP__", rankGroup);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
    if (!module)
      return {};

    wafer::tile_region_to_instr::TileRegionToInstrOptions options;
    options.allReduceSchedule = schedule;
    std::string failure;
    if (mlir::failed(
            wafer::tile_region_to_instr::convertTileRegionToInstrModule(
                *module, options, &failure))) {
      ADD_FAILURE() << failure;
      return {};
    }
    if (mlir::failed(mlir::verify(*module))) {
      ADD_FAILURE() << "lowered rank module failed verification";
      return {};
    }
    return module;
  }

  OwnedModules
  makePair(int64_t rank0Elements = 4, int64_t rank1Elements = 4,
           wafer::tile_region_to_instr::AllReduceSchedule schedule =
               wafer::tile_region_to_instr::AllReduceSchedule::Tree) {
    OwnedModules modules;
    auto rank0 = parseRank(0, rank0Elements, schedule);
    auto rank1 = parseRank(1, rank1Elements, schedule);
    if (!rank0 || !rank1)
      return {};
    modules.push_back(std::move(rank0));
    modules.push_back(std::move(rank1));
    return modules;
  }

  wafer::frontend::FrontendProgramVerificationResult
  makeProgram(int64_t elements = 4, int64_t rankCount = 2) {
    wafer::frontend::FrontendProgramVerificationResult program;
    program.logicalRankCount = rankCount;
    wafer::frontend::ProgramBoundaryBinding output;
    output.index = 0;
    output.programIndex = 0;
    output.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
    output.globalShape = {elements};
    output.localShape = {elements};
    output.dtype = "f32";
    for (int64_t rank = 0; rank < rankCount; ++rank) {
      wafer::frontend::ProgramRankSlice slice;
      slice.logicalRank = rank;
      slice.replicaId = rank;
      slice.offsets = {0};
      slice.sizes = {elements};
      slice.strides = {1};
      output.rankSlices.push_back(std::move(slice));
    }
    program.distributedOutputs.push_back(std::move(output));
    return program;
  }

  llvm::SmallVector<mlir::ModuleOp, 2> views(OwnedModules &owners) {
    llvm::SmallVector<mlir::ModuleOp, 2> modules;
    for (auto &owner : owners)
      modules.push_back(*owner);
    return modules;
  }

  wafer::InstrWDMAOp findSpillStore(mlir::ModuleOp module) {
    wafer::InstrWDMAOp result;
    module.walk([&](wafer::InstrWDMAOp store) {
      if (!result && store.getDest().getDefiningOp<mlir::memref::AllocOp>())
        result = store;
    });
    return result;
  }

  wafer::InstrRDMAOp findSpillLoad(mlir::ModuleOp module) {
    wafer::InstrRDMAOp result;
    module.walk([&](wafer::InstrRDMAOp load) {
      if (!result && load.getSource().getDefiningOp<mlir::memref::AllocOp>())
        result = load;
    });
    return result;
  }

  wafer::InstrWDMAOp findPublisher(mlir::ModuleOp module) {
    wafer::InstrWDMAOp result;
    module.walk([&](wafer::InstrWDMAOp store) {
      if (mlir::isa<mlir::BlockArgument>(store.getDest()))
        result = store;
    });
    return result;
  }

  wafer::InstrElementwiseOp findProducer(mlir::ModuleOp module) {
    wafer::InstrWDMAOp spill = findSpillStore(module);
    wafer::InstrElementwiseOp result;
    module.walk([&](wafer::InstrElementwiseOp operation) {
      if (operation.getDest() == spill.getSource())
        result = operation;
    });
    return result;
  }

  mlir::Value
  recreateRingSubview(mlir::Operation *anchor, mlir::Value value,
                      int64_t oldChunkElements, int64_t newChunkElements,
                      std::optional<int64_t> forcedOffset = std::nullopt) {
    auto oldSubview = value.getDefiningOp<mlir::memref::SubViewOp>();
    if (!oldSubview || oldSubview.getStaticOffsets().size() != 1 ||
        oldSubview.getStaticOffsets().front() == mlir::ShapedType::kDynamic ||
        oldChunkElements <= 0 || newChunkElements <= 0)
      return {};
    int64_t oldOffset = oldSubview.getStaticOffsets().front();
    if (oldOffset < 0 || oldOffset % oldChunkElements != 0)
      return {};
    int64_t logicalSlice = oldOffset / oldChunkElements;
    int64_t newOffset = forcedOffset.value_or(logicalSlice * newChunkElements);
    mlir::OpBuilder builder(anchor);
    llvm::SmallVector<mlir::OpFoldResult, 1> offsets{
        builder.getIndexAttr(newOffset)};
    llvm::SmallVector<mlir::OpFoldResult, 1> sizes{
        builder.getIndexAttr(newChunkElements)};
    llvm::SmallVector<mlir::OpFoldResult, 1> strides{builder.getIndexAttr(1)};
    return builder
        .create<mlir::memref::SubViewOp>(
            anchor->getLoc(), oldSubview.getSource(), offsets, sizes, strides)
        .getResult();
  }

  void shrinkRingProtocolToTail(OwnedModules &owners, int64_t oldChunkElements,
                                int64_t newChunkElements) {
    const int64_t newBytes = newChunkElements * 4;
    for (auto &owner : owners) {
      llvm::SmallVector<mlir::Operation *, 16> operations;
      owner->walk([&](mlir::Operation *operation) {
        if (mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTERecvOp,
                      wafer::InstrElementwiseOp, wafer::InstrGatherScatterOp>(
                operation))
          operations.push_back(operation);
      });
      for (mlir::Operation *operation : operations) {
        if (auto send = mlir::dyn_cast<wafer::InstrDTESendOp>(operation)) {
          mlir::Value replacement = recreateRingSubview(
              operation, send.getBuffer(), oldChunkElements, newChunkElements);
          ASSERT_TRUE(replacement);
          send->setOperand(0, replacement);
          send->setAttr("bytes", mlir::IntegerAttr::get(
                                     mlir::IntegerType::get(context.get(), 64),
                                     newBytes));
          continue;
        }
        if (auto recv = mlir::dyn_cast<wafer::InstrDTERecvOp>(operation)) {
          mlir::Value replacement = recreateRingSubview(
              operation, recv.getBuffer(), oldChunkElements, newChunkElements);
          ASSERT_TRUE(replacement);
          recv->setOperand(0, replacement);
          recv->setAttr("bytes", mlir::IntegerAttr::get(
                                     mlir::IntegerType::get(context.get(), 64),
                                     newBytes));
          continue;
        }
        if (auto elementwise =
                mlir::dyn_cast<wafer::InstrElementwiseOp>(operation)) {
          if (elementwise.getKind() != wafer::InstrElementwiseKind::Add)
            continue;
          for (unsigned index = 0; index < operation->getNumOperands();
               ++index) {
            mlir::Value replacement =
                recreateRingSubview(operation, operation->getOperand(index),
                                    oldChunkElements, newChunkElements);
            ASSERT_TRUE(replacement);
            operation->setOperand(index, replacement);
          }
          continue;
        }
        auto move = mlir::dyn_cast<wafer::InstrGatherScatterOp>(operation);
        if (!move || !move.getSource().getDefiningOp<mlir::memref::SubViewOp>())
          continue;
        mlir::Value source = recreateRingSubview(
            operation, move.getSource(), oldChunkElements, newChunkElements);
        mlir::Value dest = recreateRingSubview(
            operation, move.getDest(), oldChunkElements, newChunkElements);
        ASSERT_TRUE(source);
        ASSERT_TRUE(dest);
        move->setOperand(0, source);
        move->setOperand(1, dest);
        move->setAttr("byte_count",
                      mlir::IntegerAttr::get(
                          mlir::IntegerType::get(context.get(), 64), newBytes));
        move->setAttr("inner_bytes",
                      mlir::IntegerAttr::get(
                          mlir::IntegerType::get(context.get(), 64), newBytes));
      }
    }
  }

  wafer::InstrDTESendOp findRingSend(mlir::ModuleOp module, int64_t round,
                                     int64_t payloadSlice) {
    wafer::InstrDTESendOp result;
    module.walk([&](wafer::InstrDTESendOp send) {
      if (send.getMessage().getPhase() ==
              wafer::DTEProtocolPhase::AllReduceRing &&
          send.getMessage().getRound() == round &&
          send.getMessage().getPayloadSlice() == payloadSlice)
        result = send;
    });
    return result;
  }

  wafer::InstrDTERecvOp findMatchingRingRecv(OwnedModules &owners,
                                             int64_t sourceRank,
                                             wafer::InstrDTESendOp send) {
    if (!send || send.getPeer() < 0 ||
        send.getPeer() >= static_cast<int64_t>(owners.size()))
      return {};
    wafer::InstrDTERecvOp result;
    owners[static_cast<size_t>(send.getPeer())]->walk(
        [&](wafer::InstrDTERecvOp recv) {
          if (recv.getPeer() == sourceRank &&
              recv.getMessage() == send.getMessage() &&
              recv.getBytes() == send.getBytes())
            result = recv;
        });
    return result;
  }

  void expectRejectedAtomically(
      OwnedModules &owners,
      const wafer::frontend::FrontendProgramVerificationResult &program) {
    ASSERT_EQ(owners.size(), 2u);
    auto modules = views(owners);
    for (mlir::ModuleOp module : modules)
      ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
    std::vector<std::string> before;
    for (mlir::ModuleOp module : modules)
      before.push_back(snapshot(module));
    EXPECT_EQ(wafer::compiler::detail::materializeNoCPartialReductions(modules,
                                                                       program),
              0u);
    for (size_t rank = 0; rank < modules.size(); ++rank) {
      EXPECT_EQ(snapshot(modules[rank]), before[rank]);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(modules[rank])));
    }
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(NoCPartialDataflowTest,
       ElidesTreePartialSpillAndPreservesExplicitDTECompletion) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  auto modules = views(owners);
  unsigned sends = 0;
  unsigned recvs = 0;
  unsigned waits = 0;
  for (mlir::ModuleOp module : modules) {
    EXPECT_EQ(countRDMA(module), 1u);
    EXPECT_EQ(countWDMA(module), 2u);
    EXPECT_EQ(countPrivateDDRAllocations(module), 1u);
    sends += countSends(module);
    recvs += countRecvs(module);
    waits += countWaits(module);
  }

  EXPECT_EQ(wafer::compiler::detail::materializeNoCPartialReductions(
                modules, makeProgram()),
            2u);
  unsigned finalSends = 0;
  unsigned finalRecvs = 0;
  unsigned finalWaits = 0;
  for (mlir::ModuleOp module : modules) {
    EXPECT_EQ(countRDMA(module), 0u);
    EXPECT_EQ(countWDMA(module), 1u);
    EXPECT_EQ(countPrivateDDRAllocations(module), 0u);
    finalSends += countSends(module);
    finalRecvs += countRecvs(module);
    finalWaits += countWaits(module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
  }
  EXPECT_EQ(finalSends, sends);
  EXPECT_EQ(finalRecvs, recvs);
  EXPECT_EQ(finalWaits, waits);
  EXPECT_EQ(finalSends, 2u);
  EXPECT_EQ(finalRecvs, 2u);
  EXPECT_EQ(finalWaits, 4u);
}

TEST_F(NoCPartialDataflowTest, RejectsUnsupportedCollectiveCombiner) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrDTERecvOp reductionRecv;
  owners[0]->walk([&](wafer::InstrDTERecvOp recv) {
    if (recv.getMessage().getPhase() ==
        wafer::DTEProtocolPhase::AllReduceTreeReduce)
      reductionRecv = recv;
  });
  ASSERT_TRUE(reductionRecv);
  wafer::InstrElementwiseOp merge;
  owners[0]->walk([&](wafer::InstrElementwiseOp operation) {
    if (llvm::is_contained(operation.getInputs(), reductionRecv.getBuffer()))
      merge = operation;
  });
  ASSERT_TRUE(merge);
  merge.setKind(wafer::InstrElementwiseKind::Mul);
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsUnequalRankTile) {
  OwnedModules owners = makePair(/*rank0Elements=*/4, /*rank1Elements=*/2);
  ASSERT_EQ(owners.size(), 2u);
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsMissingRankWithoutMutation) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  auto allModules = views(owners);
  llvm::MutableArrayRef<mlir::ModuleOp> missingRank(allModules.data(), 1);
  std::string before = snapshot(allModules.front());
  EXPECT_EQ(wafer::compiler::detail::materializeNoCPartialReductions(
                missingRank, makeProgram()),
            0u);
  EXPECT_EQ(snapshot(allModules.front()), before);
}

TEST_F(NoCPartialDataflowTest, RejectsDeadProtocolReceive) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrDTERecvOp reductionRecv;
  owners[0]->walk([&](wafer::InstrDTERecvOp recv) {
    if (recv.getMessage().getPhase() ==
        wafer::DTEProtocolPhase::AllReduceTreeReduce)
      reductionRecv = recv;
  });
  ASSERT_TRUE(reductionRecv);
  wafer::InstrElementwiseOp merge;
  owners[0]->walk([&](wafer::InstrElementwiseOp operation) {
    if (llvm::is_contained(operation.getInputs(), reductionRecv.getBuffer()))
      merge = operation;
  });
  ASSERT_TRUE(merge);
  ASSERT_EQ(merge.getInputs().size(), 2u);
  mlir::Value local = merge.getInputs().front() == reductionRecv.getBuffer()
                          ? merge.getInputs().back()
                          : merge.getInputs().front();
  for (unsigned index = 0; index < merge.getInputs().size(); ++index)
    if (merge.getInputs()[index] == reductionRecv.getBuffer())
      merge->setOperand(index, local);
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsAllocationRootOnlySubviewProvenance) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrDTESendOp reductionSend;
  owners[1]->walk([&](wafer::InstrDTESendOp send) {
    if (send.getMessage().getPhase() ==
        wafer::DTEProtocolPhase::AllReduceTreeReduce)
      reductionSend = send;
  });
  ASSERT_TRUE(reductionSend);

  mlir::OpBuilder builder(reductionSend);
  auto bufferType =
      mlir::cast<mlir::MemRefType>(reductionSend.getBuffer().getType());
  auto backingType = mlir::MemRefType::get({8}, bufferType.getElementType(),
                                           mlir::MemRefLayoutAttrInterface{},
                                           bufferType.getMemorySpace());
  auto backing = builder.create<mlir::memref::AllocOp>(reductionSend.getLoc(),
                                                       backingType);
  llvm::SmallVector<mlir::OpFoldResult, 1> lowOffsets{builder.getIndexAttr(0)};
  llvm::SmallVector<mlir::OpFoldResult, 1> highOffsets{builder.getIndexAttr(4)};
  llvm::SmallVector<mlir::OpFoldResult, 1> sizes{builder.getIndexAttr(4)};
  llvm::SmallVector<mlir::OpFoldResult, 1> strides{builder.getIndexAttr(1)};
  auto low = builder.create<mlir::memref::SubViewOp>(
      reductionSend.getLoc(), backing, lowOffsets, sizes, strides);
  auto high = builder.create<mlir::memref::SubViewOp>(
      reductionSend.getLoc(), backing, highOffsets, sizes, strides);
  constexpr int64_t zeroStrides[] = {0, 0, 0};
  constexpr int64_t oneIterations[] = {1, 1, 1};
  builder.create<wafer::InstrGatherScatterOp>(
      reductionSend.getLoc(), reductionSend.getBuffer(), low.getResult(),
      /*byteCount=*/16, /*innerBytes=*/16, mlir::IntegerAttr{},
      mlir::IntegerAttr{}, zeroStrides, oneIterations, zeroStrides,
      oneIterations);
  reductionSend->setOperand(0, high.getResult());

  // The low subview contains the local partial while the send reads the
  // disjoint high subview. Allocation-root equality must not prove this send.
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsUnusedProtocolRound) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrDTESendOp reductionSend;
  owners[1]->walk([&](wafer::InstrDTESendOp send) {
    if (send.getMessage().getPhase() ==
        wafer::DTEProtocolPhase::AllReduceTreeReduce)
      reductionSend = send;
  });
  wafer::InstrDTERecvOp reductionRecv;
  owners[0]->walk([&](wafer::InstrDTERecvOp recv) {
    if (recv.getMessage().getPhase() ==
        wafer::DTEProtocolPhase::AllReduceTreeReduce)
      reductionRecv = recv;
  });
  ASSERT_TRUE(reductionSend);
  ASSERT_TRUE(reductionRecv);

  auto message =
      wafer::DTEMessageAttr::get(context.get(), /*communicationId=*/41,
                                 wafer::DTEProtocolPhase::AllReduceTreeReduce,
                                 /*round=*/2, /*payloadSlice=*/7);
  mlir::OpBuilder sendBuilder(reductionSend);
  mlir::Operation *sendWait = reductionSend.getToken().use_begin()->getOwner();
  sendBuilder.setInsertionPointAfter(sendWait);
  auto extraSend = sendBuilder.create<wafer::InstrDTESendOp>(
      reductionSend.getLoc(), sendBuilder.getType<mlir::async::TokenType>(),
      reductionSend.getBuffer(), mlir::Value(), reductionSend.getPeerAttr(),
      reductionSend.getBytesAttr(), message, wafer::DirectDTEBindingAttr());
  sendBuilder.create<wafer::InstrDTEWaitOp>(
      reductionSend.getLoc(), mlir::ValueRange{extraSend.getToken()});

  wafer::InstrElementwiseOp merge;
  owners[0]->walk([&](wafer::InstrElementwiseOp operation) {
    if (llvm::is_contained(operation.getInputs(), reductionRecv.getBuffer()))
      merge = operation;
  });
  ASSERT_TRUE(merge);
  mlir::OpBuilder recvBuilder(merge);
  recvBuilder.setInsertionPointAfter(merge);
  auto extraRecv = recvBuilder.create<wafer::InstrDTERecvOp>(
      reductionRecv.getLoc(), recvBuilder.getType<mlir::async::TokenType>(),
      reductionRecv.getBuffer(), mlir::Value(), reductionRecv.getPeerAttr(),
      reductionRecv.getBytesAttr(), message, wafer::DirectDTEBindingAttr());
  recvBuilder.create<wafer::InstrDTEWaitOp>(
      reductionRecv.getLoc(), mlir::ValueRange{extraRecv.getToken()});
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsDuplicateOriginMultiplicity) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrDTERecvOp reductionRecv;
  wafer::InstrDTESendOp broadcastSend;
  owners[0]->walk([&](wafer::InstrDTERecvOp recv) {
    if (recv.getMessage().getPhase() ==
        wafer::DTEProtocolPhase::AllReduceTreeReduce)
      reductionRecv = recv;
  });
  owners[0]->walk([&](wafer::InstrDTESendOp send) {
    if (send.getMessage().getPhase() ==
        wafer::DTEProtocolPhase::AllReduceTreeBroadcast)
      broadcastSend = send;
  });
  ASSERT_TRUE(reductionRecv);
  ASSERT_TRUE(broadcastSend);

  wafer::InstrElementwiseOp merge;
  owners[0]->walk([&](wafer::InstrElementwiseOp operation) {
    if (llvm::is_contained(operation.getInputs(), reductionRecv.getBuffer()))
      merge = operation;
  });
  wafer::InstrWDMAOp publisher = findPublisher(*owners[0]);
  ASSERT_TRUE(merge);
  ASSERT_TRUE(publisher);

  mlir::OpBuilder builder(merge);
  builder.setInsertionPointAfter(merge);
  auto duplicate = builder.create<mlir::memref::AllocOp>(
      merge.getLoc(), mlir::cast<mlir::MemRefType>(merge.getDest().getType()));
  llvm::SmallVector<mlir::Value, 2> duplicateInputs{merge.getDest(),
                                                    merge.getDest()};
  builder.create<wafer::InstrElementwiseOp>(
      merge.getLoc(), wafer::InstrElementwiseKind::Add, duplicateInputs,
      duplicate.getResult(), wafer::NCCWorker::Worker0);
  broadcastSend->setOperand(0, duplicate.getResult());
  publisher->setOperand(0, duplicate.getResult());

  // The first merge already contains every rank exactly once. Adding that
  // accumulator to itself keeps the same origin keys but doubles every count.
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsRankPublishingOnlyItsLocalPartial) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrRDMAOp load = findSpillLoad(*owners[1]);
  wafer::InstrWDMAOp publisher = findPublisher(*owners[1]);
  ASSERT_TRUE(load);
  ASSERT_TRUE(publisher);
  publisher->setOperand(0, load.getDest());
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsNonCompactOutputPublisher) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrWDMAOp publisher = findPublisher(*owners[1]);
  ASSERT_TRUE(publisher);
  publisher->setAttr(
      "inner_bytes",
      mlir::IntegerAttr::get(mlir::IntegerType::get(context.get(), 64), 8));
  publisher->setAttr("dst_strides",
                     mlir::DenseI64ArrayAttr::get(context.get(), {8, 0, 0}));
  publisher->setAttr("dst_iterations",
                     mlir::DenseI64ArrayAttr::get(context.get(), {2, 1, 1}));
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsOutputOverwriteAfterPublisher) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrWDMAOp publisher = findPublisher(*owners[1]);
  wafer::InstrWDMAOp spill = findSpillStore(*owners[1]);
  ASSERT_TRUE(publisher);
  ASSERT_TRUE(spill);
  mlir::OpBuilder builder(publisher);
  builder.setInsertionPointAfter(publisher);
  mlir::Operation *overwrite = builder.clone(*publisher.getOperation());
  overwrite->setOperand(0, spill.getSource());
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsProducerOverwriteAfterSnapshot) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrWDMAOp spill = findSpillStore(*owners[1]);
  wafer::InstrElementwiseOp producer = findProducer(*owners[1]);
  ASSERT_TRUE(spill);
  ASSERT_TRUE(producer);
  mlir::OpBuilder builder(spill);
  builder.setInsertionPointAfter(spill);
  builder.clone(*producer.getOperation());
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsEarlyProducerDeallocation) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrWDMAOp spill = findSpillStore(*owners[1]);
  ASSERT_TRUE(spill);
  mlir::OpBuilder builder(spill);
  builder.setInsertionPointAfter(spill);
  builder.create<mlir::memref::DeallocOp>(spill.getLoc(), spill.getSource());
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsConsumerUseBeforeReload) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrRDMAOp load = findSpillLoad(*owners[1]);
  ASSERT_TRUE(load);
  mlir::OpBuilder builder(load);
  builder.setInsertionPoint(load);
  llvm::SmallVector<mlir::Value, 2> inputs{load.getDest(), load.getDest()};
  builder.create<wafer::InstrElementwiseOp>(
      load.getLoc(), wafer::InstrElementwiseKind::Add, inputs, load.getDest(),
      wafer::NCCWorker::Worker0);
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsUnprovenConsumerAliasUse) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrRDMAOp load = findSpillLoad(*owners[1]);
  ASSERT_TRUE(load);
  mlir::OpBuilder builder(load);
  builder.setInsertionPointAfter(load);
  builder.create<mlir::memref::CastOp>(load.getLoc(), load.getDest().getType(),
                                       load.getDest());
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsConsumerUseInNestedBlock) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrRDMAOp load = findSpillLoad(*owners[1]);
  ASSERT_TRUE(load);
  mlir::OpBuilder builder(load);
  builder.setInsertionPointAfter(load);
  auto scratch = builder.create<mlir::memref::AllocOp>(
      load.getLoc(), mlir::cast<mlir::MemRefType>(load.getDest().getType()));
  auto nested = builder.create<mlir::scf::ExecuteRegionOp>(load.getLoc(),
                                                           mlir::TypeRange{});
  mlir::Block *body = builder.createBlock(&nested.getRegion());
  builder.setInsertionPointToStart(body);
  llvm::SmallVector<mlir::Value, 2> inputs{load.getDest(), load.getDest()};
  builder.create<wafer::InstrElementwiseOp>(
      load.getLoc(), wafer::InstrElementwiseKind::Add, inputs,
      scratch.getResult(), wafer::NCCWorker::Worker0);
  builder.create<mlir::scf::YieldOp>(load.getLoc());
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsLatePublisherProofFailureAtomically) {
  OwnedModules owners = makePair();
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrWDMAOp publisher = findPublisher(*owners[1]);
  ASSERT_TRUE(publisher);
  mlir::OpBuilder builder(publisher);
  builder.setInsertionPoint(publisher);
  auto scratch = builder.create<mlir::memref::AllocOp>(
      publisher.getLoc(),
      mlir::cast<mlir::MemRefType>(publisher.getDest().getType()));
  publisher->setOperand(1, scratch.getResult());
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest,
       ElidesRingPartialSpillWithSlicePreciseProvenance) {
  OwnedModules owners =
      makePair(4, 4, wafer::tile_region_to_instr::AllReduceSchedule::Ring);
  ASSERT_EQ(owners.size(), 2u);
  auto modules = views(owners);
  unsigned sends = 0;
  unsigned recvs = 0;
  for (mlir::ModuleOp module : modules) {
    EXPECT_EQ(countRDMA(module), 1u);
    EXPECT_EQ(countWDMA(module), 2u);
    sends += countSends(module);
    recvs += countRecvs(module);
  }

  EXPECT_EQ(wafer::compiler::detail::materializeNoCPartialReductions(
                modules, makeProgram()),
            2u);
  for (mlir::ModuleOp module : modules) {
    EXPECT_EQ(countRDMA(module), 0u);
    EXPECT_EQ(countWDMA(module), 1u);
    EXPECT_EQ(countPrivateDDRAllocations(module), 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
  }
  EXPECT_EQ(sends, 4u);
  EXPECT_EQ(recvs, 4u);
}

TEST_F(NoCPartialDataflowTest, RejectsRingTailWithoutMutation) {
  OwnedModules owners =
      makePair(4, 4, wafer::tile_region_to_instr::AllReduceSchedule::Ring);
  ASSERT_EQ(owners.size(), 2u);
  // Rewrite every protocol/dataflow slice from two f32 elements to one while
  // retaining the four-element publisher. Message attrs and actual subviews
  // remain mutually consistent, but the two slices cover only the first half
  // of the complete buffer.
  shrinkRingProtocolToTail(owners, /*oldChunkElements=*/2,
                           /*newChunkElements=*/1);
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsRingOverlappingActualSlice) {
  OwnedModules owners =
      makePair(4, 4, wafer::tile_region_to_instr::AllReduceSchedule::Ring);
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrDTESendOp send = findRingSend(*owners[0], /*round=*/0,
                                            /*payloadSlice=*/0);
  ASSERT_TRUE(send);
  mlir::Value duplicateSlice =
      recreateRingSubview(send, send.getBuffer(), /*oldChunkElements=*/2,
                          /*newChunkElements=*/2, /*forcedOffset=*/2);
  ASSERT_TRUE(duplicateSlice);
  send->setOperand(0, duplicateSlice);
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsRingDuplicatePayloadSlice) {
  OwnedModules owners =
      makePair(4, 4, wafer::tile_region_to_instr::AllReduceSchedule::Ring);
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrDTESendOp send = findRingSend(*owners[0], /*round=*/0,
                                            /*payloadSlice=*/0);
  wafer::InstrDTERecvOp recv =
      findMatchingRingRecv(owners, /*sourceRank=*/0, send);
  ASSERT_TRUE(send);
  ASSERT_TRUE(recv);
  auto duplicate = wafer::DTEMessageAttr::get(
      context.get(), send.getMessage().getCommunicationId(),
      wafer::DTEProtocolPhase::AllReduceRing, send.getMessage().getRound(),
      /*payloadSlice=*/1);
  send->setAttr("message", duplicate);
  recv->setAttr("message", duplicate);
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest, RejectsRingMisalignedActualSlice) {
  OwnedModules owners =
      makePair(4, 4, wafer::tile_region_to_instr::AllReduceSchedule::Ring);
  ASSERT_EQ(owners.size(), 2u);
  wafer::InstrDTESendOp send = findRingSend(*owners[0], /*round=*/0,
                                            /*payloadSlice=*/0);
  ASSERT_TRUE(send);
  // The two-element view starts one f32 into the buffer, so it straddles the
  // canonical [0, 8) and [8, 16) payload slices.
  mlir::Value straddlingSlice =
      recreateRingSubview(send, send.getBuffer(), /*oldChunkElements=*/2,
                          /*newChunkElements=*/2, /*forcedOffset=*/1);
  ASSERT_TRUE(straddlingSlice);
  send->setOperand(0, straddlingSlice);
  expectRejectedAtomically(owners, makeProgram());
}

TEST_F(NoCPartialDataflowTest,
       RingPartialPassesNoCResidentAllRankFinalization) {
  constexpr int64_t rankCount = 16;
  constexpr int64_t elements = 16;
  OwnedModules owners;
  for (int64_t rank = 0; rank < rankCount; ++rank) {
    auto module = parseRank(
        rank, elements, wafer::tile_region_to_instr::AllReduceSchedule::Ring,
        rankCount);
    ASSERT_TRUE(module);
    owners.push_back(std::move(module));
  }

  OwnedModules finalizedOwners;
  for (const auto &owner : owners)
    finalizedOwners.push_back(mlir::cast<mlir::ModuleOp>(owner.get()->clone()));
  auto finalizedModules = views(finalizedOwners);
  ASSERT_EQ(wafer::compiler::detail::materializeNoCPartialReductions(
                finalizedModules, makeProgram(elements, rankCount)),
            rankCount);
  const wafer::TargetMemoryPolicy memory =
      wafer::getDefaultWaferTargetPolicy(wafer::TileSearchEffort::Default)
          .memory;
  for (mlir::ModuleOp module : finalizedModules) {
    uint64_t terminalOperations = 0;
    EXPECT_EQ(wafer::detail::checkStaticTerminalOperationBudget(
                  module.getOperation(), terminalOperations),
              wafer::detail::StaticTerminalOperationBudgetStatus::WithinBudget);
    ASSERT_TRUE(mlir::succeeded(wafer::normalizeMinimumNCCJoins(module)));
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
    ASSERT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
        module, memory.spmBase, memory.spmLimit, memory.spmAlignment)));
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
  }
  ASSERT_TRUE(
      mlir::succeeded(wafer::compiler::detail::verifyDirectDTETransportSchedule(
          finalizedModules)));

  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(
      rankCount);
  for (size_t rank = 0; rank < frontiers.size(); ++rank)
    frontiers[rank].push_back(
        {std::move(owners[rank]), /*stableOrdinal=*/0,
         wafer::RankArtifactKind::Spill, /*reservedBaseline=*/true,
         wafer::RankBufferingKind::Single, /*bufferingPlanOrdinal=*/0});

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  ASSERT_TRUE(
      mlir::succeeded(wafer::compiler::detail::acceptWholeVariantResources(
          finalizedModules, *config)));
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, makeProgram(elements, rankCount), *config, &failure)))
      << failure;
  ASSERT_EQ(frontiers[0].size(), frontiers[1].size());
  ASSERT_GT(frontiers[0].size(), 1u);

  bool foundRingPartial = false;
  for (size_t candidate = 1; candidate < frontiers[0].size(); ++candidate) {
    unsigned rdma = 0;
    unsigned wdma = 0;
    unsigned sends = 0;
    unsigned recvs = 0;
    for (const auto &frontier : frontiers) {
      mlir::ModuleOp module = *frontier[candidate].module;
      rdma += countRDMA(module);
      wdma += countWDMA(module);
      sends += countSends(module);
      recvs += countRecvs(module);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
    }
    if (rdma == 0 && wdma == rankCount &&
        sends == static_cast<unsigned>(rankCount * 2 * (rankCount - 1)) &&
        recvs == static_cast<unsigned>(rankCount * 2 * (rankCount - 1)))
      foundRingPartial = true;
  }
  EXPECT_TRUE(foundRingPartial);
}

TEST_F(NoCPartialDataflowTest,
       PartialRunsWithoutValidInputBoundaryAndPassesResidentLateGates) {
  constexpr int64_t rankCount = 16;
  OwnedModules owners;
  for (int64_t rank = 0; rank < rankCount; ++rank) {
    auto module =
        parseRank(rank, 4, wafer::tile_region_to_instr::AllReduceSchedule::Tree,
                  rankCount);
    ASSERT_TRUE(module);
    owners.push_back(std::move(module));
  }
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(
      rankCount);
  for (size_t rank = 0; rank < frontiers.size(); ++rank)
    frontiers[rank].push_back(
        {std::move(owners[rank]), /*stableOrdinal=*/0,
         wafer::RankArtifactKind::Spill, /*reservedBaseline=*/true,
         wafer::RankBufferingKind::Single, /*bufferingPlanOrdinal=*/0});

  wafer::frontend::FrontendProgramVerificationResult program =
      makeProgram(4, rankCount);
  wafer::frontend::ProgramBoundaryBinding unrelatedInput;
  unrelatedInput.index = 0;
  unrelatedInput.programIndex = 0;
  unrelatedInput.distribution =
      wafer::frontend::ProgramDistributionKind::Replicated;
  unrelatedInput.globalShape = {4};
  unrelatedInput.localShape = {4};
  unrelatedInput.dtype = "f32";
  // Deliberately incomplete: this unrelated input role is not an admission
  // prerequisite for an already explicit partial collective.
  wafer::frontend::ProgramRankSlice onlyRank;
  onlyRank.logicalRank = 0;
  onlyRank.replicaId = 0;
  onlyRank.offsets = {0};
  onlyRank.sizes = {4};
  onlyRank.strides = {1};
  unrelatedInput.rankSlices.push_back(std::move(onlyRank));
  program.distributedInputs.push_back(std::move(unrelatedInput));

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program, *config, &failure)))
      << failure;
  ASSERT_EQ(frontiers[0].size(), frontiers[1].size());
  ASSERT_GT(frontiers[0].size(), 1u);

  bool foundResident = false;
  for (size_t candidate = 1; candidate < frontiers[0].size(); ++candidate) {
    unsigned rdma = 0;
    unsigned wdma = 0;
    unsigned sends = 0;
    unsigned recvs = 0;
    unsigned waits = 0;
    for (size_t rank = 0; rank < frontiers.size(); ++rank) {
      mlir::ModuleOp module = *frontiers[rank][candidate].module;
      rdma += countRDMA(module);
      wdma += countWDMA(module);
      sends += countSends(module);
      recvs += countRecvs(module);
      waits += countWaits(module);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
    }
    if (rdma == 0 && wdma == rankCount && sends == 2 * (rankCount - 1) &&
        recvs == 2 * (rankCount - 1) && waits == 4 * (rankCount - 1))
      foundResident = true;
  }
  EXPECT_TRUE(foundResident);
  for (const auto &frontier : frontiers) {
    EXPECT_EQ(countRDMA(*frontier.front().module), 1u);
    EXPECT_EQ(countWDMA(*frontier.front().module), 2u);
  }
}

} // namespace
