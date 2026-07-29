#include "../../lib/Wafer/Compiler/NoCResidentDataflow.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"
#include "../../lib/Wafer/Compiler/ScheduledRankFinalization.h"
#include "../../lib/Wafer/Compiler/StaticFixedSlotQualification.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/PhysicalDataflow.h"
#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"
#include "Wafer/Transforms/SoftwarePipelining.h"
#include "Wafer/Transforms/WorkerPlacement.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace {

using Candidate = wafer::compiler::detail::RankVariantCandidate;
using Frontier = wafer::compiler::detail::RankVariantFrontier;

constexpr int64_t kOperandDrivenInterfaceStableOrdinal = 16 * 12 * 4;
constexpr int64_t kPartialReductionInterfaceStableOrdinal =
    kOperandDrivenInterfaceStableOrdinal + 4;
constexpr int64_t kRefinedPartialReductionInterfaceStableOrdinal =
    kPartialReductionInterfaceStableOrdinal + 8;

struct ActualTraversalFrontiers {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers;
  std::vector<std::string> interfaceComputeIssues;
  unsigned interfaceLoopCount = 0;
  unsigned interfaceOutputWriteCount = 0;
};

static std::vector<std::string>
getComputeIssueFingerprint(mlir::ModuleOp module) {
  std::vector<std::string> result;
  module.walk([&](wafer::WaferInstructionOpInterface instruction) {
    // Boundary composition intentionally replaces input RDMA with peer DTE on
    // non-owner ranks. Partial spill-cut elimination can also remove
    // Movement-family gather/scatter copies that only staged the eliminated
    // DDR cut. Fingerprint the semantic traversal/compute and unchanged output
    // path, not those physical movement choices.
    if (mlir::isa<wafer::InstrRDMAOp, wafer::InstrDTESendOp,
                  wafer::InstrDTERecvOp, wafer::InstrDTEWaitOp,
                  wafer::InstrGatherScatterOp>(instruction.getOperation()) ||
        instruction.getInstructionFamily() == wafer::InstrFamily::DTE)
      return;
    result.push_back(instruction->getName().getStringRef().str());
  });
  llvm::sort(result);
  return result;
}

static unsigned countLoops(mlir::ModuleOp module) {
  unsigned result = 0;
  module.walk([&](mlir::scf::ForOp) { ++result; });
  return result;
}

static unsigned countOutputWrites(mlir::ModuleOp module) {
  unsigned result = 0;
  module.walk([&](wafer::InstrWDMAOp) { ++result; });
  return result;
}

static unsigned countPeerIssues(mlir::ModuleOp module) {
  unsigned result = 0;
  module.walk([&](mlir::Operation *operation) {
    result += mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTERecvOp>(operation)
                  ? 1u
                  : 0u;
  });
  return result;
}

static std::string
buildStaticFixedSlotWitnessSource(llvm::StringRef directLoopBody) {
  std::string source = R"mlir(
module {
  func.func @main(
      %input: memref<?xf16, #wafer.memory<ddr, tensor>>) {
    %slot0 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %slot1 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %fixed = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    // This overlapping root and the dynamic loop below are unrelated to the
    // admitted witness and must not turn an existential current-IR proof into
    // a universal closure inventory.
    %unrelated = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %zero = arith.constant 0.0 : f16
    %dynamic_upper = memref.dim %input, %c0
        : memref<?xf16, #wafer.memory<ddr, tensor>>
    %result:3 = scf.for %index = %c0 to %c3 step %c1
        iter_args(%current = %slot0, %next = %slot1,
                  %fixed_current = %fixed)
        -> (memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>) {
)mlir";
  source += directLoopBody.str();
  source += R"mlir(
      scf.yield %next, %current, %fixed
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
    }
    scf.for %unused = %c0 to %dynamic_upper step %c1 {
      scf.yield
    }
    return
  }
}
)mlir";
  return source;
}

enum class CompanionRecurrenceKind {
  FixedRoot,
  Unknown,
  ConflictingRootAliases,
};

static std::string
buildStaticFixedSlotCompanionSource(CompanionRecurrenceKind kind) {
  std::string source = R"mlir(
module {
  func.func @main() {
    %slot0 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %slot1 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %fixed = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %true = arith.constant true
    %zero = arith.constant 0.0 : f16
)mlir";
  if (kind == CompanionRecurrenceKind::ConflictingRootAliases) {
    source += R"mlir(
    %result:3 = scf.for %index = %c0 to %c3 step %c1
        iter_args(%current = %slot0, %next = %slot1,
                  %alias = %slot0)
        -> (memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>) {
      wafer.instr.fill %current, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      scf.yield %next, %current, %alias
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
    }
)mlir";
  } else {
    source += R"mlir(
    %result:3 = scf.for %index = %c0 to %c3 step %c1
        iter_args(%current = %slot0, %next = %slot1,
                  %fixed_current = %fixed)
        -> (memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>) {
      wafer.instr.fill %current, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
)mlir";
    if (kind == CompanionRecurrenceKind::Unknown) {
      source += R"mlir(
      %unknown = arith.select %true, %fixed_current, %fixed
          : memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield %next, %current, %unknown
)mlir";
    } else {
      source += R"mlir(
      scf.yield %next, %current, %fixed
)mlir";
    }
    source += R"mlir(
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
    }
)mlir";
  }
  source += R"mlir(
    return
  }
}
)mlir";
  return source;
}

struct PeerTransportCounts {
  unsigned loads = 0;
  unsigned sends = 0;
  unsigned recvs = 0;
  unsigned waits = 0;
};

static PeerTransportCounts countPeerTransport(mlir::ModuleOp module) {
  PeerTransportCounts counts;
  module.walk([&](mlir::Operation *operation) {
    counts.loads += mlir::isa<wafer::InstrRDMAOp>(operation) ? 1u : 0u;
    counts.sends += mlir::isa<wafer::InstrDTESendOp>(operation) ? 1u : 0u;
    counts.recvs += mlir::isa<wafer::InstrDTERecvOp>(operation) ? 1u : 0u;
    counts.waits += mlir::isa<wafer::InstrDTEWaitOp>(operation) ? 1u : 0u;
  });
  return counts;
}

static bool peerReceivesPrecedeSendsInEveryBlock(mlir::ModuleOp module) {
  llvm::SmallVector<wafer::InstrDTESendOp, 8> sends;
  llvm::SmallVector<wafer::InstrDTERecvOp, 8> receives;
  module.walk([&](wafer::InstrDTESendOp send) { sends.push_back(send); });
  module.walk(
      [&](wafer::InstrDTERecvOp receive) { receives.push_back(receive); });
  return llvm::none_of(sends, [&](wafer::InstrDTESendOp send) {
    return llvm::any_of(receives, [&](wafer::InstrDTERecvOp receive) {
      return send->getBlock() == receive->getBlock() &&
             send->isBeforeInBlock(receive);
    });
  });
}

static const wafer::compiler::detail::RankVariantCandidate *
findCorrespondingCandidate(
    const wafer::compiler::detail::RankVariantFrontier &frontier,
    const wafer::compiler::detail::RankVariantCandidate &key) {
  auto found = llvm::find_if(frontier, [&](const auto &candidate) {
    return candidate.stableOrdinal == key.stableOrdinal &&
           candidate.artifactKind == key.artifactKind &&
           candidate.bufferingKind == key.bufferingKind &&
           candidate.bufferingPlanOrdinal == key.bufferingPlanOrdinal &&
           candidate.workerPlacementKind == key.workerPlacementKind &&
           candidate.workerPlacementPlanOrdinal ==
               key.workerPlacementPlanOrdinal;
  });
  return found == frontier.end() ? nullptr : &*found;
}

static std::string
candidateSnapshot(const wafer::compiler::detail::RankVariantCandidate &slot) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << slot.stableOrdinal << ':' << static_cast<unsigned>(slot.artifactKind)
     << ':' << slot.reservedBaseline << ':'
     << static_cast<unsigned>(slot.bufferingKind) << ':'
     << slot.bufferingPlanOrdinal << ':'
     << static_cast<unsigned>(slot.workerPlacementKind) << ':'
     << slot.workerPlacementPlanOrdinal << ':';
  if (slot.module)
    slot.module.get()->print(os);
  else
    os << "<null>";
  return text;
}

class NoCResidentDataflowTest : public ::testing::Test {
protected:
  NoCResidentDataflowTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseRank() {
    return mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(
      %input: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %input : memref<4xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%source: memref<4xf32, #wafer.memory<ddr, tensor>>):
      %buffer = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %source to %buffer
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf32, #wafer.memory<ddr, tensor>>
         to memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %source
          : memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    wafer.instr.ncc_join [0]
    return
  }
})mlir",
        mlir::ParserConfig(context.get()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> deriveWorkerPlacedRank() {
    mlir::OwningOpRef<mlir::ModuleOp> module = parseRank();
    if (!module)
      return {};
    wafer::clearRankCandidatePhysicalFacts(*module);

    wafer::TileYieldOp yield;
    wafer::InstrRDMAOp inputLoad;
    module->walk([&](wafer::TileYieldOp candidate) { yield = candidate; });
    module->walk([&](wafer::InstrRDMAOp candidate) { inputLoad = candidate; });
    if (!yield || !inputLoad)
      return {};
    auto bufferType =
        mlir::dyn_cast<mlir::MemRefType>(inputLoad.getDest().getType());
    if (!bufferType)
      return {};
    mlir::OpBuilder builder(yield);
    auto zero = builder.create<mlir::arith::ConstantOp>(
        yield.getLoc(), builder.getF32FloatAttr(0.0));
    llvm::SmallVector<mlir::Value, 2> scratchBuffers;
    for (unsigned lane = 0; lane < 2; ++lane) {
      auto scratch =
          builder.create<mlir::memref::AllocOp>(yield.getLoc(), bufferType);
      scratchBuffers.push_back(scratch);
      builder.create<wafer::InstrFillOp>(yield.getLoc(), scratch, zero,
                                         wafer::FillDomainAttr{},
                                         wafer::NCCWorker::Worker0);
    }
    auto sink =
        builder.create<mlir::memref::AllocOp>(yield.getLoc(), bufferType);
    builder.create<wafer::InstrElementwiseOp>(
        yield.getLoc(), wafer::InstrElementwiseKind::Add, scratchBuffers, sink,
        wafer::NCCWorker::Worker0);

    std::string failure;
    mlir::FailureOr<wafer::NCCWorkerPlacementCandidate> placed =
        wafer::deriveDisjointNCCWorkerPlacementCandidate(*module, &failure);
    if (mlir::failed(placed)) {
      ADD_FAILURE() << failure;
      return {};
    }
    return std::move(placed->module);
  }

  bool addExistingPeerMessage(mlir::ModuleOp module, bool send, int64_t peer,
                              int64_t communicationId) {
    wafer::InstrRDMAOp boundaryLoad;
    module.walk([&](wafer::InstrRDMAOp candidate) {
      if (!boundaryLoad)
        boundaryLoad = candidate;
    });
    if (!boundaryLoad)
      return false;
    auto bufferType =
        mlir::dyn_cast<mlir::MemRefType>(boundaryLoad.getDest().getType());
    if (!bufferType)
      return false;

    mlir::OpBuilder builder(boundaryLoad);
    if (!send)
      builder.setInsertionPointAfter(boundaryLoad);
    auto buffer = builder.create<mlir::memref::AllocOp>(boundaryLoad.getLoc(),
                                                        bufferType);
    buffer->setAttr(wafer::kWaferSPMOffsetAttrName,
                    wafer::SPMOffsetAttr::get(module.getContext(), 65792));
    auto message = wafer::DTEMessageAttr::get(
        module.getContext(), communicationId,
        wafer::DTEProtocolPhase::PeerDataflow, /*round=*/9,
        /*payloadSlice=*/3);
    mlir::Value token;
    if (send) {
      token = builder
                  .create<wafer::InstrDTESendOp>(
                      boundaryLoad.getLoc(),
                      builder.getType<mlir::async::TokenType>(), buffer,
                      builder.getI64IntegerAttr(peer),
                      builder.getI64IntegerAttr(16), message,
                      wafer::DirectDTEBindingAttr())
                  .getToken();
    } else {
      token = builder
                  .create<wafer::InstrDTERecvOp>(
                      boundaryLoad.getLoc(),
                      builder.getType<mlir::async::TokenType>(), buffer,
                      builder.getI64IntegerAttr(peer),
                      builder.getI64IntegerAttr(16), message,
                      wafer::DirectDTEBindingAttr())
                  .getToken();
    }
    builder.create<wafer::InstrDTEWaitOp>(boundaryLoad.getLoc(),
                                          mlir::ValueRange{token});
    return mlir::succeeded(mlir::verify(module));
  }

  bool addPrecedingTileRegionPeerMessage(mlir::ModuleOp module,
                                         std::optional<bool> send, int64_t peer,
                                         int64_t communicationId) {
    wafer::InstrRDMAOp boundaryLoad;
    wafer::TileRegionOp tileRegion;
    module.walk([&](wafer::InstrRDMAOp candidate) {
      if (!boundaryLoad)
        boundaryLoad = candidate;
    });
    module.walk([&](wafer::TileRegionOp candidate) {
      if (!tileRegion)
        tileRegion = candidate;
    });
    if (!boundaryLoad || !tileRegion)
      return false;
    auto bufferType =
        mlir::dyn_cast<mlir::MemRefType>(boundaryLoad.getDest().getType());
    if (!bufferType)
      return false;

    mlir::OpBuilder builder(tileRegion);
    auto preparation = builder.create<wafer::TileRegionOp>(
        tileRegion.getLoc(), mlir::TypeRange{}, mlir::ValueRange{});
    mlir::Block *block = new mlir::Block();
    preparation.getBody().push_back(block);
    builder.setInsertionPointToStart(block);
    if (send) {
      auto buffer = builder.create<mlir::memref::AllocOp>(tileRegion.getLoc(),
                                                          bufferType);
      buffer->setAttr(wafer::kWaferSPMOffsetAttrName,
                      wafer::SPMOffsetAttr::get(module.getContext(), 66048));
      auto message = wafer::DTEMessageAttr::get(
          module.getContext(), communicationId,
          wafer::DTEProtocolPhase::PeerDataflow, /*round=*/10,
          /*payloadSlice=*/4);
      mlir::Value token;
      if (*send) {
        token = builder
                    .create<wafer::InstrDTESendOp>(
                        tileRegion.getLoc(),
                        builder.getType<mlir::async::TokenType>(), buffer,
                        builder.getI64IntegerAttr(peer),
                        builder.getI64IntegerAttr(16), message,
                        wafer::DirectDTEBindingAttr())
                    .getToken();
      } else {
        token = builder
                    .create<wafer::InstrDTERecvOp>(
                        tileRegion.getLoc(),
                        builder.getType<mlir::async::TokenType>(), buffer,
                        builder.getI64IntegerAttr(peer),
                        builder.getI64IntegerAttr(16), message,
                        wafer::DirectDTEBindingAttr())
                    .getToken();
      }
      builder.create<wafer::InstrDTEWaitOp>(tileRegion.getLoc(),
                                            mlir::ValueRange{token});
    }
    builder.create<wafer::TileYieldOp>(tileRegion.getLoc());
    return mlir::succeeded(mlir::verify(module));
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseHalfRank() {
    return mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %input : memref<4xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%source: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %buffer = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %source to %buffer
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.tile.yield %source
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    wafer.instr.ncc_join [0]
    return
  }
})mlir",
        mlir::ParserConfig(context.get()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseStaticTileRank() {
    return mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(
      %input: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %input : memref<4xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%source: memref<4xf32, #wafer.memory<ddr, tensor>>):
      %tile0 = memref.subview %source[0] [2] [1]
          : memref<4xf32, #wafer.memory<ddr, tensor>>
         to memref<2xf32, strided<[1], offset: 0>,
                   #wafer.memory<ddr, tensor>>
      %tile1 = memref.subview %source[2] [2] [1]
          : memref<4xf32, #wafer.memory<ddr, tensor>>
         to memref<2xf32, strided<[1], offset: 2>,
                   #wafer.memory<ddr, tensor>>
      %buffer0 = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<2xf32, #wafer.memory<spm, tensor>>
      %buffer1 = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<65792>}
          : memref<2xf32, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %tile0 to %buffer0
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<2xf32, strided<[1], offset: 0>,
                   #wafer.memory<ddr, tensor>>
         to memref<2xf32, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %tile1 to %buffer1
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<2xf32, strided<[1], offset: 2>,
                   #wafer.memory<ddr, tensor>>
         to memref<2xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %source
          : memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    wafer.instr.ncc_join [0]
    return
  }
})mlir",
        mlir::ParserConfig(context.get()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseLoopRank() {
    return mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(
      %input: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %input : memref<4xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%source: memref<4xf32, #wafer.memory<ddr, tensor>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %dest = memref.alloc()
          : memref<4xf32, #wafer.memory<ddr, tensor>>
      %loop_result = scf.for %iv = %c0 to %c4 step %c1
          iter_args(%carried = %dest)
          -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
        %loaded = memref.alloc()
            {wafer.spm.offset = #wafer.spm_offset<65536>}
            : memref<4xf32, #wafer.memory<spm, tensor>>
        %computed = memref.alloc()
            {wafer.spm.offset = #wafer.spm_offset<65792>}
            : memref<4xf32, #wafer.memory<spm, tensor>>
        wafer.instr.rdma %source to %loaded
            {byte_count = 16 : i64, inner_bytes = 16 : i64,
             src_strides = array<i64: 0, 0, 0>,
             src_iterations = array<i64: 1, 1, 1>}
            : memref<4xf32, #wafer.memory<ddr, tensor>>
           to memref<4xf32, #wafer.memory<spm, tensor>>
        wafer.instr.elementwise <add> %loaded, %loaded into %computed
            : memref<4xf32, #wafer.memory<spm, tensor>>,
              memref<4xf32, #wafer.memory<spm, tensor>>
          into memref<4xf32, #wafer.memory<spm, tensor>>
        wafer.instr.wdma %computed to %carried
            {byte_count = 16 : i64, inner_bytes = 16 : i64,
             dst_strides = array<i64: 0, 0, 0>,
             dst_iterations = array<i64: 1, 1, 1>}
            : memref<4xf32, #wafer.memory<spm, tensor>>
           to memref<4xf32, #wafer.memory<ddr, tensor>>
        scf.yield %carried
            : memref<4xf32, #wafer.memory<ddr, tensor>>
      }
      wafer.tile.yield %loop_result
          : memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    wafer.instr.ncc_join [0]
    return
  }
})mlir",
        mlir::ParserConfig(context.get()));
  }

  wafer::frontend::FrontendProgramVerificationResult program() {
    wafer::frontend::FrontendProgramVerificationResult result;
    result.logicalRankCount = 16;
    wafer::frontend::ProgramBoundaryBinding input;
    input.index = 0;
    input.programIndex = 0;
    input.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
    input.globalShape = {4};
    input.localShape = {4};
    input.dtype = "f32";
    for (int64_t rank = 0; rank < 16; ++rank) {
      wafer::frontend::ProgramRankSlice slice;
      slice.logicalRank = rank;
      slice.replicaId = rank;
      slice.offsets = {0};
      slice.sizes = {4};
      slice.strides = {1};
      input.rankSlices.push_back(std::move(slice));
    }
    result.distributedInputs.push_back(std::move(input));
    return result;
  }

  wafer::frontend::FrontendProgramVerificationResult partitionedProgram() {
    wafer::frontend::FrontendProgramVerificationResult result;
    result.logicalRankCount = 16;
    wafer::frontend::ProgramBoundaryBinding input;
    input.index = 0;
    input.programIndex = 0;
    input.distribution = wafer::frontend::ProgramDistributionKind::Partitioned;
    input.globalShape = {64};
    input.localShape = {4};
    input.dtype = "f32";
    for (int64_t rank = 0; rank < 16; ++rank) {
      wafer::frontend::ProgramRankSlice slice;
      slice.logicalRank = rank;
      slice.replicaId = 0;
      slice.offsets = {rank * 4};
      slice.sizes = {4};
      slice.strides = {1};
      input.rankSlices.push_back(std::move(slice));
    }
    result.distributedInputs.push_back(std::move(input));
    return result;
  }

  wafer::frontend::FrontendProgramVerificationResult parameterProgram() {
    wafer::frontend::FrontendProgramVerificationResult result;
    result.logicalRankCount = 16;
    result.programParameterCount = 1;
    result.programParameterShardCount = 1;
    wafer::frontend::ProgramParameterBinding parameter;
    parameter.argumentIndex = 0;
    parameter.name = "weight";
    parameter.distribution =
        wafer::frontend::ProgramDistributionKind::Replicated;
    parameter.globalShape = {4};
    parameter.localShape = {4};
    parameter.dtype = "f32";
    for (int64_t rank = 0; rank < 16; ++rank) {
      wafer::frontend::ProgramRankSlice slice;
      slice.logicalRank = rank;
      slice.replicaId = rank;
      slice.offsets = {0};
      slice.sizes = {4};
      slice.strides = {1};
      parameter.rankSlices.push_back(std::move(slice));
    }
    result.parameters.push_back(std::move(parameter));
    return result;
  }

  wafer::frontend::FrontendProgramVerificationResult
  replicatedProgram(llvm::ArrayRef<int64_t> inputShape,
                    llvm::ArrayRef<int64_t> outputShape) {
    auto makeBoundary = [&](int64_t index, llvm::ArrayRef<int64_t> shape)
        -> wafer::frontend::ProgramBoundaryBinding {
      wafer::frontend::ProgramBoundaryBinding binding;
      binding.index = index;
      binding.programIndex = index;
      binding.distribution =
          wafer::frontend::ProgramDistributionKind::Replicated;
      binding.globalShape.assign(shape.begin(), shape.end());
      binding.localShape.assign(shape.begin(), shape.end());
      binding.dtype = "f32";
      for (int64_t rank = 0; rank < 16; ++rank) {
        wafer::frontend::ProgramRankSlice slice;
        slice.logicalRank = rank;
        slice.replicaId = rank;
        slice.offsets.assign(shape.size(), 0);
        slice.sizes.assign(shape.begin(), shape.end());
        slice.strides.assign(shape.size(), 1);
        binding.rankSlices.push_back(std::move(slice));
      }
      return binding;
    };

    wafer::frontend::FrontendProgramVerificationResult result;
    result.logicalRankCount = 16;
    result.programUserInputCount = 1;
    result.distributedInputs.push_back(makeBoundary(/*index=*/0, inputShape));
    result.distributedOutputs.push_back(makeBoundary(/*index=*/0, outputShape));
    return result;
  }

  std::optional<ActualTraversalFrontiers> buildActualTraversalFrontiers(
      llvm::StringRef sourceText, int64_t interfaceStableOrdinal,
      bool duplicateBaselineLoad = false,
      wafer::TargetProfileId targetProfile =
          wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      bool requireWorkerPlacedInterface = false) {
    mlir::OwningOpRef<mlir::ModuleOp> source =
        mlir::parseSourceString<mlir::ModuleOp>(
            sourceText, mlir::ParserConfig(context.get()));
    if (!source)
      return std::nullopt;

    wafer::TensorProgramSchedulingConfig schedulingConfig;
    schedulingConfig.logicalRank = 0;
    schedulingConfig.candidateParallelism = 1;
    schedulingConfig.targetProfile = targetProfile;
    mlir::FailureOr<std::vector<wafer::ScheduledRankCandidate>> scheduled =
        wafer::buildScheduledRankCandidateFrontier(*source, schedulingConfig);
    if (mlir::failed(scheduled))
      return std::nullopt;
    mlir::FailureOr<
        std::vector<wafer::compiler::detail::FinalizedRankCandidate>>
        finalized =
            wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
                std::move(*scheduled), targetProfile);
    if (mlir::failed(finalized))
      return std::nullopt;

    const wafer::compiler::detail::FinalizedRankCandidate *baseline = nullptr;
    const wafer::compiler::detail::FinalizedRankCandidate *interface = nullptr;
    auto isUnplacedSingle =
        [](const wafer::compiler::detail::FinalizedRankCandidate &candidate) {
          return candidate.module &&
                 candidate.bufferingKind == wafer::RankBufferingKind::Single &&
                 candidate.bufferingPlanOrdinal == 0 &&
                 candidate.workerPlacementKind ==
                     wafer::RankWorkerPlacementKind::Unplaced &&
                 candidate.workerPlacementPlanOrdinal == 0;
        };
    auto isInterfaceIdentity =
        [&](const wafer::compiler::detail::FinalizedRankCandidate &candidate) {
          if (!candidate.module ||
              candidate.bufferingKind != wafer::RankBufferingKind::Single ||
              candidate.bufferingPlanOrdinal != 0)
            return false;
          if (!requireWorkerPlacedInterface)
            return candidate.workerPlacementKind ==
                       wafer::RankWorkerPlacementKind::Unplaced &&
                   candidate.workerPlacementPlanOrdinal == 0;
          return candidate.workerPlacementKind ==
                     wafer::RankWorkerPlacementKind::DisjointComponents &&
                 candidate.workerPlacementPlanOrdinal != 0;
        };
    for (const wafer::compiler::detail::FinalizedRankCandidate &candidate :
         *finalized) {
      if (isUnplacedSingle(candidate) && candidate.reservedBaseline &&
          !baseline)
        baseline = &candidate;
      if (isInterfaceIdentity(candidate) && !candidate.reservedBaseline &&
          candidate.stableOrdinal == interfaceStableOrdinal && !interface)
        interface = &candidate;
    }
    if (!baseline || !interface)
      return std::nullopt;

    unsigned interfaceLoads = 0;
    interface->module.get()->walk(
        [&](wafer::InstrRDMAOp) { ++interfaceLoads; });
    if (interfaceLoads == 0)
      return std::nullopt;
    if (requireWorkerPlacedInterface) {
      bool hasNonzeroWorker = false;
      interface->module.get()->walk([&](mlir::Operation *operation) {
        std::optional<wafer::NCCWorker> worker =
            wafer::getNCCIssueWorker(operation);
        hasNonzeroWorker |= worker && *worker != wafer::NCCWorker::Worker0;
      });
      if (!hasNonzeroWorker)
        return std::nullopt;
    }

    ActualTraversalFrontiers result;
    result.interfaceComputeIssues =
        getComputeIssueFingerprint(interface->module.get());
    result.interfaceLoopCount = countLoops(interface->module.get());
    result.interfaceOutputWriteCount =
        countOutputWrites(interface->module.get());
    result.frontiers.resize(16);
    for (wafer::compiler::detail::RankVariantFrontier &frontier :
         result.frontiers) {
      mlir::OwningOpRef<mlir::ModuleOp> baselineClone =
          mlir::cast<mlir::ModuleOp>(baseline->module.get()->clone());
      if (duplicateBaselineLoad) {
        wafer::InstrRDMAOp firstLoad;
        baselineClone->walk([&](wafer::InstrRDMAOp load) {
          if (!firstLoad)
            firstLoad = load;
        });
        if (!firstLoad)
          return std::nullopt;
        mlir::OpBuilder builder(firstLoad);
        builder.setInsertionPointAfter(firstLoad);
        builder.clone(*firstLoad.getOperation());
      }
      frontier.push_back({std::move(baselineClone), baseline->stableOrdinal,
                          baseline->artifactKind, baseline->reservedBaseline,
                          baseline->bufferingKind,
                          baseline->bufferingPlanOrdinal,
                          baseline->workerPlacementKind,
                          baseline->workerPlacementPlanOrdinal});

      frontier.push_back(
          {mlir::cast<mlir::ModuleOp>(interface->module.get()->clone()),
           interface->stableOrdinal, interface->artifactKind,
           interface->reservedBaseline, interface->bufferingKind,
           interface->bufferingPlanOrdinal, interface->workerPlacementKind,
           interface->workerPlacementPlanOrdinal});
    }
    return result;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(NoCResidentDataflowTest,
       FixedSlotWitnessRequiresDirectLoopEffectConsumption) {
  auto verifyWitness = [&](llvm::StringRef directLoopBody) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(
            buildStaticFixedSlotWitnessSource(directLoopBody),
            mlir::ParserConfig(context.get()));
    if (!module) {
      ADD_FAILURE() << "failed to parse fixed-slot witness fixture";
      return std::string("parse failure");
    }
    wafer::compiler::RankExecutable rank =
        wafer::compiler::ExecutableBundleBuilder::makeRank(
            /*logicalRank=*/0, std::move(module), "main",
            /*programBindings=*/{}, wafer::compiler::TransportContract::None);
    llvm::Error error =
        wafer::compiler::detail::verifyStaticFixedSlotQualificationEvidence(
            rank);
    return error ? llvm::toString(std::move(error)) : std::string();
  };

  EXPECT_TRUE(verifyWitness(R"mlir(
      wafer.instr.fill %current, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
)mlir")
                  .empty());

  EXPECT_TRUE(verifyWitness(R"mlir(
      %view = memref.subview %current[0] [2] [1]
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<2xf16, strided<[1]>,
                   #wafer.memory<spm, tensor>>
      wafer.instr.fill %view, %zero
          : memref<2xf16, strided<[1]>,
                   #wafer.memory<spm, tensor>>, f16
)mlir")
                  .empty());

  std::string emptyRotation = verifyWitness("");
  EXPECT_NE(emptyRotation.find("effect/issue consumption"), std::string::npos)
      << emptyRotation;

  std::string unknownAlias = verifyWitness(R"mlir(
      %condition = arith.constant true
      %selected = arith.select %condition, %current, %next
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %selected, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
)mlir");
  EXPECT_NE(unknownAlias.find("effect/issue consumption"), std::string::npos)
      << unknownAlias;

  std::string nestedDeadEffect = verifyWitness(R"mlir(
      scf.for %nested = %c0 to %c0 step %c1 {
        wafer.instr.fill %current, %zero
            : memref<4xf16, #wafer.memory<spm, tensor>>, f16
        scf.yield
      }
)mlir");
  EXPECT_NE(nestedDeadEffect.find("effect/issue consumption"),
            std::string::npos)
      << nestedDeadEffect;
}

TEST_F(NoCResidentDataflowTest,
       FixedSlotCompanionAuditsFixedButRejectsUnknownOrConflictingRecurrences) {
  auto verifyCompanion = [&](CompanionRecurrenceKind kind) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(
            buildStaticFixedSlotCompanionSource(kind),
            mlir::ParserConfig(context.get()));
    if (!module) {
      ADD_FAILURE() << "failed to parse fixed-slot companion fixture";
      return std::string("parse failure");
    }
    wafer::compiler::RankExecutable rank =
        wafer::compiler::ExecutableBundleBuilder::makeRank(
            /*logicalRank=*/0, std::move(module), "main",
            /*programBindings=*/{}, wafer::compiler::TransportContract::None);
    llvm::Error error =
        wafer::compiler::detail::verifyStaticFixedSlotCompanionEvidence(rank);
    return error ? llvm::toString(std::move(error)) : std::string();
  };

  EXPECT_TRUE(verifyCompanion(CompanionRecurrenceKind::FixedRoot).empty());

  std::string unknown = verifyCompanion(CompanionRecurrenceKind::Unknown);
  EXPECT_NE(unknown.find("cannot prove SPM iter-arg recurrence"),
            std::string::npos)
      << unknown;

  std::string conflicting =
      verifyCompanion(CompanionRecurrenceKind::ConflictingRootAliases);
  EXPECT_NE(conflicting.find("conflicting recurrence successors"),
            std::string::npos)
      << conflicting;
}

TEST_F(NoCResidentDataflowTest,
       AtomicallyBuildsOwnerLoadAndExplicitPeerFanout) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers) {
    auto module = parseRank();
    ASSERT_TRUE(module);
    frontier.push_back(
        {std::move(module), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  }
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)))
      << failure;
  for (const auto &frontier : frontiers)
    ASSERT_EQ(frontier.size(), 3u);

  const Candidate *directKey = nullptr;
  const Candidate *forwardKey = nullptr;
  for (const Candidate &candidate : llvm::drop_begin(frontiers.front())) {
    unsigned maximumRankSends = 0;
    for (const auto &frontier : frontiers) {
      const Candidate *rankCandidate =
          findCorrespondingCandidate(frontier, candidate);
      ASSERT_NE(rankCandidate, nullptr);
      maximumRankSends = std::max(
          maximumRankSends, countPeerTransport(*rankCandidate->module).sends);
    }
    if (maximumRankSends > 1)
      directKey = &candidate;
    else
      forwardKey = &candidate;
  }
  ASSERT_NE(directKey, nullptr);
  ASSERT_NE(forwardKey, nullptr);

  std::optional<int64_t> directOwner;
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    const Candidate *candidate =
        findCorrespondingCandidate(frontiers[rank], *directKey);
    ASSERT_NE(candidate, nullptr);
    PeerTransportCounts counts = countPeerTransport(*candidate->module);
    if (counts.loads == 1) {
      ASSERT_FALSE(directOwner);
      directOwner = static_cast<int64_t>(rank);
      EXPECT_EQ(counts.sends, 15u);
      EXPECT_EQ(counts.recvs, 0u);
      EXPECT_EQ(counts.waits, 15u);
    } else {
      EXPECT_EQ(counts.loads, 0u);
      EXPECT_EQ(counts.sends, 0u);
      EXPECT_EQ(counts.recvs, 1u);
      EXPECT_EQ(counts.waits, 1u);
    }
    candidate->module.get()->walk([&](mlir::Operation *operation) {
      if (auto send = mlir::dyn_cast<wafer::InstrDTESendOp>(operation))
        EXPECT_EQ(send.getMessage().getPhase(),
                  wafer::DTEProtocolPhase::PeerDataflow);
      if (auto recv = mlir::dyn_cast<wafer::InstrDTERecvOp>(operation))
        EXPECT_EQ(recv.getMessage().getPhase(),
                  wafer::DTEProtocolPhase::PeerDataflow);
    });
  }
  ASSERT_TRUE(directOwner);
  EXPECT_EQ(*directOwner, 0);
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    if (static_cast<int64_t>(rank) == *directOwner)
      continue;
    const Candidate *candidate =
        findCorrespondingCandidate(frontiers[rank], *directKey);
    candidate->module.get()->walk([&](wafer::InstrDTERecvOp recv) {
      EXPECT_EQ(recv.getPeerAttr().getInt(), *directOwner);
    });
  }

  // The second sibling is the generic receive-then-forward alternative. The
  // owner injects once, every intermediate rank waits for its exact receive
  // before forwarding, and the final rank only receives.
  unsigned forwardOwners = 0;
  unsigned forwardIntermediates = 0;
  unsigned forwardTerminals = 0;
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    const Candidate *candidate =
        findCorrespondingCandidate(frontiers[rank], *forwardKey);
    ASSERT_NE(candidate, nullptr);
    PeerTransportCounts counts = countPeerTransport(*candidate->module);
    if (counts.loads == 1) {
      ++forwardOwners;
      EXPECT_EQ(counts.sends, 1u);
      EXPECT_EQ(counts.recvs, 0u);
      EXPECT_EQ(counts.waits, 1u);
    } else if (counts.sends == 0) {
      ++forwardTerminals;
      EXPECT_EQ(counts.recvs, 1u);
      EXPECT_EQ(counts.waits, 1u);
    } else {
      ++forwardIntermediates;
      EXPECT_EQ(counts.recvs, 1u);
      EXPECT_EQ(counts.sends, 1u);
      EXPECT_EQ(counts.waits, 2u);
    }
  }
  EXPECT_EQ(forwardOwners, 1u);
  EXPECT_EQ(forwardTerminals, 1u);
  EXPECT_EQ(forwardIntermediates, 14u);

  // The reserved baseline is immutable and still performs one load per rank.
  for (auto &frontier : frontiers) {
    unsigned loads = 0;
    frontier.front().module.get()->walk([&](wafer::InstrRDMAOp) { ++loads; });
    EXPECT_EQ(loads, 1u);
  }

  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::WholeVariantSelectionStatistics statistics;
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program(), *config, diagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::Production,
      &statistics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  EXPECT_TRUE(llvm::all_of(accepted->selectedReservedBaselines,
                           [](bool reserved) { return reserved; }));
  unsigned acceptedSends = 0;
  unsigned acceptedLoads = 0;
  for (const auto &rank : accepted->ranks)
    rank.getModule().walk([&](mlir::Operation *operation) {
      acceptedSends += mlir::isa<wafer::InstrDTESendOp>(operation);
      acceptedLoads += mlir::isa<wafer::InstrRDMAOp>(operation);
    });
  EXPECT_EQ(acceptedSends, 0u) << diagnosticText;
  EXPECT_EQ(acceptedLoads, 16u) << diagnosticText;
  EXPECT_GT(statistics.noCProfitabilityEvaluations, 0u);
  EXPECT_EQ(statistics.noCProfitabilityEvaluations,
            statistics.noCProfitabilityRejected +
                statistics.noCProfitabilityIndeterminate +
                statistics.noCProfitabilityEstimated +
                statistics.noCProfitabilityProven);
  EXPECT_GT(statistics.noCProfitabilityRejected, 0u);
  EXPECT_EQ(statistics.noCProfitabilityIndeterminate, 0u);
  EXPECT_EQ(statistics.noCProfitabilityEstimated, 0u);
  EXPECT_EQ(statistics.noCProfitabilityProven, 0u);
}

TEST_F(NoCResidentDataflowTest,
       ReplicatedParameterBuildsOwnerLoadAndExplicitPeerFanout) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers)
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  wafer::frontend::FrontendProgramVerificationResult parameter =
      parameterProgram();
  ASSERT_TRUE(parameter.distributedInputs.empty());
  ASSERT_EQ(parameter.parameters.size(), 1u);
  ASSERT_EQ(parameter.parameters.front().argumentIndex, 0);
  ASSERT_EQ(parameter.parameters.front().rankSlices.size(), 16u);
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, parameter, *config, &failure)))
      << failure;
  for (const auto &frontier : frontiers)
    ASSERT_GE(frontier.size(), 2u);

  const Candidate *directKey = nullptr;
  for (const Candidate &candidate : llvm::drop_begin(frontiers.front())) {
    unsigned maximumRankSends = 0;
    for (const Frontier &frontier : frontiers) {
      const Candidate *rankCandidate =
          findCorrespondingCandidate(frontier, candidate);
      ASSERT_NE(rankCandidate, nullptr);
      maximumRankSends = std::max(
          maximumRankSends, countPeerTransport(*rankCandidate->module).sends);
    }
    if (maximumRankSends > 1) {
      directKey = &candidate;
      break;
    }
  }
  ASSERT_NE(directKey, nullptr);

  unsigned aggregateLoads = 0;
  unsigned aggregateSends = 0;
  unsigned aggregateRecvs = 0;
  unsigned ownerRanks = 0;
  for (const Frontier &frontier : frontiers) {
    const Candidate *candidate =
        findCorrespondingCandidate(frontier, *directKey);
    ASSERT_NE(candidate, nullptr);
    PeerTransportCounts counts = countPeerTransport(*candidate->module);
    aggregateLoads += counts.loads;
    aggregateSends += counts.sends;
    aggregateRecvs += counts.recvs;
    ownerRanks += counts.loads == 1 ? 1u : 0u;
    EXPECT_LE(counts.loads, 1u);
    if (counts.loads == 1) {
      EXPECT_EQ(counts.sends, 15u);
      EXPECT_EQ(counts.recvs, 0u);
      EXPECT_EQ(counts.waits, 15u);
    } else {
      EXPECT_EQ(counts.sends, 0u);
      EXPECT_EQ(counts.recvs, 1u);
      EXPECT_EQ(counts.waits, 1u);
    }
  }
  EXPECT_EQ(aggregateLoads, 1u);
  EXPECT_EQ(ownerRanks, 1u);
  EXPECT_EQ(aggregateSends, 15u);
  EXPECT_EQ(aggregateRecvs, 15u);

  unsigned baselineLoads = 0;
  for (const Frontier &frontier : frontiers)
    frontier.front().module.get()->walk(
        [&](wafer::InstrRDMAOp) { ++baselineLoads; });
  EXPECT_EQ(baselineLoads, 16u);
}

TEST_F(NoCResidentDataflowTest,
       KeepsBaselineOnParameterArgumentRelationMismatch) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  std::vector<std::string> baselineSnapshots;
  for (auto &frontier : frontiers) {
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
    baselineSnapshots.push_back(candidateSnapshot(frontier.front()));
  }
  wafer::frontend::FrontendProgramVerificationResult invalid =
      parameterProgram();
  ASSERT_TRUE(invalid.distributedInputs.empty());
  ASSERT_EQ(invalid.parameters.size(), 1u);
  invalid.parameters.front().argumentIndex = 1;
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, invalid, *config, &failure)))
      << failure;
  for (auto [rank, frontier] : llvm::enumerate(frontiers)) {
    ASSERT_EQ(frontier.size(), 1u);
    EXPECT_EQ(candidateSnapshot(frontier.front()), baselineSnapshots[rank]);
  }
}

TEST_F(NoCResidentDataflowTest, KeepsBaselineOnReplicatedParameterCoverageGap) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  std::vector<std::string> baselineSnapshots;
  for (auto &frontier : frontiers) {
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
    baselineSnapshots.push_back(candidateSnapshot(frontier.front()));
  }
  wafer::frontend::FrontendProgramVerificationResult invalid =
      parameterProgram();
  ASSERT_TRUE(invalid.distributedInputs.empty());
  ASSERT_EQ(invalid.parameters.size(), 1u);
  invalid.parameters.front().rankSlices.pop_back();
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, invalid, *config, &failure)))
      << failure;
  for (auto [rank, frontier] : llvm::enumerate(frontiers)) {
    ASSERT_EQ(frontier.size(), 1u);
    EXPECT_EQ(candidateSnapshot(frontier.front()), baselineSnapshots[rank]);
  }
}

TEST_F(NoCResidentDataflowTest, KeepsBaselineOnNonEquivalentRankSlices) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers)
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  wafer::frontend::FrontendProgramVerificationResult invalid = program();
  invalid.distributedInputs.front().rankSlices.back().offsets = {1};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, invalid, *config, &failure)));
  for (const auto &frontier : frontiers)
    EXPECT_EQ(frontier.size(), 1u);
}

TEST_F(NoCResidentDataflowTest, PartitionedBoundaryKeepsOneLoadPerUniqueShard) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers)
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, partitionedProgram(), *config, &failure)))
      << failure;
  unsigned aggregateLoads = 0;
  unsigned aggregatePeerIssues = 0;
  for (const auto &frontier : frontiers) {
    ASSERT_EQ(frontier.size(), 1u);
    frontier.front().module.get()->walk(
        [&](wafer::InstrRDMAOp) { ++aggregateLoads; });
    frontier.front().module.get()->walk([&](mlir::Operation *operation) {
      aggregatePeerIssues +=
          mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTERecvOp>(operation)
              ? 1u
              : 0u;
    });
  }
  EXPECT_EQ(aggregateLoads, 16u);
  EXPECT_EQ(aggregatePeerIssues, 0u);
}

TEST_F(NoCResidentDataflowTest, KeepsBaselineOnReplicatedCoverageGap) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers)
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  wafer::frontend::FrontendProgramVerificationResult invalid = program();
  invalid.distributedInputs.front().globalShape = {8};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, invalid, *config, &failure)));
  for (const auto &frontier : frontiers)
    EXPECT_EQ(frontier.size(), 1u);
}

TEST_F(NoCResidentDataflowTest, KeepsBaselineOnPartitionedSliceOverlap) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers)
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  wafer::frontend::FrontendProgramVerificationResult invalid =
      partitionedProgram();
  for (wafer::frontend::ProgramRankSlice &slice :
       invalid.distributedInputs.front().rankSlices)
    slice.offsets = {0};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, invalid, *config, &failure)));
  for (const auto &frontier : frontiers)
    EXPECT_EQ(frontier.size(), 1u);
}

TEST_F(NoCResidentDataflowTest, KeepsBaselineOnMismatchedPhysicalPayload) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (size_t rank = 0; rank < frontiers.size(); ++rank)
    frontiers[rank].push_back(
        {rank == 7 ? parseHalfRank() : parseRank(), /*stableOrdinal=*/0,
         wafer::RankArtifactKind::Spill, /*reservedBaseline=*/true,
         wafer::RankBufferingKind::Single, /*bufferingPlanOrdinal=*/0});
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)));
  for (const auto &frontier : frontiers)
    EXPECT_EQ(frontier.size(), 1u);
}

TEST_F(NoCResidentDataflowTest,
       KeepsRepeatedBoundaryDescriptorOutsideFanoutGroup) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers) {
    mlir::OwningOpRef<mlir::ModuleOp> module = parseRank();
    ASSERT_TRUE(module);
    module->walk([&](wafer::InstrRDMAOp load) {
      mlir::Builder builder(load.getContext());
      load.setInnerBytesAttr(builder.getI64IntegerAttr(8));
      load.setSrcStridesAttr(builder.getDenseI64ArrayAttr({8, 0, 0}));
      load.setSrcIterationsAttr(builder.getDenseI64ArrayAttr({2, 1, 1}));
    });
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    frontier.push_back(
        {std::move(module), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  }
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)))
      << failure;
  for (const auto &frontier : frontiers)
    EXPECT_EQ(frontier.size(), 1u);
}

TEST_F(NoCResidentDataflowTest, KeepsBaselineOnNonRepresentableRelation) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers)
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  wafer::frontend::FrontendProgramVerificationResult invalid = program();
  invalid.distributedInputs.front().rankSlices[5].strides = {0};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, invalid, *config, &failure)));
  for (const auto &frontier : frontiers)
    EXPECT_EQ(frontier.size(), 1u);
}

TEST_F(NoCResidentDataflowTest, SingleRankDomainDoesNotInventPeerDataflow) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(1);
  frontiers.front().push_back(
      {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
       /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
       /*bufferingPlanOrdinal=*/0});
  wafer::frontend::FrontendProgramVerificationResult singleRank = program();
  singleRank.logicalRankCount = 1;
  singleRank.distributedInputs.front().rankSlices.resize(1);
  singleRank.distributedInputs.front().rankSlices.front().logicalRank = 0;
  singleRank.distributedInputs.front().rankSlices.front().replicaId = 0;
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, singleRank, *config, &failure)))
      << failure;
  ASSERT_EQ(frontiers.front().size(), 1u);
  unsigned peerIssues = 0;
  frontiers.front().front().module->walk([&](mlir::Operation *operation) {
    peerIssues +=
        mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTERecvOp>(operation) ? 1u
                                                                           : 0u;
  });
  EXPECT_EQ(peerIssues, 0u);
}

TEST_F(NoCResidentDataflowTest,
       DistributesIndependentStaticInputTilesAcrossOwners) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers)
    frontier.push_back(
        {parseStaticTileRank(), /*stableOrdinal=*/0,
         wafer::RankArtifactKind::Spill, /*reservedBaseline=*/true,
         wafer::RankBufferingKind::Single, /*bufferingPlanOrdinal=*/0});
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)))
      << failure;
  for (const auto &frontier : frontiers)
    ASSERT_EQ(frontier.size(), 3u);

  const Candidate *directKey = nullptr;
  for (const Candidate &candidate : llvm::drop_begin(frontiers.front())) {
    unsigned maximumRankSends = 0;
    for (const Frontier &frontier : frontiers) {
      const Candidate *rankCandidate =
          findCorrespondingCandidate(frontier, candidate);
      ASSERT_NE(rankCandidate, nullptr);
      maximumRankSends = std::max(
          maximumRankSends, countPeerTransport(*rankCandidate->module).sends);
    }
    if (maximumRankSends > 1) {
      directKey = &candidate;
      break;
    }
  }
  ASSERT_NE(directKey, nullptr);

  unsigned aggregateLoads = 0;
  unsigned aggregateSends = 0;
  unsigned aggregateRecvs = 0;
  std::set<size_t> ownerRanks;
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    const Candidate *candidate =
        findCorrespondingCandidate(frontiers[rank], *directKey);
    ASSERT_NE(candidate, nullptr);
    PeerTransportCounts counts = countPeerTransport(*candidate->module);
    aggregateLoads += counts.loads;
    aggregateSends += counts.sends;
    aggregateRecvs += counts.recvs;
    if (counts.loads != 0)
      ownerRanks.insert(rank);
    EXPECT_LE(counts.loads, 1u);
    EXPECT_TRUE(peerReceivesPrecedeSendsInEveryBlock(*candidate->module));
  }
  EXPECT_EQ(aggregateLoads, 2u);
  EXPECT_EQ(aggregateSends, 30u);
  EXPECT_EQ(aggregateRecvs, 30u);
  EXPECT_EQ(ownerRanks, (std::set<size_t>{0u, 1u}));

  unsigned baselineLoads = 0;
  for (const auto &frontier : frontiers)
    frontier.front().module.get()->walk(
        [&](wafer::InstrRDMAOp) { ++baselineLoads; });
  EXPECT_EQ(baselineLoads, 32u);
}

TEST_F(NoCResidentDataflowTest,
       PreparesNewPeerReceiveBeforeExistingDirectDTETraffic) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  // Discover the semantic owner without assuming a rank ordinal in the test.
  std::vector<Frontier> probe(16);
  for (Frontier &frontier : probe)
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          probe, program(), *config, &failure)))
      << failure;
  const Candidate *probeDirect = nullptr;
  for (const Candidate &candidate : llvm::drop_begin(probe.front())) {
    unsigned maximumRankSends = 0;
    for (const Frontier &frontier : probe) {
      const Candidate *rankCandidate =
          findCorrespondingCandidate(frontier, candidate);
      ASSERT_NE(rankCandidate, nullptr);
      maximumRankSends = std::max(
          maximumRankSends, countPeerTransport(*rankCandidate->module).sends);
    }
    if (maximumRankSends > 1) {
      probeDirect = &candidate;
      break;
    }
  }
  ASSERT_NE(probeDirect, nullptr);
  std::optional<size_t> ownerRank;
  for (size_t rank = 0; rank < probe.size(); ++rank) {
    const Candidate *candidate =
        findCorrespondingCandidate(probe[rank], *probeDirect);
    ASSERT_NE(candidate, nullptr);
    if (countPeerTransport(*candidate->module).loads != 0)
      ownerRank = rank;
  }
  ASSERT_TRUE(ownerRank);
  const size_t receiverRank = (*ownerRank + 1) % probe.size();

  // Before composition message 41 is acyclic: the future receiver sends and
  // waits before its DDR load; the future owner receives after its DDR load.
  // Boundary fan-out then inserts the reverse message dependency.  The new
  // receive must be prepared before the existing send, otherwise the two
  // sender waits form a cross-message cycle that per-message Direct-DTE
  // matching cannot observe.
  constexpr int64_t existingCommunicationId = 41;
  std::vector<Frontier> frontiers(16);
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    mlir::OwningOpRef<mlir::ModuleOp> module = parseRank();
    ASSERT_TRUE(module);
    if (rank == receiverRank)
      ASSERT_TRUE(addExistingPeerMessage(*module, /*send=*/true,
                                         static_cast<int64_t>(*ownerRank),
                                         existingCommunicationId));
    if (rank == *ownerRank)
      ASSERT_TRUE(addExistingPeerMessage(*module, /*send=*/false,
                                         static_cast<int64_t>(receiverRank),
                                         existingCommunicationId));
    frontiers[rank].push_back(
        {std::move(module), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  }

  failure.clear();
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)))
      << failure;
  const Candidate *directKey = nullptr;
  for (const Candidate &candidate : llvm::drop_begin(frontiers.front())) {
    unsigned maximumRankSends = 0;
    for (const Frontier &frontier : frontiers) {
      const Candidate *rankCandidate =
          findCorrespondingCandidate(frontier, candidate);
      ASSERT_NE(rankCandidate, nullptr);
      maximumRankSends = std::max(
          maximumRankSends, countPeerTransport(*rankCandidate->module).sends);
    }
    if (maximumRankSends > 1) {
      directKey = &candidate;
      break;
    }
  }
  ASSERT_NE(directKey, nullptr);

  const Candidate *receiver =
      findCorrespondingCandidate(frontiers[receiverRank], *directKey);
  ASSERT_NE(receiver, nullptr);
  wafer::InstrDTESendOp existingSend;
  wafer::InstrDTERecvOp newReceive;
  receiver->module.get()->walk([&](wafer::InstrDTESendOp send) {
    if (send.getMessage().getCommunicationId() == existingCommunicationId)
      existingSend = send;
  });
  receiver->module.get()->walk([&](wafer::InstrDTERecvOp recv) {
    if (recv.getMessage().getCommunicationId() != existingCommunicationId)
      newReceive = recv;
  });
  ASSERT_TRUE(existingSend);
  ASSERT_TRUE(newReceive);
  ASSERT_EQ(existingSend->getBlock(), newReceive->getBlock());
  EXPECT_TRUE(newReceive->isBeforeInBlock(existingSend));

  // Retain only the target sibling and the immutable baseline. The candidate
  // remains structurally legal, but its small payload does not repay the
  // static model's message-startup and route costs.
  const int64_t targetStableOrdinal = directKey->stableOrdinal;
  const wafer::RankArtifactKind targetArtifactKind = directKey->artifactKind;
  const wafer::RankBufferingKind targetBufferingKind = directKey->bufferingKind;
  const uint32_t targetBufferingPlanOrdinal = directKey->bufferingPlanOrdinal;
  const wafer::RankWorkerPlacementKind targetWorkerPlacementKind =
      directKey->workerPlacementKind;
  const uint32_t targetWorkerPlacementPlanOrdinal =
      directKey->workerPlacementPlanOrdinal;
  ASSERT_NE(targetStableOrdinal, frontiers.front().front().stableOrdinal);
  for (Frontier &frontier : frontiers)
    llvm::erase_if(frontier, [&](const Candidate &candidate) {
      const bool matchesTarget =
          candidate.stableOrdinal == targetStableOrdinal &&
          candidate.artifactKind == targetArtifactKind &&
          candidate.bufferingKind == targetBufferingKind &&
          candidate.bufferingPlanOrdinal == targetBufferingPlanOrdinal &&
          candidate.workerPlacementKind == targetWorkerPlacementKind &&
          candidate.workerPlacementPlanOrdinal ==
              targetWorkerPlacementPlanOrdinal;
      return !candidate.reservedBaseline && !matchesTarget;
    });

  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::WholeVariantSelectionStatistics statistics;
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program(), *config, diagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::Production,
      &statistics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  EXPECT_TRUE(llvm::all_of(accepted->selectedReservedBaselines,
                           [](bool reserved) { return reserved; }));
  EXPECT_TRUE(llvm::none_of(accepted->selectedStableOrdinals,
                            [&](int64_t stableOrdinal) {
                              return stableOrdinal == targetStableOrdinal;
                            }));
  EXPECT_GT(statistics.noCProfitabilityEvaluations, 0u);
  EXPECT_EQ(statistics.noCProfitabilityEvaluations,
            statistics.noCProfitabilityRejected +
                statistics.noCProfitabilityIndeterminate +
                statistics.noCProfitabilityEstimated +
                statistics.noCProfitabilityProven);
  EXPECT_GT(statistics.noCProfitabilityRejected, 0u);
  EXPECT_EQ(statistics.noCProfitabilityIndeterminate, 0u);
  EXPECT_EQ(statistics.noCProfitabilityEstimated, 0u);
  EXPECT_EQ(statistics.noCProfitabilityProven, 0u);
}

TEST_F(NoCResidentDataflowTest,
       BuildsAcyclicMixedTransportAcrossStructuredBlockOccurrences) {
  constexpr int64_t existingCommunicationId = 73;
  std::vector<Frontier> frontiers(16);
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    mlir::OwningOpRef<mlir::ModuleOp> module = parseLoopRank();
    ASSERT_TRUE(module);
    std::optional<bool> send;
    int64_t peer = -1;
    if (rank == 0) {
      send = true;
      peer = 1;
    } else if (rank == 1) {
      send = false;
      peer = 0;
    }
    ASSERT_TRUE(addPrecedingTileRegionPeerMessage(*module, send, peer,
                                                  existingCommunicationId));
    frontiers[rank].push_back(
        {std::move(module), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  }

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)))
      << failure;

  // The existing receive is prepared and completed in a preceding sibling
  // tile region before every rank enters the loop-bearing region. The
  // synthesized fanout occurs inside the loop, so the global
  // structured-occurrence wait graph is acyclic even though the two message
  // classes live in different blocks.
  for (const Frontier &frontier : frontiers)
    ASSERT_GT(frontier.size(), 1u);

  const Candidate *directKey = nullptr;
  for (const Candidate &candidate : llvm::drop_begin(frontiers.front())) {
    unsigned maximumRankSends = 0;
    for (const Frontier &frontier : frontiers) {
      const Candidate *rankCandidate =
          findCorrespondingCandidate(frontier, candidate);
      ASSERT_NE(rankCandidate, nullptr);
      maximumRankSends = std::max(
          maximumRankSends, countPeerTransport(*rankCandidate->module).sends);
    }
    if (maximumRankSends > 1) {
      directKey = &candidate;
      break;
    }
  }
  ASSERT_NE(directKey, nullptr);

  unsigned existingSends = 0;
  unsigned existingRecvs = 0;
  unsigned loopSends = 0;
  unsigned loopRecvs = 0;
  for (const Frontier &frontier : frontiers) {
    const Candidate *candidate =
        findCorrespondingCandidate(frontier, *directKey);
    ASSERT_NE(candidate, nullptr);
    candidate->module.get()->walk([&](wafer::InstrDTESendOp send) {
      if (send.getMessage().getCommunicationId() == existingCommunicationId) {
        ++existingSends;
        EXPECT_TRUE(
            static_cast<bool>(send->getParentOfType<wafer::TileRegionOp>()));
        EXPECT_FALSE(
            static_cast<bool>(send->getParentOfType<mlir::scf::ForOp>()));
      } else {
        ++loopSends;
        EXPECT_TRUE(
            static_cast<bool>(send->getParentOfType<mlir::scf::ForOp>()));
      }
    });
    candidate->module.get()->walk([&](wafer::InstrDTERecvOp recv) {
      if (recv.getMessage().getCommunicationId() == existingCommunicationId) {
        ++existingRecvs;
        EXPECT_TRUE(
            static_cast<bool>(recv->getParentOfType<wafer::TileRegionOp>()));
        EXPECT_FALSE(
            static_cast<bool>(recv->getParentOfType<mlir::scf::ForOp>()));
      } else {
        ++loopRecvs;
        EXPECT_TRUE(
            static_cast<bool>(recv->getParentOfType<mlir::scf::ForOp>()));
      }
    });
  }
  EXPECT_EQ(existingSends, 1u);
  EXPECT_EQ(existingRecvs, 1u);
  EXPECT_EQ(loopSends, 15u);
  EXPECT_EQ(loopRecvs, 15u);

  const int64_t targetStableOrdinal = directKey->stableOrdinal;
  const wafer::RankArtifactKind targetArtifactKind = directKey->artifactKind;
  const wafer::RankBufferingKind targetBufferingKind = directKey->bufferingKind;
  const uint32_t targetBufferingPlanOrdinal = directKey->bufferingPlanOrdinal;
  const wafer::RankWorkerPlacementKind targetWorkerPlacementKind =
      directKey->workerPlacementKind;
  const uint32_t targetWorkerPlacementPlanOrdinal =
      directKey->workerPlacementPlanOrdinal;
  ASSERT_NE(targetStableOrdinal, frontiers.front().front().stableOrdinal);
  for (Frontier &frontier : frontiers)
    llvm::erase_if(frontier, [&](const Candidate &candidate) {
      const bool matchesTarget =
          candidate.stableOrdinal == targetStableOrdinal &&
          candidate.artifactKind == targetArtifactKind &&
          candidate.bufferingKind == targetBufferingKind &&
          candidate.bufferingPlanOrdinal == targetBufferingPlanOrdinal &&
          candidate.workerPlacementKind == targetWorkerPlacementKind &&
          candidate.workerPlacementPlanOrdinal ==
              targetWorkerPlacementPlanOrdinal;
      return !candidate.reservedBaseline && !matchesTarget;
    });

  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::WholeVariantSelectionStatistics statistics;
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program(), *config, diagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::Production,
      &statistics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  EXPECT_TRUE(llvm::all_of(accepted->selectedReservedBaselines,
                           [](bool reserved) { return reserved; }));
  EXPECT_TRUE(llvm::none_of(accepted->selectedStableOrdinals,
                            [&](int64_t stableOrdinal) {
                              return stableOrdinal == targetStableOrdinal;
                            }));
  EXPECT_GT(statistics.noCProfitabilityEvaluations, 0u);
  EXPECT_EQ(statistics.noCProfitabilityEvaluations,
            statistics.noCProfitabilityRejected +
                statistics.noCProfitabilityIndeterminate +
                statistics.noCProfitabilityEstimated +
                statistics.noCProfitabilityProven);
  EXPECT_GT(statistics.noCProfitabilityRejected, 0u);
  EXPECT_EQ(statistics.noCProfitabilityIndeterminate, 0u);
  EXPECT_EQ(statistics.noCProfitabilityEstimated, 0u);
  EXPECT_EQ(statistics.noCProfitabilityProven, 0u);
}

TEST_F(NoCResidentDataflowTest,
       DerivesAllRankFixedSlotSiblingFromNoCResidentLoop) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers)
    frontier.push_back(
        {parseLoopRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());

  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)))
      << failure;

  for (const auto &frontier : frontiers)
    ASSERT_EQ(frontier.size(), 5u);
  auto fixedKeyIt =
      llvm::find_if(frontiers.front(), [](const Candidate &candidate) {
        return candidate.module &&
               candidate.bufferingKind ==
                   wafer::RankBufferingKind::StaticFixedSlot &&
               candidate.bufferingPlanOrdinal != 0 &&
               countPeerIssues(*candidate.module) > 0;
      });
  ASSERT_NE(fixedKeyIt, frontiers.front().end());
  const Candidate *fixedKey = &*fixedKeyIt;
  for (auto &frontier : frontiers) {
    const Candidate *fixed = findCorrespondingCandidate(frontier, *fixedKey);
    ASSERT_NE(fixed, nullptr);
    EXPECT_GT(countPeerIssues(*fixed->module), 0u);
    EXPECT_TRUE(llvm::any_of(frontier, [&](const Candidate &candidate) {
      return candidate.stableOrdinal == fixed->stableOrdinal &&
             candidate.bufferingKind == wafer::RankBufferingKind::Single;
    }));

    unsigned plannedSlots = 0;
    fixed->module.get()->walk([&](mlir::memref::AllocOp allocation) {
      if (!wafer::isWaferSPMMemRefType(allocation.getType()))
        return;
      EXPECT_TRUE(allocation->hasAttr(wafer::kWaferSPMOffsetAttrName));
      EXPECT_FALSE(
          static_cast<bool>(allocation->getParentOfType<mlir::scf::ForOp>()));
      ++plannedSlots;
    });
    EXPECT_GE(plannedSlots, 2u);
  }
}

TEST_F(NoCResidentDataflowTest,
       RejectsFixedSlotTupleWhenAnyRankLoopBoundsOrPathDiffer) {
  enum class MismatchKind { Bounds, StructuredPath };
  for (MismatchKind mismatch :
       {MismatchKind::Bounds, MismatchKind::StructuredPath}) {
    SCOPED_TRACE(mismatch == MismatchKind::Bounds ? "bounds"
                                                  : "structured-path");
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> owners;
    owners.reserve(16);
    for (size_t rank = 0; rank < 16; ++rank) {
      mlir::OwningOpRef<mlir::ModuleOp> module = parseLoopRank();
      ASSERT_TRUE(module);
      if (rank == 1) {
        mlir::scf::ForOp loop;
        module->walk([&](mlir::scf::ForOp candidate) {
          if (!loop)
            loop = candidate;
        });
        ASSERT_TRUE(loop);
        mlir::OpBuilder builder(loop);
        auto differentUpper = builder.create<mlir::arith::ConstantIndexOp>(
            loop.getLoc(), mismatch == MismatchKind::Bounds ? 5 : 2);
        if (mismatch == MismatchKind::Bounds) {
          loop->setOperand(/*upperBound=*/1, differentUpper);
        } else {
          // The extra ordinal-zero sibling gives the real loop a distinct
          // current-IR occurrence path; its different bound prevents the
          // dummy loop from becoming a substitute correspondence match.
          builder.create<mlir::scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                                           differentUpper, loop.getStep(),
                                           mlir::ValueRange{});
        }
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      }
      owners.push_back(std::move(module));
    }

    llvm::SmallVector<mlir::ModuleOp, 16> modules;
    std::vector<std::string> snapshots;
    for (mlir::OwningOpRef<mlir::ModuleOp> &owner : owners) {
      modules.push_back(*owner);
      std::string snapshot;
      llvm::raw_string_ostream stream(snapshot);
      owner->print(stream);
      stream.flush();
      snapshots.push_back(std::move(snapshot));
    }
    mlir::scf::ForOp anchor;
    modules.front().walk([&](mlir::scf::ForOp loop) {
      if (!anchor)
        anchor = loop;
    });
    ASSERT_TRUE(anchor);

    mlir::FailureOr<llvm::SmallVector<mlir::scf::ForOp, 16>> correspondence =
        wafer::compiler::detail::resolveExactStaticLoopCorrespondence(modules,
                                                                      anchor);
    EXPECT_TRUE(mlir::failed(correspondence));

    // Resolution is one read-only all-rank transaction: no earlier rank can
    // be rewritten when a later rank lacks the exact bounds/path match.
    for (auto [owner, expected] : llvm::zip(owners, snapshots)) {
      std::string actual;
      llvm::raw_string_ostream stream(actual);
      owner->print(stream);
      stream.flush();
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST_F(NoCResidentDataflowTest,
       ComposesTypedWorkerPlacementWithDirectDTEDataflow) {
  mlir::OwningOpRef<mlir::ModuleOp> workerPrototype = deriveWorkerPlacedRank();
  ASSERT_TRUE(workerPrototype);

  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers) {
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
    frontier.push_back(
        {mlir::cast<mlir::ModuleOp>(workerPrototype->clone()),
         /*stableOrdinal=*/1, wafer::RankArtifactKind::SpillReady,
         /*reservedBaseline=*/false, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0,
         wafer::RankWorkerPlacementKind::DisjointComponents,
         /*workerPlacementPlanOrdinal=*/1});
  }

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)))
      << failure;

  for (auto &frontier : frontiers) {
    unsigned composed = 0;
    for (const Candidate &candidate : frontier) {
      if (candidate.artifactKind != wafer::RankArtifactKind::Resident ||
          candidate.workerPlacementKind !=
              wafer::RankWorkerPlacementKind::DisjointComponents ||
          countPeerIssues(*candidate.module) == 0)
        continue;
      std::set<uint32_t> workers;
      candidate.module.get()->walk([&](mlir::Operation *operation) {
        if (std::optional<wafer::NCCWorker> worker =
                wafer::getNCCIssueWorker(operation))
          workers.insert(static_cast<uint32_t>(*worker));
      });
      EXPECT_GE(workers.size(), 2u);
      EXPECT_TRUE(
          llvm::any_of(workers, [](uint32_t worker) { return worker != 0; }));
      ++composed;
    }
    EXPECT_EQ(composed, 2u);
    // Remove the source worker-only tuple so qualification must cross every
    // mixed NCC-worker + Direct-DTE whole-variant gate.
    frontier.erase(frontier.begin() + 1);
  }

  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto qualified = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program(), *config, diagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::
          QualifyWorkerPlacement);
  ASSERT_TRUE(mlir::succeeded(qualified)) << diagnosticText;
  EXPECT_TRUE(llvm::all_of(
      qualified->selectedWorkerPlacementKinds,
      [](wafer::RankWorkerPlacementKind kind) {
        return kind == wafer::RankWorkerPlacementKind::DisjointComponents;
      }));
  unsigned peerIssues = 0;
  for (const wafer::compiler::RankExecutable &rank : qualified->ranks)
    peerIssues += countPeerIssues(rank.getModule());
  EXPECT_GT(peerIssues, 0u);
}

TEST_F(NoCResidentDataflowTest,
       SkipsIncompleteNullAndWorkerPlacedTuplesButComposesFixedSlotSeed) {
  constexpr size_t originalSize = 6;
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    auto &frontier = frontiers[rank];
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
    // Neither key is complete across all ranks.
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/rank == 0 ? 10 : 11,
         wafer::RankArtifactKind::Spill, /*reservedBaseline=*/false,
         wafer::RankBufferingKind::Single, /*bufferingPlanOrdinal=*/0});
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/rank == 15 ? 21 : 20,
         wafer::RankArtifactKind::SpillReady, /*reservedBaseline=*/false,
         wafer::RankBufferingKind::Single, /*bufferingPlanOrdinal=*/0});
    // This key is complete in metadata but its imported module is absent.
    frontier.push_back(
        {mlir::OwningOpRef<mlir::ModuleOp>(), /*stableOrdinal=*/30,
         wafer::RankArtifactKind::Resident, /*reservedBaseline=*/false,
         wafer::RankBufferingKind::Single, /*bufferingPlanOrdinal=*/0});
    // A typed fixed-slot tuple is a valid seed. The worker-tagged tuple below
    // is not: its IR has no nonzero worker assignment, so metadata alone
    // cannot manufacture a worker/dataflow composition seed.
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/40, wafer::RankArtifactKind::SpillReady,
         /*reservedBaseline=*/false, wafer::RankBufferingKind::StaticFixedSlot,
         /*bufferingPlanOrdinal=*/1, wafer::RankWorkerPlacementKind::Unplaced,
         /*workerPlacementPlanOrdinal=*/0});
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/50, wafer::RankArtifactKind::SpillReady,
         /*reservedBaseline=*/false, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0,
         wafer::RankWorkerPlacementKind::DisjointComponents,
         /*workerPlacementPlanOrdinal=*/1});
  }

  std::vector<std::vector<std::string>> originalSnapshots(16);
  for (size_t rank = 0; rank < frontiers.size(); ++rank)
    for (const auto &candidate : frontiers[rank])
      originalSnapshots[rank].push_back(candidateSnapshot(candidate));

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)))
      << failure;

  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    ASSERT_EQ(frontiers[rank].size(), originalSize + 4);
    for (size_t index = 0; index < originalSize; ++index)
      EXPECT_EQ(candidateSnapshot(frontiers[rank][index]),
                originalSnapshots[rank][index]);
    unsigned appendedSingle = 0;
    unsigned appendedFixedSlot = 0;
    for (const auto &candidate :
         llvm::drop_begin(frontiers[rank], originalSize)) {
      EXPECT_EQ(candidate.artifactKind, wafer::RankArtifactKind::Resident);
      EXPECT_EQ(candidate.workerPlacementKind,
                wafer::RankWorkerPlacementKind::Unplaced);
      appendedSingle +=
          candidate.bufferingKind == wafer::RankBufferingKind::Single ? 1u : 0u;
      appendedFixedSlot +=
          candidate.bufferingKind == wafer::RankBufferingKind::StaticFixedSlot
              ? 1u
              : 0u;
    }
    EXPECT_EQ(appendedSingle, 2u);
    EXPECT_EQ(appendedFixedSlot, 2u);
  }
}

TEST_F(NoCResidentDataflowTest,
       CompleteTupleSeedCapIsDeterministicAndPreservesOldFrontier) {
  constexpr size_t optimizedSeedCount =
      wafer::compiler::detail::kNoCResidentCompleteTupleSeedLimit + 3;
  constexpr size_t originalSize = optimizedSeedCount + 1;
  auto buildFrontiers = [&]() {
    std::vector<wafer::compiler::detail::RankVariantFrontier> result(16);
    for (auto &frontier : result) {
      mlir::OwningOpRef<mlir::ModuleOp> prototype = parseRank();
      EXPECT_TRUE(prototype);
      for (size_t ordinal = 0; ordinal <= optimizedSeedCount; ++ordinal)
        frontier.push_back(
            {mlir::cast<mlir::ModuleOp>(prototype->getOperation()->clone()),
             static_cast<int64_t>(ordinal), wafer::RankArtifactKind::Spill,
             /*reservedBaseline=*/ordinal == 0,
             wafer::RankBufferingKind::Single,
             /*bufferingPlanOrdinal=*/0});
    }
    return result;
  };

  auto first = buildFrontiers();
  auto second = buildFrontiers();
  std::vector<std::vector<std::string>> originalSnapshots(16);
  for (size_t rank = 0; rank < first.size(); ++rank)
    for (const auto &candidate : first[rank])
      originalSnapshots[rank].push_back(candidateSnapshot(candidate));

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  for (auto *frontiers : {&first, &second}) {
    std::string failure;
    ASSERT_TRUE(mlir::succeeded(
        wafer::compiler::detail::appendNoCResidentDataflowCandidates(
            *frontiers, program(), *config, &failure)))
        << failure;
  }

  constexpr size_t appendedGenerationCount =
      2 * wafer::compiler::detail::kNoCResidentCompleteTupleSeedLimit;
  for (size_t rank = 0; rank < first.size(); ++rank) {
    ASSERT_EQ(first[rank].size(), originalSize + appendedGenerationCount);
    ASSERT_EQ(second[rank].size(), first[rank].size());
    for (size_t index = 0; index < originalSize; ++index)
      EXPECT_EQ(candidateSnapshot(first[rank][index]),
                originalSnapshots[rank][index]);
    for (size_t index = 0; index < first[rank].size(); ++index)
      EXPECT_EQ(candidateSnapshot(first[rank][index]),
                candidateSnapshot(second[rank][index]));
  }

  std::set<int64_t> freshOrdinals;
  for (const auto &candidate : llvm::drop_begin(first.front(), originalSize)) {
    EXPECT_EQ(candidate.artifactKind, wafer::RankArtifactKind::Resident);
    EXPECT_EQ(candidate.bufferingKind, wafer::RankBufferingKind::Single);
    EXPECT_EQ(candidate.workerPlacementKind,
              wafer::RankWorkerPlacementKind::Unplaced);
    freshOrdinals.insert(candidate.stableOrdinal);
  }
  EXPECT_EQ(freshOrdinals.size(), appendedGenerationCount);
}

TEST_F(NoCResidentDataflowTest,
       SaturatedPhysicalSeedBandsStillMaterializeGenericTuple) {
  mlir::OwningOpRef<mlir::ModuleOp> workerPrototype = deriveWorkerPlacedRank();
  ASSERT_TRUE(workerPrototype);

  constexpr size_t workerCount =
      wafer::kWorkerPlacementRankFrontierAdmissionLimit;
  constexpr size_t fixedCount = wafer::kFixedSlotRankFrontierAdmissionLimit;
  constexpr size_t genericCount = 4;
  constexpr size_t originalSize = 1 + workerCount + fixedCount + genericCount;
  std::vector<Frontier> frontiers(16);
  for (Frontier &frontier : frontiers) {
    frontier.push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
    for (size_t worker = 0; worker < workerCount; ++worker)
      frontier.push_back(
          {mlir::cast<mlir::ModuleOp>(workerPrototype->clone()),
           /*stableOrdinal=*/100 + static_cast<int64_t>(worker),
           wafer::RankArtifactKind::SpillReady,
           /*reservedBaseline=*/false, wafer::RankBufferingKind::Single,
           /*bufferingPlanOrdinal=*/0,
           wafer::RankWorkerPlacementKind::DisjointComponents,
           /*workerPlacementPlanOrdinal=*/static_cast<uint32_t>(worker) + 1});
    for (size_t fixed = 0; fixed < fixedCount; ++fixed)
      frontier.push_back(
          {parseRank(), /*stableOrdinal=*/200 + static_cast<int64_t>(fixed),
           wafer::RankArtifactKind::SpillReady,
           /*reservedBaseline=*/false,
           wafer::RankBufferingKind::StaticFixedSlot,
           /*bufferingPlanOrdinal=*/static_cast<uint32_t>(fixed) + 1,
           wafer::RankWorkerPlacementKind::Unplaced,
           /*workerPlacementPlanOrdinal=*/0});
    for (size_t generic = 0; generic < genericCount; ++generic)
      frontier.push_back(
          {parseRank(), /*stableOrdinal=*/300 + static_cast<int64_t>(generic),
           wafer::RankArtifactKind::Resident,
           /*reservedBaseline=*/false, wafer::RankBufferingKind::Single,
           /*bufferingPlanOrdinal=*/0, wafer::RankWorkerPlacementKind::Unplaced,
           /*workerPlacementPlanOrdinal=*/0});
  }

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)))
      << failure;

  for (const Frontier &frontier : frontiers) {
    ASSERT_EQ(frontier.size(), originalSize + 16);
    unsigned workerOutputs = 0;
    unsigned fixedOutputs = 0;
    unsigned genericOutputs = 0;
    for (const Candidate &candidate :
         llvm::drop_begin(frontier, originalSize)) {
      if (candidate.workerPlacementKind ==
          wafer::RankWorkerPlacementKind::DisjointComponents) {
        ++workerOutputs;
      } else if (candidate.bufferingKind ==
                 wafer::RankBufferingKind::StaticFixedSlot) {
        ++fixedOutputs;
      } else {
        ++genericOutputs;
      }
    }
    EXPECT_EQ(workerOutputs, 6u);
    EXPECT_EQ(fixedOutputs, 4u);
    // Baseline contributes two generations. Four additional generic outputs
    // prove two non-baseline Single+Unplaced correspondence seeds survived
    // simultaneous saturation of both physical bands.
    EXPECT_EQ(genericOutputs, 6u);
  }
}

TEST_F(NoCResidentDataflowTest,
       FailedGenerationsDoNotConsumeLaterWorkerAdmissionBudget) {
  mlir::OwningOpRef<mlir::ModuleOp> workerPrototype = deriveWorkerPlacedRank();
  ASSERT_TRUE(workerPrototype);
  mlir::OwningOpRef<mlir::ModuleOp> noBoundaryPrototype =
      mlir::cast<mlir::ModuleOp>(workerPrototype->clone());
  wafer::InstrRDMAOp boundaryLoad;
  noBoundaryPrototype->walk([&](wafer::InstrRDMAOp load) {
    if (!boundaryLoad)
      boundaryLoad = load;
  });
  ASSERT_TRUE(boundaryLoad);
  boundaryLoad.erase();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*noBoundaryPrototype)));

  constexpr size_t failedSeedCount = 6;
  constexpr int64_t validStableOrdinal =
      static_cast<int64_t>(failedSeedCount) + 1;
  auto buildFrontiers = [&](bool includeFailedSeeds) {
    std::vector<Frontier> result(16);
    for (Frontier &frontier : result) {
      frontier.push_back(
          {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
           /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
           /*bufferingPlanOrdinal=*/0});
      if (includeFailedSeeds) {
        for (size_t failed = 0; failed < failedSeedCount; ++failed)
          frontier.push_back(
              {mlir::cast<mlir::ModuleOp>(noBoundaryPrototype->clone()),
               /*stableOrdinal=*/static_cast<int64_t>(failed) + 1,
               wafer::RankArtifactKind::SpillReady,
               /*reservedBaseline=*/false, wafer::RankBufferingKind::Single,
               /*bufferingPlanOrdinal=*/0,
               wafer::RankWorkerPlacementKind::DisjointComponents,
               /*workerPlacementPlanOrdinal=*/
               static_cast<uint32_t>(failed) + 1});
      }
      frontier.push_back(
          {mlir::cast<mlir::ModuleOp>(workerPrototype->clone()),
           validStableOrdinal, wafer::RankArtifactKind::SpillReady,
           /*reservedBaseline=*/false, wafer::RankBufferingKind::Single,
           /*bufferingPlanOrdinal=*/0,
           wafer::RankWorkerPlacementKind::DisjointComponents,
           /*workerPlacementPlanOrdinal=*/
           static_cast<uint32_t>(failedSeedCount) + 1});
    }
    return result;
  };

  auto control = buildFrontiers(/*includeFailedSeeds=*/false);
  auto withFailures = buildFrontiers(/*includeFailedSeeds=*/true);
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  for (auto *frontiers : {&control, &withFailures}) {
    std::string failure;
    ASSERT_TRUE(mlir::succeeded(
        wafer::compiler::detail::appendNoCResidentDataflowCandidates(
            *frontiers, program(), *config, &failure)))
        << failure;
  }

  auto countAppendedWorkerTuples = [](const Frontier &frontier) {
    return llvm::count_if(frontier, [](const Candidate &candidate) {
      return candidate.artifactKind == wafer::RankArtifactKind::Resident &&
             candidate.workerPlacementKind ==
                 wafer::RankWorkerPlacementKind::DisjointComponents &&
             countPeerIssues(*candidate.module) > 0;
    });
  };
  for (size_t rank = 0; rank < control.size(); ++rank) {
    const auto controlCount = countAppendedWorkerTuples(control[rank]);
    EXPECT_GT(controlCount, 0);
    EXPECT_EQ(countAppendedWorkerTuples(withFailures[rank]), controlCount);
  }
}

TEST_F(NoCResidentDataflowTest,
       MalformedBaselineFailureLeavesEveryFrontierUnchanged) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    frontiers[rank].push_back(
        {parseRank(), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
    if (rank == 0)
      frontiers[rank].push_back({parseRank(), /*stableOrdinal=*/1,
                                 wafer::RankArtifactKind::SpillReady,
                                 /*reservedBaseline=*/true,
                                 wafer::RankBufferingKind::Single,
                                 /*bufferingPlanOrdinal=*/0});
  }
  std::vector<std::vector<std::string>> originalSnapshots(16);
  for (size_t rank = 0; rank < frontiers.size(); ++rank)
    for (const auto &candidate : frontiers[rank])
      originalSnapshots[rank].push_back(candidateSnapshot(candidate));

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  EXPECT_TRUE(
      mlir::failed(wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program(), *config, &failure)));
  EXPECT_NE(failure.find("canonical baseline"), std::string::npos);
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    ASSERT_EQ(frontiers[rank].size(), originalSnapshots[rank].size());
    for (size_t index = 0; index < frontiers[rank].size(); ++index)
      EXPECT_EQ(candidateSnapshot(frontiers[rank][index]),
                originalSnapshots[rank][index]);
  }
}

TEST_F(NoCResidentDataflowTest,
       FullSelectorRejectsSmallOperandDrivenBoundaryComposition) {
  std::optional<ActualTraversalFrontiers> actual =
      buildActualTraversalFrontiers(
          R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%input: tensor<4x6xf32>) -> tensor<4x6xf32> {
    %out = tensor.empty() : tensor<4x6xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<4x6xf32>) outs(%out : tensor<4x6xf32>) {
    ^bb0(%value: f32, %old: f32):
      %sum = arith.addf %value, %value : f32
      linalg.yield %sum : f32
    } -> tensor<4x6xf32>
    return %result : tensor<4x6xf32>
  }
}
)mlir",
          kOperandDrivenInterfaceStableOrdinal,
          /*duplicateBaselineLoad=*/true);
  ASSERT_TRUE(actual);
  ASSERT_FALSE(actual->interfaceComputeIssues.empty());
  ASSERT_GT(actual->interfaceOutputWriteCount, 0u);
  for (const auto &frontier : actual->frontiers) {
    ASSERT_EQ(frontier.size(), 2u);
    EXPECT_EQ(frontier[1].stableOrdinal, kOperandDrivenInterfaceStableOrdinal);
    EXPECT_EQ(getComputeIssueFingerprint(*frontier[1].module),
              actual->interfaceComputeIssues);
    EXPECT_EQ(countPeerIssues(*frontier[1].module), 0u);
  }

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  wafer::frontend::FrontendProgramVerificationResult frontendProgram =
      replicatedProgram(/*inputShape=*/{4, 6}, /*outputShape=*/{4, 6});
  // Keep this selection proof focused on the standard operand-driven input
  // traversal. A partitioned required output has one distinct ABI slice per
  // rank and therefore is not an output-fanout opportunity.
  auto &output = frontendProgram.distributedOutputs.front();
  output.distribution = wafer::frontend::ProgramDistributionKind::Partitioned;
  output.globalShape = {64, 6};
  for (int64_t rank = 0; rank < 16; ++rank) {
    output.rankSlices[static_cast<size_t>(rank)].replicaId = 0;
    output.rankSlices[static_cast<size_t>(rank)].offsets = {rank * 4, 0};
  }
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          actual->frontiers, frontendProgram, *config, &failure)))
      << failure;

  // The deliberately duplicate-loading baseline is not a legal fan-out seed,
  // so both fresh semantic generations must come from the complete actual
  // operand-driven tuple.
  std::set<int64_t> compositeOrdinals;
  for (const auto &candidate : actual->frontiers.front())
    if (candidate.stableOrdinal > kOperandDrivenInterfaceStableOrdinal &&
        candidate.artifactKind == wafer::RankArtifactKind::Resident &&
        candidate.bufferingKind == wafer::RankBufferingKind::Single &&
        candidate.workerPlacementKind ==
            wafer::RankWorkerPlacementKind::Unplaced)
      compositeOrdinals.insert(candidate.stableOrdinal);
  ASSERT_EQ(compositeOrdinals.size(), 2u);
  for (int64_t ordinal : compositeOrdinals)
    for (const auto &frontier : actual->frontiers) {
      auto composed = llvm::find_if(
          frontier,
          [&](const wafer::compiler::detail::RankVariantCandidate &candidate) {
            return candidate.stableOrdinal == ordinal &&
                   candidate.artifactKind ==
                       wafer::RankArtifactKind::Resident &&
                   candidate.bufferingKind ==
                       wafer::RankBufferingKind::Single &&
                   candidate.workerPlacementKind ==
                       wafer::RankWorkerPlacementKind::Unplaced;
          });
      ASSERT_NE(composed, frontier.end());
      EXPECT_EQ(getComputeIssueFingerprint(*composed->module),
                actual->interfaceComputeIssues);
      EXPECT_EQ(countLoops(*composed->module), actual->interfaceLoopCount);
      EXPECT_EQ(countOutputWrites(*composed->module),
                actual->interfaceOutputWriteCount);
      EXPECT_GT(countPeerIssues(*composed->module), 0u);
    }

  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::WholeVariantSelectionStatistics statistics;
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      actual->frontiers, frontendProgram, *config, diagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::Production,
      &statistics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->selectedStableOrdinals.size(), 16u);
  ASSERT_EQ(accepted->selectedArtifactKinds.size(), 16u);
  for (int64_t ordinal : accepted->selectedStableOrdinals)
    EXPECT_EQ(compositeOrdinals.find(ordinal), compositeOrdinals.end());
  unsigned acceptedPeerIssues = 0;
  for (const wafer::compiler::RankExecutable &rank : accepted->ranks) {
    EXPECT_EQ(getComputeIssueFingerprint(rank.getModule()),
              actual->interfaceComputeIssues);
    EXPECT_EQ(countLoops(rank.getModule()), actual->interfaceLoopCount);
    acceptedPeerIssues += countPeerIssues(rank.getModule());
  }
  EXPECT_EQ(acceptedPeerIssues, 0u);
  EXPECT_GT(statistics.noCProfitabilityEvaluations, 0u);
  EXPECT_EQ(statistics.noCProfitabilityEvaluations,
            statistics.noCProfitabilityRejected +
                statistics.noCProfitabilityIndeterminate +
                statistics.noCProfitabilityEstimated +
                statistics.noCProfitabilityProven);
  EXPECT_GT(statistics.noCProfitabilityRejected, 0u);
  EXPECT_EQ(statistics.noCProfitabilityIndeterminate, 0u);
  EXPECT_EQ(statistics.noCProfitabilityEstimated, 0u);
  EXPECT_EQ(statistics.noCProfitabilityProven, 0u);
}

TEST_F(NoCResidentDataflowTest,
       ComposesPartialReductionCompleteTupleWithReplicatedBoundary) {
  std::optional<ActualTraversalFrontiers> actual =
      buildActualTraversalFrontiers(
          R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%input: tensor<4x8xf32>) -> tensor<4xf32> {
    %out = tensor.empty() : tensor<4xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32)
        outs(%out : tensor<4xf32>) -> tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir",
          kPartialReductionInterfaceStableOrdinal);
  ASSERT_TRUE(actual);
  ASSERT_FALSE(actual->interfaceComputeIssues.empty());
  ASSERT_GT(actual->interfaceOutputWriteCount, 0u);
  const int64_t firstFreshOrdinal = kPartialReductionInterfaceStableOrdinal + 1;

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  wafer::frontend::FrontendProgramVerificationResult frontendProgram =
      replicatedProgram(/*inputShape=*/{4, 8}, /*outputShape=*/{4});
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          actual->frontiers, frontendProgram, *config, &failure)))
      << failure;

  std::set<int64_t> freshSingleOrdinals;
  for (const auto &candidate : actual->frontiers.front())
    if (candidate.stableOrdinal >= firstFreshOrdinal &&
        candidate.artifactKind == wafer::RankArtifactKind::Resident &&
        candidate.bufferingKind == wafer::RankBufferingKind::Single &&
        candidate.workerPlacementKind ==
            wafer::RankWorkerPlacementKind::Unplaced)
      freshSingleOrdinals.insert(candidate.stableOrdinal);
  // Baseline Direct/Forward are followed by the actual partial-reduction
  // Direct/Forward generations in stable seed-attempt order.
  ASSERT_EQ(freshSingleOrdinals.size(), 4u);
  for (int64_t ordinal : {firstFreshOrdinal + 2, firstFreshOrdinal + 3})
    for (const auto &frontier : actual->frontiers) {
      auto composed = llvm::find_if(
          frontier,
          [&](const wafer::compiler::detail::RankVariantCandidate &candidate) {
            return candidate.stableOrdinal == ordinal &&
                   candidate.artifactKind ==
                       wafer::RankArtifactKind::Resident &&
                   candidate.bufferingKind ==
                       wafer::RankBufferingKind::Single &&
                   candidate.workerPlacementKind ==
                       wafer::RankWorkerPlacementKind::Unplaced;
          });
      ASSERT_NE(composed, frontier.end());
      EXPECT_EQ(getComputeIssueFingerprint(*composed->module),
                actual->interfaceComputeIssues);
      EXPECT_EQ(countLoops(*composed->module), actual->interfaceLoopCount);
      EXPECT_EQ(countOutputWrites(*composed->module),
                actual->interfaceOutputWriteCount);
      EXPECT_GT(countPeerIssues(*composed->module), 0u);
    }
}

TEST_F(NoCResidentDataflowTest,
       BuildsPartialReductionCompoundButRejectsUnrolledFixedSlotMetadata) {
  std::optional<ActualTraversalFrontiers> actual =
      buildActualTraversalFrontiers(
          R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%input: tensor<8x1xf32>) -> tensor<8xf32> {
    %out = tensor.empty() : tensor<8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32)
        outs(%out : tensor<8xf32>) -> tensor<8xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
        } ins(%input : tensor<8x1xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir",
          kRefinedPartialReductionInterfaceStableOrdinal,
          /*duplicateBaselineLoad=*/false,
          wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
          /*requireWorkerPlacedInterface=*/true);
  ASSERT_TRUE(actual);

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  wafer::frontend::FrontendProgramVerificationResult frontendProgram =
      replicatedProgram(/*inputShape=*/{8, 1}, /*outputShape=*/{8});
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          actual->frontiers, frontendProgram, *config, &failure)))
      << failure;

  auto combinedIt =
      llvm::find_if(actual->frontiers.front(), [](const Candidate &candidate) {
        return candidate.module &&
               candidate.artifactKind == wafer::RankArtifactKind::Resident &&
               candidate.bufferingKind ==
                   wafer::RankBufferingKind::StaticFixedSlot &&
               candidate.bufferingPlanOrdinal != 0 &&
               candidate.workerPlacementKind ==
                   wafer::RankWorkerPlacementKind::DisjointComponents &&
               candidate.workerPlacementPlanOrdinal != 0 &&
               countPeerIssues(*candidate.module) != 0;
      });
  std::string candidateInventory;
  llvm::raw_string_ostream inventory(candidateInventory);
  for (const Candidate &candidate : actual->frontiers.front()) {
    bool hasNonzeroWorker = false;
    candidate.module.get()->walk([&](mlir::Operation *operation) {
      std::optional<wafer::NCCWorker> worker =
          wafer::getNCCIssueWorker(operation);
      hasNonzeroWorker |= worker && *worker != wafer::NCCWorker::Worker0;
    });
    inventory << "\n  ordinal=" << candidate.stableOrdinal
              << " artifact=" << static_cast<unsigned>(candidate.artifactKind)
              << " buffering=" << static_cast<unsigned>(candidate.bufferingKind)
              << "/" << candidate.bufferingPlanOrdinal << " worker="
              << static_cast<unsigned>(candidate.workerPlacementKind) << "/"
              << candidate.workerPlacementPlanOrdinal
              << " nonzero=" << hasNonzeroWorker
              << " loops=" << countLoops(*candidate.module)
              << " peers=" << countPeerIssues(*candidate.module);
    if (candidate.bufferingKind == wafer::RankBufferingKind::Single &&
        candidate.workerPlacementKind ==
            wafer::RankWorkerPlacementKind::DisjointComponents &&
        countPeerIssues(*candidate.module) != 0) {
      mlir::OwningOpRef<mlir::ModuleOp> diagnosticClone =
          mlir::cast<mlir::ModuleOp>(candidate.module.get()->clone());
      wafer::clearRankCandidatePhysicalFacts(*diagnosticClone);
      llvm::SmallVector<mlir::scf::ForOp, 4> loops;
      diagnosticClone->walk(
          [&](mlir::scf::ForOp loop) { loops.push_back(loop); });
      for (auto [index, loop] : llvm::enumerate(loops)) {
        std::string fixedFailure;
        auto fixed = wafer::deriveStaticFixedSlotPipelineCandidate(
            *diagnosticClone, loop, &fixedFailure);
        inventory << " fixed[" << index << "]="
                  << (mlir::succeeded(fixed) ? "accepted" : fixedFailure);
      }
    }
  }
  ASSERT_NE(combinedIt, actual->frontiers.front().end()) << candidateInventory;
  const int64_t stableOrdinal = combinedIt->stableOrdinal;
  const wafer::RankArtifactKind artifactKind = combinedIt->artifactKind;
  const uint32_t bufferingPlanOrdinal = combinedIt->bufferingPlanOrdinal;
  const uint32_t workerPlacementPlanOrdinal =
      combinedIt->workerPlacementPlanOrdinal;

  auto matchesCombined = [&](const Candidate &candidate) {
    return candidate.stableOrdinal == stableOrdinal &&
           candidate.artifactKind == artifactKind &&
           candidate.bufferingKind ==
               wafer::RankBufferingKind::StaticFixedSlot &&
           candidate.bufferingPlanOrdinal == bufferingPlanOrdinal &&
           candidate.workerPlacementKind ==
               wafer::RankWorkerPlacementKind::DisjointComponents &&
           candidate.workerPlacementPlanOrdinal == workerPlacementPlanOrdinal;
  };
  auto matchesWorkerOnly = [&](const Candidate &candidate) {
    return candidate.stableOrdinal == stableOrdinal &&
           candidate.artifactKind == artifactKind &&
           candidate.bufferingKind == wafer::RankBufferingKind::Single &&
           candidate.bufferingPlanOrdinal == 0 &&
           candidate.workerPlacementKind ==
               wafer::RankWorkerPlacementKind::DisjointComponents &&
           candidate.workerPlacementPlanOrdinal == workerPlacementPlanOrdinal &&
           candidate.module && countPeerIssues(*candidate.module) != 0;
  };
  for (Frontier &frontier : actual->frontiers) {
    auto candidate = llvm::find_if(frontier, matchesCombined);
    ASSERT_NE(candidate, frontier.end());
    ASSERT_NE(llvm::find_if(frontier, matchesWorkerOnly), frontier.end());
    auto evidenceModule =
        mlir::cast<mlir::ModuleOp>(candidate->module.get()->clone());
    wafer::compiler::RankExecutable evidenceRank =
        wafer::compiler::ExecutableBundleBuilder::makeRank(
            /*logicalRank=*/0, std::move(evidenceModule), "main",
            /*programBindings=*/{},
            wafer::compiler::TransportContract::DirectDTE);
    llvm::Error evidence =
        wafer::compiler::detail::verifyStaticFixedSlotQualificationEvidence(
            evidenceRank);
    ASSERT_TRUE(static_cast<bool>(evidence));
    std::string evidenceText = llvm::toString(std::move(evidence));
    EXPECT_NE(evidenceText.find("requires one positive static rotating SPM "
                                "loop with current effect/issue consumption"),
              std::string::npos)
        << evidenceText;
    PeerTransportCounts peerTransport = countPeerTransport(*candidate->module);
    EXPECT_GT(peerTransport.sends + peerTransport.recvs, 0u);
    std::set<uint32_t> workers;
    candidate->module.get()->walk([&](mlir::Operation *operation) {
      if (std::optional<wafer::NCCWorker> worker =
              wafer::getNCCIssueWorker(operation))
        workers.insert(static_cast<uint32_t>(*worker));
    });
    EXPECT_GE(workers.size(), 2u);
    EXPECT_TRUE(
        llvm::any_of(workers, [](uint32_t worker) { return worker != 0; }));
    llvm::erase_if(frontier, [&](const Candidate &slot) {
      return !slot.reservedBaseline && !matchesCombined(slot) &&
             !matchesWorkerOnly(slot);
    });
    ASSERT_EQ(frontier.size(), 3u);
  }

  // The small fixture is fully unrolled after fixed-slot derivation, so its
  // metadata cannot substitute for a current-IR rotating-loop witness. The
  // larger actual tiled compound test covers the fully accepted path.
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      actual->frontiers, frontendProgram, *config, diagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::
          QualifyNoCResidentFixedSlotWorker);
  EXPECT_TRUE(mlir::failed(accepted));
  EXPECT_NE(diagnosticText.find("gate=static-fixed-slot-qualification"),
            std::string::npos)
      << diagnosticText;
}

TEST(NoCResidentProductionTest,
     DefaultWholeVariantKeepsSmallReplicatedInputBaseline) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
    %out = tensor.empty() : tensor<4xf32>
    %squared = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      %product = arith.mulf %value, %value : f32
      linalg.yield %product : f32
    } -> tensor<4xf32>
    return %squared : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(tensorProgram);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 1;
  wafer::frontend::ProgramBoundaryBinding input;
  input.index = 0;
  input.programIndex = 0;
  input.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  input.globalShape = {4};
  input.localShape = {4};
  input.dtype = "f32";
  for (int64_t rank = 0; rank < 16; ++rank) {
    wafer::frontend::ProgramRankSlice slice;
    slice.logicalRank = rank;
    slice.replicaId = rank;
    slice.offsets = {0};
    slice.sizes = {4};
    slice.strides = {1};
    input.rankSlices.push_back(std::move(slice));
  }
  program.distributedInputs.push_back(input);
  auto output = input;
  output.distribution = wafer::frontend::ProgramDistributionKind::Partitioned;
  output.globalShape = {64};
  for (int64_t rank = 0; rank < 16; ++rank) {
    output.rankSlices[static_cast<size_t>(rank)].replicaId = 0;
    output.rankSlices[static_cast<size_t>(rank)].offsets = {rank * 4};
  }
  // Keep this source-to-package proof specific to replicated input fan-out.
  // Distinct required output slices cannot be replaced by output publication.
  program.distributedOutputs.push_back(std::move(output));
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV3(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  llvm::Expected<wafer::compiler::ExecutableBundle> bundle =
      wafer::compiler::detail::buildExecutableBundle(
          context, *tensorProgram, std::move(program), *config, diagnostics,
          std::nullopt);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnosticText << llvm::toString(bundle.takeError());
  tensorProgram = nullptr;

  unsigned loads = 0;
  unsigned inputSends = 0;
  unsigned inputRecvs = 0;
  unsigned outputSends = 0;
  unsigned outputRecvs = 0;
  unsigned stores = 0;
  unsigned waits = 0;
  for (const wafer::compiler::RankExecutable &rank :
       bundle->getRankExecutables()) {
    rank.getModule().walk([&](wafer::InstrRDMAOp) { ++loads; });
    rank.getModule().walk([&](wafer::InstrWDMAOp) { ++stores; });
    rank.getModule().walk([&](wafer::InstrDTESendOp send) {
      ASSERT_EQ(send.getMessage().getPhase(),
                wafer::DTEProtocolPhase::PeerDataflow);
      if (send.getMessage().getRound() == 0)
        ++inputSends;
      else if (send.getMessage().getRound() == 2)
        ++outputSends;
      else
        ADD_FAILURE() << "unexpected peer-dataflow send round";
    });
    rank.getModule().walk([&](wafer::InstrDTERecvOp recv) {
      ASSERT_EQ(recv.getMessage().getPhase(),
                wafer::DTEProtocolPhase::PeerDataflow);
      if (recv.getMessage().getRound() == 0)
        ++inputRecvs;
      else if (recv.getMessage().getRound() == 2)
        ++outputRecvs;
      else
        ADD_FAILURE() << "unexpected peer-dataflow receive round";
    });
    rank.getModule().walk([&](wafer::InstrDTEWaitOp) { ++waits; });
  }
  EXPECT_EQ(loads, 16u);
  EXPECT_EQ(inputSends, 0u);
  EXPECT_EQ(inputRecvs, 0u);
  EXPECT_EQ(outputSends, 0u);
  EXPECT_EQ(outputRecvs, 0u);
  // Every partitioned output slice remains locally produced and published.
  EXPECT_EQ(stores, 16u);
  EXPECT_EQ(waits, 0u);
  EXPECT_EQ(bundle->getRuntimeLaunchContract().getPhases().size(), 1u);
}

TEST(NoCResidentProductionTest,
     PartitionedInputPreservesOneDDRLoadPerUniqueShardThroughLateGates) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
    %out = tensor.empty() : tensor<4xf32>
    %squared = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      %product = arith.mulf %value, %value : f32
      linalg.yield %product : f32
    } -> tensor<4xf32>
    return %squared : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(tensorProgram);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 1;
  wafer::frontend::ProgramBoundaryBinding boundary;
  boundary.index = 0;
  boundary.programIndex = 0;
  boundary.distribution = wafer::frontend::ProgramDistributionKind::Partitioned;
  boundary.globalShape = {64};
  boundary.localShape = {4};
  boundary.dtype = "f32";
  for (int64_t rank = 0; rank < 16; ++rank) {
    wafer::frontend::ProgramRankSlice slice;
    slice.logicalRank = rank;
    slice.replicaId = 0;
    slice.offsets = {rank * 4};
    slice.sizes = {4};
    slice.strides = {1};
    boundary.rankSlices.push_back(std::move(slice));
  }
  program.distributedInputs.push_back(boundary);
  program.distributedOutputs.push_back(std::move(boundary));
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  llvm::Expected<wafer::compiler::ExecutableBundle> bundle =
      wafer::compiler::detail::buildExecutableBundle(
          context, *tensorProgram, std::move(program), *config, diagnostics,
          std::nullopt);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnosticText << llvm::toString(bundle.takeError());
  tensorProgram = nullptr;

  unsigned loads = 0;
  unsigned peerIssues = 0;
  for (const wafer::compiler::RankExecutable &rank :
       bundle->getRankExecutables()) {
    rank.getModule().walk([&](wafer::InstrRDMAOp) { ++loads; });
    rank.getModule().walk([&](mlir::Operation *operation) {
      peerIssues +=
          mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTERecvOp>(operation)
              ? 1u
              : 0u;
    });
  }
  EXPECT_EQ(loads, 16u);
  EXPECT_EQ(peerIssues, 0u);
}

} // namespace
