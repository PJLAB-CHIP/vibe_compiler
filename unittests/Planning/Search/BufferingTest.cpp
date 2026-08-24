//===- BufferingTest.cpp ---------------------------------------------===//

#include "Wafer/Planning/Search/Buffering.h"

#include "TestSupport/Planning/SpatialDemandTestSupport.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <set>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  registerWaferCoreDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

mlir::OwningOpRef<mlir::ModuleOp> parseChain(mlir::MLIRContext &context,
                                             int64_t extent) {
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n"
        "  wafer.target.topology @target\n"
        "      {card_grid = array<i64: 1, 1>, card_interconnect = \"mesh\",\n"
        "       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}\n"
        "  wafer.execution.mesh @logical {axes = [\"card\"], shape = "
        "array<i64: 1>}\n"
        "  func.func @chain(%input: tensor<"
     << extent << "xf16>) -> tensor<" << extent
     << "xf16> {\n"
        "    %e0 = tensor.empty() : tensor<"
     << extent
     << "xf16>\n"
        "    %producer = linalg.map ins(%input : tensor<"
     << extent << "xf16>) outs(%e0 : tensor<" << extent
     << "xf16>) (%v: f16) {\n"
        "      %x = arith.addf %v, %v : f16\n"
        "      linalg.yield %x : f16\n"
        "    }\n"
        "    %e1 = tensor.empty() : tensor<"
     << extent
     << "xf16>\n"
        "    %consumer = linalg.map ins(%producer : tensor<"
     << extent << "xf16>) outs(%e1 : tensor<" << extent
     << "xf16>) (%v: f16) {\n"
        "      %x = arith.mulf %v, %v : f16\n"
        "      linalg.yield %x : f16\n"
        "    }\n"
        "    return %consumer : tensor<"
     << extent << "xf16>\n  }\n}\n";
  os.flush();
  return mlir::parseSourceString<mlir::ModuleOp>(source,
                                                 mlir::ParserConfig(&context));
}

struct Prepared {
  std::unique_ptr<CardProgramAnalysis> program;
  SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
  CoupledRegionDomain coupledDomain;
  CoupledRegionAssignment coupledAssignment;
  CardTemporalDomain temporalDomain;
  CardTemporalAssignment temporalAssignment;
  CardPhysicalRepresentationDomain representationDomain;
  CardPhysicalRepresentationAssignment representationAssignment;
  CardDataMovementDomain movementDomain;
  CardDataMovementAssignment movementAssignment;
};

mlir::FailureOr<Prepared> prepare(mlir::ModuleOp module, int64_t extent,
                                  std::string *failureReason) {
  auto function = *module.getOps<mlir::func::FuncOp>().begin();
  auto dag = StructuredDAGAnalysis::create(function, failureReason);
  if (mlir::failed(dag) || dag->getNodes().size() != 2)
    return mlir::failure();
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements{
      StructuredDAGNodePlacement{0, {1}, {TileId(0)}},
      StructuredDAGNodePlacement{1, {1}, {TileId(0)}}};
  auto spatialDemand =
      wafer::test::buildTestSpatialDemand(*dag, placements, failureReason);
  if (mlir::failed(spatialDemand))
    return mlir::failure();
  auto coupled = CoupledRegionDomain::create(
      *dag, spatialDemand->spatial, spatialDemand->demand, failureReason);
  auto temporal = CardTemporalDomain::create(*dag, placements, failureReason);
  auto topology = TargetTopology::create(module, failureReason);
  if (mlir::failed(coupled) || mlir::failed(temporal) || mlir::failed(topology))
    return mlir::failure();
  CoupledRegionAssignment coupledAssignment = coupled->getFirstAssignment();
  while (true) {
    auto next = coupled->getNextAssignment(coupledAssignment);
    if (mlir::failed(next))
      return mlir::failure();
    if (!*next)
      break;
    coupledAssignment = std::move(**next);
  }
  CardTemporalAssignment temporalAssignment = temporal->getFirstAssignment();
  for (TemporalNodeAssignment &node : temporalAssignment.nodes) {
    node.iteratorTileSizes = {2};
    node.waveLoopOrder = {0};
  }
  if (!temporal->contains(temporalAssignment))
    return mlir::failure();
  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
  for (const StructuredDAGNode &node : dag->getNodes())
    operationNodes.push_back({node.operation, node.id});
  StaticOutputDomains outputDomains{{extent}};
  auto program = std::make_unique<CardProgramAnalysis>(
      std::move(*topology), llvm::SmallVector<TileId, 16>{TileId(0)},
      std::move(*dag), std::move(outputDomains), std::move(operationNodes));
  auto representation = CardPhysicalRepresentationDomain::create(
      *program, spatialDemand->spatial, spatialDemand->demand, *coupled,
      coupledAssignment, *temporal, temporalAssignment, failureReason);
  if (mlir::failed(representation))
    return mlir::failure();
  CardPhysicalRepresentationAssignment representationAssignment =
      representation->getFirstAssignment();
  auto movement = CardDataMovementDomain::create(
      *program, CardId(0), spatialDemand->spatial, spatialDemand->demand,
      *coupled, coupledAssignment, *temporal, temporalAssignment,
      *representation, representationAssignment, failureReason);
  if (mlir::failed(movement))
    return mlir::failure();
  CardDataMovementAssignment movementAssignment =
      movement->getFirstAssignment();
  if (movementAssignment.edges.size() != 1 ||
      movementAssignment.edges.front().kind != DataMovementKind::Retained)
    return mlir::failure();
  return Prepared{std::move(program),
                  std::move(spatialDemand->spatial),
                  std::move(spatialDemand->demand),
                  std::move(*coupled),
                  std::move(coupledAssignment),
                  std::move(*temporal),
                  std::move(temporalAssignment),
                  std::move(*representation),
                  std::move(representationAssignment),
                  std::move(*movement),
                  std::move(movementAssignment)};
}

CardBufferingDomain makeDomain(const Prepared &prepared) {
  auto domain = CardBufferingDomain::create(
      *prepared.program, prepared.spatial, prepared.demand,
      prepared.coupledDomain, prepared.coupledAssignment,
      prepared.temporalDomain, prepared.temporalAssignment,
      prepared.representationDomain, prepared.representationAssignment,
      prepared.movementDomain, prepared.movementAssignment);
  EXPECT_TRUE(mlir::succeeded(domain));
  return std::move(*domain);
}

std::set<uint32_t> enumerateSlotCounts(
    const CardBufferingDomain &domain,
    std::optional<CardBufferingAssignment> *fourSlot = nullptr) {
  std::set<uint32_t> result;
  CardBufferingAssignment current = domain.getFirstAssignment();
  while (true) {
    EXPECT_TRUE(domain.contains(current));
    EXPECT_EQ(current.groups.size(), 1u);
    result.insert(current.groups.front().slotCount);
    if (current.groups.front().slotCount == 4 && fourSlot)
      *fourSlot = current;
    auto next = domain.getNextAssignment(current);
    EXPECT_TRUE(mlir::succeeded(next));
    if (mlir::failed(next) || !*next)
      break;
    current = std::move(**next);
  }
  return result;
}

TEST(BufferingTest, EnumeratesEveryWaveBoundedSlotCount) {
  auto context = createContext();
  auto module = parseChain(*context, 12);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto prepared = prepare(*module, 12, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  CardBufferingDomain domain = makeDomain(*prepared);
  std::optional<CardBufferingAssignment> fourSlot;
  EXPECT_EQ(enumerateSlotCounts(domain, &fourSlot),
            (std::set<uint32_t>{1, 2, 3, 4, 5}));
  ASSERT_TRUE(fourSlot);
}

TEST(BufferingTest, TemporalStructureAloneDefinesTheFiniteSlotDomain) {
  auto context = createContext();
  std::string failureReason;
  auto fullModule = parseChain(*context, 12);
  ASSERT_TRUE(fullModule);
  auto full = prepare(*fullModule, 12, &failureReason);
  ASSERT_TRUE(mlir::succeeded(full)) << failureReason;
  EXPECT_EQ(enumerateSlotCounts(makeDomain(*full)),
            (std::set<uint32_t>{1, 2, 3, 4, 5}));

  auto tailModule = parseChain(*context, 11);
  ASSERT_TRUE(tailModule);
  auto tail = prepare(*tailModule, 11, &failureReason);
  ASSERT_TRUE(mlir::succeeded(tail)) << failureReason;
  EXPECT_EQ(enumerateSlotCounts(makeDomain(*tail)),
            (std::set<uint32_t>{1, 2, 3, 4}));
}

} // namespace
