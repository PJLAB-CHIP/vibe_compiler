//===- CompilationStages.cpp - Stage verification and registry ----------===//

#include "CompilationInternal.h"
#include "ExecutableBundleInternal.h"

#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace wafer::compiler::detail {

namespace {

mlir::OwningOpRef<mlir::ModuleOp> parseModule(llvm::StringRef filename,
                                              mlir::MLIRContext &context) {
  mlir::ParserConfig config(&context);
  return mlir::parseSourceFile<mlir::ModuleOp>(filename, config);
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
parseProgramDirectoryModule(llvm::StringRef programDirectory,
                            mlir::MLIRContext &context) {
  std::string modulePath =
      programFile(programDirectory, {llvm::StringRef("functions"),
                                     llvm::StringRef("forward.mlir")});
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(modulePath, context);
  if (module && mlir::succeeded(mlir::verify(*module)))
    return module;
  return {};
}

mlir::LogicalResult verifyProgramDirectoryMetadata(
    mlir::ModuleOp module, llvm::StringRef programDirectory,
    llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result) {
  std::string verifierDiagnostics;
  llvm::raw_string_ostream verifierStream(verifierDiagnostics);
  mlir::LogicalResult status = wafer::frontend::verifyProgramDirectory(
      module, programDirectory, verifierStream, result);
  if (mlir::succeeded(status))
    return status;

  verifierStream.flush();
  llvm::StringRef remaining(verifierDiagnostics);
  while (!remaining.empty()) {
    auto [line, rest] = remaining.split('\n');
    if (!line.empty())
      reject(diagnostics, line);
    remaining = rest;
  }
  return mlir::failure();
}

bool hasPostSpmdMarker(mlir::ModuleOp module,
                       llvm::StringRef programDirectory) {
  if (isRegularFile(
          programFile(programDirectory,
                      {llvm::StringRef("functions"),
                       llvm::StringRef("forward.parameter_shards.json")})))
    return true;

  if (module->getAttr("mhlo.spmd_parameters_shardings"))
    return true;
  bool found = false;
  module.walk([&](mlir::Operation *operation) {
    if (operation->getAttr("mhlo.spmd_parameters_shardings")) {
      found = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

namespace {

bool containsDialectAttribute(mlir::Attribute attribute,
                              llvm::StringRef dialectNamespace);

bool containsDialectType(mlir::Type type, llvm::StringRef dialectNamespace) {
  if (type.getDialect().getNamespace() == dialectNamespace)
    return true;
  if (auto rankedTensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    if (mlir::Attribute encoding = rankedTensor.getEncoding())
      if (containsDialectAttribute(encoding, dialectNamespace))
        return true;
  }
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
    if (containsDialectAttribute(memref.getLayout(), dialectNamespace) ||
        containsDialectAttribute(memref.getMemorySpace(), dialectNamespace))
      return true;
  }
  if (auto unrankedMemref = mlir::dyn_cast<mlir::UnrankedMemRefType>(type))
    if (containsDialectAttribute(unrankedMemref.getMemorySpace(),
                                 dialectNamespace))
      return true;
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    return containsDialectType(shaped.getElementType(), dialectNamespace);
  if (auto function = mlir::dyn_cast<mlir::FunctionType>(type)) {
    return llvm::any_of(function.getInputs(),
                        [&](mlir::Type nested) {
                          return containsDialectType(nested, dialectNamespace);
                        }) ||
           llvm::any_of(function.getResults(), [&](mlir::Type nested) {
             return containsDialectType(nested, dialectNamespace);
           });
  }
  if (auto tuple = mlir::dyn_cast<mlir::TupleType>(type))
    return llvm::any_of(tuple.getTypes(), [&](mlir::Type nested) {
      return containsDialectType(nested, dialectNamespace);
    });
  return false;
}

bool containsDialectAttribute(mlir::Attribute attribute,
                              llvm::StringRef dialectNamespace) {
  if (attribute.getDialect().getNamespace() == dialectNamespace)
    return true;
  if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attribute))
    return llvm::any_of(array, [&](mlir::Attribute nested) {
      return containsDialectAttribute(nested, dialectNamespace);
    });
  if (auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(attribute))
    return llvm::any_of(dictionary, [&](mlir::NamedAttribute nested) {
      llvm::StringRef name = nested.getName().getValue();
      return (name.consume_front(dialectNamespace) && name.starts_with('.')) ||
             containsDialectAttribute(nested.getValue(), dialectNamespace);
    });
  if (auto type = mlir::dyn_cast<mlir::TypeAttr>(attribute))
    return containsDialectType(type.getValue(), dialectNamespace);
  if (auto typed = mlir::dyn_cast<mlir::TypedAttr>(attribute))
    return containsDialectType(typed.getType(), dialectNamespace);
  return false;
}

} // namespace

bool containsDialectSemantics(mlir::ModuleOp module,
                              llvm::StringRef dialectNamespace) {
  bool found = false;
  module.walk([&](mlir::Operation *operation) {
    if (operation->getName().getDialectNamespace() == dialectNamespace) {
      found = true;
      return mlir::WalkResult::interrupt();
    }
    if (llvm::any_of(operation->getAttrs(),
                     [&](mlir::NamedAttribute attr) {
                       llvm::StringRef name = attr.getName().getValue();
                       return (name.consume_front(dialectNamespace) &&
                               name.starts_with('.')) ||
                              containsDialectAttribute(attr.getValue(),
                                                       dialectNamespace);
                     }) ||
        llvm::any_of(operation->getOperandTypes(),
                     [&](mlir::Type type) {
                       return containsDialectType(type, dialectNamespace);
                     }) ||
        llvm::any_of(operation->getResultTypes(), [&](mlir::Type type) {
          return containsDialectType(type, dialectNamespace);
        })) {
      found = true;
      return mlir::WalkResult::interrupt();
    }
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument argument : block.getArguments())
          if (containsDialectType(argument.getType(), dialectNamespace)) {
            found = true;
            return mlir::WalkResult::interrupt();
          }
    return mlir::WalkResult::advance();
  });
  return found;
}

mlir::LogicalResult verifyStablehloStageOperations(mlir::ModuleOp module) {
  mlir::Operation *illegal = nullptr;
  module.walk([&](mlir::Operation *operation) {
    llvm::StringRef dialect = operation->getName().getDialectNamespace();
    bool allowed =
        dialect == "builtin" || dialect == "func" || dialect == "stablehlo" ||
        mlir::isa<wafer::TargetTopologyOp, wafer::ExecutionMeshOp>(operation);
    if (!allowed) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!illegal)
    return mlir::success();
  return illegal->emitOpError(
      "is not legal in an exporter or post-SPMD StableHLO program");
}

namespace {

bool isTensorProgramStageWaferOperation(mlir::Operation *operation) {
  return mlir::isa<wafer::TargetTopologyOp, wafer::ExecutionMeshOp,
                   wafer::LinalgExtCollectiveYieldOp>(operation) ||
         mlir::isa<wafer::WaferLinalgExtCollectiveOpInterface>(operation);
}

} // namespace

mlir::LogicalResult verifyTensorProgramStageOperations(mlir::ModuleOp module) {
  mlir::Operation *illegal = nullptr;
  module.walk([&](mlir::Operation *operation) {
    llvm::StringRef dialect = operation->getName().getDialectNamespace();
    bool allowed =
        dialect == "builtin" || dialect == "func" || dialect == "arith" ||
        dialect == "math" || dialect == "tensor" || dialect == "linalg" ||
        dialect == "scf" || dialect == "cf" ||
        (dialect == "wafer" && isTensorProgramStageWaferOperation(operation));
    if (!allowed) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!illegal)
    return mlir::success();
  return illegal->emitOpError(
      "is not legal in a verified structured tensor-program artifact");
}

namespace {

bool hasExactSingleCardTopology(wafer::TargetTopologyOp topology) {
  llvm::ArrayRef<int64_t> cardGrid = topology.getCardGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> tileGrid = topology.getTileGridAttr().asArrayRef();
  return topology.getSymName() == "default" && cardGrid.size() == 2 &&
         cardGrid[0] == 1 && cardGrid[1] == 1 && tileGrid.size() == 2 &&
         tileGrid[0] == 4 && tileGrid[1] == 4 &&
         topology.getCardInterconnectAttr().getValue() == "mesh" &&
         topology.getUnavailableTilesAttr().empty();
}

bool hasExactExecutionMesh(wafer::ExecutionMeshOp mesh,
                           const ExecutionConfig &config) {
  if (mesh.getSymName() != "default_mesh" ||
      mesh.getTopologyAttr().getValue() != "default")
    return false;

  mlir::ArrayAttr axes = mesh.getAxesAttr();
  llvm::ArrayRef<int64_t> shape = mesh.getShapeAttr().asArrayRef();
  if (axes.size() != 1 ||
      mlir::cast<mlir::StringAttr>(axes[0]).getValue() != "rank" ||
      shape.size() != 1 || shape[0] != config.getRankCount())
    return false;

  llvm::StringRef policy = mesh.getPolicyAttr().getValue();
  llvm::ArrayRef<int64_t> endpoints = mesh.getEndpointsAttr().asArrayRef();
  if (config.getRankCount() == 16)
    return policy == "all_available" && endpoints.empty();
  return policy == "explicit" && endpoints.size() == 4 && endpoints[0] == 0 &&
         endpoints[1] == 0 && endpoints[2] == 0 && endpoints[3] == 0;
}

mlir::LogicalResult collectExecutionFacts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<wafer::TargetTopologyOp> &topologies,
    llvm::SmallVectorImpl<wafer::ExecutionMeshOp> &meshes) {
  topologies.clear();
  meshes.clear();
  for (wafer::TargetTopologyOp operation :
       module.getOps<wafer::TargetTopologyOp>())
    topologies.push_back(operation);
  for (wafer::ExecutionMeshOp operation :
       module.getOps<wafer::ExecutionMeshOp>())
    meshes.push_back(operation);

  bool hasNestedExecutionFacts = false;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::TargetTopologyOp, wafer::ExecutionMeshOp>(operation) &&
        operation->getParentOp() != module.getOperation()) {
      hasNestedExecutionFacts = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (hasNestedExecutionFacts)
    return module.emitOpError(
        "target topology and execution mesh must be direct module members");
  return mlir::success();
}

} // namespace

mlir::LogicalResult
verifyExactExecutionConfigInternal(mlir::ModuleOp module,
                                   const ExecutionConfig &config) {
  llvm::SmallVector<wafer::TargetTopologyOp, 2> topologies;
  llvm::SmallVector<wafer::ExecutionMeshOp, 2> meshes;
  if (mlir::failed(collectExecutionFacts(module, topologies, meshes)))
    return mlir::failure();

  if (topologies.size() != 1)
    return module.emitOpError(
        "typed compilation requires exactly one target topology");
  if (meshes.size() != 1)
    return module.emitOpError(
        "typed compilation requires exactly one execution mesh");
  if (!hasExactSingleCardTopology(topologies.front()))
    return module.emitOpError(
        "target topology does not match the single-card 1x1-card/4x4-tile "
        "ExecutionConfig");
  if (!hasExactExecutionMesh(meshes.front(), config))
    return module.emitOpError(
        "execution mesh does not exactly match the requested execution-ranks");
  return mlir::success();
}

mlir::LogicalResult
materializeOrVerifyExactExecutionConfig(mlir::ModuleOp module,
                                        const ExecutionConfig &config) {
  llvm::SmallVector<wafer::TargetTopologyOp, 2> topologies;
  llvm::SmallVector<wafer::ExecutionMeshOp, 2> meshes;
  if (mlir::failed(collectExecutionFacts(module, topologies, meshes)))
    return mlir::failure();

  if (topologies.size() > 1)
    return module.emitOpError(
        "typed compilation requires exactly one target topology");
  if (meshes.size() > 1)
    return module.emitOpError(
        "typed compilation requires exactly one execution mesh");
  if (topologies.empty() && !meshes.empty())
    return module.emitOpError(
        "execution mesh cannot precede its target topology");

  if (topologies.empty()) {
    mlir::PassManager manager(module.getContext());
    wafer::support::attachCompileTiming(manager, "execution-config");
    manager.addPass(wafer::createMaterializeTargetTopologyPass());
    if (mlir::failed(manager.run(module)))
      return mlir::failure();
  }

  if (meshes.empty()) {
    wafer::MaterializeExecutionMeshPassOptions options;
    options.topologyName = "default";
    if (config.getRankCount() == 1) {
      options.policy = "explicit";
      options.shape = "1";
      options.endpoints = "0,0,0,0";
    }
    mlir::PassManager manager(module.getContext());
    wafer::support::attachCompileTiming(manager, "execution-config");
    manager.addPass(wafer::createMaterializeExecutionMeshPass(options));
    if (mlir::failed(manager.run(module)))
      return mlir::failure();
  }
  return verifyExactExecutionConfigInternal(module, config);
}

void eraseTargetTopologyAndExecutionMesh(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *> operations;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::TargetTopologyOp, wafer::ExecutionMeshOp>(operation))
      operations.push_back(operation);
  });
  for (mlir::Operation *operation : operations)
    operation->erase();
}

void registerCompilationDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  wafer::registerImporterDialects(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
}

bool runPassPipeline(mlir::ModuleOp module,
                     void (*builder)(mlir::OpPassManager &)) {
  mlir::PassManager manager(module.getContext());
  wafer::support::attachCompileTiming(manager, "stablehlo-to-linalg");
  builder(manager);
  return mlir::failed(manager.run(module));
}

mlir::LogicalResult
verifyExactExecutionConfig(mlir::ModuleOp module,
                           const ExecutionConfig &executionConfig) {
  return verifyExactExecutionConfigInternal(module, executionConfig);
}

} // namespace wafer::compiler::detail
