//===- Compilation.cpp - Typed Wafer compiler orchestration --------------===//

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Compiler/Testing.h"

#include "ExecutableBundleInternal.h"
#include "TargetArtifactInternal.h"

#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/Frontend/Program.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#ifdef __linux__
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace wafer::compiler {

llvm::Expected<ExecutionConfig>
ExecutionConfig::createForSingleCard(int64_t executionRankCount) {
  if (executionRankCount != 1 && executionRankCount != 16)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "execution-ranks must be exactly 1 or 16 for the single-card compiler");
  return ExecutionConfig(executionRankCount);
}

llvm::Expected<CompilationRequest>
CompilationRequest::create(llvm::StringRef sourceProgramDirectory,
                           ExecutionConfig executionConfig) {
  if (sourceProgramDirectory.empty())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "source program directory must not be empty");
  return CompilationRequest(sourceProgramDirectory, executionConfig);
}

namespace {

bool reject(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << "wafer-compile: " << message << "\n";
  return true;
}

bool pathEntryExists(llvm::StringRef path) {
  llvm::sys::fs::file_type type =
      llvm::sys::fs::get_file_type(path, /*Follow=*/false);
  return type != llvm::sys::fs::file_type::file_not_found &&
         type != llvm::sys::fs::file_type::status_error;
}

bool isDirectory(llvm::StringRef path) {
  llvm::sys::fs::file_status status;
  return !llvm::sys::fs::status(path, status) &&
         llvm::sys::fs::is_directory(status);
}

bool isRegularFile(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::regular_file;
}

bool createDirectory(llvm::StringRef path, llvm::raw_ostream &diagnostics) {
  if (std::error_code error = llvm::sys::fs::create_directories(path))
    return reject(diagnostics, "failed to create directory '" + path.str() +
                                   "': " + error.message());
  return false;
}

std::string programFile(llvm::StringRef programDirectory,
                        llvm::ArrayRef<llvm::StringRef> components) {
  llvm::SmallString<256> path(programDirectory);
  for (llvm::StringRef component : components)
    llvm::sys::path::append(path, component);
  return path.str().str();
}

bool copyDirectory(llvm::StringRef sourceDirectory,
                   llvm::StringRef destinationDirectory,
                   llvm::raw_ostream &diagnostics) {
  if (!isDirectory(sourceDirectory))
    return reject(diagnostics,
                  "source program directory is not a directory: '" +
                      sourceDirectory.str() + "'");
  if (createDirectory(destinationDirectory, diagnostics))
    return true;

  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(sourceDirectory, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return reject(diagnostics, "failed to walk directory '" +
                                     sourceDirectory.str() +
                                     "': " + error.message());

    llvm::StringRef source = iterator->path();
    if (iterator->type() == llvm::sys::fs::file_type::symlink_file)
      return reject(diagnostics,
                    "program directory contains unsupported symbolic link: '" +
                        source.str() + "'");
    llvm::StringRef relative = source;
    if (!relative.consume_front(sourceDirectory))
      return reject(diagnostics, "failed to derive source program member path");
    if (relative.starts_with(llvm::sys::path::get_separator()))
      relative = relative.drop_front();

    llvm::SmallString<256> destination(destinationDirectory);
    llvm::sys::path::append(destination, relative);

    llvm::sys::fs::file_status status;
    if (std::error_code statusError = llvm::sys::fs::status(source, status))
      return reject(diagnostics, "failed to stat '" + source.str() +
                                     "': " + statusError.message());

    if (llvm::sys::fs::is_directory(status)) {
      if (createDirectory(destination, diagnostics))
        return true;
      continue;
    }
    if (!llvm::sys::fs::is_regular_file(status))
      return reject(
          diagnostics,
          "program directory contains unsupported non-regular member: '" +
              source.str() + "'");

    llvm::SmallString<256> parent(destination);
    llvm::sys::path::remove_filename(parent);
    if (createDirectory(parent, diagnostics))
      return true;
    if (std::error_code copyError =
            llvm::sys::fs::copy_file(source, destination))
      return reject(diagnostics, "failed to copy '" + source.str() +
                                     "': " + copyError.message());
  }

  if (error)
    return reject(diagnostics, "failed to walk directory '" +
                                   sourceDirectory.str() +
                                   "': " + error.message());
  return false;
}

bool validateRegularDirectoryTree(llvm::StringRef root,
                                  llvm::raw_ostream &diagnostics) {
  if (llvm::sys::fs::get_file_type(root, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return reject(diagnostics, "helper output is not a real directory: '" +
                                   root.str() + "'");

  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(root, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return reject(diagnostics, "failed to walk helper output directory: " +
                                     error.message());
    llvm::sys::fs::file_type type = iterator->type();
    if (type != llvm::sys::fs::file_type::directory_file &&
        type != llvm::sys::fs::file_type::regular_file)
      return reject(diagnostics,
                    "helper output contains unsupported non-regular member: '" +
                        iterator->path() + "'");
  }
  if (error)
    return reject(diagnostics,
                  "failed to walk helper output directory: " + error.message());
  return false;
}

bool mergeMissingProgramMembers(llvm::StringRef sourceDirectory,
                                llvm::StringRef destinationDirectory,
                                llvm::raw_ostream &diagnostics) {
  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(sourceDirectory, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return reject(diagnostics, "failed to walk source program members: " +
                                     error.message());

    llvm::StringRef source = iterator->path();
    llvm::StringRef relative = source;
    if (!relative.consume_front(sourceDirectory))
      return reject(diagnostics, "failed to derive source program member path");
    if (relative.starts_with(llvm::sys::path::get_separator()))
      relative = relative.drop_front();

    llvm::SmallString<256> destination(destinationDirectory);
    llvm::sys::path::append(destination, relative);
    llvm::sys::fs::file_type sourceType = iterator->type();
    llvm::sys::fs::file_status destinationStatus;
    std::error_code destinationStatusError =
        llvm::sys::fs::status(destination, destinationStatus, /*follow=*/false);
    bool destinationMissing = false;
    if (destinationStatusError) {
      if (destinationStatusError == std::errc::no_such_file_or_directory) {
        destinationMissing = true;
      } else {
        return reject(diagnostics, "failed to inspect helper output member '" +
                                       relative.str() + "': " +
                                       destinationStatusError.message());
      }
    }
    llvm::sys::fs::file_type destinationType = destinationStatus.type();

    if (sourceType == llvm::sys::fs::file_type::directory_file) {
      if (destinationMissing) {
        if (createDirectory(destination, diagnostics))
          return true;
      } else if (destinationType != llvm::sys::fs::file_type::directory_file) {
        return reject(
            diagnostics,
            "helper output changes a source directory into a file: '" +
                relative.str() + "'");
      }
      continue;
    }
    if (sourceType != llvm::sys::fs::file_type::regular_file)
      return reject(
          diagnostics,
          "source program contains unsupported non-regular member: '" +
              source.str() + "'");
    if (!destinationMissing) {
      if (destinationType != llvm::sys::fs::file_type::regular_file)
        return reject(
            diagnostics,
            "helper output changes a source file into a directory: '" +
                relative.str() + "'");
      continue;
    }
    llvm::SmallString<256> parent(destination);
    llvm::sys::path::remove_filename(parent);
    if (createDirectory(parent, diagnostics))
      return true;
    if (std::error_code copyError =
            llvm::sys::fs::copy_file(source, destination))
      return reject(diagnostics, "failed to preserve source program member '" +
                                     relative.str() +
                                     "': " + copyError.message());
  }
  if (error)
    return reject(diagnostics,
                  "failed to walk source program members: " + error.message());
  return false;
}

bool writeProgramModule(mlir::ModuleOp module, llvm::StringRef programDirectory,
                        llvm::raw_ostream &diagnostics) {
  std::string functionsDirectory =
      programFile(programDirectory, {llvm::StringRef("functions")});
  if (createDirectory(functionsDirectory, diagnostics))
    return true;

  std::string modulePath =
      programFile(programDirectory, {llvm::StringRef("functions"),
                                     llvm::StringRef("forward.mlir")});
  std::error_code error;
  llvm::raw_fd_ostream stream(modulePath, error, llvm::sys::fs::OF_Text);
  if (error)
    return reject(diagnostics,
                  "failed to write '" + modulePath + "': " + error.message());
  module->print(stream);
  stream << "\n";
  stream.close();
  if (stream.has_error())
    return reject(diagnostics, "failed to close '" + modulePath + "'");
  return false;
}

mlir::OwningOpRef<mlir::ModuleOp> parseModule(llvm::StringRef filename,
                                              mlir::MLIRContext &context) {
  mlir::ParserConfig config(&context);
  return mlir::parseSourceFile<mlir::ModuleOp>(filename, config);
}

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
    wafer::frontend::FrontendProgramVerificationResult *result = nullptr) {
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

bool isGroupedStageWaferOperation(mlir::Operation *operation) {
  return mlir::isa<wafer::TargetTopologyOp, wafer::ExecutionMeshOp,
                   wafer::GroupOp, wafer::GroupYieldOp,
                   wafer::LinalgExtCollectiveYieldOp>(operation) ||
         mlir::isa<wafer::WaferLinalgExtCollectiveOpInterface>(operation);
}

mlir::LogicalResult verifyGroupedStageOperations(mlir::ModuleOp module) {
  mlir::Operation *illegal = nullptr;
  module.walk([&](mlir::Operation *operation) {
    llvm::StringRef dialect = operation->getName().getDialectNamespace();
    bool allowed =
        dialect == "builtin" || dialect == "func" || dialect == "arith" ||
        dialect == "math" || dialect == "tensor" || dialect == "linalg" ||
        dialect == "scf" || dialect == "cf" ||
        (dialect == "wafer" && isGroupedStageWaferOperation(operation));
    if (!allowed) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }

    bool ungroupedRoot =
        ((mlir::isa<mlir::linalg::LinalgOp>(operation) &&
          !mlir::isa<mlir::linalg::FillOp>(operation)) ||
         mlir::isa<wafer::WaferLinalgExtCollectiveOpInterface>(operation)) &&
        !operation->getParentOfType<wafer::GroupOp>();
    if (ungroupedRoot) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!illegal)
    return mlir::success();
  return illegal->emitOpError(
      "is not legal in a verified grouped-program artifact");
}

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
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
}

bool runPassPipeline(mlir::ModuleOp module,
                     void (*builder)(mlir::OpPassManager &)) {
  mlir::PassManager manager(module.getContext());
  builder(manager);
  return mlir::failed(manager.run(module));
}

bool runSpmdHelper(llvm::StringRef helper,
                   llvm::StringRef inputProgramDirectory,
                   llvm::StringRef outputProgramDirectory,
                   const ExecutionConfig &config,
                   llvm::raw_ostream &diagnostics) {
  std::string helperStorage = helper.str();
  std::string inputStorage = inputProgramDirectory.str();
  std::string outputStorage = outputProgramDirectory.str();
  std::string rankCountStorage = std::to_string(config.getRankCount());
  llvm::SmallVector<llvm::StringRef, 9> arguments = {
      helperStorage,    "--input-program-dir",
      inputStorage,     "--output-program-dir",
      outputStorage,    "--entry-function",
      "forward",        "--logical-rank-count",
      rankCountStorage,
  };
  int exitCode = llvm::sys::ExecuteAndWait(helperStorage, arguments);
  if (exitCode == 0)
    return false;
  std::string message = "XLA SPMD partitioner helper failed";
  if (exitCode > 0)
    message += " with exit code " + std::to_string(exitCode);
  return reject(diagnostics, message);
}

bool makeAbsoluteNormalizedPath(llvm::StringRef path,
                                llvm::SmallVectorImpl<char> &storage,
                                llvm::raw_ostream &diagnostics) {
  storage.assign(path.begin(), path.end());
  if (std::error_code error = llvm::sys::fs::make_absolute(storage))
    return reject(diagnostics,
                  "failed to make path absolute: " + error.message());
  llvm::sys::path::remove_dots(storage, /*remove_dot_dot=*/true);
  return false;
}

bool resolveThroughExistingAncestor(llvm::StringRef path,
                                    llvm::SmallVectorImpl<char> &storage,
                                    llvm::raw_ostream &diagnostics) {
  llvm::SmallString<256> existing(path);
  llvm::SmallVector<std::string, 8> missingComponents;
  while (!pathEntryExists(existing)) {
    llvm::StringRef filename = llvm::sys::path::filename(existing);
    if (filename.empty())
      return reject(diagnostics,
                    "failed to locate an existing output path ancestor");
    missingComponents.push_back(filename.str());
    llvm::sys::path::remove_filename(existing);
  }

  llvm::SmallString<256> canonicalExisting;
  if (std::error_code error =
          llvm::sys::fs::real_path(existing, canonicalExisting))
    return reject(diagnostics, "failed to resolve output path ancestor '" +
                                   existing.str().str() +
                                   "': " + error.message());
  for (const std::string &component : llvm::reverse(missingComponents))
    llvm::sys::path::append(canonicalExisting, component);
  storage.assign(canonicalExisting.begin(), canonicalExisting.end());
  return false;
}

bool pathIsWithin(llvm::StringRef path, llvm::StringRef directory) {
  if (path == directory)
    return true;
  if (!path.starts_with(directory) || path.size() <= directory.size())
    return false;
  return llvm::sys::path::is_separator(path[directory.size()]);
}

bool publishDirectoryNoReplace(llvm::StringRef source,
                               llvm::StringRef destination,
                               llvm::raw_ostream &diagnostics) {
#ifdef __linux__
  std::string sourceStorage = source.str();
  std::string destinationStorage = destination.str();
  if (::syscall(SYS_renameat2, AT_FDCWD, sourceStorage.c_str(), AT_FDCWD,
                destinationStorage.c_str(), RENAME_NOREPLACE) == 0)
    return false;

  int errorNumber = errno;
  if (errorNumber == EEXIST)
    return reject(diagnostics,
                  "output program directory appeared before publication; "
                  "refusing to replace it");
  return reject(
      diagnostics,
      "failed to atomically publish grouped program directory "
      "without replacement: " +
          std::error_code(errorNumber, std::generic_category()).message());
#else
  (void)source;
  (void)destination;
  return reject(
      diagnostics,
      "atomic no-replace directory publication is unsupported on this host");
#endif
}

} // namespace

mlir::LogicalResult
detail::verifyExactExecutionConfig(mlir::ModuleOp module,
                                   const ExecutionConfig &executionConfig) {
  return verifyExactExecutionConfigInternal(module, executionConfig);
}

static llvm::Expected<ExecutableBundle>
compileGroupedProgramToExecutableBundleImpl(
    llvm::StringRef groupedProgramDirectory, ExecutionConfig executionConfig,
    llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank) {
  auto fail = [&](llvm::StringRef message) -> llvm::Error {
    reject(diagnostics, message);
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   message.str().c_str());
  };
  if (groupedProgramDirectory.empty())
    return fail("grouped program directory must not be empty");
  if (!isDirectory(groupedProgramDirectory))
    return fail("grouped program path is not a directory");
  if (validateRegularDirectoryTree(groupedProgramDirectory, diagnostics))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "grouped program tree is not regular");

  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        diagnostic.print(diagnostics);
        diagnostics << "\n";
        return mlir::success();
      });

  mlir::OwningOpRef<mlir::ModuleOp> groupedModule =
      parseProgramDirectoryModule(groupedProgramDirectory, *context);
  if (!groupedModule)
    return fail("failed to parse verified grouped program module");
  if (mlir::failed(
          verifyExactExecutionConfigInternal(*groupedModule, executionConfig)))
    return fail("grouped program does not match ExecutionConfig");
  if (!hasPostSpmdMarker(*groupedModule, groupedProgramDirectory))
    return fail("grouped program is missing its post-SPMD marker");
  if (containsDialectSemantics(*groupedModule, "stablehlo") ||
      containsDialectSemantics(*groupedModule, "sdy") ||
      mlir::failed(verifyGroupedStageOperations(*groupedModule)))
    return fail("input is not a verified grouped-program artifact");

  frontend::FrontendProgramVerificationResult program;
  if (mlir::failed(verifyProgramDirectoryMetadata(
          *groupedModule, groupedProgramDirectory, diagnostics, &program)))
    return fail("grouped program metadata verification failed");
  if (program.logicalRankCount != executionConfig.getRankCount())
    return fail("typed program rank domain does not match ExecutionConfig");
  if (program.parameters.size() != program.programParameterCount ||
      program.constants.size() != program.programConstantCount)
    return fail("typed program resources do not cover all parameters and "
                "constants");

  return detail::buildExecutableBundle(context, *groupedModule,
                                       std::move(program), executionConfig,
                                       diagnostics, failAfterLogicalRank);
}

llvm::Expected<ExecutableBundle>
compileGroupedProgramToExecutableBundle(llvm::StringRef groupedProgramDirectory,
                                        ExecutionConfig executionConfig,
                                        llvm::raw_ostream &diagnostics) {
  return compileGroupedProgramToExecutableBundleImpl(
      groupedProgramDirectory, executionConfig, diagnostics, std::nullopt);
}

static mlir::LogicalResult compileProgramImpl(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    std::optional<int64_t> failAfterTargetLogicalRank) {
#if !defined(WAFER_ENABLE_STABLEHLO) || !defined(WAFER_ENABLE_SHARDY)
  (void)request;
  (void)outputProgramDirectory;
  (void)xlaSpmdPartitionerHelper;
  (void)targetToolchain;
  (void)failAfterLogicalRank;
  (void)failAfterTargetLogicalRank;
  reject(diagnostics,
         "StableHLO and SPMD partitioner dependencies are required");
  return mlir::failure();
#else
  if (outputProgramDirectory.empty()) {
    reject(diagnostics, "output program directory must not be empty");
    return mlir::failure();
  }
  if (xlaSpmdPartitionerHelper.empty()) {
    reject(diagnostics, "XLA SPMD partitioner helper path must not be empty");
    return mlir::failure();
  }

  llvm::SmallString<256> canonicalSource;
  if (std::error_code error = llvm::sys::fs::real_path(
          request.getSourceProgramDirectory(), canonicalSource)) {
    reject(diagnostics, "failed to resolve source program directory '" +
                            request.getSourceProgramDirectory().str() +
                            "': " + error.message());
    return mlir::failure();
  }
  if (!isDirectory(canonicalSource)) {
    reject(diagnostics, "source program directory is not a directory: '" +
                            canonicalSource.str().str() + "'");
    return mlir::failure();
  }

  llvm::SmallString<256> absoluteOutput;
  if (makeAbsoluteNormalizedPath(outputProgramDirectory, absoluteOutput,
                                 diagnostics))
    return mlir::failure();
  llvm::StringRef outputName = llvm::sys::path::filename(absoluteOutput);
  if (outputName.empty()) {
    reject(diagnostics, "output program directory must name a directory");
    return mlir::failure();
  }
  if (pathEntryExists(absoluteOutput)) {
    reject(diagnostics,
           "refusing to replace existing output program directory: '" +
               absoluteOutput.str().str() + "'");
    return mlir::failure();
  }

  llvm::SmallString<256> outputParent(absoluteOutput);
  llvm::sys::path::remove_filename(outputParent);
  if (outputParent.empty())
    outputParent = ".";

  llvm::SmallString<256> prospectiveCanonicalOutputParent;
  if (resolveThroughExistingAncestor(
          outputParent, prospectiveCanonicalOutputParent, diagnostics))
    return mlir::failure();
  llvm::SmallString<256> prospectiveCanonicalOutput(
      prospectiveCanonicalOutputParent);
  llvm::sys::path::append(prospectiveCanonicalOutput, outputName);
  if (pathIsWithin(prospectiveCanonicalOutput, canonicalSource)) {
    reject(diagnostics,
           "output program directory must not equal or be nested under the "
           "source program directory");
    return mlir::failure();
  }

  if (createDirectory(outputParent, diagnostics))
    return mlir::failure();

  llvm::SmallString<256> canonicalOutputParent;
  if (std::error_code error =
          llvm::sys::fs::real_path(outputParent, canonicalOutputParent)) {
    reject(diagnostics, "failed to resolve output parent directory '" +
                            outputParent.str().str() + "': " + error.message());
    return mlir::failure();
  }
  llvm::SmallString<256> canonicalOutput(canonicalOutputParent);
  llvm::sys::path::append(canonicalOutput, outputName);
  if (pathIsWithin(canonicalOutput, canonicalSource)) {
    reject(diagnostics,
           "output program directory must not equal or be nested under the "
           "source program directory");
    return mlir::failure();
  }
  if (pathEntryExists(canonicalOutput)) {
    reject(diagnostics,
           "refusing to replace existing output program directory: '" +
               canonicalOutput.str().str() + "'");
    return mlir::failure();
  }

  llvm::SmallString<256> stagingPrefix(canonicalOutputParent);
  llvm::sys::path::append(stagingPrefix, ".wafer-compile-staging");
  llvm::SmallString<256> transactionRoot;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          stagingPrefix, transactionRoot)) {
    reject(diagnostics, "failed to create compilation staging directory: " +
                            error.message());
    return mlir::failure();
  }
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(transactionRoot); });

  llvm::SmallString<256> sourceSnapshot(transactionRoot);
  llvm::sys::path::append(sourceSnapshot, "source");
  if (copyDirectory(canonicalSource, sourceSnapshot, diagnostics))
    return mlir::failure();

  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        diagnostic.print(diagnostics);
        diagnostics << "\n";
        return mlir::success();
      });

  mlir::OwningOpRef<mlir::ModuleOp> sourceModule =
      parseProgramDirectoryModule(sourceSnapshot, context);
  if (!sourceModule)
    return mlir::failure();
  if (hasPostSpmdMarker(*sourceModule, sourceSnapshot)) {
    reject(diagnostics,
           "source program is already SPMD-partitioned; typed compilation "
           "requires an exporter program");
    return mlir::failure();
  }
  if (containsDialectSemantics(*sourceModule, "sdy")) {
    reject(diagnostics,
           "source program contains Shardy semantics that the external XLA "
           "SPMD helper boundary cannot consume");
    return mlir::failure();
  }
  if (mlir::failed(verifyStablehloStageOperations(*sourceModule)))
    return mlir::failure();
  if (mlir::failed(materializeOrVerifyExactExecutionConfig(
          *sourceModule, request.getExecutionConfig())))
    return mlir::failure();
  if (mlir::failed(verifyProgramDirectoryMetadata(*sourceModule, sourceSnapshot,
                                                  diagnostics)))
    return mlir::failure();

  llvm::SmallString<256> propagatedProgram(transactionRoot);
  llvm::sys::path::append(propagatedProgram, "propagated");
  if (copyDirectory(sourceSnapshot, propagatedProgram, diagnostics))
    return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> helperModule = sourceModule->clone();
  eraseTargetTopologyAndExecutionMesh(*helperModule);
  if (mlir::failed(mlir::verify(*helperModule)))
    return mlir::failure();
  if (writeProgramModule(*helperModule, propagatedProgram, diagnostics))
    return mlir::failure();

  llvm::SmallString<256> groupedProgram(transactionRoot);
  llvm::sys::path::append(groupedProgram, "grouped");
  if (runSpmdHelper(xlaSpmdPartitionerHelper, propagatedProgram, groupedProgram,
                    request.getExecutionConfig(), diagnostics))
    return mlir::failure();

  if (validateRegularDirectoryTree(groupedProgram, diagnostics))
    return mlir::failure();
  for (llvm::StringRef requiredMember :
       {llvm::StringRef("forward.mlir"), llvm::StringRef("forward.meta")}) {
    if (!isRegularFile(programFile(
            groupedProgram, {llvm::StringRef("functions"), requiredMember}))) {
      reject(diagnostics,
             "XLA SPMD partitioner output is missing required regular member "
             "'functions/" +
                 requiredMember.str() + "'");
      return mlir::failure();
    }
  }
  if (!isRegularFile(
          programFile(groupedProgram,
                      {llvm::StringRef("functions"),
                       llvm::StringRef("forward.parameter_shards.json")}))) {
    reject(diagnostics,
           "XLA SPMD partitioner output is missing its partition marker");
    return mlir::failure();
  }
  if (mergeMissingProgramMembers(sourceSnapshot, groupedProgram, diagnostics) ||
      validateRegularDirectoryTree(groupedProgram, diagnostics))
    return mlir::failure();

  mlir::OwningOpRef<mlir::ModuleOp> groupedModule =
      parseProgramDirectoryModule(groupedProgram, context);
  if (!groupedModule)
    return mlir::failure();
  if (mlir::failed(materializeOrVerifyExactExecutionConfig(
          *groupedModule, request.getExecutionConfig())))
    return mlir::failure();
  if (!hasPostSpmdMarker(*groupedModule, groupedProgram)) {
    reject(diagnostics,
           "XLA SPMD partitioner output is missing its partition marker");
    return mlir::failure();
  }
  if (containsDialectSemantics(*groupedModule, "sdy")) {
    reject(diagnostics,
           "XLA SPMD partitioner output still contains Shardy semantics");
    return mlir::failure();
  }
  if (mlir::failed(verifyStablehloStageOperations(*groupedModule)))
    return mlir::failure();
  if (mlir::failed(verifyProgramDirectoryMetadata(*groupedModule,
                                                  groupedProgram, diagnostics)))
    return mlir::failure();
  if (runPassPipeline(*groupedModule, wafer::buildStablehloToLinalgPipeline) ||
      runPassPipeline(*groupedModule, wafer::buildFormLogicalGroupsPipeline))
    return mlir::failure();
  if (containsDialectSemantics(*groupedModule, "stablehlo") ||
      containsDialectSemantics(*groupedModule, "sdy")) {
    reject(diagnostics,
           "grouped program still contains frontend or sharding operations");
    return mlir::failure();
  }
  if (mlir::failed(verifyGroupedStageOperations(*groupedModule)))
    return mlir::failure();
  if (writeProgramModule(*groupedModule, groupedProgram, diagnostics))
    return mlir::failure();

  mlir::OwningOpRef<mlir::ModuleOp> verifiedGroupedModule =
      parseProgramDirectoryModule(groupedProgram, context);
  if (!verifiedGroupedModule)
    return mlir::failure();
  if (mlir::failed(verifyExactExecutionConfigInternal(
          *verifiedGroupedModule, request.getExecutionConfig())))
    return mlir::failure();
  if (!hasPostSpmdMarker(*verifiedGroupedModule, groupedProgram)) {
    reject(diagnostics,
           "grouped-program readback is missing its partition marker");
    return mlir::failure();
  }
  if (containsDialectSemantics(*verifiedGroupedModule, "stablehlo") ||
      containsDialectSemantics(*verifiedGroupedModule, "sdy")) {
    reject(diagnostics,
           "grouped-program readback contains frontend or sharding semantics");
    return mlir::failure();
  }
  if (mlir::failed(verifyGroupedStageOperations(*verifiedGroupedModule)) ||
      mlir::failed(verifyProgramDirectoryMetadata(*verifiedGroupedModule,
                                                  groupedProgram, diagnostics)))
    return mlir::failure();

  llvm::Expected<ExecutableBundle> executableBundle =
      compileGroupedProgramToExecutableBundleImpl(
          groupedProgram, request.getExecutionConfig(), diagnostics,
          failAfterLogicalRank);
  if (!executableBundle) {
    llvm::consumeError(executableBundle.takeError());
    return mlir::failure();
  }

  llvm::SmallString<256> groupedModules(groupedProgram);
  llvm::sys::path::append(groupedModules, "modules");
  if (pathEntryExists(groupedModules)) {
    reject(diagnostics,
           "grouped program contains reserved target artifact member "
           "'modules'");
    return mlir::failure();
  }

  llvm::SmallString<256> stagedTargetArtifacts(transactionRoot);
  llvm::sys::path::append(stagedTargetArtifacts, "target-artifacts");
  {
    llvm::Expected<TargetArtifactBundle> targetArtifacts =
        detail::compileExecutableBundleToTargetArtifactsImpl(
            *executableBundle, stagedTargetArtifacts, targetToolchain,
            diagnostics, failAfterTargetLogicalRank);
    if (!targetArtifacts) {
      llvm::consumeError(targetArtifacts.takeError());
      return mlir::failure();
    }
  }

  llvm::SmallString<256> stagedModules(stagedTargetArtifacts);
  llvm::sys::path::append(stagedModules, "modules");
  if (std::error_code error =
          llvm::sys::fs::rename(stagedModules, groupedModules)) {
    reject(diagnostics,
           "failed to attach verified target modules: " + error.message());
    return mlir::failure();
  }
  if (std::error_code error = llvm::sys::fs::remove(stagedTargetArtifacts)) {
    reject(diagnostics,
           "failed to remove empty target artifact staging root: " +
               error.message());
    return mlir::failure();
  }

  if (publishDirectoryNoReplace(groupedProgram, canonicalOutput, diagnostics))
    return mlir::failure();
  return mlir::success();
#endif
}

mlir::LogicalResult compileProgram(CompilationRequest request,
                                   llvm::StringRef outputProgramDirectory,
                                   llvm::StringRef xlaSpmdPartitionerHelper,
                                   const TargetToolchain &targetToolchain,
                                   llvm::raw_ostream &diagnostics) {
  return compileProgramImpl(std::move(request), outputProgramDirectory,
                            xlaSpmdPartitionerHelper, targetToolchain,
                            diagnostics, std::nullopt, std::nullopt);
}

mlir::LogicalResult testing::compileProgramWithRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLogicalRank < 0 ||
      failAfterLogicalRank >= request.getExecutionConfig().getRankCount()) {
    reject(diagnostics, "test-only failure rank is outside ExecutionConfig");
    return mlir::failure();
  }
  return compileProgramImpl(std::move(request), outputProgramDirectory,
                            xlaSpmdPartitionerHelper, targetToolchain,
                            diagnostics, failAfterLogicalRank, std::nullopt);
}

mlir::LogicalResult testing::compileProgramWithTargetRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLogicalRank < 0 ||
      failAfterLogicalRank >= request.getExecutionConfig().getRankCount()) {
    reject(diagnostics,
           "test-only target failure rank is outside ExecutionConfig");
    return mlir::failure();
  }
  return compileProgramImpl(std::move(request), outputProgramDirectory,
                            xlaSpmdPartitionerHelper, targetToolchain,
                            diagnostics, std::nullopt, failAfterLogicalRank);
}

} // namespace wafer::compiler
