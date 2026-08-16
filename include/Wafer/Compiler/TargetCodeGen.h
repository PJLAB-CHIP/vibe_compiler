//===- TargetCodeGen.h - Wafer target code generation ----------*- C++ -*-===//

#ifndef WAFER_COMPILER_TARGETCODEGEN_H
#define WAFER_COMPILER_TARGETCODEGEN_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/IR/WaferDialect.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

/// Closed union of what one ordered tile entry argument references. The kind
/// plus `resourceIndex` form the exact typed reference: an external program
/// port, a package-owned TargetTensor, or an entry-local requirement. This
/// enum describes entry arguments only; it does not own bytes, file ranges,
/// device addresses, or pointer-row storage.
enum class TileEntryArgumentKind {
  /// Caller-bound program input port (`resourceIndex` is the program-boundary
  /// input index).
  ExternalInput,
  /// Package-owned parameter/constant TargetTensor (`resourceIndex` is the
  /// accepted entry argument index; the parameter/constant distinction lives
  /// in the referenced program tensor's role).
  TargetTensor,
  /// Caller-visible output port (`resourceIndex` is the program-boundary
  /// output index).
  ExternalOutput,
  /// Entry-local compiler-managed default DDR arena.
  Workspace,
  /// Entry-local profiler capture record.
  ProfileRecord,
  /// Entry-local Direct-DTE status.
  TransportStatus,
};

/// Host access pattern of one tile entry argument. The producer sets it from
/// the closed kind; consumers verify consistency instead of re-deriving it.
enum class TileEntryArgumentAccess { ReadOnly, WriteOnly, ReadWrite };

/// One ordered argument of one Tile target entry: ordinal, closed kind,
/// exactly one typed reference, target descriptor facts, and access. It never
/// owns bytes, a file range, a device address, or pointer-row storage; the
/// current kernel pointer row is only the runtime representation that reads
/// this schema.
struct TileEntryArgument {
  int64_t ordinal = -1;
  TileEntryArgumentKind kind = TileEntryArgumentKind::ExternalInput;
  int64_t resourceIndex = -1;
  std::string name;
  std::string dtype;
  MemLayout layout;
  std::vector<int64_t> shape;
  int64_t byteSize = -1;
  int64_t alignment = -1;
  TileEntryArgumentAccess access = TileEntryArgumentAccess::ReadOnly;
};

/// One fully translated target LLVM module and the context that owns all of
/// its uniqued IR state. The module is immutable after construction: target
/// linking and host-side consumers must share this verified translation
/// instead of independently lowering the accepted Tile again.
class TargetLLVMModule {
public:
  ~TargetLLVMModule();
  TargetLLVMModule(TargetLLVMModule &&);
  TargetLLVMModule &operator=(TargetLLVMModule &&);
  TargetLLVMModule(const TargetLLVMModule &) = delete;
  TargetLLVMModule &operator=(const TargetLLVMModule &) = delete;

  CardId getCardId() const { return cardId; }
  TileId getTileId() const { return tileId; }
  LaunchSlotId getLaunchSlotId() const { return launchSlotId; }
  llvm::StringRef getEntrySymbol() const { return entrySymbol; }
  TargetIdentityId getTargetIdentityId() const { return targetIdentity; }
  KernelRuntimeABIId getKernelRuntimeABIId() const { return kernelRuntimeABI; }
  llvm::StringRef getModuleFormat() const { return moduleFormat; }
  llvm::StringRef getModuleIdentifier() const;
  llvm::StringRef getTargetTriple() const;
  const std::vector<TileEntryArgument> &getTileEntryArguments() const {
    return tileEntryArguments;
  }
  const llvm::Module &getModule() const;

private:
  friend struct TargetLLVMModulesBuilder;

  TargetLLVMModule(CardId cardId, TileId tileId,
                   LaunchSlotId launchSlotId, llvm::StringRef entrySymbol,
                   TargetIdentityId targetIdentity,
                   KernelRuntimeABIId kernelRuntimeABI,
                   llvm::StringRef moduleFormat,
                   std::vector<TileEntryArgument> tileEntryArguments,
                   std::unique_ptr<llvm::LLVMContext> context,
                   std::unique_ptr<llvm::Module> module);

  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlotId;
  std::string entrySymbol;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId kernelRuntimeABI;
  std::string moduleFormat;
  std::vector<TileEntryArgument> tileEntryArguments;
  // Declaration order is intentional: reverse destruction destroys the
  // module before the context that owns its uniqued state.
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
};

/// Atomic owner of the complete target LLVM Tile domain. This is an
/// invocation-local boundary and deliberately has no serialization form.
class TargetLLVMModules {
public:
  ~TargetLLVMModules();
  TargetLLVMModules(TargetLLVMModules &&);
  TargetLLVMModules &operator=(TargetLLVMModules &&);
  TargetLLVMModules(const TargetLLVMModules &) = delete;
  TargetLLVMModules &operator=(const TargetLLVMModules &) = delete;

  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const RuntimeLaunchContract &getRuntimeLaunchContract() const {
    return runtimeLaunchContract;
  }
  const std::vector<TargetLLVMModule> &getModules() const { return modules; }

private:
  friend struct TargetLLVMModulesBuilder;

  TargetLLVMModules(ExecutionConfig executionConfig,
                    RuntimeLaunchContract runtimeLaunchContract,
                    std::vector<TargetLLVMModule> modules)
      : executionConfig(executionConfig),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        modules(std::move(modules)) {}

  ExecutionConfig executionConfig;
  RuntimeLaunchContract runtimeLaunchContract;
  std::vector<TargetLLVMModule> modules;
};

/// Invocation-local external tool and resource facts consumed by target
/// device linking: the toolchain interpreters, the pinned TX8 dependency
/// root, and the Wafer CRT/ABI resources. Every path is validated by the
/// driver resolver before this value is constructed.
class TargetToolchain {
public:
  static llvm::Expected<TargetToolchain>
  create(llvm::StringRef pythonExecutable, llvm::StringRef deviceLinkerScript,
         llvm::StringRef llvmClangXX, llvm::StringRef tx8DepsRoot,
         llvm::StringRef waferIncludeDir, llvm::StringRef waferCrtSource,
         llvm::StringRef waferCrtIncludeDir);

  llvm::StringRef getPythonExecutable() const { return pythonExecutable; }
  llvm::StringRef getDeviceLinkerScript() const { return deviceLinkerScript; }
  llvm::StringRef getLLVMClangXX() const { return llvmClangXX; }
  llvm::StringRef getTx8DepsRoot() const { return tx8DepsRoot; }
  llvm::StringRef getWaferIncludeDir() const { return waferIncludeDir; }
  llvm::StringRef getWaferCrtSource() const { return waferCrtSource; }
  llvm::StringRef getWaferCrtIncludeDir() const { return waferCrtIncludeDir; }

private:
  TargetToolchain(llvm::StringRef pythonExecutable,
                  llvm::StringRef deviceLinkerScript, llvm::StringRef llvmClangXX,
                  llvm::StringRef tx8DepsRoot, llvm::StringRef waferIncludeDir,
                  llvm::StringRef waferCrtSource,
                  llvm::StringRef waferCrtIncludeDir)
      : pythonExecutable(pythonExecutable.str()),
        deviceLinkerScript(deviceLinkerScript.str()),
        llvmClangXX(llvmClangXX.str()), tx8DepsRoot(tx8DepsRoot.str()),
        waferIncludeDir(waferIncludeDir.str()),
        waferCrtSource(waferCrtSource.str()),
        waferCrtIncludeDir(waferCrtIncludeDir.str()) {}

  std::string pythonExecutable;
  std::string deviceLinkerScript;
  std::string llvmClangXX;
  std::string tx8DepsRoot;
  std::string waferIncludeDir;
  std::string waferCrtSource;
  std::string waferCrtIncludeDir;
};

class CompiledProgram;

/// Owner-backed result retained by downstream consumers that need the same
/// accepted Tile domain and the exact target LLVM modules consumed by
/// target module linking. Neither member is reconstructed from the package or
/// serialized as a side channel.
class CompiledProgram {
public:
  CompiledProgram(CompiledProgram &&) = default;
  CompiledProgram &operator=(CompiledProgram &&) = default;
  CompiledProgram(const CompiledProgram &) = delete;
  CompiledProgram &operator=(const CompiledProgram &) = delete;

  const CardExecutable &getCardExecutable() const {
    return cardExecutable;
  }
  const TargetLLVMModules &getTargetLLVMModules() const {
    return targetLLVMModules;
  }
  const CompilationIRTrace &getIRTrace() const { return irTrace; }

private:
  friend llvm::Expected<CompiledProgram> compileProgramWithTargetLLVMModules(
      CompilationRequest request, llvm::StringRef outputDirectory,
      llvm::StringRef xlaSpmdPartitionerHelper,
      const TargetToolchain &targetToolchain, CompilationOptions options,
      llvm::raw_ostream &diagnostics);

  CompiledProgram(CardExecutable cardExecutable,
                  TargetLLVMModules targetLLVMModules,
                  CompilationIRTrace irTrace)
      : cardExecutable(std::move(cardExecutable)),
        targetLLVMModules(std::move(targetLLVMModules)),
        irTrace(std::move(irTrace)) {}

  CardExecutable cardExecutable;
  TargetLLVMModules targetLLVMModules;
  CompilationIRTrace irTrace;
};

/// Runs the same compilation transaction as compileProgram while
/// retaining the owner-backed CardExecutable, the exact target LLVM modules
/// used to create the linked target modules, and the IR trace. This is the
/// internal qualification/debug entry; no second lowering is performed and
/// the members never enter the ordinary public result.
llvm::Expected<CompiledProgram> compileProgramWithTargetLLVMModules(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics);

/// Prepares the fixed Kernel Runtime ABI, lowers, translates, and verifies
/// every accepted Tile before atomically returning an owner-backed
/// LLVM modules. This function does not write target modules or a
/// package.
llvm::Expected<TargetLLVMModules>
compileCardExecutableToTargetLLVMModules(
    const CardExecutable &cardExecutable,
    llvm::raw_ostream &diagnostics);

/// Identifier for one linked target module. Physical
/// Tile to payload coverage is represented only by VerifiedTargetTileInterface;
/// this identifier carries no implicit topology or launch-slot semantics.
class TargetModuleId {
public:
  TargetModuleId() = delete;
  explicit constexpr TargetModuleId(uint64_t value) : value(value) {}

  constexpr uint64_t getValue() const { return value; }

  friend constexpr bool operator==(TargetModuleId lhs, TargetModuleId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(TargetModuleId lhs, TargetModuleId rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(TargetModuleId lhs, TargetModuleId rhs) {
    return lhs.value < rhs.value;
  }

private:
  uint64_t value;
};

/// Semantic role of one externally visible target-module function. Symbol
/// spelling is only a loader locator; the runtime launch contract interprets
/// roles.
enum class TargetExportRole { Prepare, Main };

class VerifiedTargetExport {
public:
  TargetExportRole getRole() const { return role; }
  llvm::StringRef getSymbol() const { return symbol; }

private:
  friend struct LinkedTargetModulesBuilder;

  VerifiedTargetExport(TargetExportRole role, llvm::StringRef symbol)
      : role(role), symbol(symbol.str()) {}

  TargetExportRole role;
  std::string symbol;
};

class VerifiedTargetModule {
public:
  TargetModuleId getId() const { return id; }
  llvm::StringRef getRelativePath() const { return relativePath; }
  llvm::StringRef getContentDigest() const { return contentDigest; }
  TargetIdentityId getTargetIdentityId() const { return targetIdentity; }
  KernelRuntimeABIId getKernelRuntimeABIId() const { return kernelRuntimeABI; }
  llvm::StringRef getModuleFormat() const { return moduleFormat; }
  const std::vector<VerifiedTargetExport> &getExports() const {
    return exports;
  }

private:
  friend struct LinkedTargetModulesBuilder;

  VerifiedTargetModule(TargetModuleId id, llvm::StringRef relativePath,
                       llvm::StringRef contentDigest,
                       TargetIdentityId targetIdentity,
                       KernelRuntimeABIId kernelRuntimeABI,
                       llvm::StringRef moduleFormat,
                       std::vector<VerifiedTargetExport> exports)
      : id(id), relativePath(relativePath.str()),
        contentDigest(contentDigest.str()), targetIdentity(targetIdentity),
        kernelRuntimeABI(kernelRuntimeABI), moduleFormat(moduleFormat.str()),
        exports(std::move(exports)) {}

  TargetModuleId id;
  std::string relativePath;
  std::string contentDigest;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId kernelRuntimeABI;
  std::string moduleFormat;
  std::vector<VerifiedTargetExport> exports;
};

class VerifiedTargetTileInterface {
public:
  CardId getCardId() const { return cardId; }
  TileId getTileId() const { return tileId; }
  LaunchSlotId getLaunchSlotId() const { return launchSlotId; }
  TargetModuleId getModuleId() const { return moduleId; }
  const std::vector<TileEntryArgument> &getTileEntryArguments() const {
    return tileEntryArguments;
  }

private:
  friend struct LinkedTargetModulesBuilder;

  VerifiedTargetTileInterface(CardId cardId,
                              TileId tileId,
                              LaunchSlotId launchSlotId,
                              TargetModuleId moduleId,
                              std::vector<TileEntryArgument> tileEntryArguments)
      : cardId(cardId), tileId(tileId),
        launchSlotId(launchSlotId), moduleId(moduleId),
        tileEntryArguments(std::move(tileEntryArguments)) {}

  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlotId;
  TargetModuleId moduleId;
  std::vector<TileEntryArgument> tileEntryArguments;
};

class LinkedTargetModules {
public:
  LinkedTargetModules(LinkedTargetModules &&) = default;
  LinkedTargetModules &operator=(LinkedTargetModules &&) = default;
  LinkedTargetModules(const LinkedTargetModules &) = delete;
  LinkedTargetModules &operator=(const LinkedTargetModules &) = delete;

  llvm::StringRef getRootDirectory() const { return rootDirectory; }
  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const RuntimeLaunchContract &getRuntimeLaunchContract() const {
    return runtimeLaunchContract;
  }
  const std::vector<VerifiedTargetModule> &getModules() const {
    return modules;
  }
  const std::vector<VerifiedTargetTileInterface> &getTileInterfaces() const {
    return tileInterfaces;
  }

private:
  friend struct LinkedTargetModulesBuilder;

  LinkedTargetModules(llvm::StringRef rootDirectory,
                      ExecutionConfig executionConfig,
                      RuntimeLaunchContract runtimeLaunchContract,
                      std::vector<VerifiedTargetModule> modules,
                      std::vector<VerifiedTargetTileInterface> tileInterfaces)
      : rootDirectory(rootDirectory.str()), executionConfig(executionConfig),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        modules(std::move(modules)), tileInterfaces(std::move(tileInterfaces)) {
  }

  std::string rootDirectory;
  ExecutionConfig executionConfig;
  RuntimeLaunchContract runtimeLaunchContract;
  std::vector<VerifiedTargetModule> modules;
  std::vector<VerifiedTargetTileInterface> tileInterfaces;
};

/// Links the verified LLVM modules, checks the resulting target modules, and
/// writes outputDirectory with no-replace semantics. This function never
/// re-runs ABI preparation, target lowering, or LLVM translation.
llvm::Expected<LinkedTargetModules> linkTargetLLVMModules(
    const TargetLLVMModules &targetLLVMModules, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_TARGETCODEGEN_H
