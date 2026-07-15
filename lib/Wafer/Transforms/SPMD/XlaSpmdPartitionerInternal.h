#ifndef WAFER_LIB_TRANSFORMS_SPMD_XLASPMDPARTITIONERINTERNAL_H
#define WAFER_LIB_TRANSFORMS_SPMD_XLASPMDPARTITIONERINTERNAL_H

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace wafer::xla_spmd_helper {

namespace fs = std::filesystem;

struct Options {
  fs::path inputProgramDir;
  fs::path outputProgramDir;
  std::string entryFunction = "forward";
  int64_t logicalRankCount = 0;
};

struct InputLocation {
  std::string type;
  std::string name;
  int64_t position = -1;
};

struct TensorSignature {
  std::vector<int64_t> shape;
  std::string dtype;
};

struct ProgramMetadata {
  llvm::json::Object root;
  std::vector<InputLocation> inputLocations;
  std::vector<TensorSignature> inputSignatures;
  std::vector<TensorSignature> outputSignatures;
};

struct ParameterShard {
  int64_t rank = 0;
  int64_t replicaId = 0;
  std::string file;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

struct ParameterBinding {
  int64_t argumentIndex = 0;
  std::string name;
  std::string dtype;
  std::string distribution;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  std::vector<ParameterShard> shards;
};

struct DistributedRank {
  int64_t rank = 0;
  int64_t replicaId = 0;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

struct DistributedBoundaryBinding {
  int64_t index = 0;
  std::string dtype;
  std::string distribution;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  std::vector<DistributedRank> ranks;
};

struct DistributedBoundary {
  int64_t logicalRankCount = 0;
  std::vector<DistributedBoundaryBinding> inputs;
  std::vector<DistributedBoundaryBinding> outputs;
};

void printUsage();
absl::StatusOr<Options> parseOptions(int argc, char **argv);

absl::StatusOr<std::string> readFile(const fs::path &path);
absl::Status writeFile(const fs::path &path, std::string_view content);
absl::Status writeBytes(const fs::path &path,
                        const std::vector<uint8_t> &content);
absl::Status copyConstantPayloads(const Options &options,
                                  const ProgramMetadata &meta);

absl::StatusOr<ProgramMetadata> parseMeta(const fs::path &path);
std::string metaToString(ProgramMetadata meta,
                         std::vector<TensorSignature> inputSignatures,
                         std::vector<TensorSignature> outputSignatures,
                         DistributedBoundary boundary);
absl::StatusOr<mlir::func::FuncOp> findMain(mlir::ModuleOp module);
absl::StatusOr<std::vector<TensorSignature>>
inputSignaturesFromFunc(mlir::func::FuncOp func, const ProgramMetadata &meta);
absl::StatusOr<std::vector<TensorSignature>>
outputSignaturesFromFunc(mlir::func::FuncOp func, const ProgramMetadata &meta);
std::string bindingsToJson(const std::string &functionName,
                           int64_t logicalRankCount,
                           const std::vector<ParameterBinding> &bindings);

std::vector<int64_t> shapeDims(const xla::Shape &shape);
std::vector<int64_t> zeros(size_t size);
std::vector<int64_t> ones(size_t size);
absl::StatusOr<DistributedBoundary> buildDistributedBoundary(
    const ProgramMetadata &meta, const xla::HloModule &distributedModule,
    const xla::HloModule &partitionedModule, int64_t logicalRankCount);
absl::Status
materializeParameterShards(const Options &options, const ProgramMetadata &meta,
                           const xla::HloModule &prePartitionModule,
                           const xla::HloModule &partitionedModule,
                           std::vector<ParameterBinding> &bindings);

void canonicalizeFrontendShardingAttrs(mlir::ModuleOp module);
absl::StatusOr<std::unique_ptr<xla::HloModule>>
stablehloToHloModule(mlir::ModuleOp module, int64_t logicalRankCount);
absl::Status prepareSpmdPartitioning(xla::HloModule *module);
absl::Status runSpmdPartitioner(xla::HloModule *module,
                                int64_t logicalRankCount);
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
hloModuleToStablehlo(mlir::MLIRContext &context, xla::HloModule *module,
                     const std::vector<std::string> &parameterShardings);
std::string moduleToString(mlir::ModuleOp module);

absl::Status run(const Options &options);

} // namespace wafer::xla_spmd_helper

#endif
