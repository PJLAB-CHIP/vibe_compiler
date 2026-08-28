//===- XlaSpmdProgram.cpp - Program-directory SPMD orchestration -----===//

#include "XlaSpmdPartitionerInternal.h"

#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/pjrt/mlir_to_hlo.h"

#include <optional>
#include <utility>

namespace wafer::xla_spmd_helper {

absl::Status run(const Options &options) {
  fs::remove_all(options.outputProgramDir);
  fs::create_directories(options.outputProgramDir / "functions");

  TF_ASSIGN_OR_RETURN(
      ProgramMetadata meta,
      parseMeta(options.inputProgramDir / "functions" / "forward.meta"));
  std::optional<llvm::StringRef> metadataName = meta.root.getString("name");
  if (!metadataName || *metadataName != options.entryFunction)
    return absl::InvalidArgumentError(
        "program metadata name must match the canonical entry 'forward'");
  TF_ASSIGN_OR_RETURN(
      std::string mlirText,
      readFile(options.inputProgramDir / "functions" / "forward.mlir"));

  mlir::MLIRContext inputContext;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> inputModule,
                      xla::ParseMlirModuleString(mlirText, inputContext));
  canonicalizeFrontendShardingAttrs(*inputModule);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModule> prePartitionModule,
      stablehloToHloModule(*inputModule, options.numPartitions));

  std::unique_ptr<xla::HloModule> distributedModule =
      prePartitionModule->Clone();
  TF_RETURN_IF_ERROR(prepareSpmdPartitioning(distributedModule.get()));

  std::vector<std::string> parameterShardings;
  const xla::HloComputation *preEntry = distributedModule->entry_computation();
  for (int64_t index = 0; index < preEntry->num_parameters(); ++index) {
    const xla::HloInstruction *parameter =
        preEntry->parameter_instruction(index);
    parameterShardings.push_back(parameter->has_sharding()
                                     ? parameter->sharding().ToString()
                                     : std::string("{replicated}"));
  }

  std::unique_ptr<xla::HloModule> partitionedModule =
      distributedModule->Clone();
  TF_RETURN_IF_ERROR(
      runSpmdPartitioner(partitionedModule.get(), options.numPartitions));
  mlir::MLIRContext outputContext;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> outputModule,
                      hloModuleToStablehlo(outputContext,
                                           partitionedModule.get(),
                                           parameterShardings));
  TF_ASSIGN_OR_RETURN(mlir::func::FuncOp mainFunc, findMain(*outputModule));
  TF_ASSIGN_OR_RETURN(std::vector<TensorSignature> inputSignatures,
                      inputSignaturesFromFunc(mainFunc, meta));
  TF_ASSIGN_OR_RETURN(std::vector<TensorSignature> outputSignatures,
                      outputSignaturesFromFunc(mainFunc, meta));
  TF_ASSIGN_OR_RETURN(DistributedBoundary distributedBoundary,
                      buildDistributedBoundary(meta, *distributedModule,
                                               *partitionedModule,
                                               options.numPartitions));

  std::vector<ParameterBinding> bindings;
  TF_RETURN_IF_ERROR(materializeParameterShards(
      options, meta, *distributedModule, *partitionedModule, bindings));
  TF_RETURN_IF_ERROR(copyConstantPayloads(options, meta));

  TF_RETURN_IF_ERROR(
      writeFile(options.outputProgramDir / "functions" / "forward.mlir",
                moduleToString(*outputModule)));
  TF_RETURN_IF_ERROR(
      writeFile(options.outputProgramDir / "functions" / "forward.meta",
                metaToString(std::move(meta), std::move(inputSignatures),
                             std::move(outputSignatures),
                             std::move(distributedBoundary))));
  TF_RETURN_IF_ERROR(writeFile(
      options.outputProgramDir / "functions" / "forward.parameter_shards.json",
      bindingsToJson(options.entryFunction, options.numPartitions,
                     bindings)));
  return absl::OkStatus();
}

} // namespace wafer::xla_spmd_helper
