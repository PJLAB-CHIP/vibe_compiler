//===- XlaSpmdPartitioning.cpp - StableHLO and XLA SPMD bridge -------===//

#include "XlaSpmdPartitionerInternal.h"

#include "absl/container/inlined_vector.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/PassManager.h"
#include "tsl/platform/statusor.h"
#include "xla/client/xla_computation.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/spmd/shardy/shardy_xla_pass.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/service/spmd/spmd_prepare.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <limits>
#include <optional>

namespace wafer::xla_spmd_helper {

constexpr std::string_view kMhloShardingAttr = "mhlo.sharding";
constexpr std::string_view kStablehloShardingAttr = "stablehlo.sharding";

static std::optional<std::string>
canonicalizeSequentialIotaSharding(std::string_view sharding) {
  std::string_view text = absl::StripAsciiWhitespace(sharding);
  if (!absl::StartsWith(text, "{devices=[") || !absl::EndsWith(text, "}") ||
      text.find("<=") != std::string_view::npos) {
    return std::nullopt;
  }

  size_t dimsStart = text.find('[');
  size_t dimsEnd = text.find(']', dimsStart);
  if (dimsStart == std::string_view::npos ||
      dimsEnd == std::string_view::npos) {
    return std::nullopt;
  }

  std::string_view dimsText =
      text.substr(dimsStart + 1, dimsEnd - dimsStart - 1);
  std::vector<int64_t> dims;
  int64_t deviceCount = 1;
  for (std::string_view token : absl::StrSplit(dimsText, ',')) {
    token = absl::StripAsciiWhitespace(token);
    int64_t dim = 0;
    if (token.empty() || !absl::SimpleAtoi(token, &dim) || dim <= 0)
      return std::nullopt;
    if (deviceCount > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    deviceCount *= dim;
    dims.push_back(dim);
  }
  if (dims.empty())
    return std::nullopt;

  std::string_view rest = absl::StripAsciiWhitespace(
      text.substr(dimsEnd + 1, text.size() - dimsEnd - 2));
  size_t deviceListEnd = 0;
  while (deviceListEnd < rest.size()) {
    unsigned char c = static_cast<unsigned char>(rest[deviceListEnd]);
    if (!std::isdigit(c) && rest[deviceListEnd] != ',' && !std::isspace(c)) {
      break;
    }
    ++deviceListEnd;
  }

  std::string_view deviceList =
      absl::StripAsciiWhitespace(rest.substr(0, deviceListEnd));
  if (deviceList.empty())
    return std::nullopt;

  std::vector<int64_t> devices;
  for (std::string_view token : absl::StrSplit(deviceList, ',')) {
    token = absl::StripAsciiWhitespace(token);
    int64_t device = 0;
    if (token.empty() || !absl::SimpleAtoi(token, &device))
      return std::nullopt;
    devices.push_back(device);
  }
  if (devices.size() != static_cast<size_t>(deviceCount))
    return std::nullopt;
  for (auto [index, device] : llvm::enumerate(devices)) {
    if (device != static_cast<int64_t>(index))
      return std::nullopt;
  }

  std::string_view suffix = rest.substr(deviceListEnd);
  return absl::StrCat("{devices=[", dimsText, "]<=[", deviceCount, "]", suffix,
                      "}");
}

static void canonicalizeShardingStringAttr(mlir::Operation *op,
                                           llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return;
  if (std::optional<std::string> normalized =
          canonicalizeSequentialIotaSharding(attr.getValue().str())) {
    op->setAttr(attrName, mlir::StringAttr::get(op->getContext(), *normalized));
  }
}

static void
canonicalizeFunctionBoundaryShardingAttrs(mlir::func::FuncOp func,
                                          llvm::StringRef attrName) {
  for (int64_t index = 0; index < func.getNumArguments(); ++index) {
    auto attr = func.getArgAttrOfType<mlir::StringAttr>(index, attrName);
    if (!attr)
      continue;
    if (std::optional<std::string> normalized =
            canonicalizeSequentialIotaSharding(attr.getValue().str())) {
      func.setArgAttr(index, attrName,
                      mlir::StringAttr::get(func.getContext(), *normalized));
    }
  }

  for (int64_t index = 0; index < func.getNumResults(); ++index) {
    auto attr = func.getResultAttrOfType<mlir::StringAttr>(index, attrName);
    if (!attr)
      continue;
    if (std::optional<std::string> normalized =
            canonicalizeSequentialIotaSharding(attr.getValue().str())) {
      func.setResultAttr(index, attrName,
                         mlir::StringAttr::get(func.getContext(), *normalized));
    }
  }
}

void canonicalizeFrontendShardingAttrs(mlir::ModuleOp module) {
  module.walk([&](mlir::Operation *op) {
    canonicalizeShardingStringAttr(op, kMhloShardingAttr);
    canonicalizeShardingStringAttr(op, kStablehloShardingAttr);
    if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
      canonicalizeFunctionBoundaryShardingAttrs(func, kMhloShardingAttr);
      canonicalizeFunctionBoundaryShardingAttrs(func, kStablehloShardingAttr);
    }
  });
}

absl::StatusOr<std::unique_ptr<xla::HloModule>>
stablehloToHloModule(mlir::ModuleOp module, int64_t numPartitions) {
  xla::XlaComputation computation;
  TF_RETURN_IF_ERROR(xla::MlirToXlaComputation(
      module, computation, /*use_tuple_args=*/false, /*return_tuple=*/false,
      /*use_shardy=*/true));

  xla::HloModuleConfig config;
  xla::ProgramShape programShape(computation.proto().host_program_shape());
  config.SetDefaultComputationLayout(programShape);
  config.set_use_spmd_partitioning(true);
  config.set_num_partitions(numPartitions);
  config.set_replica_count(1);
  absl::InlinedVector<bool, 8> allowParams(programShape.parameters_size(),
                                           false);
  config.set_allow_spmd_sharding_propagation_to_parameters(allowParams);
  absl::InlinedVector<bool, 8> allowOutputs(
      programShape.result().IsTuple()
          ? programShape.result().tuple_shapes_size()
          : 1,
      true);
  config.set_allow_spmd_sharding_propagation_to_output(allowOutputs);

  return xla::HloModule::CreateFromProto(computation.proto(), config);
}

absl::Status prepareSpmdPartitioning(xla::HloModule *module) {
  xla::HloPassPipeline pipeline("wafer-sharding-propagation");
  pipeline.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                     /*allow_mixed_precision=*/false);
  pipeline.AddPass<xla::sdy::ShardyXLA>();
  pipeline.AddPass<xla::spmd::SpmdPrepare>();
  pipeline.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                     /*allow_mixed_precision=*/false);
  TF_RETURN_IF_ERROR(pipeline.Run(module).status());
  return absl::OkStatus();
}

absl::Status runSpmdPartitioner(xla::HloModule *module,
                                int64_t numPartitions) {
  xla::spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  auto collectiveOpsCreator =
      xla::spmd::GetDefaultCollectiveOpsCreator(numPartitions,
                                                /*num_replicas=*/1);

  xla::HloPassPipeline pipeline("wafer-spmd-partitioning");
  pipeline.AddPass<xla::spmd::SpmdPartitioner>(
      numPartitions, /*num_replicas=*/1, options, collectiveOpsCreator);
  pipeline.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                     /*allow_mixed_precision=*/false);
  TF_RETURN_IF_ERROR(pipeline.Run(module).status());
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
hloModuleToStablehlo(mlir::MLIRContext &context, xla::HloModule *module,
                     const std::vector<std::string> &parameterShardings) {
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlirModule,
      xla::ConvertHloToMlirHlo(context, module,
                               /*import_all_computations=*/false,
                               /*flatten_computation_args_result=*/true));

  mlir::PassManager pm(&context);
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (mlir::failed(pm.run(*mlirModule)))
    return absl::InternalError(
        "failed to legalize partitioned HLO to StableHLO");

  std::vector<mlir::Attribute> shardingAttrs;
  shardingAttrs.reserve(parameterShardings.size());
  for (const std::string &sharding : parameterShardings)
    shardingAttrs.push_back(mlir::StringAttr::get(&context, sharding));
  (*mlirModule)
      ->setAttr("mhlo.spmd_parameters_shardings",
                mlir::ArrayAttr::get(&context, shardingAttrs));
  (*mlirModule)
      ->setAttr("mhlo.use_auto_spmd_partitioning",
                mlir::BoolAttr::get(&context, false));
  return mlirModule;
}

std::string moduleToString(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os);
  os << "\n";
  return text;
}

} // namespace wafer::xla_spmd_helper
