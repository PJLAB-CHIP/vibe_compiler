//===- CanonicalPlanningTestSupport.cpp -------------------------------===//

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"

#include "llvm/Support/raw_ostream.h"

#include <type_traits>
#include <utility>

namespace wafer::test {

mlir::FailureOr<CanonicalPlanningPrefix>
buildCanonicalPlanningPrefix(const compiler::detail::StructuredDAGAnalysis &dag,
                             llvm::ArrayRef<TileId> tiles,
                             std::string *failureReason) {
  auto coordinate = compiler::detail::buildCanonicalSpatialAssignment(
      dag, tiles, failureReason);
  if (mlir::failed(coordinate))
    return mlir::failure();
  auto demandSession = compiler::detail::DemandPlanningSession::create(
      dag, analysis::IndexRelationLimits(), failureReason);
  if (mlir::failed(demandSession))
    return mlir::failure();
  analysis::ExactDemandOutcome demandOutcome =
      demandSession->query(coordinate->assignment);
  const analysis::ExactDemandProof *demand =
      analysis::getExactDemandProof(demandOutcome);
  if (!demand) {
    if (failureReason)
      *failureReason = std::visit(
          [](const auto &value) -> std::string {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, analysis::ExactDemandProof>)
              return {};
            else
              return value.detail;
          },
          demandOutcome);
    return mlir::failure();
  }
  auto rootAnalysis = compiler::detail::RootRegionWorkAnalysis::create(
      dag, coordinate->assignment, *demand, failureReason);
  if (mlir::failed(rootAnalysis))
    return mlir::failure();
  std::vector<analysis::RootRegionWork> rootWorks;
  for (const compiler::detail::StructuredDAGNode &node : dag.getNodes()) {
    const compiler::detail::SemanticRootKey *root =
        rootAnalysis->getRoot(node.id);
    if (!root)
      return mlir::failure();
    for (TileId tile : tiles) {
      analysis::RootRegionWorkOutcome outcome =
          rootAnalysis->query(*root, tile);
      if (std::holds_alternative<analysis::NoRootRegionWork>(outcome))
        continue;
      const analysis::RootRegionWork *work =
          analysis::getRootRegionWork(outcome);
      if (!work) {
        if (failureReason)
          *failureReason = std::visit(
              [](const auto &value) -> std::string {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, analysis::RootRegionWork> ||
                              std::is_same_v<T, analysis::NoRootRegionWork>)
                  return {};
                else
                  return value.detail;
              },
              outcome);
        return mlir::failure();
      }
      rootWorks.push_back(*work);
    }
  }
  compiler::detail::CanonicalRegionPlanOutcome regionOutcome =
      compiler::detail::buildCanonicalRegionPlan(rootWorks);
  const compiler::detail::RegionPlan *regions =
      compiler::detail::getRegionPlan(regionOutcome);
  if (!regions) {
    if (failureReason)
      *failureReason =
          std::get<compiler::detail::BrokenRegionPlan>(regionOutcome).detail;
    return mlir::failure();
  }
  compiler::detail::CanonicalTemporalPlanOutcome temporalOutcome =
      compiler::detail::buildCanonicalTemporalPlan(*regions, rootWorks);
  const compiler::detail::TemporalPlan *temporal =
      compiler::detail::getTemporalPlan(temporalOutcome);
  if (!temporal) {
    if (failureReason)
      *failureReason =
          std::get<compiler::detail::BrokenTemporalPlan>(temporalOutcome)
              .detail;
    return mlir::failure();
  }
  return CanonicalPlanningPrefix{coordinate->assignment, *demand,
                                 std::move(rootWorks), *regions, *temporal};
}

std::string buildFlashDecodingPlanningFixture(int64_t queryExtent,
                                              int64_t keyValueExtent) {
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (m, k2)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @decode(
      %query: tensor<2x)mlir"
         << queryExtent << "x128xf16>, %key: tensor<2x" << keyValueExtent
         << "x128xf16>,\n"
         << "      %value: tensor<2x" << keyValueExtent
         << "x64xf16>, %scale: f32,\n"
         << "      %mask: tensor<" << queryExtent << "x" << keyValueExtent
         << "xf16>) -> tensor<2x" << queryExtent << "x64xf16> {\n"
         << "    %out = tensor.empty() : tensor<2x" << queryExtent
         << "x64xf16>\n"
         << "    %result = wafer.linalg_ext.attention\n"
         << "        ins(%query, %key, %value, %scale, %mask :\n"
         << "            tensor<2x" << queryExtent << "x128xf16>, tensor<2x"
         << keyValueExtent << "x128xf16>,\n"
         << "            tensor<2x" << keyValueExtent << "x64xf16>, f32, "
         << "tensor<" << queryExtent << "x" << keyValueExtent << "xf16>)\n"
         << "        outs(%out : tensor<2x" << queryExtent << "x64xf16>)\n"
         << "        algorithm(<flash_decoding>)\n"
         << "        indexing_maps = [#q, #k, #v, #s, #mask, #o]\n"
         << "        -> tensor<2x" << queryExtent << "x64xf16>\n"
         << "    return %result : tensor<2x" << queryExtent << "x64xf16>\n"
         << "  }\n"
         << "}\n";
  return source;
}

std::string buildFlashAttentionPlanningFixture(int64_t queryExtent,
                                               int64_t keyValueExtent,
                                               bool withMask) {
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (m, k2)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @prefill(
      %query: tensor<2x)mlir"
         << queryExtent << "x128xf16>, %key: tensor<2x" << keyValueExtent
         << "x128xf16>,\n"
         << "      %value: tensor<2x" << keyValueExtent
         << "x64xf16>, %scale: f32";
  if (withMask)
    stream << ", %mask: tensor<" << queryExtent << "x" << keyValueExtent
           << "xf16>";
  stream << ") -> tensor<2x" << queryExtent << "x64xf16> {\n"
         << "    %out = tensor.empty() : tensor<2x" << queryExtent
         << "x64xf16>\n"
         << "    %result = wafer.linalg_ext.attention\n"
         << "        ins(%query, %key, %value, %scale";
  if (withMask)
    stream << ", %mask";
  stream << " : tensor<2x" << queryExtent << "x128xf16>, tensor<2x"
         << keyValueExtent << "x128xf16>, tensor<2x" << keyValueExtent
         << "x64xf16>, f32";
  if (withMask)
    stream << ", tensor<" << queryExtent << "x" << keyValueExtent << "xf16>";
  stream << ")\n"
         << "        outs(%out : tensor<2x" << queryExtent << "x64xf16>)\n"
         << "        algorithm(<flash_attention>)\n"
         << "        indexing_maps = [#q, #k, #v, #s";
  if (withMask)
    stream << ", #mask";
  stream << ", #o]\n"
         << "        -> tensor<2x" << queryExtent << "x64xf16>\n"
         << "    return %result : tensor<2x" << queryExtent << "x64xf16>\n"
         << "  }\n"
         << "}\n";
  return source;
}

namespace {

std::string buildRank4AttentionPlanningFixture(
    int64_t batchExtent, int64_t headExtent, int64_t queryExtent,
    int64_t keyValueExtent, int64_t queryKeyExtent, int64_t valueExtent,
    bool withMask, bool decoding) {
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << R"mlir(
#q = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k1)>
#k = affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, k1)>
#v = affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, n)>
#s = affine_map<(b, h, m, k1, k2, n) -> ()>
#mask = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k2)>
#o = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, n)>
module {
  func.func @)mlir"
         << (decoding ? "decode" : "prefill") << "(\n"
         << "      %query: tensor<" << batchExtent << "x" << headExtent << "x"
         << queryExtent << "x" << queryKeyExtent << "xf16>,\n"
         << "      %key: tensor<" << batchExtent << "x" << headExtent << "x"
         << keyValueExtent << "x" << queryKeyExtent << "xf16>,\n"
         << "      %value: tensor<" << batchExtent << "x" << headExtent << "x"
         << keyValueExtent << "x" << valueExtent
         << "xf16>, %scale: f32";
  if (withMask)
    stream << ",\n      %mask: tensor<" << batchExtent << "x" << headExtent
           << "x" << queryExtent << "x" << keyValueExtent << "xf16>";
  stream << ") -> tensor<" << batchExtent << "x" << headExtent << "x"
         << queryExtent << "x" << valueExtent << "xf16> {\n"
         << "    %out = tensor.empty() : tensor<" << batchExtent << "x"
         << headExtent << "x" << queryExtent << "x" << valueExtent
         << "xf16>\n"
         << "    %result = wafer.linalg_ext.attention\n"
         << "        ins(%query, %key, %value, %scale";
  if (withMask)
    stream << ", %mask";
  stream << " :\n            tensor<" << batchExtent << "x" << headExtent << "x"
         << queryExtent << "x" << queryKeyExtent << "xf16>, tensor<"
         << batchExtent << "x" << headExtent << "x" << keyValueExtent << "x"
         << queryKeyExtent << "xf16>,\n            tensor<" << batchExtent << "x"
         << headExtent << "x" << keyValueExtent << "x" << valueExtent
         << "xf16>, f32";
  if (withMask)
    stream << ", tensor<" << batchExtent << "x" << headExtent << "x"
           << queryExtent << "x" << keyValueExtent << "xf16>";
  stream << ")\n"
         << "        outs(%out : tensor<" << batchExtent << "x" << headExtent
         << "x" << queryExtent << "x" << valueExtent << "xf16>)\n"
         << "        algorithm(<"
         << (decoding ? "flash_decoding" : "flash_attention") << ">)\n"
         << "        indexing_maps = [#q, #k, #v, #s";
  if (withMask)
    stream << ", #mask";
  stream << ", #o]\n"
         << "        -> tensor<" << batchExtent << "x" << headExtent << "x"
         << queryExtent << "x" << valueExtent << "xf16>\n"
         << "    return %result : tensor<" << batchExtent << "x" << headExtent
         << "x" << queryExtent << "x" << valueExtent << "xf16>\n"
         << "  }\n"
         << "}\n";
  return source;
}

} // namespace

std::string buildRank4FlashAttentionPlanningFixture(
    int64_t batchExtent, int64_t headExtent, int64_t queryExtent,
    int64_t keyValueExtent, int64_t queryKeyExtent, int64_t valueExtent,
    bool withMask) {
  return buildRank4AttentionPlanningFixture(
      batchExtent, headExtent, queryExtent, keyValueExtent, queryKeyExtent,
      valueExtent, withMask, /*decoding=*/false);
}

std::string buildRank4FlashDecodingPlanningFixture(
    int64_t batchExtent, int64_t headExtent, int64_t queryExtent,
    int64_t keyValueExtent, int64_t queryKeyExtent, int64_t valueExtent,
    bool withMask) {
  return buildRank4AttentionPlanningFixture(
      batchExtent, headExtent, queryExtent, keyValueExtent, queryKeyExtent,
      valueExtent, withMask, /*decoding=*/true);
}

std::string buildMultiK2FlashDecodingPlanningFixture() {
  return R"mlir(
#q = affine_map<(b, m, k1, k20, k21, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k20, k21, n) -> (b, k20, k21, k1)>
#v = affine_map<(b, m, k1, k20, k21, n) -> (b, k20, k21, n)>
#s = affine_map<(b, m, k1, k20, k21, n) -> ()>
#mask = affine_map<(b, m, k1, k20, k21, n) -> (m, k20, k21)>
#o = affine_map<(b, m, k1, k20, k21, n) -> (b, m, n)>
module {
  func.func @decode(
      %query: tensor<2x1025x128xf16>, %key: tensor<2x33x31x128xf16>,
      %value: tensor<2x33x31x64xf16>, %scale: f32,
      %mask: tensor<1025x33x31xf16>) -> tensor<2x1025x64xf16> {
    %out = tensor.empty() : tensor<2x1025x64xf16>
    %result = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale, %mask :
            tensor<2x1025x128xf16>, tensor<2x33x31x128xf16>,
            tensor<2x33x31x64xf16>, f32, tensor<1025x33x31xf16>)
        outs(%out : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>)
        indexing_maps = [#q, #k, #v, #s, #mask, #o]
        -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}
)mlir";
}

std::string buildTwoFlashAttentionPlanningFixture() {
  return R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @two_roots(
      %query: tensor<2x1025x128xf16>, %key: tensor<2x1031x128xf16>,
      %value: tensor<2x1031x64xf16>, %scale: f32)
      -> (tensor<2x1025x64xf16>, tensor<2x1025x64xf16>) {
    %left_out = tensor.empty() : tensor<2x1025x64xf16>
    %left = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%left_out : tensor<2x1025x64xf16>)
        algorithm(<flash_attention>)
        indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x64xf16>
    %right_out = tensor.empty() : tensor<2x1025x64xf16>
    %right = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%right_out : tensor<2x1025x64xf16>)
        algorithm(<flash_attention>)
        indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x64xf16>
    return %left, %right : tensor<2x1025x64xf16>, tensor<2x1025x64xf16>
  }
}
)mlir";
}

} // namespace wafer::test
