//===- ProfileInstrumentationModel.cpp - Profiler instrumentation model -===//

#include "Wafer/Package/Profile/ProfileInstrumentationModel.h"

#include "Wafer/Target/Core/TargetCall.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"

#include <optional>
#include <string>
#include <utility>

namespace wafer::runtime {

void emitStaticCostMetric(llvm::json::OStream &json,
                          const ProfileStaticCostMetric &metric) {
  json.object([&] {
    json.attribute("knowledge", metric.knowledge);
    if (metric.value)
      json.attribute("value", std::to_string(*metric.value));
    else
      json.attribute("value", llvm::json::Value(nullptr));
    json.attribute("reason", metric.reason);
  });
}


void emitStaticTileWork(llvm::json::OStream &json,
                        const ProfileStaticTileWork &work) {
  auto emitMetric = [&](llvm::StringRef name,
                        const ProfileStaticCostMetric &metric) {
    json.attributeBegin(name);
    emitStaticCostMetric(json, metric);
    json.attributeEnd();
  };
  json.object([&] {
    emitMetric("npu_f16_bf16_logical_ops", work.npuF16Bf16LogicalOps);
    emitMetric("npu_other_logical_ops", work.npuOtherLogicalOps);
    emitMetric("vector_f16_bf16_logical_ops", work.vectorF16Bf16LogicalOps);
    emitMetric("vector_f32_logical_ops", work.vectorF32LogicalOps);
    emitMetric("vector_other_logical_ops", work.vectorOtherLogicalOps);
    emitMetric("ddr_read_bytes", work.ddrReadBytes);
    emitMetric("ddr_write_bytes", work.ddrWriteBytes);
    emitMetric("spm_movement_bytes", work.spmMovementBytes);
    emitMetric("noc_transmit_bytes", work.nocTransmitBytes);
    emitMetric("noc_receive_bytes", work.nocReceiveBytes);
    json.attributeObject("directional_noc_transmit_bytes", [&] {
      emitMetric("north", work.directionalNoCTransmitBytes.north);
      emitMetric("east", work.directionalNoCTransmitBytes.east);
      emitMetric("south", work.directionalNoCTransmitBytes.south);
      emitMetric("west", work.directionalNoCTransmitBytes.west);
    });
  });
}


void emitStaticCostModel(llvm::json::OStream &json,
                         const ProfileStaticCostModel &model) {
  json.object([&] {
    json.attribute("model", model.model);
    json.attribute("scope", model.scope);
    json.attributeObject("rates", [&] {
      json.attribute("card_ddr_bytes_per_second",
                     model.rates.cardDDRBytesPerSecond);
      json.attribute("directional_noc_bytes_per_second",
                     model.rates.directionalNoCBytesPerSecond);
      json.attribute("f16_bf16_npu_logical_ops_per_second_per_tile",
                     model.rates.f16Bf16NPULogicalOpsPerSecondPerTile);
      json.attribute("f16_bf16_vector_logical_ops_per_second_per_tile",
                     model.rates.f16Bf16VectorLogicalOpsPerSecondPerTile);
      json.attribute("f32_vector_logical_ops_per_second_per_tile",
                     model.rates.f32VectorLogicalOpsPerSecondPerTile);
      if (model.rates.spmMovementBytesPerSecond)
        json.attribute("spm_movement_bytes_per_second",
                       *model.rates.spmMovementBytesPerSecond);
      else
        json.attribute("spm_movement_bytes_per_second",
                       llvm::json::Value(nullptr));
    });
    json.attributeArray("tiles", [&] {
      std::vector<const ProfileStaticTileCost *> tiles;
      tiles.reserve(model.tiles.size());
      for (const ProfileStaticTileCost &tile : model.tiles)
        tiles.push_back(&tile);
      llvm::sort(tiles, [](const auto *lhs, const auto *rhs) {
        return lhs->launchSlot.getValue() < rhs->launchSlot.getValue();
      });
      for (const ProfileStaticTileCost *tile : tiles)
        json.object([&] {
          json.attribute("card_id", tile->cardId.getValue());
          json.attribute("tile_id", tile->tileId.getValue());
          json.attribute("launch_slot", tile->launchSlot.getValue());
          json.attributeBegin("work");
          emitStaticTileWork(json, tile->work);
          json.attributeEnd();
        });
    });
  });
}


llvm::StringRef stringifyProfileCaptureKind(ProfileCaptureKind capture) {
  switch (capture) {
  case ProfileCaptureKind::Count:
    return "count";
  case ProfileCaptureKind::Trace:
    return "trace";
  }
  llvm_unreachable("unknown profile capture kind");
}

llvm::StringRef stringifyProfileTSMEngine(ProfileTSMEngine engine) {
  switch (engine) {
  case ProfileTSMEngine::CT:
    return "CT";
  case ProfileTSMEngine::NE:
    return "NE";
  case ProfileTSMEngine::RDMA:
    return "RDMA";
  case ProfileTSMEngine::WDMA:
    return "WDMA";
  case ProfileTSMEngine::TDMA:
    return "TDMA";
  case ProfileTSMEngine::DirectDTE:
    return "DIRECT_DTE";
  }
  llvm_unreachable("unknown profile TSM engine");
}

llvm::StringRef stringifyProfileTargetSiteKind(ProfileTargetSiteKind kind) {
  switch (kind) {
  case ProfileTargetSiteKind::NCCCommand:
    return "ncc-command";
  case ProfileTargetSiteKind::NCCCompletion:
    return "ncc-completion";
  case ProfileTargetSiteKind::DirectDTEControl:
    return "direct-dte-control";
  case ProfileTargetSiteKind::DirectDTEIssue:
    return "direct-dte-issue";
  case ProfileTargetSiteKind::DirectDTEWait:
    return "direct-dte-wait";
  }
  llvm_unreachable("unknown profile target site kind");
}

void writeProfileStaticCostModel(llvm::json::OStream &json,
                                 const ProfileStaticCostModel &model) {
  emitStaticCostModel(json, model);
}

ProfileTargetSiteKind
getProfileTargetSiteKind(const TargetCallDescriptor &descriptor) {
  if (std::optional<TargetCallTSMEngine> engine =
          getTargetCallTSMEngine(descriptor)) {
    if (*engine != TargetCallTSMEngine::DirectDTE)
      return ProfileTargetSiteKind::NCCCommand;
    const auto *builtin = std::get_if<TargetCallBuiltin>(&descriptor.semantic);
    if (!builtin)
      llvm_unreachable("Direct-DTE target call is not a builtin");
    if (*builtin == TargetCallBuiltin::DirectDTESendIssue)
      return ProfileTargetSiteKind::DirectDTEIssue;
    if (*builtin == TargetCallBuiltin::DirectDTEWait)
      return ProfileTargetSiteKind::DirectDTEWait;
    llvm_unreachable("unknown Direct-DTE engine target call");
  }

  const auto *builtin = std::get_if<TargetCallBuiltin>(&descriptor.semantic);
  if (!builtin)
    llvm_unreachable(
        "non-builtin target call without an NCC issue-domain engine");
  switch (*builtin) {
  case TargetCallBuiltin::NCCJoin:
    return ProfileTargetSiteKind::NCCCompletion;
  case TargetCallBuiltin::DirectDTEBegin:
  case TargetCallBuiltin::DirectDTEBeginAfterPrepare:
  case TargetCallBuiltin::DirectDTESendPrepare:
  case TargetCallBuiltin::DirectDTERecvPrepare:
  case TargetCallBuiltin::DirectDTEFinish:
    return ProfileTargetSiteKind::DirectDTEControl;
  case TargetCallBuiltin::RDMA:
  case TargetCallBuiltin::WDMA:
  case TargetCallBuiltin::GatherScatter:
  case TargetCallBuiltin::Memset:
  case TargetCallBuiltin::Bit2FP:
  case TargetCallBuiltin::MaskMove:
  case TargetCallBuiltin::Gemm:
  case TargetCallBuiltin::GemmOriented:
  case TargetCallBuiltin::TDMAPad:
  case TargetCallBuiltin::TDMAImg2Col:
  case TargetCallBuiltin::DirectDTESendIssue:
  case TargetCallBuiltin::DirectDTEWait:
    llvm_unreachable("engine target call lost its typed issue domain");
  }
  llvm_unreachable("unknown target-call builtin");
}

} // namespace wafer::runtime
