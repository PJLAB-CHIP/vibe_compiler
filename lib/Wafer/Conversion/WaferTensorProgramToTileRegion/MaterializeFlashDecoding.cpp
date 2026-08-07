//===- MaterializeFlashDecoding.cpp - Decode split-KV materialization ---===//

#include "Internal.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"

using namespace wafer;

namespace wafer {
#define GEN_PASS_DEF_MATERIALIZEFLASHDECODINGPASS
#include "Wafer/Transforms/WaferPasses.h.inc"
} // namespace wafer

namespace wafer {
namespace {

static mlir::FailureOr<llvm::SmallVector<int64_t, 6>>
parseOutputTileSizes(llvm::StringRef text) {
  llvm::SmallVector<int64_t, 6> result;
  text = text.trim();
  if (text.empty())
    return mlir::failure();
  while (!text.empty()) {
    auto [part, rest] = text.split(',');
    int64_t value = 0;
    if (part.trim().empty() || part.trim().getAsInteger(10, value) ||
        value <= 0)
      return mlir::failure();
    result.push_back(value);
    text = rest;
  }
  return result;
}

struct MaterializeFlashDecodingPass
    : public impl::MaterializeFlashDecodingPassBase<
          MaterializeFlashDecodingPass> {
  using impl::MaterializeFlashDecodingPassBase<
      MaterializeFlashDecodingPass>::MaterializeFlashDecodingPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    auto tileSizes = parseOutputTileSizes(outputTileSizes);
    if (mlir::failed(tileSizes) || keyValueTileSize <= 0 || splitCount <= 1) {
      module.emitError()
          << "flash_decoding_materialization_failed: output and K/V tile "
             "sizes must be positive and split count must exceed one";
      return signalPassFailure();
    }

    mlir::func::FuncOp function =
        tensor_program_to_tile_region::findSingleStandaloneTensorProgram(
            module);
    if (!function) {
      module.emitError()
          << "flash_decoding_materialization_failed: expected one "
             "standalone tensor program";
      return signalPassFailure();
    }
    std::string failureReason;
    tensor_program_to_tile_region::TensorProgramScope scope(function);
    llvm::SmallVector<int64_t, 2> reductionTileSizes{keyValueTileSize,
                                                     splitCount};
    if (mlir::failed(
            tensor_program_to_tile_region::materializeCompleteFlashTraversal(
                scope, *tileSizes, reductionTileSizes,
                tensor_program_to_tile_region::AttentionImplementationKind::
                    SplitKV,
                &failureReason))) {
      module.emitError() << "flash_decoding_materialization_failed: "
                         << failureReason;
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace wafer
