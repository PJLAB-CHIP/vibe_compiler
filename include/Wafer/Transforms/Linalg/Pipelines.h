//===- Pipelines.h - Linalg transform pipelines -------------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_PIPELINES_H
#define WAFER_TRANSFORMS_LINALG_PIPELINES_H

namespace mlir {
class OpPassManager;
}

namespace wafer {

void buildDecomposeOnlineAttentionPipeline(mlir::OpPassManager &pm);

} // namespace wafer

#endif // WAFER_TRANSFORMS_LINALG_PIPELINES_H
