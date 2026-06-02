//===- wafer-opt.cpp - Wafer optimizer driver ----------------------------===//

#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/InitAll.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef WAFER_ENABLE_SHARDY
#include "shardy/dialect/sdy/transforms/passes.h"
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  wafer::registerImporterDialects(registry);
  wafer::registerWaferTransformPasses();
  wafer::registerWaferPipelines();
  mlir::func::registerInlinerExtension(registry);
#ifdef WAFER_ENABLE_SHARDY
  mlir::sdy::registerAllSdyPassesAndPipelines();
#endif

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Wafer optimizer driver\n", registry));
}
