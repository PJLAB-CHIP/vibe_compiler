//===- Passes.td - Linalg-level transforms ----------------*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_PASSES_TD
#define WAFER_TRANSFORMS_LINALG_PASSES_TD

def FoldStaticTensorOpsPass
    : Pass<"wafer-fold-static-tensor-ops", "mlir::ModuleOp"> {
  let summary =
      "Evaluate bounded static tensor and structured Linalg expressions";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::tensor::TensorDialect"];
}

def FormAttentionOpsPass
    : Pass<"wafer-form-attention-ops", "mlir::func::FuncOp"> {
  let summary =
      "Form self-contained attention ops from proven structured tensor graphs";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::math::MathDialect",
                           "mlir::tensor::TensorDialect",
                           "wafer::WaferDialect"];
}

def NormalizeStructuredTensorGraphPass
    : Pass<"wafer-normalize-structured-tensor-graph", "mlir::func::FuncOp"> {
  let summary =
      "Normalize exact pure Tensor/Linalg access graphs with bounded equality saturation";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::math::MathDialect",
                           "mlir::tensor::TensorDialect"];
  let options = [
    Option<"maximumRelationQueries", "maximum-relation-queries", "uint64_t",
           /*default=*/"8192",
           "Maximum IndexRelation service calls admitted by one component request">,
    Option<"maximumENodes", "maximum-e-nodes", "uint64_t",
           /*default=*/"4096",
           "Maximum e-nodes admitted by one structured component request">,
    Option<"maximumMatches", "maximum-matches", "uint64_t",
           /*default=*/"8192",
           "Maximum exact rewrite matches admitted by one component request">,
    Option<"maximumIterations", "maximum-iterations", "uint32_t",
           /*default=*/"8",
           "Maximum deterministic equality-saturation iterations">
  ];
  let statistics = [
    Statistic<"numComponents", "components",
              "Number of structured components sent to the e-graph">,
    Statistic<"numInputOperations", "input-operations",
              "Number of operations before structured graph normalization">,
    Statistic<"numOutputOperations", "output-operations",
              "Number of operations after structured graph normalization">,
    Statistic<"numChangedComponents", "changed-components",
              "Number of components replaced by a strictly dominating graph">,
    Statistic<"numMultiRuleChangedComponents", "multi-rule-changed-components",
              "Number of changed components in which multiple dynamic rule kinds applied">,
    Statistic<"numBudgetExhaustedComponents", "budget-exhausted-components",
              "Number of components preserved after deterministic budget exhaustion">,
    Statistic<"numENodes", "e-nodes", "Number of explored e-nodes">,
    Statistic<"numEClasses", "e-classes", "Number of explored e-classes">,
    Statistic<"numRewriteMatches", "rewrite-matches",
              "Number of admitted exact rewrite matches">,
    Statistic<"numEClassMerges", "e-class-merges",
              "Number of successful e-class merges">,
    Statistic<"numRebuildWork", "rebuild-work",
              "Number of deterministic egg rebuild work items">,
    Statistic<"numIterations", "iterations",
              "Number of equality-saturation iterations">,
    Statistic<"numExtractionWork", "extraction-work",
              "Number of extractor cost evaluations">,
    Statistic<"numABIInputRecords", "abi-input-records",
              "Number of C ABI input node, child and relation records">,
    Statistic<"numABIOutputRecords", "abi-output-records",
              "Number of C ABI output node, child and relation records">,
    Statistic<"numABIInputBytes", "abi-input-bytes",
              "Number of serialized C ABI input bytes">,
    Statistic<"numABIOutputBytes", "abi-output-bytes",
              "Number of serialized C ABI output bytes">,
    Statistic<"numAccessTransformsRemoved", "access-transforms-removed",
              "Number of explicit access-transform occurrences removed">,
    Statistic<"numMultiUseAccessPropagations", "multi-use-access-propagations",
              "Number of shared projected accesses propagated to all supported Linalg data users">,
    Statistic<"numConcatTransformsRemoved", "concat-transforms-removed",
              "Number of concat assembly occurrences removed">,
    Statistic<"numRelationQueries", "relation-queries",
              "Number of request-local IndexRelation service calls">,
    Statistic<"numIdentityApplications", "identity-applications",
              "Number of Access identity rule applications">,
    Statistic<"numCompositionApplications", "composition-applications",
              "Number of Access composition rule applications">,
    Statistic<"numComputeAbsorptionApplications", "compute-absorption-applications",
              "Number of Compute operand Access absorption applications">,
    Statistic<"numConcatApplications", "concat-applications",
              "Number of Concat normalization rule applications">,
    Statistic<"numResultReindexApplications", "result-reindex-applications",
              "Number of restricted Compute result reindex applications">
  ];
}

#endif // WAFER_TRANSFORMS_LINALG_PASSES_TD
