//===- FormGroups.cpp - Form tensor-level Wafer groups -------------------===//

#include "Wafer/Transforms/Passes.h"

#include "Support/AttentionGemmUtils.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace wafer {
namespace {

static bool isTensorValue(mlir::Value value) {
  return mlir::isa<mlir::TensorType>(value.getType());
}

static bool hasStaticMismatch(int64_t lhs, int64_t rhs) {
  return lhs != mlir::ShapedType::kDynamic &&
         rhs != mlir::ShapedType::kDynamic && lhs != rhs;
}

static std::optional<mlir::Attribute>
getScalarConstantAttr(mlir::Attribute attr) {
  if (mlir::isa<mlir::FloatAttr, mlir::IntegerAttr>(attr))
    return attr;
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return std::nullopt;
  return dense.getSplatValue<mlir::Attribute>();
}

static std::optional<mlir::Attribute> getScalarConstantAttr(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    return getScalarConstantAttr(constant.getValue());
  if (auto extract = value.getDefiningOp<mlir::tensor::ExtractOp>()) {
    if (extract.getIndices().empty())
      return getScalarConstantAttr(extract.getTensor());
  }
  return std::nullopt;
}

static std::optional<mlir::Attribute>
getReduceInitValueAttr(mlir::Value output) {
  auto fill = output.getDefiningOp<mlir::linalg::FillOp>();
  if (!fill || fill.getInputs().size() != 1)
    return std::nullopt;
  return getScalarConstantAttr(fill.getInputs()[0]);
}

static bool canFormSingleMatmulGroup(mlir::linalg::MatmulOp matmul) {
  if (matmul->getParentOfType<wafer::GroupOp>())
    return false;
  if (matmul->getNumResults() != 1 || matmul.getOutputs().size() != 1)
    return false;
  if (!llvm::all_of(matmul.getInputs(), isTensorValue))
    return false;
  if (!llvm::all_of(matmul.getOutputs(), isTensorValue))
    return false;
  return true;
}

static bool canFormSingleAttentionGemmGroup(mlir::linalg::GenericOp generic) {
  if (generic->getParentOfType<wafer::GroupOp>())
    return false;
  if (!matchAttentionGemm(generic))
    return false;
  if (!llvm::all_of(generic.getDpsInputs(), isTensorValue))
    return false;
  if (!llvm::all_of(generic.getDpsInits(), isTensorValue))
    return false;
  return true;
}

static std::optional<wafer::ComputeElementwiseKind>
mapElementwiseKind(mlir::linalg::ElementwiseKind kind) {
  switch (kind) {
  case mlir::linalg::ElementwiseKind::add:
    return wafer::ComputeElementwiseKind::Add;
  case mlir::linalg::ElementwiseKind::sub:
    return wafer::ComputeElementwiseKind::Sub;
  case mlir::linalg::ElementwiseKind::mul:
    return wafer::ComputeElementwiseKind::Mul;
  case mlir::linalg::ElementwiseKind::div:
    return wafer::ComputeElementwiseKind::Div;
  case mlir::linalg::ElementwiseKind::max_signed:
    return wafer::ComputeElementwiseKind::Max;
  case mlir::linalg::ElementwiseKind::min_signed:
    return wafer::ComputeElementwiseKind::Min;
  case mlir::linalg::ElementwiseKind::negf:
    return wafer::ComputeElementwiseKind::Neg;
  case mlir::linalg::ElementwiseKind::reciprocal:
    return wafer::ComputeElementwiseKind::Recip;
  case mlir::linalg::ElementwiseKind::sqrt:
    return wafer::ComputeElementwiseKind::Sqrt;
  case mlir::linalg::ElementwiseKind::rsqrt:
    return wafer::ComputeElementwiseKind::Rsqrt;
  case mlir::linalg::ElementwiseKind::exp:
    return wafer::ComputeElementwiseKind::Exp;
  case mlir::linalg::ElementwiseKind::tanh:
    return wafer::ComputeElementwiseKind::Tanh;
  default:
    return std::nullopt;
  }
}

static bool areBlockArguments(mlir::Value lhs, mlir::Value rhs,
                              mlir::BlockArgument arg0,
                              mlir::BlockArgument arg1) {
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

static std::optional<wafer::ComputeReduceKind>
mapReduceKind(mlir::linalg::ReduceOp reduce) {
  if (reduce->getNumResults() != 1 || reduce.getInputs().size() != 1 ||
      reduce.getInits().size() != 1 || reduce.getRegion().empty())
    return std::nullopt;

  mlir::Block &body = reduce.getRegion().front();
  if (body.getNumArguments() != 2 || !body.getTerminator() ||
      body.getTerminator()->getNumOperands() != 1)
    return std::nullopt;

  mlir::Value yielded = body.getTerminator()->getOperand(0);
  if (auto add = yielded.getDefiningOp<mlir::arith::AddFOp>())
    if (areBlockArguments(add->getOperand(0), add->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Sum;
  if (auto add = yielded.getDefiningOp<mlir::arith::AddIOp>())
    if (areBlockArguments(add->getOperand(0), add->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Sum;
  if (auto max = yielded.getDefiningOp<mlir::arith::MaximumFOp>())
    if (areBlockArguments(max->getOperand(0), max->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Max;
  if (auto max = yielded.getDefiningOp<mlir::arith::MaxSIOp>())
    if (areBlockArguments(max->getOperand(0), max->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Max;
  if (auto min = yielded.getDefiningOp<mlir::arith::MinimumFOp>())
    if (areBlockArguments(min->getOperand(0), min->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Min;
  if (auto min = yielded.getDefiningOp<mlir::arith::MinSIOp>())
    if (areBlockArguments(min->getOperand(0), min->getOperand(1),
                          body.getArgument(0), body.getArgument(1)))
      return wafer::ComputeReduceKind::Min;

  return std::nullopt;
}

static bool canFormSingleReduceGroup(mlir::linalg::ReduceOp reduce) {
  if (reduce->getParentOfType<wafer::GroupOp>())
    return false;
  if (!mapReduceKind(reduce))
    return false;
  if (reduce->getNumResults() != 1 || reduce.getInputs().size() != 1 ||
      reduce.getInits().size() != 1)
    return false;
  if (!llvm::all_of(reduce.getInputs(), isTensorValue) ||
      !llvm::all_of(reduce.getInits(), isTensorValue))
    return false;
  if (!getReduceInitValueAttr(reduce.getInits()[0]))
    return false;

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(reduce.getInputs()[0].getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(reduce->getResult(0).getType());
  if (!inputType || !resultType || reduce.getInits()[0].getType() != resultType)
    return false;
  if (inputType.getElementType() != resultType.getElementType())
    return false;

  llvm::DenseSet<int64_t> reducedDims;
  for (int64_t dim : reduce.getDimensions()) {
    if (dim < 0 || dim >= inputType.getRank())
      return false;
    if (!reducedDims.insert(dim).second)
      return false;
  }
  if (resultType.getRank() !=
      inputType.getRank() - static_cast<int64_t>(reducedDims.size()))
    return false;

  int64_t resultDim = 0;
  for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
    if (reducedDims.contains(inputDim))
      continue;
    if (hasStaticMismatch(inputType.getDimSize(inputDim),
                          resultType.getDimSize(resultDim)))
      return false;
    ++resultDim;
  }
  return true;
}

static bool
canFormSingleElementwiseGroup(mlir::linalg::ElementwiseOp elementwise) {
  if (elementwise->getParentOfType<wafer::GroupOp>())
    return false;
  if (!mapElementwiseKind(elementwise.getKind()))
    return false;
  if (elementwise->getNumResults() != 1 || elementwise.getOutputs().size() != 1)
    return false;
  if (!llvm::all_of(elementwise.getInputs(), isTensorValue))
    return false;
  if (!llvm::all_of(elementwise.getOutputs(), isTensorValue))
    return false;

  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      elementwise->getResult(0).getType());
  if (!resultType || elementwise.getOutputs()[0].getType() != resultType)
    return false;

  llvm::SmallVector<mlir::AffineMap> maps = elementwise.getIndexingMapsArray();
  if (maps.size() !=
      elementwise.getInputs().size() + elementwise.getOutputs().size())
    return false;
  mlir::AffineMap resultMap = maps.back();
  if (resultMap.getNumDims() != resultType.getRank() ||
      resultMap.getNumSymbols() != 0 || !resultMap.isIdentity())
    return false;

  for (auto [index, input] : llvm::enumerate(elementwise.getInputs())) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || inputType.getElementType() != resultType.getElementType())
      return false;

    mlir::AffineMap inputMap = maps[index];
    if (inputMap.getNumDims() != resultType.getRank() ||
        inputMap.getNumSymbols() != 0 ||
        inputMap.getNumResults() != inputType.getRank() ||
        !inputMap.isProjectedPermutation())
      return false;

    for (auto [dim, expr] : llvm::enumerate(inputMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr || dimExpr.getPosition() >= resultType.getRank())
        return false;
      if (hasStaticMismatch(inputType.getDimSize(dim),
                            resultType.getDimSize(dimExpr.getPosition())))
        return false;
    }
  }

  for (mlir::Value input : elementwise.getInputs()) {
    if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
      return false;
  }
  return true;
}

static void addMappedBlockArguments(mlir::Block *block, mlir::ValueRange values,
                                    mlir::IRMapping &mapping) {
  for (mlir::Value value : values) {
    mlir::BlockArgument arg =
        block->addArgument(value.getType(), value.getLoc());
    mapping.map(value, arg);
  }
}

static void formGroup(mlir::Operation *root, mlir::ValueRange rootInputs,
                      mlir::ValueRange rootOuts) {
  llvm::SmallVector<mlir::Value> inputs(rootInputs.begin(), rootInputs.end());
  llvm::SmallVector<mlir::Value> outs(rootOuts.begin(), rootOuts.end());

  mlir::OpBuilder builder(root);
  auto group = builder.create<wafer::GroupOp>(
      root->getLoc(), root->getResultTypes(), inputs, outs);

  mlir::Block *body = new mlir::Block();
  group.getBody().push_back(body);

  mlir::IRMapping mapping;
  addMappedBlockArguments(body, inputs, mapping);
  addMappedBlockArguments(body, outs, mapping);

  mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(body);
  mlir::Operation *cloned = bodyBuilder.clone(*root, mapping);
  bodyBuilder.create<wafer::GroupYieldOp>(root->getLoc(), cloned->getResults());

  root->replaceAllUsesWith(group->getResults());
  root->erase();
}

struct FormGroupsPass
    : public mlir::PassWrapper<FormGroupsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormGroupsPass)

  llvm::StringRef getArgument() const final { return "wafer-form-groups"; }

  llvm::StringRef getDescription() const final {
    return "form conservative tensor-level Wafer logical groups";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, wafer::WaferDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::Operation *> roots;
    getOperation().walk([&](mlir::Operation *op) {
      if (auto matmul = mlir::dyn_cast<mlir::linalg::MatmulOp>(op)) {
        if (canFormSingleMatmulGroup(matmul))
          roots.push_back(op);
        return;
      }
      if (auto reduce = mlir::dyn_cast<mlir::linalg::ReduceOp>(op)) {
        if (canFormSingleReduceGroup(reduce))
          roots.push_back(op);
        return;
      }
      if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op)) {
        if (canFormSingleAttentionGemmGroup(generic))
          roots.push_back(op);
        return;
      }
      if (auto elementwise = mlir::dyn_cast<mlir::linalg::ElementwiseOp>(op)) {
        if (canFormSingleElementwiseGroup(elementwise))
          roots.push_back(op);
      }
    });

    for (mlir::Operation *root : roots) {
      if (auto matmul = mlir::dyn_cast<mlir::linalg::MatmulOp>(root))
        formGroup(root, matmul.getInputs(), matmul.getOutputs());
      else if (auto reduce = mlir::dyn_cast<mlir::linalg::ReduceOp>(root))
        formGroup(root, reduce.getInputs(), reduce.getInits());
      else if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(root)) {
        llvm::SmallVector<mlir::Value> inputs = generic.getDpsInputs();
        formGroup(root, inputs, generic.getDpsInits());
      } else if (auto elementwise =
                     mlir::dyn_cast<mlir::linalg::ElementwiseOp>(root))
        formGroup(root, elementwise.getInputs(), elementwise.getOutputs());
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createFormGroupsPass() {
  return std::make_unique<FormGroupsPass>();
}

} // namespace wafer
