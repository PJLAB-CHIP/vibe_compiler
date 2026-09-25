//===- TensorSubsetInterpreter.h - Independent subset oracle ---*- C++ -*-===//
#ifndef WAFER_TESTSUPPORT_TRANSFORMS_TENSORSUBSETINTERPRETER_H
#define WAFER_TESTSUPPORT_TRANSFORMS_TENSORSUBSETINTERPRETER_H
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "gtest/gtest.h"
#include <functional>
#include <limits>
#include <numeric>
#include <vector>
namespace wafer::test {
// An independent bounded interpreter for element identity, not floating point
// arithmetic. Empty elements are poison; every observed element must be read
// from an actual source. Subsets use ordinary row-major coordinate iteration.
struct TensorSubsetInterpreter {
  using Tensor = std::vector<int64_t>;
  static constexpr int64_t poison = std::numeric_limits<int64_t>::min();
  llvm::DenseMap<mlir::Value, Tensor> tensors;
  llvm::DenseMap<mlir::Value, int64_t> inductions;
  llvm::DenseSet<mlir::Operation *> sourceReads;
  std::vector<Tensor> observations;
  uint64_t dynamicCopies = 0;
  std::function<void(mlir::linalg::GenericOp, TensorSubsetInterpreter &)>
      compute;

  int64_t index(mlir::OpFoldResult input) {
    if (auto c = mlir::getConstantIntValue(input))
      return *c;
    auto value = mlir::cast<mlir::Value>(input);
    if (auto found = inductions.find(value); found != inductions.end())
      return found->second;
    if (auto apply = value.getDefiningOp<mlir::affine::AffineApplyOp>()) {
      llvm::SmallVector<mlir::Attribute> operands, folded;
      for (mlir::Value operand : apply.getMapOperands())
        operands.push_back(mlir::IntegerAttr::get(
            mlir::IndexType::get(value.getContext()), index(operand)));
      EXPECT_TRUE(
          mlir::succeeded(apply.getAffineMap().constantFold(operands, folded)));
      return mlir::cast<mlir::IntegerAttr>(folded.front()).getInt();
    }
    auto *op = value.getDefiningOp();
    if (op && op->getNumOperands() == 2) {
      int64_t lhs = index(op->getOperand(0)), rhs = index(op->getOperand(1));
      if (mlir::isa<mlir::arith::AddIOp>(op))
        return lhs + rhs;
      if (mlir::isa<mlir::arith::SubIOp>(op))
        return lhs - rhs;
      if (mlir::isa<mlir::arith::OrIOp>(op))
        return value.getType().isInteger(1) ? (lhs != 0) || (rhs != 0)
                                            : lhs | rhs;
      if (mlir::isa<mlir::arith::MulIOp>(op))
        return lhs * rhs;
      if (mlir::isa<mlir::arith::AndIOp>(op))
        return value.getType().isInteger(1) ? (lhs != 0) && (rhs != 0)
                                            : lhs & rhs;
      if (mlir::isa<mlir::arith::XOrIOp>(op))
        return value.getType().isInteger(1) ? (lhs != 0) != (rhs != 0)
                                            : lhs ^ rhs;
      if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
        switch (cmp.getPredicate()) {
        case mlir::arith::CmpIPredicate::eq:
          return lhs == rhs;
        case mlir::arith::CmpIPredicate::sge:
          return lhs >= rhs;
        case mlir::arith::CmpIPredicate::slt:
          return lhs < rhs;
        default:
          break;
        }
      }
    }
    ADD_FAILURE() << "unsupported interpreter index operation";
    return 0;
  }

  std::vector<size_t> subset(llvm::ArrayRef<int64_t> shape,
                             llvm::ArrayRef<mlir::OpFoldResult> offsets,
                             llvm::ArrayRef<mlir::OpFoldResult> sizes,
                             llvm::ArrayRef<mlir::OpFoldResult> strides) {
    std::vector<size_t> result{0};
    for (unsigned axis = 0; axis < shape.size(); ++axis) {
      std::vector<size_t> next;
      int64_t offset = index(offsets[axis]), size = index(sizes[axis]),
              stride = index(strides[axis]);
      for (size_t prefix : result)
        for (int64_t i = 0; i < size; ++i) {
          int64_t coordinate = offset + i * stride;
          EXPECT_GE(coordinate, 0);
          EXPECT_LT(coordinate, shape[axis]);
          next.push_back(prefix * shape[axis] + coordinate);
        }
      result = std::move(next);
    }
    return result;
  }

  void run(mlir::Block &body) {
    for (mlir::Operation &op : body) {
      if (::testing::Test::HasFatalFailure())
        return;
      if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(op)) {
        tensors[empty] = Tensor(empty.getType().getNumElements(), poison);
      } else if (auto extract =
                     mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op)) {
        auto coordinates = subset(
            extract.getSourceType().getShape(), extract.getMixedOffsets(),
            extract.getMixedSizes(), extract.getMixedStrides());
        Tensor value;
        const auto &source = tensors[extract.getSource()];
        for (size_t at : coordinates) {
          ASSERT_LT(at, source.size());
          if (sourceReads.contains(&op)) {
            ASSERT_NE(source[at], poison);
          }
          value.push_back(source[at]);
        }
        tensors[extract] = std::move(value);
        dynamicCopies += sourceReads.contains(&op);
      } else if (auto insert =
                     mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op)) {
        Tensor value = tensors[insert.getDest()];
        const auto &source = tensors[insert.getSource()];
        auto coordinates =
            subset(insert.getType().getShape(), insert.getMixedOffsets(),
                   insert.getMixedSizes(), insert.getMixedStrides());
        ASSERT_EQ(coordinates.size(), source.size());
        for (auto [i, at] : llvm::enumerate(coordinates)) {
          ASSERT_LT(at, value.size());
          value[at] = source[i];
        }
        tensors[insert] = std::move(value);
      } else if (mlir::isa<mlir::tensor::CollapseShapeOp,
                           mlir::tensor::ExpandShapeOp, mlir::tensor::CastOp>(
                     op)) {
        Tensor value = tensors[op.getOperand(0)];
        tensors[op.getResult(0)] = std::move(value);
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        std::vector<Tensor> carried;
        for (mlir::Value initial : loop.getInitArgs())
          carried.push_back(tensors[initial]);
        int64_t step = index(loop.getStep());
        ASSERT_GT(step, 0);
        for (int64_t i = index(loop.getLowerBound());
             i < index(loop.getUpperBound()); i += step) {
          inductions[loop.getInductionVar()] = i;
          for (auto [argument, value] :
               llvm::zip_equal(loop.getRegionIterArgs(), carried))
            tensors[argument] = value;
          run(*loop.getBody());
          auto yield =
              mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
          for (auto [i, value] : llvm::enumerate(yield.getResults()))
            carried[i] = tensors[value];
        }
        for (auto [result, value] : llvm::zip_equal(loop.getResults(), carried))
          tensors[result] = value;
      } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        auto &region = index(branch.getCondition()) ? branch.getThenRegion()
                                                    : branch.getElseRegion();
        ASSERT_FALSE(region.empty());
        run(region.front());
        auto yield =
            mlir::cast<mlir::scf::YieldOp>(region.front().getTerminator());
        for (auto [result, value] :
             llvm::zip_equal(branch.getResults(), yield.getResults())) {
          Tensor carried = tensors[value];
          tensors[result] = std::move(carried);
        }
      } else if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op)) {
        ASSERT_TRUE(bool(compute));
        compute(generic, *this);
      } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        for (mlir::Value operand : call.getOperands())
          if (mlir::isa<mlir::RankedTensorType>(operand.getType())) {
            const auto &value = tensors[operand];
            ASSERT_FALSE(llvm::is_contained(value, poison));
            observations.push_back(value);
          }
      }
    }
  }

  void run(mlir::func::FuncOp function) {
    for (auto [i, argument] : llvm::enumerate(function.getArguments())) {
      auto type = mlir::cast<mlir::RankedTensorType>(argument.getType());
      Tensor value(type.getNumElements());
      std::iota(value.begin(), value.end(), int64_t(i + 1) * 1000000);
      tensors[argument] = std::move(value);
    }
    run(function.getBody().front());
  }
};

} // namespace wafer::test
#endif
