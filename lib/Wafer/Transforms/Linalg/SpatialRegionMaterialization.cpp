//===- SpatialRegionMaterialization.cpp - Selected structural IR --------===//

#include "Wafer/Transforms/Linalg/SpatialRegionMaterialization.h"

#include "OnlineAttentionMaterialization.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Linalg/StructuredTiling.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer {
namespace {

using compiler::detail::DemandFragmentId;
using compiler::detail::ExecutionInstanceId;
using compiler::detail::LocalUseBinding;
using compiler::detail::materializeOnlineAttentionFinalize;
using compiler::detail::materializeOnlineAttentionStateMerge;
using compiler::detail::materializeOnlineAttentionTile;
using compiler::detail::OnlineAttentionState;
using compiler::detail::RegionExecutionId;
using compiler::detail::RegionGroupPlan;
using compiler::detail::ReplicaExecutionId;
using compiler::detail::RequiredMergeExecution;
using compiler::detail::RequiredRootExecution;

template <typename T>
mlir::FailureOr<T> fail(SpatialRegionMaterializationFailure *failure,
                        SpatialRegionMaterializationFailureKind kind,
                        llvm::StringRef detail) {
  if (failure) {
    failure->kind = kind;
    failure->detail = detail.str();
  }
  return mlir::failure();
}

void recordFailure(SpatialRegionMaterializationFailure *failure,
                   SpatialRegionMaterializationFailureKind kind,
                   llvm::StringRef detail) {
  if (!failure)
    return;
  failure->kind = kind;
  failure->detail = detail.str();
}

struct SourceValueKey {
  enum class Kind : uint8_t { FunctionArgument, StructuredResult };
  Kind kind = Kind::FunctionArgument;
  uint32_t owner = 0;
  unsigned result = 0;

  friend bool operator<(const SourceValueKey &lhs, const SourceValueKey &rhs) {
    return std::tie(lhs.kind, lhs.owner, lhs.result) <
           std::tie(rhs.kind, rhs.owner, rhs.result);
  }
  friend bool operator==(const SourceValueKey &lhs, const SourceValueKey &rhs) {
    return lhs.kind == rhs.kind && lhs.owner == rhs.owner &&
           lhs.result == rhs.result;
  }
};

struct SourceSliceKey {
  SourceValueKey source;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;

  friend bool operator<(const SourceSliceKey &lhs, const SourceSliceKey &rhs) {
    return std::tie(lhs.source, lhs.offsets, lhs.sizes) <
           std::tie(rhs.source, rhs.offsets, rhs.sizes);
  }
};

struct ProducedValueKey {
  analysis::RootBoundaryId source;
  std::optional<compiler::detail::LogicalShardId> shard;
  std::optional<compiler::detail::ReductionGroupId> reductionGroup;
  TileId tile{0};

  friend bool operator<(const ProducedValueKey &lhs,
                        const ProducedValueKey &rhs) {
    return std::tie(lhs.source, lhs.shard, lhs.reductionGroup) <
               std::tie(rhs.source, rhs.shard, rhs.reductionGroup) ||
           (std::tie(lhs.source, lhs.shard, lhs.reductionGroup) ==
                std::tie(rhs.source, rhs.shard, rhs.reductionGroup) &&
            lhs.tile.getValue() < rhs.tile.getValue());
  }
};

struct GroupResult {
  ProducedValueKey key;
  unsigned resultNumber = 0;
};

struct GroupBoundaryInput {
  DemandFragmentId fragment;
  TileId sourceTile{0};
  mlir::Value destination;
};

struct CoupledStateKey {
  compiler::detail::ReductionGroupId group;
  compiler::detail::LogicalShardId contribution;
  CoupledReductionComponentKind component =
      CoupledReductionComponentKind::Maximum;

  friend bool operator<(const CoupledStateKey &lhs,
                        const CoupledStateKey &rhs) {
    return std::tie(lhs.group, lhs.contribution, lhs.component) <
           std::tie(rhs.group, rhs.contribution, rhs.component);
  }
};

struct CoupledStateBoundaryRequirement {
  TileId sourceTile{0};
  mlir::RankedTensorType type;
};

struct StandardPartialKey {
  compiler::detail::ReductionGroupId group;
  compiler::detail::LogicalShardId contribution;
  unsigned result = 0;

  friend bool operator<(const StandardPartialKey &lhs,
                        const StandardPartialKey &rhs) {
    return std::tie(lhs.group, lhs.contribution, lhs.result) <
           std::tie(rhs.group, rhs.contribution, rhs.result);
  }
};

struct StandardPartialBoundaryRequirement {
  TileId sourceTile{0};
  mlir::RankedTensorType type;
};

struct GroupStandardPartialResult {
  StandardPartialKey key;
  unsigned resultNumber = 0;
};

struct GroupStandardPartialInput {
  StandardPartialKey key;
  TileId sourceTile{0};
  mlir::Value destination;
};

struct GroupCoupledStateResult {
  CoupledStateKey key;
  unsigned resultNumber = 0;
};

struct GroupCoupledStateInput {
  CoupledStateKey key;
  TileId sourceTile{0};
  mlir::Value destination;
};

struct GroupArtifact {
  TileId tile{0};
  TileRegionOp region;
  llvm::SmallVector<GroupResult, 4> results;
  llvm::SmallVector<GroupBoundaryInput, 8> boundaryInputs;
  llvm::SmallVector<GroupCoupledStateResult, 8> coupledStateResults;
  llvm::SmallVector<GroupCoupledStateInput, 8> coupledStateInputs;
  llvm::SmallVector<GroupStandardPartialResult, 4> standardPartialResults;
  llvm::SmallVector<GroupStandardPartialInput, 4> standardPartialInputs;
};

const analysis::RootRegionWork *
findWork(llvm::ArrayRef<analysis::RootRegionWork> works,
         const analysis::RootRegionWorkId &id) {
  auto found = llvm::find_if(works, [&](const analysis::RootRegionWork &work) {
    return work.id == id;
  });
  return found == works.end() ? nullptr : &*found;
}

const analysis::RootExecutionWork *
findExecution(const analysis::RootRegionWork &work,
              const compiler::detail::LogicalShardId &shard) {
  auto found = llvm::find_if(work.execution, [&](const auto &execution) {
    return execution.shard == shard;
  });
  return found == work.execution.end() ? nullptr : &*found;
}

mlir::FailureOr<mlir::RankedTensorType> getCoupledStateType(
    const analysis::CoupledReductionComponentRequirement &component) {
  auto normalized = analysis::normalizeFiniteExactIndexSet(component.domain);
  if (mlir::failed(normalized) || normalized->getBoxes().size() != 1)
    return mlir::failure();
  auto box = normalized->getBoxes().front();
  if (box.sizes.empty() ||
      llvm::any_of(box.sizes, [](int64_t size) { return size <= 0; }))
    return mlir::failure();
  return mlir::RankedTensorType::get(box.sizes, component.elementType);
}

const compiler::detail::StructuredOperationNodeMapping *
findNode(llvm::ArrayRef<compiler::detail::StructuredOperationNodeMapping> nodes,
         mlir::Operation *operation) {
  auto found = llvm::find_if(
      nodes, [&](const auto &node) { return node.operation == operation; });
  return found == nodes.end() ? nullptr : &*found;
}

std::optional<llvm::SmallVector<int64_t, 6>>
getStaticValues(llvm::ArrayRef<mlir::OpFoldResult> values) {
  llvm::SmallVector<int64_t, 6> result;
  result.reserve(values.size());
  for (mlir::OpFoldResult value : values) {
    std::optional<int64_t> constant = mlir::getConstantIntValue(value);
    if (!constant)
      return std::nullopt;
    result.push_back(*constant);
  }
  return result;
}

std::optional<SourceValueKey> getSourceValueKey(
    mlir::Value value, mlir::func::FuncOp sourceFunction,
    llvm::ArrayRef<compiler::detail::StructuredOperationNodeMapping> nodes) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (argument.getOwner() == &sourceFunction.getBody().front())
      return SourceValueKey{SourceValueKey::Kind::FunctionArgument,
                            argument.getArgNumber(), 0};
    return std::nullopt;
  }
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  const auto *node = result ? findNode(nodes, result.getOwner()) : nullptr;
  if (!node)
    return std::nullopt;
  return SourceValueKey{SourceValueKey::Kind::StructuredResult,
                        node->structuredNodeId, result.getResultNumber()};
}

mlir::FailureOr<mlir::func::FuncOp>
getSourceFunction(mlir::ModuleOp source,
                  llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
                  SpatialRegionMaterializationFailure *failure) {
  mlir::func::FuncOp function;
  for (const analysis::RootRegionWork &work : rootWorks) {
    mlir::func::FuncOp candidate =
        work.rootOperation
            ? work.rootOperation->getParentOfType<mlir::func::FuncOp>()
            : mlir::func::FuncOp{};
    if (!candidate || candidate.isExternal())
      return fail<mlir::func::FuncOp>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "root work is not owned by a defined TensorProgram function");
    if (function && function != candidate)
      return fail<mlir::func::FuncOp>(
          failure, SpatialRegionMaterializationFailureKind::Unsupported,
          "one structural transaction cannot span multiple TensorProgram "
          "functions");
    function = candidate;
  }
  if (!function || !function.getBody().hasOneBlock())
    return fail<mlir::func::FuncOp>(
        failure, SpatialRegionMaterializationFailureKind::Unsupported,
        "structural materialization requires one single-block TensorProgram");
  return function;
}

bool isSharedTopLevelFact(mlir::Operation &operation,
                          mlir::func::FuncOp sourceFunction) {
  return &operation != sourceFunction.getOperation() &&
         !mlir::isa<mlir::func::FuncOp>(operation);
}

mlir::LogicalResult materializeTileFunctionClosure(
    mlir::ModuleOp source, mlir::func::FuncOp sourceFunction, TileModuleOp tile,
    SpatialRegionMaterializationFailure *failure) {
  std::map<std::string, mlir::func::FuncOp> sourceFunctions;
  for (mlir::func::FuncOp function : source.getOps<mlir::func::FuncOp>())
    sourceFunctions.emplace(function.getSymName().str(), function);

  std::set<std::string> pending;
  std::set<std::string> cloned;
  auto collectCalls = [&](mlir::Operation *root) {
    root->walk([&](mlir::func::CallOp call) {
      pending.insert(call.getCallee().str());
    });
  };
  collectCalls(tile);
  mlir::OpBuilder builder(&tile.getBody().front(),
                          tile.getBody().front().end());
  while (!pending.empty()) {
    std::string name = *pending.begin();
    pending.erase(pending.begin());
    if (!cloned.insert(name).second)
      continue;
    auto sourceSymbol = sourceFunctions.find(name);
    if (sourceSymbol == sourceFunctions.end() ||
        sourceSymbol->second == sourceFunction) {
      recordFailure(failure,
                    SpatialRegionMaterializationFailureKind::Unsupported,
                    "Tile execution references a function outside the "
                    "materializable helper closure");
      return mlir::failure();
    }
    mlir::IRMapping mapping;
    mlir::Operation *helper = builder.clone(*sourceSymbol->second, mapping);
    collectCalls(helper);
  }
  return mlir::success();
}

mlir::FailureOr<llvm::SmallVector<RegionExecutionId, 8>>
collectGroupExecutions(const RegionGroupPlan &group,
                       llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
                       SpatialRegionMaterializationFailure *failure) {
  llvm::SmallVector<RegionExecutionId, 8> executions;
  std::set<RegionExecutionId> seen;
  for (const auto &execution : group.executions) {
    if (!seen.insert(execution.id).second)
      return fail<llvm::SmallVector<RegionExecutionId, 8>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "RegionPlan repeats one required execution");
    executions.push_back(execution.id);
  }
  for (const auto &replica : group.replicas) {
    if (!seen.insert(replica.id).second)
      return fail<llvm::SmallVector<RegionExecutionId, 8>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "RegionPlan repeats one replica execution");
    executions.push_back(replica.id);
  }
  auto sourceOperation = [&](const RegionExecutionId &execution) {
    analysis::RootRegionWorkId id;
    if (const auto *required = std::get_if<ExecutionInstanceId>(&execution))
      id = std::visit([](const auto &entry) { return entry.work; },
                      required->source);
    else
      id = std::get<ReplicaExecutionId>(execution).producer.work;
    const analysis::RootRegionWork *work = findWork(rootWorks, id);
    return work ? work->rootOperation : nullptr;
  };
  llvm::stable_sort(executions, [&](const RegionExecutionId &lhs,
                                    const RegionExecutionId &rhs) {
    mlir::Operation *lhsOp = sourceOperation(lhs);
    mlir::Operation *rhsOp = sourceOperation(rhs);
    if (lhsOp && rhsOp && lhsOp->getBlock() == rhsOp->getBlock() &&
        lhsOp != rhsOp)
      return lhsOp->isBeforeInBlock(rhsOp);
    return lhs < rhs;
  });
  return executions;
}

mlir::LogicalResult
validateRegionChoice(llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
                     const compiler::detail::RegionPlan &regionPlan,
                     SpatialRegionMaterializationFailure *failure) {
  std::set<ExecutionInstanceId> expectedExecutions;
  std::set<DemandFragmentId> expectedFragments;
  std::map<analysis::RootRegionWorkId, const analysis::RootRegionWork *> works;
  for (const analysis::RootRegionWork &work : rootWorks) {
    works.emplace(work.id, &work);
    std::set<compiler::detail::LogicalShardId> shards;
    for (const analysis::RootExecutionWork &execution : work.execution) {
      if (execution.shard.root != work.id.root ||
          !shards.insert(execution.shard).second ||
          !expectedExecutions
               .insert(ExecutionInstanceId{
                   RequiredRootExecution{work.id, execution.shard}})
               .second) {
        recordFailure(failure,
                      SpatialRegionMaterializationFailureKind::BrokenContract,
                      "root work contains a duplicate or foreign execution");
        return mlir::failure();
      }
      for (const auto &interval : execution.iterationDomain)
        if (interval.offset < 0 || interval.size <= 0) {
          recordFailure(failure,
                        SpatialRegionMaterializationFailureKind::BrokenContract,
                        "root work contains an invalid iteration interval");
          return mlir::failure();
        }
    }
    std::set<compiler::detail::ReductionGroupId> merges;
    for (const analysis::ReductionMergeRequirement &merge : work.merges)
      if (merge.group.root != work.id.root || merge.mergeTile != work.id.tile ||
          !merges.insert(merge.group).second ||
          !expectedExecutions
               .insert(ExecutionInstanceId{
                   RequiredMergeExecution{work.id, merge.group}})
               .second) {
        recordFailure(failure,
                      SpatialRegionMaterializationFailureKind::BrokenContract,
                      "root work contains a duplicate or foreign merge");
        return mlir::failure();
      }

    for (const analysis::RootBoundaryWork &boundary : work.boundaries)
      for (const analysis::RootBoundaryUseWork &use : boundary.consumerUses) {
        if (use.eligibleFinalOwners.empty()) {
          expectedFragments.insert(
              {boundary.id, use.id, std::nullopt, std::nullopt, std::nullopt});
          continue;
        }
        for (const analysis::OwnerIntersection &owner : use.eligibleFinalOwners)
          expectedFragments.insert({boundary.id, use.id, owner.ownerShard,
                                    owner.reductionGroup, owner.tile});
      }
  }

  std::set<ExecutionInstanceId> actualExecutions;
  std::set<ReplicaExecutionId> actualReplicas;
  std::set<DemandFragmentId> actualFragments;
  for (const RegionGroupPlan &group : regionPlan.groups) {
    auto findConsumerWork = [&](const DemandFragmentId &fragment)
        -> const analysis::RootRegionWork * {
      auto found = llvm::find_if(group.mandatoryRoots, [&](const auto &id) {
        if (id.root != fragment.use.destinationShard.root)
          return false;
        auto work = works.find(id);
        return work != works.end() &&
               findExecution(*work->second, fragment.use.destinationShard) !=
                   nullptr;
      });
      if (found == group.mandatoryRoots.end())
        return nullptr;
      return works.at(*found);
    };

    for (const auto &execution : group.executions) {
      analysis::RootRegionWorkId workId = std::visit(
          [](const auto &source) { return source.work; }, execution.id.source);
      if (!llvm::is_contained(group.mandatoryRoots, workId) ||
          !actualExecutions.insert(execution.id).second) {
        recordFailure(failure,
                      SpatialRegionMaterializationFailureKind::BrokenContract,
                      "Region group repeats or imports a required execution");
        return mlir::failure();
      }
    }
    for (const auto &replica : group.replicas) {
      const analysis::RootRegionWork *producer =
          findWork(rootWorks, replica.id.producer.work);
      if (!producer || !findExecution(*producer, replica.id.producer.shard) ||
          !producer->rootOperation ||
          !mlir::isMemoryEffectFree(producer->rootOperation)) {
        recordFailure(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "explicit Region replica is not a pure selected execution");
        return mlir::failure();
      }
      if (!actualReplicas.insert(replica.id).second) {
        recordFailure(failure,
                      SpatialRegionMaterializationFailureKind::BrokenContract,
                      "RegionPlan repeats one explicit replica");
        return mlir::failure();
      }
    }
    for (const auto &binding : group.externalBindings) {
      if (!findConsumerWork(binding.fragment) ||
          !actualFragments.insert(binding.fragment).second) {
        recordFailure(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "external binding is duplicated or has no local consumer");
        return mlir::failure();
      }
    }
    for (const LocalUseBinding &binding : group.localBindings) {
      if (!findConsumerWork(binding.fragment) ||
          !actualFragments.insert(binding.fragment).second) {
        recordFailure(failure,
                      SpatialRegionMaterializationFailureKind::BrokenContract,
                      "local binding is duplicated or has no local consumer");
        return mlir::failure();
      }
      bool hasProducer = false;
      if (const auto *required =
              std::get_if<ExecutionInstanceId>(&binding.producer))
        hasProducer = llvm::any_of(group.executions, [&](const auto &entry) {
          return entry.id == *required;
        });
      else {
        const auto &replica = std::get<ReplicaExecutionId>(binding.producer);
        hasProducer = llvm::any_of(group.replicas, [&](const auto &entry) {
          return entry.id == replica;
        });
      }
      if (!hasProducer) {
        recordFailure(failure,
                      SpatialRegionMaterializationFailureKind::BrokenContract,
                      "local binding has no selected producer execution");
        return mlir::failure();
      }
    }
    for (const auto &replica : group.replicas)
      if (!llvm::any_of(group.localBindings, [&](const auto &binding) {
            const auto *selected =
                std::get_if<ReplicaExecutionId>(&binding.producer);
            return selected && *selected == replica.id &&
                   binding.fragment == replica.id.fragment;
          })) {
        recordFailure(failure,
                      SpatialRegionMaterializationFailureKind::BrokenContract,
                      "explicit replica has no local use binding");
        return mlir::failure();
      }

    if (group.mandatoryRoots.size() > 1) {
      std::map<analysis::RootRegionWorkId, std::set<analysis::RootRegionWorkId>>
          adjacency;
      for (const auto &work : group.mandatoryRoots)
        adjacency[work];
      for (const LocalUseBinding &binding : group.localBindings) {
        const analysis::RootRegionWork *consumer =
            findConsumerWork(binding.fragment);
        analysis::RootRegionWorkId producer;
        if (const auto *required =
                std::get_if<ExecutionInstanceId>(&binding.producer))
          producer = std::visit([](const auto &source) { return source.work; },
                                required->source);
        else
          producer =
              std::get<ReplicaExecutionId>(binding.producer).producer.work;
        if (!consumer || !adjacency.count(producer) || producer == consumer->id)
          continue;
        adjacency[producer].insert(consumer->id);
        adjacency[consumer->id].insert(producer);
      }
      std::set<analysis::RootRegionWorkId> reached{
          group.mandatoryRoots.front()};
      llvm::SmallVector<analysis::RootRegionWorkId, 8> worklist{
          group.mandatoryRoots.front()};
      while (!worklist.empty()) {
        analysis::RootRegionWorkId current = worklist.pop_back_val();
        for (const analysis::RootRegionWorkId &next : adjacency[current])
          if (reached.insert(next).second)
            worklist.push_back(next);
      }
      if (reached.size() != group.mandatoryRoots.size()) {
        recordFailure(failure,
                      SpatialRegionMaterializationFailureKind::BrokenContract,
                      "Region group is not connected by selected local uses");
        return mlir::failure();
      }
    }
  }
  if (actualExecutions != expectedExecutions ||
      actualFragments != expectedFragments) {
    recordFailure(failure,
                  SpatialRegionMaterializationFailureKind::BrokenContract,
                  "RegionPlan does not cover required executions and demand "
                  "fragments exactly");
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult materializeTileLocalSplatConstants(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::arith::ConstantOp, 8> constants;
  module.walk([&](mlir::arith::ConstantOp constant) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(constant.getType());
    auto value = mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
    if (type && type.hasStaticShape() && value && value.isSplat() &&
        constant->getParentOfType<TileRegionOp>())
      constants.push_back(constant);
  });
  mlir::IRRewriter rewriter(module.getContext());
  for (mlir::arith::ConstantOp constant : constants) {
    auto type = mlir::cast<mlir::RankedTensorType>(constant.getType());
    auto value = mlir::cast<mlir::DenseElementsAttr>(constant.getValue());
    auto scalarValue = value.getSplatValue<mlir::TypedAttr>();
    if (!scalarValue || scalarValue.getType() != type.getElementType())
      return mlir::failure();
    auto floatValue = mlir::dyn_cast<mlir::FloatAttr>(scalarValue);
    if (floatValue && floatValue.getValue().isExactlyValue(2.0)) {
      llvm::SmallVector<mlir::Value, 8> worklist{constant.getResult()};
      llvm::DenseSet<mlir::Value> visited;
      while (!worklist.empty()) {
        mlir::Value current = worklist.pop_back_val();
        if (!visited.insert(current).second)
          continue;
        for (mlir::OpOperand &use : current.getUses()) {
          mlir::Operation *user = use.getOwner();
          if (mlir::isa<mlir::tensor::ExtractSliceOp,
                        mlir::tensor::ExpandShapeOp,
                        mlir::tensor::CollapseShapeOp, mlir::tensor::CastOp>(
                  user)) {
            for (mlir::Value result : user->getResults())
              worklist.push_back(result);
            continue;
          }
          auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(user);
          if (!linalg)
            continue;
          auto indexedInputs = llvm::enumerate(linalg.getDpsInputOperands());
          auto input = llvm::find_if(indexedInputs, [&](auto indexed) {
            return indexed.value() == &use;
          });
          if (input == indexedInputs.end())
            continue;
          auto indexedInput = *input;
          if (indexedInput.index() >= linalg.getRegionInputArgs().size())
            continue;
          mlir::BlockArgument exponent =
              linalg.getRegionInputArgs()[indexedInput.index()];
          llvm::SmallVector<mlir::math::PowFOp, 2> powers;
          linalg->walk([&](mlir::math::PowFOp power) {
            if (power.getRhs() == exponent)
              powers.push_back(power);
          });
          for (mlir::math::PowFOp power : powers) {
            rewriter.setInsertionPoint(power);
            mlir::Value two = rewriter.create<mlir::arith::ConstantOp>(
                power.getLoc(), scalarValue);
            power->setOperand(1, two);
          }
        }
      }
    }
    rewriter.setInsertionPoint(constant);
    mlir::Value scalar = rewriter.create<mlir::arith::ConstantOp>(
        constant.getLoc(), scalarValue);
    mlir::Value empty = rewriter.create<mlir::tensor::EmptyOp>(
        constant.getLoc(), type.getShape(), type.getElementType(),
        type.getEncoding());
    mlir::Value fill =
        rewriter.create<mlir::linalg::FillOp>(constant.getLoc(), scalar, empty)
            .getResult(0);
    rewriter.replaceOp(constant, fill);
  }
  return mlir::success();
}

struct GroupBuilder {
  struct CompactSupportTile {
    mlir::Value value;
    llvm::SmallVector<int64_t, 4> offsets;
    llvm::SmallVector<int64_t, 4> sizes;
  };

  mlir::ModuleOp source;
  mlir::func::FuncOp sourceFunction;
  mlir::func::FuncOp entryFunction;
  mlir::OpBuilder &entryBuilder;
  const RegionGroupPlan &group;
  llvm::ArrayRef<compiler::detail::StructuredOperationNodeMapping> nodes;
  llvm::ArrayRef<analysis::RootRegionWork> rootWorks;
  const std::map<DemandFragmentId, ProducedValueKey> &fragmentSources;
  const std::map<ProducedValueKey, mlir::Value> &produced;
  const std::map<CoupledStateKey, CoupledStateBoundaryRequirement>
      &coupledStateBoundaryRequirements;
  const std::map<CoupledStateKey, mlir::Value> &producedCoupledStates;
  const std::map<StandardPartialKey, StandardPartialBoundaryRequirement>
      &standardPartialBoundaryRequirements;
  const std::map<StandardPartialKey, mlir::Value> &producedStandardPartials;
  std::map<SourceValueKey, mlir::BlockArgument> &entrySourceArguments;
  std::map<DemandFragmentId, mlir::BlockArgument> &entryExternalArguments;
  std::map<CoupledStateKey, mlir::BlockArgument> &entryCoupledStateArguments;
  std::map<StandardPartialKey, mlir::BlockArgument>
      &entryStandardPartialArguments;
  SpatialRegionMaterializationFailure *failure = nullptr;

  mlir::Region regionBody;
  mlir::Block *body = nullptr;
  mlir::OpBuilder builder;
  mlir::IRMapping mapping;
  std::map<SourceValueKey, mlir::BlockArgument> boundaryArguments;
  std::map<SourceSliceKey, mlir::BlockArgument> slicedBoundaryArguments;
  std::map<DemandFragmentId, mlir::BlockArgument> fragmentArguments;
  std::map<CoupledStateKey, mlir::BlockArgument> coupledStateArguments;
  std::map<StandardPartialKey, mlir::BlockArgument> standardPartialArguments;
  llvm::SmallVector<mlir::Value, 8> regionInputs;
  llvm::SmallVector<GroupBoundaryInput, 8> boundaryInputs;
  llvm::SmallVector<GroupCoupledStateInput, 8> coupledStateInputs;
  llvm::SmallVector<GroupStandardPartialInput, 4> standardPartialInputs;
  llvm::SmallVector<GroupResult, 4> results;
  llvm::SmallVector<GroupCoupledStateResult, 8> coupledStateResults;
  llvm::SmallVector<GroupStandardPartialResult, 4> standardPartialResults;
  std::map<RegionExecutionId, llvm::SmallVector<mlir::Value, 2>>
      executionResults;
  std::map<RegionExecutionId,
           llvm::SmallVector<std::pair<CoupledStateKey, mlir::Value>, 3>>
      executionCoupledStates;
  std::map<RegionExecutionId,
           llvm::SmallVector<std::pair<StandardPartialKey, mlir::Value>, 2>>
      executionStandardPartials;
  llvm::DenseMap<mlir::Value, CompactSupportTile> compactSupportTiles;

  GroupBuilder(
      mlir::ModuleOp source, mlir::func::FuncOp sourceFunction,
      mlir::func::FuncOp entryFunction, mlir::OpBuilder &entryBuilder,
      const RegionGroupPlan &group,
      llvm::ArrayRef<compiler::detail::StructuredOperationNodeMapping> nodes,
      llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
      const std::map<DemandFragmentId, ProducedValueKey> &fragmentSources,
      const std::map<ProducedValueKey, mlir::Value> &produced,
      const std::map<CoupledStateKey, CoupledStateBoundaryRequirement>
          &coupledStateBoundaryRequirements,
      const std::map<CoupledStateKey, mlir::Value> &producedCoupledStates,
      const std::map<StandardPartialKey, StandardPartialBoundaryRequirement>
          &standardPartialBoundaryRequirements,
      const std::map<StandardPartialKey, mlir::Value> &producedStandardPartials,
      std::map<SourceValueKey, mlir::BlockArgument> &entrySourceArguments,
      std::map<DemandFragmentId, mlir::BlockArgument> &entryExternalArguments,
      std::map<CoupledStateKey, mlir::BlockArgument>
          &entryCoupledStateArguments,
      std::map<StandardPartialKey, mlir::BlockArgument>
          &entryStandardPartialArguments,
      SpatialRegionMaterializationFailure *failure)
      : source(source), sourceFunction(sourceFunction),
        entryFunction(entryFunction), entryBuilder(entryBuilder), group(group),
        nodes(nodes), rootWorks(rootWorks), fragmentSources(fragmentSources),
        produced(produced),
        coupledStateBoundaryRequirements(coupledStateBoundaryRequirements),
        producedCoupledStates(producedCoupledStates),
        standardPartialBoundaryRequirements(
            standardPartialBoundaryRequirements),
        producedStandardPartials(producedStandardPartials),
        entrySourceArguments(entrySourceArguments),
        entryExternalArguments(entryExternalArguments),
        entryCoupledStateArguments(entryCoupledStateArguments),
        entryStandardPartialArguments(entryStandardPartialArguments),
        failure(failure), builder(source.getContext()) {
    regionBody.push_back(new mlir::Block());
    body = &regionBody.front();
    builder.setInsertionPointToStart(body);
  }

  mlir::BlockArgument addRegionInput(mlir::Value value) {
    regionInputs.push_back(value);
    return body->addArgument(value.getType(), value.getLoc());
  }

  mlir::BlockArgument getOrCreateEntrySourceArgument(const SourceValueKey &key,
                                                     mlir::Type type,
                                                     mlir::Location location) {
    auto found = entrySourceArguments.find(key);
    if (found != entrySourceArguments.end())
      return found->second;
    unsigned index = entryFunction.getNumArguments();
    entryFunction.insertArgument(index, type, mlir::DictionaryAttr{}, location);
    return entrySourceArguments
        .emplace(key, entryFunction.getBody().front().getArgument(index))
        .first->second;
  }

  mlir::FailureOr<mlir::Value> getOrCreateBoundary(mlir::Value sourceValue) {
    std::optional<SourceValueKey> key =
        getSourceValueKey(sourceValue, sourceFunction, nodes);
    if (!key)
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "external structured value has no stable source identity");
    auto found = boundaryArguments.find(*key);
    if (found != boundaryArguments.end())
      return found->second;

    if (key->kind == SourceValueKey::Kind::StructuredResult)
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "structured Region input has no selected demand fragment");
    mlir::Value entryArgument = getOrCreateEntrySourceArgument(
        *key, sourceValue.getType(), sourceValue.getLoc());
    mlir::BlockArgument argument = addRegionInput(entryArgument);
    boundaryArguments.emplace(*key, argument);
    return argument;
  }

  mlir::FailureOr<mlir::Value> getOrCreateSlicedBoundary(
      mlir::Value sourceValue,
      const analysis::StaticRectangularIndexSet &requested) {
    std::optional<SourceValueKey> sourceKey =
        getSourceValueKey(sourceValue, sourceFunction, nodes);
    auto sourceType =
        mlir::dyn_cast<mlir::RankedTensorType>(sourceValue.getType());
    if (!sourceKey ||
        sourceKey->kind != SourceValueKey::Kind::FunctionArgument ||
        !sourceType || !sourceType.hasStaticShape() ||
        requested.offsets.size() != static_cast<size_t>(sourceType.getRank()) ||
        requested.sizes.size() != static_cast<size_t>(sourceType.getRank()))
      return mlir::failure();
    for (auto [offset, size, extent] : llvm::zip_equal(
             requested.offsets, requested.sizes, sourceType.getShape())) {
      int64_t end = 0;
      if (offset < 0 || size <= 0 || llvm::AddOverflow(offset, size, end) ||
          end > extent)
        return mlir::failure();
    }

    SourceSliceKey key{
        *sourceKey,
        std::vector<int64_t>(requested.offsets.begin(),
                             requested.offsets.end()),
        std::vector<int64_t>(requested.sizes.begin(), requested.sizes.end())};
    auto found = slicedBoundaryArguments.find(key);
    if (found != slicedBoundaryArguments.end())
      return found->second;

    mlir::Value entryArgument = getOrCreateEntrySourceArgument(
        *sourceKey, sourceType, sourceValue.getLoc());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    for (auto [offset, size] :
         llvm::zip_equal(requested.offsets, requested.sizes)) {
      offsets.push_back(entryBuilder.getIndexAttr(offset));
      sizes.push_back(entryBuilder.getIndexAttr(size));
      strides.push_back(entryBuilder.getIndexAttr(1));
    }
    auto slice = entryBuilder.create<mlir::tensor::ExtractSliceOp>(
        sourceValue.getLoc(), entryArgument, offsets, sizes, strides);
    mlir::BlockArgument argument = addRegionInput(slice.getResult());
    slicedBoundaryArguments.emplace(std::move(key), argument);
    return argument;
  }

  mlir::FailureOr<mlir::Value>
  getOrCreateFragmentValue(mlir::Value sourceValue,
                           const DemandFragmentId &fragment) {
    auto local =
        llvm::find_if(group.localBindings, [&](const LocalUseBinding &binding) {
          return binding.fragment == fragment;
        });
    if (local != group.localBindings.end()) {
      auto produced = executionResults.find(local->producer);
      if (produced == executionResults.end() ||
          fragment.source.index >= produced->second.size())
        return fail<mlir::Value>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "local fragment has no actual producer value");
      mlir::Value value = produced->second[fragment.source.index];
      if (value.getType() != sourceValue.getType())
        return fail<mlir::Value>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "local fragment producer type differs from its source");
      return value;
    }

    if (fragment.source.kind == analysis::RootBoundaryKind::Constant ||
        fragment.source.kind == analysis::RootBoundaryKind::CapturedValue)
      return mapSupportValue(sourceValue);
    auto found = fragmentArguments.find(fragment);
    if (found != fragmentArguments.end())
      return found->second;
    std::optional<SourceValueKey> key =
        getSourceValueKey(sourceValue, sourceFunction, nodes);
    if (!key) {
      mlir::IRMapping supportMapping;
      std::function<mlir::FailureOr<mlir::Value>(mlir::Value)> mapSupport =
          [&](mlir::Value value) -> mlir::FailureOr<mlir::Value> {
        if (mlir::Value mapped = supportMapping.lookupOrNull(value))
          return mapped;
        if (mlir::isa<mlir::BlockArgument>(value)) {
          auto mapped = getOrCreateBoundary(value);
          if (mlir::succeeded(mapped))
            supportMapping.map(value, *mapped);
          return mapped;
        }
        auto result = mlir::dyn_cast<mlir::OpResult>(value);
        mlir::Operation *definition = result ? result.getOwner() : nullptr;
        if (!definition || !mlir::isMemoryEffectFree(definition))
          return mlir::failure();
        if (findNode(nodes, definition)) {
          auto mapped = getOrCreateFragmentValue(value, fragment);
          if (mlir::succeeded(mapped))
            supportMapping.map(value, *mapped);
          return mapped;
        }
        for (mlir::Value operand : definition->getOperands()) {
          auto mapped = mapSupport(operand);
          if (mlir::failed(mapped))
            return mlir::failure();
          supportMapping.map(operand, *mapped);
        }
        mlir::Operation *cloned = builder.clone(*definition, supportMapping);
        for (auto [oldResult, newResult] :
             llvm::zip_equal(definition->getResults(), cloned->getResults()))
          supportMapping.map(oldResult, newResult);
        return supportMapping.lookup(value);
      };
      auto mapped = mapSupport(sourceValue);
      if (mlir::succeeded(mapped))
        return mapped;
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "external fragment support chain cannot reach an actual endpoint");
    }

    if (key->kind == SourceValueKey::Kind::FunctionArgument) {
      mlir::FailureOr<mlir::Value> argument = getOrCreateBoundary(sourceValue);
      if (mlir::succeeded(argument))
        fragmentArguments.emplace(fragment,
                                  mlir::cast<mlir::BlockArgument>(*argument));
      return argument;
    }

    auto planned = fragmentSources.find(fragment);
    if (planned == fragmentSources.end())
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "external fragment has no selected producer endpoint");
    mlir::Value actualInput;
    if (planned->second.tile == group.tile) {
      auto current = produced.find(planned->second);
      if (current == produced.end())
        return fail<mlir::Value>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "same-Tile Region input was materialized before its producer");
      actualInput = current->second;
    } else {
      auto external = entryExternalArguments.find(fragment);
      if (external == entryExternalArguments.end()) {
        unsigned index = entryFunction.getNumArguments();
        auto attrs = mlir::DictionaryAttr::get(
            builder.getContext(),
            {builder.getNamedAttr(kWaferCrossTileBoundaryInputAttrName,
                                  builder.getUnitAttr())});
        entryFunction.insertArgument(index, sourceValue.getType(), attrs,
                                     sourceValue.getLoc());
        external =
            entryExternalArguments
                .emplace(fragment,
                         entryFunction.getBody().front().getArgument(index))
                .first;
      }
      actualInput = external->second;
    }
    mlir::BlockArgument argument = addRegionInput(actualInput);
    fragmentArguments.emplace(fragment, argument);
    if (planned->second.tile != group.tile)
      boundaryInputs.push_back({fragment, planned->second.tile, argument});
    return argument;
  }

  mlir::FailureOr<mlir::Value>
  getOrCreateCoupledStateBoundary(const CoupledStateKey &key) {
    auto existing = coupledStateArguments.find(key);
    if (existing != coupledStateArguments.end())
      return existing->second;
    auto required = coupledStateBoundaryRequirements.find(key);
    if (required == coupledStateBoundaryRequirements.end() ||
        !required->second.type)
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "coupled state has no required boundary type");

    mlir::Value actualInput;
    if (required->second.sourceTile == group.tile) {
      bool definedInCurrentRegion = false;
      for (const auto &[execution, states] : executionCoupledStates) {
        (void)execution;
        auto local = llvm::find_if(states, [&](const auto &entry) {
          return !(entry.first < key) && !(key < entry.first);
        });
        if (local != states.end()) {
          actualInput = local->second;
          definedInCurrentRegion = true;
          break;
        }
      }
      if (definedInCurrentRegion)
        return actualInput;
      auto current = producedCoupledStates.find(key);
      if (!actualInput && current != producedCoupledStates.end())
        actualInput = current->second;
      if (!actualInput)
        return fail<mlir::Value>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "same-Tile coupled state was materialized before its producer");
    } else {
      auto external = entryCoupledStateArguments.find(key);
      if (external == entryCoupledStateArguments.end()) {
        unsigned index = entryFunction.getNumArguments();
        auto attrs = mlir::DictionaryAttr::get(
            builder.getContext(),
            {builder.getNamedAttr(kWaferCrossTileBoundaryInputAttrName,
                                  builder.getUnitAttr())});
        entryFunction.insertArgument(index, required->second.type, attrs,
                                     source.getLoc());
        external =
            entryCoupledStateArguments
                .emplace(key,
                         entryFunction.getBody().front().getArgument(index))
                .first;
      }
      actualInput = external->second;
    }
    mlir::BlockArgument argument = addRegionInput(actualInput);
    coupledStateArguments.emplace(key, argument);
    if (required->second.sourceTile != group.tile)
      coupledStateInputs.push_back(
          {key, required->second.sourceTile, argument});
    return argument;
  }

  mlir::FailureOr<mlir::Value>
  getOrCreateStandardPartialBoundary(const StandardPartialKey &key) {
    auto existing = standardPartialArguments.find(key);
    if (existing != standardPartialArguments.end())
      return existing->second;
    auto required = standardPartialBoundaryRequirements.find(key);
    if (required == standardPartialBoundaryRequirements.end() ||
        !required->second.type)
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "standard partial has no required boundary type");

    mlir::Value actualInput;
    if (required->second.sourceTile == group.tile) {
      for (const auto &[execution, partials] : executionStandardPartials) {
        (void)execution;
        auto local = llvm::find_if(partials, [&](const auto &entry) {
          return !(entry.first < key) && !(key < entry.first);
        });
        if (local != partials.end()) {
          actualInput = local->second;
          break;
        }
      }
      if (actualInput)
        return actualInput;
      auto current = producedStandardPartials.find(key);
      if (current != producedStandardPartials.end())
        actualInput = current->second;
      if (!actualInput)
        return fail<mlir::Value>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "same-Tile standard partial was materialized before its producer");
    } else {
      auto external = entryStandardPartialArguments.find(key);
      if (external == entryStandardPartialArguments.end()) {
        unsigned index = entryFunction.getNumArguments();
        auto attrs = mlir::DictionaryAttr::get(
            builder.getContext(),
            {builder.getNamedAttr(kWaferCrossTileBoundaryInputAttrName,
                                  builder.getUnitAttr())});
        entryFunction.insertArgument(index, required->second.type, attrs,
                                     source.getLoc());
        external =
            entryStandardPartialArguments
                .emplace(key,
                         entryFunction.getBody().front().getArgument(index))
                .first;
      }
      actualInput = external->second;
    }
    mlir::BlockArgument argument = addRegionInput(actualInput);
    standardPartialArguments.emplace(key, argument);
    if (required->second.sourceTile != group.tile)
      standardPartialInputs.push_back(
          {key, required->second.sourceTile, argument});
    return argument;
  }

  const analysis::ExactIndexSet *
  findFragmentDomain(const DemandFragmentId &fragment) const {
    auto work = llvm::find_if(rootWorks, [&](const auto &candidate) {
      return candidate.id.root == fragment.use.destinationShard.root &&
             findExecution(candidate, fragment.use.destinationShard);
    });
    if (work == rootWorks.end())
      return nullptr;
    auto boundary = llvm::find_if(work->boundaries, [&](const auto &candidate) {
      return candidate.id == fragment.source;
    });
    if (boundary == work->boundaries.end())
      return nullptr;
    auto use =
        llvm::find_if(boundary->consumerUses, [&](const auto &candidate) {
          return candidate.id == fragment.use;
        });
    if (use == boundary->consumerUses.end())
      return nullptr;
    if (use->eligibleFinalOwners.empty())
      return use->requiredDomain ? &*use->requiredDomain : nullptr;
    auto owner = llvm::find_if(
        use->eligibleFinalOwners,
        [&](const analysis::OwnerIntersection &entry) {
          return entry.ownerShard == fragment.ownerShard &&
                 entry.reductionGroup == fragment.reductionGroup &&
                 (!fragment.ownerTile || entry.tile == *fragment.ownerTile);
        });
    if (owner == use->eligibleFinalOwners.end())
      return nullptr;
    return &owner->domain;
  }

  mlir::FailureOr<mlir::Value> materializeSupportAfterFragmentAssembly(
      mlir::Value value,
      const llvm::SmallVector<DemandFragmentId, 4> &fragments,
      mlir::IRMapping &supportMapping) {
    if (mlir::Value mapped = supportMapping.lookupOrNull(value))
      return mapped;
    if (std::optional<SourceValueKey> key =
            getSourceValueKey(value, sourceFunction, nodes)) {
      mlir::FailureOr<mlir::Value> mapped = mlir::failure();
      if (key->kind == SourceValueKey::Kind::FunctionArgument) {
        mapped = getOrCreateBoundary(value);
      } else {
        llvm::SmallVector<DemandFragmentId, 4> matchingFragments;
        for (const DemandFragmentId &fragment : fragments)
          for (const analysis::RootRegionWork &work : rootWorks) {
            auto boundary =
                llvm::find_if(work.boundaries,
                              [&](const analysis::RootBoundaryWork &candidate) {
                                return candidate.id == fragment.source &&
                                       candidate.sourceValue == value;
                              });
            if (boundary != work.boundaries.end()) {
              matchingFragments.push_back(fragment);
              break;
            }
          }
        if (!matchingFragments.empty())
          mapped = assembleFragments(value, matchingFragments);
      }
      if (mlir::succeeded(mapped))
        supportMapping.map(value, *mapped);
      return mapped;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *definition = result ? result.getOwner() : nullptr;
    if (!definition || !mlir::isMemoryEffectFree(definition))
      return mlir::failure();
    bool hasShapedOperand = false;
    for (mlir::Value operand : definition->getOperands()) {
      mlir::FailureOr<mlir::Value> mapped =
          mlir::isa<mlir::ShapedType>(operand.getType())
              ? materializeSupportAfterFragmentAssembly(operand, fragments,
                                                        supportMapping)
              : mapSupportValue(operand, supportMapping);
      if (mlir::failed(mapped))
        return mlir::failure();
      hasShapedOperand |= mlir::isa<mlir::ShapedType>(operand.getType());
      supportMapping.map(operand, *mapped);
    }
    if (!hasShapedOperand)
      return mapSupportValue(value, supportMapping);
    llvm::SetVector<mlir::Value> captures;
    mlir::getUsedValuesDefinedAbove(definition->getRegions(), captures);
    for (mlir::Value capture : captures)
      if (mlir::failed(mapSupportValue(capture, supportMapping)))
        return mlir::failure();
    mlir::Operation *cloned = builder.clone(*definition, supportMapping);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(definition->getResults(), cloned->getResults()))
      supportMapping.map(oldResult, newResult);
    return supportMapping.lookup(value);
  }

  mlir::FailureOr<mlir::Value>
  assembleFragments(mlir::Value sourceValue,
                    llvm::SmallVector<DemandFragmentId, 4> fragments) {
    if (!getSourceValueKey(sourceValue, sourceFunction, nodes)) {
      auto sourceResult = mlir::dyn_cast<mlir::OpResult>(sourceValue);
      mlir::Operation *definition =
          sourceResult ? sourceResult.getOwner() : nullptr;
      if (definition && mlir::isMemoryEffectFree(definition) &&
          llvm::none_of(definition->getOperandTypes(),
                        llvm::IsaPred<mlir::ShapedType>))
        return mapSupportValue(sourceValue);
      mlir::IRMapping supportMapping;
      auto mapped = materializeSupportAfterFragmentAssembly(
          sourceValue, fragments, supportMapping);
      if (mlir::succeeded(mapped))
        return mapped;
      std::string detail =
          "external support chain cannot be applied after fragment assembly";
      if (mlir::Operation *definition = sourceValue.getDefiningOp())
        detail += ": " + definition->getName().getStringRef().str();
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::Unsupported,
          detail);
    }
    const bool hadFragments = !fragments.empty();
    llvm::erase_if(fragments, [&](const DemandFragmentId &fragment) {
      const analysis::ExactIndexSet *domain = findFragmentDomain(fragment);
      return domain && domain->isEmpty();
    });
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(sourceValue.getType());
    if (fragments.empty() && hadFragments) {
      if (!tensorType || !tensorType.hasStaticShape())
        return fail<mlir::Value>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "exact-empty structural input requires a static ranked tensor");
      return builder
          .create<mlir::tensor::EmptyOp>(sourceValue.getLoc(),
                                         tensorType.getShape(),
                                         tensorType.getElementType())
          .getResult();
    }
    if (fragments.size() == 1)
      return getOrCreateFragmentValue(sourceValue, fragments.front());
    if (fragments.empty())
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "structural fragment assembly has no selected input");
    if (!tensorType || !tensorType.hasStaticShape())
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::Unsupported,
          "structural fan-in requires a static ranked tensor");
    auto empty = builder.create<mlir::tensor::EmptyOp>(
        sourceValue.getLoc(), tensorType.getShape(),
        tensorType.getElementType());
    mlir::Value assembled = empty.getResult();
    for (const DemandFragmentId &fragment : fragments) {
      const analysis::ExactIndexSet *domain = findFragmentDomain(fragment);
      if (!domain) {
        std::string detail =
            "external fragment has no exact current demand domain";
        if (mlir::Operation *definition = sourceValue.getDefiningOp())
          detail += ": " + definition->getName().getStringRef().str();
        return fail<mlir::Value>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            detail);
      }
      auto normalized = analysis::normalizeFiniteExactIndexSet(*domain);
      if (mlir::failed(normalized) || normalized->getBoxes().empty())
        return fail<mlir::Value>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "external fragment is not a finite rectangular union");
      mlir::FailureOr<mlir::Value> endpoint =
          getOrCreateFragmentValue(sourceValue, fragment);
      if (mlir::failed(endpoint))
        return mlir::failure();
      for (const auto &box : normalized->getBoxes()) {
        if (box.offsets.size() != static_cast<size_t>(tensorType.getRank()) ||
            box.sizes.size() != static_cast<size_t>(tensorType.getRank()))
          return fail<mlir::Value>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "external fragment rank does not match its tensor");
        llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
        llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
        llvm::SmallVector<mlir::OpFoldResult, 4> strides;
        for (auto [offset, size] : llvm::zip_equal(box.offsets, box.sizes)) {
          offsets.push_back(builder.getIndexAttr(offset));
          sizes.push_back(builder.getIndexAttr(size));
          strides.push_back(builder.getIndexAttr(1));
        }
        auto slice = builder.create<mlir::tensor::ExtractSliceOp>(
            sourceValue.getLoc(), *endpoint, offsets, sizes, strides);
        assembled = builder
                        .create<mlir::tensor::InsertSliceOp>(
                            sourceValue.getLoc(), slice.getResult(), assembled,
                            offsets, sizes, strides)
                        .getResult();
      }
    }
    return assembled;
  }

  mlir::FailureOr<mlir::Value> mapSupportValue(mlir::Value value,
                                               mlir::IRMapping &valueMapping) {
    if (mlir::Value mapped = valueMapping.lookupOrNull(value))
      return mapped;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::FailureOr<mlir::Value> boundary = getOrCreateBoundary(argument);
      if (mlir::succeeded(boundary))
        valueMapping.map(value, *boundary);
      return boundary;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *definition = result ? result.getOwner() : nullptr;
    if (!definition ||
        definition->getBlock() != &sourceFunction.getBody().front())
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::Unsupported,
          "support value escapes the single-block TensorProgram");
    if (findNode(nodes, definition)) {
      mlir::FailureOr<mlir::Value> boundary = getOrCreateBoundary(value);
      if (mlir::succeeded(boundary))
        valueMapping.map(value, *boundary);
      return boundary;
    }
    if (!mlir::isMemoryEffectFree(definition))
      return fail<mlir::Value>(
          failure, SpatialRegionMaterializationFailureKind::Unsupported,
          "support closure crosses an effectful operation");
    for (mlir::Value operand : definition->getOperands())
      if (mlir::failed(mapSupportValue(operand, valueMapping)))
        return mlir::failure();
    llvm::SetVector<mlir::Value> captures;
    mlir::getUsedValuesDefinedAbove(definition->getRegions(), captures);
    for (mlir::Value capture : captures)
      if (mlir::failed(mapSupportValue(capture, valueMapping)))
        return mlir::failure();
    mlir::Operation *cloned = builder.clone(*definition, valueMapping);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(definition->getResults(), cloned->getResults()))
      valueMapping.map(oldResult, newResult);
    return valueMapping.lookup(value);
  }

  mlir::FailureOr<mlir::Value> mapSupportValue(mlir::Value value) {
    return mapSupportValue(value, mapping);
  }

  llvm::SmallVector<const LocalUseBinding *, 2>
  findLocalBindings(const RegionExecutionId &consumer,
                    unsigned operandNumber) const {
    const compiler::detail::LogicalShardId *shard = nullptr;
    if (const auto *required = std::get_if<ExecutionInstanceId>(&consumer)) {
      const auto *root = std::get_if<RequiredRootExecution>(&required->source);
      shard = root ? &root->shard : nullptr;
    } else {
      shard = &std::get<ReplicaExecutionId>(consumer).producer.shard;
    }
    if (!shard)
      return {};
    llvm::SmallVector<const LocalUseBinding *, 2> bindings;
    for (const LocalUseBinding &binding : group.localBindings)
      if (binding.fragment.use.operand == operandNumber &&
          binding.fragment.use.destinationShard == *shard)
        bindings.push_back(&binding);
    return bindings;
  }

  llvm::SmallVector<const DemandFragmentId *, 2>
  findExternalBindings(const RegionExecutionId &consumer,
                       unsigned operandNumber, mlir::Value operandValue) const {
    const compiler::detail::LogicalShardId *shard = nullptr;
    if (const auto *required = std::get_if<ExecutionInstanceId>(&consumer)) {
      const auto *root = std::get_if<RequiredRootExecution>(&required->source);
      shard = root ? &root->shard : nullptr;
    } else {
      shard = &std::get<ReplicaExecutionId>(consumer).producer.shard;
    }
    llvm::SmallVector<const DemandFragmentId *, 2> result;
    if (!shard)
      return result;
    const bool directSource =
        getSourceValueKey(operandValue, sourceFunction, nodes).has_value();
    for (const auto &binding : group.externalBindings) {
      const DemandFragmentId &fragment = binding.fragment;
      if (fragment.use.operand != operandNumber ||
          fragment.use.destinationShard != *shard)
        continue;
      if (directSource) {
        bool sameCurrentSource = false;
        for (const analysis::RootRegionWork &work : rootWorks) {
          auto boundary =
              llvm::find_if(work.boundaries,
                            [&](const analysis::RootBoundaryWork &candidate) {
                              return candidate.id == fragment.source &&
                                     candidate.sourceValue == operandValue;
                            });
          if (boundary != work.boundaries.end()) {
            sameCurrentSource = true;
            break;
          }
        }
        if (!sameCurrentSource)
          continue;
      }
      result.push_back(&fragment);
    }
    return result;
  }

  mlir::FailureOr<mlir::Value> materializeCompactSupportTile(
      mlir::Value value, const analysis::StaticRectangularIndexSet &requested,
      llvm::ArrayRef<DemandFragmentId> fragments) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!type || !type.hasStaticShape() ||
        requested.offsets.size() != static_cast<size_t>(type.getRank()) ||
        requested.sizes.size() != static_cast<size_t>(type.getRank()))
      return mlir::failure();
    for (auto [offset, size, extent] :
         llvm::zip_equal(requested.offsets, requested.sizes, type.getShape())) {
      int64_t end = 0;
      if (offset < 0 || size <= 0 || llvm::AddOverflow(offset, size, end) ||
          end > extent)
        return mlir::failure();
    }

    if (value.getDefiningOp<mlir::tensor::EmptyOp>())
      return builder
          .create<mlir::tensor::EmptyOp>(value.getLoc(), requested.sizes,
                                         type.getElementType(),
                                         type.getEncoding())
          .getResult();

    if (auto inserted = value.getDefiningOp<mlir::tensor::InsertSliceOp>()) {
      auto staticOffsets = getStaticValues(inserted.getMixedOffsets());
      auto staticSizes = getStaticValues(inserted.getMixedSizes());
      auto staticStrides = getStaticValues(inserted.getMixedStrides());
      auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
          inserted.getSource().getType());
      if (!staticOffsets || !staticSizes || !staticStrides || !sourceType ||
          sourceType.getRank() != type.getRank() ||
          !llvm::equal(*staticSizes, sourceType.getShape()) ||
          !llvm::all_of(*staticStrides,
                        [](int64_t stride) { return stride == 1; }))
        return mlir::failure();
      mlir::FailureOr<mlir::Value> compact = materializeCompactSupportTile(
          inserted.getDest(), requested, fragments);
      if (mlir::failed(compact))
        return mlir::failure();

      analysis::StaticRectangularIndexSet sourceRequest;
      llvm::SmallVector<int64_t, 4> compactOffsets;
      bool intersects = true;
      for (auto [requestedOffset, requestedSize, insertedOffset, insertedSize] :
           llvm::zip_equal(requested.offsets, requested.sizes, *staticOffsets,
                           *staticSizes)) {
        int64_t requestedEnd = requestedOffset + requestedSize;
        int64_t insertedEnd = insertedOffset + insertedSize;
        int64_t begin = std::max(requestedOffset, insertedOffset);
        int64_t end = std::min(requestedEnd, insertedEnd);
        if (begin >= end) {
          intersects = false;
          break;
        }
        sourceRequest.offsets.push_back(begin - insertedOffset);
        sourceRequest.sizes.push_back(end - begin);
        compactOffsets.push_back(begin - requestedOffset);
      }
      if (!intersects)
        return compact;
      mlir::FailureOr<mlir::Value> sourceTile = materializeCompactSupportTile(
          inserted.getSource(), sourceRequest, fragments);
      if (mlir::failed(sourceTile))
        return mlir::failure();
      llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
      llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
      llvm::SmallVector<mlir::OpFoldResult, 4> strides;
      for (int64_t offset : compactOffsets)
        offsets.push_back(builder.getIndexAttr(offset));
      for (int64_t size : sourceRequest.sizes)
        sizes.push_back(builder.getIndexAttr(size));
      for (int64_t dimension = 0; dimension < type.getRank(); ++dimension)
        strides.push_back(builder.getIndexAttr(1));
      return builder
          .create<mlir::tensor::InsertSliceOp>(
              value.getLoc(), *sourceTile, *compact, offsets, sizes, strides)
          .getResult();
    }

    mlir::Value full;
    if (std::optional<SourceValueKey> key =
            getSourceValueKey(value, sourceFunction, nodes)) {
      if (key->kind == SourceValueKey::Kind::FunctionArgument) {
        return getOrCreateSlicedBoundary(value, requested);
      } else {
        llvm::SmallVector<DemandFragmentId, 4> matching;
        for (const DemandFragmentId &fragment : fragments)
          for (const analysis::RootRegionWork &work : rootWorks) {
            auto boundary =
                llvm::find_if(work.boundaries,
                              [&](const analysis::RootBoundaryWork &candidate) {
                                return candidate.id == fragment.source &&
                                       candidate.sourceValue == value;
                              });
            if (boundary != work.boundaries.end()) {
              matching.push_back(fragment);
              break;
            }
          }
        if (matching.empty())
          return mlir::failure();
        auto assembled = assembleFragments(value, matching);
        if (mlir::failed(assembled))
          return mlir::failure();
        full = *assembled;
      }
    } else {
      llvm::SmallVector<DemandFragmentId, 4> selectedFragments(
          fragments.begin(), fragments.end());
      mlir::IRMapping supportMapping;
      auto mapped = materializeSupportAfterFragmentAssembly(
          value, selectedFragments, supportMapping);
      if (mlir::failed(mapped))
        mapped = mapSupportValue(value);
      if (mlir::failed(mapped))
        return mlir::failure();
      full = *mapped;
    }
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    for (int64_t offset : requested.offsets)
      offsets.push_back(builder.getIndexAttr(offset));
    for (int64_t size : requested.sizes)
      sizes.push_back(builder.getIndexAttr(size));
    for (int64_t dimension = 0; dimension < type.getRank(); ++dimension)
      strides.push_back(builder.getIndexAttr(1));
    return builder
        .create<mlir::tensor::ExtractSliceOp>(value.getLoc(), full, offsets,
                                              sizes, strides)
        .getResult();
  }

  mlir::FailureOr<mlir::Value>
  mapExecutionOperand(const RegionExecutionId &consumer,
                      mlir::OpOperand &operand) {
    llvm::SmallVector<const DemandFragmentId *, 2> external =
        findExternalBindings(consumer, operand.getOperandNumber(),
                             operand.get());
    llvm::SmallVector<DemandFragmentId, 4> fragments;
    for (const LocalUseBinding *binding :
         findLocalBindings(consumer, operand.getOperandNumber()))
      fragments.push_back(binding->fragment);
    for (const DemandFragmentId *fragment : external)
      fragments.push_back(*fragment);
    const compiler::detail::LogicalShardId *destinationShard = nullptr;
    if (const auto *required = std::get_if<ExecutionInstanceId>(&consumer)) {
      const auto *root = std::get_if<RequiredRootExecution>(&required->source);
      destinationShard = root ? &root->shard : nullptr;
    } else {
      destinationShard = &std::get<ReplicaExecutionId>(consumer).producer.shard;
    }
    if (destinationShard &&
        operand.get().getDefiningOp<mlir::tensor::InsertSliceOp>()) {
      const analysis::RootRegionWork *work =
          findWork(rootWorks, analysis::RootRegionWorkId{destinationShard->root,
                                                         group.tile});
      auto operandWork =
          work ? llvm::find_if(work->operands,
                               [&](const auto &candidate) {
                                 return candidate.operand ==
                                        operand.getOperandNumber();
                               })
               : std::vector<analysis::RootOperandWork>::const_iterator{};
      if (work && operandWork != work->operands.end()) {
        auto use = llvm::find_if(operandWork->uses, [&](const auto &candidate) {
          return candidate.id.operand == operand.getOperandNumber() &&
                 candidate.id.destinationShard == *destinationShard;
        });
        if (use != operandWork->uses.end()) {
          auto normalized =
              analysis::normalizeFiniteExactIndexSet(use->operandDemand);
          if (mlir::succeeded(normalized) &&
              normalized->getBoxes().size() == 1) {
            const auto box = normalized->getBoxes().front();
            mlir::FailureOr<mlir::Value> compact =
                materializeCompactSupportTile(operand.get(), box, fragments);
            if (mlir::succeeded(compact)) {
              auto fullType =
                  mlir::cast<mlir::RankedTensorType>(operand.get().getType());
              mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
                  operand.get().getLoc(), fullType.getShape(),
                  fullType.getElementType(), fullType.getEncoding());
              llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
              llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
              llvm::SmallVector<mlir::OpFoldResult, 4> strides;
              for (int64_t offset : box.offsets)
                offsets.push_back(builder.getIndexAttr(offset));
              for (int64_t size : box.sizes)
                sizes.push_back(builder.getIndexAttr(size));
              for (int64_t dimension = 0; dimension < fullType.getRank();
                   ++dimension)
                strides.push_back(builder.getIndexAttr(1));
              mlir::Value full = builder
                                     .create<mlir::tensor::InsertSliceOp>(
                                         operand.get().getLoc(), *compact,
                                         empty, offsets, sizes, strides)
                                     .getResult();
              compactSupportTiles.try_emplace(
                  full, CompactSupportTile{*compact, box.offsets, box.sizes});
              return full;
            }
          }
        }
      }
    }
    if (!fragments.empty())
      return assembleFragments(operand.get(), std::move(fragments));
    return mapSupportValue(operand.get());
  }

  mlir::LogicalResult applyCompactSupportTiles() {
    for (auto &[full, compact] : compactSupportTiles) {
      llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> slices;
      for (mlir::Operation *user : full.getUsers())
        if (auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user))
          slices.push_back(slice);
      for (mlir::tensor::ExtractSliceOp slice : slices) {
        auto offsets = getStaticValues(slice.getMixedOffsets());
        auto sizes = getStaticValues(slice.getMixedSizes());
        auto strides = getStaticValues(slice.getMixedStrides());
        if (!offsets || !sizes || !strides || *offsets != compact.offsets ||
            *sizes != compact.sizes ||
            !llvm::all_of(*strides, [](int64_t stride) { return stride == 1; }))
          continue;
        if (slice.getType() != compact.value.getType())
          return mlir::failure();
        slice.getResult().replaceAllUsesWith(compact.value);
        slice.erase();
      }
    }
    return mlir::success();
  }

  void eraseDeadCompactSupportWrappers() {
    llvm::SmallVector<mlir::Value, 4> completed;
    for (auto &[full, compact] : compactSupportTiles) {
      (void)compact;
      if (!full.use_empty())
        continue;
      llvm::SmallVector<mlir::Operation *, 8> worklist;
      if (mlir::Operation *definition = full.getDefiningOp())
        worklist.push_back(definition);
      while (!worklist.empty()) {
        mlir::Operation *operation = worklist.pop_back_val();
        if (!operation->getBlock() || !mlir::isOpTriviallyDead(operation))
          continue;
        llvm::SmallVector<mlir::Operation *, 4> producers;
        for (mlir::Value operand : operation->getOperands())
          if (mlir::Operation *producer = operand.getDefiningOp())
            producers.push_back(producer);
        operation->erase();
        llvm::append_range(worklist, producers);
      }
      completed.push_back(full);
    }
    for (mlir::Value full : completed)
      compactSupportTiles.erase(full);
  }

  mlir::FailureOr<std::pair<const analysis::RootRegionWork *,
                            const analysis::RootExecutionWork *>>
  getExecutionWork(const RegionExecutionId &execution) {
    analysis::RootRegionWorkId workId;
    const compiler::detail::LogicalShardId *shard = nullptr;
    if (const auto *required = std::get_if<ExecutionInstanceId>(&execution)) {
      if (std::holds_alternative<RequiredMergeExecution>(required->source))
        return fail<std::pair<const analysis::RootRegionWork *,
                              const analysis::RootExecutionWork *>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "structural reduction merge materialization is not available");
      const auto &root = std::get<RequiredRootExecution>(required->source);
      workId = root.work;
      shard = &root.shard;
    } else {
      const auto &replica = std::get<ReplicaExecutionId>(execution);
      workId = replica.producer.work;
      shard = &replica.producer.shard;
    }
    const analysis::RootRegionWork *work = findWork(rootWorks, workId);
    const analysis::RootExecutionWork *piece =
        work && shard ? findExecution(*work, *shard) : nullptr;
    const bool isReplica =
        std::holds_alternative<ReplicaExecutionId>(execution);
    if (!work || !piece || !work->rootOperation ||
        (!isReplica && work->id.tile != group.tile))
      return fail<std::pair<const analysis::RootRegionWork *,
                            const analysis::RootExecutionWork *>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "Region execution has no matching root work on its selected Tile");
    return std::make_pair(work, piece);
  }

  mlir::FailureOr<std::pair<const analysis::RootRegionWork *,
                            const analysis::ReductionMergeRequirement *>>
  getMergeWork(const RequiredMergeExecution &execution) {
    const analysis::RootRegionWork *work = findWork(rootWorks, execution.work);
    if (!work || work->id.tile != group.tile)
      return fail<std::pair<const analysis::RootRegionWork *,
                            const analysis::ReductionMergeRequirement *>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "merge execution has no matching selected merge work");
    auto merge = llvm::find_if(work->merges, [&](const auto &candidate) {
      return candidate.group == execution.group;
    });
    if (merge == work->merges.end())
      return fail<std::pair<const analysis::RootRegionWork *,
                            const analysis::ReductionMergeRequirement *>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "merge execution has no matching selected merge work");
    return std::make_pair(work, &*merge);
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
  materializeActualStandardMergeExecution(
      const RequiredMergeExecution &execution) {
    auto workAndMerge = getMergeWork(execution);
    if (mlir::failed(workAndMerge))
      return mlir::failure();
    const analysis::RootRegionWork &work = *workAndMerge->first;
    const analysis::ReductionMergeRequirement &merge = *workAndMerge->second;
    auto tiling =
        mlir::dyn_cast_or_null<mlir::TilingInterface>(work.rootOperation);
    auto partialInterface =
        mlir::dyn_cast_or_null<mlir::PartialReductionOpInterface>(
            work.rootOperation);
    auto dps = mlir::dyn_cast_or_null<mlir::DestinationStyleOpInterface>(
        work.rootOperation);
    if (!tiling || !partialInterface || !dps || merge.coupledRule ||
        merge.contributions.empty() || merge.results.empty() ||
        dps.getNumDpsInits() != work.rootOperation->getNumResults() ||
        merge.results.size() != work.rootOperation->getNumResults() ||
        llvm::any_of(llvm::enumerate(merge.results), [](const auto &entry) {
          return entry.value().result != entry.index();
        }))
      return fail<llvm::SmallVector<mlir::Value, 2>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "standard merge contract is incomplete");

    llvm::SmallVector<mlir::utils::IteratorType, 6> iteratorTypes =
        tiling.getLoopIteratorTypes();
    llvm::SmallVector<int, 2> reductionDimensions;
    for (auto [dimension, iteratorType] : llvm::enumerate(iteratorTypes))
      if (iteratorType == mlir::utils::IteratorType::reduction)
        reductionDimensions.push_back(static_cast<int>(dimension));
    if (reductionDimensions.empty())
      return fail<llvm::SmallVector<mlir::Value, 2>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "standard merge has no reduction dimension");

    llvm::SmallVector<analysis::StaticRectangularIndexSet, 8> boxes;
    for (const analysis::ReductionContribution &contribution :
         merge.contributions) {
      auto normalized =
          analysis::normalizeFiniteExactIndexSet(contribution.iterationDomain);
      if (mlir::failed(normalized) || normalized->getBoxes().size() != 1)
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "standard contribution is not one exact iteration rectangle");
      boxes.push_back(normalized->getBoxes().front());
    }
    const size_t rank = iteratorTypes.size();
    if (boxes.empty() || boxes.front().offsets.size() != rank ||
        boxes.front().sizes.size() != rank)
      return fail<llvm::SmallVector<mlir::Value, 2>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "standard contribution rank does not match its operation");
    llvm::SmallVector<int64_t, 6> globalOffsets(boxes.front().offsets.begin(),
                                                boxes.front().offsets.end());
    llvm::SmallVector<int64_t, 6> globalEnds(rank);
    for (size_t dimension = 0; dimension < rank; ++dimension)
      if (boxes.front().sizes[dimension] <= 0 ||
          llvm::AddOverflow(boxes.front().offsets[dimension],
                            boxes.front().sizes[dimension],
                            globalEnds[dimension]))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "standard contribution extent is invalid");
    int64_t contributionVolume = 0;
    for (auto [boxIndex, box] : llvm::enumerate(boxes)) {
      if (box.offsets.size() != rank || box.sizes.size() != rank)
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "standard contribution rectangles have different ranks");
      int64_t volume = 1;
      for (size_t dimension = 0; dimension < rank; ++dimension) {
        int64_t end = 0;
        int64_t nextVolume = 0;
        if (box.sizes[dimension] <= 0 ||
            llvm::AddOverflow(box.offsets[dimension], box.sizes[dimension],
                              end) ||
            llvm::MulOverflow(volume, box.sizes[dimension], nextVolume))
          return fail<llvm::SmallVector<mlir::Value, 2>>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "standard contribution size is not representable");
        volume = nextVolume;
        if (iteratorTypes[dimension] != mlir::utils::IteratorType::reduction &&
            (box.offsets[dimension] != boxes.front().offsets[dimension] ||
             box.sizes[dimension] != boxes.front().sizes[dimension]))
          return fail<llvm::SmallVector<mlir::Value, 2>>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "standard contributions disagree on parallel iteration work");
        globalOffsets[dimension] =
            std::min(globalOffsets[dimension], box.offsets[dimension]);
        globalEnds[dimension] = std::max(globalEnds[dimension], end);
      }
      int64_t nextTotal = 0;
      if (llvm::AddOverflow(contributionVolume, volume, nextTotal))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "standard contribution volume is not representable");
      contributionVolume = nextTotal;
      for (size_t previous = 0; previous < boxIndex; ++previous) {
        bool overlaps = true;
        for (size_t dimension = 0; dimension < rank; ++dimension) {
          int64_t previousEnd = boxes[previous].offsets[dimension] +
                                boxes[previous].sizes[dimension];
          int64_t currentEnd = box.offsets[dimension] + box.sizes[dimension];
          overlaps &= boxes[previous].offsets[dimension] < currentEnd &&
                      box.offsets[dimension] < previousEnd;
        }
        if (overlaps)
          return fail<llvm::SmallVector<mlir::Value, 2>>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "standard contribution rectangles overlap");
      }
    }
    llvm::SmallVector<int64_t, 6> globalSizes(rank);
    int64_t globalVolume = 1;
    for (size_t dimension = 0; dimension < rank; ++dimension) {
      globalSizes[dimension] = globalEnds[dimension] - globalOffsets[dimension];
      int64_t nextVolume = 0;
      if (globalSizes[dimension] <= 0 ||
          llvm::MulOverflow(globalVolume, globalSizes[dimension], nextVolume))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "standard merge rectangle is not representable");
      globalVolume = nextVolume;
    }
    if (globalVolume != contributionVolume)
      return fail<llvm::SmallVector<mlir::Value, 2>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "standard contributions do not exactly cover their merge rectangle");

    llvm::SmallVector<mlir::Value, 2> assembledPartials;
    for (const analysis::ReductionResultSlice &result : merge.results) {
      if (result.result >= work.rootOperation->getNumResults())
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "standard merge result index is out of range");
      auto resultType = mlir::cast<mlir::RankedTensorType>(
          work.rootOperation->getResult(result.result).getType());
      mlir::Value assembled = builder
                                  .create<mlir::tensor::EmptyOp>(
                                      work.rootOperation->getLoc(), globalSizes,
                                      resultType.getElementType())
                                  .getResult();
      for (auto [contribution, box] :
           llvm::zip_equal(merge.contributions, boxes)) {
        StandardPartialKey key{merge.group, contribution.shard, result.result};
        auto partial = getOrCreateStandardPartialBoundary(key);
        if (mlir::failed(partial))
          return mlir::failure();
        llvm::SmallVector<mlir::OpFoldResult, 6> offsets;
        llvm::SmallVector<mlir::OpFoldResult, 6> sizes;
        llvm::SmallVector<mlir::OpFoldResult, 6> strides;
        for (size_t dimension = 0; dimension < rank; ++dimension) {
          offsets.push_back(builder.getIndexAttr(box.offsets[dimension] -
                                                 globalOffsets[dimension]));
          sizes.push_back(builder.getIndexAttr(box.sizes[dimension]));
          strides.push_back(builder.getIndexAttr(1));
        }
        assembled = builder
                        .create<mlir::tensor::InsertSliceOp>(
                            work.rootOperation->getLoc(), *partial, assembled,
                            offsets, sizes, strides)
                        .getResult();
      }
      assembledPartials.push_back(assembled);
    }

    llvm::SmallVector<mlir::OpFoldResult, 6> iterationOffsets;
    llvm::SmallVector<mlir::OpFoldResult, 6> iterationSizes;
    for (size_t dimension = 0; dimension < rank; ++dimension) {
      iterationOffsets.push_back(
          builder.getIndexAttr(globalOffsets[dimension]));
      iterationSizes.push_back(builder.getIndexAttr(globalSizes[dimension]));
    }
    llvm::SmallVector<mlir::Value, 2> resultDestinations;
    for (const analysis::ReductionResultSlice &result : merge.results) {
      mlir::Value sourceInit = dps.getDpsInitOperand(result.result)->get();
      auto mappedInit = mapSupportValue(sourceInit);
      llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
      llvm::SmallVector<mlir::OpFoldResult> resultSizes;
      if (mlir::failed(mappedInit) ||
          mlir::failed(tiling.getResultTilePosition(
              builder, result.result, iterationOffsets, iterationSizes,
              resultOffsets, resultSizes)))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "standard merge cannot materialize its single destination init");
      auto initType =
          mlir::cast<mlir::RankedTensorType>((*mappedInit).getType());
      llvm::SmallVector<mlir::OpFoldResult, 4> strides(initType.getRank(),
                                                       builder.getIndexAttr(1));
      resultDestinations.push_back(builder
                                       .create<mlir::tensor::ExtractSliceOp>(
                                           work.rootOperation->getLoc(),
                                           *mappedInit, resultOffsets,
                                           resultSizes, strides)
                                       .getResult());
    }

    mlir::IRMapping adapterMapping;
    llvm::SmallVector<mlir::Operation *, 4> placeholders;
    for (mlir::OpOperand &operand : work.rootOperation->getOpOperands()) {
      std::optional<unsigned> initIndex;
      for (auto [index, value] : llvm::enumerate(dps.getDpsInits()))
        if (value == operand.get()) {
          initIndex = index;
          break;
        }
      if (initIndex) {
        adapterMapping.map(operand.get(), resultDestinations[*initIndex]);
        continue;
      }
      auto type =
          mlir::dyn_cast<mlir::RankedTensorType>(operand.get().getType());
      if (type && type.hasStaticShape()) {
        auto empty = builder.create<mlir::tensor::EmptyOp>(
            work.rootOperation->getLoc(), type.getShape(),
            type.getElementType());
        placeholders.push_back(empty);
        adapterMapping.map(operand.get(), empty.getResult());
        continue;
      }
      auto mapped = mapSupportValue(operand.get());
      if (mlir::failed(mapped))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "standard merge cannot localize a non-tensor operand");
      adapterMapping.map(operand.get(), *mapped);
    }
    mlir::Operation *adapter =
        builder.clone(*work.rootOperation, adapterMapping);
    for (auto [result, destination] :
         llvm::zip_equal(adapter->getResults(), resultDestinations))
      result.setType(destination.getType());
    auto adapterPartial =
        mlir::cast<mlir::PartialReductionOpInterface>(adapter);
    auto merged =
        adapterPartial.mergeReductions(builder, work.rootOperation->getLoc(),
                                       assembledPartials, reductionDimensions);
    if (mlir::failed(merged) ||
        merged->replacements.size() != merge.results.size()) {
      if (adapter->use_empty())
        adapter->erase();
      return fail<llvm::SmallVector<mlir::Value, 2>>(
          failure, SpatialRegionMaterializationFailureKind::Unsupported,
          "standard partials cannot form one actual merge");
    }
    if (adapter->use_empty())
      adapter->erase();
    for (mlir::Operation *placeholder : llvm::reverse(placeholders))
      if (placeholder->use_empty())
        placeholder->erase();

    llvm::SmallVector<mlir::Value, 2> fullResults;
    for (auto [result, mergedValue] :
         llvm::zip_equal(merge.results, merged->replacements)) {
      auto fullType = mlir::cast<mlir::RankedTensorType>(
          work.rootOperation->getResult(result.result).getType());
      auto normalizedResult =
          analysis::normalizeFiniteExactIndexSet(result.domain);
      if (mlir::failed(normalizedResult) ||
          normalizedResult->getBoxes().size() != 1)
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "standard merge result is not one exact tensor tile");
      auto box = normalizedResult->getBoxes().front();
      llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
      llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
      llvm::SmallVector<mlir::OpFoldResult, 4> strides;
      for (auto [offset, size] : llvm::zip_equal(box.offsets, box.sizes)) {
        offsets.push_back(builder.getIndexAttr(offset));
        sizes.push_back(builder.getIndexAttr(size));
        strides.push_back(builder.getIndexAttr(1));
      }
      fullResults.push_back(
          builder
              .create<mlir::tensor::InsertSliceOp>(
                  work.rootOperation->getLoc(), mergedValue,
                  builder
                      .create<mlir::tensor::EmptyOp>(
                          work.rootOperation->getLoc(), fullType.getShape(),
                          fullType.getElementType())
                      .getResult(),
                  offsets, sizes, strides)
              .getResult());
    }
    executionResults.emplace(ExecutionInstanceId{execution}, fullResults);
    return fullResults;
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
  materializeActualCoupledMergeExecution(
      const RequiredMergeExecution &execution) {
    auto workAndMerge = getMergeWork(execution);
    if (mlir::failed(workAndMerge))
      return mlir::failure();
    const analysis::RootRegionWork &work = *workAndMerge->first;
    const analysis::ReductionMergeRequirement &merge = *workAndMerge->second;
    auto attention =
        mlir::dyn_cast_or_null<LinalgExtAttentionOp>(work.rootOperation);
    if (!attention ||
        attention.getAlgorithm() != AttentionAlgorithm::FlashDecoding ||
        !merge.coupledRule || merge.contributions.size() < 2 ||
        merge.components.size() != 3 || merge.results.size() != 1)
      return fail<llvm::SmallVector<mlir::Value, 2>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "flash_decoding merge contract is incomplete");

    llvm::SmallVector<OnlineAttentionState, 8> contributionStates;
    for (const analysis::ReductionContribution &contribution :
         merge.contributions) {
      OnlineAttentionState state;
      for (const analysis::CoupledReductionComponentRequirement &component :
           merge.components) {
        CoupledStateKey key{merge.group, contribution.shard, component.kind};
        mlir::FailureOr<mlir::Value> value =
            getOrCreateCoupledStateBoundary(key);
        if (mlir::failed(value))
          return mlir::failure();
        switch (component.kind) {
        case CoupledReductionComponentKind::Maximum:
          state.maximum = *value;
          break;
        case CoupledReductionComponentKind::Sum:
          state.sum = *value;
          break;
        case CoupledReductionComponentKind::Accumulator:
          state.accumulator = *value;
          break;
        }
      }
      if (!state.accumulator || !state.maximum || !state.sum)
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "flash_decoding contribution is missing one coupled state");
      contributionStates.push_back(state);
    }

    OnlineAttentionState merged = contributionStates.front();
    for (const OnlineAttentionState &state :
         llvm::ArrayRef(contributionStates).drop_front()) {
      auto next = materializeOnlineAttentionStateMerge(attention, merged, state,
                                                       builder);
      if (mlir::failed(next))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "flash_decoding states cannot form an actual coupled merge");
      merged = *next;
    }
    mlir::FailureOr<mlir::Value> finalized =
        materializeOnlineAttentionFinalize(attention, merged, builder);
    auto normalized =
        analysis::normalizeFiniteExactIndexSet(merge.results.front().domain);
    if (mlir::failed(finalized) || mlir::failed(normalized) ||
        normalized->getBoxes().size() != 1)
      return fail<llvm::SmallVector<mlir::Value, 2>>(
          failure, SpatialRegionMaterializationFailureKind::Unsupported,
          "flash_decoding final result is not one exact tensor tile");
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        work.rootOperation->getResult(merge.results.front().result).getType());
    auto finalizedType =
        mlir::dyn_cast<mlir::RankedTensorType>(finalized->getType());
    auto resultBox = normalized->getBoxes().front();
    if (!resultType || !resultType.hasStaticShape() || !finalizedType ||
        finalizedType.getShape() != llvm::ArrayRef<int64_t>(resultBox.sizes))
      return fail<llvm::SmallVector<mlir::Value, 2>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "flash_decoding final state does not match its output piece");
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    for (auto [offset, size] :
         llvm::zip_equal(resultBox.offsets, resultBox.sizes)) {
      offsets.push_back(builder.getIndexAttr(offset));
      sizes.push_back(builder.getIndexAttr(size));
      strides.push_back(builder.getIndexAttr(1));
    }
    mlir::Value fullResult =
        builder
            .create<mlir::tensor::InsertSliceOp>(
                attention.getLoc(), *finalized,
                builder
                    .create<mlir::tensor::EmptyOp>(attention.getLoc(),
                                                   resultType.getShape(),
                                                   resultType.getElementType())
                    .getResult(),
                offsets, sizes, strides)
            .getResult();
    llvm::SmallVector<mlir::Value, 2> fullResults{fullResult};
    executionResults.emplace(ExecutionInstanceId{execution}, fullResults);
    return fullResults;
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
  materializeExecution(const RegionExecutionId &execution) {
    auto workAndPiece = getExecutionWork(execution);
    if (mlir::failed(workAndPiece))
      return mlir::failure();
    const analysis::RootRegionWork &work = *workAndPiece->first;
    const analysis::RootExecutionWork &piece = *workAndPiece->second;
    mlir::Operation *sourceRoot = work.rootOperation;
    if (!findNode(nodes, sourceRoot))
      return fail<llvm::SmallVector<mlir::Value, 2>>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "root work has no structured node mapping");

    const analysis::RootContributionWork *coupledContribution = nullptr;
    for (const analysis::RootContributionWork &contribution :
         work.contributions)
      if (contribution.contribution.shard == piece.shard &&
          contribution.coupledRule) {
        coupledContribution = &contribution;
        break;
      }
    const analysis::RootContributionWork *standardContribution = nullptr;
    for (const analysis::RootContributionWork &contribution :
         work.contributions)
      if (contribution.contribution.shard == piece.shard &&
          !contribution.coupledRule) {
        standardContribution = &contribution;
        break;
      }
    auto attention = mlir::dyn_cast<LinalgExtAttentionOp>(sourceRoot);
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(sourceRoot);

    mlir::IRMapping rootMapping;
    llvm::SmallVector<mlir::Operation *, 2> temporaryInitPlaceholders;
    for (mlir::OpOperand &operand : sourceRoot->getOpOperands()) {
      bool isInit = false;
      if (dps)
        isInit = llvm::is_contained(dps.getDpsInits(), operand.get());
      if (attention && isInit)
        continue;
      if (standardContribution && isInit) {
        auto type =
            mlir::dyn_cast<mlir::RankedTensorType>(operand.get().getType());
        if (!type || !type.hasStaticShape())
          return fail<llvm::SmallVector<mlir::Value, 2>>(
              failure, SpatialRegionMaterializationFailureKind::Unsupported,
              "standard contribution init requires a static tensor type");
        auto placeholder = builder.create<mlir::tensor::EmptyOp>(
            sourceRoot->getLoc(), type.getShape(), type.getElementType());
        rootMapping.map(operand.get(), placeholder.getResult());
        temporaryInitPlaceholders.push_back(placeholder);
        continue;
      }
      mlir::FailureOr<mlir::Value> mapped =
          mapExecutionOperand(execution, operand);
      if (mlir::failed(mapped))
        return mlir::failure();
      rootMapping.map(operand.get(), *mapped);
    }
    llvm::SetVector<mlir::Value> captures;
    mlir::getUsedValuesDefinedAbove(sourceRoot->getRegions(), captures);
    for (mlir::Value capture : captures) {
      mlir::FailureOr<mlir::Value> mapped = mapSupportValue(capture);
      if (mlir::failed(mapped))
        return mlir::failure();
      rootMapping.map(capture, *mapped);
    }
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    for (const auto &interval : piece.iterationDomain) {
      offsets.push_back(builder.getIndexAttr(interval.offset));
      sizes.push_back(builder.getIndexAttr(interval.size));
    }
    if (attention &&
        attention.getAlgorithm() == AttentionAlgorithm::FlashAttention) {
      if (coupledContribution || !work.merges.empty())
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "flash_attention cannot have spatial K2 contributions or merge");
      mlir::Value mappedMask =
          attention.getMask() ? rootMapping.lookupOrNull(attention.getMask())
                              : mlir::Value{};
      auto state = materializeOnlineAttentionTile(
          attention, rootMapping.lookupOrNull(attention.getQuery()),
          rootMapping.lookupOrNull(attention.getKey()),
          rootMapping.lookupOrNull(attention.getValue()),
          rootMapping.lookupOrNull(attention.getScale()), mappedMask, offsets,
          sizes, builder);
      if (mlir::failed(state))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "flash_attention spatial tile cannot form online state IR");
      if (mlir::failed(applyCompactSupportTiles()))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::CompilerFailure,
            "flash_attention compact support tile replacement failed");
      eraseDeadCompactSupportWrappers();
      mlir::FailureOr<mlir::Value> finalized =
          materializeOnlineAttentionFinalize(attention, *state, builder);
      llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
      llvm::SmallVector<mlir::OpFoldResult> resultSizes;
      if (mlir::failed(finalized) ||
          mlir::failed(attention.getResultTilePosition(
              builder, 0, offsets, sizes, resultOffsets, resultSizes)))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "flash_attention spatial result tile cannot be finalized");
      auto fullType = mlir::dyn_cast<mlir::RankedTensorType>(
          sourceRoot->getResult(0).getType());
      if (!fullType || !fullType.hasStaticShape())
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "flash_attention result requires a static ranked tensor");
      mlir::Value empty = builder
                              .create<mlir::tensor::EmptyOp>(
                                  sourceRoot->getLoc(), fullType.getShape(),
                                  fullType.getElementType())
                              .getResult();
      llvm::SmallVector<mlir::OpFoldResult, 6> strides(fullType.getRank(),
                                                       builder.getIndexAttr(1));
      mlir::Value fullResult = builder
                                   .create<mlir::tensor::InsertSliceOp>(
                                       sourceRoot->getLoc(), *finalized, empty,
                                       resultOffsets, resultSizes, strides)
                                   .getResult();
      llvm::SmallVector<mlir::Value, 2> results{fullResult};
      executionResults.emplace(execution, results);
      return results;
    }

    if (attention && coupledContribution) {
      if (attention.getAlgorithm() != AttentionAlgorithm::FlashDecoding)
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "only flash_decoding may have spatial coupled contributions");
      mlir::Value mappedMask =
          attention.getMask() ? rootMapping.lookupOrNull(attention.getMask())
                              : mlir::Value{};
      auto state = materializeOnlineAttentionTile(
          attention, rootMapping.lookupOrNull(attention.getQuery()),
          rootMapping.lookupOrNull(attention.getKey()),
          rootMapping.lookupOrNull(attention.getValue()),
          rootMapping.lookupOrNull(attention.getScale()), mappedMask, offsets,
          sizes, builder);
      if (mlir::failed(state))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "flash_decoding contribution cannot form online state IR");
      if (mlir::failed(applyCompactSupportTiles()))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::CompilerFailure,
            "flash_decoding compact support tile replacement failed");
      eraseDeadCompactSupportWrappers();
      auto stateValue = [&](CoupledReductionComponentKind kind) -> mlir::Value {
        switch (kind) {
        case CoupledReductionComponentKind::Maximum:
          return state->maximum;
        case CoupledReductionComponentKind::Sum:
          return state->sum;
        case CoupledReductionComponentKind::Accumulator:
          return state->accumulator;
        }
        return {};
      };
      llvm::SmallVector<std::pair<CoupledStateKey, mlir::Value>, 3> states;
      for (const analysis::CoupledReductionComponentSlice &component :
           coupledContribution->contribution.components) {
        CoupledStateKey key{coupledContribution->group, piece.shard,
                            component.kind};
        mlir::Value value = stateValue(component.kind);
        auto required = coupledStateBoundaryRequirements.find(key);
        if (!value || required == coupledStateBoundaryRequirements.end() ||
            required->second.type != value.getType())
          return fail<llvm::SmallVector<mlir::Value, 2>>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "flash_decoding state does not match its actual endpoint");
        states.push_back({key, value});
      }
      if (states.size() != 3)
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "flash_decoding contribution does not contain three states");
      executionCoupledStates.emplace(execution, std::move(states));
      llvm::SmallVector<mlir::Value, 2> noValues;
      executionResults.emplace(execution, noValues);
      return noValues;
    }

    mlir::Operation *adapter = builder.clone(*sourceRoot, rootMapping);
    if (standardContribution) {
      std::string partialFailure;
      auto partial = materializePartialReductionTile(adapter, builder, offsets,
                                                     sizes, &partialFailure);
      if (mlir::failed(partial)) {
        if (failure) {
          failure->kind = SpatialRegionMaterializationFailureKind::Unsupported;
          failure->detail = std::move(partialFailure);
        }
        if (adapter->use_empty())
          adapter->erase();
        return mlir::failure();
      }
      if (mlir::failed(applyCompactSupportTiles()))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::CompilerFailure,
            "partial reduction compact support tile replacement failed");
      if (llvm::any_of(partial->partialOperations,
                       [](mlir::Operation *operation) {
                         return !operation ||
                                mlir::failed(mlir::verify(operation));
                       }) ||
          llvm::any_of(
              partial->mergeOperations, [](mlir::Operation *operation) {
                return !operation || mlir::failed(mlir::verify(operation));
              }))
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "PartialReductionOpInterface produced an invalid noncanonical "
            "reduction tile");
      llvm::SmallVector<std::pair<StandardPartialKey, mlir::Value>, 2>
          partialValues;
      for (const analysis::ReductionResultSlice &result :
           standardContribution->contribution.results) {
        if (result.result >= partial->partialValues.size())
          return fail<llvm::SmallVector<mlir::Value, 2>>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "standard contribution result has no partial value");
        StandardPartialKey key{standardContribution->group, piece.shard,
                               result.result};
        auto required = standardPartialBoundaryRequirements.find(key);
        mlir::Value value = partial->partialValues[result.result];
        if (required == standardPartialBoundaryRequirements.end() ||
            required->second.type != value.getType())
          return fail<llvm::SmallVector<mlir::Value, 2>>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "standard partial does not match its actual endpoint type");
        partialValues.push_back({key, value});
      }
      for (mlir::Operation *operation :
           llvm::reverse(partial->mergeOperations)) {
        if (!operation->use_empty())
          return fail<llvm::SmallVector<mlir::Value, 2>>(
              failure, SpatialRegionMaterializationFailureKind::CompilerFailure,
              "temporary standard merge unexpectedly has live users");
        operation->erase();
      }
      if (adapter->use_empty())
        adapter->erase();
      eraseDeadCompactSupportWrappers();
      for (mlir::Operation *placeholder :
           llvm::reverse(temporaryInitPlaceholders)) {
        for (mlir::Operation *user :
             llvm::make_early_inc_range(placeholder->getUsers()))
          if (user->use_empty() && mlir::isMemoryEffectFree(user))
            user->erase();
        if (placeholder->use_empty())
          placeholder->erase();
      }
      executionStandardPartials.emplace(execution, std::move(partialValues));
      llvm::SmallVector<mlir::Value, 2> noValues;
      executionResults.emplace(execution, noValues);
      return noValues;
    }
    std::string tilingFailure;
    mlir::FailureOr<IterationTileMaterialization> tiled =
        materializeOperationFromIterationTile(adapter, builder, offsets, sizes,
                                              &tilingFailure);
    if (mlir::failed(tiled)) {
      if (failure) {
        failure->kind = SpatialRegionMaterializationFailureKind::Unsupported;
        failure->detail = std::move(tilingFailure);
      }
      if (adapter->use_empty())
        adapter->erase();
      return mlir::failure();
    }
    if (mlir::failed(applyCompactSupportTiles()))
      return fail<llvm::SmallVector<mlir::Value, 2>>(
          failure, SpatialRegionMaterializationFailureKind::CompilerFailure,
          "spatial compact support tile replacement failed");

    llvm::SmallVector<mlir::Value, 2> fullResults;
    fullResults.reserve(tiled->tiledValues.size());
    for (auto [resultNumber, tiledValue] :
         llvm::enumerate(tiled->tiledValues)) {
      auto fullType = mlir::dyn_cast<mlir::RankedTensorType>(
          sourceRoot->getResult(resultNumber).getType());
      if (!fullType || !fullType.hasStaticShape())
        return fail<llvm::SmallVector<mlir::Value, 2>>(
            failure, SpatialRegionMaterializationFailureKind::Unsupported,
            "spatial result requires a static ranked tensor type");
      auto empty = builder.create<mlir::tensor::EmptyOp>(
          sourceRoot->getLoc(), fullType.getShape(), fullType.getElementType());
      llvm::SmallVector<mlir::OpFoldResult, 4> strides(fullType.getRank(),
                                                       builder.getIndexAttr(1));
      auto inserted = builder.create<mlir::tensor::InsertSliceOp>(
          sourceRoot->getLoc(), tiledValue, empty.getResult(),
          tiled->resultOffsets[resultNumber], tiled->resultSizes[resultNumber],
          strides);
      fullResults.push_back(inserted.getResult());
    }
    if (adapter->use_empty())
      adapter->erase();
    eraseDeadCompactSupportWrappers();
    executionResults.emplace(execution, fullResults);
    return fullResults;
  }

  mlir::FailureOr<GroupArtifact> build() {
    auto executions = collectGroupExecutions(group, rootWorks, failure);
    if (mlir::failed(executions))
      return mlir::failure();
    // Every selected nonempty external fragment must have an actual
    // destination endpoint before the Region is committed.
    for (const auto &binding : group.externalBindings) {
      const DemandFragmentId &fragment = binding.fragment;
      if (fragment.source.kind != analysis::RootBoundaryKind::StructuredResult)
        continue;
      const analysis::ExactIndexSet *domain = findFragmentDomain(fragment);
      if (domain && domain->isEmpty())
        continue;
      auto sourceWork = llvm::find_if(rootWorks, [&](const auto &work) {
        return work.id.root == fragment.source.semantic && work.rootOperation &&
               fragment.source.index < work.rootOperation->getNumResults();
      });
      if (sourceWork == rootWorks.end())
        return fail<GroupArtifact>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "external Region fragment has no structured source operation");
      if (mlir::failed(getOrCreateFragmentValue(
              sourceWork->rootOperation->getResult(fragment.source.index),
              fragment)))
        return mlir::failure();
    }
    llvm::SmallVector<mlir::Value, 4> returned;
    std::set<ProducedValueKey> resultKeys;
    std::set<CoupledStateKey> stateKeys;
    std::set<StandardPartialKey> partialKeys;
    for (const RegionExecutionId &execution : *executions) {
      const auto *required = std::get_if<ExecutionInstanceId>(&execution);
      const auto *merge =
          required ? std::get_if<RequiredMergeExecution>(&required->source)
                   : nullptr;
      const analysis::RootRegionWork *work = nullptr;
      const analysis::RootExecutionWork *piece = nullptr;
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> values =
          mlir::failure();
      if (merge) {
        auto workAndMerge = getMergeWork(*merge);
        if (mlir::failed(workAndMerge))
          return mlir::failure();
        work = workAndMerge->first;
        values = workAndMerge->second->coupledRule
                     ? materializeActualCoupledMergeExecution(*merge)
                     : materializeActualStandardMergeExecution(*merge);
      } else {
        auto workAndPiece = getExecutionWork(execution);
        if (mlir::failed(workAndPiece))
          return mlir::failure();
        work = workAndPiece->first;
        piece = workAndPiece->second;
        values = materializeExecution(execution);
      }
      if (mlir::failed(values))
        return mlir::failure();
      auto stateValues = executionCoupledStates.find(execution);
      if (stateValues != executionCoupledStates.end()) {
        for (const auto &[key, value] : stateValues->second) {
          if (!value || !stateKeys.insert(key).second)
            return fail<GroupArtifact>(
                failure,
                SpatialRegionMaterializationFailureKind::BrokenContract,
                "Region produces a duplicate or missing coupled state");
          coupledStateResults.push_back(
              {key, static_cast<unsigned>(returned.size())});
          returned.push_back(value);
        }
      }
      auto partialValues = executionStandardPartials.find(execution);
      if (partialValues != executionStandardPartials.end()) {
        for (const auto &[key, value] : partialValues->second) {
          if (!value || !partialKeys.insert(key).second)
            return fail<GroupArtifact>(
                failure,
                SpatialRegionMaterializationFailureKind::BrokenContract,
                "Region produces a duplicate or missing standard partial");
          standardPartialResults.push_back(
              {key, static_cast<unsigned>(returned.size())});
          returned.push_back(value);
        }
      }
      if (std::holds_alternative<ReplicaExecutionId>(execution))
        continue;
      for (const analysis::RootResultWork &result : work->results) {
        const bool owned = merge ? result.reductionGroup &&
                                       *result.reductionGroup == merge->group
                                 : result.ownerShard && piece &&
                                       *result.ownerShard == piece->shard;
        if (!owned || result.result >= values->size())
          continue;
        ProducedValueKey key{analysis::RootBoundaryId{
                                 analysis::RootBoundaryKind::StructuredResult,
                                 work->id.root, result.result},
                             result.ownerShard, result.reductionGroup,
                             group.tile};
        if (!resultKeys.insert(key).second)
          continue;
        GroupResult groupResult{key, static_cast<unsigned>(returned.size())};
        results.push_back(groupResult);
        returned.push_back((*values)[result.result]);
      }
    }
    builder.create<TileYieldOp>(source.getLoc(), returned);
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    llvm::transform(returned, std::back_inserter(resultTypes),
                    [](mlir::Value value) { return value.getType(); });
    entryBuilder.setInsertionPointToEnd(&entryFunction.getBody().front());
    auto region = entryBuilder.create<TileRegionOp>(source.getLoc(),
                                                    resultTypes, regionInputs);
    region.getBody().takeBody(regionBody);

    GroupArtifact artifact;
    artifact.tile = group.tile;
    artifact.region = region;
    artifact.results = std::move(results);
    artifact.boundaryInputs = std::move(boundaryInputs);
    artifact.coupledStateResults = std::move(coupledStateResults);
    artifact.coupledStateInputs = std::move(coupledStateInputs);
    artifact.standardPartialResults = std::move(standardPartialResults);
    artifact.standardPartialInputs = std::move(standardPartialInputs);
    return artifact;
  }
};

} // namespace

mlir::FailureOr<SpatialRegionMaterializationResult> materializeSpatialRegions(
    mlir::ModuleOp source, CardId cardId, llvm::ArrayRef<TileId> availableTiles,
    llvm::ArrayRef<compiler::detail::StructuredOperationNodeMapping>
        operationNodes,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const compiler::detail::RegionPlan &regionPlan,
    SpatialRegionMaterializationFailure *failure) {
  if (failure)
    *failure = {};
  if (!source || availableTiles.empty() || operationNodes.empty() ||
      rootWorks.empty() || regionPlan.groups.empty())
    return fail<SpatialRegionMaterializationResult>(
        failure, SpatialRegionMaterializationFailureKind::BrokenContract,
        "structural materialization input is incomplete");
  if (mlir::failed(mlir::verify(source)))
    return fail<SpatialRegionMaterializationResult>(
        failure, SpatialRegionMaterializationFailureKind::BrokenContract,
        "structural materialization requires verifier-valid source IR");
  auto sourceFunction = getSourceFunction(source, rootWorks, failure);
  if (mlir::failed(sourceFunction))
    return mlir::failure();

  llvm::DenseSet<mlir::Operation *> mappedOperations;
  std::set<uint32_t> mappedNodeIds;
  for (const auto &mapping : operationNodes)
    if (!mapping.operation ||
        mapping.operation->getParentOfType<mlir::func::FuncOp>() !=
            *sourceFunction ||
        !mappedOperations.insert(mapping.operation).second ||
        !mappedNodeIds.insert(mapping.structuredNodeId).second)
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "structured node mapping is stale, duplicated, or outside the "
          "TensorProgram");
  auto sourceReturn = mlir::dyn_cast<mlir::func::ReturnOp>(
      sourceFunction->getBody().front().getTerminator());
  if (!sourceReturn)
    return fail<SpatialRegionMaterializationResult>(
        failure, SpatialRegionMaterializationFailureKind::BrokenContract,
        "TensorProgram has no func.return terminator");
  for (mlir::Value output : sourceReturn.getOperands()) {
    if (!mlir::isa<mlir::ShapedType>(output.getType()))
      continue;
    auto result = mlir::dyn_cast<mlir::OpResult>(output);
    if (!result || !findNode(operationNodes, result.getOwner())) {
      std::string detail;
      llvm::raw_string_ostream stream(detail);
      stream << "structural materialization requires each shaped program "
                "output to be a direct structured result; owner=";
      if (mlir::Operation *owner = output.getDefiningOp())
        stream << owner->getName() << " output_type=" << output.getType()
               << " operands=[";
      if (mlir::Operation *owner = output.getDefiningOp())
        llvm::interleaveComma(
            owner->getOperands(), stream, [&](mlir::Value operand) {
              if (mlir::Operation *definition = operand.getDefiningOp())
                stream << definition->getName();
              else
                stream << "block_argument";
              stream << ':' << operand.getType();
            });
      if (output.getDefiningOp())
        stream << ']';
      else
        stream << "block_argument";
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::Unsupported,
          detail);
    }
  }

  std::set<int64_t> tileIds;
  llvm::SmallVector<TileId, 16> sortedTiles(availableTiles.begin(),
                                            availableTiles.end());
  llvm::sort(sortedTiles, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  for (TileId tile : sortedTiles)
    if (tile.getValue() < 0 || !tileIds.insert(tile.getValue()).second)
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "available Tile domain is malformed");

  std::set<analysis::RootRegionWorkId> workIds;
  for (const analysis::RootRegionWork &work : rootWorks)
    if (!work.rootOperation || !workIds.insert(work.id).second ||
        !tileIds.count(work.id.tile.getValue()))
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "root work is missing, duplicated, or assigned to an unavailable "
          "Tile");
  llvm::DenseSet<mlir::Operation *> checkedAttentionRoots;
  for (const analysis::RootRegionWork &work : rootWorks) {
    auto attention =
        mlir::dyn_cast_or_null<LinalgExtAttentionOp>(work.rootOperation);
    if (!attention || !checkedAttentionRoots.insert(attention).second)
      continue;
    bool hasCoupledContribution = false;
    bool hasCoupledMerge = false;
    for (const analysis::RootRegionWork &candidate : rootWorks) {
      if (candidate.id.root != work.id.root)
        continue;
      hasCoupledContribution |=
          llvm::any_of(candidate.contributions, [](const auto &contribution) {
            return contribution.coupledRule;
          });
      for (const analysis::ReductionMergeRequirement &merge :
           candidate.merges) {
        if (!merge.coupledRule)
          continue;
        hasCoupledMerge = true;
        if (merge.contributions.size() < 2)
          return fail<SpatialRegionMaterializationResult>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "flash_decoding requires at least two spatial K2 contributions");
      }
    }
    if (attention.getAlgorithm() == AttentionAlgorithm::FlashAttention &&
        (hasCoupledContribution || hasCoupledMerge))
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "flash_attention cannot consume a multi-contribution Spatial choice");
    if (attention.getAlgorithm() == AttentionAlgorithm::FlashDecoding &&
        (!hasCoupledContribution || !hasCoupledMerge))
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "flash_decoding requires contributions and one selected merge owner");
  }
  std::set<analysis::RootRegionWorkId> groupedWorks;
  for (const RegionGroupPlan &group : regionPlan.groups) {
    if (!tileIds.count(group.tile.getValue()) || group.mandatoryRoots.empty())
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "Region group has no valid Tile or mandatory root");
    for (const auto &work : group.mandatoryRoots)
      if (!workIds.count(work) || work.tile != group.tile ||
          !groupedWorks.insert(work).second)
        return fail<SpatialRegionMaterializationResult>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "Region groups do not partition root work exactly");
  }
  if (groupedWorks != workIds)
    return fail<SpatialRegionMaterializationResult>(
        failure, SpatialRegionMaterializationFailureKind::BrokenContract,
        "RegionPlan omits selected root work");
  if (mlir::failed(validateRegionChoice(rootWorks, regionPlan, failure)))
    return mlir::failure();

  SpatialRegionMaterializationResult result;
  result.module = mlir::ModuleOp::create(source.getLoc());
  result.module->getOperation()->setAttrs(source->getAttrDictionary());
  mlir::OpBuilder moduleBuilder(result.module->getBodyRegion());
  mlir::IRMapping declarationMapping;
  for (mlir::Operation &operation : source.getBody()->without_terminator())
    if (isSharedTopLevelFact(operation, *sourceFunction))
      moduleBuilder.clone(operation, declarationMapping);

  std::map<int64_t, TileModuleOp> tileModules;
  for (TileId tileId : sortedTiles) {
    auto tile = moduleBuilder.create<TileModuleOp>(
        source.getLoc(), moduleBuilder.getI64IntegerAttr(cardId.getValue()),
        moduleBuilder.getI64IntegerAttr(tileId.getValue()));
    tile.getBody().push_back(new mlir::Block());
    tileModules.emplace(tileId.getValue(), tile);
  }

  struct PendingBoundary {
    DemandFragmentId fragment;
    TileId sourceTile{0};
    TileId destinationTile{0};
    mlir::Value destination;
  };
  struct PendingCoupledStateBoundary {
    CoupledStateKey key;
    mlir::Value destination;
  };
  struct PendingStandardPartialBoundary {
    StandardPartialKey key;
    mlir::Value destination;
  };
  std::map<ProducedValueKey, mlir::Value> produced;
  llvm::SmallVector<PendingBoundary, 16> pendingBoundaries;
  llvm::SmallVector<PendingCoupledStateBoundary, 16>
      pendingCoupledStateBoundaries;
  llvm::SmallVector<PendingStandardPartialBoundary, 16>
      pendingStandardPartialBoundaries;
  std::set<ProducedValueKey> plannedResults;
  std::map<ProducedValueKey, size_t> plannedResultGroups;
  for (auto [groupIndex, group] : llvm::enumerate(regionPlan.groups)) {
    auto executions = collectGroupExecutions(group, rootWorks, failure);
    if (mlir::failed(executions))
      return mlir::failure();
    for (const RegionExecutionId &execution : *executions) {
      const auto *required = std::get_if<ExecutionInstanceId>(&execution);
      const auto *merge =
          required ? std::get_if<RequiredMergeExecution>(&required->source)
                   : nullptr;
      const auto *root =
          required ? std::get_if<RequiredRootExecution>(&required->source)
                   : nullptr;
      if (!root && !merge)
        continue;
      const analysis::RootRegionWork *work =
          findWork(rootWorks, root ? root->work : merge->work);
      const analysis::RootExecutionWork *piece =
          work && root ? findExecution(*work, root->shard) : nullptr;
      const analysis::ReductionMergeRequirement *mergeWork = nullptr;
      if (work && merge) {
        auto found = llvm::find_if(work->merges, [&](const auto &candidate) {
          return candidate.group == merge->group;
        });
        if (found != work->merges.end())
          mergeWork = &*found;
      }
      if (!work || (root && !piece))
        return fail<SpatialRegionMaterializationResult>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "Region execution has no selected root work");
      if (merge && !mergeWork)
        return fail<SpatialRegionMaterializationResult>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "Region execution has no selected merge work");
      for (const analysis::RootResultWork &resultWork : work->results) {
        const bool owned = root
                               ? resultWork.ownerShard &&
                                     *resultWork.ownerShard == piece->shard
                               : resultWork.reductionGroup &&
                                     *resultWork.reductionGroup == merge->group;
        if (!owned)
          continue;
        ProducedValueKey key{analysis::RootBoundaryId{
                                 analysis::RootBoundaryKind::StructuredResult,
                                 work->id.root, resultWork.result},
                             resultWork.ownerShard, resultWork.reductionGroup,
                             group.tile};
        if (!plannedResults.insert(key).second ||
            !plannedResultGroups.emplace(key, groupIndex).second)
          return fail<SpatialRegionMaterializationResult>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "RegionPlan assigns one structured result endpoint twice");
      }
    }
  }

  std::map<CoupledStateKey, CoupledStateBoundaryRequirement>
      coupledStateBoundaryRequirements;
  for (const analysis::RootRegionWork &work : rootWorks)
    for (const analysis::ReductionMergeRequirement &merge : work.merges) {
      if (!merge.coupledRule || merge.contributions.size() < 2 ||
          merge.components.size() != 3)
        continue;
      for (const analysis::ReductionContribution &contribution :
           merge.contributions) {
        for (const analysis::CoupledReductionComponentRequirement &component :
             merge.components) {
          auto type = getCoupledStateType(component);
          CoupledStateKey key{merge.group, contribution.shard, component.kind};
          if (mlir::failed(type) ||
              !coupledStateBoundaryRequirements
                   .emplace(key,
                            CoupledStateBoundaryRequirement{contribution.tile,
                                                            *type})
                   .second)
            return fail<SpatialRegionMaterializationResult>(
                failure,
                SpatialRegionMaterializationFailureKind::BrokenContract,
                "coupled state endpoint is invalid or duplicated");
        }
      }
    }
  std::map<CoupledStateKey, mlir::Value> producedCoupledStates;
  std::map<StandardPartialKey, StandardPartialBoundaryRequirement>
      standardPartialBoundaryRequirements;
  for (const analysis::RootRegionWork &work : rootWorks)
    for (const analysis::ReductionMergeRequirement &merge : work.merges) {
      if (merge.coupledRule || merge.contributions.empty())
        continue;
      for (const analysis::ReductionContribution &contribution :
           merge.contributions) {
        const analysis::RootRegionWork *contributionWork =
            findWork(rootWorks, {work.id.root, contribution.tile});
        const analysis::RootExecutionWork *piece =
            contributionWork
                ? findExecution(*contributionWork, contribution.shard)
                : nullptr;
        if (!piece || !contributionWork->rootOperation)
          return fail<SpatialRegionMaterializationResult>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "standard partial has no selected contribution work");
        llvm::SmallVector<int64_t, 6> shape;
        for (const auto &interval : piece->iterationDomain)
          shape.push_back(interval.size);
        for (const analysis::ReductionResultSlice &result :
             contribution.results) {
          if (result.result >= contributionWork->rootOperation->getNumResults())
            return fail<SpatialRegionMaterializationResult>(
                failure,
                SpatialRegionMaterializationFailureKind::BrokenContract,
                "standard partial result index is out of range");
          auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
              contributionWork->rootOperation->getResult(result.result)
                  .getType());
          StandardPartialKey key{merge.group, contribution.shard,
                                 result.result};
          if (!resultType || shape.empty() ||
              !standardPartialBoundaryRequirements
                   .emplace(key,
                            StandardPartialBoundaryRequirement{
                                contribution.tile,
                                mlir::RankedTensorType::get(
                                    shape, resultType.getElementType())})
                   .second)
            return fail<SpatialRegionMaterializationResult>(
                failure,
                SpatialRegionMaterializationFailureKind::BrokenContract,
                "standard partial endpoint is invalid or duplicated");
        }
      }
    }
  std::map<StandardPartialKey, mlir::Value> producedStandardPartials;

  std::map<DemandFragmentId, ProducedValueKey> fragmentSources;
  auto recordFragmentSource = [&](const DemandFragmentId &fragment) {
    if (fragment.source.kind != analysis::RootBoundaryKind::StructuredResult ||
        !fragment.ownerTile)
      return;
    ProducedValueKey key{fragment.source, fragment.ownerShard,
                         fragment.reductionGroup, *fragment.ownerTile};
    auto matched = plannedResults.find(key);
    if (matched != plannedResults.end())
      fragmentSources.emplace(fragment, *matched);
  };
  std::set<DemandFragmentId> externalFragments;
  for (const RegionGroupPlan &group : regionPlan.groups)
    for (const auto &binding : group.externalBindings) {
      const DemandFragmentId &fragment = binding.fragment;
      if (!externalFragments.insert(fragment).second)
        return fail<SpatialRegionMaterializationResult>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "RegionPlan repeats one external demand fragment");
      recordFragmentSource(fragment);
    }

  std::vector<std::set<size_t>> groupSuccessors(regionPlan.groups.size());
  std::vector<size_t> groupIndegrees(regionPlan.groups.size(), 0);
  std::map<compiler::detail::LogicalShardId, size_t> shardExecutionGroups;
  std::map<compiler::detail::ReductionGroupId, size_t> mergeExecutionGroups;
  for (auto [groupIndex, group] : llvm::enumerate(regionPlan.groups))
    for (const compiler::detail::ExecutionInstancePlan &execution :
         group.executions) {
      if (const auto *root =
              std::get_if<RequiredRootExecution>(&execution.id.source)) {
        if (!shardExecutionGroups.emplace(root->shard, groupIndex).second)
          return fail<SpatialRegionMaterializationResult>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "RegionPlan assigns one shard execution to multiple groups");
      } else {
        const auto &merge =
            std::get<RequiredMergeExecution>(execution.id.source);
        if (!mergeExecutionGroups.emplace(merge.group, groupIndex).second)
          return fail<SpatialRegionMaterializationResult>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "RegionPlan assigns one merge execution to multiple groups");
      }
    }
  auto addGroupEdge = [&](size_t producer, size_t consumer) {
    if (producer == consumer)
      return;
    if (groupSuccessors[producer].insert(consumer).second)
      ++groupIndegrees[consumer];
  };
  for (auto [consumerIndex, group] : llvm::enumerate(regionPlan.groups)) {
    for (const auto &binding : group.externalBindings) {
      if (binding.fragment.source.kind !=
          analysis::RootBoundaryKind::StructuredResult)
        continue;
      auto source = fragmentSources.find(binding.fragment);
      if (source == fragmentSources.end())
        continue;
      auto producer = plannedResultGroups.find(source->second);
      if (producer == plannedResultGroups.end())
        return fail<SpatialRegionMaterializationResult>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "RegionPlan external binding has no producer group");
      if (producer->second == consumerIndex)
        return fail<SpatialRegionMaterializationResult>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "RegionPlan retains a local structured result as an external "
            "binding");
      addGroupEdge(producer->second, consumerIndex);
    }
  }
  for (const auto &[key, requirement] : coupledStateBoundaryRequirements) {
    (void)requirement;
    auto producer = shardExecutionGroups.find(key.contribution);
    auto consumer = mergeExecutionGroups.find(key.group);
    if (producer == shardExecutionGroups.end() ||
        consumer == mergeExecutionGroups.end())
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "coupled state boundary has no producer or merge group");
    addGroupEdge(producer->second, consumer->second);
  }
  for (const auto &[key, requirement] : standardPartialBoundaryRequirements) {
    (void)requirement;
    auto producer = shardExecutionGroups.find(key.contribution);
    auto consumer = mergeExecutionGroups.find(key.group);
    if (producer == shardExecutionGroups.end() ||
        consumer == mergeExecutionGroups.end())
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "standard partial boundary has no producer or merge group");
    addGroupEdge(producer->second, consumer->second);
  }
  llvm::SmallVector<size_t, 64> readyGroups;
  for (auto [groupIndex, indegree] : llvm::enumerate(groupIndegrees))
    if (indegree == 0)
      readyGroups.push_back(groupIndex);
  llvm::SmallVector<unsigned, 64> groupRanks(regionPlan.groups.size(), 0);
  unsigned nextGroupRank = 0;
  while (!readyGroups.empty()) {
    auto next = llvm::min_element(readyGroups, [&](size_t lhs, size_t rhs) {
      return regionPlan.groups[lhs] < regionPlan.groups[rhs];
    });
    size_t groupIndex = *next;
    readyGroups.erase(next);
    groupRanks[groupIndex] = nextGroupRank++;
    for (size_t successor : groupSuccessors[groupIndex])
      if (--groupIndegrees[successor] == 0)
        readyGroups.push_back(successor);
  }
  if (nextGroupRank != regionPlan.groups.size())
    return fail<SpatialRegionMaterializationResult>(
        failure, SpatialRegionMaterializationFailureKind::BrokenContract,
        "RegionPlan external dependency graph is cyclic");

  for (TileId tileId : sortedTiles) {
    TileModuleOp tile = tileModules[tileId.getValue()];
    mlir::Block &tileBody = tile.getBody().front();
    mlir::OpBuilder tileBuilder(&tileBody, tileBody.begin());
    auto entry = tileBuilder.create<mlir::func::FuncOp>(
        source.getLoc(), "entry",
        tileBuilder.getFunctionType(sourceFunction->getArgumentTypes(),
                                    /*results=*/{}));
    entry.addEntryBlock();
    mlir::Block &entryBody = entry.getBody().front();
    mlir::OpBuilder entryBuilder(&entryBody, entryBody.end());
    std::map<SourceValueKey, mlir::BlockArgument> sourceArguments;
    for (auto [index, argument] :
         llvm::enumerate(sourceFunction->getArguments()))
      sourceArguments.emplace(
          SourceValueKey{SourceValueKey::Kind::FunctionArgument,
                         static_cast<uint32_t>(index), 0},
          entry.getArgument(index));
    std::map<DemandFragmentId, mlir::BlockArgument> externalArguments;
    std::map<CoupledStateKey, mlir::BlockArgument> coupledStateArguments;
    std::map<StandardPartialKey, mlir::BlockArgument> standardPartialArguments;

    llvm::SmallVector<size_t, 8> remaining;
    for (auto [index, group] : llvm::enumerate(regionPlan.groups))
      if (group.tile == tileId)
        remaining.push_back(index);
    llvm::sort(remaining, [&](size_t lhs, size_t rhs) {
      return std::tie(groupRanks[lhs], regionPlan.groups[lhs]) <
             std::tie(groupRanks[rhs], regionPlan.groups[rhs]);
    });
    while (!remaining.empty()) {
      auto ready = llvm::find_if(remaining, [&](size_t index) {
        const RegionGroupPlan &group = regionPlan.groups[index];
        for (const auto &binding : group.externalBindings) {
          const DemandFragmentId &fragment = binding.fragment;
          if (fragment.source.kind !=
              analysis::RootBoundaryKind::StructuredResult)
            continue;
          auto planned = fragmentSources.find(fragment);
          if (planned != fragmentSources.end() &&
              planned->second.tile == tileId &&
              produced.find(planned->second) == produced.end())
            return false;
        }
        for (const auto &execution : group.executions) {
          const auto *merge =
              std::get_if<RequiredMergeExecution>(&execution.id.source);
          const analysis::RootRegionWork *work =
              merge ? findWork(rootWorks, merge->work) : nullptr;
          if (!work)
            continue;
          auto requirement =
              llvm::find_if(work->merges, [&](const auto &candidate) {
                return candidate.group == merge->group;
              });
          if (requirement != work->merges.end() && requirement->coupledRule) {
            for (const analysis::ReductionContribution &contribution :
                 requirement->contributions) {
              if (contribution.tile != tileId)
                continue;
              const bool producedInCurrentGroup =
                  llvm::any_of(group.executions, [&](const auto &candidate) {
                    const auto *root = std::get_if<RequiredRootExecution>(
                        &candidate.id.source);
                    return root && root->shard == contribution.shard;
                  });
              if (producedInCurrentGroup)
                continue;
              for (const analysis::CoupledReductionComponentRequirement
                       &component : requirement->components) {
                CoupledStateKey key{requirement->group, contribution.shard,
                                    component.kind};
                if (producedCoupledStates.find(key) ==
                    producedCoupledStates.end())
                  return false;
              }
            }
          } else if (requirement != work->merges.end()) {
            for (const analysis::ReductionContribution &contribution :
                 requirement->contributions) {
              if (contribution.tile != tileId)
                continue;
              const bool producedInCurrentGroup =
                  llvm::any_of(group.executions, [&](const auto &candidate) {
                    const auto *root = std::get_if<RequiredRootExecution>(
                        &candidate.id.source);
                    return root && root->shard == contribution.shard;
                  });
              if (producedInCurrentGroup)
                continue;
              for (const analysis::ReductionResultSlice &result :
                   contribution.results) {
                StandardPartialKey key{requirement->group, contribution.shard,
                                       result.result};
                if (producedStandardPartials.find(key) ==
                    producedStandardPartials.end())
                  return false;
              }
            }
          }
        }
        return true;
      });
      if (ready == remaining.end())
        return fail<SpatialRegionMaterializationResult>(
            failure, SpatialRegionMaterializationFailureKind::BrokenContract,
            "same-Tile Region dependencies are cyclic or missing");
      size_t groupIndex = *ready;
      remaining.erase(ready);
      entryBuilder.setInsertionPointToEnd(&entryBody);
      const RegionGroupPlan &group = regionPlan.groups[groupIndex];
      GroupBuilder groupBuilder(
          source, *sourceFunction, entry, entryBuilder, group, operationNodes,
          rootWorks, fragmentSources, produced,
          coupledStateBoundaryRequirements, producedCoupledStates,
          standardPartialBoundaryRequirements, producedStandardPartials,
          sourceArguments, externalArguments, coupledStateArguments,
          standardPartialArguments, failure);
      auto artifact = groupBuilder.build();
      if (mlir::failed(artifact))
        return mlir::failure();
      for (const GroupResult &groupResult : artifact->results) {
        if (groupResult.resultNumber >= artifact->region.getNumResults() ||
            !produced
                 .try_emplace(groupResult.key, artifact->region.getResult(
                                                   groupResult.resultNumber))
                 .second)
          return fail<SpatialRegionMaterializationResult>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "materialized Region result identity is duplicated");
      }
      for (const GroupCoupledStateResult &state :
           artifact->coupledStateResults) {
        if (state.resultNumber >= artifact->region.getNumResults() ||
            !producedCoupledStates
                 .try_emplace(state.key,
                              artifact->region.getResult(state.resultNumber))
                 .second)
          return fail<SpatialRegionMaterializationResult>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "materialized coupled state endpoint is duplicated");
      }
      for (const GroupStandardPartialResult &partial :
           artifact->standardPartialResults) {
        if (partial.resultNumber >= artifact->region.getNumResults() ||
            !producedStandardPartials
                 .try_emplace(partial.key,
                              artifact->region.getResult(partial.resultNumber))
                 .second)
          return fail<SpatialRegionMaterializationResult>(
              failure, SpatialRegionMaterializationFailureKind::BrokenContract,
              "materialized standard partial endpoint is duplicated");
      }
      for (const GroupBoundaryInput &boundary : artifact->boundaryInputs)
        pendingBoundaries.push_back({boundary.fragment, boundary.sourceTile,
                                     tileId, boundary.destination});
      for (const GroupCoupledStateInput &state : artifact->coupledStateInputs)
        pendingCoupledStateBoundaries.push_back({state.key, state.destination});
      for (const GroupStandardPartialInput &partial :
           artifact->standardPartialInputs)
        pendingStandardPartialBoundaries.push_back(
            {partial.key, partial.destination});
    }
    entryBuilder.setInsertionPointToEnd(&entryBody);
    entryBuilder.create<mlir::func::ReturnOp>(source.getLoc());
    entry.setFunctionType(mlir::FunctionType::get(
        source.getContext(), entry.getArgumentTypes(), /*results=*/{}));
    if (mlir::failed(materializeTileFunctionClosure(source, *sourceFunction,
                                                    tile, failure)))
      return mlir::failure();
  }

  for (const PendingBoundary &pending : pendingBoundaries) {
    auto planned = fragmentSources.find(pending.fragment);
    auto sourceValue = planned != fragmentSources.end()
                           ? produced.find(planned->second)
                           : produced.end();
    if (planned == fragmentSources.end() || sourceValue == produced.end())
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "cross-Tile Region relation has no actual producer endpoint");
    result.relations.boundaryRelations.push_back(
        {sourceValue->second, pending.destination});
  }

  for (const PendingCoupledStateBoundary &pending :
       pendingCoupledStateBoundaries) {
    auto sourceValue = producedCoupledStates.find(pending.key);
    if (sourceValue == producedCoupledStates.end())
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "cross-Tile coupled state has no actual producer endpoint");
    result.relations.boundaryRelations.push_back(
        {sourceValue->second, pending.destination});
  }

  for (const PendingStandardPartialBoundary &pending :
       pendingStandardPartialBoundaries) {
    auto sourceValue = producedStandardPartials.find(pending.key);
    if (sourceValue == producedStandardPartials.end())
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "cross-Tile standard partial has no actual producer endpoint");
    result.relations.boundaryRelations.push_back(
        {sourceValue->second, pending.destination});
  }

  for (auto [outputIndex, output] :
       llvm::enumerate(sourceReturn.getOperands())) {
    auto sourceResult = mlir::dyn_cast<mlir::OpResult>(output);
    auto sourceWork = sourceResult
                          ? llvm::find_if(rootWorks,
                                          [&](const auto &work) {
                                            return work.rootOperation ==
                                                   sourceResult.getOwner();
                                          })
                          : rootWorks.end();
    if (sourceWork == rootWorks.end())
      continue;
    analysis::RootBoundaryId sourceId{
        analysis::RootBoundaryKind::StructuredResult, sourceWork->id.root,
        sourceResult.getResultNumber()};
    bool foundOutput = false;
    for (const auto &[key, endpoint] : produced)
      if (key.source == sourceId) {
        result.relations.structuralOutputs.push_back(
            {static_cast<unsigned>(outputIndex), endpoint});
        foundOutput = true;
      }
    if (!foundOutput)
      return fail<SpatialRegionMaterializationResult>(
          failure, SpatialRegionMaterializationFailureKind::BrokenContract,
          "observable structured result has no selected Tile endpoint");
  }

  if (mlir::failed(materializeTileLocalSplatConstants(*result.module)) ||
      mlir::failed(mlir::verify(*result.module)) ||
      mlir::failed(verifyTileModuleCollection(*result.module)) ||
      mlir::failed(verifyStructuralTileRegions(*result.module)) ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          result.module->getOperation(), result.relations))) {
    return fail<SpatialRegionMaterializationResult>(
        failure, SpatialRegionMaterializationFailureKind::CompilerFailure,
        "structural materialization produced invalid IR or stale current "
        "relations");
  }
  return result;
}

} // namespace wafer
