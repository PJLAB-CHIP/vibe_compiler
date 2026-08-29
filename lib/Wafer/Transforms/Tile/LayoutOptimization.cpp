//===- LayoutOptimization.cpp - Current layout cleanup -------===//

#include "Wafer/Transforms/Tile/LayoutOptimization.h"

#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <vector>

namespace wafer::compiler::detail {
namespace {

enum class ChoiceKind : uint8_t { Keep, Erase, Reuse };

struct LayoutChoice {
  LayoutMaterializeOp operation;
  LayoutMaterializeOp replacement;
  std::vector<ChoiceKind> states;
};

using MaterializationKey = std::pair<mlir::Value, mlir::Type>;

bool isReadOnlyUse(mlir::OpOperand &use) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(use.getOwner());
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> instances;
  effects.getEffectsOnValue(use.get(), instances);
  return !instances.empty() && llvm::all_of(instances, [](const auto &effect) {
    return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
  });
}

bool resultHasOnlyReadUses(LayoutMaterializeOp operation) {
  return llvm::all_of(operation.getResult().getUses(),
                      [](mlir::OpOperand &use) { return isReadOnlyUse(use); });
}

bool hasNoInterveningSourceWrite(LayoutMaterializeOp prior,
                                 LayoutMaterializeOp current,
                                 mlir::AliasAnalysis &aliases) {
  if (prior->getBlock() != current->getBlock() ||
      !prior->isBeforeInBlock(current))
    return false;
  for (mlir::Operation *operation = prior->getNextNode();
       operation && operation != current.getOperation();
       operation = operation->getNextNode())
    if (aliases.getModRef(operation, current.getSource()).isMod())
      return false;
  return true;
}

void retargetRelationValue(StructuredMaterializationRelations &relations,
                           mlir::Value oldValue, mlir::Value newValue) {
  auto retarget = [&](auto &entries) {
    for (auto &entry : entries)
      if (entry.buffer == oldValue)
        entry.buffer = newValue;
  };
  retarget(relations.operationResultBuffers);
  retarget(relations.operandBuffers);
  retarget(relations.scratchBuffers);
  retarget(relations.outputBuffers);
  retarget(relations.ddrBuffers);
  retarget(relations.partialReductionContributions);
  retarget(relations.partialReductionMergeInputs);
  for (auto &entry : relations.structuralOutputs)
    if (entry.endpoint == oldValue)
      entry.endpoint = newValue;
  for (auto &entry : relations.boundaryRelations) {
    if (entry.sourceEndpoint == oldValue)
      entry.sourceEndpoint = newValue;
    if (entry.destinationEndpoint == oldValue)
      entry.destinationEndpoint = newValue;
  }
}

void preserveEmissionOwners(StructuredMaterializationRelations &relations,
                            mlir::Operation *oldOperation,
                            mlir::Operation *newOperation) {
  llvm::SmallVector<uint32_t, 4> owners;
  for (const StructuredOperationEmissionRelation &entry :
       relations.operationEmissions)
    if (entry.operation == oldOperation &&
        !llvm::is_contained(owners, entry.structuredNodeId))
      owners.push_back(entry.structuredNodeId);
  for (uint32_t owner : owners)
    if (!llvm::any_of(relations.operationEmissions, [&](const auto &entry) {
          return entry.operation == newOperation &&
                 entry.structuredNodeId == owner;
        }))
      relations.operationEmissions.push_back({owner, newOperation});
}

} // namespace

LayoutOptimizationResult
optimizeTileLayouts(llvm::MutableArrayRef<LayoutOptimizationInput> modules,
                    uint64_t workLimit) {
  LayoutOptimizationResult result;
  result.statistics.invocations = 1;
  result.statistics.hardOnlyInvocations = 1;
  if (modules.empty() || workLimit == 0) {
    result.status = workLimit == 0 ? ExactPBQPStatus::Indeterminate
                                   : ExactPBQPStatus::BrokenContract;
    result.detail = modules.empty() ? "current layout query has no Tile IR"
                                    : "current layout PBQP has no work budget";
    return result;
  }

  std::vector<LayoutChoice> choices;
  ExactPBQPProblem problem;
  for (LayoutOptimizationInput &candidate : modules) {
    if (!candidate.module || !candidate.relations ||
        mlir::failed(mlir::verify(candidate.module)) ||
        mlir::failed(checkStructuredBufferRelationsCurrent(
            candidate.module, *candidate.relations))) {
      result.detail = "current layout query received invalid IR or stale "
                      "buffer relations";
      return result;
    }

    mlir::DominanceInfo dominance(candidate.module);
    mlir::AliasAnalysis aliases(candidate.module);
    llvm::DenseMap<MaterializationKey,
                   llvm::SmallVector<LayoutMaterializeOp, 2>>
        available;
    candidate.module.walk([&](LayoutMaterializeOp operation) {
      ++result.statistics.layoutMaterializationsBefore;
      LayoutChoice choice;
      choice.operation = operation;
      if (operation.getResult().use_empty()) {
        choice.states = {ChoiceKind::Erase, ChoiceKind::Keep};
      } else if (resultHasOnlyReadUses(operation)) {
        MaterializationKey key{operation.getSource(),
                               operation.getResult().getType()};
        auto found = available.find(key);
        if (found != available.end()) {
          auto replacement =
              llvm::find_if(found->second, [&](LayoutMaterializeOp prior) {
                return hasNoInterveningSourceWrite(prior, operation, aliases) &&
                       llvm::all_of(operation.getResult().getUses(),
                                    [&](mlir::OpOperand &use) {
                                      return dominance.dominates(
                                          prior.getResult(), use.getOwner());
                                    });
              });
          if (replacement != found->second.end()) {
            choice.replacement = *replacement;
            choice.states = {ChoiceKind::Reuse, ChoiceKind::Keep};
          }
        }
        available[key].push_back(operation);
      }
      if (choice.states.empty())
        choice.states = {ChoiceKind::Keep};
      problem.variables.push_back(ExactPBQPVariable{
          std::vector<ExactPBQPCost>(choice.states.size(), 0)});
      choices.push_back(std::move(choice));
    });
  }

  // An empty current layout domain is a valid identity transformation. Keep a
  // one-state variable so the shared exact solver still supplies the same
  // typed budget and determinism contract as non-empty domains.
  if (problem.variables.empty())
    problem.variables.push_back(ExactPBQPVariable{{0}});
  ExactPBQPResult solved = solveExactPBQP(problem, workLimit);
  result.status = solved.status;
  result.statistics.solverWork = solved.work;
  if (solved.status != ExactPBQPStatus::Optimal) {
    result.detail = "current layout PBQP did not produce an Optimal assignment";
    return result;
  }
  if (solved.assignment.size() != problem.variables.size() ||
      solved.assignment.size() < choices.size()) {
    result.status = ExactPBQPStatus::BrokenContract;
    result.detail = "current layout PBQP returned an incomplete assignment";
    return result;
  }

  llvm::DenseMap<mlir::Operation *, size_t> choiceIndices;
  for (auto [index, choice] : llvm::enumerate(choices))
    choiceIndices.try_emplace(choice.operation, index);
  std::vector<ChoiceKind> selectedChoices;
  std::vector<LayoutMaterializeOp> selectedReplacements(choices.size());
  selectedChoices.reserve(choices.size());
  for (auto [index, choice] : llvm::enumerate(choices)) {
    const uint32_t state = solved.assignment[index];
    if (state >= choice.states.size()) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail = "current layout PBQP selected an invalid state";
      return result;
    }
    const ChoiceKind selected = choice.states[state];
    selectedChoices.push_back(selected);
    if (selected != ChoiceKind::Reuse)
      continue;
    auto replacement = choiceIndices.find(choice.replacement);
    if (replacement == choiceIndices.end() || replacement->second >= index) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail = "current layout reuse has no earlier selected owner";
      return result;
    }
    selectedReplacements[index] =
        selectedChoices[replacement->second] == ChoiceKind::Reuse
            ? selectedReplacements[replacement->second]
            : choice.replacement;
    if (!selectedReplacements[index]) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail = "current layout reuse resolved to an empty owner";
      return result;
    }
  }

  // All failure-prone checks completed above. Each actual mutation is a
  // same-typed SSA replacement or dead-op erase and is performed through one
  // rewriter per owning module.
  for (LayoutOptimizationInput &candidate : modules) {
    mlir::IRRewriter rewriter(candidate.module->getContext());
    for (auto [index, choice] : llvm::enumerate(choices)) {
      if (choice.operation->getParentOfType<mlir::ModuleOp>() !=
          candidate.module)
        continue;
      switch (selectedChoices[index]) {
      case ChoiceKind::Keep:
        break;
      case ChoiceKind::Erase:
        rewriter.eraseOp(choice.operation);
        ++result.statistics.unusedMaterializationsErased;
        break;
      case ChoiceKind::Reuse:
        preserveEmissionOwners(*candidate.relations, choice.operation,
                               selectedReplacements[index]);
        retargetRelationValue(*candidate.relations,
                              choice.operation.getResult(),
                              selectedReplacements[index].getResult());
        rewriter.replaceAllUsesWith(choice.operation.getResult(),
                                    selectedReplacements[index].getResult());
        rewriter.eraseOp(choice.operation);
        ++result.statistics.sharedMaterializationsReused;
        break;
      }
    }

    // Reuse can make an earlier materialization dead. Close that local DCE
    // chain without introducing another planning decision.
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<LayoutMaterializeOp, 4> dead;
      candidate.module.walk([&](LayoutMaterializeOp operation) {
        if (operation.getResult().use_empty())
          dead.push_back(operation);
      });
      for (LayoutMaterializeOp operation : dead) {
        rewriter.eraseOp(operation);
        ++result.statistics.unusedMaterializationsErased;
        changed = true;
      }
    }
    retainCurrentStructuredBufferRelations(candidate.module,
                                           *candidate.relations);
    if (mlir::failed(checkStructuredBufferRelationsCurrent(
            candidate.module, *candidate.relations)) ||
        mlir::failed(mlir::verify(candidate.module))) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail =
          "current layout assignment produced invalid IR or stale relations";
      return result;
    }
    candidate.module.walk([&](LayoutMaterializeOp) {
      ++result.statistics.layoutMaterializationsAfter;
    });
  }
  return result;
}

} // namespace wafer::compiler::detail
