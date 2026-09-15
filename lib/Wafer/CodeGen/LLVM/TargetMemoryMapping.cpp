//===- TargetMemoryMapping.cpp - Reuse read mappings across loops ------===//

#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"

#include "Wafer/Support/CompileTiming.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/TargetParser/Triple.h"

namespace wafer::compiler::detail {
namespace {

static std::optional<TargetCallBuiltin> getBuiltin(llvm::Value *value) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(value);
  auto *callee = call ? call->getCalledFunction() : nullptr;
  const auto *descriptor =
      callee ? findTargetCallDescriptor(callee->getName()) : nullptr;
  const auto *builtin = descriptor
                            ? std::get_if<TargetCallBuiltin>(&descriptor->semantic)
                            : nullptr;
  return builtin ? std::optional<TargetCallBuiltin>(*builtin) : std::nullopt;
}

static bool isSPMPointer(llvm::Value *value) {
  while (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(value))
    value = gep->getPointerOperand();
  return getBuiltin(value) == TargetCallBuiltin::SPMMapping;
}

// Mapping is an acquire, so ordinary LICM cannot treat the SDK function as
// readnone. Only move it through this witnessed DDR-read-only instruction
// stream. Joins, publication/acquire, DDR writers and unknown calls block it.
static bool preservesDDRVisibility(const llvm::Loop &loop) {
  for (auto *block : loop.blocks())
    for (auto &instruction : *block) {
      if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&instruction)) {
        llvm::Value *pointer = load->getPointerOperand();
        while (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(pointer))
          pointer = gep->getPointerOperand();
        auto builtin = getBuiltin(pointer);
        if (load->isAtomic() ||
            (builtin != TargetCallBuiltin::DDRReadMapping &&
             builtin != TargetCallBuiltin::SPMMapping))
          return false;
        continue;
      }
      if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&instruction)) {
        if (store->isAtomic() || !isSPMPointer(store->getPointerOperand()))
          return false;
        continue;
      }
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
        if (auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(call)) {
          if (intrinsic->doesNotAccessMemory())
            continue;
          return false;
        }
        auto builtin = getBuiltin(call);
        if (builtin == TargetCallBuiltin::DDRReadMapping ||
            builtin == TargetCallBuiltin::SPMMapping ||
            builtin == TargetCallBuiltin::RDMA)
          continue;
        return false;
      }
      if (instruction.mayWriteToMemory())
        return false;
    }
  return true;
}

} // namespace

void hoistReadOnlyDDRMemoryMappings(llvm::Module &module) {
  uint64_t hoisted = 0;
  for (auto &function : module) {
    if (function.isDeclaration())
      continue;
    bool hasMapping = false;
    for (auto &block : function)
      for (auto &instruction : block)
        hasMapping |= getBuiltin(&instruction) == TargetCallBuiltin::DDRReadMapping;
    if (!hasMapping)
      continue;
    llvm::DominatorTree dominators(function);
    llvm::LoopInfo loops(dominators);
    llvm::AssumptionCache assumptions(function);
    llvm::TargetLibraryInfoImpl libraryImpl(llvm::Triple(module.getTargetTriple()));
    llvm::TargetLibraryInfo library(libraryImpl);
    llvm::ScalarEvolution evolution(function, library, assumptions, dominators, loops);
    auto orderedLoops = loops.getLoopsInPreorder();
    for (llvm::Loop *loop : llvm::reverse(orderedLoops)) {
      auto *preheader = loop->getLoopPreheader();
      auto *latch = loop->getLoopLatch();
      if (!preheader || !latch || !preservesDDRVisibility(*loop))
        continue;
      const auto *backedges = llvm::dyn_cast<llvm::SCEVConstant>(
          evolution.getBackedgeTakenCount(loop));
      // A positive exact backedge count proves an execution of the body.
      // Zero/unknown-trip loops retain the acquire at the original access.
      if (!backedges || backedges->getAPInt().isZero())
        continue;
      llvm::SmallVector<llvm::CallInst *, 4> mappings;
      for (auto *block : loop->blocks())
        for (auto &instruction : *block) {
          if (getBuiltin(&instruction) != TargetCallBuiltin::DDRReadMapping)
            continue;
          auto *call = llvm::cast<llvm::CallInst>(&instruction);
          if (dominators.dominates(block, latch) &&
              llvm::all_of(call->args(), [&](llvm::Value *value) {
                return loop->isLoopInvariant(value);
              }))
            mappings.push_back(call);
        }
      for (auto *call : mappings) {
        call->moveBefore(preheader->getTerminator());
        ++hoisted;
      }
    }
  }
  support::addCompileCounter("target-memory-mapping", "hoisted", hoisted);
}

} // namespace wafer::compiler::detail
