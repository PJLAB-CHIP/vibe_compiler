//===- ReadyOrder.cpp - Actual instruction ready ordering ----------------===//

#include "Wafer/Transforms/PhysicalDataflow.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace wafer {
namespace {

struct BufferAccess {
  mlir::Value value;
  bool write = false;
};

static bool isReadyOrderOperation(mlir::Operation *operation) {
  return mlir::isa<WaferInstructionOpInterface, SyncLocalFenceOp>(operation);
}

static bool isLocalFence(mlir::Operation *operation) {
  auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!interface)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> effects;
  interface.getEffects(effects);
  return llvm::any_of(effects, [](const auto &effect) {
    return llvm::isa<WaferSyncResource>(effect.getResource()) &&
           llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect());
  });
}

static void collectBufferAccesses(
    mlir::Operation *operation,
    llvm::SmallVectorImpl<BufferAccess> &accesses) {
  auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!interface)
    return;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> effects;
  interface.getEffects(effects);
  for (const auto &effect : effects) {
    mlir::Value value = effect.getValue();
    if (!value)
      continue;
    bool read = llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
    bool write = llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect());
    if (!read && !write)
      continue;
    auto found = llvm::find_if(accesses, [&](const BufferAccess &access) {
      return access.value == value;
    });
    if (found == accesses.end())
      accesses.push_back({value, write});
    else
      found->write |= write;
  }
}

static unsigned getReadyPriority(mlir::Operation *operation) {
  auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(operation);
  if (!instruction)
    return 3;
  switch (instruction.getInstructionFamily()) {
  case InstrFamily::RDMA:
  case InstrFamily::WDMA:
  case InstrFamily::TDMA:
    return 0;
  case InstrFamily::DTE:
    return 1;
  case InstrFamily::CT:
  case InstrFamily::NE:
    return 2;
  }
  llvm_unreachable("unknown instruction family");
}

static unsigned scheduleRun(llvm::ArrayRef<mlir::Operation *> operations) {
  if (operations.size() < 2)
    return 0;

  const unsigned count = operations.size();
  llvm::DenseMap<mlir::Operation *, unsigned> indices;
  for (auto [index, operation] : llvm::enumerate(operations))
    indices.try_emplace(operation, static_cast<unsigned>(index));

  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> successors(count);
  llvm::SmallVector<unsigned, 16> indegrees(count, 0);
  auto addEdge = [&](unsigned from, unsigned to) {
    if (from == to || llvm::is_contained(successors[from], to))
      return;
    successors[from].push_back(to);
    ++indegrees[to];
  };

  llvm::DenseMap<mlir::Value, unsigned> lastWriters;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<unsigned, 4>> readers;
  std::optional<unsigned> lastFence;
  for (unsigned index = 0; index < count; ++index) {
    mlir::Operation *operation = operations[index];
    for (mlir::Value operand : operation->getOperands()) {
      mlir::Operation *definition = operand.getDefiningOp();
      auto found = indices.find(definition);
      if (found != indices.end())
        addEdge(found->second, index);
    }

    if (lastFence)
      addEdge(*lastFence, index);
    if (isLocalFence(operation)) {
      for (unsigned predecessor = 0; predecessor < index; ++predecessor)
        addEdge(predecessor, index);
      lastFence = index;
      continue;
    }

    llvm::SmallVector<BufferAccess, 4> accesses;
    collectBufferAccesses(operation, accesses);
    for (const BufferAccess &access : accesses) {
      auto writer = lastWriters.find(access.value);
      if (writer != lastWriters.end())
        addEdge(writer->second, index);
      if (!access.write) {
        readers[access.value].push_back(index);
        continue;
      }
      auto activeReaders = readers.find(access.value);
      if (activeReaders != readers.end()) {
        for (unsigned reader : activeReaders->second)
          addEdge(reader, index);
        activeReaders->second.clear();
      }
      lastWriters[access.value] = index;
    }
  }

  llvm::SmallVector<unsigned, 16> order;
  llvm::SmallVector<bool, 16> emitted(count, false);
  while (order.size() != count) {
    unsigned selected = count;
    unsigned selectedPriority = std::numeric_limits<unsigned>::max();
    for (unsigned index = 0; index < count; ++index) {
      if (emitted[index] || indegrees[index] != 0)
        continue;
      unsigned priority = getReadyPriority(operations[index]);
      if (selected == count || priority < selectedPriority) {
        selected = index;
        selectedPriority = priority;
      }
    }
    if (selected == count)
      return 0;
    emitted[selected] = true;
    order.push_back(selected);
    for (unsigned successor : successors[selected])
      --indegrees[successor];
  }

  unsigned moved = 0;
  for (auto [position, originalIndex] : llvm::enumerate(order))
    moved += position != originalIndex;
  if (moved == 0)
    return 0;

  mlir::Block *block = operations.front()->getBlock();
  auto insertionPoint = operations.back()->getIterator();
  ++insertionPoint;
  for (unsigned originalIndex : order)
    operations[originalIndex]->moveBefore(block, insertionPoint);
  return moved;
}

static unsigned scheduleBlock(mlir::Block &block) {
  unsigned moved = 0;
  llvm::SmallVector<mlir::Operation *, 16> run;
  auto flush = [&] {
    moved += scheduleRun(run);
    run.clear();
  };
  for (mlir::Operation &operation : block) {
    if (isReadyOrderOperation(&operation))
      run.push_back(&operation);
    else
      flush();
  }
  flush();
  return moved;
}

static unsigned scheduleRegion(mlir::Region &region) {
  unsigned moved = 0;
  for (mlir::Block &block : region) {
    for (mlir::Operation &operation : block)
      for (mlir::Region &nested : operation.getRegions())
        moved += scheduleRegion(nested);
    moved += scheduleBlock(block);
  }
  return moved;
}

} // namespace

unsigned scheduleIndependentInstructionsByReadyOrder(mlir::Operation *scope) {
  if (!scope)
    return 0;
  unsigned moved = 0;
  for (mlir::Region &region : scope->getRegions())
    moved += scheduleRegion(region);
  return moved;
}

} // namespace wafer
