//===- NoCPartialDataflow.cpp - Typed partial/reduction residency -------===//

#include "NoCPartialDataflow.h"

#include "Wafer/Analysis/StaticBufferRange.h"
#include "Wafer/Compiler/GlobalTileRelation.h"
#include "Wafer/IR/Common/OpVerifierUtils.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

namespace wafer::compiler::detail {
namespace {

enum class ReductionProtocolKind {
  AllReduceTree,
  AllReduceRing,
};

struct ReductionProtocolKey {
  int64_t communicationId = -1;
  ReductionProtocolKind kind = ReductionProtocolKind::AllReduceTree;

  friend bool operator<(const ReductionProtocolKey &lhs,
                        const ReductionProtocolKey &rhs) {
    return std::tie(lhs.communicationId, lhs.kind) <
           std::tie(rhs.communicationId, rhs.kind);
  }
  friend bool operator==(const ReductionProtocolKey &lhs,
                         const ReductionProtocolKey &rhs) {
    return lhs.communicationId == rhs.communicationId && lhs.kind == rhs.kind;
  }
};

struct PartialSpillCut {
  int64_t logicalRank = -1;
  mlir::memref::AllocOp spillRoot;
  InstrWDMAOp store;
  InstrRDMAOp load;
  mlir::Value producerBuffer;
  mlir::Value consumerBuffer;
  int64_t bytes = 0;
  llvm::SmallVector<mlir::memref::DeallocOp, 2> spillDeallocations;
  llvm::SmallVector<mlir::memref::CastOp, 4> spillCasts;
  llvm::SmallVector<mlir::memref::DeallocOp, 2> consumerDeallocations;
};

struct PublisherIdentity {
  int64_t outputIndex = -1;
  StaticTileRegion globalTile;
  mlir::Type bufferType;
};

struct MessageKey {
  int64_t sourceRank = -1;
  int64_t destinationRank = -1;
  int64_t communicationId = -1;
  DTEProtocolPhase phase = DTEProtocolPhase::AllReduceTreeReduce;
  int64_t round = -1;
  int64_t payloadSlice = -1;
  int64_t bytes = -1;

  friend bool operator<(const MessageKey &lhs, const MessageKey &rhs) {
    return std::tie(lhs.sourceRank, lhs.destinationRank, lhs.communicationId,
                    lhs.phase, lhs.round, lhs.payloadSlice, lhs.bytes) <
           std::tie(rhs.sourceRank, rhs.destinationRank, rhs.communicationId,
                    rhs.phase, rhs.round, rhs.payloadSlice, rhs.bytes);
  }
  friend bool operator==(const MessageKey &lhs, const MessageKey &rhs) {
    return lhs.sourceRank == rhs.sourceRank &&
           lhs.destinationRank == rhs.destinationRank &&
           lhs.communicationId == rhs.communicationId &&
           lhs.phase == rhs.phase && lhs.round == rhs.round &&
           lhs.payloadSlice == rhs.payloadSlice && lhs.bytes == rhs.bytes;
  }
};

using analysis::StaticBufferRange;
using analysis::StaticByteRange;

struct ProvenanceBinding {
  StaticByteRange bytes;
  unsigned node = 0;
};

struct CutProtocolProof {
  PartialSpillCut *cut = nullptr;
  ReductionProtocolKey protocol;
  PublisherIdentity publisher;
  unsigned publisherNode = 0;
  std::map<MessageKey, unsigned> sendNodes;
  std::set<MessageKey> receiveMessages;
  llvm::SmallVector<mlir::OpOperand *, 8> consumerUses;
  llvm::SmallVector<ProvenanceBinding, 8> publisherPieces;
  std::map<MessageKey, ProvenanceBinding> sendBindings;
};

enum class ProvenanceNodeKind {
  LocalPartial,
  Receive,
  Merge,
  Project,
};

struct ProvenanceNode {
  ProvenanceNodeKind kind = ProvenanceNodeKind::LocalPartial;
  StaticByteRange bytes;
  int64_t logicalRank = -1;
  MessageKey message;
  llvm::SmallVector<unsigned, 2> inputs;
  std::optional<InstrElementwiseKind> combiner;
};

class ProvenanceArena {
public:
  unsigned createLocal(int64_t logicalRank, StaticByteRange bytes) {
    ProvenanceNode node;
    node.kind = ProvenanceNodeKind::LocalPartial;
    node.bytes = bytes;
    node.logicalRank = logicalRank;
    nodes.push_back(std::move(node));
    return nodes.size() - 1;
  }
  unsigned createLocal(int64_t logicalRank) {
    return createLocal(logicalRank, StaticByteRange{});
  }

  unsigned createReceive(const MessageKey &message, StaticByteRange bytes) {
    ProvenanceNode node;
    node.kind = ProvenanceNodeKind::Receive;
    node.bytes = bytes;
    node.message = message;
    nodes.push_back(std::move(node));
    return nodes.size() - 1;
  }
  unsigned createReceive(const MessageKey &message) {
    return createReceive(message, StaticByteRange{});
  }

  unsigned createMerge(StaticByteRange bytes, llvm::ArrayRef<unsigned> inputs,
                       InstrElementwiseKind combiner) {
    ProvenanceNode node;
    node.kind = ProvenanceNodeKind::Merge;
    node.bytes = bytes;
    llvm::append_range(node.inputs, inputs);
    node.combiner = combiner;
    nodes.push_back(std::move(node));
    return nodes.size() - 1;
  }
  unsigned createMerge(llvm::ArrayRef<unsigned> inputs,
                       InstrElementwiseKind combiner) {
    return createMerge(StaticByteRange{}, inputs, combiner);
  }

  unsigned createProject(StaticByteRange bytes, unsigned input) {
    ProvenanceNode node;
    node.kind = ProvenanceNodeKind::Project;
    node.bytes = bytes;
    node.inputs.push_back(input);
    nodes.push_back(std::move(node));
    return nodes.size() - 1;
  }

  std::optional<StaticByteRange> getNodeBytes(unsigned node) const {
    if (node == 0 || node >= nodes.size())
      return std::nullopt;
    return nodes[node].bytes;
  }

  llvm::ArrayRef<ProvenanceNode> getNodes() const { return nodes; }

private:
  // Node zero is reserved so a default-constructed proof cannot accidentally
  // name a real provenance expression.
  llvm::SmallVector<ProvenanceNode, 32> nodes{ProvenanceNode{}};
};

struct TreeProtocolInventory {
  int64_t root = -1;
  llvm::SmallVector<int64_t, 16> parent;
  llvm::SmallVector<llvm::SmallVector<int64_t, 2>, 16> children;
  std::set<MessageKey> messages;
};

struct RingProtocolInventory {
  int64_t chunkBytes = 0;
  llvm::SmallVector<int64_t, 16> successor;
  llvm::SmallVector<int64_t, 16> predecessor;
  std::set<MessageKey> messages;
};

struct BufferStateEntry {
  StaticBufferRange storage;
  unsigned node = 0;
};

class BufferState {
public:
  std::optional<unsigned> lookup(const StaticBufferRange &storage,
                                 ProvenanceArena &arena) const {
    const BufferStateEntry *result = nullptr;
    for (const BufferStateEntry &entry : entries) {
      if (entry.storage.root != storage.root ||
          !analysis::staticByteRangeContains(entry.storage.bytes,
                                             storage.bytes))
        continue;
      if (result)
        return std::nullopt;
      result = &entry;
    }
    if (!result)
      return std::nullopt;
    std::optional<StaticByteRange> nodeBytes = arena.getNodeBytes(result->node);
    if (!nodeBytes ||
        !analysis::staticByteRangeContains(*nodeBytes, storage.bytes))
      return std::nullopt;
    if (*nodeBytes == storage.bytes)
      return result->node;
    return arena.createProject(storage.bytes, result->node);
  }

  bool hasRoot(mlir::Value root) const {
    return llvm::any_of(entries, [&](const BufferStateEntry &entry) {
      return entry.storage.root == root;
    });
  }

  bool assign(const StaticBufferRange &storage, unsigned node) {
    if (!storage.root || storage.bytes.begin < 0 ||
        storage.bytes.end <= storage.bytes.begin || node == 0)
      return false;
    llvm::SmallVector<BufferStateEntry, 16> updated;
    for (const BufferStateEntry &entry : entries) {
      if (entry.storage.root != storage.root ||
          analysis::staticByteRangesAreDisjoint(entry.storage.bytes,
                                                storage.bytes)) {
        updated.push_back(entry);
        continue;
      }
      if (entry.storage.bytes.begin < storage.bytes.begin)
        updated.push_back(BufferStateEntry{
            StaticBufferRange{entry.storage.root,
                              StaticByteRange{entry.storage.bytes.begin,
                                              storage.bytes.begin}},
            entry.node});
      if (storage.bytes.end < entry.storage.bytes.end)
        updated.push_back(BufferStateEntry{
            StaticBufferRange{
                entry.storage.root,
                StaticByteRange{storage.bytes.end, entry.storage.bytes.end}},
            entry.node});
    }
    updated.push_back(BufferStateEntry{storage, node});
    entries = std::move(updated);
    return true;
  }

  bool exactCover(const StaticBufferRange &storage,
                  llvm::SmallVectorImpl<ProvenanceBinding> &pieces,
                  ProvenanceArena &arena) const {
    llvm::SmallVector<const BufferStateEntry *, 8> candidates;
    for (const BufferStateEntry &entry : entries) {
      if (entry.storage.root != storage.root)
        continue;
      if (!analysis::staticByteRangeContains(storage.bytes,
                                             entry.storage.bytes))
        return false;
      candidates.push_back(&entry);
    }
    llvm::sort(candidates,
               [](const BufferStateEntry *lhs, const BufferStateEntry *rhs) {
                 return lhs->storage.bytes.begin < rhs->storage.bytes.begin;
               });
    int64_t cursor = storage.bytes.begin;
    for (const BufferStateEntry *entry : candidates) {
      if (entry->storage.bytes.begin != cursor || entry->node == 0)
        return false;
      unsigned node = entry->node;
      if (node == 0)
        return false;
      node = arena.createProject(entry->storage.bytes, node);
      pieces.push_back(ProvenanceBinding{entry->storage.bytes, node});
      cursor = entry->storage.bytes.end;
    }
    return !pieces.empty() && cursor == storage.bytes.end;
  }

  llvm::ArrayRef<BufferStateEntry> getEntries() const { return entries; }

private:
  llvm::SmallVector<BufferStateEntry, 16> entries;
};

static std::optional<StaticByteRange>
getMessagePayloadRange(const MessageKey &message, int64_t completeBytes) {
  if (completeBytes <= 0 || message.bytes <= 0)
    return std::nullopt;
  if (message.phase == DTEProtocolPhase::AllReduceTreeReduce ||
      message.phase == DTEProtocolPhase::AllReduceTreeBroadcast)
    return message.bytes == completeBytes
               ? std::optional<StaticByteRange>(
                     StaticByteRange{0, completeBytes})
               : std::nullopt;
  if (message.phase != DTEProtocolPhase::AllReduceRing ||
      message.payloadSlice < 0 ||
      message.payloadSlice >
          std::numeric_limits<int64_t>::max() / message.bytes)
    return std::nullopt;
  int64_t begin = message.payloadSlice * message.bytes;
  if (begin > completeBytes || message.bytes > completeBytes - begin)
    return std::nullopt;
  return StaticByteRange{begin, begin + message.bytes};
}

static std::optional<StaticBufferRange>
getCompleteBufferRange(mlir::Value value, int64_t completeBytes) {
  std::optional<StaticBufferRange> range =
      analysis::resolveStaticBufferRange(value);
  if (!range || range->bytes != StaticByteRange{0, completeBytes})
    return std::nullopt;
  return range;
}

static mlir::Value stripTransparentCasts(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      auto sourceType =
          mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
      auto resultType =
          mlir::dyn_cast<mlir::MemRefType>(cast.getResult().getType());
      if (!sourceType || !resultType ||
          sourceType.getShape() != resultType.getShape() ||
          sourceType.getElementType() != resultType.getElementType() ||
          sourceType.getMemorySpace() != resultType.getMemorySpace())
        break;
      value = cast.getSource();
      continue;
    }
    break;
  }
  return value;
}

static mlir::Value getAliasRoot(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    mlir::Value castSource = stripTransparentCasts(value);
    if (castSource != value) {
      value = castSource;
      continue;
    }
    auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
        value.getDefiningOp());
    if (!view)
      break;
    mlir::Value source = view.getViewSource();
    if (!source || source == value)
      break;
    value = source;
  }
  return value;
}

/// Returns an identity suitable for full-buffer provenance. Range-changing
/// views deliberately fail closed: allocation-root equality alone cannot
/// prove that two subviews denote the same bytes.
static std::optional<mlir::Value> getExactBufferIdentity(mlir::Value value) {
  mlir::Value identity = stripTransparentCasts(value);
  if (!identity || mlir::isa_and_nonnull<mlir::ViewLikeOpInterface>(
                       identity.getDefiningOp()))
    return std::nullopt;
  return identity;
}

static bool sameRoot(mlir::Value lhs, mlir::Value rhs) {
  if (!lhs || !rhs)
    return false;
  std::optional<StaticBufferRange> left =
      analysis::resolveStaticBufferRange(lhs);
  std::optional<StaticBufferRange> right =
      analysis::resolveStaticBufferRange(rhs);
  if (left && right)
    return left->root == right->root;
  return getAliasRoot(lhs) == getAliasRoot(rhs);
}

static std::optional<int64_t> getPhysicalBytes(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || !memref.hasStaticShape())
    return std::nullopt;
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(memref);
  return info && info->physicalBytes > 0
             ? std::optional<int64_t>(info->physicalBytes)
             : std::nullopt;
}

static bool isUnitDescriptor(llvm::ArrayRef<int64_t> strides,
                             llvm::ArrayRef<int64_t> iterations) {
  return strides.size() == 3 && iterations.size() == 3 &&
         llvm::all_of(strides, [](int64_t value) { return value == 0; }) &&
         llvm::all_of(iterations, [](int64_t value) { return value == 1; });
}

static bool isCompleteStore(InstrWDMAOp store, int64_t bytes) {
  return (!store.getSrcOffsetAttr() ||
          store.getSrcOffsetAttr().getInt() == 0) &&
         (!store.getDstOffsetAttr() ||
          store.getDstOffsetAttr().getInt() == 0) &&
         store.getByteCountAttr().getInt() == bytes &&
         store.getInnerBytesAttr().getInt() == bytes &&
         isUnitDescriptor(store.getDstStrides(), store.getDstIterations());
}

static bool isCompleteLoad(InstrRDMAOp load, int64_t bytes) {
  return (!load.getSrcOffsetAttr() || load.getSrcOffsetAttr().getInt() == 0) &&
         (!load.getDstOffsetAttr() || load.getDstOffsetAttr().getInt() == 0) &&
         load.getByteCountAttr().getInt() == bytes &&
         load.getInnerBytesAttr().getInt() == bytes &&
         isUnitDescriptor(load.getSrcStrides(), load.getSrcIterations());
}

static bool isTransparentCast(mlir::memref::CastOp cast, mlir::Value source) {
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto resultType =
      mlir::dyn_cast<mlir::MemRefType>(cast.getResult().getType());
  return cast.getSource() == source && sourceType && resultType &&
         sourceType.getShape() == resultType.getShape() &&
         sourceType.getElementType() == resultType.getElementType() &&
         sourceType.getMemorySpace() == resultType.getMemorySpace();
}

static bool hasOnlySpillUses(mlir::Value value, InstrWDMAOp store,
                             InstrRDMAOp load,
                             llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (owner == store.getOperation() && use.getOperandNumber() == 1)
      continue;
    if (owner == load.getOperation() && use.getOperandNumber() == 0)
      continue;
    if (mlir::isa<mlir::memref::DeallocOp>(owner))
      continue;
    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(owner)) {
      if (use.getOperandNumber() != 0 || !isTransparentCast(cast, value) ||
          !hasOnlySpillUses(cast.getResult(), store, load, visited))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

static bool operationWritesRoot(mlir::Operation *operation, mlir::Value value) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [&](const auto &instance) {
    return instance.getValue() && sameRoot(instance.getValue(), value) &&
           mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
  });
}

static mlir::Operation *findLastWriterBefore(mlir::Value value,
                                             mlir::Operation *before) {
  mlir::Operation *writer = nullptr;
  mlir::Block *block = before ? before->getBlock() : nullptr;
  if (!block)
    return nullptr;
  for (mlir::Operation &operation : *block) {
    if (&operation == before)
      break;
    if (!operationWritesRoot(&operation, value))
      continue;
    writer = &operation;
  }
  return writer;
}

static bool producerSnapshotRemainsStable(mlir::ModuleOp module,
                                          mlir::Value producer,
                                          mlir::Operation *snapshotStore) {
  bool stable = true;
  module.walk([&](mlir::Operation *operation) {
    if (!stable || operation == snapshotStore)
      return;
    bool referencesProducer =
        llvm::any_of(operation->getOperands(), [&](mlir::Value operand) {
          return sameRoot(operand, producer);
        });
    if (!referencesProducer)
      return;
    if (operation->getBlock() == snapshotStore->getBlock() &&
        operation->isBeforeInBlock(snapshotStore))
      return;

    auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (!effects) {
      stable = false;
      return;
    }
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
    effects.getEffects(instances);
    bool sawProducerEffect = false;
    for (const auto &instance : instances) {
      if (!instance.getValue() || !sameRoot(instance.getValue(), producer))
        continue;
      sawProducerEffect = true;
      if (!mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect())) {
        stable = false;
        return;
      }
    }
    if (!sawProducerEffect)
      stable = false;
  });
  return stable;
}

static bool isTypedPartialProducer(mlir::Operation *operation) {
  auto instruction =
      mlir::dyn_cast_or_null<WaferInstructionOpInterface>(operation);
  return instruction &&
         (instruction.getInstructionFamily() == InstrFamily::CT ||
          instruction.getInstructionFamily() == InstrFamily::NE);
}

static llvm::SmallVector<PartialSpillCut, 4>
collectPartialSpillCuts(mlir::ModuleOp module, int64_t rank) {
  llvm::SmallVector<mlir::memref::AllocOp, 8> roots;
  module.walk([&](mlir::memref::AllocOp allocation) {
    if (wafer::detail::hasWaferMemorySpace(allocation.getType(),
                                           MemorySpace::DDR))
      roots.push_back(allocation);
  });
  llvm::SmallVector<PartialSpillCut, 4> cuts;
  for (mlir::memref::AllocOp root : roots) {
    llvm::SmallVector<InstrWDMAOp, 2> stores;
    llvm::SmallVector<InstrRDMAOp, 2> loads;
    module.walk([&](InstrWDMAOp store) {
      if (sameRoot(store.getDest(), root.getResult()))
        stores.push_back(store);
    });
    module.walk([&](InstrRDMAOp load) {
      if (sameRoot(load.getSource(), root.getResult()))
        loads.push_back(load);
    });
    if (stores.size() != 1 || loads.size() != 1)
      continue;

    InstrWDMAOp store = stores.front();
    InstrRDMAOp load = loads.front();
    if (store->getBlock() != load->getBlock() || !store->isBeforeInBlock(load))
      continue;
    std::optional<int64_t> producerBytes =
        getPhysicalBytes(store.getSource().getType());
    std::optional<int64_t> spillBytes = getPhysicalBytes(root.getType());
    std::optional<int64_t> consumerBytes =
        getPhysicalBytes(load.getDest().getType());
    std::optional<mlir::Value> producerIdentity =
        getExactBufferIdentity(store.getSource());
    std::optional<mlir::Value> consumerIdentity =
        getExactBufferIdentity(load.getDest());
    mlir::Operation *producer =
        findLastWriterBefore(store.getSource(), store.getOperation());
    if (!producerBytes || !spillBytes || !consumerBytes || !producerIdentity ||
        !consumerIdentity || *producerBytes != *spillBytes ||
        *producerBytes != *consumerBytes ||
        store.getSource().getType() != load.getDest().getType() ||
        !load.getDest().getDefiningOp<mlir::memref::AllocOp>() ||
        !isCompleteStore(store, *producerBytes) ||
        !isCompleteLoad(load, *producerBytes) ||
        !isTypedPartialProducer(producer) ||
        !producerSnapshotRemainsStable(module, store.getSource(),
                                       store.getOperation()))
      continue;

    llvm::DenseSet<mlir::Value> visited;
    if (!hasOnlySpillUses(root.getResult(), store, load, visited))
      continue;

    PartialSpillCut cut{
        rank,           root,          store, load, store.getSource(),
        load.getDest(), *producerBytes};
    llvm::DenseSet<mlir::Operation *> cleanup;
    for (mlir::Value alias : visited)
      for (mlir::OpOperand &use : alias.getUses()) {
        mlir::Operation *owner = use.getOwner();
        if (!cleanup.insert(owner).second)
          continue;
        if (auto deallocation = mlir::dyn_cast<mlir::memref::DeallocOp>(owner))
          cut.spillDeallocations.push_back(deallocation);
        else if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(owner);
                 cast && isTransparentCast(cast, alias))
          cut.spillCasts.push_back(cast);
      }

    mlir::Value consumerRoot = getAliasRoot(cut.consumerBuffer);
    if (auto allocation = consumerRoot.getDefiningOp<mlir::memref::AllocOp>()) {
      for (mlir::OpOperand &use : allocation.getResult().getUses())
        if (auto deallocation =
                mlir::dyn_cast<mlir::memref::DeallocOp>(use.getOwner()))
          cut.consumerDeallocations.push_back(deallocation);
    }
    cuts.push_back(std::move(cut));
  }
  return cuts;
}

static std::optional<ReductionProtocolKey>
getReductionProtocol(DTEMessageAttr message) {
  switch (message.getPhase()) {
  case DTEProtocolPhase::AllReduceTreeReduce:
  case DTEProtocolPhase::AllReduceTreeBroadcast:
    return ReductionProtocolKey{message.getCommunicationId(),
                                ReductionProtocolKind::AllReduceTree};
  case DTEProtocolPhase::AllReduceRing:
    return ReductionProtocolKey{message.getCommunicationId(),
                                ReductionProtocolKind::AllReduceRing};
  case DTEProtocolPhase::CollectivePermute:
  case DTEProtocolPhase::AllToAll:
  case DTEProtocolPhase::AllGatherDirect:
  case DTEProtocolPhase::AllGatherRing:
  case DTEProtocolPhase::ReduceScatterDirect:
  case DTEProtocolPhase::ReduceScatterRing:
  case DTEProtocolPhase::PeerDataflow:
    return std::nullopt;
  }
  return std::nullopt;
}

static bool isSupportedCombiner(InstrElementwiseKind kind) {
  return kind == InstrElementwiseKind::Add ||
         kind == InstrElementwiseKind::Max || kind == InstrElementwiseKind::Min;
}

static const frontend::ProgramRankSlice *
findRankSlice(llvm::ArrayRef<frontend::ProgramRankSlice> slices, int64_t rank) {
  const frontend::ProgramRankSlice *result = nullptr;
  for (const frontend::ProgramRankSlice &slice : slices) {
    if (slice.logicalRank != rank)
      continue;
    if (result)
      return nullptr;
    result = &slice;
  }
  return result;
}

static std::optional<PublisherIdentity>
resolvePublisher(InstrWDMAOp store, int64_t rank,
                 const frontend::FrontendProgramVerificationResult &program,
                 int64_t expectedBytes) {
  std::optional<mlir::Value> destinationIdentity =
      getExactBufferIdentity(store.getDest());
  std::optional<mlir::Value> sourceIdentity =
      getExactBufferIdentity(store.getSource());
  if (!destinationIdentity || !sourceIdentity ||
      getPhysicalBytes(store.getSource().getType()) != expectedBytes ||
      getPhysicalBytes(store.getDest().getType()) != expectedBytes ||
      !isCompleteStore(store, expectedBytes))
    return std::nullopt;

  mlir::Value destination = *destinationIdentity;
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(destination);
  if (!argument)
    return std::nullopt;
  mlir::Block *owner = argument.getOwner();
  auto tileRegion =
      owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
            : TileRegionOp();
  if (!tileRegion || tileRegion.getBody().empty() ||
      owner != &tileRegion.getBody().front() ||
      argument.getArgNumber() >= tileRegion.getInputs().size() ||
      program.distributedOutputs.empty() ||
      tileRegion.getInputs().size() < program.distributedOutputs.size())
    return std::nullopt;

  size_t outputBase =
      tileRegion.getInputs().size() - program.distributedOutputs.size();
  if (argument.getArgNumber() < outputBase)
    return std::nullopt;
  size_t outputOrdinal = argument.getArgNumber() - outputBase;
  if (outputOrdinal >= program.distributedOutputs.size())
    return std::nullopt;
  const frontend::ProgramBoundaryBinding &binding =
      program.distributedOutputs[outputOrdinal];
  if (binding.index != static_cast<int64_t>(outputOrdinal) ||
      binding.distribution != frontend::ProgramDistributionKind::Replicated)
    return std::nullopt;

  auto destinationType =
      mlir::dyn_cast<mlir::MemRefType>(store.getDest().getType());
  std::optional<mlir::RankedTensorType> logical =
      destinationType ? wafer::detail::getLogicalTensorType(destinationType)
                      : std::nullopt;
  if (!destinationType || !logical ||
      logical->getShape() != llvm::ArrayRef<int64_t>(binding.localShape))
    return std::nullopt;
  const frontend::ProgramRankSlice *slice =
      findRankSlice(binding.rankSlices, rank);
  if (!slice)
    return std::nullopt;

  StaticTileRegion local;
  local.offsets.assign(binding.localShape.size(), 0);
  local.sizes = binding.localShape;
  local.strides.assign(binding.localShape.size(), 1);
  llvm::Expected<StaticTileRegion> global = mapRankLocalTileToGlobal(
      *slice, binding.globalShape, binding.localShape, local);
  if (!global) {
    llvm::consumeError(global.takeError());
    return std::nullopt;
  }
  return PublisherIdentity{binding.index, std::move(*global),
                           store.getSource().getType()};
}

static bool isAfterInSameBlock(mlir::Operation *operation,
                               mlir::Operation *anchor) {
  return operation && anchor && operation->getBlock() == anchor->getBlock() &&
         anchor->isBeforeInBlock(operation);
}

static llvm::SmallVector<ReductionProtocolKey, 4>
collectProtocolsAfter(PartialSpillCut &cut) {
  std::set<ReductionProtocolKey> protocols;
  cut.load->getParentOfType<mlir::ModuleOp>().walk(
      [&](mlir::Operation *operation) {
        if (!isAfterInSameBlock(operation, cut.load.getOperation()))
          return;
        std::optional<ReductionProtocolKey> protocol;
        if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
          protocol = getReductionProtocol(send.getMessage());
        else if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation))
          protocol = getReductionProtocol(recv.getMessage());
        if (protocol)
          protocols.insert(*protocol);
      });
  return {protocols.begin(), protocols.end()};
}

static bool tokenHasExactWait(mlir::Value token) {
  if (!token.hasOneUse())
    return false;
  auto wait = mlir::dyn_cast<InstrDTEWaitOp>(*token.getUsers().begin());
  mlir::Operation *issue = token.getDefiningOp();
  return wait && issue && issue->getBlock() == wait->getBlock() &&
         issue->isBeforeInBlock(wait);
}

static bool isCompleteGatherScatter(InstrGatherScatterOp move, int64_t bytes) {
  std::optional<StaticBufferRange> source =
      analysis::resolveStaticBufferRange(move.getSource());
  std::optional<StaticBufferRange> dest =
      analysis::resolveStaticBufferRange(move.getDest());
  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(move.getSource().getType());
  auto destType = mlir::dyn_cast<mlir::MemRefType>(move.getDest().getType());
  return source && dest && source->bytes.length() == bytes &&
         dest->bytes.length() == bytes && sourceType && destType &&
         sourceType.getElementType() == destType.getElementType() &&
         sourceType.getMemorySpace() == destType.getMemorySpace() &&
         (!move.getSrcOffsetAttr() || move.getSrcOffsetAttr().getInt() == 0) &&
         (!move.getDstOffsetAttr() || move.getDstOffsetAttr().getInt() == 0) &&
         move.getByteCountAttr().getInt() == bytes &&
         move.getInnerBytesAttr().getInt() == bytes &&
         isUnitDescriptor(move.getSrcStrides(), move.getSrcIterations()) &&
         isUnitDescriptor(move.getDstStrides(), move.getDstIterations());
}

static MessageKey getSendMessageKey(InstrDTESendOp send, int64_t rank) {
  return MessageKey{rank,
                    static_cast<int64_t>(send.getPeer()),
                    send.getMessage().getCommunicationId(),
                    send.getMessage().getPhase(),
                    send.getMessage().getRound(),
                    send.getMessage().getPayloadSlice(),
                    static_cast<int64_t>(send.getBytes())};
}

static MessageKey getRecvMessageKey(InstrDTERecvOp recv, int64_t rank) {
  return MessageKey{static_cast<int64_t>(recv.getPeer()),
                    rank,
                    recv.getMessage().getCommunicationId(),
                    recv.getMessage().getPhase(),
                    recv.getMessage().getRound(),
                    recv.getMessage().getPayloadSlice(),
                    static_cast<int64_t>(recv.getBytes())};
}

static bool operationTouchesRoot(mlir::Operation *operation, mlir::Value root) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [&](const auto &instance) {
    return instance.getValue() && sameRoot(instance.getValue(), root);
  });
}

static bool isFinalOutputWriter(mlir::ModuleOp module,
                                mlir::Operation *publisher,
                                mlir::Value outputRoot) {
  if (!publisher || !outputRoot)
    return false;
  bool finalWriter = true;
  module.walk([&](mlir::Operation *operation) {
    if (!finalWriter || operation == publisher ||
        !operationWritesRoot(operation, outputRoot))
      return;
    if (operation->getBlock() != publisher->getBlock() ||
        publisher->isBeforeInBlock(operation))
      finalWriter = false;
  });
  return finalWriter;
}

static std::optional<CutProtocolProof>
proveCutProtocol(PartialSpillCut &cut, const ReductionProtocolKey &protocol,
                 const frontend::FrontendProgramVerificationResult &program,
                 ProvenanceArena &arena) {
  mlir::Block *block = cut.load->getBlock();
  if (!block)
    return std::nullopt;

  std::optional<StaticBufferRange> consumer =
      getCompleteBufferRange(cut.consumerBuffer, cut.bytes);
  if (!consumer)
    return std::nullopt;

  BufferState state;
  if (!state.assign(
          *consumer,
          arena.createLocal(cut.logicalRank, StaticByteRange{0, cut.bytes})))
    return std::nullopt;

  struct PendingReceive {
    StaticBufferRange storage;
    StaticByteRange payload;
    MessageKey message;
  };
  llvm::DenseMap<mlir::Value, PendingReceive> pendingReceives;
  std::optional<PublisherIdentity> publisher;
  unsigned publisherNode = 0;
  llvm::SmallVector<ProvenanceBinding, 8> publisherPieces;
  mlir::Operation *publisherOperation = nullptr;
  mlir::Value publisherDestination;
  std::map<MessageKey, unsigned> sendNodes;
  std::map<MessageKey, ProvenanceBinding> sendBindings;
  std::set<MessageKey> receiveMessages;
  llvm::DenseSet<mlir::Operation *> tracedOperations;
  auto aliasesState = [&](mlir::Value value) {
    return value && llvm::any_of(state.getEntries(), [&](const auto &entry) {
             return sameRoot(value, entry.storage.root);
           });
  };

  for (mlir::Operation &operation : *block) {
    if (!isAfterInSameBlock(&operation, cut.load.getOperation()))
      continue;

    if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(&operation)) {
      for (mlir::Value token : wait.getTokens()) {
        auto pending = pendingReceives.find(token);
        if (pending == pendingReceives.end())
          continue;
        unsigned node = arena.createReceive(pending->second.message,
                                            pending->second.payload);
        if (!state.assign(pending->second.storage, node))
          return std::nullopt;
        pendingReceives.erase(token);
      }
      continue;
    }

    if (publisherOperation) {
      if (operationWritesRoot(&operation, publisherDestination))
        return std::nullopt;
      if (!mlir::isa<mlir::memref::DeallocOp>(&operation))
        for (const BufferStateEntry &entry : state.getEntries())
          if (operationTouchesRoot(&operation, entry.storage.root))
            return std::nullopt;
    }

    for (const auto &entry : pendingReceives)
      if (operationTouchesRoot(&operation, entry.second.storage.root))
        return std::nullopt;

    if (auto move = mlir::dyn_cast<InstrGatherScatterOp>(&operation)) {
      std::optional<StaticBufferRange> source =
          analysis::resolveStaticBufferRange(move.getSource());
      std::optional<StaticBufferRange> dest =
          analysis::resolveStaticBufferRange(move.getDest());
      if (!source || !dest) {
        if (aliasesState(move.getSource()) || aliasesState(move.getDest()))
          return std::nullopt;
        continue;
      }
      std::optional<unsigned> sourceState = state.lookup(*source, arena);
      if (!sourceState) {
        if (state.hasRoot(dest->root))
          return std::nullopt;
        continue;
      }
      if (source->bytes != dest->bytes ||
          !isCompleteGatherScatter(move, source->bytes.length()) ||
          !state.assign(*dest, *sourceState))
        return std::nullopt;
      tracedOperations.insert(&operation);
      continue;
    }

    if (auto elementwise = mlir::dyn_cast<InstrElementwiseOp>(&operation)) {
      llvm::SmallVector<unsigned, 4> inputs;
      std::optional<StaticByteRange> inputBytes;
      for (mlir::Value input : elementwise.getInputs()) {
        std::optional<StaticBufferRange> storage =
            analysis::resolveStaticBufferRange(input);
        if (!storage) {
          if (aliasesState(input))
            return std::nullopt;
          continue;
        }
        std::optional<unsigned> inputState = state.lookup(*storage, arena);
        if (!inputState) {
          if (state.hasRoot(storage->root))
            return std::nullopt;
          continue;
        }
        if (inputBytes && *inputBytes != storage->bytes)
          return std::nullopt;
        inputBytes = storage->bytes;
        inputs.push_back(*inputState);
      }
      std::optional<StaticBufferRange> dest =
          analysis::resolveStaticBufferRange(elementwise.getDest());
      if (!dest) {
        if (!inputs.empty() || aliasesState(elementwise.getDest()))
          return std::nullopt;
        continue;
      }
      if (inputs.empty()) {
        if (state.hasRoot(dest->root))
          return std::nullopt;
        continue;
      }
      if (inputs.size() != elementwise.getInputs().size() || !inputBytes ||
          *inputBytes != dest->bytes ||
          !isSupportedCombiner(elementwise.getKind()) ||
          dest->bytes.length() <= 0)
        return std::nullopt;
      unsigned node =
          arena.createMerge(dest->bytes, inputs, elementwise.getKind());
      if (!state.assign(*dest, node))
        return std::nullopt;
      tracedOperations.insert(&operation);
      continue;
    }

    if (auto send = mlir::dyn_cast<InstrDTESendOp>(&operation)) {
      std::optional<ReductionProtocolKey> candidate =
          getReductionProtocol(send.getMessage());
      if (!candidate || !(*candidate == protocol))
        continue;
      std::optional<StaticBufferRange> storage =
          analysis::resolveStaticBufferRange(send.getBuffer());
      if (!storage)
        return std::nullopt;
      MessageKey message = getSendMessageKey(send, cut.logicalRank);
      std::optional<StaticByteRange> payload =
          getMessagePayloadRange(message, cut.bytes);
      std::optional<unsigned> current = state.lookup(*storage, arena);
      if (!payload || storage->bytes != *payload || !current ||
          storage->bytes.length() != static_cast<int64_t>(send.getBytes()) ||
          !tokenHasExactWait(send.getToken()) ||
          !sendNodes.emplace(message, *current).second ||
          !sendBindings.emplace(message, ProvenanceBinding{*payload, *current})
               .second)
        return std::nullopt;
      tracedOperations.insert(&operation);
      continue;
    }

    if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(&operation)) {
      std::optional<ReductionProtocolKey> candidate =
          getReductionProtocol(recv.getMessage());
      std::optional<StaticBufferRange> storage =
          analysis::resolveStaticBufferRange(recv.getBuffer());
      if (!storage) {
        if ((candidate && *candidate == protocol) ||
            aliasesState(recv.getBuffer()))
          return std::nullopt;
        continue;
      }
      if (!candidate || !(*candidate == protocol)) {
        if (state.hasRoot(storage->root))
          return std::nullopt;
        continue;
      }
      MessageKey message = getRecvMessageKey(recv, cut.logicalRank);
      std::optional<StaticByteRange> payload =
          getMessagePayloadRange(message, cut.bytes);
      if (!payload || storage->bytes != *payload ||
          storage->bytes.length() != static_cast<int64_t>(recv.getBytes()) ||
          !tokenHasExactWait(recv.getToken()) ||
          !receiveMessages.insert(message).second ||
          !pendingReceives
               .try_emplace(recv.getToken(),
                            PendingReceive{*storage, *payload, message})
               .second)
        return std::nullopt;
      tracedOperations.insert(&operation);
      continue;
    }

    if (auto store = mlir::dyn_cast<InstrWDMAOp>(&operation)) {
      std::optional<StaticBufferRange> source =
          getCompleteBufferRange(store.getSource(), cut.bytes);
      if (!source) {
        if (aliasesState(store.getSource()))
          return std::nullopt;
        continue;
      }
      llvm::SmallVector<ProvenanceBinding, 8> pieces;
      if (!state.exactCover(*source, pieces, arena))
        continue;
      std::optional<PublisherIdentity> candidate =
          resolvePublisher(store, cut.logicalRank, program, cut.bytes);
      if (!candidate || publisher)
        return std::nullopt;
      publisher = std::move(*candidate);
      publisherPieces = std::move(pieces);
      if (publisherPieces.size() == 1)
        publisherNode = publisherPieces.front().node;
      publisherOperation = &operation;
      publisherDestination = getAliasRoot(store.getDest());
      tracedOperations.insert(&operation);
      continue;
    }

    for (const BufferStateEntry &entry : state.getEntries())
      if (operationWritesRoot(&operation, entry.storage.root))
        return std::nullopt;
  }

  if (!pendingReceives.empty() || !publisher || publisherPieces.empty() ||
      sendNodes.empty() || receiveMessages.empty() ||
      !isFinalOutputWriter(cut.load->getParentOfType<mlir::ModuleOp>(),
                           publisherOperation, publisherDestination))
    return std::nullopt;

  llvm::SmallVector<mlir::OpOperand *, 8> consumerUses;
  for (mlir::OpOperand &use : cut.consumerBuffer.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (owner == cut.load.getOperation() && use.getOperandNumber() == 1)
      continue;
    if (mlir::isa<mlir::memref::DeallocOp>(owner))
      continue;
    if (!tracedOperations.contains(owner) ||
        !isAfterInSameBlock(owner, cut.load.getOperation()) ||
        (owner != publisherOperation &&
         !owner->isBeforeInBlock(publisherOperation)) ||
        operationWritesRoot(owner, cut.consumerBuffer))
      return std::nullopt;
    consumerUses.push_back(&use);
  }
  CutProtocolProof proof;
  proof.cut = &cut;
  proof.protocol = protocol;
  proof.publisher = std::move(*publisher);
  proof.publisherNode = publisherNode;
  proof.sendNodes = std::move(sendNodes);
  proof.receiveMessages = std::move(receiveMessages);
  proof.consumerUses = std::move(consumerUses);
  proof.publisherPieces = std::move(publisherPieces);
  proof.sendBindings = std::move(sendBindings);
  return proof;
}

static std::optional<TreeProtocolInventory>
collectTreeProtocolInventory(llvm::ArrayRef<mlir::ModuleOp> modules,
                             const ReductionProtocolKey &protocol) {
  std::map<MessageKey, unsigned> sends;
  std::map<MessageKey, unsigned> recvs;
  bool valid = true;

  for (auto [rank, module] : llvm::enumerate(modules)) {
    mlir::ModuleOp mutableModule = module;
    mutableModule.walk([&](mlir::Operation *operation) {
      if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation)) {
        std::optional<ReductionProtocolKey> candidate =
            getReductionProtocol(send.getMessage());
        if (!candidate || !(*candidate == protocol))
          return;
        int64_t peer = send.getPeer();
        if (peer < 0 || peer >= static_cast<int64_t>(modules.size()) ||
            !tokenHasExactWait(send.getToken())) {
          valid = false;
          return;
        }
        MessageKey key = getSendMessageKey(send, static_cast<int64_t>(rank));
        ++sends[key];
        return;
      }
      auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation);
      if (!recv)
        return;
      std::optional<ReductionProtocolKey> candidate =
          getReductionProtocol(recv.getMessage());
      if (!candidate || !(*candidate == protocol))
        return;
      int64_t peer = recv.getPeer();
      if (peer < 0 || peer >= static_cast<int64_t>(modules.size()) ||
          !tokenHasExactWait(recv.getToken())) {
        valid = false;
        return;
      }
      MessageKey key = getRecvMessageKey(recv, static_cast<int64_t>(rank));
      ++recvs[key];
    });
  }

  if (!valid || sends.empty() || sends != recvs ||
      llvm::any_of(sends, [](const auto &entry) { return entry.second != 1; }))
    return std::nullopt;

  const size_t rankCount = modules.size();
  TreeProtocolInventory inventory;
  inventory.parent.assign(rankCount, -1);
  inventory.children.resize(rankCount);
  std::map<int64_t, MessageKey> reduceByChild;
  std::set<MessageKey> broadcast;
  for (const auto &entry : sends) {
    const MessageKey &message = entry.first;
    inventory.messages.insert(message);
    if (message.phase == DTEProtocolPhase::AllReduceTreeReduce) {
      if (message.sourceRank == message.destinationRank ||
          inventory.parent[static_cast<size_t>(message.sourceRank)] != -1)
        return std::nullopt;
      inventory.parent[static_cast<size_t>(message.sourceRank)] =
          message.destinationRank;
      reduceByChild.emplace(message.sourceRank, message);
      continue;
    }
    if (message.phase != DTEProtocolPhase::AllReduceTreeBroadcast)
      return std::nullopt;
    broadcast.insert(message);
  }
  if (reduceByChild.size() + 1 != rankCount ||
      broadcast.size() + 1 != rankCount)
    return std::nullopt;

  for (size_t rank = 0; rank < rankCount; ++rank) {
    int64_t parent = inventory.parent[rank];
    if (parent < 0) {
      if (inventory.root >= 0)
        return std::nullopt;
      inventory.root = static_cast<int64_t>(rank);
      continue;
    }
    inventory.children[static_cast<size_t>(parent)].push_back(
        static_cast<int64_t>(rank));
  }
  if (inventory.root < 0)
    return std::nullopt;

  llvm::SmallVector<int64_t, 16> depth(rankCount, -1);
  depth[static_cast<size_t>(inventory.root)] = 0;
  llvm::SmallVector<int64_t, 16> worklist{inventory.root};
  while (!worklist.empty()) {
    int64_t parent = worklist.pop_back_val();
    for (int64_t child : inventory.children[static_cast<size_t>(parent)]) {
      if (depth[static_cast<size_t>(child)] >= 0)
        return std::nullopt;
      depth[static_cast<size_t>(child)] =
          depth[static_cast<size_t>(parent)] + 1;
      worklist.push_back(child);
    }
  }
  if (llvm::any_of(depth, [](int64_t value) { return value < 0; }))
    return std::nullopt;

  for (const auto &entry : reduceByChild) {
    const MessageKey &reduce = entry.second;
    if (reduce.round != depth[static_cast<size_t>(reduce.sourceRank)])
      return std::nullopt;
    MessageKey reverse{reduce.destinationRank,
                       reduce.sourceRank,
                       reduce.communicationId,
                       DTEProtocolPhase::AllReduceTreeBroadcast,
                       reduce.round,
                       reduce.payloadSlice,
                       reduce.bytes};
    if (!broadcast.erase(reverse))
      return std::nullopt;
  }
  if (!broadcast.empty())
    return std::nullopt;
  return inventory;
}

static std::optional<RingProtocolInventory>
collectRingProtocolInventory(llvm::ArrayRef<mlir::ModuleOp> modules,
                             const ReductionProtocolKey &protocol) {
  if (protocol.kind != ReductionProtocolKind::AllReduceRing ||
      modules.size() < 2 ||
      modules.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;

  std::map<MessageKey, unsigned> sends;
  std::map<MessageKey, unsigned> recvs;
  bool valid = true;
  for (auto [rank, module] : llvm::enumerate(modules)) {
    mlir::ModuleOp mutableModule = module;
    mutableModule.walk([&](mlir::Operation *operation) {
      if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation)) {
        std::optional<ReductionProtocolKey> candidate =
            getReductionProtocol(send.getMessage());
        if (!candidate || !(*candidate == protocol))
          return;
        int64_t peer = send.getPeer();
        if (peer < 0 || peer >= static_cast<int64_t>(modules.size()) ||
            !tokenHasExactWait(send.getToken())) {
          valid = false;
          return;
        }
        ++sends[getSendMessageKey(send, static_cast<int64_t>(rank))];
        return;
      }
      auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation);
      if (!recv)
        return;
      std::optional<ReductionProtocolKey> candidate =
          getReductionProtocol(recv.getMessage());
      if (!candidate || !(*candidate == protocol))
        return;
      int64_t peer = recv.getPeer();
      if (peer < 0 || peer >= static_cast<int64_t>(modules.size()) ||
          !tokenHasExactWait(recv.getToken())) {
        valid = false;
        return;
      }
      ++recvs[getRecvMessageKey(recv, static_cast<int64_t>(rank))];
    });
  }
  if (!valid || sends.empty() || sends != recvs ||
      llvm::any_of(sends, [](const auto &entry) { return entry.second != 1; }))
    return std::nullopt;

  const int64_t rankCount = static_cast<int64_t>(modules.size());
  if (rankCount - 1 > std::numeric_limits<int64_t>::max() / INT64_C(2))
    return std::nullopt;
  const int64_t roundCount = 2 * (rankCount - 1);
  if (static_cast<uint64_t>(rankCount) >
      std::numeric_limits<size_t>::max() / static_cast<uint64_t>(roundCount))
    return std::nullopt;
  const size_t expectedMessages = static_cast<size_t>(
      static_cast<uint64_t>(rankCount) * static_cast<uint64_t>(roundCount));
  if (sends.size() != expectedMessages)
    return std::nullopt;

  RingProtocolInventory inventory;
  inventory.successor.assign(static_cast<size_t>(rankCount), -1);
  inventory.predecessor.assign(static_cast<size_t>(rankCount), -1);
  llvm::SmallVector<std::set<int64_t>, 16> sources(roundCount);
  llvm::SmallVector<std::set<int64_t>, 16> destinations(roundCount);
  llvm::SmallVector<std::set<int64_t>, 16> slices(roundCount);
  for (const auto &entry : sends) {
    const MessageKey &message = entry.first;
    if (message.phase != DTEProtocolPhase::AllReduceRing ||
        message.communicationId != protocol.communicationId ||
        message.sourceRank < 0 || message.sourceRank >= rankCount ||
        message.destinationRank < 0 || message.destinationRank >= rankCount ||
        message.sourceRank == message.destinationRank || message.round < 0 ||
        message.round >= roundCount || message.payloadSlice < 0 ||
        message.payloadSlice >= rankCount || message.bytes <= 0)
      return std::nullopt;
    if (inventory.chunkBytes == 0)
      inventory.chunkBytes = message.bytes;
    else if (inventory.chunkBytes != message.bytes)
      return std::nullopt;

    int64_t &successor =
        inventory.successor[static_cast<size_t>(message.sourceRank)];
    int64_t &predecessor =
        inventory.predecessor[static_cast<size_t>(message.destinationRank)];
    if ((successor >= 0 && successor != message.destinationRank) ||
        (predecessor >= 0 && predecessor != message.sourceRank))
      return std::nullopt;
    successor = message.destinationRank;
    predecessor = message.sourceRank;
    if (!sources[static_cast<size_t>(message.round)]
             .insert(message.sourceRank)
             .second ||
        !destinations[static_cast<size_t>(message.round)]
             .insert(message.destinationRank)
             .second ||
        !slices[static_cast<size_t>(message.round)]
             .insert(message.payloadSlice)
             .second)
      return std::nullopt;
    inventory.messages.insert(message);
  }
  for (int64_t round = 0; round < roundCount; ++round)
    if (sources[static_cast<size_t>(round)].size() !=
            static_cast<size_t>(rankCount) ||
        destinations[static_cast<size_t>(round)].size() !=
            static_cast<size_t>(rankCount) ||
        slices[static_cast<size_t>(round)].size() !=
            static_cast<size_t>(rankCount))
      return std::nullopt;

  llvm::SmallVector<unsigned char, 16> visited(static_cast<size_t>(rankCount),
                                               0);
  int64_t current = 0;
  for (int64_t step = 0; step < rankCount; ++step) {
    if (current < 0 || current >= rankCount ||
        visited[static_cast<size_t>(current)])
      return std::nullopt;
    visited[static_cast<size_t>(current)] = true;
    current = inventory.successor[static_cast<size_t>(current)];
  }
  if (current != 0 || llvm::any_of(visited, [](bool value) { return !value; }))
    return std::nullopt;
  return inventory;
}

struct ResolvedProvenance {
  std::map<int64_t, uint64_t> originMultiplicity;
  std::set<MessageKey> receives;
  std::set<InstrElementwiseKind> combiners;
};

static bool resolveProvenance(
    unsigned node, llvm::ArrayRef<ProvenanceNode> nodes,
    const std::map<MessageKey, unsigned> &sendNodes,
    llvm::MutableArrayRef<unsigned char> marks,
    llvm::MutableArrayRef<std::optional<ResolvedProvenance>> cache) {
  if (node == 0 || node >= nodes.size())
    return false;
  if (marks[node] == 2)
    return cache[node].has_value();
  if (marks[node] == 1)
    return false;
  marks[node] = 1;

  ResolvedProvenance result;
  const ProvenanceNode &expression = nodes[node];
  switch (expression.kind) {
  case ProvenanceNodeKind::LocalPartial:
    if (expression.logicalRank < 0 ||
        expression.bytes.end <= expression.bytes.begin)
      return false;
    result.originMultiplicity.emplace(expression.logicalRank, 1);
    break;
  case ProvenanceNodeKind::Receive: {
    auto send = sendNodes.find(expression.message);
    if (send == sendNodes.end() || send->second >= nodes.size() ||
        nodes[send->second].bytes != expression.bytes ||
        !resolveProvenance(send->second, nodes, sendNodes, marks, cache))
      return false;
    result = *cache[send->second];
    result.receives.insert(expression.message);
    break;
  }
  case ProvenanceNodeKind::Merge:
    if (!expression.combiner || expression.inputs.empty() ||
        expression.bytes.end <= expression.bytes.begin)
      return false;
    for (unsigned input : expression.inputs) {
      if (input >= nodes.size() || nodes[input].bytes != expression.bytes ||
          !resolveProvenance(input, nodes, sendNodes, marks, cache))
        return false;
      const ResolvedProvenance &resolved = *cache[input];
      for (const auto &[rank, count] : resolved.originMultiplicity) {
        uint64_t &aggregate = result.originMultiplicity[rank];
        if (count > std::numeric_limits<uint64_t>::max() - aggregate)
          return false;
        aggregate += count;
      }
      result.receives.insert(resolved.receives.begin(),
                             resolved.receives.end());
      result.combiners.insert(resolved.combiners.begin(),
                              resolved.combiners.end());
    }
    result.combiners.insert(*expression.combiner);
    break;
  case ProvenanceNodeKind::Project: {
    if (expression.inputs.size() != 1 ||
        expression.inputs.front() >= nodes.size() ||
        !analysis::staticByteRangeContains(
            nodes[expression.inputs.front()].bytes, expression.bytes) ||
        !resolveProvenance(expression.inputs.front(), nodes, sendNodes, marks,
                           cache))
      return false;
    result = *cache[expression.inputs.front()];
    break;
  }
  }

  cache[node] = std::move(result);
  marks[node] = 2;
  return true;
}

static bool verifyCompleteProof(llvm::ArrayRef<mlir::ModuleOp> modules,
                                llvm::ArrayRef<CutProtocolProof> proofs,
                                const ReductionProtocolKey &protocol,
                                const ProvenanceArena &arena) {
  if (proofs.size() != modules.size() || proofs.empty() || !proofs.front().cut)
    return false;

  const CutProtocolProof &anchor = proofs.front();
  std::map<MessageKey, unsigned> sendNodes;
  std::map<MessageKey, ProvenanceBinding> sendBindings;
  std::set<MessageKey> receiveMessages;
  for (const CutProtocolProof &proof : proofs) {
    if (!proof.cut || !(proof.protocol == protocol) ||
        proof.cut->bytes != anchor.cut->bytes ||
        proof.cut->producerBuffer.getType() !=
            anchor.cut->producerBuffer.getType() ||
        proof.publisher.outputIndex != anchor.publisher.outputIndex ||
        proof.publisher.bufferType != anchor.publisher.bufferType ||
        compareStaticTiles(proof.publisher.globalTile,
                           anchor.publisher.globalTile) !=
            StaticTileRelation::Equivalent)
      return false;
    if (proof.publisherPieces.empty() ||
        proof.sendNodes.size() != proof.sendBindings.size())
      return false;
    int64_t publisherCursor = 0;
    for (const ProvenanceBinding &piece : proof.publisherPieces) {
      if (piece.node == 0 || piece.bytes.begin != publisherCursor ||
          piece.bytes.end <= piece.bytes.begin ||
          piece.bytes.end > proof.cut->bytes)
        return false;
      publisherCursor = piece.bytes.end;
    }
    if (publisherCursor != proof.cut->bytes ||
        (proof.publisherNode != 0 &&
         (proof.publisherPieces.size() != 1 ||
          proof.publisherPieces.front().node != proof.publisherNode)))
      return false;
    for (const auto &entry : proof.sendNodes) {
      auto binding = proof.sendBindings.find(entry.first);
      if (binding == proof.sendBindings.end() ||
          binding->second.node != entry.second ||
          !sendNodes.insert(entry).second ||
          !sendBindings.insert(*binding).second)
        return false;
    }
    receiveMessages.insert(proof.receiveMessages.begin(),
                           proof.receiveMessages.end());
  }

  llvm::ArrayRef<ProvenanceNode> nodes = arena.getNodes();
  llvm::SmallVector<unsigned char, 64> marks(nodes.size(), 0);
  llvm::SmallVector<std::optional<ResolvedProvenance>, 64> cache(nodes.size());
  std::map<int64_t, uint64_t> allRanks;
  for (size_t rank = 0; rank < modules.size(); ++rank)
    allRanks.emplace(static_cast<int64_t>(rank), 1);

  std::set<InstrElementwiseKind> combiners;
  std::set<MessageKey> usedReceives;
  auto resolveAndAccumulate =
      [&](unsigned node, const std::map<int64_t, uint64_t> &expected) -> bool {
    if (!resolveProvenance(node, nodes, sendNodes, marks, cache))
      return false;
    const ResolvedProvenance &resolved = *cache[node];
    if (resolved.originMultiplicity != expected)
      return false;
    combiners.insert(resolved.combiners.begin(), resolved.combiners.end());
    usedReceives.insert(resolved.receives.begin(), resolved.receives.end());
    return true;
  };
  auto validateSendBinding = [&](const MessageKey &message,
                                 unsigned node) -> bool {
    auto binding = sendBindings.find(message);
    std::optional<StaticByteRange> payload =
        getMessagePayloadRange(message, anchor.cut->bytes);
    return binding != sendBindings.end() && payload &&
           binding->second.node == node && binding->second.bytes == *payload &&
           node < nodes.size() && nodes[node].bytes == binding->second.bytes;
  };

  if (protocol.kind == ReductionProtocolKind::AllReduceTree) {
    std::optional<TreeProtocolInventory> inventory =
        collectTreeProtocolInventory(modules, protocol);
    if (!inventory || receiveMessages != inventory->messages ||
        sendNodes.size() != inventory->messages.size() ||
        llvm::any_of(inventory->messages, [&](const MessageKey &message) {
          return message.bytes != anchor.cut->bytes ||
                 !sendNodes.count(message);
        }))
      return false;

    llvm::SmallVector<std::map<int64_t, uint64_t>, 16> subtrees(modules.size());
    std::function<void(int64_t)> buildSubtree = [&](int64_t rank) {
      std::map<int64_t, uint64_t> &subtree =
          subtrees[static_cast<size_t>(rank)];
      subtree.emplace(rank, 1);
      for (int64_t child : inventory->children[static_cast<size_t>(rank)]) {
        buildSubtree(child);
        const std::map<int64_t, uint64_t> &childSubtree =
            subtrees[static_cast<size_t>(child)];
        subtree.insert(childSubtree.begin(), childSubtree.end());
      }
    };
    buildSubtree(inventory->root);

    for (const auto &entry : sendNodes) {
      const std::map<int64_t, uint64_t> &expected =
          entry.first.phase == DTEProtocolPhase::AllReduceTreeReduce
              ? subtrees[static_cast<size_t>(entry.first.sourceRank)]
              : allRanks;
      if (!validateSendBinding(entry.first, entry.second) ||
          !resolveAndAccumulate(entry.second, expected))
        return false;
    }
    for (const CutProtocolProof &proof : proofs)
      for (const ProvenanceBinding &piece : proof.publisherPieces)
        if (piece.node >= nodes.size() ||
            nodes[piece.node].bytes != piece.bytes ||
            !resolveAndAccumulate(piece.node, allRanks))
          return false;
    return combiners.size() == 1 && usedReceives == inventory->messages;
  }

  if (protocol.kind != ReductionProtocolKind::AllReduceRing)
    return false;
  std::optional<RingProtocolInventory> inventory =
      collectRingProtocolInventory(modules, protocol);
  const int64_t rankCount = static_cast<int64_t>(modules.size());
  if (!inventory || inventory->chunkBytes <= 0 ||
      inventory->chunkBytes > std::numeric_limits<int64_t>::max() / rankCount ||
      inventory->chunkBytes * rankCount != anchor.cut->bytes ||
      receiveMessages != inventory->messages ||
      sendNodes.size() != inventory->messages.size())
    return false;

  for (const auto &entry : sendNodes) {
    if (!inventory->messages.count(entry.first) ||
        !validateSendBinding(entry.first, entry.second))
      return false;
    std::map<int64_t, uint64_t> expected;
    if (entry.first.round < rankCount - 1) {
      int64_t origin = entry.first.sourceRank;
      for (int64_t step = 0; step <= entry.first.round; ++step) {
        if (origin < 0 || origin >= rankCount ||
            !expected.emplace(origin, 1).second)
          return false;
        origin = inventory->predecessor[static_cast<size_t>(origin)];
      }
    } else {
      expected = allRanks;
    }
    if (!resolveAndAccumulate(entry.second, expected))
      return false;
  }

  for (const CutProtocolProof &proof : proofs) {
    if (proof.publisherPieces.size() != static_cast<size_t>(rankCount))
      return false;
    for (auto [slice, piece] : llvm::enumerate(proof.publisherPieces)) {
      if (slice > static_cast<size_t>(std::numeric_limits<int64_t>::max() /
                                      inventory->chunkBytes))
        return false;
      int64_t begin = static_cast<int64_t>(slice) * inventory->chunkBytes;
      if (piece.bytes !=
              StaticByteRange{begin, begin + inventory->chunkBytes} ||
          piece.node >= nodes.size() ||
          nodes[piece.node].bytes != piece.bytes ||
          !resolveAndAccumulate(piece.node, allRanks))
        return false;
    }
  }
  return combiners.size() == 1 && usedReceives == inventory->messages;
}

static void eraseIfDead(mlir::Operation *operation) {
  if (operation &&
      llvm::all_of(operation->getResults(),
                   [](mlir::Value result) { return result.use_empty(); }))
    operation->erase();
}

static void applyCut(CutProtocolProof &proof) {
  PartialSpillCut &cut = *proof.cut;
  for (mlir::memref::DeallocOp deallocation : cut.consumerDeallocations)
    if (deallocation->getBlock())
      deallocation.erase();

  for (mlir::OpOperand *use : proof.consumerUses)
    use->set(cut.producerBuffer);

  cut.load.erase();
  cut.store.erase();
  for (mlir::memref::DeallocOp deallocation : cut.spillDeallocations)
    if (deallocation->getBlock())
      deallocation.erase();

  llvm::DenseSet<mlir::Operation *> spillCastOperations;
  llvm::SmallVector<mlir::memref::CastOp, 4> deadCasts;
  for (mlir::memref::CastOp cast : cut.spillCasts) {
    spillCastOperations.insert(cast.getOperation());
    if (cast.getResult().use_empty())
      deadCasts.push_back(cast);
  }
  while (!deadCasts.empty()) {
    mlir::memref::CastOp cast = deadCasts.pop_back_val();
    if (!cast.getResult().use_empty())
      continue;
    auto sourceCast = cast.getSource().getDefiningOp<mlir::memref::CastOp>();
    cast.erase();
    if (sourceCast && spillCastOperations.contains(sourceCast.getOperation()) &&
        sourceCast.getResult().use_empty())
      deadCasts.push_back(sourceCast);
  }
  eraseIfDead(cut.spillRoot.getOperation());
  eraseIfDead(getAliasRoot(cut.consumerBuffer).getDefiningOp());
}

} // namespace

unsigned materializeNoCPartialReductions(
    llvm::MutableArrayRef<mlir::ModuleOp> modules,
    const frontend::FrontendProgramVerificationResult &program) {
  wafer::support::ScopedCompileTimingSpan timing(
      "transformation", "materializeNoCPartialReductions", "total");
  auto phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "materializeNoCPartialReductions",
      "collectPartialSpillCuts");
  if (modules.size() < 2 ||
      modules.size() != static_cast<size_t>(program.logicalRankCount) ||
      program.distributedOutputs.empty())
    return 0;

  llvm::SmallVector<llvm::SmallVector<PartialSpillCut, 4>, 16> cutsByRank;
  cutsByRank.reserve(modules.size());
  std::set<ReductionProtocolKey> commonProtocols;
  bool firstRank = true;
  for (auto [rank, module] : llvm::enumerate(modules)) {
    cutsByRank.push_back(
        collectPartialSpillCuts(module, static_cast<int64_t>(rank)));
    std::set<ReductionProtocolKey> rankProtocols;
    for (PartialSpillCut &cut : cutsByRank.back()) {
      llvm::SmallVector<ReductionProtocolKey, 4> protocols =
          collectProtocolsAfter(cut);
      rankProtocols.insert(protocols.begin(), protocols.end());
    }
    if (firstRank) {
      commonProtocols = std::move(rankProtocols);
      firstRank = false;
    } else {
      std::set<ReductionProtocolKey> intersection;
      std::set_intersection(commonProtocols.begin(), commonProtocols.end(),
                            rankProtocols.begin(), rankProtocols.end(),
                            std::inserter(intersection, intersection.begin()));
      commonProtocols = std::move(intersection);
    }
  }
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "materializeNoCPartialReductions",
      "prove-and-verify-protocol");
  for (const ReductionProtocolKey &protocol : commonProtocols) {
    ProvenanceArena arena;
    llvm::SmallVector<CutProtocolProof, 16> proofs;
    bool ambiguous = false;
    for (auto &rankCuts : cutsByRank) {
      std::optional<CutProtocolProof> rankProof;
      for (PartialSpillCut &cut : rankCuts) {
        std::optional<CutProtocolProof> proof =
            proveCutProtocol(cut, protocol, program, arena);
        if (!proof)
          continue;
        if (rankProof) {
          ambiguous = true;
          break;
        }
        rankProof = std::move(*proof);
      }
      if (ambiguous || !rankProof) {
        ambiguous = true;
        break;
      }
      proofs.push_back(std::move(*rankProof));
    }
    bool complete =
        !ambiguous && verifyCompleteProof(modules, proofs, protocol, arena);
    if (!complete)
      continue;

    // All proof and complete-domain checks precede the first mutation.  The
    // caller can therefore discard later-gate failures without exposing a
    // partially rewritten rank tuple.
    phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
        "transformation-phase", "materializeNoCPartialReductions", "applyCut");
    for (CutProtocolProof &proof : proofs)
      applyCut(proof);
    return static_cast<unsigned>(proofs.size());
  }
  return 0;
}

} // namespace wafer::compiler::detail
