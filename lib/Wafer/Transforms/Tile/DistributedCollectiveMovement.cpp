//===- DistributedCollectiveMovement.cpp - Actual Tile collectives -------===//

#include "DistributedCollectiveMovement.h"

#include "Wafer/Planning/PhysicalDataflow/CollectiveAlgorithms.h"
#include "Wafer/IR/Topology/TargetTopology.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

// MLIR FailureOr exposes LogicalResult conversion instead of optional's
// bool/has_value API. Every dereference below is dominated by an explicit
// presence check; clang-tidy cannot follow the MLIR conversion checks.
// NOLINTBEGIN(bugprone-unchecked-optional-access)

namespace wafer::compiler::detail {

mlir::FailureOr<llvm::SmallVector<uint64_t, 16>>
buildMinimumHopTileRing(const TargetTopology &topology,
                        llvm::ArrayRef<uint64_t> participants) {
  // This adapter is the target-owned boundary. The generic ring builder does
  // not impose TileId's signed representation or a card identity; TX81's
  // topology oracle does.
  if (llvm::any_of(participants, [](uint64_t participant) {
        return participant >
               static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
      }))
    return mlir::failure();
  return buildMinimumHopRing(
      participants, /*maximumParticipants=*/16,
      [&](uint64_t lhs, uint64_t rhs) {
        return topology.getOnCardShortestHopDistance(
            CardId(0), TileId(static_cast<int64_t>(lhs)),
            TileId(static_cast<int64_t>(rhs)));
      });
}

namespace {

struct MessageKey {
  int64_t source = 0;
  int64_t destination = 0;
  int64_t communication = 0;
  int64_t round = 0;
  int64_t payloadSlice = 0;

  friend bool operator<(const MessageKey &lhs, const MessageKey &rhs) {
    return std::tie(lhs.source, lhs.destination, lhs.communication, lhs.round,
                    lhs.payloadSlice) < std::tie(rhs.source, rhs.destination,
                                                 rhs.communication, rhs.round,
                                                 rhs.payloadSlice);
  }
};

struct PeerPair {
  CommPeerSendOp send;
  CommPeerRecvOp receive;
  int64_t source = 0;
  int64_t destination = 0;
};

struct CoordinateIndex {
  unsigned row = 0;
  unsigned column = 0;
};

struct CompletePersonalizedComponent {
  int64_t communication = 0;
  llvm::SmallVector<int64_t, 16> participants;
  llvm::SmallVector<int64_t, 4> rows;
  llvm::SmallVector<int64_t, 4> columns;
  std::map<int64_t, CoordinateIndex> coordinates;
  std::map<std::pair<int64_t, int64_t>, PeerPair> pairs;
  mlir::MemRefType pieceType;
  uint64_t pieceBytes = 0;
};

static DistributedCollectiveMovementResult
fail(DistributedCollectiveMovementFailureKind kind, llvm::StringRef detail) {
  DistributedCollectiveMovementResult result;
  result.failure = kind;
  result.detail = detail.str();
  return result;
}

static std::optional<int64_t> getTileId(mlir::Operation *operation) {
  TileModuleOp tile = operation->getParentOfType<TileModuleOp>();
  if (!tile || !tile.getTileIdAttr())
    return std::nullopt;
  return tile.getTileIdAttr().getInt();
}

static MessageKey getKey(CommPeerSendOp operation, int64_t tile) {
  DTEMessageAttr message = operation.getMessageAttr();
  return MessageKey{tile, operation.getPeerAttr().getInt(),
                    message.getCommunicationId(), message.getRound(),
                    message.getPayloadSlice()};
}

static MessageKey getKey(CommPeerRecvOp operation, int64_t tile) {
  DTEMessageAttr message = operation.getMessageAttr();
  return MessageKey{operation.getPeerAttr().getInt(), tile,
                    message.getCommunicationId(), message.getRound(),
                    message.getPayloadSlice()};
}

static bool hasSupportedPieceType(mlir::MemRefType type, uint64_t bytes) {
  if (!type || type.getRank() == 0 || !type.hasStaticShape() ||
      !type.getLayout().isIdentity())
    return false;
  MemoryAttr memory = getWaferMemoryAttr(type);
  if (!memory || memory.getSpace() != MemorySpace::SPM ||
      (memory.getLayout() != MemLayout::Tensor &&
       memory.getLayout() != MemLayout::NTensor))
    return false;
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  return info && info->physicalBytes > 0 &&
         static_cast<uint64_t>(info->physicalBytes) == bytes;
}

static bool logicalTypesMatch(mlir::Type lhsType, mlir::Type rhsType,
                              uint64_t bytes) {
  auto lhs = mlir::dyn_cast<mlir::MemRefType>(lhsType);
  auto rhs = mlir::dyn_cast<mlir::MemRefType>(rhsType);
  if (!lhs || !rhs || lhs.getShape() != rhs.getShape() ||
      lhs.getElementType() != rhs.getElementType())
    return false;
  MemoryAttr lhsMemory = getWaferMemoryAttr(lhs);
  MemoryAttr rhsMemory = getWaferMemoryAttr(rhs);
  std::optional<WaferPhysicalTensorInfo> lhsInfo =
      computeWaferPhysicalTensorInfo(lhs);
  std::optional<WaferPhysicalTensorInfo> rhsInfo =
      computeWaferPhysicalTensorInfo(rhs);
  return lhsMemory && rhsMemory && lhsMemory == rhsMemory && lhsInfo &&
         rhsInfo && lhsInfo->physicalBytes > 0 && rhsInfo->physicalBytes > 0 &&
         static_cast<uint64_t>(lhsInfo->physicalBytes) == bytes &&
         static_cast<uint64_t>(rhsInfo->physicalBytes) == bytes;
}

static std::optional<mlir::MemRefType>
getAggregateType(mlir::MemRefType pieceType, unsigned count,
                 uint64_t pieceBytes) {
  if (!hasSupportedPieceType(pieceType, pieceBytes) || count < 2)
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> shape(pieceType.getShape());
  if (llvm::MulOverflow(shape.front(), static_cast<int64_t>(count),
                        shape.front()))
    return std::nullopt;
  auto aggregate =
      mlir::MemRefType::get(shape, pieceType.getElementType(),
                            pieceType.getLayout(), pieceType.getMemorySpace());
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(aggregate);
  if (pieceBytes > std::numeric_limits<uint64_t>::max() / count)
    return std::nullopt;
  return info && info->physicalBytes > 0 &&
                 static_cast<uint64_t>(info->physicalBytes) ==
                     pieceBytes * count
             ? std::optional<mlir::MemRefType>(aggregate)
             : std::nullopt;
}

static mlir::FailureOr<mlir::Value>
createSlot(mlir::OpBuilder &builder, mlir::Location location,
           mlir::Value aggregate, mlir::MemRefType pieceType, unsigned slot) {
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets(pieceType.getRank(),
                                                   builder.getIndexAttr(0));
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  for (int64_t size : pieceType.getShape()) {
    sizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }
  int64_t firstOffset = 0;
  if (llvm::MulOverflow(static_cast<int64_t>(slot), pieceType.getDimSize(0),
                        firstOffset))
    return mlir::failure();
  offsets.front() = builder.getIndexAttr(firstOffset);
  auto subview = builder.create<mlir::memref::SubViewOp>(
      location, aggregate, offsets, sizes, strides);
  auto type = mlir::dyn_cast<mlir::MemRefType>(subview.getType());
  if (!type || type.getShape() != pieceType.getShape() ||
      type.getElementType() != pieceType.getElementType() ||
      type.getMemorySpace() != pieceType.getMemorySpace()) {
    subview.erase();
    return mlir::failure();
  }
  return subview.getResult();
}

static bool
allOperationsShareOneRegion(int64_t tile,
                            const CompletePersonalizedComponent &component,
                            TileRegionOp &region) {
  for (const auto &[edge, pair] : component.pairs) {
    mlir::Operation *operation = nullptr;
    if (edge.first == tile)
      operation = pair.send;
    else if (edge.second == tile)
      operation = pair.receive;
    if (!operation)
      continue;
    TileRegionOp current = operation->getParentOfType<TileRegionOp>();
    if (!current || (region && current != region))
      return false;
    region = current;
  }
  return static_cast<bool>(region);
}

static mlir::FailureOr<llvm::SmallVector<PeerPair, 64>>
collectMatchedPeerPairs(mlir::ModuleOp module) {
  std::map<MessageKey, CommPeerSendOp> sends;
  std::map<MessageKey, CommPeerRecvOp> receives;
  bool malformed = false;
  module.walk([&](CommPeerSendOp operation) {
    std::optional<int64_t> tile = getTileId(operation);
    if (!tile || !sends.emplace(getKey(operation, *tile), operation).second)
      malformed = true;
  });
  module.walk([&](CommPeerRecvOp operation) {
    std::optional<int64_t> tile = getTileId(operation);
    if (!tile ||
        !receives.emplace(getKey(operation, *tile), operation).second)
      malformed = true;
  });
  if (malformed || sends.size() != receives.size())
    return mlir::failure();
  llvm::SmallVector<PeerPair, 64> pairs;
  for (auto &[key, send] : sends) {
    auto found = receives.find(key);
    if (found == receives.end())
      return mlir::failure();
    pairs.push_back(PeerPair{send, found->second, key.source, key.destination});
  }
  return pairs;
}

static llvm::SmallVector<CompletePersonalizedComponent, 2>
findCompletePersonalizedComponents(mlir::ModuleOp module,
                                   const TargetTopology &topology,
                                   std::string &failureReason) {
  mlir::FailureOr<llvm::SmallVector<PeerPair, 64>> matched =
      collectMatchedPeerPairs(module);
  if (mlir::failed(matched)) {
    failureReason = "current Tile peer messages have duplicate or missing "
                    "physical endpoints";
    return {};
  }

  std::map<int64_t, llvm::SmallVector<PeerPair, 32>> byCommunication;
  for (PeerPair &pair : *matched) {
    byCommunication[pair.send.getMessageAttr().getCommunicationId()].push_back(
        pair);
  }

  llvm::SmallVector<CompletePersonalizedComponent, 2> result;
  constexpr CardId cardId(0);
  for (auto &[communication, pairs] : byCommunication) {
    std::set<int64_t> participantSet;
    std::map<std::pair<int64_t, int64_t>, PeerPair> pairMap;
    bool duplicate = false;
    for (const PeerPair &pair : pairs) {
      participantSet.insert(pair.source);
      participantSet.insert(pair.destination);
      duplicate |=
          !pairMap.emplace(std::make_pair(pair.source, pair.destination), pair)
               .second;
    }
    const size_t participantCount = participantSet.size();
    if (duplicate || participantCount < 4 ||
        pairMap.size() != participantCount * (participantCount - 1))
      continue;
    bool complete = true;
    for (int64_t source : participantSet)
      for (int64_t destination : participantSet)
        if (source != destination &&
            !pairMap.count(std::make_pair(source, destination)))
          complete = false;
    if (!complete)
      continue;

    PeerPair &first = pairMap.begin()->second;
    auto pieceType =
        mlir::dyn_cast<mlir::MemRefType>(first.receive.getBuffer().getType());
    int64_t signedPieceBytes = first.send.getBytesAttr().getInt();
    if (signedPieceBytes <= 0)
      continue;
    uint64_t pieceBytes = static_cast<uint64_t>(signedPieceBytes);
    if (!hasSupportedPieceType(pieceType, pieceBytes))
      continue;
    for (auto &[edge, pair] : pairMap)
      if (!logicalTypesMatch(pair.send.getBuffer().getType(), pieceType,
                             pieceBytes) ||
          !logicalTypesMatch(pair.receive.getBuffer().getType(), pieceType,
                             pieceBytes) ||
          pair.send.getBytesAttr().getInt() != signedPieceBytes ||
          pair.receive.getBytesAttr().getInt() != signedPieceBytes ||
          !pair.send.getToken().use_empty() ||
          !pair.receive.getToken().use_empty())
        complete = false;
    if (!complete)
      continue;

    llvm::SmallVector<int64_t, 4> rows;
    llvm::SmallVector<int64_t, 4> columns;
    std::map<int64_t, TileCoordinate> physicalCoordinates;
    for (int64_t participant : participantSet) {
      std::optional<TileCoordinate> coordinate =
          topology.getTileCoordinate(TileId(participant));
      if (!coordinate ||
          !topology.isTileAvailable(cardId, TileId(participant))) {
        complete = false;
        break;
      }
      const TileCoordinate coordinateValue = *coordinate;
      physicalCoordinates.emplace(participant, coordinateValue);
      if (!llvm::is_contained(rows, coordinateValue.y))
        rows.push_back(coordinateValue.y);
      if (!llvm::is_contained(columns, coordinateValue.x))
        columns.push_back(coordinateValue.x);
    }
    llvm::sort(rows);
    llvm::sort(columns);
    if (!complete || rows.size() < 2 || columns.size() < 2 ||
        rows.size() * columns.size() != participantCount)
      continue;
    for (int64_t row : rows)
      for (int64_t column : columns) {
        std::optional<TileId> tile =
            topology.getTileId(TileCoordinate{row, column});
        if (!tile || !participantSet.count(tile->getValue()))
          complete = false;
      }
    if (!complete)
      continue;
    if (!getAggregateType(pieceType, rows.size(), pieceBytes) ||
        !getAggregateType(pieceType, columns.size(), pieceBytes))
      continue;

    CompletePersonalizedComponent component;
    component.communication = communication;
    component.participants.assign(participantSet.begin(), participantSet.end());
    component.rows = std::move(rows);
    component.columns = std::move(columns);
    component.pairs = std::move(pairMap);
    component.pieceType = pieceType;
    component.pieceBytes = pieceBytes;
    for (const auto &[tile, coordinate] : physicalCoordinates) {
      auto row = llvm::find(component.rows, coordinate.y);
      auto column = llvm::find(component.columns, coordinate.x);
      component.coordinates.emplace(
          tile, CoordinateIndex{
                    static_cast<unsigned>(row - component.rows.begin()),
                    static_cast<unsigned>(column - component.columns.begin())});
    }
    bool oneRegionPerTile = true;
    for (int64_t tile : component.participants) {
      TileRegionOp region;
      oneRegionPerTile &= allOperationsShareOneRegion(tile, component, region);
    }
    if (oneRegionPerTile)
      result.push_back(std::move(component));
  }
  return result;
}

static mlir::Operation *
getInsertionAnchor(int64_t tile,
                   const CompletePersonalizedComponent &component) {
  mlir::Operation *anchor = nullptr;
  for (const auto &[edge, pair] : component.pairs) {
    mlir::Operation *operation = nullptr;
    if (edge.first == tile)
      operation = pair.send;
    else if (edge.second == tile)
      operation = pair.receive;
    if (!operation)
      continue;
    if (!anchor || operation->isBeforeInBlock(anchor))
      anchor = operation;
  }
  return anchor;
}

static int64_t getTileAt(const CompletePersonalizedComponent &component,
                         unsigned row, unsigned column) {
  for (const auto &[tile, coordinate] : component.coordinates)
    if (coordinate.row == row && coordinate.column == column)
      return tile;
  return -1;
}

static mlir::LogicalResult
materializeComponent(CompletePersonalizedComponent &component,
                     DistributedCollectiveMovementResult &statistics) {
  mlir::MLIRContext *context = component.pieceType.getContext();
  const unsigned rowCount = component.rows.size();
  const unsigned columnCount = component.columns.size();
  std::optional<mlir::MemRefType> rowAggregateType =
      getAggregateType(component.pieceType, rowCount, component.pieceBytes);
  std::optional<mlir::MemRefType> columnAggregateType =
      getAggregateType(component.pieceType, columnCount, component.pieceBytes);
  if (!rowAggregateType || !columnAggregateType)
    return mlir::failure();

  std::map<std::pair<int64_t, unsigned>, mlir::Value> phaseOnePacks;
  std::map<std::pair<int64_t, int64_t>, mlir::Value> phaseOneReceives;
  std::map<std::pair<int64_t, unsigned>, mlir::Value> phaseTwoPacks;
  std::map<std::pair<int64_t, unsigned>, mlir::Value> phaseTwoReceives;

  for (int64_t tile : component.participants) {
    mlir::Operation *anchor = getInsertionAnchor(tile, component);
    if (!anchor)
      return mlir::failure();
    mlir::OpBuilder builder(anchor);
    mlir::Location location = anchor->getLoc();
    const CoordinateIndex coordinate = component.coordinates.at(tile);

    for (unsigned sourceColumn = 0; sourceColumn < columnCount;
         ++sourceColumn) {
      if (sourceColumn == coordinate.column)
        continue;
      int64_t sourceTile = getTileAt(component, coordinate.row, sourceColumn);
      if (sourceTile < 0)
        return mlir::failure();
      mlir::Value aggregate =
          builder.create<mlir::memref::AllocOp>(location, *rowAggregateType);
      auto message = DTEMessageAttr::get(
          context, component.communication,
          static_cast<int64_t>(
              (coordinate.column + columnCount - sourceColumn) % columnCount -
              1),
          static_cast<int64_t>(sourceColumn * columnCount + coordinate.column));
      builder.create<CommPeerRecvOp>(location, aggregate, sourceTile,
                                     component.pieceBytes * rowCount, message);
      phaseOneReceives.emplace(std::make_pair(tile, sourceTile), aggregate);
      ++statistics.createdPeerReceives;
    }

    for (unsigned destinationColumn = 0; destinationColumn < columnCount;
         ++destinationColumn) {
      mlir::Value aggregate =
          builder.create<mlir::memref::AllocOp>(location, *rowAggregateType);
      phaseOnePacks.emplace(std::make_pair(tile, destinationColumn), aggregate);
      for (unsigned destinationRow = 0; destinationRow < rowCount;
           ++destinationRow) {
        int64_t destinationTile =
            getTileAt(component, destinationRow, destinationColumn);
        if (destinationTile < 0)
          return mlir::failure();
        if (destinationTile == tile)
          continue;
        auto pair = component.pairs.find(std::make_pair(tile, destinationTile));
        if (pair == component.pairs.end())
          return mlir::failure();
        mlir::FailureOr<mlir::Value> slot = createSlot(
            builder, location, aggregate, component.pieceType, destinationRow);
        if (mlir::failed(slot))
          return mlir::failure();
        builder.create<mlir::memref::CopyOp>(
            location, pair->second.send.getBuffer(), *slot);
        ++statistics.packCopies;
      }
      if (destinationColumn == coordinate.column)
        continue;
      int64_t destinationTile =
          getTileAt(component, coordinate.row, destinationColumn);
      unsigned delta =
          (destinationColumn + columnCount - coordinate.column) % columnCount;
      auto message = DTEMessageAttr::get(
          context, component.communication, static_cast<int64_t>(delta - 1),
          static_cast<int64_t>(coordinate.column * columnCount +
                               destinationColumn));
      builder.create<CommPeerSendOp>(location, aggregate, destinationTile,
                                     component.pieceBytes * rowCount, message);
      ++statistics.createdPeerSends;
    }
  }

  for (int64_t tile : component.participants) {
    mlir::Operation *anchor = getInsertionAnchor(tile, component);
    mlir::OpBuilder builder(anchor);
    mlir::Location location = anchor->getLoc();
    const CoordinateIndex coordinate = component.coordinates.at(tile);

    for (unsigned sourceRow = 0; sourceRow < rowCount; ++sourceRow) {
      if (sourceRow == coordinate.row)
        continue;
      int64_t sourceTile = getTileAt(component, sourceRow, coordinate.column);
      mlir::Value aggregate =
          builder.create<mlir::memref::AllocOp>(location, *columnAggregateType);
      unsigned delta = (coordinate.row + rowCount - sourceRow) % rowCount;
      auto message = DTEMessageAttr::get(
          context, component.communication,
          static_cast<int64_t>(columnCount - 1 + delta - 1),
          static_cast<int64_t>(columnCount * columnCount +
                               sourceRow * rowCount + coordinate.row));
      builder.create<CommPeerRecvOp>(location, aggregate, sourceTile,
                                     component.pieceBytes * columnCount,
                                     message);
      phaseTwoReceives.emplace(std::make_pair(tile, sourceRow), aggregate);
      ++statistics.createdPeerReceives;
    }

    for (unsigned destinationRow = 0; destinationRow < rowCount;
         ++destinationRow) {
      mlir::Value aggregate =
          builder.create<mlir::memref::AllocOp>(location, *columnAggregateType);
      phaseTwoPacks.emplace(std::make_pair(tile, destinationRow), aggregate);
      for (unsigned sourceColumn = 0; sourceColumn < columnCount;
           ++sourceColumn) {
        int64_t sourceTile = getTileAt(component, coordinate.row, sourceColumn);
        int64_t destinationTile =
            getTileAt(component, destinationRow, coordinate.column);
        if (sourceTile < 0 || destinationTile < 0)
          return mlir::failure();
        if (sourceTile == destinationTile)
          continue;
        mlir::Value phaseOne;
        if (sourceColumn == coordinate.column) {
          phaseOne = phaseOnePacks.at(std::make_pair(tile, coordinate.column));
        } else {
          auto found = phaseOneReceives.find(std::make_pair(tile, sourceTile));
          if (found == phaseOneReceives.end())
            return mlir::failure();
          phaseOne = found->second;
        }
        mlir::FailureOr<mlir::Value> sourceSlot = createSlot(
            builder, location, phaseOne, component.pieceType, destinationRow);
        mlir::FailureOr<mlir::Value> destinationSlot = createSlot(
            builder, location, aggregate, component.pieceType, sourceColumn);
        if (mlir::failed(sourceSlot) || mlir::failed(destinationSlot))
          return mlir::failure();
        builder.create<mlir::memref::CopyOp>(location, *sourceSlot,
                                             *destinationSlot);
        ++statistics.packCopies;
      }
      if (destinationRow == coordinate.row)
        continue;
      int64_t destinationTile =
          getTileAt(component, destinationRow, coordinate.column);
      unsigned delta = (destinationRow + rowCount - coordinate.row) % rowCount;
      auto message = DTEMessageAttr::get(
          context, component.communication,
          static_cast<int64_t>(columnCount - 1 + delta - 1),
          static_cast<int64_t>(columnCount * columnCount +
                               coordinate.row * rowCount + destinationRow));
      builder.create<CommPeerSendOp>(location, aggregate, destinationTile,
                                     component.pieceBytes * columnCount,
                                     message);
      ++statistics.createdPeerSends;
    }
  }

  mlir::IRRewriter rewriter(context);
  for (auto &[edge, pair] : component.pairs) {
    const CoordinateIndex sourceCoordinate =
        component.coordinates.at(edge.first);
    const CoordinateIndex destinationCoordinate =
        component.coordinates.at(edge.second);
    mlir::Value finalAggregate;
    if (sourceCoordinate.row == destinationCoordinate.row) {
      finalAggregate = phaseTwoPacks.at(
          std::make_pair(edge.second, destinationCoordinate.row));
    } else {
      finalAggregate = phaseTwoReceives.at(
          std::make_pair(edge.second, sourceCoordinate.row));
    }
    rewriter.setInsertionPoint(pair.receive);
    mlir::FailureOr<mlir::Value> slot =
        createSlot(rewriter, pair.receive.getLoc(), finalAggregate,
                   component.pieceType, sourceCoordinate.column);
    if (mlir::failed(slot))
      return mlir::failure();
    mlir::Value oldReceiveBuffer = pair.receive.getBuffer();
    mlir::Operation *oldReceiveOwner = oldReceiveBuffer.getDefiningOp();
    rewriter.eraseOp(pair.send);
    rewriter.eraseOp(pair.receive);
    oldReceiveBuffer.replaceAllUsesWith(*slot);
    if (auto allocation =
            mlir::dyn_cast_or_null<mlir::memref::AllocOp>(oldReceiveOwner);
        allocation && allocation->use_empty())
      rewriter.eraseOp(allocation);
    ++statistics.removedPeerSends;
    ++statistics.removedPeerReceives;
  }
  ++statistics.components;
  return mlir::success();
}

struct AssociativeMergeTree {
  ComputeElementwiseOp root;
  ComputeElementwiseKind kind = ComputeElementwiseKind::Add;
  llvm::SmallVector<mlir::Operation *, 16> operations;
  llvm::SmallVector<mlir::Value, 16> leaves;
  mlir::Value localLeaf;
};

using RingContributionQuery = std::function<mlir::Value(uint64_t, unsigned)>;
using RingPieceTypeQuery = std::function<mlir::MemRefType(unsigned)>;
using RingPieceBytesQuery = std::function<uint64_t(unsigned)>;
using RingInsertionAnchorQuery = std::function<mlir::Operation *(uint64_t)>;

static mlir::FailureOr<std::map<uint64_t, mlir::Value>>
materializeRingReduction(llvm::ArrayRef<uint64_t> ring, int64_t communication,
                         ComputeElementwiseKind kind,
                         const RingContributionQuery &getContribution,
                         const RingPieceTypeQuery &getPieceType,
                         const RingPieceBytesQuery &getPieceBytes,
                         const RingInsertionAnchorQuery &getAnchor,
                         DistributedCollectiveMovementResult &statistics) {
  if (ring.size() < 2)
    return mlir::failure();
  std::map<uint64_t, unsigned> positions;
  for (auto [position, tile] : llvm::enumerate(ring))
    positions.emplace(tile, static_cast<unsigned>(position));
  std::map<uint64_t, mlir::Value> finalPartials;
  for (auto [positionValue, tile] : llvm::enumerate(ring)) {
    const unsigned position = static_cast<unsigned>(positionValue);
    mlir::Operation *anchor = getAnchor(tile);
    if (!anchor)
      return mlir::failure();
    mlir::OpBuilder builder(anchor);
    const uint64_t next = ring[(position + 1) % ring.size()];
    const uint64_t previous = ring[(position + ring.size() - 1) % ring.size()];
    mlir::Value accumulated;
    for (unsigned round = 0; round + 1 < ring.size(); ++round) {
      const unsigned sendChunk =
          (position + ring.size() - round - 1) % ring.size();
      const unsigned receiveChunk =
          (position + ring.size() - round - 2) % ring.size();
      mlir::MemRefType receiveType = getPieceType(receiveChunk);
      const uint64_t receiveBytes = getPieceBytes(receiveChunk);
      if (!receiveType || receiveBytes == 0)
        return mlir::failure();
      mlir::Value sendBuffer =
          round == 0 ? getContribution(tile, sendChunk) : accumulated;
      mlir::Value local = getContribution(tile, receiveChunk);
      if (!sendBuffer || !local)
        return mlir::failure();
      auto receiveBuffer =
          builder.create<mlir::memref::AllocOp>(anchor->getLoc(), receiveType);
      auto receiveMessage = DTEMessageAttr::get(
          receiveType.getContext(), communication, round, receiveChunk);
      auto sendMessage = DTEMessageAttr::get(receiveType.getContext(),
                                             communication, round, sendChunk);
      builder.create<CommPeerRecvOp>(anchor->getLoc(), receiveBuffer, previous,
                                     receiveBytes, receiveMessage);
      builder.create<CommPeerSendOp>(anchor->getLoc(), sendBuffer, next,
                                     getPieceBytes(sendChunk), sendMessage);
      auto combined = builder.create<ComputeElementwiseOp>(
          anchor->getLoc(), receiveType,
          ComputeElementwiseKindAttr::get(receiveType.getContext(), kind),
          mlir::ValueRange{local, receiveBuffer}, mlir::ArrayAttr{});
      accumulated = combined.getResult();
      ++statistics.createdPeerSends;
      ++statistics.createdPeerReceives;
      ++statistics.reductionCombines;
    }
    finalPartials.emplace(tile, accumulated);
  }
  return finalPartials;
}

static bool isAssociativeReductionKind(ComputeElementwiseKind kind) {
  return kind == ComputeElementwiseKind::Add ||
         kind == ComputeElementwiseKind::Max ||
         kind == ComputeElementwiseKind::Min;
}

static bool hasIdentityIndexing(ComputeElementwiseOp operation) {
  mlir::ArrayAttr maps = operation.getIndexingMapsAttr();
  if (!maps)
    return true;
  if (maps.size() != operation.getInputs().size() + 1)
    return false;
  return llvm::all_of(maps, [](mlir::Attribute attribute) {
    auto map = mlir::dyn_cast<mlir::AffineMapAttr>(attribute);
    return map && map.getValue().isIdentity();
  });
}

static bool collectAssociativeMergeTree(
    mlir::Value value, TileRegionOp region, ComputeElementwiseKind kind,
    mlir::MemRefType pieceType, ComputeElementwiseOp root,
    llvm::SmallVectorImpl<mlir::Operation *> &operations,
    llvm::SmallVectorImpl<mlir::Value> &leaves,
    std::set<mlir::Operation *> &visited) {
  auto operation = value.getDefiningOp<ComputeElementwiseOp>();
  if (!operation || operation->getParentOfType<TileRegionOp>() != region ||
      operation.getKind() != kind || operation.getInputs().size() != 2 ||
      operation.getResult().getType() != pieceType ||
      !hasIdentityIndexing(operation)) {
    if (llvm::is_contained(leaves, value))
      return false;
    leaves.push_back(value);
    return true;
  }
  if (!visited.insert(operation).second)
    return false;
  if (operation != root) {
    if (!operation.getResult().hasOneUse())
      return false;
    mlir::Operation *user = *operation.getResult().getUsers().begin();
    if (!mlir::isa<ComputeElementwiseOp>(user))
      return false;
  }
  operations.push_back(operation);
  for (mlir::Value input : operation.getInputs())
    if (!collectAssociativeMergeTree(input, region, kind, pieceType, root,
                                     operations, leaves, visited))
      return false;
  return true;
}

static std::optional<AssociativeMergeTree>
findAssociativeMergeTree(int64_t destination,
                         CompletePersonalizedComponent &component) {
  llvm::SmallVector<mlir::Value, 16> remoteLeaves;
  TileRegionOp region;
  for (int64_t source : component.participants) {
    if (source == destination)
      continue;
    PeerPair &pair = component.pairs.at({source, destination});
    remoteLeaves.push_back(pair.receive.getBuffer());
    TileRegionOp current = pair.receive->getParentOfType<TileRegionOp>();
    if (!current || (region && region != current))
      return std::nullopt;
    region = current;
  }
  if (!region)
    return std::nullopt;

  std::optional<AssociativeMergeTree> selected;
  region.walk([&](ComputeElementwiseOp root) {
    if (selected || !isAssociativeReductionKind(root.getKind()) ||
        root.getInputs().size() != 2 ||
        root.getResult().getType() != component.pieceType ||
        !hasIdentityIndexing(root))
      return;
    AssociativeMergeTree candidate;
    candidate.root = root;
    candidate.kind = root.getKind();
    std::set<mlir::Operation *> visited;
    if (!collectAssociativeMergeTree(
            root.getResult(), region, root.getKind(), component.pieceType, root,
            candidate.operations, candidate.leaves, visited) ||
        candidate.leaves.size() != component.participants.size())
      return;
    for (mlir::Value remote : remoteLeaves)
      if (!llvm::is_contained(candidate.leaves, remote))
        return;
    llvm::SmallVector<mlir::Value, 2> local;
    for (mlir::Value leaf : candidate.leaves)
      if (!llvm::is_contained(remoteLeaves, leaf))
        local.push_back(leaf);
    if (local.size() != 1 ||
        !logicalTypesMatch(local.front().getType(), component.pieceType,
                           component.pieceBytes))
      return;
    candidate.localLeaf = local.front();
    selected = std::move(candidate);
  });
  return selected;
}

static mlir::LogicalResult materializeReduceScatterComponent(
    CompletePersonalizedComponent &component, const TargetTopology &topology,
    DistributedCollectiveMovementResult &statistics) {
  std::map<int64_t, AssociativeMergeTree> merges;
  std::optional<ComputeElementwiseKind> kind;
  for (int64_t destination : component.participants) {
    std::optional<AssociativeMergeTree> merge =
        findAssociativeMergeTree(destination, component);
    if (!merge || (kind && *kind != merge->kind))
      return mlir::failure();
    kind = merge->kind;
    merges.emplace(destination, std::move(*merge));
  }
  if (!kind)
    return mlir::failure();
  llvm::SmallVector<uint64_t, 16> participantIds;
  for (int64_t participant : component.participants) {
    if (participant < 0)
      return mlir::failure();
    participantIds.push_back(static_cast<uint64_t>(participant));
  }
  mlir::FailureOr<llvm::SmallVector<uint64_t, 16>> ring =
      buildMinimumHopTileRing(topology, participantIds);
  if (mlir::failed(ring))
    return mlir::failure();
  std::map<uint64_t, mlir::Operation *> anchors;
  for (uint64_t tile : *ring) {
    mlir::Operation *anchor =
        getInsertionAnchor(static_cast<int64_t>(tile), component);
    if (!anchor)
      return mlir::failure();
    anchors.emplace(tile, anchor);
  }
  auto finalPartials = materializeRingReduction(
      *ring, component.communication, *kind,
      [&](uint64_t source, unsigned destinationPosition) -> mlir::Value {
        const uint64_t destination = (*ring)[destinationPosition];
        if (source == destination)
          return merges.at(static_cast<int64_t>(destination)).localLeaf;
        return component.pairs
            .at({static_cast<int64_t>(source),
                 static_cast<int64_t>(destination)})
            .send.getBuffer();
      },
      [&](unsigned) { return component.pieceType; },
      [&](unsigned) { return component.pieceBytes; },
      [&](uint64_t tile) { return anchors.at(tile); }, statistics);
  if (mlir::failed(finalPartials))
    return mlir::failure();

  mlir::IRRewriter rewriter(component.pieceType.getContext());
  for (auto &[destination, merge] : merges)
    merge.root.getResult().replaceAllUsesWith(
        finalPartials->at(static_cast<uint64_t>(destination)));
  for (auto &[destination, merge] : merges) {
    (void)destination;
    for (mlir::Operation *operation : merge.operations) {
      if (!operation->use_empty())
        return mlir::failure();
      rewriter.eraseOp(operation);
    }
  }
  for (auto &[edge, pair] : component.pairs) {
    (void)edge;
    mlir::Value receiveBuffer = pair.receive.getBuffer();
    mlir::Operation *receiveOwner = receiveBuffer.getDefiningOp();
    rewriter.eraseOp(pair.send);
    rewriter.eraseOp(pair.receive);
    if (auto allocation =
            mlir::dyn_cast_or_null<mlir::memref::AllocOp>(receiveOwner);
        allocation && allocation->use_empty())
      rewriter.eraseOp(allocation);
    ++statistics.removedPeerSends;
    ++statistics.removedPeerReceives;
  }
  ++statistics.reduceScatterComponents;
  return mlir::success();
}

struct LeadingChunk {
  int64_t offset = 0;
  int64_t size = 0;
  mlir::MemRefType type;
  uint64_t bytes = 0;
};

struct RingAllReduceComponent {
  int64_t communication = 0;
  uint64_t rootTile = 0;
  llvm::SmallVector<uint64_t, 16> participants;
  AssociativeMergeTree merge;
  std::map<uint64_t, mlir::Value> contributions;
  std::map<uint64_t, mlir::Value> publishedResults;
  llvm::SmallVector<PeerPair, 32> oldPairs;
  mlir::MemRefType fullType;
  uint64_t fullBytes = 0;
};

static std::optional<llvm::SmallVector<LeadingChunk, 16>>
buildLeadingChunks(mlir::MemRefType fullType, unsigned count) {
  if (!fullType || !fullType.hasStaticShape() ||
      !fullType.getLayout().isIdentity() || fullType.getRank() < 1 ||
      count < 2 || fullType.getDimSize(0) < static_cast<int64_t>(count))
    return std::nullopt;
  MemoryAttr memory = getWaferMemoryAttr(fullType);
  if (!memory || memory.getSpace() != MemorySpace::SPM ||
      (memory.getLayout() != MemLayout::Tensor &&
       memory.getLayout() != MemLayout::NTensor))
    return std::nullopt;
  const int64_t leading = fullType.getDimSize(0);
  const int64_t quotient = leading / static_cast<int64_t>(count);
  const int64_t remainder = leading % static_cast<int64_t>(count);
  int64_t offset = 0;
  llvm::SmallVector<LeadingChunk, 16> chunks;
  for (unsigned position = 0; position < count; ++position) {
    const int64_t size = quotient + (position < remainder ? 1 : 0);
    llvm::SmallVector<int64_t, 4> shape(fullType.getShape());
    shape.front() = size;
    auto type =
        mlir::MemRefType::get(shape, fullType.getElementType(),
                              fullType.getLayout(), fullType.getMemorySpace());
    std::optional<WaferPhysicalTensorInfo> info =
        computeWaferPhysicalTensorInfo(type);
    if (!info || info->physicalBytes <= 0)
      return std::nullopt;
    chunks.push_back(LeadingChunk{offset, size, type,
                                  static_cast<uint64_t>(info->physicalBytes)});
    offset += size;
  }
  if (offset != leading)
    return std::nullopt;
  return chunks;
}

static mlir::FailureOr<mlir::Value>
createLeadingChunk(mlir::OpBuilder &builder, mlir::Location location,
                   mlir::Value fullBuffer, mlir::MemRefType fullType,
                   const LeadingChunk &chunk) {
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets(fullType.getRank(),
                                                   builder.getIndexAttr(0));
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  offsets.front() = builder.getIndexAttr(chunk.offset);
  for (auto [dimension, size] : llvm::enumerate(fullType.getShape())) {
    sizes.push_back(builder.getIndexAttr(dimension == 0 ? chunk.size : size));
    strides.push_back(builder.getIndexAttr(1));
  }
  auto view = builder.create<mlir::memref::SubViewOp>(location, fullBuffer,
                                                      offsets, sizes, strides);
  auto type = mlir::dyn_cast<mlir::MemRefType>(view.getType());
  if (!type || type.getShape() != chunk.type.getShape() ||
      !logicalTypesMatch(type, chunk.type, chunk.bytes)) {
    view.erase();
    return mlir::failure();
  }
  return view.getResult();
}

static std::optional<RingAllReduceComponent>
findRingAllReduceComponent(ComputeElementwiseOp root,
                           llvm::MutableArrayRef<PeerPair> pairs,
                           llvm::DenseSet<mlir::Operation *> &claimed) {
  auto fullType = mlir::dyn_cast<mlir::MemRefType>(root.getResult().getType());
  if (!fullType || !isAssociativeReductionKind(root.getKind()) ||
      !hasIdentityIndexing(root))
    return std::nullopt;
  std::optional<WaferPhysicalTensorInfo> fullInfo =
      computeWaferPhysicalTensorInfo(fullType);
  if (!fullInfo || fullInfo->physicalBytes <= 0 ||
      !hasSupportedPieceType(fullType,
                             static_cast<uint64_t>(fullInfo->physicalBytes)))
    return std::nullopt;
  TileRegionOp rootRegion = root->getParentOfType<TileRegionOp>();
  std::optional<int64_t> rootTileId = getTileId(root);
  if (!rootRegion || !rootTileId || *rootTileId < 0)
    return std::nullopt;

  AssociativeMergeTree merge;
  merge.root = root;
  merge.kind = root.getKind();
  std::set<mlir::Operation *> visited;
  if (!collectAssociativeMergeTree(root.getResult(), rootRegion, root.getKind(),
                                   fullType, root, merge.operations,
                                   merge.leaves, visited) ||
      merge.leaves.size() < 2 || merge.leaves.size() > 16)
    return std::nullopt;

  llvm::DenseMap<mlir::Value, unsigned> receivePair;
  for (auto [index, pair] : llvm::enumerate(pairs))
    receivePair.try_emplace(pair.receive.getBuffer(),
                            static_cast<unsigned>(index));
  llvm::SmallVector<unsigned, 16> faninIndices;
  std::map<uint64_t, mlir::Value> contributions;
  llvm::SmallVector<mlir::Value, 2> localLeaves;
  for (mlir::Value leaf : merge.leaves) {
    auto found = receivePair.find(leaf);
    if (found == receivePair.end()) {
      localLeaves.push_back(leaf);
      continue;
    }
    PeerPair &pair = pairs[found->second];
    if (pair.destination != *rootTileId || pair.source < 0 ||
        !logicalTypesMatch(pair.send.getBuffer().getType(), fullType,
                           static_cast<uint64_t>(fullInfo->physicalBytes)) ||
        !contributions
             .emplace(static_cast<uint64_t>(pair.source), pair.send.getBuffer())
             .second)
      return std::nullopt;
    faninIndices.push_back(found->second);
  }
  if (localLeaves.size() != 1 ||
      !contributions
           .emplace(static_cast<uint64_t>(*rootTileId), localLeaves.front())
           .second ||
      contributions.size() != merge.leaves.size())
    return std::nullopt;
  merge.localLeaf = localLeaves.front();

  std::map<uint64_t, mlir::Value> holders;
  holders.emplace(static_cast<uint64_t>(*rootTileId), root.getResult());
  llvm::DenseSet<unsigned> faninSet(faninIndices.begin(), faninIndices.end());
  llvm::DenseSet<unsigned> fanoutSet;
  bool progress = true;
  while (progress) {
    progress = false;
    for (auto [indexValue, pair] : llvm::enumerate(pairs)) {
      const unsigned index = static_cast<unsigned>(indexValue);
      if (faninSet.contains(index) || fanoutSet.contains(index) ||
          pair.source < 0 || pair.destination < 0)
        continue;
      auto holder = holders.find(static_cast<uint64_t>(pair.source));
      if (holder == holders.end() || pair.send.getBuffer() != holder->second ||
          holders.count(static_cast<uint64_t>(pair.destination)) ||
          !logicalTypesMatch(pair.receive.getBuffer().getType(), fullType,
                             static_cast<uint64_t>(fullInfo->physicalBytes)))
        continue;
      holders.emplace(static_cast<uint64_t>(pair.destination),
                      pair.receive.getBuffer());
      fanoutSet.insert(index);
      progress = true;
    }
  }
  if (holders.size() != contributions.size() ||
      fanoutSet.size() + 1 != contributions.size())
    return std::nullopt;
  for (const auto &[participant, contribution] : contributions) {
    (void)contribution;
    if (!holders.count(participant))
      return std::nullopt;
  }
  if (!buildLeadingChunks(fullType, contributions.size()))
    return std::nullopt;

  RingAllReduceComponent component;
  component.rootTile = static_cast<uint64_t>(*rootTileId);
  component.merge = std::move(merge);
  component.contributions = std::move(contributions);
  component.publishedResults = std::move(holders);
  component.fullType = fullType;
  component.fullBytes = static_cast<uint64_t>(fullInfo->physicalBytes);
  component.communication = std::numeric_limits<int64_t>::max();
  for (const auto &[participant, contribution] : component.contributions) {
    (void)contribution;
    component.participants.push_back(participant);
  }
  llvm::sort(component.participants);
  for (unsigned index : faninIndices) {
    if (claimed.contains(pairs[index].send) ||
        claimed.contains(pairs[index].receive))
      return std::nullopt;
    component.oldPairs.push_back(pairs[index]);
    component.communication =
        std::min(component.communication,
                 pairs[index].send.getMessageAttr().getCommunicationId());
  }
  for (unsigned index : fanoutSet) {
    if (claimed.contains(pairs[index].send) ||
        claimed.contains(pairs[index].receive))
      return std::nullopt;
    component.oldPairs.push_back(pairs[index]);
    component.communication =
        std::min(component.communication,
                 pairs[index].send.getMessageAttr().getCommunicationId());
  }
  if (component.communication == std::numeric_limits<int64_t>::max())
    return std::nullopt;
  for (PeerPair &pair : component.oldPairs) {
    claimed.insert(pair.send);
    claimed.insert(pair.receive);
  }
  return component;
}

static mlir::Operation *
getAllReduceInsertionAnchor(uint64_t tile, RingAllReduceComponent &component) {
  mlir::Operation *anchor = nullptr;
  for (PeerPair &pair : component.oldPairs) {
    mlir::Operation *operation = nullptr;
    if (pair.source == static_cast<int64_t>(tile))
      operation = pair.send;
    else if (pair.destination == static_cast<int64_t>(tile))
      operation = pair.receive;
    if (operation && (!anchor || operation->isBeforeInBlock(anchor)))
      anchor = operation;
  }
  if (tile == component.rootTile &&
      (!anchor || component.merge.root->isBeforeInBlock(anchor)))
    anchor = component.merge.root;
  return anchor;
}

static mlir::LogicalResult
materializeAllReduceComponent(RingAllReduceComponent &component,
                              const TargetTopology &topology,
                              DistributedCollectiveMovementResult &statistics) {
  mlir::FailureOr<llvm::SmallVector<uint64_t, 16>> ring =
      buildMinimumHopTileRing(topology, component.participants);
  std::optional<llvm::SmallVector<LeadingChunk, 16>> chunks =
      buildLeadingChunks(component.fullType, component.participants.size());
  if (mlir::failed(ring) || !chunks)
    return mlir::failure();
  std::map<uint64_t, mlir::Operation *> anchors;
  for (uint64_t tile : *ring) {
    mlir::Operation *anchor = getAllReduceInsertionAnchor(tile, component);
    if (!anchor ||
        anchor->getBlock() != component.contributions.at(tile).getParentBlock())
      return mlir::failure();
    anchors.emplace(tile, anchor);
  }

  std::map<std::pair<uint64_t, unsigned>, mlir::Value> contributionChunks;
  std::map<std::pair<uint64_t, unsigned>, mlir::Value> resultChunks;
  std::map<uint64_t, mlir::Value> results;
  for (uint64_t tile : *ring) {
    mlir::Operation *anchor = anchors.at(tile);
    mlir::OpBuilder builder(anchor);
    mlir::Value result = builder.create<mlir::memref::AllocOp>(
        anchor->getLoc(), component.fullType);
    results.emplace(tile, result);
    for (auto [position, chunk] : llvm::enumerate(*chunks)) {
      auto contribution = createLeadingChunk(builder, anchor->getLoc(),
                                             component.contributions.at(tile),
                                             component.fullType, chunk);
      auto resultChunk = createLeadingChunk(builder, anchor->getLoc(), result,
                                            component.fullType, chunk);
      if (mlir::failed(contribution) || mlir::failed(resultChunk))
        return mlir::failure();
      contributionChunks.emplace(
          std::make_pair(tile, static_cast<unsigned>(position)),
          *contribution);
      resultChunks.emplace(
          std::make_pair(tile, static_cast<unsigned>(position)),
          *resultChunk);
    }
  }

  auto partials = materializeRingReduction(
      *ring, component.communication, component.merge.kind,
      [&](uint64_t tile, unsigned chunk) {
        return contributionChunks.at({tile, chunk});
      },
      [&](unsigned chunk) { return (*chunks)[chunk].type; },
      [&](unsigned chunk) { return (*chunks)[chunk].bytes; },
      [&](uint64_t tile) { return anchors.at(tile); }, statistics);
  if (mlir::failed(partials))
    return mlir::failure();

  for (auto [positionValue, tile] : llvm::enumerate(*ring)) {
    const unsigned position = static_cast<unsigned>(positionValue);
    mlir::Operation *anchor = anchors.at(tile);
    mlir::OpBuilder builder(anchor);
    builder.create<mlir::memref::CopyOp>(anchor->getLoc(), partials->at(tile),
                                         resultChunks.at({tile, position}));
    ++statistics.resultCopies;
    const uint64_t next = (*ring)[(position + 1) % ring->size()];
    const uint64_t previous =
        (*ring)[(position + ring->size() - 1) % ring->size()];
    for (unsigned round = 0; round + 1 < ring->size(); ++round) {
      const unsigned sendChunk =
          (position + ring->size() - round) % ring->size();
      const unsigned receiveChunk =
          (position + ring->size() - round - 1) % ring->size();
      const int64_t protocolRound =
          static_cast<int64_t>(ring->size() - 1 + round);
      auto receiveMessage = DTEMessageAttr::get(component.fullType.getContext(),
                                                component.communication,
                                                protocolRound, receiveChunk);
      auto sendMessage = DTEMessageAttr::get(component.fullType.getContext(),
                                             component.communication,
                                             protocolRound, sendChunk);
      builder.create<CommPeerRecvOp>(
          anchor->getLoc(), resultChunks.at({tile, receiveChunk}), previous,
          (*chunks)[receiveChunk].bytes, receiveMessage);
      builder.create<CommPeerSendOp>(anchor->getLoc(),
                                     resultChunks.at({tile, sendChunk}), next,
                                     (*chunks)[sendChunk].bytes, sendMessage);
      ++statistics.createdPeerSends;
      ++statistics.createdPeerReceives;
    }
  }

  mlir::IRRewriter rewriter(component.fullType.getContext());
  llvm::SmallVector<std::pair<mlir::Value, mlir::Operation *>, 16>
      oldReceiveBuffers;
  for (PeerPair &pair : component.oldPairs) {
    oldReceiveBuffers.push_back(
        {pair.receive.getBuffer(), pair.receive.getBuffer().getDefiningOp()});
    rewriter.eraseOp(pair.send);
    rewriter.eraseOp(pair.receive);
    ++statistics.removedPeerSends;
    ++statistics.removedPeerReceives;
  }
  component.merge.root.getResult().replaceAllUsesWith(
      results.at(component.rootTile));
  for (auto &[tile, oldResult] : component.publishedResults) {
    if (tile == component.rootTile)
      continue;
    oldResult.replaceAllUsesWith(results.at(tile));
  }
  for (mlir::Operation *operation : component.merge.operations) {
    if (!operation->use_empty())
      return mlir::failure();
    rewriter.eraseOp(operation);
  }
  for (const auto &[buffer, owner] : oldReceiveBuffers) {
    (void)buffer;
    if (auto allocation = mlir::dyn_cast_or_null<mlir::memref::AllocOp>(owner);
        allocation && allocation->use_empty())
      rewriter.eraseOp(allocation);
  }
  ++statistics.allReduceComponents;
  return mlir::success();
}

} // namespace

DistributedCollectiveMovementResult
materializeDimensionOrderedAllToAll(mlir::ModuleOp module) {
  if (!module || mlir::failed(mlir::verify(module)))
    return fail(DistributedCollectiveMovementFailureKind::BrokenContract,
                "dimension-ordered AllToAll requires verified current Tile "
                "IR");
  std::string topologyFailure;
  mlir::FailureOr<TargetTopology> topology =
      TargetTopology::create(module, &topologyFailure);
  if (mlir::failed(topology))
    return fail(DistributedCollectiveMovementFailureKind::BrokenContract,
                topologyFailure);
  std::string discoveryFailure;
  llvm::SmallVector<CompletePersonalizedComponent, 2> components =
      findCompletePersonalizedComponents(module, *topology,
                                         discoveryFailure);
  if (!discoveryFailure.empty())
    return fail(DistributedCollectiveMovementFailureKind::BrokenContract,
                discoveryFailure);

  DistributedCollectiveMovementResult result;
  for (CompletePersonalizedComponent &component : components)
    if (mlir::failed(materializeComponent(component, result)))
      return fail(DistributedCollectiveMovementFailureKind::CompilerFailure,
                  "preflighted dimension-ordered AllToAll could not be "
                  "materialized");
  if (mlir::failed(mlir::verify(module)))
    return fail(DistributedCollectiveMovementFailureKind::CompilerFailure,
                "dimension-ordered AllToAll produced invalid current Tile "
                "IR");
  return result;
}

DistributedCollectiveMovementResult
materializeRingReduceScatter(mlir::ModuleOp module) {
  if (!module || mlir::failed(mlir::verify(module)))
    return fail(DistributedCollectiveMovementFailureKind::BrokenContract,
                "Ring ReduceScatter requires verified current Tile IR");
  std::string topologyFailure;
  mlir::FailureOr<TargetTopology> topology =
      TargetTopology::create(module, &topologyFailure);
  if (mlir::failed(topology))
    return fail(DistributedCollectiveMovementFailureKind::BrokenContract,
                topologyFailure);
  std::string discoveryFailure;
  llvm::SmallVector<CompletePersonalizedComponent, 2> components =
      findCompletePersonalizedComponents(module, *topology,
                                         discoveryFailure);
  if (!discoveryFailure.empty())
    return fail(DistributedCollectiveMovementFailureKind::BrokenContract,
                discoveryFailure);

  DistributedCollectiveMovementResult result;
  for (CompletePersonalizedComponent &component : components) {
    bool isReduction = true;
    for (int64_t destination : component.participants)
      isReduction &=
          static_cast<bool>(findAssociativeMergeTree(destination, component));
    if (!isReduction)
      continue;
    if (mlir::failed(
            materializeReduceScatterComponent(component, *topology,
                                              result)))
      return fail(DistributedCollectiveMovementFailureKind::CompilerFailure,
                  "preflighted Ring ReduceScatter could not be materialized");
  }
  if (mlir::failed(mlir::verify(module)))
    return fail(DistributedCollectiveMovementFailureKind::CompilerFailure,
                "Ring ReduceScatter produced invalid current Tile IR");
  return result;
}

DistributedCollectiveMovementResult
materializeRingAllReduce(mlir::ModuleOp module) {
  if (!module || mlir::failed(mlir::verify(module)))
    return fail(DistributedCollectiveMovementFailureKind::BrokenContract,
                "Ring AllReduce requires verified current Tile IR");
  std::string topologyFailure;
  mlir::FailureOr<TargetTopology> topology =
      TargetTopology::create(module, &topologyFailure);
  if (mlir::failed(topology))
    return fail(DistributedCollectiveMovementFailureKind::BrokenContract,
                topologyFailure);
  auto pairs = collectMatchedPeerPairs(module);
  if (mlir::failed(pairs))
    return fail(DistributedCollectiveMovementFailureKind::BrokenContract,
                "current Tile peer messages have duplicate or missing "
                "physical endpoints");

  llvm::SmallVector<RingAllReduceComponent, 2> components;
  llvm::DenseSet<mlir::Operation *> claimed;
  module.walk([&](ComputeElementwiseOp root) {
    if (auto component = findRingAllReduceComponent(root, *pairs,
                                                    claimed))
      components.push_back(std::move(*component));
  });
  DistributedCollectiveMovementResult result;
  for (RingAllReduceComponent &component : components)
    if (mlir::failed(
            materializeAllReduceComponent(component, *topology, result)))
      return fail(DistributedCollectiveMovementFailureKind::CompilerFailure,
                  "preflighted Ring AllReduce could not be materialized");
  if (mlir::failed(mlir::verify(module)))
    return fail(DistributedCollectiveMovementFailureKind::CompilerFailure,
                "Ring AllReduce produced invalid current Tile IR");
  return result;
}

} // namespace wafer::compiler::detail

// NOLINTEND(bugprone-unchecked-optional-access)
