//===- IndexRelationTest.cpp - MLIR-backed index relation tests ----------===//

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Analysis/PhysicalDataflow/PhysicalAccessRelation.h"
#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

#include <limits>

namespace {

using wafer::analysis::IndexRelation;
using wafer::analysis::IndexRelationLimits;
using wafer::analysis::IndexRelationQueryResult;
using wafer::analysis::IndexRelationResult;
using wafer::analysis::IndexRelationStatus;
using wafer::analysis::IndexSetResult;
using wafer::analysis::PhysicalAccessRelation;
using wafer::analysis::PhysicalLayoutRelation;
using wafer::analysis::StaticRectangularIndexSetPiecesResult;
using wafer::analysis::StaticRectangularIndexSetResult;
using wafer::analysis::TransferRealizability;

TEST(IndexRelationTest, RepresentsIdentityPermutationAndBroadcastExactly) {
  IndexRelationResult identity = IndexRelation::identity({2, 3});
  ASSERT_TRUE(identity.isExact());
  EXPECT_TRUE(identity.get()->contains({1, 2}, {1, 2}));
  EXPECT_FALSE(identity.get()->contains({1, 2}, {0, 2}));

  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  IndexRelationResult permutation = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1, d0}, &context),
      /*destinationShape=*/{2, 3}, /*sourceShape=*/{3, 2});
  ASSERT_TRUE(permutation.isExact());
  EXPECT_TRUE(permutation.get()->contains({1, 2}, {2, 1}));

  IndexRelationResult broadcast = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1}, &context),
      /*destinationShape=*/{4, 3}, /*sourceShape=*/{3});
  ASSERT_TRUE(broadcast.isExact());
  EXPECT_TRUE(broadcast.get()->contains({0, 2}, {2}));
  EXPECT_TRUE(broadcast.get()->contains({3, 2}, {2}));

  IndexRelationResult constantProjection = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(1, 0, {mlir::getAffineConstantExpr(0, &context)},
                           &context),
      /*destinationShape=*/{4}, /*sourceShape=*/{3});
  ASSERT_TRUE(constantProjection.isExact());
  StaticRectangularIndexSetPiecesResult constantImage =
      constantProjection.get()->getExactStaticRectangularImagePieces(
          /*destinationOffsets=*/{0}, /*destinationSizes=*/{4});
  ASSERT_TRUE(constantImage.isExact()) << constantImage.reason;
  ASSERT_EQ(constantImage.domains.size(), 1u);
  EXPECT_EQ(constantImage.domains.front().offsets,
            llvm::SmallVector<int64_t>({0}));
  EXPECT_EQ(constantImage.domains.front().sizes,
            llvm::SmallVector<int64_t>({1}));

  std::optional<mlir::AffineMap> recovered =
      permutation.get()->getProjectedAffineMap(&context);
  ASSERT_TRUE(recovered);
  EXPECT_EQ(*recovered, mlir::AffineMap::get(2, 0, {d1, d0}, &context));
}

TEST(IndexRelationTest,
     RecoversRectangularUnionDisjunctsWithoutGenericSetEquality) {
  IndexSetResult first =
      IndexRelation::staticRectangularDomain(/*offsets=*/{0, 0},
                                             /*sizes=*/{16, 64});
  IndexSetResult second =
      IndexRelation::staticRectangularDomain(/*offsets=*/{0, 64},
                                             /*sizes=*/{16, 64});
  ASSERT_TRUE(first.isExact());
  ASSERT_TRUE(second.isExact());
  IndexSetResult combined{wafer::analysis::IndexRelationStatus::Exact,
                          first.set->unionSet(*second.set),
                          {}};

  StaticRectangularIndexSetPiecesResult pieces =
      combined.getExactStaticRectangularDisjuncts();

  ASSERT_TRUE(pieces.isExact()) << pieces.reason;
  ASSERT_EQ(pieces.domains.size(), 2u);
  EXPECT_EQ(pieces.domains[0].offsets, llvm::SmallVector<int64_t>({0, 0}));
  EXPECT_EQ(pieces.domains[0].sizes, llvm::SmallVector<int64_t>({16, 64}));
  EXPECT_EQ(pieces.domains[1].offsets, llvm::SmallVector<int64_t>({0, 64}));
  EXPECT_EQ(pieces.domains[1].sizes, llvm::SmallVector<int64_t>({16, 64}));
}

TEST(IndexRelationTest,
     SingletonReshapeCompositionKeepsProjectedRectangleImage) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d2 = mlir::getAffineDimExpr(2, &context);
  IndexRelationResult matmulInput = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(3, 0, {d0, d2}, &context),
      /*destinationShape=*/{16, 4096, 4096},
      /*sourceShape=*/{16, 4096});
  ASSERT_TRUE(matmulInput.isExact());
  IndexRelationResult insertUnitDimension =
      IndexRelation::staticReshape(/*destinationShape=*/{16, 4096},
                                   /*sourceShape=*/{1, 16, 4096});
  ASSERT_TRUE(insertUnitDimension.isExact());

  IndexRelationResult composed =
      matmulInput.get()->compose(*insertUnitDimension.get());
  ASSERT_TRUE(composed.isExact());
  IndexSetResult fullSource = IndexRelation::staticDomain({1, 16, 4096});
  ASSERT_TRUE(fullSource.isExact());
  IndexRelationResult restricted =
      composed.get()->intersectSourceDomain(*fullSource.set);
  ASSERT_TRUE(restricted.isExact());

  StaticRectangularIndexSetResult first =
      restricted.get()->getExactStaticRectangularImage(
          /*destinationOffsets=*/{0, 0, 0},
          /*destinationSizes=*/{16, 256, 4096});
  StaticRectangularIndexSetResult second =
      restricted.get()->getExactStaticRectangularImage(
          /*destinationOffsets=*/{0, 256, 0},
          /*destinationSizes=*/{16, 256, 4096});
  ASSERT_TRUE(first.isExact()) << first.reason;
  ASSERT_TRUE(second.isExact()) << second.reason;
  EXPECT_EQ(first.domain->offsets, llvm::SmallVector<int64_t>({0, 0, 0}));
  EXPECT_EQ(first.domain->sizes, llvm::SmallVector<int64_t>({1, 16, 4096}));
  EXPECT_EQ(second.domain->offsets, first.domain->offsets);
  EXPECT_EQ(second.domain->sizes, first.domain->sizes);
}

TEST(IndexRelationTest,
     CollapsingReshapeCompositionKeepsMixedRadixRectangleImage) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  mlir::AffineExpr d2 = mlir::getAffineDimExpr(2, &context);
  mlir::AffineExpr d3 = mlir::getAffineDimExpr(3, &context);
  IndexRelationResult transpose = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(4, 0, {d0, d2, d1, d3}, &context),
      /*destinationShape=*/{1, 32, 16, 128},
      /*sourceShape=*/{1, 16, 32, 128});
  ASSERT_TRUE(transpose.isExact());
  IndexRelationResult collapse =
      IndexRelation::staticReshape(/*destinationShape=*/{1, 16, 32, 128},
                                   /*sourceShape=*/{16, 4096});
  ASSERT_TRUE(collapse.isExact());
  IndexRelationResult composed = transpose.get()->compose(*collapse.get());
  ASSERT_TRUE(composed.isExact());

  StaticRectangularIndexSetResult complete =
      composed.get()->getExactStaticRectangularImage(
          /*destinationOffsets=*/{0, 0, 0, 0},
          /*destinationSizes=*/{1, 32, 16, 128});
  StaticRectangularIndexSetResult shard =
      composed.get()->getExactStaticRectangularImage(
          /*destinationOffsets=*/{0, 2, 0, 0},
          /*destinationSizes=*/{1, 2, 16, 128});
  ASSERT_TRUE(complete.isExact()) << complete.reason;
  ASSERT_TRUE(shard.isExact()) << shard.reason;
  EXPECT_EQ(complete.domain->offsets, llvm::SmallVector<int64_t>({0, 0}));
  EXPECT_EQ(complete.domain->sizes, llvm::SmallVector<int64_t>({16, 4096}));
  EXPECT_EQ(shard.domain->offsets, llvm::SmallVector<int64_t>({0, 256}));
  EXPECT_EQ(shard.domain->sizes, llvm::SmallVector<int64_t>({16, 256}));

  StaticRectangularIndexSetPiecesResult strided =
      composed.get()->getExactStaticRectangularImagePieces(
          /*destinationOffsets=*/{0, 0, 0, 0},
          /*destinationSizes=*/{1, 32, 16, 8});
  ASSERT_TRUE(strided.isExact()) << strided.reason;
  ASSERT_EQ(strided.domains.size(), 32u);
  for (auto [ordinal, piece] : llvm::enumerate(strided.domains)) {
    EXPECT_EQ(piece.offsets,
              llvm::SmallVector<int64_t>({0, int64_t(ordinal) * 128}));
    EXPECT_EQ(piece.sizes, llvm::SmallVector<int64_t>({16, 8}));
  }
}

TEST(IndexRelationTest,
     ExpandingReshapeCompositionKeepsMixedRadixRectangleImage) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d2 = mlir::getAffineDimExpr(2, &context);
  IndexRelationResult matmulInput = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(3, 0, {d0, d2}, &context),
      /*destinationShape=*/{16, 4096, 4096},
      /*sourceShape=*/{16, 4096});
  ASSERT_TRUE(matmulInput.isExact());
  IndexRelationResult expand =
      IndexRelation::staticReshape(/*destinationShape=*/{16, 4096},
                                   /*sourceShape=*/{1, 16, 32, 128});
  ASSERT_TRUE(expand.isExact());
  IndexRelationResult composed = matmulInput.get()->compose(*expand.get());
  ASSERT_TRUE(composed.isExact());

  StaticRectangularIndexSetResult outputShard =
      composed.get()->getExactStaticRectangularImage(
          /*destinationOffsets=*/{0, 512, 0},
          /*destinationSizes=*/{16, 256, 4096});
  StaticRectangularIndexSetResult reductionShard =
      composed.get()->getExactStaticRectangularImage(
          /*destinationOffsets=*/{0, 0, 256},
          /*destinationSizes=*/{16, 4096, 256});
  ASSERT_TRUE(outputShard.isExact()) << outputShard.reason;
  ASSERT_TRUE(reductionShard.isExact()) << reductionShard.reason;
  EXPECT_EQ(outputShard.domain->offsets,
            llvm::SmallVector<int64_t>({0, 0, 0, 0}));
  EXPECT_EQ(outputShard.domain->sizes,
            llvm::SmallVector<int64_t>({1, 16, 32, 128}));
  EXPECT_EQ(reductionShard.domain->offsets,
            llvm::SmallVector<int64_t>({0, 0, 2, 0}));
  EXPECT_EQ(reductionShard.domain->sizes,
            llvm::SmallVector<int64_t>({1, 16, 2, 128}));
}

TEST(PhysicalAccessRelationTest,
     ComposesEveryLayoutPairAcrossDtypesBlocksAndTails) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  const llvm::SmallVector<int64_t, 4> shape{2, 17, 197};
  IndexRelationResult identity = IndexRelation::identity(shape);
  ASSERT_TRUE(identity.isExact());

  const llvm::SmallVector<mlir::Type, 5> elementTypes{
      mlir::Float16Type::get(&context), mlir::BFloat16Type::get(&context),
      mlir::Float32Type::get(&context), mlir::IntegerType::get(&context, 8),
      mlir::IntegerType::get(&context, 1)};
  const llvm::SmallVector<wafer::MemLayout, 4> layouts{
      wafer::MemLayout::Tensor, wafer::MemLayout::NTensor, wafer::MemLayout::Cx,
      wafer::MemLayout::NCx};
  const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 6> points{
      {0, 0, 0},   {0, 3, 63},  {0, 3, 64},
      {1, 7, 127}, {1, 7, 128}, {1, 16, 196}};

  for (auto [dtypeIndex, elementType] : llvm::enumerate(elementTypes)) {
    llvm::SmallVector<mlir::MemRefType, 4> types;
    for (wafer::MemLayout layout : layouts) {
      auto memory =
          wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM, layout);
      mlir::MemRefType type = mlir::MemRefType::get(
          shape, elementType, mlir::MemRefLayoutAttrInterface{}, memory);
      types.push_back(type);

      mlir::FailureOr<PhysicalAccessRelation> access =
          PhysicalAccessRelation::create(type, shape, *identity.get(),
                                         /*requireInjective=*/true);
      ASSERT_TRUE(mlir::succeeded(access));
      auto encoding =
          mlir::dyn_cast<wafer::WaferPhysicalEncodingAttrInterface>(memory);
      ASSERT_TRUE(encoding);
      EXPECT_EQ(access->getPhysicalFootprintBytes(),
                *encoding.getPhysicalFootprintBytes(type));
      EXPECT_EQ(access->getMinimumAlignmentBytes(),
                *encoding.getMinimumAlignmentBytes(type));
      for (llvm::ArrayRef<int64_t> point : points) {
        SCOPED_TRACE(testing::Message()
                     << "dtype=" << dtypeIndex << " layout="
                     << static_cast<unsigned>(layout) << " point=" << point[0]
                     << "," << point[1] << "," << point[2]);
        mlir::FailureOr<wafer::WaferPhysicalElementSpan> composed =
            access->getPhysicalElementSpan(point);
        mlir::FailureOr<wafer::WaferPhysicalElementSpan> direct =
            encoding.getPhysicalElementSpan(type, point);
        ASSERT_TRUE(mlir::succeeded(composed));
        ASSERT_TRUE(mlir::succeeded(direct));
        EXPECT_EQ(*composed, *direct);
      }
    }

    for (mlir::MemRefType source : types)
      for (mlir::MemRefType dest : types)
        EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveMappedTransfer(
            source, dest, shape, *identity.get(), *identity.get())));
  }
}

TEST(PhysicalAccessRelationTest,
     NormalizesDtypeSpecificBitOffsetsToPhysicalTraversal) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  const llvm::SmallVector<int64_t, 3> shape{2, 2, 197};
  IndexRelationResult identity = IndexRelation::identity(shape);
  ASSERT_TRUE(identity.isExact());
  auto makeType = [&](mlir::Type elementType, wafer::MemLayout layout) {
    return mlir::MemRefType::get(
        shape, elementType, mlir::MemRefLayoutAttrInterface{},
        wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM, layout));
  };

  mlir::MemRefType f16NCx =
      makeType(mlir::Float16Type::get(&context), wafer::MemLayout::NCx);
  mlir::MemRefType bf16NCx =
      makeType(mlir::BFloat16Type::get(&context), wafer::MemLayout::NCx);
  mlir::MemRefType f32NCx =
      makeType(mlir::Float32Type::get(&context), wafer::MemLayout::NCx);
  mlir::MemRefType i1NCx =
      makeType(mlir::IntegerType::get(&context, 1), wafer::MemLayout::NCx);
  mlir::MemRefType f16Tensor =
      makeType(mlir::Float16Type::get(&context), wafer::MemLayout::Tensor);
  mlir::MemRefType f32Tensor =
      makeType(mlir::Float32Type::get(&context), wafer::MemLayout::Tensor);
  mlir::MemRefType i1Tensor =
      makeType(mlir::IntegerType::get(&context, 1), wafer::MemLayout::Tensor);

  EXPECT_TRUE(mlir::succeeded(TransferRealizability::provePhysicalTraversal(
      f16NCx, bf16NCx, shape, *identity.get(), *identity.get())));
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::provePhysicalTraversal(
      f16Tensor, f32Tensor, shape, *identity.get(), *identity.get())));
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::provePhysicalTraversal(
      i1Tensor, f32Tensor, shape, *identity.get(), *identity.get())));
  EXPECT_TRUE(mlir::failed(TransferRealizability::provePhysicalTraversal(
      f16NCx, f32NCx, shape, *identity.get(), *identity.get())));
  EXPECT_TRUE(mlir::failed(TransferRealizability::provePhysicalTraversal(
      f16NCx, i1NCx, shape, *identity.get(), *identity.get())));
  EXPECT_TRUE(mlir::failed(TransferRealizability::provePhysicalTraversal(
      f16Tensor, f16NCx, shape, *identity.get(), *identity.get())));

  const llvm::SmallVector<int64_t, 2> blockShape{2, 65};
  IndexRelationResult blockIdentity = IndexRelation::identity(blockShape);
  ASSERT_TRUE(blockIdentity.isExact());
  auto cxMemory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                         wafer::MemLayout::Cx);
  mlir::MemRefType f16Cx =
      mlir::MemRefType::get(blockShape, mlir::Float16Type::get(&context),
                            mlir::MemRefLayoutAttrInterface{}, cxMemory);
  mlir::MemRefType i8Cx =
      mlir::MemRefType::get(blockShape, mlir::IntegerType::get(&context, 8),
                            mlir::MemRefLayoutAttrInterface{}, cxMemory);
  mlir::MemRefType i1Cx =
      mlir::MemRefType::get(blockShape, mlir::IntegerType::get(&context, 1),
                            mlir::MemRefLayoutAttrInterface{}, cxMemory);
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::provePhysicalTraversal(
      i1Cx, f16Cx, blockShape, *blockIdentity.get(), *blockIdentity.get())));
  EXPECT_TRUE(mlir::failed(TransferRealizability::provePhysicalTraversal(
      f16Cx, i8Cx, blockShape, *blockIdentity.get(), *blockIdentity.get())));
}

TEST(PhysicalAccessRelationTest,
     ComposesPermutationAndEnforcesWriterInjectivity) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();
  mlir::Type f16 = mlir::Float16Type::get(&context);
  auto cxMemory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                         wafer::MemLayout::Cx);

  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  IndexRelationResult transpose = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1, d0}, &context),
      /*destinationShape=*/{5, 97}, /*sourceShape=*/{97, 5});
  ASSERT_TRUE(transpose.isExact());
  mlir::MemRefType transposedEndpoint = mlir::MemRefType::get(
      {97, 5}, f16, mlir::MemRefLayoutAttrInterface{}, cxMemory);
  mlir::FailureOr<PhysicalAccessRelation> transposeAccess =
      PhysicalAccessRelation::create(transposedEndpoint, {5, 97},
                                     *transpose.get(),
                                     /*requireInjective=*/true);
  ASSERT_TRUE(mlir::succeeded(transposeAccess));
  EXPECT_EQ(*transposeAccess->getLogicalPoint({3, 64}),
            (llvm::SmallVector<int64_t, 4>{64, 3}));

  IndexRelationResult broadcast = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1}, &context),
      /*destinationShape=*/{4, 97}, /*sourceShape=*/{97});
  ASSERT_TRUE(broadcast.isExact());
  mlir::MemRefType broadcastEndpoint = mlir::MemRefType::get(
      {97}, f16, mlir::MemRefLayoutAttrInterface{}, cxMemory);
  EXPECT_TRUE(mlir::succeeded(PhysicalAccessRelation::create(
      broadcastEndpoint, {4, 97}, *broadcast.get(),
      /*requireInjective=*/false)));
  EXPECT_TRUE(mlir::failed(PhysicalAccessRelation::create(
      broadcastEndpoint, {4, 97}, *broadcast.get(),
      /*requireInjective=*/true)));
}

TEST(PhysicalLayoutRelationTest,
     NormalizesBlockedEncodingPiecesIntoExactPresburgerMap) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();
  auto memory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                       wafer::MemLayout::Cx);
  mlir::MemRefType type =
      mlir::MemRefType::get({2, 17, 197}, mlir::Float16Type::get(&context),
                            mlir::MemRefLayoutAttrInterface{}, memory);

  mlir::FailureOr<PhysicalLayoutRelation> layout =
      PhysicalLayoutRelation::create(type);
  ASSERT_TRUE(mlir::succeeded(layout));
  ASSERT_EQ(layout->getPieces().size(), 2u);
  EXPECT_EQ(layout->getPieces()[0].logicalTilePeriods,
            (llvm::SmallVector<int64_t, 4>{0, 0, 64}));
  EXPECT_EQ(layout->getPieces()[1].logicalTilePeriods,
            (llvm::SmallVector<int64_t, 4>{0, 0, 0}));
  EXPECT_TRUE(
      layout->getLogicalToPhysicalBitOffset().isFunctional().isProvenTrue());
  EXPECT_TRUE(
      layout->getLogicalToPhysicalBitOffset().isInjective().isProvenTrue());

  std::optional<wafer::WaferStaticPhysicalOffsetCalculator> calculator =
      wafer::WaferStaticPhysicalOffsetCalculator::create(type);
  ASSERT_TRUE(calculator);
  for (llvm::SmallVector<int64_t, 4> point :
       {llvm::SmallVector<int64_t, 4>{0, 3, 63},
        llvm::SmallVector<int64_t, 4>{0, 3, 64},
        llvm::SmallVector<int64_t, 4>{1, 16, 196}}) {
    std::optional<int64_t> byteOffset = calculator->getByteOffset(point);
    ASSERT_TRUE(byteOffset);
    EXPECT_TRUE(layout->getLogicalToPhysicalBitOffset().contains(
        point, {*byteOffset * 8}));
  }
}

TEST(IndexRelationTest, ProjectedAffineMapIsDerivedAndFailsClosed) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  IndexRelationResult slice = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d0 * 2 + 1, d1 * -1 + 6}, &context),
      /*destinationShape=*/{3, 7}, /*sourceShape=*/{6, 7});
  ASSERT_TRUE(slice.isExact());
  std::optional<mlir::AffineMap> recovered =
      slice.get()->getProjectedAffineMap(&context);
  ASSERT_TRUE(recovered);
  EXPECT_EQ(*recovered,
            mlir::AffineMap::get(2, 0, {d0 * 2 + 1, d1 * -1 + 6}, &context));

  IndexRelationResult reshape =
      IndexRelation::staticReshape(/*destinationShape=*/{6},
                                   /*sourceShape=*/{2, 3});
  ASSERT_TRUE(reshape.isExact());
  EXPECT_FALSE(reshape.get()->getProjectedAffineMap(&context));
}

TEST(IndexRelationTest, ComposesSliceAndReshapePointwise) {
  IndexRelationResult slice = IndexRelation::staticSlice(
      /*destinationShape=*/{2, 2}, /*sourceShape=*/{4, 4},
      /*offsets=*/{1, 1}, /*strides=*/{1, 1});
  IndexRelationResult reshape =
      IndexRelation::staticReshape(/*destinationShape=*/{4, 4},
                                   /*sourceShape=*/{16});
  ASSERT_TRUE(slice.isExact());
  ASSERT_TRUE(reshape.isExact());

  IndexRelationResult composed = slice.get()->compose(*reshape.get());
  ASSERT_TRUE(composed.isExact());
  EXPECT_TRUE(composed.get()->contains({0, 0}, {5}));
  EXPECT_TRUE(composed.get()->contains({1, 1}, {10}));
  EXPECT_FALSE(composed.get()->contains({1, 1}, {11}));
}

TEST(IndexRelationTest, ComputesImagePreimageAndDomainIntersections) {
  IndexRelationResult slice = IndexRelation::staticSlice(
      /*destinationShape=*/{2, 2}, /*sourceShape=*/{4, 5},
      /*offsets=*/{1, 2}, /*strides=*/{1, 1});
  IndexSetResult destinationDomain = IndexRelation::staticDomain({1, 2});
  IndexSetResult sourceDomain = IndexRelation::staticDomain({3, 4});
  ASSERT_TRUE(slice.isExact());
  ASSERT_TRUE(destinationDomain.isExact());
  ASSERT_TRUE(sourceDomain.isExact());

  IndexSetResult image = slice.get()->image(*destinationDomain.set);
  ASSERT_TRUE(image.isExact());
  EXPECT_TRUE(image.contains({1, 2}));
  EXPECT_TRUE(image.contains({1, 3}));
  EXPECT_FALSE(image.contains({2, 2}));

  IndexSetResult preimage = slice.get()->preimage(*sourceDomain.set);
  ASSERT_TRUE(preimage.isExact());
  EXPECT_TRUE(preimage.contains({0, 0}));
  EXPECT_TRUE(preimage.contains({1, 1}));
  EXPECT_FALSE(preimage.contains({0, 2}));

  IndexRelationResult destinationRestricted =
      slice.get()->intersectDestinationDomain(*destinationDomain.set);
  ASSERT_TRUE(destinationRestricted.isExact());
  EXPECT_TRUE(destinationRestricted.get()->contains({0, 1}, {1, 3}));
  EXPECT_FALSE(destinationRestricted.get()->contains({1, 1}, {2, 3}));

  IndexRelationResult sourceRestricted =
      slice.get()->intersectSourceDomain(*sourceDomain.set);
  ASSERT_TRUE(sourceRestricted.isExact());
  EXPECT_TRUE(sourceRestricted.get()->contains({1, 1}, {2, 3}));
  EXPECT_FALSE(sourceRestricted.get()->contains({1, 1}, {2, 4}));
}

TEST(IndexRelationTest,
     RecoversExactDenseDemandFromOneToManyReductionAndWindowRelations) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);

  IndexRelationResult reduction = IndexRelation::fromCommonIterationDomain(
      mlir::AffineMap::get(2, 0, {d0}, &context),
      /*destinationShape=*/{8}, mlir::AffineMap::get(2, 0, {d0, d1}, &context),
      /*sourceShape=*/{8, 4}, /*iterationShape=*/{8, 4});
  IndexSetResult reductionShard =
      IndexRelation::staticRectangularDomain(/*offsets=*/{2}, /*sizes=*/{3});
  ASSERT_TRUE(reduction.isExact());
  EXPECT_FALSE(reduction.get()->isFunctional().isProvenTrue());
  ASSERT_TRUE(reductionShard.isExact());
  IndexSetResult reductionDemand = reduction.get()->image(*reductionShard.set);
  ASSERT_TRUE(reductionDemand.isExact());
  StaticRectangularIndexSetResult reductionRectangle =
      reductionDemand.getExactStaticRectangularDomain();
  ASSERT_TRUE(reductionRectangle.isExact()) << reductionRectangle.reason;
  EXPECT_EQ(reductionRectangle.domain->offsets,
            (llvm::SmallVector<int64_t, 4>{2, 0}));
  EXPECT_EQ(reductionRectangle.domain->sizes,
            (llvm::SmallVector<int64_t, 4>{3, 4}));
  StaticRectangularIndexSetResult reductionDirect =
      reduction.get()->getExactStaticRectangularImage(/*offsets=*/{2},
                                                      /*sizes=*/{3});
  ASSERT_TRUE(reductionDirect.isExact()) << reductionDirect.reason;
  EXPECT_EQ(reductionDirect.domain->offsets,
            reductionRectangle.domain->offsets);
  EXPECT_EQ(reductionDirect.domain->sizes, reductionRectangle.domain->sizes);

  IndexRelationResult window = IndexRelation::fromCommonIterationDomain(
      mlir::AffineMap::get(2, 0, {d0}, &context),
      /*destinationShape=*/{4}, mlir::AffineMap::get(2, 0, {d0 + d1}, &context),
      /*sourceShape=*/{6}, /*iterationShape=*/{4, 3});
  IndexSetResult windowShard =
      IndexRelation::staticRectangularDomain(/*offsets=*/{1}, /*sizes=*/{2});
  ASSERT_TRUE(window.isExact());
  ASSERT_TRUE(windowShard.isExact());
  IndexSetResult windowDemand = window.get()->image(*windowShard.set);
  ASSERT_TRUE(windowDemand.isExact());
  StaticRectangularIndexSetResult windowRectangle =
      windowDemand.getExactStaticRectangularDomain();
  ASSERT_TRUE(windowRectangle.isExact()) << windowRectangle.reason;
  EXPECT_EQ(windowRectangle.domain->offsets,
            (llvm::SmallVector<int64_t, 4>{1}));
  EXPECT_EQ(windowRectangle.domain->sizes, (llvm::SmallVector<int64_t, 4>{4}));
  StaticRectangularIndexSetResult windowDirect =
      window.get()->getExactStaticRectangularImage(/*offsets=*/{1},
                                                   /*sizes=*/{2});
  ASSERT_TRUE(windowDirect.isExact()) << windowDirect.reason;
  EXPECT_EQ(windowDirect.domain->offsets, windowRectangle.domain->offsets);
  EXPECT_EQ(windowDirect.domain->sizes, windowRectangle.domain->sizes);
}

TEST(IndexRelationTest, DoesNotReplaceExactStridedDemandWithBoundingBox) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  IndexRelationResult stride = IndexRelation::fromCommonIterationDomain(
      mlir::AffineMap::get(1, 0, {d0}, &context),
      /*destinationShape=*/{4}, mlir::AffineMap::get(1, 0, {d0 * 2}, &context),
      /*sourceShape=*/{7}, /*iterationShape=*/{4});
  IndexSetResult shard = IndexRelation::staticDomain({4});
  ASSERT_TRUE(stride.isExact());
  ASSERT_TRUE(shard.isExact());
  IndexSetResult demand = stride.get()->image(*shard.set);
  ASSERT_TRUE(demand.isExact());
  EXPECT_TRUE(demand.contains({0}));
  EXPECT_TRUE(demand.contains({2}));
  EXPECT_FALSE(demand.contains({1}));
  StaticRectangularIndexSetResult rectangle =
      demand.getExactStaticRectangularDomain();
  EXPECT_FALSE(rectangle.isExact());
  EXPECT_EQ(rectangle.status, IndexRelationStatus::Unsupported);
  EXPECT_NE(rectangle.reason.find("not one dense static rectangle"),
            std::string::npos);
}

TEST(IndexRelationTest, ProjectedRectangleFastPathRejectsOutOfBoundsDomain) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  IndexRelationResult broadcast = IndexRelation::fromCommonIterationDomain(
      mlir::AffineMap::get(2, 0, {d0, d1}, &context),
      /*destinationShape=*/{8, 4}, mlir::AffineMap::get(2, 0, {d1}, &context),
      /*sourceShape=*/{4}, /*iterationShape=*/{8, 4});
  ASSERT_TRUE(broadcast.isExact());

  StaticRectangularIndexSetResult demand =
      broadcast.get()->getExactStaticRectangularImage(/*offsets=*/{2, 1},
                                                      /*sizes=*/{3, 2});
  ASSERT_TRUE(demand.isExact()) << demand.reason;
  EXPECT_EQ(demand.domain->offsets, (llvm::SmallVector<int64_t, 4>{1}));
  EXPECT_EQ(demand.domain->sizes, (llvm::SmallVector<int64_t, 4>{2}));

  StaticRectangularIndexSetResult outOfBounds =
      broadcast.get()->getExactStaticRectangularImage(/*offsets=*/{7, 1},
                                                      /*sizes=*/{2, 2});
  EXPECT_FALSE(outOfBounds.isExact());
  EXPECT_EQ(outOfBounds.status, IndexRelationStatus::Invalid);
  EXPECT_NE(outOfBounds.reason.find("out of bounds"), std::string::npos);
}

TEST(IndexRelationTest, ProjectedRectangleFastPathPreservesSourceBounds) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);

  // The affine expression is identity, but the exact bounded relation exists
  // only on [0,4). The projected fast path must not return the unbounded map
  // image [0,8).
  IndexRelationResult clipped = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(1, 0, {d0}, &context),
      /*destinationShape=*/{8}, /*sourceShape=*/{4});
  ASSERT_TRUE(clipped.isExact());
  StaticRectangularIndexSetResult image =
      clipped.get()->getExactStaticRectangularImage(/*offsets=*/{0},
                                                    /*sizes=*/{8});
  ASSERT_TRUE(image.isExact()) << image.reason;
  EXPECT_EQ(image.domain->offsets, (llvm::SmallVector<int64_t, 4>{0}));
  EXPECT_EQ(image.domain->sizes, (llvm::SmallVector<int64_t, 4>{4}));

  // Source queries are clipped symmetrically by the destination bound.
  IndexRelationResult reverseClip = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(1, 0, {d0}, &context),
      /*destinationShape=*/{4}, /*sourceShape=*/{8});
  IndexSetResult fullSource = IndexRelation::staticDomain({8});
  ASSERT_TRUE(reverseClip.isExact());
  ASSERT_TRUE(fullSource.isExact());
  IndexSetResult preimage = reverseClip.get()->preimage(*fullSource.set);
  ASSERT_TRUE(preimage.isExact()) << preimage.reason;
  EXPECT_TRUE(preimage.contains({0}));
  EXPECT_TRUE(preimage.contains({3}));
  EXPECT_FALSE(preimage.contains({4}));
}

TEST(PhysicalAccessRelationTest, RejectsAffineMapClippedByEndpointBounds) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  IndexRelationResult clipped = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(1, 0, {d0 + 5}, &context),
      /*destinationShape=*/{10}, /*sourceShape=*/{10});
  ASSERT_TRUE(clipped.isExact());
  EXPECT_FALSE(clipped.get()->hasTotalBoundedAffineMapConstruction());

  auto memory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                       wafer::MemLayout::Tensor);
  mlir::MemRefType endpoint =
      mlir::MemRefType::get({10}, mlir::Float16Type::get(&context),
                            mlir::MemRefLayoutAttrInterface{}, memory);
  EXPECT_TRUE(mlir::failed(PhysicalAccessRelation::create(
      endpoint, /*iterationShape=*/{10}, *clipped.get(),
      /*requireInjective=*/true)));
}

TEST(PhysicalAccessRelationTest,
     AcceptsLargeInBoundsConstantTupleByConstruction) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  mlir::AffineMap tuple = mlir::AffineMap::get(
      /*dimCount=*/0, /*symbolCount=*/0,
      {mlir::getAffineConstantExpr(1, &context),
       mlir::getAffineConstantExpr(1023, &context),
       mlir::getAffineConstantExpr(127, &context)},
      &context);
  IndexRelationResult relation = IndexRelation::fromAffineMap(
      tuple, /*destinationShape=*/{}, /*sourceShape=*/{2, 1024, 128});
  ASSERT_TRUE(relation.isExact());
  EXPECT_TRUE(relation.get()->hasTotalBoundedAffineMapConstruction());

  auto memory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                       wafer::MemLayout::Tensor);
  mlir::MemRefType endpoint =
      mlir::MemRefType::get({2, 1024, 128}, mlir::Float16Type::get(&context),
                            mlir::MemRefLayoutAttrInterface{}, memory);
  EXPECT_TRUE(mlir::succeeded(PhysicalAccessRelation::create(
      endpoint, /*iterationShape=*/{}, *relation.get(),
      /*requireInjective=*/false)));

  mlir::AffineMap outside = mlir::AffineMap::get(
      /*dimCount=*/0, /*symbolCount=*/0,
      {mlir::getAffineConstantExpr(2, &context),
       mlir::getAffineConstantExpr(0, &context),
       mlir::getAffineConstantExpr(0, &context)},
      &context);
  IndexRelationResult clipped = IndexRelation::fromAffineMap(
      outside, /*destinationShape=*/{}, /*sourceShape=*/{2, 1024, 128});
  ASSERT_TRUE(clipped.isExact());
  EXPECT_FALSE(clipped.get()->hasTotalBoundedAffineMapConstruction());
  EXPECT_TRUE(mlir::failed(PhysicalAccessRelation::create(
      endpoint, /*iterationShape=*/{}, *clipped.get(),
      /*requireInjective=*/false)));
}

TEST(PhysicalAccessRelationTest,
     AcceptsLargeProjectedSliceWithNonzeroConstantByConstruction) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  mlir::AffineMap slice = mlir::AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/0,
      {d0, d1, mlir::getAffineConstantExpr(127, &context)}, &context);
  IndexRelationResult relation =
      IndexRelation::fromAffineMap(slice, /*destinationShape=*/{2, 1024},
                                   /*sourceShape=*/{2, 1024, 128});
  ASSERT_TRUE(relation.isExact());
  EXPECT_TRUE(relation.get()->hasTotalBoundedAffineMapConstruction());

  auto memory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                       wafer::MemLayout::NCx);
  mlir::MemRefType endpoint =
      mlir::MemRefType::get({2, 1024, 128}, mlir::Float16Type::get(&context),
                            mlir::MemRefLayoutAttrInterface{}, memory);
  mlir::FailureOr<PhysicalAccessRelation> access =
      PhysicalAccessRelation::create(endpoint, /*iterationShape=*/{2, 1024},
                                     *relation.get(),
                                     /*requireInjective=*/true);
  ASSERT_TRUE(mlir::succeeded(access));
  EXPECT_TRUE(mlir::succeeded(access->getPhysicalElementSpan({1, 1023})));

  mlir::AffineMap outside = mlir::AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/0,
      {d0, d1, mlir::getAffineConstantExpr(128, &context)}, &context);
  IndexRelationResult clipped =
      IndexRelation::fromAffineMap(outside, /*destinationShape=*/{2, 1024},
                                   /*sourceShape=*/{2, 1024, 128});
  ASSERT_TRUE(clipped.isExact());
  EXPECT_FALSE(clipped.get()->hasTotalBoundedAffineMapConstruction());
  EXPECT_TRUE(mlir::failed(PhysicalAccessRelation::create(
      endpoint, /*iterationShape=*/{2, 1024}, *clipped.get(),
      /*requireInjective=*/true)));
}

TEST(IndexRelationTest, RecoversNonemptyZeroDimensionalRectangleWithoutSolver) {
  IndexSetResult scalarDomain = IndexRelation::staticDomain({});
  ASSERT_TRUE(scalarDomain.isExact());
  StaticRectangularIndexSetResult rectangle =
      scalarDomain.getExactStaticRectangularDomain();
  ASSERT_TRUE(rectangle.isExact()) << rectangle.reason;
  EXPECT_TRUE(rectangle.domain->offsets.empty());
  EXPECT_TRUE(rectangle.domain->sizes.empty());
}

TEST(IndexRelationTest, DomainRestrictionInvalidatesProjectedRectangleProof) {
  IndexRelationResult identity = IndexRelation::identity({10});
  IndexSetResult prefix =
      IndexRelation::staticRectangularDomain(/*offsets=*/{0}, /*sizes=*/{5});
  ASSERT_TRUE(identity.isExact());
  ASSERT_TRUE(prefix.isExact());

  IndexRelationResult restricted =
      identity.get()->intersectDestinationDomain(*prefix.set);
  ASSERT_TRUE(restricted.isExact());
  StaticRectangularIndexSetResult clippedImage =
      restricted.get()->getExactStaticRectangularImage(/*offsets=*/{5},
                                                       /*sizes=*/{1});
  EXPECT_FALSE(clippedImage.isExact());
  EXPECT_EQ(clippedImage.status, IndexRelationStatus::Unsupported);
}

TEST(IndexRelationTest,
     CompleteReductionShortcutRequiresCompleteSourceCoverage) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr zero = mlir::getAffineConstantExpr(0, &context);
  IndexRelationResult reduction = IndexRelation::fromCommonIterationDomain(
      mlir::AffineMap::get(2, 0, {zero}, &context),
      /*destinationShape=*/{1},
      mlir::AffineMap::get(2, 0, {d0, zero}, &context),
      /*sourceShape=*/{4, 4}, /*iterationShape=*/{4, 4});
  ASSERT_TRUE(reduction.isExact());

  StaticRectangularIndexSetResult image =
      reduction.get()->getExactStaticRectangularImage(/*offsets=*/{0},
                                                      /*sizes=*/{1});
  ASSERT_TRUE(image.isExact()) << image.reason;
  EXPECT_EQ(image.domain->offsets, (llvm::SmallVector<int64_t, 4>{0, 0}));
  EXPECT_EQ(image.domain->sizes, (llvm::SmallVector<int64_t, 4>{4, 1}));
}

TEST(IndexRelationTest, CompositionDoesNotDropIntermediateBounds) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  IndexRelationResult toIntermediate = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(1, 0, {d0}, &context),
      /*destinationShape=*/{8}, /*sourceShape=*/{4});
  IndexRelationResult fromIntermediate = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(1, 0, {d0}, &context),
      /*destinationShape=*/{8}, /*sourceShape=*/{8});
  ASSERT_TRUE(toIntermediate.isExact());
  ASSERT_TRUE(fromIntermediate.isExact());
  IndexRelationResult composed =
      toIntermediate.get()->compose(*fromIntermediate.get());
  ASSERT_TRUE(composed.isExact());
  StaticRectangularIndexSetResult image =
      composed.get()->getExactStaticRectangularImage(/*offsets=*/{0},
                                                     /*sizes=*/{8});
  ASSERT_TRUE(image.isExact()) << image.reason;
  EXPECT_EQ(image.domain->sizes, (llvm::SmallVector<int64_t, 4>{4}));
}

TEST(IndexRelationTest, ProvesFunctionalInjectiveBijectiveAndContainment) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  IndexRelationResult identity = IndexRelation::identity({2, 3});
  IndexRelationResult permutation = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1, d0}, &context), {2, 3}, {3, 2});
  IndexRelationResult broadcast = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1}, &context), {4, 3}, {3});
  ASSERT_TRUE(identity.isExact());
  ASSERT_TRUE(permutation.isExact());
  ASSERT_TRUE(broadcast.isExact());

  EXPECT_TRUE(identity.get()->isFunctional().isProvenTrue());
  EXPECT_TRUE(identity.get()->isInjective().isProvenTrue());
  EXPECT_TRUE(identity.get()->isBijective().isProvenTrue());
  EXPECT_TRUE(permutation.get()->isBijective().isProvenTrue());
  EXPECT_TRUE(broadcast.get()->isFunctional().isProvenTrue());
  IndexRelationQueryResult broadcastInjective = broadcast.get()->isInjective();
  ASSERT_EQ(broadcastInjective.status, IndexRelationStatus::Exact);
  EXPECT_EQ(broadcastInjective.value, false);

  IndexRelationResult inverseBroadcast = broadcast.get()->inverse();
  ASSERT_TRUE(inverseBroadcast.isExact());
  IndexRelationQueryResult inverseFunctional =
      inverseBroadcast.get()->isFunctional();
  ASSERT_EQ(inverseFunctional.status, IndexRelationStatus::Exact);
  EXPECT_EQ(inverseFunctional.value, false);

  IndexRelationResult full = IndexRelation::identity({4});
  IndexRelationResult prefix = IndexRelation::staticSlice({2}, {4}, {0}, {1});
  ASSERT_TRUE(full.isExact());
  ASSERT_TRUE(prefix.isExact());
  EXPECT_TRUE(full.get()->isEquivalentTo(*full.get()).isProvenTrue());
  EXPECT_TRUE(prefix.get()->implies(*full.get()).isProvenTrue());
  IndexRelationQueryResult reverseImplication =
      full.get()->implies(*prefix.get());
  ASSERT_EQ(reverseImplication.status, IndexRelationStatus::Exact);
  EXPECT_EQ(reverseImplication.value, false);
}

TEST(IndexRelationTest, RepresentsSegmentedConcatPiecesExactly) {
  IndexRelationResult first = IndexRelation::staticConcatPiece(
      /*destinationShape=*/{2, 5}, /*sourceShape=*/{2, 2}, /*axis=*/1,
      /*destinationOffset=*/0);
  IndexRelationResult second = IndexRelation::staticConcatPiece(
      /*destinationShape=*/{2, 5}, /*sourceShape=*/{2, 3}, /*axis=*/1,
      /*destinationOffset=*/2);
  ASSERT_TRUE(first.isExact());
  ASSERT_TRUE(second.isExact());
  EXPECT_TRUE(first.get()->contains({1, 1}, {1, 1}));
  EXPECT_FALSE(first.get()->contains({1, 2}, {1, 0}));
  EXPECT_TRUE(second.get()->contains({1, 2}, {1, 0}));
  EXPECT_TRUE(second.get()->contains({0, 4}, {0, 2}));
  EXPECT_FALSE(second.get()->contains({0, 1}, {0, 0}));

  EXPECT_EQ(IndexRelation::staticConcatPiece({2, 5}, {2, 3}, 1, 3).status,
            IndexRelationStatus::Invalid);
  EXPECT_EQ(IndexRelation::staticConcatPiece({2, 5}, {3, 3}, 1, 2).status,
            IndexRelationStatus::Invalid);
}

TEST(IndexRelationTest, ProvesCurrentViewDmaGatherScatterAndStagedRoutes) {
  mlir::DialectRegistry registry;
  registry.insert<wafer::WaferDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();
  mlir::Type f16 = mlir::Float16Type::get(&context);
  auto ddrTensor = wafer::MemoryAttr::get(&context, wafer::MemorySpace::DDR,
                                          wafer::MemLayout::Tensor);
  auto spmTensor = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                          wafer::MemLayout::Tensor);
  auto spmCx = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                      wafer::MemLayout::Cx);
  auto makeType = [&](llvm::ArrayRef<int64_t> shape, wafer::MemoryAttr memory) {
    return mlir::MemRefType::get(shape, f16, mlir::MemRefLayoutAttrInterface{},
                                 memory);
  };

  mlir::MemRefType boundary = makeType({2, 65}, ddrTensor);
  mlir::MemRefType temporary = makeType({2, 65}, spmTensor);
  mlir::MemRefType encoded = makeType({2, 65}, spmCx);
  IndexRelationResult identity = IndexRelation::identity({2, 65});
  ASSERT_TRUE(identity.isExact());
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveCompactDma(
      boundary, temporary, *identity.get())));
  EXPECT_TRUE(mlir::failed(TransferRealizability::proveCompactDma(
      boundary, encoded, *identity.get())));
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveGatherScatter(
      temporary, encoded, *identity.get())));
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveStagedMovement(
      boundary, temporary, encoded, *identity.get(), *identity.get())));

  mlir::MemRefType largeBoundary = makeType({16, 4096}, ddrTensor);
  mlir::MemRefType largeTemporary = makeType({16, 4096}, spmTensor);
  mlir::MemRefType largeEncoded = makeType({16, 4096}, spmCx);
  IndexRelationResult largeIdentity = IndexRelation::identity({16, 4096});
  ASSERT_TRUE(largeIdentity.isExact());
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveStagedMovement(
      largeBoundary, largeTemporary, largeEncoded, *largeIdentity.get(),
      *largeIdentity.get())));

  auto stridedLayout = mlir::StridedLayoutAttr::get(&context, 0, {4096, 1});
  auto stridedBoundary =
      mlir::MemRefType::get({16, 2048}, f16, stridedLayout, ddrTensor);
  mlir::MemRefType tiledTemporary = makeType({16, 2048}, spmTensor);
  IndexRelationResult tiledIdentity = IndexRelation::identity({16, 2048});
  ASSERT_TRUE(tiledIdentity.isExact());
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveCompactDma(
      stridedBoundary, tiledTemporary, *tiledIdentity.get())));

  mlir::MemRefType compactReshape = makeType({5, 26}, spmTensor);
  mlir::MemRefType encodedReshape = makeType({5, 26}, spmCx);
  IndexRelationResult reshape = IndexRelation::staticReshape({5, 26}, {2, 65});
  ASSERT_TRUE(reshape.isExact());
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveMetadataView(
      temporary, compactReshape, *reshape.get(),
      /*destinationMayWrite=*/true)));
  EXPECT_TRUE(mlir::failed(TransferRealizability::proveMetadataView(
      encoded, encodedReshape, *reshape.get(),
      /*destinationMayWrite=*/true)));
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveGatherScatter(
      encoded, encodedReshape, *reshape.get())));

  auto transposedStrides = mlir::StridedLayoutAttr::get(&context, 0, {1, 2});
  mlir::MemRefType compact2x2 = makeType({2, 2}, spmTensor);
  mlir::MemRefType transposed2x2 =
      mlir::MemRefType::get({2, 2}, f16, transposedStrides, spmTensor);
  IndexRelationResult identity2x2 = IndexRelation::identity({2, 2});
  ASSERT_TRUE(identity2x2.isExact());
  EXPECT_TRUE(mlir::failed(TransferRealizability::proveMetadataView(
      compact2x2, transposed2x2, *identity2x2.get(),
      /*destinationMayWrite=*/false)));

  auto offsetLayout = mlir::StridedLayoutAttr::get(&context, 1, {2, 1});
  mlir::MemRefType offset2x2 =
      mlir::MemRefType::get({2, 2}, f16, offsetLayout, spmTensor);
  EXPECT_TRUE(mlir::failed(TransferRealizability::proveMetadataView(
      compact2x2, offset2x2, *identity2x2.get(),
      /*destinationMayWrite=*/false)));

  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  IndexRelationResult permutation = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1, d0}, &context), {3, 2}, {2, 3});
  ASSERT_TRUE(permutation.isExact());
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveGatherScatter(
      makeType({2, 3}, spmTensor), makeType({3, 2}, spmTensor),
      *permutation.get())));
  IndexRelationResult permutationDest = IndexRelation::identity({3, 2});
  ASSERT_TRUE(permutationDest.isExact());
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveMappedTransfer(
      makeType({2, 3}, spmTensor), makeType({3, 2}, spmTensor), {3, 2},
      *permutation.get(), *permutationDest.get())));
  EXPECT_TRUE(mlir::failed(TransferRealizability::proveMetadataView(
      makeType({2, 3}, spmTensor), makeType({3, 2}, spmTensor),
      *permutation.get(), /*destinationMayWrite=*/true)));
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveGatherScatter(
      makeType({2, 3}, spmTensor), makeType({3, 2}, spmTensor),
      *permutation.get())));

  IndexRelationResult broadcast = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1}, &context), {4, 3}, {3});
  ASSERT_TRUE(broadcast.isExact());
  IndexRelationResult broadcastDest = IndexRelation::identity({4, 3});
  ASSERT_TRUE(broadcastDest.isExact());
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveMappedTransfer(
      makeType({3}, spmTensor), makeType({4, 3}, spmTensor), {4, 3},
      *broadcast.get(), *broadcastDest.get())));
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveGatherScatter(
      makeType({3}, spmTensor), makeType({4, 3}, spmTensor),
      *broadcast.get())));
  EXPECT_TRUE(mlir::failed(TransferRealizability::proveMetadataView(
      makeType({3}, spmTensor), makeType({4, 3}, spmTensor), *broadcast.get(),
      /*destinationMayWrite=*/true)));
}

TEST(PhysicalAccessRelationTest,
     ProvesBlockedReshapeEquivalenceWithoutElementEnumeration) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();
  mlir::Type f16 = mlir::Float16Type::get(&context);
  auto cx = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                   wafer::MemLayout::Cx);
  auto ncx = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                    wafer::MemLayout::NCx);
  auto makeType = [&](llvm::ArrayRef<int64_t> shape, wafer::MemoryAttr memory) {
    return mlir::MemRefType::get(shape, f16, mlir::MemRefLayoutAttrInterface{},
                                 memory);
  };

  EXPECT_TRUE(
      mlir::succeeded(TransferRealizability::proveStaticReshapeMetadataView(
          makeType({2, 64}, cx), makeType({1, 2, 64}, cx),
          /*destinationMayWrite=*/true)));
  EXPECT_TRUE(
      mlir::succeeded(TransferRealizability::proveStaticReshapeMetadataView(
          makeType({2, 3, 64}, ncx), makeType({2, 1, 3, 64}, ncx),
          /*destinationMayWrite=*/true)));
  EXPECT_TRUE(
      mlir::failed(TransferRealizability::proveStaticReshapeMetadataView(
          makeType({2, 65}, cx), makeType({5, 26}, cx),
          /*destinationMayWrite=*/true)));

  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  IndexRelationResult largeTranspose = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1, d0}, &context), {4096, 4096},
      {4096, 4096});
  ASSERT_TRUE(largeTranspose.isExact());
  EXPECT_TRUE(mlir::succeeded(TransferRealizability::proveGatherScatter(
      makeType({4096, 4096}, cx), makeType({4096, 4096}, cx),
      *largeTranspose.get())));
}

TEST(IndexRelationTest, ResolvesMixedSliceOperandsWithValueBounds) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  mlir::Value one =
      builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
  mlir::Value two =
      builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 2);

  IndexRelationResult relation = IndexRelation::slice(
      /*destinationShape=*/{2, 2}, /*sourceShape=*/{5, 5},
      /*offsets=*/{mlir::OpFoldResult(one), mlir::OpFoldResult(one)},
      /*strides=*/{mlir::OpFoldResult(two), mlir::OpFoldResult(two)});
  ASSERT_TRUE(relation.isExact());
  EXPECT_TRUE(relation.get()->contains({1, 1}, {3, 3}));
}

TEST(IndexRelationTest, DistinguishesBoundUnsupportedInvalidAndBudgetFailure) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineMap identity = mlir::AffineMap::get(1, 0, {d0}, &context);
  IndexRelationResult bound = IndexRelation::fromAffineMap(
      identity, /*destinationShape=*/{mlir::ShapedType::kDynamic},
      /*sourceShape=*/{mlir::ShapedType::kDynamic});
  EXPECT_EQ(bound.status, IndexRelationStatus::SoundBound);
  ASSERT_TRUE(bound.get());
  IndexRelationResult composedBound = bound.get()->compose(*bound.get());
  EXPECT_EQ(composedBound.status, IndexRelationStatus::SoundBound);
  EXPECT_FALSE(composedBound.isExact());
  EXPECT_EQ(bound.get()->isFunctional().status,
            IndexRelationStatus::SoundBound);

  mlir::AffineExpr symbol = mlir::getAffineSymbolExpr(0, &context);
  IndexRelationResult unsupported = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(1, 1, {d0 + symbol}, &context), {4}, {4});
  EXPECT_EQ(unsupported.status, IndexRelationStatus::Unsupported);
  EXPECT_FALSE(unsupported.get());

  IndexRelationResult invalid = IndexRelation::staticReshape({2, 3}, {5});
  EXPECT_EQ(invalid.status, IndexRelationStatus::Invalid);
  IndexRelationResult overflow =
      IndexRelation::staticReshape({std::numeric_limits<int64_t>::max(), 2},
                                   {std::numeric_limits<int64_t>::max(), 2});
  EXPECT_EQ(overflow.status, IndexRelationStatus::Invalid);

  IndexRelationResult exhausted =
      IndexRelation::fromAffineMap(identity, {4}, {4},
                                   IndexRelationLimits{/*maxVariables=*/1,
                                                       /*maxDisjuncts=*/1});
  EXPECT_EQ(exhausted.status, IndexRelationStatus::ResourceExhausted);

  IndexSetResult exhaustedDomain = IndexRelation::staticDomain(
      {4, 4}, IndexRelationLimits{/*maxVariables=*/1, /*maxDisjuncts=*/1});
  EXPECT_EQ(exhaustedDomain.status, IndexRelationStatus::ResourceExhausted);

  IndexRelationResult exactIdentity = IndexRelation::identity({4});
  ASSERT_TRUE(exactIdentity.isExact());
  EXPECT_EQ(exactIdentity.get()
                ->isEquivalentTo(*exactIdentity.get(),
                                 IndexRelationLimits{/*maxVariables=*/1,
                                                     /*maxDisjuncts=*/1})
                .status,
            IndexRelationStatus::ResourceExhausted);
}

TEST(IndexRelationTest, RepresentsStaticInsertSliceExactly) {
  IndexRelationResult relation = IndexRelation::staticInsertSlice(
      /*destinationShape=*/{8, 4}, /*sourceShape=*/{3, 2},
      /*offsets=*/{2, 1});
  ASSERT_TRUE(relation.isExact());
  EXPECT_TRUE(relation.get()->contains({3, 2}, {1, 1}));
  EXPECT_TRUE(relation.get()->contains({2, 1}, {0, 0}));
  EXPECT_TRUE(relation.get()->contains({4, 2}, {2, 1}));
  EXPECT_FALSE(relation.get()->contains({1, 1}, {0, 0}));
  EXPECT_FALSE(relation.get()->contains({8, 1}, {0, 0}));
  EXPECT_FALSE(relation.get()->contains({5, 1}, {2, 0}));

  // Out-of-domain pieces are invalid, never silently clamped.
  EXPECT_EQ(IndexRelation::staticInsertSlice({8, 4}, {3, 2}, {6, 1}).status,
            IndexRelationStatus::Invalid);
  EXPECT_EQ(IndexRelation::staticInsertSlice({8, 4}, {3, 2}, {2, 3}).status,
            IndexRelationStatus::Invalid);
  // Negative offsets and rank mismatches are invalid.
  EXPECT_EQ(IndexRelation::staticInsertSlice({8, 4}, {3, 2}, {-1, 1}).status,
            IndexRelationStatus::Invalid);
  EXPECT_EQ(IndexRelation::staticInsertSlice({8}, {3, 2}, {0, 0}).status,
            IndexRelationStatus::Invalid);
}

TEST(IndexRelationTest,
     GenericRectangleRecoveryRejectsOverBudgetStructureBeforeSolverWork) {
  IndexSetResult domain = IndexRelation::staticDomain({4});
  ASSERT_TRUE(domain.isExact());

  IndexRelationLimits constraintLimit;
  constraintLimit.maxConstraintsPerDisjunct = 1;
  EXPECT_EQ(domain.getExactStaticRectangularDomain(constraintLimit).status,
            IndexRelationStatus::ResourceExhausted);

  IndexRelationLimits coefficientLimit;
  coefficientLimit.maxAbsoluteCoefficient = 2;
  EXPECT_EQ(domain.getExactStaticRectangularDomain(coefficientLimit).status,
            IndexRelationStatus::ResourceExhausted);
}

} // namespace
