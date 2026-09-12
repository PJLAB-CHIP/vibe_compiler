//===- FormalOperations.cpp - Checked formal operation construction -----===//

#include "Wafer/Simulator/Reference/FormalOperations.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace wafer {

namespace {

bool isKnownRoundingMode(TargetRoundingMode mode) {
  return llvm::is_contained(getTargetRoundingModes(), mode);
}

bool isKnownElementwiseOperation(TargetElementwiseOperation operation) {
  return llvm::is_contained(getTargetElementwiseOperations(), operation);
}

bool isKnownReduceOperation(TargetReduceOperation operation) {
  return llvm::is_contained(getTargetReduceOperations(), operation);
}

bool isKnownReduceDimension(TargetReduceDimension dimension) {
  switch (dimension) {
  case TargetReduceDimension::Trailing0:
  case TargetReduceDimension::Trailing1:
  case TargetReduceDimension::Trailing2:
  case TargetReduceDimension::Trailing3:
  case TargetReduceDimension::Trailing2And1:
  case TargetReduceDimension::Trailing2And1And0:
    return true;
  }
  return false;
}

bool isAlignedLayout(PhysicalTensorLayout layout) {
  return layout == PhysicalTensorLayout::Cx ||
         layout == PhysicalTensorLayout::NCx;
}

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

llvm::Error requireEngineFormat(TargetFormatEngine engine, LogicalFormat format,
                                llvm::StringRef role) {
  const TargetFormatEncodingRecord *record =
      findTargetFormatEncoding(engine, format);
  if (!record)
    return llvm::createStringError(
        llvm::errc::not_supported,
        "%s format '%s' is not compiler-emittable for target engine '%s'",
        role.str().c_str(), stringifyLogicalFormat(format).str().c_str(),
        stringifyTargetFormatEngine(engine).str().c_str());
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FormalGemmGeometry>
getCanonicalFormalGemmGeometry(uint64_t rank) {
  if (rank < 2)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "canonical numeric GEMM axes require rank at least 2");
  const uint64_t matrixBase = rank - 2;
  std::vector<uint64_t> batchDimensions;
  if (matrixBase > batchDimensions.max_size())
    return llvm::createStringError(
        llvm::errc::result_out_of_range,
        "canonical numeric GEMM batch-axis count exceeds the host size domain");
  batchDimensions.reserve(static_cast<size_t>(matrixBase));
  for (uint64_t dimension = 0; dimension < matrixBase; ++dimension)
    batchDimensions.push_back(dimension);
  return FormalGemmGeometry{batchDimensions, matrixBase, matrixBase + 1,
                            batchDimensions, matrixBase, matrixBase + 1,
                            batchDimensions, matrixBase, matrixBase + 1};
}

llvm::Expected<FormalConvertOperation>
createFormalConvertOperation(uint16_t opcode, PhysicalTensorDescriptor source,
                             PhysicalTensorDescriptor destination,
                             std::optional<TargetConvertParameter> parameter) {
  const TargetConvertRoute *route = findTargetConvertRoute(opcode);
  if (!route)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown CT convert opcode %u",
                                   static_cast<unsigned>(opcode));
  if (source.getFormat() != route->source ||
      destination.getFormat() != route->destination)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "CT convert route endpoints do not match source/destination tensors");
  if (source.getElementCount() != destination.getElementCount())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "CT convert source and destination must have the same static element "
        "count");
  if (destination.getElementCount() > std::numeric_limits<uint32_t>::max())
    return llvm::createStringError(
        llvm::errc::result_out_of_range,
        "CT convert destination element count must fit uint32_t");

  switch (route->parameterKind) {
  case TargetConvertParameterKind::None:
    if (parameter)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "parameterless CT convert route '%s' "
                                     "rejects every optional parameter",
                                     route->canonicalSpelling.str().c_str());
    break;
  case TargetConvertParameterKind::RoundingMode: {
    if (!parameter)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT convert route '%s' requires exactly one rounding-mode parameter",
          route->canonicalSpelling.str().c_str());
    std::optional<TargetRoundingMode> mode = parameter->getRoundingMode();
    if (!mode)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT convert route '%s' requires a rounding-mode parameter, not a "
          "zero-point parameter",
          route->canonicalSpelling.str().c_str());
    if (!isKnownRoundingMode(*mode))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT convert route '%s' carries an unknown rounding mode %u",
          route->canonicalSpelling.str().c_str(), static_cast<unsigned>(*mode));
    break;
  }
  case TargetConvertParameterKind::ZeroPoint:
    if (!parameter)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "CT convert route '%s' requires exactly "
                                     "one uint32 zero-point parameter",
                                     route->canonicalSpelling.str().c_str());
    if (!parameter->getZeroPoint())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT convert route '%s' requires a zero-point parameter, not a "
          "rounding-mode parameter",
          route->canonicalSpelling.str().c_str());
    break;
  }

  return FormalConvertOperation{opcode, std::move(source),
                                std::move(destination), parameter};
}

llvm::Expected<FormalElementwiseOperation>
createFormalElementwiseOperation(TargetElementwiseOperation operation,
                                 std::vector<PhysicalTensorDescriptor> inputs,
                                 PhysicalTensorDescriptor destination) {
  if (!isKnownElementwiseOperation(operation))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown numeric elementwise operation");
  const unsigned arity = getTargetElementwiseArity(operation);
  if (inputs.size() != arity)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "numeric elementwise operation expects %u input(s), got %zu", arity,
        inputs.size());
  if (destination.getElementCount() > std::numeric_limits<uint32_t>::max())
    return llvm::createStringError(
        llvm::errc::result_out_of_range,
        "CT elementwise destination element count must fit uint32_t");
  if (llvm::Error error =
          requireEngineFormat(TargetFormatEngine::CT, destination.getFormat(),
                              "CT elementwise destination"))
    return std::move(error);

  for (const PhysicalTensorDescriptor &input : inputs) {
    if (input.getShape() != destination.getShape() ||
        input.getElementCount() != destination.getElementCount())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT elementwise input shapes must match destination shape");
    if (llvm::Error error = requireEngineFormat(
            TargetFormatEngine::CT, input.getFormat(), "CT elementwise input"))
      return std::move(error);
  }

  if (isTargetElementwiseLogic(operation)) {
    if (destination.getFormat() != LogicalFormat::Bool ||
        std::any_of(inputs.begin(), inputs.end(),
                    [](const PhysicalTensorDescriptor &key) {
                      return key.getFormat() != LogicalFormat::Bool;
                    }))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "logical CT elementwise operations "
                                     "require BOOL inputs and destination");
  } else if (isTargetElementwiseRelation(operation)) {
    if (destination.getFormat() != LogicalFormat::Bool ||
        inputs.front().getFormat() == LogicalFormat::Bool ||
        std::any_of(inputs.begin(), inputs.end(),
                    [&](const PhysicalTensorDescriptor &key) {
                      return key.getFormat() != inputs.front().getFormat();
                    }))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "relation CT elementwise operations require matching numeric inputs "
          "and BOOL destination");
  } else if (destination.getFormat() == LogicalFormat::Bool ||
             std::any_of(inputs.begin(), inputs.end(),
                         [&](const PhysicalTensorDescriptor &key) {
                           return key.getFormat() != destination.getFormat();
                         })) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "numeric CT elementwise input and destination formats must match");
  }

  return FormalElementwiseOperation{operation, std::move(inputs),
                                    std::move(destination)};
}

llvm::Expected<FormalGemmOperation> createFormalGemmOperation(
    PhysicalTensorDescriptor lhs, PhysicalTensorDescriptor rhs,
    PhysicalTensorDescriptor destination, uint32_t m, uint32_t k, uint32_t n,
    uint32_t batchCount, FormalGemmGeometry geometry,
    TargetGemmOrientation lhsOrientation,
    TargetGemmOrientation rhsOrientation) {
  if (lhs.getFormat() != rhs.getFormat() ||
      (lhs.getFormat() != destination.getFormat() &&
       !((lhs.getFormat() == LogicalFormat::F16 ||
          lhs.getFormat() == LogicalFormat::BF16) &&
         destination.getFormat() == LogicalFormat::F32)))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM requires matching inputs and a matching or f32 destination");
  if (llvm::Error error = requireEngineFormat(TargetFormatEngine::NE,
                                              lhs.getFormat(), "NE GEMM"))
    return std::move(error);
  if (!isAlignedLayout(lhs.getLayout()) || !isAlignedLayout(rhs.getLayout()) ||
      !isAlignedLayout(destination.getLayout()))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM tensors must use cx or ncx layout markers");

  const size_t rank = lhs.getShape().size();
  if (rank < 2 || rhs.getShape().size() != rank ||
      destination.getShape().size() != rank)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM tensors must have the same rank of at least 2");
  static_assert(std::numeric_limits<size_t>::max() <=
                    std::numeric_limits<uint64_t>::max(),
                "numeric GEMM rank must convert losslessly to its axis domain");
  llvm::Expected<FormalGemmGeometry> canonicalAxes =
      getCanonicalFormalGemmGeometry(static_cast<uint64_t>(rank));
  if (!canonicalAxes)
    return canonicalAxes.takeError();
  if (!(geometry == *canonicalAxes))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM requires canonical leading batch and trailing M/K/N axes");
  if (m == 0 || k == 0 || n == 0 || batchCount == 0 ||
      m > std::numeric_limits<uint16_t>::max() ||
      k > std::numeric_limits<uint16_t>::max() ||
      n > std::numeric_limits<uint16_t>::max() ||
      batchCount > std::numeric_limits<uint16_t>::max())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM m, k, n and batch count must be positive uint16 values");
  for (const PhysicalTensorDescriptor *tensor : {&lhs, &rhs, &destination})
    if (std::any_of(tensor->getShape().begin(), tensor->getShape().end(),
                    [](uint64_t dimension) { return dimension == 0; }))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "NE GEMM tensor dimensions must be positive");

  const size_t matrixBase = rank - 2;
  uint64_t inferredBatch = 1;
  for (size_t index = 0; index < matrixBase; ++index) {
    if (lhs.getShape()[index] != rhs.getShape()[index] ||
        lhs.getShape()[index] != destination.getShape()[index])
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "NE GEMM canonical leading batch dimensions must match");
    if (!checkedMultiply(inferredBatch, lhs.getShape()[index], inferredBatch) ||
        inferredBatch > std::numeric_limits<uint16_t>::max())
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "NE GEMM canonical batch product must fit uint16_t");
  }
  size_t lhsMDim = lhsOrientation == TargetGemmOrientation::Normal
                       ? matrixBase
                       : matrixBase + 1;
  size_t lhsKDim = lhsOrientation == TargetGemmOrientation::Normal
                       ? matrixBase + 1
                       : matrixBase;
  size_t rhsKDim = rhsOrientation == TargetGemmOrientation::Normal
                       ? matrixBase
                       : matrixBase + 1;
  size_t rhsNDim = rhsOrientation == TargetGemmOrientation::Normal
                       ? matrixBase + 1
                       : matrixBase;
  if (lhs.getShape()[lhsMDim] != m || lhs.getShape()[lhsKDim] != k ||
      rhs.getShape()[rhsKDim] != k || rhs.getShape()[rhsNDim] != n ||
      destination.getShape()[matrixBase] != m ||
      destination.getShape()[matrixBase + 1] != n)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM stored operand shapes must match their explicit "
        "orientations and m/k/n");
  if (inferredBatch != batchCount)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM batch count must equal the leading batch shape product");

  return FormalGemmOperation{std::move(lhs),
                             std::move(rhs),
                             std::move(destination),
                             static_cast<uint16_t>(m),
                             static_cast<uint16_t>(k),
                             static_cast<uint16_t>(n),
                             static_cast<uint16_t>(batchCount),
                             std::move(geometry),
                             lhsOrientation,
                             rhsOrientation};
}

llvm::Expected<FormalReduceOperation> createFormalReduceOperation(
    TargetReduceOperation operation, PhysicalTensorDescriptor input,
    PhysicalTensorDescriptor destination, TargetReduceDimension dimension) {
  if (!isKnownReduceOperation(operation))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown numeric reduce operation");
  if (!isKnownReduceDimension(dimension))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown native CT reduce dimension");
  if (input.getFormat() != destination.getFormat())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce input and destination formats must match");
  if (llvm::Error error = requireEngineFormat(
          TargetFormatEngine::CT, input.getFormat(), "native CT reduce"))
    return std::move(error);

  const size_t rank = input.getShape().size();
  if (rank != 4 || destination.getShape().size() != 4)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce operands must use rank4 NHWC geometry");
  if (input.getLayout() != PhysicalTensorLayout::NCx ||
      destination.getLayout() != PhysicalTensorLayout::NCx)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce operands must use ncx layout");
  for (uint64_t value : input.getShape())
    if (value == 0 || value > std::numeric_limits<uint16_t>::max())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "native CT reduce input dimensions must be positive uint16 values");

  std::vector<size_t> reduced =
      getTargetReduceLogicalDimensions(dimension, rank);
  if (reduced.empty())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce dimension is not valid for the input rank");
  if (dimension == TargetReduceDimension::Trailing3 ||
      dimension == TargetReduceDimension::Trailing2And1And0)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce N/HWC axes have no supported target contract");
  std::vector<uint64_t> expectedShape(input.getShape().begin(),
                                      input.getShape().end());
  for (size_t index : reduced)
    expectedShape[index] = 1;
  if (destination.getShape() != llvm::ArrayRef<uint64_t>(expectedShape))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce destination must retain input rank and set reduced "
        "dimensions to one");

  return FormalReduceOperation{operation, std::move(input),
                               std::move(destination), dimension};
}

} // namespace wafer
