//===- NumericCommand.cpp - Numeric command semantics --------------------===//

#include "Wafer/Target/NumericSemantics.h"

#include "NumericSemanticsInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace wafer {
using numeric_semantics_internal::digestCanonical;
using numeric_semantics_internal::isValidDigest;

namespace {

constexpr NumericRoundingMode kRoundingModes[] = {
    NumericRoundingMode::NearestEven,    NumericRoundingMode::TowardZero,
    NumericRoundingMode::TowardPositive, NumericRoundingMode::TowardNegative,
    NumericRoundingMode::Stochastic,
};
constexpr NumericElementwiseOperation kElementwiseOperations[] = {
    NumericElementwiseOperation::Abs,
    NumericElementwiseOperation::Recip,
    NumericElementwiseOperation::Square,
    NumericElementwiseOperation::Sqrt,
    NumericElementwiseOperation::Rsqrt,
    NumericElementwiseOperation::Neg,
    NumericElementwiseOperation::Max,
    NumericElementwiseOperation::Min,
    NumericElementwiseOperation::Add,
    NumericElementwiseOperation::Sub,
    NumericElementwiseOperation::Mul,
    NumericElementwiseOperation::Div,
    NumericElementwiseOperation::Eq,
    NumericElementwiseOperation::Ne,
    NumericElementwiseOperation::Ge,
    NumericElementwiseOperation::Gt,
    NumericElementwiseOperation::Le,
    NumericElementwiseOperation::Lt,
    NumericElementwiseOperation::LogicNot,
    NumericElementwiseOperation::LogicAnd,
    NumericElementwiseOperation::LogicOr,
    NumericElementwiseOperation::LogicXor,
    NumericElementwiseOperation::Log2,
    NumericElementwiseOperation::Ln,
    NumericElementwiseOperation::Pow2,
    NumericElementwiseOperation::Exp,
    NumericElementwiseOperation::ExpLp,
    NumericElementwiseOperation::Sin,
    NumericElementwiseOperation::Cos,
    NumericElementwiseOperation::Tanh,
    NumericElementwiseOperation::Sigmoid,
    NumericElementwiseOperation::Relu,
    NumericElementwiseOperation::SatRelu,
    NumericElementwiseOperation::LeakyRelu,
    NumericElementwiseOperation::Softplus,
};
constexpr NumericReduceOperation kReduceOperations[] = {
    NumericReduceOperation::Sum,
    NumericReduceOperation::Max,
    NumericReduceOperation::Min,
    NumericReduceOperation::Avg,
};

bool isKnownLayout(NumericTensorLayout layout) {
  switch (layout) {
  case NumericTensorLayout::Tensor:
  case NumericTensorLayout::NTensor:
  case NumericTensorLayout::Cx:
  case NumericTensorLayout::NCx:
    return true;
  }
  return false;
}

bool isKnownRoundingMode(NumericRoundingMode mode) {
  switch (mode) {
  case NumericRoundingMode::NearestEven:
  case NumericRoundingMode::TowardZero:
  case NumericRoundingMode::TowardPositive:
  case NumericRoundingMode::TowardNegative:
  case NumericRoundingMode::Stochastic:
    return true;
  }
  return false;
}

bool isKnownElementwiseOperation(NumericElementwiseOperation operation) {
  return std::find(std::begin(kElementwiseOperations),
                   std::end(kElementwiseOperations),
                   operation) != std::end(kElementwiseOperations);
}

bool isKnownReduceOperation(NumericReduceOperation operation) {
  return std::find(std::begin(kReduceOperations), std::end(kReduceOperations),
                   operation) != std::end(kReduceOperations);
}

bool isKnownReduceDimension(NativeCTReduceDimension dimension) {
  switch (dimension) {
  case NativeCTReduceDimension::Trailing0:
  case NativeCTReduceDimension::Trailing1:
  case NativeCTReduceDimension::Trailing2:
  case NativeCTReduceDimension::Trailing3:
  case NativeCTReduceDimension::Trailing2And1:
  case NativeCTReduceDimension::Trailing2And1And0:
    return true;
  }
  return false;
}

bool isAlignedLayout(NumericTensorLayout layout) {
  return layout == NumericTensorLayout::Cx ||
         layout == NumericTensorLayout::NCx;
}

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

std::string makeTensorDigest(LogicalFormat format, NumericTensorLayout layout,
                             llvm::ArrayRef<uint64_t> shape,
                             uint64_t elementCount) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << "wafer-numeric-tensor-key-v1\n"
         << "format=" << stringifyLogicalFormat(format) << '\n'
         << "layout=" << stringifyNumericTensorLayout(layout) << '\n'
         << "rank=" << shape.size() << '\n';
  for (auto [index, dimension] : llvm::enumerate(shape))
    stream << "dim-" << index << '=' << dimension << '\n';
  stream << "element-count=" << elementCount << '\n';
  stream.flush();
  return digestCanonical(canonical);
}

void writeTensorDigest(llvm::raw_ostream &stream, llvm::StringRef role,
                       const NumericTensorKey &tensor) {
  stream << role << "-tensor-digest=" << tensor.getDigest() << '\n';
}

std::string makeCommandKeyDigest(TargetProfileId targetProfile,
                                 const NumericCommandPayload &payload) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << "wafer-numeric-command-key-v2\n"
         << "target=" << stringifyTargetProfileId(targetProfile) << '\n';
  std::visit(
      [&](const auto &command) {
        using Command = std::decay_t<decltype(command)>;
        if constexpr (std::is_same_v<Command, NumericCTConvertCommand>) {
          const TargetConvertRoute *route =
              findTargetConvertRoute(targetProfile, command.opcode);
          if (!route)
            llvm::report_fatal_error("validated convert key lost its route");
          stream << "family=ct-convert\n"
                 << "opcode=" << command.opcode << '\n'
                 << "route=" << route->canonicalSpelling << '\n';
          writeTensorDigest(stream, "source", command.source);
          writeTensorDigest(stream, "destination", command.destination);
          if (!command.parameter) {
            stream << "parameter=none\n";
          } else if (std::optional<NumericRoundingMode> mode =
                         command.parameter->getRoundingMode()) {
            stream << "rounding-mode=" << stringifyNumericRoundingMode(*mode)
                   << '\n';
          } else {
            stream << "zero-point=" << *command.parameter->getZeroPoint()
                   << '\n';
          }
        } else if constexpr (std::is_same_v<Command,
                                            NumericCTElementwiseCommand>) {
          stream << "family=ct-elementwise\n"
                 << "operation="
                 << stringifyNumericElementwiseOperation(command.operation)
                 << '\n'
                 << "arity=" << command.inputs.size() << '\n';
          for (auto [index, input] : llvm::enumerate(command.inputs))
            writeTensorDigest(
                stream, (llvm::Twine("input-") + llvm::Twine(index)).str(),
                input);
          writeTensorDigest(stream, "destination", command.destination);
        } else if constexpr (std::is_same_v<Command, NumericNEGemmCommand>) {
          stream << "family=ne-gemm\n";
          writeTensorDigest(stream, "lhs", command.lhs);
          writeTensorDigest(stream, "rhs", command.rhs);
          writeTensorDigest(stream, "destination", command.destination);
          stream << "m=" << command.m << '\n'
                 << "k=" << command.k << '\n'
                 << "n=" << command.n << '\n'
                 << "batch-count=" << command.batchCount << '\n';
          for (auto [index, dimension] :
               llvm::enumerate(command.axes.lhsBatchDimensions))
            stream << "lhs-batch-dim-" << index << '=' << dimension << '\n';
          stream << "lhs-m-dim=" << command.axes.lhsMDimension << '\n'
                 << "lhs-contracting-dim="
                 << command.axes.lhsContractingDimension << '\n';
          for (auto [index, dimension] :
               llvm::enumerate(command.axes.rhsBatchDimensions))
            stream << "rhs-batch-dim-" << index << '=' << dimension << '\n';
          stream << "rhs-contracting-dim="
                 << command.axes.rhsContractingDimension << '\n'
                 << "rhs-n-dim=" << command.axes.rhsNDimension << '\n';
          for (auto [index, dimension] :
               llvm::enumerate(command.axes.destinationBatchDimensions))
            stream << "destination-batch-dim-" << index << '=' << dimension
                   << '\n';
          stream << "destination-m-dim=" << command.axes.destinationMDimension
                 << '\n'
                 << "destination-n-dim=" << command.axes.destinationNDimension
                 << '\n';
        } else if constexpr (std::is_same_v<Command,
                                            NumericNativeCTReduceCommand>) {
          stream << "family=native-ct-reduce\n"
                 << "operation="
                 << stringifyNumericReduceOperation(command.operation) << '\n'
                 << "dimension="
                 << stringifyNativeCTReduceDimension(command.dimension) << '\n';
          writeTensorDigest(stream, "input", command.input);
          writeTensorDigest(stream, "destination", command.destination);
        }
      },
      payload);
  stream.flush();
  return digestCanonical(canonical);
}

llvm::Error requireEngineFormat(TargetProfileId targetProfile,
                                TargetFormatEngine engine, LogicalFormat format,
                                llvm::StringRef role) {
  const TargetFormatEncodingRecord *record =
      findTargetFormatEncoding(targetProfile, engine, format);
  if (!record || !record->isSupported())
    return llvm::createStringError(
        llvm::errc::not_supported,
        "%s format '%s' is not compiler-emittable for target engine '%s'",
        role.str().c_str(), stringifyLogicalFormat(format).str().c_str(),
        stringifyTargetFormatEngine(engine).str().c_str());
  return llvm::Error::success();
}

std::vector<size_t> getReducedDimensionsImpl(NativeCTReduceDimension dimension,
                                             size_t rank) {
  auto trailing = [&](size_t index) -> std::optional<size_t> {
    if (index >= rank)
      return std::nullopt;
    return rank - 1 - index;
  };
  std::vector<size_t> dimensions;
  switch (dimension) {
  case NativeCTReduceDimension::Trailing0:
    if (auto value = trailing(0))
      dimensions.push_back(*value);
    break;
  case NativeCTReduceDimension::Trailing1:
    if (auto value = trailing(1))
      dimensions.push_back(*value);
    break;
  case NativeCTReduceDimension::Trailing2:
    if (auto value = trailing(2))
      dimensions.push_back(*value);
    break;
  case NativeCTReduceDimension::Trailing3:
    if (auto value = trailing(3))
      dimensions.push_back(*value);
    break;
  case NativeCTReduceDimension::Trailing2And1: {
    std::optional<size_t> first = trailing(2);
    std::optional<size_t> second = trailing(1);
    if (first && second)
      dimensions = {*first, *second};
    break;
  }
  case NativeCTReduceDimension::Trailing2And1And0: {
    std::optional<size_t> first = trailing(2);
    std::optional<size_t> second = trailing(1);
    std::optional<size_t> third = trailing(0);
    if (first && second && third)
      dimensions = {*first, *second, *third};
    break;
  }
  }
  return dimensions;
}

} // namespace

std::vector<size_t>
getNativeCTReduceLogicalDimensions(NativeCTReduceDimension dimension,
                                   size_t rank) {
  return getReducedDimensionsImpl(dimension, rank);
}

llvm::StringRef stringifyNumericTensorLayout(NumericTensorLayout layout) {
  switch (layout) {
  case NumericTensorLayout::Tensor:
    return "tensor";
  case NumericTensorLayout::NTensor:
    return "ntensor";
  case NumericTensorLayout::Cx:
    return "cx";
  case NumericTensorLayout::NCx:
    return "ncx";
  }
  llvm_unreachable("numeric tensor layout is not registered");
}

llvm::Expected<NumericTensorKey>
NumericTensorKey::create(LogicalFormat format, NumericTensorLayout layout,
                         std::vector<uint64_t> shape) {
  if (!findLogicalFormatDescriptor(format))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "numeric tensor has an unknown format");
  if (!isKnownLayout(layout))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "numeric tensor has an unknown layout");
  uint64_t elementCount = 1;
  for (uint64_t dimension : shape) {
    if (dimension > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "numeric tensor dimension must fit the static IR dimension domain");
    if (!checkedMultiply(elementCount, dimension, elementCount))
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "numeric tensor static element count overflows uint64_t");
  }
  std::string digest = makeTensorDigest(format, layout, shape, elementCount);
  if (!isValidDigest(digest))
    llvm::report_fatal_error("invalid numeric tensor digest");
  return NumericTensorKey(format, layout, std::move(shape), elementCount,
                          std::move(digest));
}

llvm::ArrayRef<NumericRoundingMode> getNumericRoundingModes() {
  return kRoundingModes;
}

llvm::Expected<NumericRoundingMode> parseNumericRoundingMode(uint8_t value) {
  if (value <= static_cast<uint8_t>(NumericRoundingMode::Stochastic))
    return static_cast<NumericRoundingMode>(value);
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown numeric rounding mode %u",
                                 static_cast<unsigned>(value));
}

llvm::StringRef stringifyNumericRoundingMode(NumericRoundingMode mode) {
  switch (mode) {
  case NumericRoundingMode::NearestEven:
    return "nearest-even";
  case NumericRoundingMode::TowardZero:
    return "toward-zero";
  case NumericRoundingMode::TowardPositive:
    return "toward-positive-infinity";
  case NumericRoundingMode::TowardNegative:
    return "toward-negative-infinity";
  case NumericRoundingMode::Stochastic:
    return "stochastic";
  }
  llvm_unreachable("numeric rounding mode is not registered");
}

std::optional<NumericRoundingMode>
NumericConvertParameter::getRoundingMode() const {
  if (kind != Kind::RoundingMode)
    return std::nullopt;
  return static_cast<NumericRoundingMode>(payload);
}

std::optional<uint32_t> NumericConvertParameter::getZeroPoint() const {
  if (kind != Kind::ZeroPoint)
    return std::nullopt;
  return payload;
}

llvm::ArrayRef<NumericElementwiseOperation> getNumericElementwiseOperations() {
  return kElementwiseOperations;
}

llvm::StringRef
stringifyNumericElementwiseOperation(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::Abs:
    return "abs";
  case NumericElementwiseOperation::Recip:
    return "recip";
  case NumericElementwiseOperation::Square:
    return "square";
  case NumericElementwiseOperation::Sqrt:
    return "sqrt";
  case NumericElementwiseOperation::Rsqrt:
    return "rsqrt";
  case NumericElementwiseOperation::Neg:
    return "neg";
  case NumericElementwiseOperation::Max:
    return "max";
  case NumericElementwiseOperation::Min:
    return "min";
  case NumericElementwiseOperation::Add:
    return "add";
  case NumericElementwiseOperation::Sub:
    return "sub";
  case NumericElementwiseOperation::Mul:
    return "mul";
  case NumericElementwiseOperation::Div:
    return "div";
  case NumericElementwiseOperation::Eq:
    return "eq";
  case NumericElementwiseOperation::Ne:
    return "ne";
  case NumericElementwiseOperation::Ge:
    return "ge";
  case NumericElementwiseOperation::Gt:
    return "gt";
  case NumericElementwiseOperation::Le:
    return "le";
  case NumericElementwiseOperation::Lt:
    return "lt";
  case NumericElementwiseOperation::LogicNot:
    return "logic_not";
  case NumericElementwiseOperation::LogicAnd:
    return "logic_and";
  case NumericElementwiseOperation::LogicOr:
    return "logic_or";
  case NumericElementwiseOperation::LogicXor:
    return "logic_xor";
  case NumericElementwiseOperation::Log2:
    return "log2";
  case NumericElementwiseOperation::Ln:
    return "ln";
  case NumericElementwiseOperation::Pow2:
    return "pow2";
  case NumericElementwiseOperation::Exp:
    return "exp";
  case NumericElementwiseOperation::ExpLp:
    return "exp_lp";
  case NumericElementwiseOperation::Sin:
    return "sin";
  case NumericElementwiseOperation::Cos:
    return "cos";
  case NumericElementwiseOperation::Tanh:
    return "tanh";
  case NumericElementwiseOperation::Sigmoid:
    return "sigmoid";
  case NumericElementwiseOperation::Relu:
    return "relu";
  case NumericElementwiseOperation::SatRelu:
    return "satrelu";
  case NumericElementwiseOperation::LeakyRelu:
    return "leakyrelu";
  case NumericElementwiseOperation::Softplus:
    return "softplus";
  }
  llvm_unreachable("numeric elementwise operation is not registered");
}

unsigned getNumericElementwiseArity(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::Abs:
  case NumericElementwiseOperation::Recip:
  case NumericElementwiseOperation::Square:
  case NumericElementwiseOperation::Sqrt:
  case NumericElementwiseOperation::Rsqrt:
  case NumericElementwiseOperation::Neg:
  case NumericElementwiseOperation::LogicNot:
  case NumericElementwiseOperation::Log2:
  case NumericElementwiseOperation::Ln:
  case NumericElementwiseOperation::Pow2:
  case NumericElementwiseOperation::Exp:
  case NumericElementwiseOperation::ExpLp:
  case NumericElementwiseOperation::Sin:
  case NumericElementwiseOperation::Cos:
  case NumericElementwiseOperation::Tanh:
  case NumericElementwiseOperation::Sigmoid:
  case NumericElementwiseOperation::Relu:
  case NumericElementwiseOperation::SatRelu:
  case NumericElementwiseOperation::LeakyRelu:
  case NumericElementwiseOperation::Softplus:
    return 1;
  case NumericElementwiseOperation::Max:
  case NumericElementwiseOperation::Min:
  case NumericElementwiseOperation::Add:
  case NumericElementwiseOperation::Sub:
  case NumericElementwiseOperation::Mul:
  case NumericElementwiseOperation::Div:
  case NumericElementwiseOperation::Eq:
  case NumericElementwiseOperation::Ne:
  case NumericElementwiseOperation::Ge:
  case NumericElementwiseOperation::Gt:
  case NumericElementwiseOperation::Le:
  case NumericElementwiseOperation::Lt:
  case NumericElementwiseOperation::LogicAnd:
  case NumericElementwiseOperation::LogicOr:
  case NumericElementwiseOperation::LogicXor:
    return 2;
  }
  llvm_unreachable("numeric elementwise operation is not registered");
}

bool isNumericElementwiseRelation(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::Eq:
  case NumericElementwiseOperation::Ne:
  case NumericElementwiseOperation::Ge:
  case NumericElementwiseOperation::Gt:
  case NumericElementwiseOperation::Le:
  case NumericElementwiseOperation::Lt:
    return true;
  default:
    return false;
  }
}

bool isNumericElementwiseLogic(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::LogicNot:
  case NumericElementwiseOperation::LogicAnd:
  case NumericElementwiseOperation::LogicOr:
  case NumericElementwiseOperation::LogicXor:
    return true;
  default:
    return false;
  }
}

llvm::ArrayRef<NumericReduceOperation> getNumericReduceOperations() {
  return kReduceOperations;
}

llvm::StringRef stringifyNumericReduceOperation(NumericReduceOperation kind) {
  switch (kind) {
  case NumericReduceOperation::Sum:
    return "sum";
  case NumericReduceOperation::Max:
    return "max";
  case NumericReduceOperation::Min:
    return "min";
  case NumericReduceOperation::Avg:
    return "avg";
  }
  llvm_unreachable("numeric reduce operation is not registered");
}

llvm::StringRef
stringifyNativeCTReduceDimension(NativeCTReduceDimension dimension) {
  switch (dimension) {
  case NativeCTReduceDimension::Trailing0:
    return "trailing-0";
  case NativeCTReduceDimension::Trailing1:
    return "trailing-1";
  case NativeCTReduceDimension::Trailing2:
    return "trailing-2";
  case NativeCTReduceDimension::Trailing3:
    return "trailing-3";
  case NativeCTReduceDimension::Trailing2And1:
    return "trailing-2-and-1";
  case NativeCTReduceDimension::Trailing2And1And0:
    return "trailing-2-and-1-and-0";
  }
  llvm_unreachable("native CT reduce dimension is not registered");
}

llvm::StringRef stringifyNumericCommandFamily(NumericCommandFamily family) {
  switch (family) {
  case NumericCommandFamily::CTConvert:
    return "ct-convert";
  case NumericCommandFamily::CTElementwise:
    return "ct-elementwise";
  case NumericCommandFamily::NEGemm:
    return "ne-gemm";
  case NumericCommandFamily::NativeCTReduce:
    return "native-ct-reduce";
  }
  llvm_unreachable("numeric command family is not registered");
}

llvm::Expected<NumericGemmAxes> getCanonicalNumericGemmAxes(uint64_t rank) {
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
  return NumericGemmAxes{batchDimensions, matrixBase, matrixBase + 1,
                         batchDimensions, matrixBase, matrixBase + 1,
                         batchDimensions, matrixBase, matrixBase + 1};
}

NumericCommandFamily NumericCommandKey::getFamily() const {
  if (std::holds_alternative<NumericCTConvertCommand>(payload))
    return NumericCommandFamily::CTConvert;
  if (std::holds_alternative<NumericCTElementwiseCommand>(payload))
    return NumericCommandFamily::CTElementwise;
  if (std::holds_alternative<NumericNEGemmCommand>(payload))
    return NumericCommandFamily::NEGemm;
  return NumericCommandFamily::NativeCTReduce;
}

const NumericCTConvertCommand *NumericCommandKey::getCTConvert() const {
  return std::get_if<NumericCTConvertCommand>(&payload);
}
const NumericCTElementwiseCommand *NumericCommandKey::getCTElementwise() const {
  return std::get_if<NumericCTElementwiseCommand>(&payload);
}
const NumericNEGemmCommand *NumericCommandKey::getNEGemm() const {
  return std::get_if<NumericNEGemmCommand>(&payload);
}
const NumericNativeCTReduceCommand *
NumericCommandKey::getNativeCTReduce() const {
  return std::get_if<NumericNativeCTReduceCommand>(&payload);
}

llvm::Expected<NumericCommandKey> NumericCommandKey::createCTConvert(
    TargetProfileId targetProfile, uint16_t opcode, NumericTensorKey source,
    NumericTensorKey destination,
    std::optional<NumericConvertParameter> parameter) {
  const TargetConvertRoute *route =
      findTargetConvertRoute(targetProfile, opcode);
  if (!route)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "unknown CT convert route for target profile '%s' and opcode %u",
        stringifyTargetProfileId(targetProfile).str().c_str(),
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
    std::optional<NumericRoundingMode> mode = parameter->getRoundingMode();
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

  NumericCommandPayload payload = NumericCTConvertCommand{
      opcode, std::move(source), std::move(destination), parameter};
  std::string digest = makeCommandKeyDigest(targetProfile, payload);
  return NumericCommandKey(targetProfile, std::move(payload),
                           std::move(digest));
}

llvm::Expected<NumericCommandKey> NumericCommandKey::createCTElementwise(
    TargetProfileId targetProfile, NumericElementwiseOperation operation,
    std::vector<NumericTensorKey> inputs, NumericTensorKey destination) {
  if (!isKnownElementwiseOperation(operation))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown numeric elementwise operation");
  const unsigned arity = getNumericElementwiseArity(operation);
  if (inputs.size() != arity)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "numeric elementwise operation expects %u input(s), got %zu", arity,
        inputs.size());
  if (destination.getElementCount() > std::numeric_limits<uint32_t>::max())
    return llvm::createStringError(
        llvm::errc::result_out_of_range,
        "CT elementwise destination element count must fit uint32_t");
  if (llvm::Error error = requireEngineFormat(
          targetProfile, TargetFormatEngine::CT, destination.getFormat(),
          "CT elementwise destination"))
    return std::move(error);

  for (const NumericTensorKey &input : inputs) {
    if (input.getShape() != destination.getShape() ||
        input.getElementCount() != destination.getElementCount())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT elementwise input shapes must match destination shape");
    if (llvm::Error error =
            requireEngineFormat(targetProfile, TargetFormatEngine::CT,
                                input.getFormat(), "CT elementwise input"))
      return std::move(error);
  }

  if (isNumericElementwiseLogic(operation)) {
    if (destination.getFormat() != LogicalFormat::Bool ||
        std::any_of(inputs.begin(), inputs.end(),
                    [](const NumericTensorKey &key) {
                      return key.getFormat() != LogicalFormat::Bool;
                    }))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "logical CT elementwise operations "
                                     "require BOOL inputs and destination");
  } else if (isNumericElementwiseRelation(operation)) {
    if (destination.getFormat() != LogicalFormat::Bool ||
        inputs.front().getFormat() == LogicalFormat::Bool ||
        std::any_of(inputs.begin(), inputs.end(),
                    [&](const NumericTensorKey &key) {
                      return key.getFormat() != inputs.front().getFormat();
                    }))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "relation CT elementwise operations require matching numeric inputs "
          "and BOOL destination");
  } else if (destination.getFormat() == LogicalFormat::Bool ||
             std::any_of(inputs.begin(), inputs.end(),
                         [&](const NumericTensorKey &key) {
                           return key.getFormat() != destination.getFormat();
                         })) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "numeric CT elementwise input and destination formats must match");
  }

  NumericCommandPayload payload = NumericCTElementwiseCommand{
      operation, std::move(inputs), std::move(destination)};
  std::string digest = makeCommandKeyDigest(targetProfile, payload);
  return NumericCommandKey(targetProfile, std::move(payload),
                           std::move(digest));
}

llvm::Expected<NumericCommandKey> NumericCommandKey::createNEGemm(
    TargetProfileId targetProfile, NumericTensorKey lhs, NumericTensorKey rhs,
    NumericTensorKey destination, uint32_t m, uint32_t k, uint32_t n,
    uint32_t batchCount, NumericGemmAxes axes) {
  if (lhs.getFormat() != rhs.getFormat() ||
      lhs.getFormat() != destination.getFormat())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM lhs, rhs and destination formats must match");
  if (llvm::Error error = requireEngineFormat(
          targetProfile, TargetFormatEngine::NE, lhs.getFormat(), "NE GEMM"))
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
  llvm::Expected<NumericGemmAxes> canonicalAxes =
      getCanonicalNumericGemmAxes(static_cast<uint64_t>(rank));
  if (!canonicalAxes)
    return canonicalAxes.takeError();
  if (!(axes == *canonicalAxes))
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
  for (const NumericTensorKey *tensor : {&lhs, &rhs, &destination})
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
  if (lhs.getShape()[matrixBase] != m || lhs.getShape()[matrixBase + 1] != k ||
      rhs.getShape()[matrixBase] != k || rhs.getShape()[matrixBase + 1] != n ||
      destination.getShape()[matrixBase] != m ||
      destination.getShape()[matrixBase + 1] != n)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM requires canonical trailing lhs[M,K], rhs[K,N], dest[M,N] "
        "axes matching m/k/n");
  if (inferredBatch != batchCount)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM batch count must equal the leading batch shape product");

  NumericCommandPayload payload =
      NumericNEGemmCommand{std::move(lhs),
                           std::move(rhs),
                           std::move(destination),
                           static_cast<uint16_t>(m),
                           static_cast<uint16_t>(k),
                           static_cast<uint16_t>(n),
                           static_cast<uint16_t>(batchCount),
                           std::move(axes)};
  std::string digest = makeCommandKeyDigest(targetProfile, payload);
  return NumericCommandKey(targetProfile, std::move(payload),
                           std::move(digest));
}

llvm::Expected<NumericCommandKey> NumericCommandKey::createNativeCTReduce(
    TargetProfileId targetProfile, NumericReduceOperation operation,
    NumericTensorKey input, NumericTensorKey destination,
    NativeCTReduceDimension dimension) {
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
  if (input.getFormat() == LogicalFormat::Bool)
    return llvm::createStringError(
        llvm::errc::not_supported,
        "native CT reduce does not accept the BOOL-specific CT format row");
  if (llvm::Error error =
          requireEngineFormat(targetProfile, TargetFormatEngine::CT,
                              input.getFormat(), "native CT reduce"))
    return std::move(error);

  const size_t rank = input.getShape().size();
  if (rank < 1 || rank > 4)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce input rank must be in [1, 4]");
  NumericTensorLayout expectedInputLayout =
      rank > 2 ? NumericTensorLayout::NCx : NumericTensorLayout::Cx;
  NumericTensorLayout expectedDestinationLayout =
      destination.getShape().size() > 2 ? NumericTensorLayout::NCx
                                        : NumericTensorLayout::Cx;
  if (input.getLayout() != expectedInputLayout ||
      destination.getLayout() != expectedDestinationLayout)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce rank <= 2 requires cx and rank > 2 requires ncx");
  for (uint64_t value : input.getShape())
    if (value == 0 || value > std::numeric_limits<uint16_t>::max())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "native CT reduce input dimensions must be positive uint16 values");

  std::vector<size_t> reduced =
      getNativeCTReduceLogicalDimensions(dimension, rank);
  if (reduced.empty())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce dimension is not valid for the input rank");
  std::vector<uint64_t> expectedShape;
  for (size_t index = 0; index < rank; ++index)
    if (std::find(reduced.begin(), reduced.end(), index) == reduced.end())
      expectedShape.push_back(input.getShape()[index]);
  if (destination.getShape() != llvm::ArrayRef<uint64_t>(expectedShape))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce destination shape must contain exactly the "
        "non-reduced input dimensions");

  NumericCommandPayload payload = NumericNativeCTReduceCommand{
      operation, std::move(input), std::move(destination), dimension};
  std::string digest = makeCommandKeyDigest(targetProfile, payload);
  return NumericCommandKey(targetProfile, std::move(payload),
                           std::move(digest));
}

} // namespace wafer
