//===- ReferenceProgramNumeric.cpp - Reference numeric semantics -----===//

#include "ReferenceProgramInterpreterInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace wafer::compiler::reference_detail {
namespace {

static float maximumF32(float lhs, float rhs) {
  if (std::isnan(lhs) || std::isnan(rhs))
    return std::numeric_limits<float>::quiet_NaN();
  if (lhs == 0.0f && rhs == 0.0f)
    return std::signbit(lhs) && std::signbit(rhs) ? -0.0f : 0.0f;
  return std::max(lhs, rhs);
}

static float minimumF32(float lhs, float rhs) {
  if (std::isnan(lhs) || std::isnan(rhs))
    return std::numeric_limits<float>::quiet_NaN();
  if (lhs == 0.0f && rhs == 0.0f)
    return std::signbit(lhs) || std::signbit(rhs) ? -0.0f : 0.0f;
  return std::min(lhs, rhs);
}

} // namespace

llvm::Error ProgramInterpreter::executeConvert(const Command &command) {
  auto source = lookup(command.source);
  auto dest = lookup(command.dest);
  if (!source)
    return source.takeError();
  if (!dest)
    return dest.takeError();

  std::vector<llvm::APInt> converted;
  converted.reserve(source->type.getNumElements());
  if (llvm::Error error = forEachLogicalIndex(
          source->type.getShape(),
          [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
            auto value = readNumeric(*source, index, command.sourceFormat);
            if (!value)
              return value.takeError();
            auto result =
                command.stochasticRounding
                    ? convertNumericStochastic(*value, command.destFormat,
                                               nextStochasticBits())
                    : convertNumeric(*value, command.destFormat,
                                     command.roundingMode);
            if (!result)
              return result.takeError();
            converted.push_back(std::move(*result));
            return llvm::Error::success();
          }))
    return error;

  size_t position = 0;
  if (llvm::Error error = forEachLogicalIndex(
          dest->type.getShape(),
          [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
            if (position >= converted.size())
              return invalid(
                  "projected convert destination has excess elements");
            return writeNumericBits(*dest, index, command.destFormat,
                                    converted[position++]);
          }))
    return error;
  if (position != converted.size())
    return invalid("projected convert destination has too few elements");
  return llvm::Error::success();
}

llvm::Error ProgramInterpreter::executeGemm(const Command &command) {
  auto lhs = lookup(command.lhs);
  auto rhs = lookup(command.rhs);
  auto dest = lookup(command.dest);
  if (!lhs)
    return lhs.takeError();
  if (!rhs)
    return rhs.takeError();
  if (!dest)
    return dest.takeError();
  return forEachLogicalIndex(
      command.batchShape,
      [&](llvm::ArrayRef<int64_t> batchIndex) -> llvm::Error {
        for (int64_t m = 0; m < command.m; ++m)
          for (int64_t n = 0; n < command.n; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < command.k; ++k) {
              llvm::SmallVector<int64_t> lhsIndex(batchIndex.begin(),
                                                  batchIndex.end());
              llvm::SmallVector<int64_t> rhsIndex(batchIndex.begin(),
                                                  batchIndex.end());
              lhsIndex.append({m, k});
              rhsIndex.append({k, n});
              auto lhsValue = readF32(*lhs, lhsIndex);
              auto rhsValue = readF32(*rhs, rhsIndex);
              if (!lhsValue)
                return lhsValue.takeError();
              if (!rhsValue)
                return rhsValue.takeError();
              sum += *lhsValue * *rhsValue;
            }
            llvm::SmallVector<int64_t> destIndex(batchIndex.begin(),
                                                 batchIndex.end());
            destIndex.append({m, n});
            if (llvm::Error error = writeF32(*dest, destIndex, sum))
              return error;
          }
        return llvm::Error::success();
      });
}

llvm::Error ProgramInterpreter::executeReduce(const Command &command) {
  auto input = lookup(command.source);
  auto dest = lookup(command.dest);
  if (!input)
    return input.takeError();
  if (!dest)
    return dest.takeError();

  const Scalar *init = &command.scalarValue;
  if (command.reduceInit) {
    auto found = scalars.find(*command.reduceInit);
    if (found == scalars.end())
      return invalid("reduce scalar initialization is unavailable");
    init = &found->second;
  }
  auto initValue = convertScalarToF32(*init);
  if (!initValue)
    return initValue.takeError();
  if (llvm::Error error = forEachLogicalIndex(
          dest->type.getShape(), [&](llvm::ArrayRef<int64_t> index) {
            return writeF32(*dest, index, *initValue);
          }))
    return error;

  return forEachLogicalIndex(
      input->type.getShape(), [&](llvm::ArrayRef<int64_t> inputIndex) {
        llvm::SmallVector<int64_t> destIndex;
        for (auto [dimension, value] : llvm::enumerate(inputIndex))
          if (!llvm::is_contained(command.reduceDimensions,
                                  static_cast<int64_t>(dimension)))
            destIndex.push_back(value);
        auto value = readF32(*input, inputIndex);
        auto accumulator = readF32(*dest, destIndex);
        if (!value)
          return value.takeError();
        if (!accumulator)
          return accumulator.takeError();

        float result = 0.0f;
        switch (command.reduceKind) {
        case wafer::InstrReduceKind::Sum:
          result = *accumulator + *value;
          break;
        case wafer::InstrReduceKind::Max:
          result = maximumF32(*accumulator, *value);
          break;
        case wafer::InstrReduceKind::Min:
          result = minimumF32(*accumulator, *value);
          break;
        case wafer::InstrReduceKind::Avg:
          return invalid("unsupported projected reduce kind");
        }
        return writeF32(*dest, destIndex, result);
      });
}

llvm::Error ProgramInterpreter::executeElementwise(const Command &command) {
  auto dest = lookup(command.dest);
  if (!dest)
    return dest.takeError();
  std::vector<BufferView> inputs;
  for (ValueId value : command.inputs) {
    auto input = lookup(value);
    if (!input)
      return input.takeError();
    inputs.push_back(std::move(*input));
  }
  return forEachLogicalIndex(
      dest->type.getShape(), [&](llvm::ArrayRef<int64_t> index) {
        llvm::SmallVector<float> values;
        for (const BufferView &input : inputs) {
          auto value = readF32(input, index);
          if (!value)
            return value.takeError();
          values.push_back(*value);
        }
        float result = 0.0f;
        switch (command.elementwiseKind) {
        case wafer::InstrElementwiseKind::Abs:
          result = std::fabs(values[0]);
          break;
        case wafer::InstrElementwiseKind::Recip:
          result = 1.0f / values[0];
          break;
        case wafer::InstrElementwiseKind::Square:
          result = values[0] * values[0];
          break;
        case wafer::InstrElementwiseKind::Sqrt:
          result = std::sqrt(values[0]);
          break;
        case wafer::InstrElementwiseKind::Rsqrt:
          result = 1.0f / std::sqrt(values[0]);
          break;
        case wafer::InstrElementwiseKind::Neg:
          result = -values[0];
          break;
        case wafer::InstrElementwiseKind::Max:
          result = maximumF32(values[0], values[1]);
          break;
        case wafer::InstrElementwiseKind::Min:
          result = minimumF32(values[0], values[1]);
          break;
        case wafer::InstrElementwiseKind::Add:
          result = values[0] + values[1];
          break;
        case wafer::InstrElementwiseKind::Sub:
          result = values[0] - values[1];
          break;
        case wafer::InstrElementwiseKind::Mul:
          result = values[0] * values[1];
          break;
        case wafer::InstrElementwiseKind::Div:
          result = values[0] / values[1];
          break;
        case wafer::InstrElementwiseKind::Log2:
          result = std::log2(values[0]);
          break;
        case wafer::InstrElementwiseKind::Ln:
          result = std::log(values[0]);
          break;
        case wafer::InstrElementwiseKind::Pow2:
          result = std::exp2(values[0]);
          break;
        case wafer::InstrElementwiseKind::Exp:
        case wafer::InstrElementwiseKind::ExpLp:
          result = std::exp(values[0]);
          break;
        case wafer::InstrElementwiseKind::Sin:
          result = std::sin(values[0]);
          break;
        case wafer::InstrElementwiseKind::Cos:
          result = std::cos(values[0]);
          break;
        case wafer::InstrElementwiseKind::Tanh:
          result = std::tanh(values[0]);
          break;
        case wafer::InstrElementwiseKind::Sigmoid:
          result = 1.0f / (1.0f + std::exp(-values[0]));
          break;
        case wafer::InstrElementwiseKind::Relu:
        case wafer::InstrElementwiseKind::SatRelu:
          result = std::max(0.0f, values[0]);
          break;
        case wafer::InstrElementwiseKind::Softplus:
          result = std::log1p(std::exp(values[0]));
          break;
        default:
          return invalid("unsupported projected elementwise kind");
        }
        return writeF32(*dest, index, result);
      });
}

llvm::Expected<float>
ProgramInterpreter::convertScalarToF32(const Scalar &scalar) {
  if (scalar.floating) {
    llvm::APFloat value = *scalar.floating;
    bool losesInfo = false;
    value.convert(llvm::APFloat::IEEEsingle(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return value.convertToFloat();
  }
  if (scalar.integer) {
    if (scalar.integer->getBitWidth() > 64)
      return unsupported("fill integer wider than 64 bits");
    if (scalar.integerIsUnsigned)
      return static_cast<float>(scalar.integer->getZExtValue());
    return static_cast<float>(scalar.integer->getSExtValue());
  }
  return invalid("projected scalar has no value");
}

llvm::Error ProgramInterpreter::executeFill(const Command &command) {
  auto dest = lookup(command.dest);
  if (!dest)
    return dest.takeError();
  auto scalar = scalars.find(command.scalar);
  if (scalar == scalars.end())
    return invalid("fill scalar is unavailable");
  if (dest->type.getElementType().isInteger(1)) {
    if (!scalar->second.integer || scalar->second.integer->getBitWidth() != 1)
      return invalid("i1 fill requires an i1 scalar");
    bool value = !scalar->second.integer->isZero();
    return forEachLogicalIndex(dest->type.getShape(),
                               [&](llvm::ArrayRef<int64_t> index) {
                                 return writeI1(*dest, index, value);
                               });
  }
  auto value = convertScalarToF32(scalar->second);
  if (!value)
    return value.takeError();
  return forEachLogicalIndex(dest->type.getShape(),
                             [&](llvm::ArrayRef<int64_t> index) {
                               return writeF32(*dest, index, *value);
                             });
}

llvm::Error ProgramInterpreter::executeBit2Fp(const Command &command) {
  auto source = lookup(command.source);
  auto dest = lookup(command.dest);
  if (!source)
    return source.takeError();
  if (!dest)
    return dest.takeError();
  return forEachLogicalIndex(
      dest->type.getShape(), [&](llvm::ArrayRef<int64_t> index) {
        auto value = readI1(*source, index);
        if (!value)
          return value.takeError();
        return writeF32(*dest, index, *value ? 1.0f : 0.0f);
      });
}

llvm::Error ProgramInterpreter::executeMaskMove(const Command &command) {
  auto source = lookup(command.source);
  auto mask = lookup(command.mask);
  auto dest = lookup(command.dest);
  if (!source)
    return source.takeError();
  if (!mask)
    return mask.takeError();
  if (!dest)
    return dest.takeError();
  return forEachLogicalIndex(dest->type.getShape(),
                             [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
                               auto maskValue = readF32(*mask, index);
                               if (!maskValue)
                                 return maskValue.takeError();
                               if (*maskValue == 0.0f)
                                 return llvm::Error::success();
                               auto sourceValue = readF32(*source, index);
                               if (!sourceValue)
                                 return sourceValue.takeError();
                               return writeF32(*dest, index, *sourceValue);
                             });
}

uint64_t ProgramInterpreter::nextStochasticBits() {
  uint64_t value = (stochasticState += UINT64_C(0x9e3779b97f4a7c15));
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

} // namespace wafer::compiler::reference_detail
