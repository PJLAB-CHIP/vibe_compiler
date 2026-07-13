//===- ReferenceExecutor.cpp - Accepted-rank semantic execution ----------===//

#include "Wafer/Compiler/ReferenceExecutor.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

struct ReferenceExecutionResultBuilder {
  static ReferenceExecutionResult
  make(int64_t logicalRank, std::vector<ReferenceOutputBinding> outputs) {
    return ReferenceExecutionResult(logicalRank, std::move(outputs));
  }
};

namespace {

llvm::Error invalid(llvm::StringRef message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

std::optional<int64_t> getDTypeByteWidth(llvm::StringRef dtype) {
  if (dtype == "i1")
    return std::nullopt;
  if (dtype == "i8" || dtype == "ui8")
    return 1;
  if (dtype == "i16" || dtype == "ui16" || dtype == "f16" || dtype == "bf16")
    return 2;
  if (dtype == "i32" || dtype == "ui32" || dtype == "f32" || dtype == "tf32")
    return 4;
  if (dtype == "i64" || dtype == "ui64" || dtype == "f64")
    return 8;
  return std::nullopt;
}

std::optional<int64_t> getCompactByteCount(llvm::StringRef dtype,
                                           llvm::ArrayRef<int64_t> shape) {
  std::optional<int64_t> elementBytes = getDTypeByteWidth(dtype);
  if (!elementBytes)
    return std::nullopt;
  int64_t bytes = *elementBytes;
  for (int64_t dim : shape) {
    if (dim < 0 ||
        (dim != 0 && bytes > std::numeric_limits<int64_t>::max() / dim))
      return std::nullopt;
    bytes *= dim;
  }
  return bytes;
}

std::string elementDType(mlir::Type type) {
  if (type.isF32())
    return "f32";
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return (integer.isUnsigned() ? "ui" : "i") +
           std::to_string(integer.getWidth());
  return {};
}

struct Storage {
  std::vector<uint8_t> bytes;
};

struct BufferView {
  std::shared_ptr<Storage> storage;
  int64_t base = 0;
  int64_t viewOffset = 0;
  int64_t physicalBytes = 0;
  mlir::MemRefType type;
};

template <typename Callback>
llvm::Error forEachLogicalIndex(llvm::ArrayRef<int64_t> shape,
                                Callback callback) {
  for (int64_t dim : shape)
    if (dim < 0)
      return invalid("reference executor requires static tensor shapes");
  if (llvm::is_contained(shape, int64_t{0}))
    return llvm::Error::success();
  llvm::SmallVector<int64_t> index(shape.size(), 0);
  if (shape.empty())
    return callback(index);
  while (true) {
    if (llvm::Error error = callback(index))
      return error;
    int64_t dim = static_cast<int64_t>(shape.size()) - 1;
    for (; dim >= 0; --dim) {
      if (++index[dim] < shape[dim])
        break;
      index[dim] = 0;
    }
    if (dim < 0)
      break;
  }
  return llvm::Error::success();
}

llvm::Expected<int64_t> getAbsoluteOffset(const BufferView &buffer,
                                          llvm::ArrayRef<int64_t> indices,
                                          int64_t byteWidth) {
  std::optional<int64_t> relative =
      wafer::computeWaferPhysicalElementByteOffset(buffer.type, indices);
  if (!relative)
    return invalid("cannot map logical index to accepted physical layout");
  if (*relative < 0 || byteWidth < 0 ||
      *relative > buffer.physicalBytes - byteWidth)
    return invalid("logical access exceeds accepted buffer extent");
  int64_t absolute = buffer.base + buffer.viewOffset + *relative;
  if (absolute < 0 ||
      absolute > static_cast<int64_t>(buffer.storage->bytes.size()) - byteWidth)
    return invalid("logical access exceeds reference storage arena");
  return absolute;
}

llvm::Expected<float> readF32(const BufferView &buffer,
                              llvm::ArrayRef<int64_t> indices) {
  if (!buffer.type.getElementType().isF32())
    return invalid("reference floating compute currently requires f32");
  auto offset = getAbsoluteOffset(buffer, indices, sizeof(float));
  if (!offset)
    return offset.takeError();
  float value;
  std::memcpy(&value, buffer.storage->bytes.data() + *offset, sizeof(value));
  return value;
}

llvm::Error writeF32(const BufferView &buffer, llvm::ArrayRef<int64_t> indices,
                     float value) {
  if (!buffer.type.getElementType().isF32())
    return invalid("reference floating compute currently requires f32");
  auto offset = getAbsoluteOffset(buffer, indices, sizeof(float));
  if (!offset)
    return offset.takeError();
  std::memcpy(buffer.storage->bytes.data() + *offset, &value, sizeof(value));
  return llvm::Error::success();
}

class RankInterpreter {
public:
  explicit RankInterpreter(const RankExecutable &rank) : rank(rank) {
    spmArena = std::make_shared<Storage>();
    ddrArena = std::make_shared<Storage>();
  }

  llvm::Expected<ReferenceExecutionResult>
  run(llvm::ArrayRef<ReferenceInputBinding> inputs) {
    mlir::func::FuncOp entry =
        rank.getModule().lookupSymbol<mlir::func::FuncOp>(
            rank.getEntrySymbol());
    if (!entry)
      return invalid("accepted rank entry function is missing");
    if (!entry.getBody().hasOneBlock())
      return invalid("reference executor requires a single-block entry");

    if (llvm::Error error = bindEntryArguments(entry, inputs))
      return std::move(error);
    auto returned = executeBlock(entry.getBody().front());
    if (!returned)
      return returned.takeError();

    std::vector<ReferenceOutputBinding> outputs;
    for (const RankProgramBinding &binding : rank.getProgramBindings()) {
      if (binding.role != ProgramResourceRole::Output)
        continue;
      if (binding.index < 0 ||
          binding.index >= static_cast<int64_t>(returned->size()))
        return invalid("output binding index is outside entry results");
      auto tensor = exportTensor((*returned)[binding.index], binding.dtype,
                                 binding.localShape);
      if (!tensor)
        return tensor.takeError();
      outputs.push_back({binding.index, std::move(*tensor)});
    }
    return ReferenceExecutionResultBuilder::make(rank.getLogicalRank(),
                                                 std::move(outputs));
  }

private:
  llvm::Error bindEntryArguments(mlir::func::FuncOp entry,
                                 llvm::ArrayRef<ReferenceInputBinding> inputs) {
    llvm::SmallVector<bool> used(inputs.size(), false);
    for (const RankProgramBinding &binding : rank.getProgramBindings()) {
      if (binding.role == ProgramResourceRole::Output)
        continue;
      int64_t match = -1;
      for (auto [position, input] : llvm::enumerate(inputs)) {
        if (input.role == binding.role && input.index == binding.index) {
          if (match >= 0)
            return invalid("duplicate reference input binding");
          match = static_cast<int64_t>(position);
        }
      }
      if (match < 0)
        return invalid("missing reference input binding");
      used[match] = true;
      if (inputs[match].tensor.getDType() != binding.dtype ||
          inputs[match].tensor.getShape() !=
              llvm::ArrayRef<int64_t>(binding.localShape))
        return invalid(
            "reference input tensor disagrees with typed rank binding");
      if (binding.index < 0 || binding.index >= entry.getNumArguments())
        return invalid("program input index is outside entry arguments");
      auto type = mlir::dyn_cast<mlir::MemRefType>(
          entry.getArgument(binding.index).getType());
      if (!type)
        return invalid("program input entry argument is not a memref");
      auto imported = importTensor(inputs[match].tensor, type);
      if (!imported)
        return imported.takeError();
      buffers[entry.getArgument(binding.index)] = std::move(*imported);
    }
    if (llvm::is_contained(used, false))
      return invalid("unexpected reference input binding");
    return llvm::Error::success();
  }

  llvm::Expected<BufferView> importTensor(const ReferenceTensor &tensor,
                                          mlir::MemRefType type) {
    auto info = wafer::computeWaferPhysicalTensorInfo(type);
    if (!info || info->physicalBytes < 0 || info->elementBytes <= 0)
      return invalid("input has no static accepted physical layout");
    if (elementDType(type.getElementType()) != tensor.getDType() ||
        type.getShape() != tensor.getShape())
      return invalid("input tensor type disagrees with entry memref");
    BufferView view{std::make_shared<Storage>(), 0, 0, info->physicalBytes,
                    type};
    view.storage->bytes.resize(info->physicalBytes, 0);
    int64_t linearByte = 0;
    if (llvm::Error error = forEachLogicalIndex(
            type.getShape(), [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
              auto offset = getAbsoluteOffset(view, index, info->elementBytes);
              if (!offset)
                return offset.takeError();
              std::memcpy(view.storage->bytes.data() + *offset,
                          tensor.getBytes().data() + linearByte,
                          info->elementBytes);
              linearByte += info->elementBytes;
              return llvm::Error::success();
            }))
      return std::move(error);
    return view;
  }

  llvm::Expected<ReferenceTensor> exportTensor(const BufferView &view,
                                               llvm::StringRef dtype,
                                               llvm::ArrayRef<int64_t> shape) {
    auto bytes = getCompactByteCount(dtype, shape);
    if (!bytes)
      return invalid("output tensor has unsupported dtype or shape");
    if (elementDType(view.type.getElementType()) != dtype ||
        view.type.getShape() != shape)
      return invalid("entry result disagrees with typed output binding");
    std::vector<uint8_t> compact(*bytes);
    int64_t linearByte = 0;
    auto info = wafer::computeWaferPhysicalTensorInfo(view.type);
    if (!info || info->elementBytes <= 0)
      return invalid("output has no static accepted physical layout");
    if (llvm::Error error = forEachLogicalIndex(
            shape, [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
              auto offset = getAbsoluteOffset(view, index, info->elementBytes);
              if (!offset)
                return offset.takeError();
              std::memcpy(compact.data() + linearByte,
                          view.storage->bytes.data() + *offset,
                          info->elementBytes);
              linearByte += info->elementBytes;
              return llvm::Error::success();
            }))
      return std::move(error);
    return ReferenceTensor::create(dtype, shape, compact);
  }

  llvm::Expected<std::vector<BufferView>> executeBlock(mlir::Block &block) {
    for (mlir::Operation &operation : block) {
      if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(operation)) {
        auto allocated = allocate(alloc);
        if (!allocated)
          return allocated.takeError();
        buffers[alloc.getResult()] = std::move(*allocated);
        continue;
      }
      if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(operation)) {
        auto source = lookup(cast.getSource());
        if (!source)
          return source.takeError();
        auto resultType = mlir::dyn_cast<mlir::MemRefType>(cast.getType());
        auto info = resultType
                        ? wafer::computeWaferPhysicalTensorInfo(resultType)
                        : std::nullopt;
        if (!info || info->physicalBytes < 0 ||
            info->physicalBytes > source->physicalBytes)
          return invalid("memref.cast changes accepted physical extent");
        source->type = resultType;
        source->physicalBytes = info->physicalBytes;
        buffers[cast.getResult()] = *source;
        continue;
      }
      if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(operation)) {
        if (auto value = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue()))
          scalars[constant.getResult()] = value.getValueAsDouble();
        else if (auto value =
                     mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
          scalars[constant.getResult()] = value.getValue().getSExtValue();
        else
          return invalid("unsupported scalar constant in accepted rank");
        continue;
      }
      if (auto tile = mlir::dyn_cast<wafer::TileRegionOp>(operation)) {
        mlir::Block &body = tile.getBody().front();
        if (body.getNumArguments() != tile.getInputs().size())
          return invalid("tile region argument arity mismatch");
        for (auto [argument, input] :
             llvm::zip_equal(body.getArguments(), tile.getInputs())) {
          auto found = buffers.find(input);
          if (found == buffers.end())
            return invalid("tile region input buffer is unavailable");
          buffers[argument] = found->second;
        }
        auto yielded = executeBlock(body);
        if (!yielded)
          return yielded.takeError();
        if (yielded->size() != tile.getNumResults())
          return invalid("tile region result arity mismatch");
        for (auto [result, value] :
             llvm::zip_equal(tile.getResults(), *yielded))
          buffers[result] = value;
        continue;
      }
      if (auto rdma = mlir::dyn_cast<wafer::InstrRDMAOp>(operation)) {
        if (llvm::Error error = executeRDMA(rdma))
          return std::move(error);
        continue;
      }
      if (auto wdma = mlir::dyn_cast<wafer::InstrWDMAOp>(operation)) {
        if (llvm::Error error = executeWDMA(wdma))
          return std::move(error);
        continue;
      }
      if (auto movement =
              mlir::dyn_cast<wafer::InstrGatherScatterOp>(operation)) {
        if (llvm::Error error = executeGatherScatter(movement))
          return std::move(error);
        continue;
      }
      if (auto gemm = mlir::dyn_cast<wafer::InstrGemmOp>(operation)) {
        if (llvm::Error error = executeGemm(gemm))
          return std::move(error);
        continue;
      }
      if (auto elementwise =
              mlir::dyn_cast<wafer::InstrElementwiseOp>(operation)) {
        if (llvm::Error error = executeElementwise(elementwise))
          return std::move(error);
        continue;
      }
      if (auto fill = mlir::dyn_cast<wafer::InstrFillOp>(operation)) {
        if (llvm::Error error = executeFill(fill))
          return std::move(error);
        continue;
      }
      if (mlir::isa<wafer::SyncLocalFenceOp>(operation))
        continue;
      if (mlir::isa<wafer::TileYieldOp, mlir::func::ReturnOp>(operation)) {
        std::vector<BufferView> result;
        for (mlir::Value value : operation.getOperands()) {
          auto found = buffers.find(value);
          if (found == buffers.end())
            return invalid("returned buffer is unavailable");
          result.push_back(found->second);
        }
        return result;
      }
      return invalid(("unsupported operation in accepted rank: " +
                      operation.getName().getStringRef())
                         .str());
    }
    return invalid("accepted block has no return-like terminator");
  }

  llvm::Expected<BufferView> allocate(mlir::memref::AllocOp alloc) {
    mlir::MemRefType type = alloc.getType();
    auto info = wafer::computeWaferPhysicalTensorInfo(type);
    if (!info || info->physicalBytes < 0)
      return invalid("allocation has no static accepted physical extent");
    int64_t offset = -1;
    std::shared_ptr<Storage> arena;
    if (wafer::isWaferSPMMemRefType(type)) {
      auto attr = alloc->getAttrOfType<wafer::SPMOffsetAttr>(
          wafer::kWaferSPMOffsetAttrName);
      if (!attr)
        return invalid("SPM allocation is missing accepted offset");
      offset = attr.getOffset();
      arena = spmArena;
    } else if (wafer::isWaferDDRMemRefType(type)) {
      auto attr = alloc->getAttrOfType<wafer::DDROffsetAttr>(
          wafer::kWaferDDROffsetAttrName);
      if (!attr)
        return invalid("DDR allocation is missing accepted offset");
      offset = attr.getOffset();
      arena = ddrArena;
    } else {
      return invalid("allocation is outside Wafer DDR/SPM memory");
    }
    if (offset < 0 ||
        info->physicalBytes > std::numeric_limits<int64_t>::max() - offset)
      return invalid("accepted allocation range overflows");
    int64_t end = offset + info->physicalBytes;
    if (end > static_cast<int64_t>(arena->bytes.size()))
      arena->bytes.resize(end, 0);
    return BufferView{std::move(arena), offset, 0, info->physicalBytes, type};
  }

  llvm::Expected<BufferView> lookup(mlir::Value value) {
    auto found = buffers.find(value);
    if (found == buffers.end())
      return invalid("instruction operand buffer is unavailable");
    return found->second;
  }

  llvm::Error copyChunks(const BufferView &source, const BufferView &dest,
                         llvm::ArrayRef<int64_t> sourceOffsets,
                         llvm::ArrayRef<int64_t> destOffsets,
                         int64_t innerBytes) {
    if (sourceOffsets.size() != destOffsets.size() || innerBytes < 0)
      return invalid("movement descriptor has inconsistent chunk counts");
    std::vector<uint8_t> payload(sourceOffsets.size() * innerBytes);
    for (auto [position, offset] : llvm::enumerate(sourceOffsets)) {
      if (offset < 0 || offset > source.physicalBytes - innerBytes)
        return invalid("movement source descriptor exceeds buffer extent");
      int64_t absolute = source.base + source.viewOffset + offset;
      if (absolute < 0 ||
          absolute >
              static_cast<int64_t>(source.storage->bytes.size()) - innerBytes)
        return invalid("movement source exceeds storage arena");
      std::memcpy(payload.data() + position * innerBytes,
                  source.storage->bytes.data() + absolute, innerBytes);
    }
    for (auto [position, offset] : llvm::enumerate(destOffsets)) {
      if (offset < 0 || offset > dest.physicalBytes - innerBytes)
        return invalid("movement destination descriptor exceeds buffer extent");
      int64_t absolute = dest.base + dest.viewOffset + offset;
      if (absolute < 0 ||
          absolute >
              static_cast<int64_t>(dest.storage->bytes.size()) - innerBytes)
        return invalid("movement destination exceeds storage arena");
      std::memcpy(dest.storage->bytes.data() + absolute,
                  payload.data() + position * innerBytes, innerBytes);
    }
    return llvm::Error::success();
  }

  llvm::Error validateMovementByteCount(uint64_t byteCount, uint64_t innerBytes,
                                        size_t chunkCount) {
    if (innerBytes == 0 ||
        chunkCount > std::numeric_limits<uint64_t>::max() / innerBytes ||
        byteCount != chunkCount * innerBytes)
      return invalid("movement byte_count disagrees with its descriptor");
    return llvm::Error::success();
  }

  llvm::Expected<std::vector<int64_t>>
  descriptorOffsets(llvm::ArrayRef<int64_t> strides,
                    llvm::ArrayRef<int64_t> iterations, int64_t base = 0) {
    if (strides.size() != 3 || iterations.size() != 3 || base < 0)
      return invalid("movement descriptor must have three non-negative levels");
    std::vector<int64_t> offsets;
    for (int64_t i0 = 0; i0 < iterations[0]; ++i0)
      for (int64_t i1 = 0; i1 < iterations[1]; ++i1)
        for (int64_t i2 = 0; i2 < iterations[2]; ++i2) {
          if (i0 < 0 || i1 < 0 || i2 < 0 || strides[0] < 0 || strides[1] < 0 ||
              strides[2] < 0)
            return invalid("movement descriptor contains a negative field");
          offsets.push_back(base + i0 * strides[0] + i1 * strides[1] +
                            i2 * strides[2]);
        }
    return offsets;
  }

  llvm::Error executeRDMA(wafer::InstrRDMAOp op) {
    auto source = lookup(op.getSource());
    auto dest = lookup(op.getDest());
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();
    auto sourceOffsets =
        descriptorOffsets(op.getSrcStrides(), op.getSrcIterations());
    if (!sourceOffsets)
      return sourceOffsets.takeError();
    if (llvm::Error error = validateMovementByteCount(
            op.getByteCount(), op.getInnerBytes(), sourceOffsets->size()))
      return error;
    std::vector<int64_t> destOffsets(sourceOffsets->size());
    for (auto [index, offset] : llvm::enumerate(destOffsets))
      offset = index * op.getInnerBytes();
    return copyChunks(*source, *dest, *sourceOffsets, destOffsets,
                      op.getInnerBytes());
  }

  llvm::Error executeWDMA(wafer::InstrWDMAOp op) {
    auto source = lookup(op.getSource());
    auto dest = lookup(op.getDest());
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();
    auto destOffsets =
        descriptorOffsets(op.getDstStrides(), op.getDstIterations());
    if (!destOffsets)
      return destOffsets.takeError();
    if (llvm::Error error = validateMovementByteCount(
            op.getByteCount(), op.getInnerBytes(), destOffsets->size()))
      return error;
    std::vector<int64_t> sourceOffsets(destOffsets->size());
    for (auto [index, offset] : llvm::enumerate(sourceOffsets))
      offset = index * op.getInnerBytes();
    return copyChunks(*source, *dest, sourceOffsets, *destOffsets,
                      op.getInnerBytes());
  }

  llvm::Error executeGatherScatter(wafer::InstrGatherScatterOp op) {
    auto source = lookup(op.getSource());
    auto dest = lookup(op.getDest());
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();
    auto sourceOffsets =
        descriptorOffsets(op.getSrcStrides(), op.getSrcIterations(),
                          op.getSrcOffset().value_or(0));
    auto destOffsets =
        descriptorOffsets(op.getDstStrides(), op.getDstIterations(),
                          op.getDstOffset().value_or(0));
    if (!sourceOffsets)
      return sourceOffsets.takeError();
    if (!destOffsets)
      return destOffsets.takeError();
    if (sourceOffsets->size() != destOffsets->size())
      return invalid("movement source/destination descriptors disagree");
    if (llvm::Error error = validateMovementByteCount(
            op.getByteCount(), op.getInnerBytes(), sourceOffsets->size()))
      return error;
    return copyChunks(*source, *dest, *sourceOffsets, *destOffsets,
                      op.getInnerBytes());
  }

  llvm::Error executeGemm(wafer::InstrGemmOp op) {
    if (op.getBatchCount().value_or(1) != 1)
      return invalid("batched GEMM is not yet supported by reference executor");
    auto lhs = lookup(op.getLhs());
    auto rhs = lookup(op.getRhs());
    auto dest = lookup(op.getDest());
    if (!lhs)
      return lhs.takeError();
    if (!rhs)
      return rhs.takeError();
    if (!dest)
      return dest.takeError();
    if (lhs->type.getRank() != 2 || rhs->type.getRank() != 2 ||
        dest->type.getRank() != 2 ||
        lhs->type.getShape() !=
            llvm::ArrayRef<int64_t>({static_cast<int64_t>(op.getM()),
                                     static_cast<int64_t>(op.getK())}) ||
        rhs->type.getShape() !=
            llvm::ArrayRef<int64_t>({static_cast<int64_t>(op.getK()),
                                     static_cast<int64_t>(op.getN())}) ||
        dest->type.getShape() !=
            llvm::ArrayRef<int64_t>({static_cast<int64_t>(op.getM()),
                                     static_cast<int64_t>(op.getN())}))
      return invalid("GEMM dimensions disagree with accepted buffer shapes");
    for (int64_t m = 0; m < static_cast<int64_t>(op.getM()); ++m)
      for (int64_t n = 0; n < static_cast<int64_t>(op.getN()); ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < static_cast<int64_t>(op.getK()); ++k) {
          auto lhsValue = readF32(*lhs, {m, k});
          auto rhsValue = readF32(*rhs, {k, n});
          if (!lhsValue)
            return lhsValue.takeError();
          if (!rhsValue)
            return rhsValue.takeError();
          sum += *lhsValue * *rhsValue;
        }
        if (llvm::Error error = writeF32(*dest, {m, n}, sum))
          return error;
      }
    return llvm::Error::success();
  }

  llvm::Error executeElementwise(wafer::InstrElementwiseOp op) {
    auto dest = lookup(op.getDest());
    if (!dest)
      return dest.takeError();
    std::vector<BufferView> inputs;
    for (mlir::Value value : op.getInputs()) {
      auto input = lookup(value);
      if (!input)
        return input.takeError();
      if (input->type.getShape() != dest->type.getShape())
        return invalid("elementwise input shape disagrees with destination");
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
          switch (op.getKind()) {
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
            result = std::max(values[0], values[1]);
            break;
          case wafer::InstrElementwiseKind::Min:
            result = std::min(values[0], values[1]);
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
            return invalid("unsupported reference elementwise kind");
          }
          return writeF32(*dest, index, result);
        });
  }

  llvm::Error executeFill(wafer::InstrFillOp op) {
    auto dest = lookup(op.getDest());
    if (!dest)
      return dest.takeError();
    auto scalar = scalars.find(op.getValue());
    if (scalar == scalars.end())
      return invalid("fill scalar is unavailable");
    return forEachLogicalIndex(
        dest->type.getShape(), [&](llvm::ArrayRef<int64_t> index) {
          return writeF32(*dest, index, static_cast<float>(scalar->second));
        });
  }

  const RankExecutable &rank;
  std::shared_ptr<Storage> spmArena;
  std::shared_ptr<Storage> ddrArena;
  llvm::DenseMap<mlir::Value, BufferView> buffers;
  llvm::DenseMap<mlir::Value, double> scalars;
};

} // namespace

llvm::Expected<ReferenceTensor>
ReferenceTensor::create(llvm::StringRef dtype, llvm::ArrayRef<int64_t> shape,
                        llvm::ArrayRef<uint8_t> bytes) {
  std::optional<int64_t> expected = getCompactByteCount(dtype, shape);
  if (!expected)
    return invalid("reference tensor has unsupported dtype or shape");
  if (*expected != static_cast<int64_t>(bytes.size()))
    return invalid(
        "reference tensor byte count disagrees with dtype and shape");
  return ReferenceTensor(dtype.str(), std::vector<int64_t>(shape),
                         std::vector<uint8_t>(bytes));
}

llvm::Expected<ReferenceExecutionResult>
executeReferenceRank(const ExecutableBundle &bundle, int64_t logicalRank,
                     llvm::ArrayRef<ReferenceInputBinding> inputs) {
  if (logicalRank < 0 ||
      logicalRank >= bundle.getExecutionConfig().getRankCount())
    return invalid("reference logical rank is outside executable bundle");
  const RankExecutable *match = nullptr;
  for (const RankExecutable &rank : bundle.getRankExecutables()) {
    if (rank.getLogicalRank() != logicalRank)
      continue;
    if (match)
      return invalid("executable bundle contains a duplicate logical rank");
    match = &rank;
  }
  if (!match)
    return invalid("executable bundle is missing requested logical rank");
  return RankInterpreter(*match).run(inputs);
}

} // namespace wafer::compiler
