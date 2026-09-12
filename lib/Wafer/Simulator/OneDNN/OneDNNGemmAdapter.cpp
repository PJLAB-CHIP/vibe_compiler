//===- OneDNNGemmAdapter.cpp - Qualified oneDNN GEMM adapter --------===//

#include "OneDNNTensorNumericInternal.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace wafer::onednn_detail {

namespace {

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

std::optional<dnnl::memory::data_type>
getDNNLAdapterDataType(LogicalFormat format) {
  switch (format) {
  case LogicalFormat::F16:
  case LogicalFormat::BF16:
  case LogicalFormat::F32:
    return dnnl::memory::data_type::f32;
  default:
    return std::nullopt;
  }
}

uint64_t getElementBytes(LogicalFormat format) {
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(format);
  return descriptor && !descriptor->bitpacked ? descriptor->storageBits / 8 : 0;
}

uint32_t widenF16ToF32Bits(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & UINT16_C(0x8000)) << 16;
  uint32_t exponent = (bits >> 10) & UINT16_C(0x1f);
  uint32_t fraction = bits & UINT16_C(0x03ff);
  if (exponent == 0) {
    if (fraction == 0)
      return sign;
    int32_t unbiasedExponent = -14;
    while ((fraction & UINT32_C(0x0400)) == 0) {
      fraction <<= 1;
      --unbiasedExponent;
    }
    return sign | (static_cast<uint32_t>(unbiasedExponent + 127) << 23) |
           ((fraction & UINT32_C(0x03ff)) << 13);
  }
  if (exponent == UINT32_C(0x1f))
    return sign | UINT32_C(0x7f800000) | (fraction << 13);
  exponent = exponent - 15 + 127;
  return sign | (exponent << 23) | (fraction << 13);
}

uint32_t widenToF32Bits(LogicalFormat format, uint64_t bits) {
  switch (format) {
  case LogicalFormat::F16:
    return widenF16ToF32Bits(static_cast<uint16_t>(bits));
  case LogicalFormat::BF16:
    return static_cast<uint32_t>(bits) << 16;
  case LogicalFormat::F32:
    return static_cast<uint32_t>(bits);
  default:
    llvm_unreachable("unsupported oneDNN dense adapter format");
  }
}

llvm::Expected<std::vector<uint8_t>>
makeF32DenseBytes(llvm::ArrayRef<RawLogicalValue> values,
                  LogicalFormat format) {
  constexpr uint64_t elementBytes = sizeof(uint32_t);
  uint64_t byteCount = 0;
  if (!checkedMultiply(values.size(), elementBytes, byteCount) ||
      byteCount > std::numeric_limits<size_t>::max())
    return onednnError(OneDNNTensorNumericErrorCode::WorkCountOverflow,
                       "dense adapter byte count overflows");
  std::vector<uint8_t> bytes(static_cast<size_t>(byteCount));
  for (size_t index = 0; index < values.size(); ++index) {
    if (values[index].format != format)
      return onednnError(OneDNNTensorNumericErrorCode::InvalidInputEncoding,
                         "dense adapter saw a mixed logical format");
    switch (format) {
    case LogicalFormat::F16:
    case LogicalFormat::BF16:
    case LogicalFormat::F32:
      break;
    default:
      return onednnError(OneDNNTensorNumericErrorCode::UnsupportedFormat,
                         "F32 dense adapter received an unsupported format");
    }
    // The physical codec produced these values with the selected strict
    // decode policy, so repeating scalar canonicalization here would only
    // revalidate the same bits before widening them to the backend format.
    const uint32_t f32Bits = widenToF32Bits(format, values[index].bits);
    for (uint64_t byte = 0; byte < elementBytes; ++byte)
      bytes[index * elementBytes + byte] =
          static_cast<uint8_t>(f32Bits >> (8 * byte));
  }
  return bytes;
}

llvm::Expected<std::vector<RawLogicalValue>>
makeRawValues(llvm::ArrayRef<uint8_t> bytes, LogicalFormat format,
              uint64_t count) {
  uint64_t elementBytes = getElementBytes(format);
  uint64_t expectedBytes = 0;
  if (elementBytes == 0 ||
      !checkedMultiply(count, elementBytes, expectedBytes) ||
      expectedBytes != bytes.size())
    return onednnError(OneDNNTensorNumericErrorCode::BackendExecutionFailure,
                       "backend dense result has an inconsistent size");
  std::vector<RawLogicalValue> values;
  values.reserve(static_cast<size_t>(count));
  for (uint64_t index = 0; index < count; ++index) {
    uint64_t bits = 0;
    for (uint64_t byte = 0; byte < elementBytes; ++byte)
      bits |= static_cast<uint64_t>(
                  bytes[static_cast<size_t>(index * elementBytes + byte)])
              << (8 * byte);
    llvm::Expected<RawLogicalValue> value =
        makeRawLogicalValue(format, bits, NonCanonicalEncodingPolicy::Reject);
    if (!value)
      return onednnError(OneDNNTensorNumericErrorCode::BackendExecutionFailure,
                         llvm::toString(value.takeError()));
    values.push_back(*value);
  }
  return values;
}

dnnl::memory::dims toDNNLDims(llvm::ArrayRef<uint64_t> shape) {
  dnnl::memory::dims result;
  result.reserve(shape.size());
  for (uint64_t dimension : shape)
    result.push_back(static_cast<dnnl_dim_t>(dimension));
  return result;
}

dnnl::memory::dims getDenseStrides(const dnnl::memory::dims &dims) {
  dnnl::memory::dims strides(dims.size(), 1);
  for (size_t index = dims.size(); index > 1; --index)
    strides[index - 2] = strides[index - 1] * dims[index - 1];
  return strides;
}

struct ValidatedGemm {
  const FormalGemmOperation *gemm = nullptr;
  LogicalFormat format = LogicalFormat::F32;
  uint64_t fusedMultiplyAdds = 0;
};

llvm::Expected<ValidatedGemm>
validateGemm(const FormalGemmOperation &operation,
             llvm::ArrayRef<OneDNNTensorStorage> inputs,
             const OneDNNTensorStorage &destinationTemplate) {
  if (operation.psum)
    return onednnError(OneDNNTensorNumericErrorCode::UnsupportedOperation,
                       "oneDNN GEMM adapter has no psum qualification");
  const FormalGemmOperation *gemm = &operation;
  if (inputs.size() != 2)
    return onednnError(OneDNNTensorNumericErrorCode::InputArityMismatch,
                       "NE GEMM onednn execution requires two inputs");
  if (inputs[0].getKey() != gemm->lhs || inputs[1].getKey() != gemm->rhs ||
      destinationTemplate.getKey() != gemm->destination)
    return onednnError(OneDNNTensorNumericErrorCode::InvalidPhysicalStorage,
                       "onednn tensor keys do not exactly match the operation");
  if (gemm->lhs.getFormat() != gemm->rhs.getFormat() ||
      gemm->lhs.getFormat() != gemm->destination.getFormat() ||
      !getDNNLAdapterDataType(gemm->lhs.getFormat()))
    return onednnError(OneDNNTensorNumericErrorCode::UnsupportedFormat,
                       "onednn GEMM only admits same-format f16, bf16 or f32");
  if (gemm->lhs.getShape().size() < 2 || gemm->lhs.getShape().size() > 12 ||
      gemm->rhs.getShape().size() != gemm->lhs.getShape().size() ||
      gemm->destination.getShape().size() != gemm->lhs.getShape().size())
    return onednnError(OneDNNTensorNumericErrorCode::UnsupportedOperation,
                       "oneDNN MatMul requires a common rank in [2, 12]");
  uint64_t outputCount = 0;
  uint64_t fusedMultiplyAdds = 0;
  if (!checkedMultiply(gemm->batchCount, gemm->m, outputCount) ||
      !checkedMultiply(outputCount, gemm->n, outputCount) ||
      outputCount != gemm->destination.getElementCount() ||
      !checkedMultiply(outputCount, gemm->k, fusedMultiplyAdds))
    return onednnError(OneDNNTensorNumericErrorCode::WorkCountOverflow,
                       "NE GEMM work count is inconsistent or overflows");
  return ValidatedGemm{gemm, gemm->lhs.getFormat(), fusedMultiplyAdds};
}

llvm::Expected<uint64_t>
computeTensorBufferBytes(llvm::ArrayRef<OneDNNTensorStorage> inputs,
                         const OneDNNTensorStorage &destination,
                         OneDNNNumericWorkBudget budget) {
  uint64_t total = 0;
  auto add = [&](uint64_t bytes) -> bool {
    return checkedAdd(total, bytes, total);
  };
  for (const OneDNNTensorStorage &input : inputs)
    if (!add(input.getStorage().size()))
      return onednnError(OneDNNTensorNumericErrorCode::WorkCountOverflow,
                         "physical input byte count overflows");
  if (!add(destination.getStorage().size()))
    return onednnError(OneDNNTensorNumericErrorCode::WorkCountOverflow,
                       "physical destination byte count overflows");
  constexpr uint64_t elementBytes = sizeof(uint32_t);
  for (const OneDNNTensorStorage &input : inputs) {
    uint64_t dense = 0;
    if (!checkedMultiply(input.getKey().getElementCount(), elementBytes,
                         dense) ||
        !add(dense))
      return onednnError(OneDNNTensorNumericErrorCode::WorkCountOverflow,
                         "dense input byte count overflows");
  }
  uint64_t denseDestination = 0;
  if (!checkedMultiply(destination.getKey().getElementCount(), elementBytes,
                       denseDestination) ||
      !add(denseDestination))
    return onednnError(OneDNNTensorNumericErrorCode::WorkCountOverflow,
                       "dense destination byte count overflows");
  if (total > budget.getMaximumTotalBytes())
    return onednnError(OneDNNTensorNumericErrorCode::TotalByteBudgetExceeded,
                       llvm::Twine("onednn adapter requires at least ") +
                           llvm::Twine(total) + " bytes but budget allows " +
                           llvm::Twine(budget.getMaximumTotalBytes()));
  return total;
}

llvm::Expected<std::string> descriptorDigest(
    dnnl::memory::desc descriptor, llvm::StringRef implementation,
    uint64_t scratchpadBytes, const OneDNNExecutionEnvironment &environment,
    const FormalGemmOperation &operation, LogicalFormat targetFormat) {
  std::vector<uint8_t> blob = descriptor.get_blob();
  llvm::SmallString<512> identity;
  llvm::raw_svector_ostream stream(identity);
  appendField(stream, "schema", "wafer-onednn-descriptor");
  appendField(stream, "environment", environment.getDigest());
  appendField(stream, "adapter", getOneDNNAdapterContractDigest());
  llvm::Expected<std::string> problemDigest =
      computeOneDNNGemmProblemDigest(operation);
  if (!problemDigest)
    return problemDigest.takeError();
  appendField(stream, "problem", *problemDigest);
  appendField(stream, "backend_dense_format", "f32");
  appendField(stream, "target_format", stringifyLogicalFormat(targetFormat));
  appendField(stream, "implementation", implementation);
  appendField(stream, "scratchpad_bytes", scratchpadBytes);
  appendField(stream, "weights_descriptor_blob", sha256(blob));
  appendField(stream, "weights_descriptor_size", descriptor.get_size());
  return sha256(identity);
}

} // namespace

llvm::Expected<detail::UnqualifiedOneDNNExecutionResult>
executeOneDNN(const OneDNNExecutionEnvironment &environment,
              const FormalGemmOperation &operation,
              llvm::ArrayRef<OneDNNTensorStorage> inputs,
              const OneDNNTensorStorage &destinationTemplate,
              OneDNNNumericWorkBudget budget) {
  llvm::Expected<ValidatedGemm> validatedGemm =
      validateGemm(operation, inputs, destinationTemplate);
  if (!validatedGemm)
    return validatedGemm.takeError();
  llvm::Expected<uint64_t> tensorBufferBytes =
      computeTensorBufferBytes(inputs, destinationTemplate, budget);
  if (!tensorBufferBytes)
    return tensorBufferBytes.takeError();

  std::fenv_t savedFloatingEnvironment;
  if (std::fegetenv(&savedFloatingEnvironment) != 0)
    return onednnError(
        OneDNNTensorNumericErrorCode::BackendConfigurationFailure,
        "could not snapshot the caller floating environment");
  const uint32_t savedMXCSR = readMXCSR();
  auto restoreFloatingEnvironment = llvm::make_scope_exit([&] {
    std::fesetenv(&savedFloatingEnvironment);
#if defined(__x86_64__) || defined(_M_X64)
    _mm_setcsr(savedMXCSR);
#endif
  });

  llvm::Expected<std::vector<RawLogicalValue>> lhsValues =
      unpackOneDNNTensorLogicalValues(inputs[0]);
  if (!lhsValues)
    return lhsValues.takeError();
  llvm::Expected<std::vector<RawLogicalValue>> rhsValues =
      unpackOneDNNTensorLogicalValues(inputs[1]);
  if (!rhsValues)
    return rhsValues.takeError();
  llvm::Expected<std::vector<uint8_t>> lhsDense =
      makeF32DenseBytes(*lhsValues, validatedGemm->format);
  if (!lhsDense)
    return lhsDense.takeError();
  llvm::Expected<std::vector<uint8_t>> rhsDense =
      makeF32DenseBytes(*rhsValues, validatedGemm->format);
  if (!rhsDense)
    return rhsDense.takeError();

  constexpr uint64_t elementBytes = sizeof(uint32_t);
  uint64_t destinationBytes = 0;
  if (!checkedMultiply(validatedGemm->gemm->destination.getElementCount(),
                       elementBytes, destinationBytes) ||
      destinationBytes > std::numeric_limits<size_t>::max())
    return onednnError(OneDNNTensorNumericErrorCode::WorkCountOverflow,
                       "dense destination allocation overflows");
  std::vector<uint8_t> destinationDense(static_cast<size_t>(destinationBytes),
                                        0);

  try {
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream executionStream(engine);
    std::optional<dnnl::memory::data_type> dataType =
        getDNNLAdapterDataType(validatedGemm->format);
    dnnl::memory::dims lhsDims =
        toDNNLDims(validatedGemm->gemm->lhs.getShape());
    dnnl::memory::dims rhsDims =
        toDNNLDims(validatedGemm->gemm->rhs.getShape());
    dnnl::memory::dims destinationDims =
        toDNNLDims(validatedGemm->gemm->destination.getShape());
    dnnl::memory::desc lhsDescriptor(lhsDims, *dataType,
                                     getDenseStrides(lhsDims));
    dnnl::memory::desc rhsPlainDescriptor(rhsDims, *dataType,
                                          getDenseStrides(rhsDims));
    dnnl::memory::desc rhsAnyDescriptor(rhsDims, *dataType,
                                        dnnl::memory::format_tag::any);
    dnnl::memory::desc destinationDescriptor(destinationDims, *dataType,
                                             getDenseStrides(destinationDims));
    dnnl::primitive_attr attributes;
    attributes.set_deterministic(true);
    attributes.set_fpmath_mode(dnnl::fpmath_mode::strict, true);
    attributes.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    dnnl::matmul::primitive_desc primitiveDescriptor(
        engine, lhsDescriptor, rhsAnyDescriptor, destinationDescriptor,
        attributes);
    dnnl::memory::desc resolvedWeights = primitiveDescriptor.weights_desc();
    dnnl::memory::desc scratchpadDescriptor =
        primitiveDescriptor.scratchpad_desc();
    const uint64_t scratchpadBytes = scratchpadDescriptor.get_size();
    const uint64_t reorderedBytes = resolvedWeights.get_size();
    const bool needsReorder = resolvedWeights != rhsPlainDescriptor;
    if (scratchpadBytes > budget.getMaximumScratchpadBytes())
      return onednnError(OneDNNTensorNumericErrorCode::ScratchpadBudgetExceeded,
                         llvm::Twine("oneDNN scratchpad requires ") +
                             llvm::Twine(scratchpadBytes) + " bytes");
    const uint64_t reorderBudget = needsReorder ? reorderedBytes : 0;
    if (reorderBudget > budget.getMaximumReorderBytes())
      return onednnError(OneDNNTensorNumericErrorCode::ReorderBudgetExceeded,
                         llvm::Twine("oneDNN weights reorder requires ") +
                             llvm::Twine(reorderBudget) + " bytes");
    uint64_t totalBytes = *tensorBufferBytes;
    if (!checkedAdd(totalBytes, scratchpadBytes, totalBytes) ||
        !checkedAdd(totalBytes, reorderBudget, totalBytes))
      return onednnError(OneDNNTensorNumericErrorCode::WorkCountOverflow,
                         "oneDNN scratch/reorder byte count overflows");
    if (totalBytes > budget.getMaximumTotalBytes())
      return onednnError(OneDNNTensorNumericErrorCode::TotalByteBudgetExceeded,
                         llvm::Twine("onednn execution requires ") +
                             llvm::Twine(totalBytes) +
                             " bytes but budget allows " +
                             llvm::Twine(budget.getMaximumTotalBytes()));

    std::vector<uint8_t> reorderedStorage;
    if (needsReorder)
      reorderedStorage.resize(static_cast<size_t>(reorderedBytes));
    std::vector<uint8_t> scratchpad(static_cast<size_t>(scratchpadBytes));
    dnnl::memory lhsMemory(lhsDescriptor, engine, lhsDense->data());
    dnnl::memory rhsPlainMemory(rhsPlainDescriptor, engine, rhsDense->data());
    dnnl::memory rhsMemory =
        needsReorder
            ? dnnl::memory(resolvedWeights, engine, reorderedStorage.data())
            : rhsPlainMemory;
    uint64_t reorderInvocations = 0;
    if (needsReorder) {
      dnnl::reorder(rhsPlainMemory, rhsMemory)
          .execute(executionStream, rhsPlainMemory, rhsMemory);
      executionStream.wait();
      reorderInvocations = 1;
    }
    dnnl::memory destinationMemory(destinationDescriptor, engine,
                                   destinationDense.data());
    dnnl::memory scratchpadMemory(scratchpadDescriptor, engine,
                                  scratchpad.data());
    dnnl::matmul primitive(primitiveDescriptor);
    primitive.execute(executionStream,
                      {{DNNL_ARG_SRC, lhsMemory},
                       {DNNL_ARG_WEIGHTS, rhsMemory},
                       {DNNL_ARG_DST, destinationMemory},
                       {DNNL_ARG_SCRATCHPAD, scratchpadMemory}});
    executionStream.wait();

    llvm::Expected<std::vector<RawLogicalValue>> accumulators =
        makeRawValues(destinationDense, LogicalFormat::F32,
                      validatedGemm->gemm->destination.getElementCount());
    if (!accumulators)
      return accumulators.takeError();
    std::vector<RawLogicalValue> destinationValues;
    destinationValues.reserve(accumulators->size());
    for (RawLogicalValue accumulator : *accumulators) {
      llvm::Expected<FormalNumericResult> finalized =
          evaluateFormalGemmFinalize(operation, accumulator);
      if (!finalized)
        return onednnError(
            OneDNNTensorNumericErrorCode::BackendExecutionFailure,
            llvm::toString(finalized.takeError()));
      destinationValues.push_back(finalized->value);
    }
    llvm::Expected<OneDNNTensorStorage> packed =
        packIntoTemplate(validatedGemm->gemm->destination, destinationValues,
                         destinationTemplate.getStorage().vec());
    if (!packed)
      return packed.takeError();
    const char *implementationPointer = primitiveDescriptor.impl_info_str();
    std::string implementation =
        implementationPointer ? implementationPointer : "";
    if (implementation.empty())
      return onednnError(OneDNNTensorNumericErrorCode::BackendDescriptorFailure,
                         "oneDNN returned an empty implementation identity");
    llvm::Expected<std::string> resolvedDescriptorDigest =
        descriptorDigest(resolvedWeights, implementation, scratchpadBytes,
                         environment, operation, validatedGemm->format);
    if (!resolvedDescriptorDigest)
      return resolvedDescriptorDigest.takeError();
    OneDNNDispatchEvidence evidence{/*matmulInvocations=*/1,
                                    reorderInvocations,
                                    /*formalFusedMultiplyAdds=*/0,
                                    totalBytes,
                                    scratchpadBytes,
                                    implementation,
                                    std::move(*resolvedDescriptorDigest)};
    return detail::UnqualifiedOneDNNExecutionResult{std::move(*packed),
                                                    std::move(evidence)};
  } catch (const dnnl::error &error) {
    return onednnError(OneDNNTensorNumericErrorCode::BackendExecutionFailure,
                       llvm::Twine("oneDNN status ") +
                           llvm::Twine(static_cast<int>(error.status)) + ": " +
                           error.what());
  } catch (const std::exception &error) {
    return onednnError(OneDNNTensorNumericErrorCode::BackendExecutionFailure,
                       error.what());
  }
}

} // namespace wafer::onednn_detail
