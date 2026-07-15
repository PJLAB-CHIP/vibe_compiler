//===- BulkTensorNumeric.cpp - Qualified oneDNN tensor execution --------===//

#include "Wafer/Target/BulkTensorNumeric.h"

#include "BulkTensorNumericInternal.h"
#include "Wafer/Target/PhysicalTensorCodec.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_version_hash.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <algorithm>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <gnu/libc-version.h>
#include <sched.h>
#include <sys/utsname.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifndef WAFER_BULK_DEPENDENCY_RECORD_SHA256
#error "managed bulk dependency record identity is required"
#endif
#ifndef WAFER_BULK_ONEDNN_LIBRARY_SHA256
#error "managed oneDNN library identity is required"
#endif
#ifndef WAFER_BULK_ONEDNN_VERSION
#error "managed oneDNN version is required"
#endif
#ifndef WAFER_BULK_ONEDNN_COMMIT
#error "managed oneDNN commit is required"
#endif

namespace wafer {
namespace {

llvm::Error bulkError(BulkTensorNumericErrorCode code,
                      const llvm::Twine &detail) {
  return llvm::make_error<BulkTensorNumericError>(code, detail.str());
}

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

uint32_t readMXCSR();

std::string sha256(llvm::StringRef payload) {
  llvm::SHA256 hasher;
  hasher.update(payload);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

std::string sha256(llvm::ArrayRef<uint8_t> payload) {
  llvm::SHA256 hasher;
  hasher.update(payload);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

void appendField(llvm::raw_ostream &stream, llvm::StringRef name,
                 llvm::StringRef value) {
  stream << name << '=' << value.size() << ':' << value << '\n';
}

void appendField(llvm::raw_ostream &stream, llvm::StringRef name,
                 uint64_t value) {
  stream << name << '=' << value << '\n';
}

llvm::Expected<BulkTensorStorage>
packIntoTemplate(const NumericTensorKey &key,
                 llvm::ArrayRef<RawLogicalValue> values,
                 std::vector<uint8_t> storage) {
  llvm::Expected<std::vector<uint8_t>> packed =
      packPhysicalTensorLogicalValues(key, values, storage);
  if (!packed)
    return bulkError(BulkTensorNumericErrorCode::InvalidInputEncoding,
                     llvm::toString(packed.takeError()));
  return BulkTensorStorage::create(key, std::move(*packed));
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

llvm::Expected<std::vector<uint8_t>>
makeF32DenseBytes(llvm::ArrayRef<RawLogicalValue> values,
                  LogicalFormat format) {
  constexpr uint64_t elementBytes = sizeof(uint32_t);
  uint64_t byteCount = 0;
  if (!checkedMultiply(values.size(), elementBytes, byteCount) ||
      byteCount > std::numeric_limits<size_t>::max())
    return bulkError(BulkTensorNumericErrorCode::WorkCountOverflow,
                     "dense adapter byte count overflows");
  std::vector<uint8_t> bytes(static_cast<size_t>(byteCount));
  for (size_t index = 0; index < values.size(); ++index) {
    if (values[index].format != format)
      return bulkError(BulkTensorNumericErrorCode::InvalidInputEncoding,
                       "dense adapter saw a mixed logical format");
    const llvm::fltSemantics *semantics = nullptr;
    unsigned storageBits = 0;
    switch (format) {
    case LogicalFormat::F16:
      semantics = &llvm::APFloat::IEEEhalf();
      storageBits = 16;
      break;
    case LogicalFormat::BF16:
      semantics = &llvm::APFloat::BFloat();
      storageBits = 16;
      break;
    case LogicalFormat::F32:
      semantics = &llvm::APFloat::IEEEsingle();
      storageBits = 32;
      break;
    default:
      return bulkError(BulkTensorNumericErrorCode::UnsupportedFormat,
                       "F32 dense adapter received an unsupported format");
    }
    llvm::Expected<RawLogicalValue> canonical = makeRawLogicalValue(
        format, values[index].bits, NonCanonicalEncodingPolicy::Reject);
    if (!canonical)
      return bulkError(BulkTensorNumericErrorCode::InvalidInputEncoding,
                       llvm::toString(canonical.takeError()));
    llvm::APFloat floating(*semantics, llvm::APInt(storageBits, canonical->bits,
                                                   /*isSigned=*/false));
    bool losesInfo = false;
    llvm::APFloat::opStatus status =
        floating.convert(llvm::APFloat::IEEEsingle(),
                         llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    if ((status & (llvm::APFloat::opOverflow | llvm::APFloat::opUnderflow)) !=
            0 ||
        losesInfo)
      return bulkError(BulkTensorNumericErrorCode::InvalidInputEncoding,
                       "f16/bf16 input did not widen exactly to f32");
    const uint32_t f32Bits =
        static_cast<uint32_t>(floating.bitcastToAPInt().getZExtValue());
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
    return bulkError(BulkTensorNumericErrorCode::BackendExecutionFailure,
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
      return bulkError(BulkTensorNumericErrorCode::BackendExecutionFailure,
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

struct GemmPreflight {
  const NumericNEGemmCommand *gemm = nullptr;
  LogicalFormat format = LogicalFormat::F32;
  uint64_t fusedMultiplyAdds = 0;
};

llvm::Expected<GemmPreflight>
preflightGemm(const ResolvedNumericCommand &command,
              llvm::ArrayRef<BulkTensorStorage> inputs,
              const BulkTensorStorage &destinationTemplate) {
  if (!command.isSupported() ||
      command.getFamily() != NumericCommandFamily::NEGemm ||
      command.getFormalKernelKind() != FormalKernelKind::Gemm ||
      command.getFormalBackendKind() !=
          FormalNumericBackendKind::LLVMAPFloatAPInt ||
      !command.getSemantics())
    return bulkError(BulkTensorNumericErrorCode::UnsupportedResolvedCommand,
                     "bulk backend requires a complete supported NE GEMM");
  const NumericNEGemmCommand *gemm = command.getCommandKey().getNEGemm();
  if (!gemm)
    return bulkError(BulkTensorNumericErrorCode::UnsupportedResolvedCommand,
                     "resolved NE GEMM lost its command payload");
  if (inputs.size() != 2)
    return bulkError(BulkTensorNumericErrorCode::InputArityMismatch,
                     "NE GEMM bulk execution requires two inputs");
  if (inputs[0].getKey() != gemm->lhs || inputs[1].getKey() != gemm->rhs ||
      destinationTemplate.getKey() != gemm->destination)
    return bulkError(BulkTensorNumericErrorCode::InvalidPhysicalStorage,
                     "bulk tensor keys do not exactly match the command");
  if (gemm->lhs.getFormat() != gemm->rhs.getFormat() ||
      gemm->lhs.getFormat() != gemm->destination.getFormat() ||
      !getDNNLAdapterDataType(gemm->lhs.getFormat()))
    return bulkError(BulkTensorNumericErrorCode::UnsupportedFormat,
                     "bulk GEMM only admits same-format f16, bf16 or f32");
  if (gemm->lhs.getShape().size() < 2 || gemm->lhs.getShape().size() > 12 ||
      gemm->rhs.getShape().size() != gemm->lhs.getShape().size() ||
      gemm->destination.getShape().size() != gemm->lhs.getShape().size())
    return bulkError(BulkTensorNumericErrorCode::UnsupportedResolvedCommand,
                     "oneDNN MatMul requires a common rank in [2, 12]");
  uint64_t outputCount = 0;
  uint64_t fusedMultiplyAdds = 0;
  if (!checkedMultiply(gemm->batchCount, gemm->m, outputCount) ||
      !checkedMultiply(outputCount, gemm->n, outputCount) ||
      outputCount != gemm->destination.getElementCount() ||
      !checkedMultiply(outputCount, gemm->k, fusedMultiplyAdds))
    return bulkError(BulkTensorNumericErrorCode::WorkCountOverflow,
                     "NE GEMM work count is inconsistent or overflows");
  return GemmPreflight{gemm, gemm->lhs.getFormat(), fusedMultiplyAdds};
}

llvm::Expected<uint64_t>
preflightBaseBytes(llvm::ArrayRef<BulkTensorStorage> inputs,
                   const BulkTensorStorage &destination,
                   const GemmPreflight &preflight,
                   BulkNumericWorkBudget budget) {
  uint64_t total = 0;
  auto add = [&](uint64_t bytes) -> bool {
    return checkedAdd(total, bytes, total);
  };
  for (const BulkTensorStorage &input : inputs)
    if (!add(input.getStorage().size()))
      return bulkError(BulkTensorNumericErrorCode::WorkCountOverflow,
                       "physical input byte count overflows");
  if (!add(destination.getStorage().size()))
    return bulkError(BulkTensorNumericErrorCode::WorkCountOverflow,
                     "physical destination byte count overflows");
  constexpr uint64_t elementBytes = sizeof(uint32_t);
  for (const BulkTensorStorage &input : inputs) {
    uint64_t dense = 0;
    if (!checkedMultiply(input.getKey().getElementCount(), elementBytes,
                         dense) ||
        !add(dense))
      return bulkError(BulkTensorNumericErrorCode::WorkCountOverflow,
                       "dense input byte count overflows");
  }
  uint64_t denseDestination = 0;
  if (!checkedMultiply(destination.getKey().getElementCount(), elementBytes,
                       denseDestination) ||
      !add(denseDestination))
    return bulkError(BulkTensorNumericErrorCode::WorkCountOverflow,
                     "dense destination byte count overflows");
  if (total > budget.getMaximumTotalBytes())
    return bulkError(BulkTensorNumericErrorCode::TotalByteBudgetExceeded,
                     llvm::Twine("bulk adapter requires at least ") +
                         llvm::Twine(total) + " bytes but budget allows " +
                         llvm::Twine(budget.getMaximumTotalBytes()));
  return total;
}

std::string descriptorDigest(dnnl::memory::desc descriptor,
                             llvm::StringRef implementation,
                             uint64_t scratchpadBytes,
                             const BulkExecutionEnvironment &environment,
                             const ResolvedNumericCommand &command,
                             LogicalFormat targetFormat) {
  std::vector<uint8_t> blob = descriptor.get_blob();
  llvm::SmallString<512> identity;
  llvm::raw_svector_ostream stream(identity);
  appendField(stream, "schema", "wafer-bulk-descriptor-v1");
  appendField(stream, "environment", environment.getDigest());
  appendField(stream, "adapter", getBulkAdapterIdentityDigest());
  appendField(stream, "resolution", command.getDigest());
  appendField(stream, "backend_dense_format", "f32");
  appendField(stream, "target_format", stringifyLogicalFormat(targetFormat));
  appendField(stream, "implementation", implementation);
  appendField(stream, "scratchpad_bytes", scratchpadBytes);
  appendField(stream, "weights_descriptor_blob", sha256(blob));
  appendField(stream, "weights_descriptor_size", descriptor.get_size());
  return sha256(identity);
}

llvm::Expected<detail::UnqualifiedBulkExecutionResult>
executeOneDNN(const BulkExecutionEnvironment &environment,
              const ResolvedNumericCommand &command,
              llvm::ArrayRef<BulkTensorStorage> inputs,
              const BulkTensorStorage &destinationTemplate,
              BulkNumericWorkBudget budget) {
  llvm::Expected<GemmPreflight> preflight =
      preflightGemm(command, inputs, destinationTemplate);
  if (!preflight)
    return preflight.takeError();
  llvm::Expected<uint64_t> baseBytes =
      preflightBaseBytes(inputs, destinationTemplate, *preflight, budget);
  if (!baseBytes)
    return baseBytes.takeError();

  std::fenv_t savedFloatingEnvironment;
  if (std::fegetenv(&savedFloatingEnvironment) != 0)
    return bulkError(BulkTensorNumericErrorCode::BackendConfigurationFailure,
                     "could not snapshot the caller floating environment");
  const uint32_t savedMXCSR = readMXCSR();
  auto restoreFloatingEnvironment = llvm::make_scope_exit([&] {
    std::fesetenv(&savedFloatingEnvironment);
#if defined(__x86_64__) || defined(_M_X64)
    _mm_setcsr(savedMXCSR);
#endif
  });

  llvm::Expected<std::vector<RawLogicalValue>> lhsValues =
      unpackBulkTensorLogicalValues(inputs[0]);
  if (!lhsValues)
    return lhsValues.takeError();
  llvm::Expected<std::vector<RawLogicalValue>> rhsValues =
      unpackBulkTensorLogicalValues(inputs[1]);
  if (!rhsValues)
    return rhsValues.takeError();
  llvm::Expected<std::vector<uint8_t>> lhsDense =
      makeF32DenseBytes(*lhsValues, preflight->format);
  if (!lhsDense)
    return lhsDense.takeError();
  llvm::Expected<std::vector<uint8_t>> rhsDense =
      makeF32DenseBytes(*rhsValues, preflight->format);
  if (!rhsDense)
    return rhsDense.takeError();

  constexpr uint64_t elementBytes = sizeof(uint32_t);
  uint64_t destinationBytes = 0;
  if (!checkedMultiply(preflight->gemm->destination.getElementCount(),
                       elementBytes, destinationBytes) ||
      destinationBytes > std::numeric_limits<size_t>::max())
    return bulkError(BulkTensorNumericErrorCode::WorkCountOverflow,
                     "dense destination allocation overflows");
  std::vector<uint8_t> destinationDense(static_cast<size_t>(destinationBytes),
                                        0);

  try {
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream executionStream(engine);
    std::optional<dnnl::memory::data_type> dataType =
        getDNNLAdapterDataType(preflight->format);
    dnnl::memory::dims lhsDims = toDNNLDims(preflight->gemm->lhs.getShape());
    dnnl::memory::dims rhsDims = toDNNLDims(preflight->gemm->rhs.getShape());
    dnnl::memory::dims destinationDims =
        toDNNLDims(preflight->gemm->destination.getShape());
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
      return bulkError(BulkTensorNumericErrorCode::ScratchpadBudgetExceeded,
                       llvm::Twine("oneDNN scratchpad requires ") +
                           llvm::Twine(scratchpadBytes) + " bytes");
    const uint64_t reorderBudget = needsReorder ? reorderedBytes : 0;
    if (reorderBudget > budget.getMaximumReorderBytes())
      return bulkError(BulkTensorNumericErrorCode::ReorderBudgetExceeded,
                       llvm::Twine("oneDNN weights reorder requires ") +
                           llvm::Twine(reorderBudget) + " bytes");
    uint64_t totalBytes = *baseBytes;
    if (!checkedAdd(totalBytes, scratchpadBytes, totalBytes) ||
        !checkedAdd(totalBytes, reorderBudget, totalBytes))
      return bulkError(BulkTensorNumericErrorCode::WorkCountOverflow,
                       "oneDNN scratch/reorder byte count overflows");
    if (totalBytes > budget.getMaximumTotalBytes())
      return bulkError(BulkTensorNumericErrorCode::TotalByteBudgetExceeded,
                       llvm::Twine("bulk execution requires ") +
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
                      preflight->gemm->destination.getElementCount());
    if (!accumulators)
      return accumulators.takeError();
    std::vector<RawLogicalValue> destinationValues;
    destinationValues.reserve(accumulators->size());
    for (RawLogicalValue accumulator : *accumulators) {
      llvm::Expected<FormalNumericResult> finalized =
          evaluateFormalGemmFinalize(command, accumulator);
      if (!finalized)
        return bulkError(BulkTensorNumericErrorCode::BackendExecutionFailure,
                         llvm::toString(finalized.takeError()));
      destinationValues.push_back(finalized->value);
    }
    llvm::Expected<BulkTensorStorage> packed =
        packIntoTemplate(preflight->gemm->destination, destinationValues,
                         destinationTemplate.getStorage().vec());
    if (!packed)
      return packed.takeError();
    const char *implementationPointer = primitiveDescriptor.impl_info_str();
    std::string implementation =
        implementationPointer ? implementationPointer : "";
    if (implementation.empty())
      return bulkError(BulkTensorNumericErrorCode::BackendDescriptorFailure,
                       "oneDNN returned an empty implementation identity");
    BulkDispatchEvidence evidence{
        /*matmulInvocations=*/1,
        reorderInvocations,
        /*formalFusedMultiplyAdds=*/0,
        totalBytes,
        scratchpadBytes,
        implementation,
        descriptorDigest(resolvedWeights, implementation, scratchpadBytes,
                         environment, command, preflight->format)};
    return detail::UnqualifiedBulkExecutionResult{std::move(*packed),
                                                  std::move(evidence)};
  } catch (const dnnl::error &error) {
    return bulkError(BulkTensorNumericErrorCode::BackendExecutionFailure,
                     llvm::Twine("oneDNN status ") +
                         llvm::Twine(static_cast<int>(error.status)) + ": " +
                         error.what());
  } catch (const std::exception &error) {
    return bulkError(BulkTensorNumericErrorCode::BackendExecutionFailure,
                     error.what());
  }
}

std::string makeFeaturesDigest() {
  llvm::StringMap<bool, llvm::MallocAllocator> features =
      llvm::sys::getHostCPUFeatures();
  std::vector<std::pair<std::string, bool>> ordered;
  ordered.reserve(features.size());
  for (const auto &feature : features)
    ordered.emplace_back(feature.getKey().str(), feature.getValue());
  llvm::sort(ordered, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  llvm::SmallString<1024> payload;
  llvm::raw_svector_ostream stream(payload);
  for (const auto &[name, enabled] : ordered)
    stream << name << '=' << (enabled ? '1' : '0') << '\n';
  return sha256(payload);
}

std::string readHostText(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/true,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return "unavailable";
  return (*buffer)->getBuffer().trim().str();
}

std::string makeStableCPUIdentity() {
  std::string cpuInfo = readHostText("/proc/cpuinfo");
  if (cpuInfo == "unavailable")
    return cpuInfo;
  llvm::SmallString<512> selected;
  llvm::raw_svector_ostream stream(selected);
  for (llvm::StringRef line : llvm::split(llvm::StringRef(cpuInfo), '\n')) {
    llvm::StringRef key = line.take_front(line.find(':')).trim();
    if (key == "vendor_id" || key == "cpu family" || key == "model" ||
        key == "stepping" || key == "microcode")
      stream << line.trim() << '\n';
    if (key == "microcode")
      break;
  }
  return selected.empty() ? "unavailable" : sha256(selected);
}

std::string makeLoaderIdentity() {
  std::string maps = readHostText("/proc/self/maps");
  if (maps == "unavailable")
    return maps;
  for (llvm::StringRef line : llvm::split(llvm::StringRef(maps), '\n')) {
    size_t separator = line.rfind(' ');
    if (separator == llvm::StringRef::npos)
      continue;
    llvm::StringRef path = line.drop_front(separator + 1).trim();
    if (!path.contains("ld-linux") && !path.contains("/ld-"))
      continue;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
    if (!buffer)
      return "unavailable";
    return sha256((*buffer)->getBuffer());
  }
  return "unavailable";
}

std::string makeHostPlatformDigest() {
  llvm::SmallString<2048> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-host-platform-v1");
  appendField(stream, "process_triple", llvm::sys::getProcessTriple());
  appendField(stream, "cpu_identity", makeStableCPUIdentity());
  appendField(stream, "cpu_online",
              readHostText("/sys/devices/system/cpu/online"));
  appendField(stream, "numa_online",
              readHostText("/sys/devices/system/node/online"));
  appendField(stream, "loader", makeLoaderIdentity());
#if defined(__linux__)
  struct utsname identity{};
  if (::uname(&identity) == 0) {
    appendField(stream, "kernel_sysname", identity.sysname);
    appendField(stream, "kernel_release", identity.release);
    appendField(stream, "kernel_version", identity.version);
    appendField(stream, "kernel_machine", identity.machine);
  } else {
    appendField(stream, "kernel", "unavailable");
  }
  appendField(stream, "libc_version", gnu_get_libc_version());
  appendField(stream, "libc_release", gnu_get_libc_release());
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (::sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    appendField(
        stream, "affinity",
        sha256(llvm::ArrayRef<uint8_t>(
            reinterpret_cast<const uint8_t *>(&affinity), sizeof(affinity))));
  else
    appendField(stream, "affinity", "unavailable");
#else
  appendField(stream, "kernel", "unavailable");
  appendField(stream, "libc", "unavailable");
  appendField(stream, "affinity", "unavailable");
#endif
  return sha256(payload);
}

uint32_t readMXCSR() {
#if defined(__x86_64__) || defined(_M_X64)
  return _mm_getcsr();
#else
  return 0;
#endif
}

uint32_t readMXCSRControl() {
  // Sticky exception status bits 0..5 are invocation state, not an execution
  // policy. All control bits, including masks/RC/FTZ/DAZ, are identity.
  return readMXCSR() & ~UINT32_C(0x3f);
}

} // namespace

llvm::Expected<BulkTensorStorage>
BulkTensorStorage::create(NumericTensorKey key, std::vector<uint8_t> storage) {
  llvm::Expected<uint64_t> bytes = getBulkTensorPhysicalBytes(key);
  if (!bytes)
    return bytes.takeError();
  if (*bytes != storage.size())
    return bulkError(BulkTensorNumericErrorCode::InvalidPhysicalStorage,
                     llvm::Twine("tensor requires ") + llvm::Twine(*bytes) +
                         " physical bytes, got " + llvm::Twine(storage.size()));
  return BulkTensorStorage(std::move(key), std::move(storage));
}

llvm::StringRef stringifyBulkQualificationKind(BulkQualificationKind kind) {
  switch (kind) {
  case BulkQualificationKind::BitExact:
    return "bit-exact";
  case BulkQualificationKind::ProfileBounded:
    return "profile-bounded";
  }
  llvm_unreachable("bulk qualification kind is not registered");
}

llvm::StringRef
stringifyBulkTensorNumericErrorCode(BulkTensorNumericErrorCode code) {
  switch (code) {
  case BulkTensorNumericErrorCode::UnsupportedResolvedCommand:
    return "unsupported-resolved-command";
  case BulkTensorNumericErrorCode::UnsupportedFormat:
    return "unsupported-format";
  case BulkTensorNumericErrorCode::InvalidPhysicalLayout:
    return "invalid-physical-layout";
  case BulkTensorNumericErrorCode::InvalidPhysicalStorage:
    return "invalid-physical-storage";
  case BulkTensorNumericErrorCode::InvalidInputEncoding:
    return "invalid-input-encoding";
  case BulkTensorNumericErrorCode::InputArityMismatch:
    return "input-arity-mismatch";
  case BulkTensorNumericErrorCode::AdmissionMismatch:
    return "admission-mismatch";
  case BulkTensorNumericErrorCode::EnvironmentMismatch:
    return "environment-mismatch";
  case BulkTensorNumericErrorCode::WorkCountOverflow:
    return "work-count-overflow";
  case BulkTensorNumericErrorCode::TotalByteBudgetExceeded:
    return "total-byte-budget-exceeded";
  case BulkTensorNumericErrorCode::ScratchpadBudgetExceeded:
    return "scratchpad-budget-exceeded";
  case BulkTensorNumericErrorCode::ReorderBudgetExceeded:
    return "reorder-budget-exceeded";
  case BulkTensorNumericErrorCode::BackendConfigurationFailure:
    return "backend-configuration-failure";
  case BulkTensorNumericErrorCode::BackendDescriptorFailure:
    return "backend-descriptor-failure";
  case BulkTensorNumericErrorCode::BackendExecutionFailure:
    return "backend-execution-failure";
  case BulkTensorNumericErrorCode::BackendOutputMismatch:
    return "backend-output-mismatch";
  }
  llvm_unreachable("bulk tensor numeric error code is not registered");
}

char BulkTensorNumericError::ID;

void BulkTensorNumericError::log(llvm::raw_ostream &stream) const {
  stream << "bulk tensor numeric " << stringifyBulkTensorNumericErrorCode(code)
         << ": " << detail;
}

std::error_code BulkTensorNumericError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::StringRef getBulkAdapterIdentityDigest() {
  static const std::string digest = [] {
    llvm::SmallString<512> payload;
    llvm::raw_svector_ostream stream(payload);
    appendField(stream, "schema", "wafer-bulk-adapter-v1");
    appendField(stream, "input", "target-owned-physical-codec-layout");
    appendField(stream, "backend_dense_format", "f32");
    appendField(stream, "backend_primitive",
                "one-matmul-optional-weight-reorder");
    appendField(stream, "destination", "formal-gemm-finalize-and-target-pack");
    appendField(stream, "atomic_commit", "private-temporary-before-publish");
    return sha256(payload);
  }();
  return digest;
}

llvm::Expected<BulkExecutionEnvironment>
createManagedBulkExecutionEnvironment() {
  static std::mutex mutex;
  static std::optional<BulkExecutionEnvironment> cached;
  static std::optional<std::string> failure;
  std::lock_guard<std::mutex> lock(mutex);
  if (cached) {
    if (std::fegetround() != cached->getFloatingRoundingMode() ||
        readMXCSRControl() != cached->getMXCSR())
      return bulkError(BulkTensorNumericErrorCode::EnvironmentMismatch,
                       "current caller fenv/MXCSR control no longer matches "
                       "the managed sequential environment");
    if (llvm::sys::getHostCPUName() != cached->getHostCPUName() ||
        makeFeaturesDigest() != cached->getHostFeaturesDigest() ||
        makeHostPlatformDigest() != cached->getHostPlatformDigest())
      return bulkError(BulkTensorNumericErrorCode::EnvironmentMismatch,
                       "current host CPU/platform/affinity no longer matches "
                       "the managed sequential environment");
    return *cached;
  }
  if (failure)
    return bulkError(BulkTensorNumericErrorCode::BackendConfigurationFailure,
                     *failure);
  auto fail = [&](const llvm::Twine &detail)
      -> llvm::Expected<BulkExecutionEnvironment> {
    failure = detail.str();
    return bulkError(BulkTensorNumericErrorCode::BackendConfigurationFailure,
                     *failure);
  };
  if (std::fegetround() != FE_TONEAREST)
    return fail("initial sequential profile requires FE_TONEAREST");
#if !defined(__x86_64__) && !defined(_M_X64)
  return fail("initial sequential profile requires x86-64 MXCSR control");
#endif
  const uint32_t mxcsr = readMXCSRControl();
  if ((mxcsr & UINT32_C(0x8040)) != 0 || (mxcsr & UINT32_C(0x6000)) != 0 ||
      (mxcsr & UINT32_C(0x1f80)) != UINT32_C(0x1f80))
    return fail("initial sequential profile requires MXCSR RNE, FTZ=0, "
                "DAZ=0 and all exceptions masked");
  if (dnnl::set_max_cpu_isa(dnnl::cpu_isa::isa_default) !=
      dnnl::status::success)
    return fail(
        "oneDNN max CPU ISA policy was already initialized or rejected");
  if (dnnl::set_cpu_isa_hints(dnnl::cpu_isa_hints::no_hints) !=
      dnnl::status::success)
    return fail(
        "oneDNN CPU ISA hints policy was already initialized or rejected");
  try {
    dnnl::set_primitive_cache_capacity(0);
  } catch (const dnnl::error &error) {
    return fail(llvm::Twine("oneDNN primitive cache policy failed: ") +
                error.what());
  }
  const dnnl::version_t *runtimeVersion = dnnl::version();
  if (!runtimeVersion || runtimeVersion->major != DNNL_VERSION_MAJOR ||
      runtimeVersion->minor != DNNL_VERSION_MINOR ||
      runtimeVersion->patch != DNNL_VERSION_PATCH || !runtimeVersion->hash ||
      llvm::StringRef(runtimeVersion->hash) != DNNL_VERSION_HASH ||
      runtimeVersion->cpu_runtime != DNNL_RUNTIME_SEQ)
    return fail("loaded oneDNN version/commit/runtime identity mismatch");
  dnnl::cpu_isa effectiveISA = dnnl::get_effective_cpu_isa();
  const char *isaName =
      dnnl_cpu_isa2str(static_cast<dnnl_cpu_isa_t>(effectiveISA));
  if (!isaName || !*isaName)
    return fail("oneDNN returned an empty effective CPU ISA");

  llvm::SmallString<512> backendPayload;
  llvm::raw_svector_ostream backendStream(backendPayload);
  appendField(backendStream, "schema", "wafer-bulk-backend-v1");
  appendField(backendStream, "name", "oneDNN");
  appendField(backendStream, "version", WAFER_BULK_ONEDNN_VERSION);
  appendField(backendStream, "commit", WAFER_BULK_ONEDNN_COMMIT);
  const std::string dependencyRecordDigest =
      (llvm::Twine("sha256:") + WAFER_BULK_DEPENDENCY_RECORD_SHA256).str();
  const std::string libraryDigest =
      (llvm::Twine("sha256:") + WAFER_BULK_ONEDNN_LIBRARY_SHA256).str();
  appendField(backendStream, "dependency_record", dependencyRecordDigest);
  appendField(backendStream, "library", libraryDigest);
  BulkBackendIdentity backend("oneDNN", WAFER_BULK_ONEDNN_VERSION,
                              WAFER_BULK_ONEDNN_COMMIT, dependencyRecordDigest,
                              libraryDigest, sha256(backendPayload));

  std::string cpuName = llvm::sys::getHostCPUName().str();
  std::string featuresDigest = makeFeaturesDigest();
  std::string hostPlatformDigest = makeHostPlatformDigest();
  llvm::SmallString<1024> environmentPayload;
  llvm::raw_svector_ostream environmentStream(environmentPayload);
  appendField(environmentStream, "schema", "wafer-bulk-environment-v1");
  appendField(environmentStream, "backend", backend.getDigest());
  appendField(environmentStream, "host_cpu", cpuName);
  appendField(environmentStream, "host_features", featuresDigest);
  appendField(environmentStream, "host_platform", hostPlatformDigest);
  appendField(environmentStream, "effective_isa", isaName);
  appendField(environmentStream, "fenv_round", std::fegetround());
  appendField(environmentStream, "mxcsr", mxcsr);
  appendField(environmentStream, "thread_runtime", "seq-caller-worker-v1");
  cached = BulkExecutionEnvironment(
      std::move(backend), std::move(cpuName), std::move(featuresDigest),
      std::move(hostPlatformDigest), isaName, std::fegetround(), mxcsr,
      "seq-caller-worker-v1", sha256(environmentPayload));
  return *cached;
}

llvm::Expected<uint64_t>
getBulkTensorPhysicalBytes(const NumericTensorKey &key) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return bulkError(BulkTensorNumericErrorCode::InvalidPhysicalLayout,
                     llvm::toString(bytes.takeError()));
  return *bytes;
}

llvm::Expected<std::vector<RawLogicalValue>>
unpackBulkTensorLogicalValues(const BulkTensorStorage &tensor) {
  llvm::Expected<std::vector<RawLogicalValue>> values =
      unpackPhysicalTensorLogicalValues(tensor.getKey(), tensor.getStorage());
  if (!values)
    return bulkError(BulkTensorNumericErrorCode::InvalidInputEncoding,
                     llvm::toString(values.takeError()));
  return values;
}

llvm::Expected<BulkTensorStorage>
packBulkTensorLogicalValues(const NumericTensorKey &key,
                            llvm::ArrayRef<RawLogicalValue> values,
                            uint8_t paddingFill) {
  llvm::Expected<uint64_t> bytes = getBulkTensorPhysicalBytes(key);
  if (!bytes)
    return bytes.takeError();
  if (*bytes > std::numeric_limits<size_t>::max())
    return bulkError(BulkTensorNumericErrorCode::WorkCountOverflow,
                     "physical tensor footprint exceeds host size_t");
  return packIntoTemplate(
      key, values,
      std::vector<uint8_t>(static_cast<size_t>(*bytes), paddingFill));
}

std::string
computeBulkTensorPayloadDigest(llvm::ArrayRef<BulkTensorStorage> tensors) {
  llvm::SmallString<1024> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-bulk-payload-v1");
  appendField(stream, "tensor_count", tensors.size());
  for (auto [index, tensor] : llvm::enumerate(tensors)) {
    appendField(stream,
                (llvm::Twine("tensor_") + llvm::Twine(index) + "_key").str(),
                tensor.getKey().getDigest());
    appendField(stream,
                (llvm::Twine("tensor_") + llvm::Twine(index) + "_bytes").str(),
                sha256(tensor.getStorage()));
  }
  return sha256(payload);
}

std::string computeBulkTensorStorageDigest(const BulkTensorStorage &tensor) {
  llvm::SmallString<512> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-bulk-storage-v1");
  appendField(stream, "key", tensor.getKey().getDigest());
  appendField(stream, "bytes", sha256(tensor.getStorage()));
  return sha256(payload);
}

namespace detail {

llvm::Expected<UnqualifiedBulkExecutionResult>
executeBulkTensorForQualification(const BulkExecutionEnvironment &environment,
                                  const ResolvedNumericCommand &command,
                                  llvm::ArrayRef<BulkTensorStorage> inputs,
                                  const BulkTensorStorage &destinationTemplate,
                                  BulkNumericWorkBudget budget) {
  llvm::Expected<BulkExecutionEnvironment> current =
      createManagedBulkExecutionEnvironment();
  if (!current)
    return current.takeError();
  if (current->getDigest() != environment.getDigest())
    return bulkError(BulkTensorNumericErrorCode::EnvironmentMismatch,
                     "requested bulk environment is not current");
  return executeOneDNN(environment, command, inputs, destinationTemplate,
                       budget);
}

} // namespace detail

llvm::Expected<BulkTensorNumericResult>
executeAdmittedBulkTensorNumeric(const BulkExecutionEnvironment &environment,
                                 const BulkBackendAdmission &admission,
                                 const ResolvedNumericCommand &command,
                                 llvm::ArrayRef<BulkTensorStorage> inputs,
                                 const BulkTensorStorage &destinationTemplate,
                                 BulkNumericWorkBudget budget) {
  if (admission.getEnvironmentDigest() != environment.getDigest())
    return bulkError(BulkTensorNumericErrorCode::EnvironmentMismatch,
                     "bulk admission does not match the current environment");
  if (admission.getAdapterDigest() != getBulkAdapterIdentityDigest() ||
      !command.getSemantics() ||
      admission.getSemanticProfileDigest() !=
          command.getSemantics()->getDigest() ||
      admission.getResolutionDigest() != command.getDigest() ||
      admission.getInputPayloadDigest() !=
          computeBulkTensorPayloadDigest(inputs) ||
      admission.getDestinationTemplateDigest() !=
          computeBulkTensorStorageDigest(destinationTemplate))
    return bulkError(BulkTensorNumericErrorCode::AdmissionMismatch,
                     "bulk command, payload or destination template is not "
                     "the frozen qualified row");
  llvm::Expected<detail::UnqualifiedBulkExecutionResult> result =
      detail::executeBulkTensorForQualification(environment, command, inputs,
                                                destinationTemplate, budget);
  if (!result)
    return result.takeError();
  if (result->evidence.implementation !=
          admission.getExpectedImplementation() ||
      result->evidence.resolvedDescriptorDigest !=
          admission.getExpectedResolvedDescriptorDigest())
    return bulkError(BulkTensorNumericErrorCode::AdmissionMismatch,
                     "oneDNN implementation or resolved descriptor changed "
                     "from the frozen qualified row");
  if (computeBulkTensorStorageDigest(result->destination) !=
      admission.getExpectedBackendOutputDigest())
    return bulkError(BulkTensorNumericErrorCode::BackendOutputMismatch,
                     "oneDNN output changed from the frozen validated result");
  return BulkTensorNumericResult{std::move(result->destination),
                                 admission.getFormalFlags(),
                                 std::move(result->evidence)};
}

} // namespace wafer
