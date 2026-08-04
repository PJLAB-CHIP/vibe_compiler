//===- TargetFormat.cpp - Closed target format registry ------------------===//

#include "Wafer/Target/TargetFormat.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstddef>

namespace wafer {
namespace {

using Format = LogicalFormat;
using Category = LogicalFormatCategory;
using Engine = TargetFormatEngine;
using Constraint = TargetFormatConstraint;
using Parameter = TargetConvertParameterKind;

constexpr LogicalFormatDescriptor kLogicalFormats[] = {
    {Format::I8, "i8", 8, 8, Category::SignedInteger, 0, 0, UINT64_C(0xff),
     false, false, false, false},
    {Format::I16, "i16", 16, 16, Category::SignedInteger, 0, 0,
     UINT64_C(0xffff), false, false, false, false},
    {Format::F16, "f16", 16, 16, Category::BinaryFloatingPoint, 5, 11,
     UINT64_C(0xffff), false, true, true, true},
    {Format::BF16, "bf16", 16, 16, Category::BinaryFloatingPoint, 8, 8,
     UINT64_C(0xffff), false, true, true, true},
    {Format::I32, "i32", 32, 32, Category::SignedInteger, 0, 0,
     UINT64_C(0xffffffff), false, false, false, false},
    {Format::F32, "f32", 32, 32, Category::BinaryFloatingPoint, 8, 24,
     UINT64_C(0xffffffff), false, true, true, true},
    {Format::TF32, "tf32", 32, 19, Category::BinaryFloatingPoint, 8, 11,
     UINT64_C(0xffffe000), false, true, true, true},
    {Format::Bool, "bool", 1, 1, Category::Boolean, 0, 0, UINT64_C(0x1), true,
     false, false, false},
    {Format::U8, "u8", 8, 8, Category::UnsignedInteger, 0, 0, UINT64_C(0xff),
     false, false, false, false},
    {Format::U16, "u16", 16, 16, Category::UnsignedInteger, 0, 0,
     UINT64_C(0xffff), false, false, false, false},
    {Format::U32, "u32", 32, 32, Category::UnsignedInteger, 0, 0,
     UINT64_C(0xffffffff), false, false, false, false},
    {Format::I64, "i64", 64, 64, Category::SignedInteger, 0, 0, UINT64_MAX,
     false, false, false, false},
    {Format::U64, "u64", 64, 64, Category::UnsignedInteger, 0, 0, UINT64_MAX,
     false, false, false, false},
};

// The only hand-written public Data_Format code facts. These rows describe
// the vendor enum for the current target, not permission for any engine to
// emit it.
constexpr TargetDataFormatCodeRecord kTargetDataFormatCodes[] = {
    {Format::I8, 0},   {Format::I16, 1},
    {Format::F16, 2},  {Format::BF16, 3},
    {Format::I32, 4},  {Format::F32, 5},
    {Format::TF32, 6}, {Format::Bool, 7},
    {Format::U8, 8},   {Format::U16, 9},
    {Format::U32, 10}, {Format::I64, 11},
    {Format::U64, 12},
};

constexpr std::optional<uint8_t> findDataFormatCode(Format format) {
  for (const TargetDataFormatCodeRecord &record : kTargetDataFormatCodes)
    if (record.format == format)
      return record.dataFormatCode;
  return std::nullopt;
}

constexpr TargetFormatEngine kTargetFormatEngines[] = {
    Engine::RDMA, Engine::WDMA, Engine::TDMA, Engine::CT, Engine::NE,
};

constexpr TargetFormatEncodingRecord
supported(Engine engine, Format format,
          Constraint constraint = Constraint::None) {
  return {engine, format, constraint, *findDataFormatCode(format)};
}

// This is intentionally an explicit 65-row closed matrix. Adding a logical
// format or engine must add each new row and keep the completeness
// assertion below true; absence is never interpreted as unsupported.
constexpr TargetFormatEncodingRecord kTargetFormatEncodings[] = {
    supported(Engine::RDMA, Format::I8),
    supported(Engine::RDMA, Format::I16),
    supported(Engine::RDMA, Format::F16),
    supported(Engine::RDMA, Format::BF16),
    supported(Engine::RDMA, Format::I32),
    supported(Engine::RDMA, Format::F32),
    supported(Engine::RDMA, Format::TF32),
    supported(Engine::RDMA, Format::Bool,
              Constraint::BitpackedDMA),
    supported(Engine::RDMA, Format::U8),
    supported(Engine::RDMA, Format::U16),
    supported(Engine::RDMA, Format::U32),
    supported(Engine::RDMA, Format::I64),
    supported(Engine::RDMA, Format::U64),

    supported(Engine::WDMA, Format::I8),
    supported(Engine::WDMA, Format::I16),
    supported(Engine::WDMA, Format::F16),
    supported(Engine::WDMA, Format::BF16),
    supported(Engine::WDMA, Format::I32),
    supported(Engine::WDMA, Format::F32),
    supported(Engine::WDMA, Format::TF32),
    supported(Engine::WDMA, Format::Bool,
              Constraint::BitpackedDMA),
    supported(Engine::WDMA, Format::U8),
    supported(Engine::WDMA, Format::U16),
    supported(Engine::WDMA, Format::U32),
    supported(Engine::WDMA, Format::I64),
    supported(Engine::WDMA, Format::U64),

    supported(Engine::TDMA, Format::I8),
    supported(Engine::TDMA, Format::I16),
    supported(Engine::TDMA, Format::F16),
    supported(Engine::TDMA, Format::BF16),
    supported(Engine::TDMA, Format::I32),
    supported(Engine::TDMA, Format::F32),
    supported(Engine::TDMA, Format::TF32),
    supported(Engine::TDMA, Format::Bool, Constraint::BitpackedLayout),
    supported(Engine::TDMA, Format::U8),
    supported(Engine::TDMA, Format::U16),
    supported(Engine::TDMA, Format::U32),
    supported(Engine::TDMA, Format::I64),
    supported(Engine::TDMA, Format::U64),

    supported(Engine::CT, Format::I8),
    supported(Engine::CT, Format::I16),
    supported(Engine::CT, Format::F16),
    supported(Engine::CT, Format::BF16),
    supported(Engine::CT, Format::I32),
    supported(Engine::CT, Format::F32),
    supported(Engine::CT, Format::TF32),
    supported(Engine::CT, Format::Bool, Constraint::BitpackedLayout),
    supported(Engine::CT, Format::U8),
    supported(Engine::CT, Format::U16),
    supported(Engine::CT, Format::U32),
    supported(Engine::CT, Format::I64),
    supported(Engine::CT, Format::U64),

    supported(Engine::NE, Format::I8),
    supported(Engine::NE, Format::I16),
    supported(Engine::NE, Format::F16),
    supported(Engine::NE, Format::BF16),
    supported(Engine::NE, Format::I32),
    supported(Engine::NE, Format::F32),
    supported(Engine::NE, Format::TF32),
    supported(Engine::NE, Format::Bool, Constraint::BitpackedLayout),
    supported(Engine::NE, Format::U8),
    supported(Engine::NE, Format::U16),
    supported(Engine::NE, Format::U32),
    supported(Engine::NE, Format::I64),
    supported(Engine::NE, Format::U64),
};

constexpr TargetConvertRoute kTargetConvertRoutes[] = {
    {139, "int8_fp16", Format::I8, Format::F16, Parameter::ZeroPoint},
    {140, "int8_bf16", Format::I8, Format::BF16,
     Parameter::ZeroPoint},
    {141, "int8_fp32", Format::I8, Format::F32, Parameter::ZeroPoint},
    {142, "int8_tf32", Format::I8, Format::TF32,
     Parameter::ZeroPoint},
    {143, "int16_fp16", Format::I16, Format::F16, Parameter::None},
    {144, "int16_bf16", Format::I16, Format::BF16,
     Parameter::RoundingMode},
    {145, "int16_fp32", Format::I16, Format::F32,
     Parameter::RoundingMode},
    {146, "int16_tf32", Format::I16, Format::TF32,
     Parameter::RoundingMode},
    {147, "int32_fp16", Format::I32, Format::F16,
     Parameter::RoundingMode},
    {148, "int32_bf16", Format::I32, Format::BF16,
     Parameter::RoundingMode},
    {149, "int32_fp32", Format::I32, Format::F32,
     Parameter::RoundingMode},
    {150, "int32_tf32", Format::I32, Format::TF32,
     Parameter::RoundingMode},
    {151, "bf16_int8", Format::BF16, Format::I8, Parameter::None},
    {152, "bf16_int16", Format::BF16, Format::I16,
     Parameter::RoundingMode},
    {153, "bf16_int32", Format::BF16, Format::I32,
     Parameter::RoundingMode},
    {154, "bf16_fp16", Format::BF16, Format::F16, Parameter::None},
    {155, "bf16_fp32", Format::BF16, Format::F32, Parameter::None},
    {156, "bf16_tf32", Format::BF16, Format::TF32, Parameter::None},
    {157, "fp16_int8", Format::F16, Format::I8,
     Parameter::RoundingMode},
    {158, "fp16_int16", Format::F16, Format::I16,
     Parameter::RoundingMode},
    {159, "fp16_int32", Format::F16, Format::I32,
     Parameter::RoundingMode},
    {160, "fp16_bf16", Format::F16, Format::BF16,
     Parameter::RoundingMode},
    {161, "fp16_fp32", Format::F16, Format::F32, Parameter::None},
    {162, "fp16_tf32", Format::F16, Format::TF32, Parameter::None},
    {163, "fp32_int8", Format::F32, Format::I8,
     Parameter::RoundingMode},
    {164, "fp32_int16", Format::F32, Format::I16,
     Parameter::RoundingMode},
    {165, "fp32_int32", Format::F32, Format::I32,
     Parameter::RoundingMode},
    {166, "fp32_fp16", Format::F32, Format::F16,
     Parameter::RoundingMode},
    {167, "fp32_bf16", Format::F32, Format::BF16,
     Parameter::RoundingMode},
    {168, "fp32_tf32", Format::F32, Format::TF32,
     Parameter::RoundingMode},
    {169, "tf32_int8", Format::TF32, Format::I8,
     Parameter::RoundingMode},
    {170, "tf32_int16", Format::TF32, Format::I16,
     Parameter::RoundingMode},
    {171, "tf32_int32", Format::TF32, Format::I32,
     Parameter::RoundingMode},
    {172, "tf32_fp16", Format::TF32, Format::F16, Parameter::None},
    {173, "tf32_bf16", Format::TF32, Format::BF16,
     Parameter::RoundingMode},
    {174, "tf32_fp32", Format::TF32, Format::F32, Parameter::None},
};

template <typename T, size_t N> constexpr size_t arrayLength(const T (&)[N]) {
  return N;
}

constexpr uint64_t lowBitMask(uint8_t width) {
  return width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1;
}

constexpr bool hasCompleteAndValidLogicalFormatRegistry() {
  if (arrayLength(kLogicalFormats) != 13)
    return false;
  for (size_t index = 0; index < arrayLength(kLogicalFormats); ++index) {
    const LogicalFormatDescriptor &descriptor = kLogicalFormats[index];
    if (descriptor.storageBits == 0 || descriptor.storageBits > 64 ||
        descriptor.semanticBits == 0 ||
        descriptor.semanticBits > descriptor.storageBits)
      return false;
    const uint8_t semanticShift =
        descriptor.storageBits - descriptor.semanticBits;
    if (descriptor.canonicalMask !=
        (lowBitMask(descriptor.semanticBits) << semanticShift))
      return false;

    for (size_t other = index + 1; other < arrayLength(kLogicalFormats);
         ++other)
      if (kLogicalFormats[other].format == descriptor.format)
        return false;

    switch (descriptor.category) {
    case Category::SignedInteger:
    case Category::UnsignedInteger:
      if (descriptor.exponentBits != 0 || descriptor.precisionBits != 0 ||
          descriptor.bitpacked || descriptor.hasInfinity || descriptor.hasNaN ||
          descriptor.hasSubnormal)
        return false;
      break;
    case Category::BinaryFloatingPoint:
      if (descriptor.exponentBits == 0 || descriptor.precisionBits <= 1 ||
          descriptor.semanticBits !=
              descriptor.exponentBits + descriptor.precisionBits ||
          descriptor.bitpacked || !descriptor.hasInfinity ||
          !descriptor.hasNaN || !descriptor.hasSubnormal)
        return false;
      break;
    case Category::Boolean:
      if (descriptor.storageBits != 1 || descriptor.semanticBits != 1 ||
          descriptor.exponentBits != 0 || descriptor.precisionBits != 0 ||
          !descriptor.bitpacked || descriptor.hasInfinity ||
          descriptor.hasNaN || descriptor.hasSubnormal)
        return false;
      break;
    }
  }
  return true;
}

constexpr bool hasCompleteAndUniqueDataFormatCodeRegistry() {
  if (arrayLength(kTargetDataFormatCodes) != arrayLength(kLogicalFormats))
    return false;
  for (size_t index = 0; index < arrayLength(kTargetDataFormatCodes); ++index) {
    const TargetDataFormatCodeRecord &record = kTargetDataFormatCodes[index];
    size_t formatCount = 0;
    for (const LogicalFormatDescriptor &format : kLogicalFormats)
      if (format.format == record.format)
        ++formatCount;
    if (formatCount != 1)
      return false;
    for (size_t other = index + 1; other < arrayLength(kTargetDataFormatCodes);
         ++other) {
      const TargetDataFormatCodeRecord &candidate =
          kTargetDataFormatCodes[other];
      if (candidate.format == record.format ||
          candidate.dataFormatCode == record.dataFormatCode)
        return false;
    }
  }
  return true;
}

constexpr bool hasCompleteAndConsistentEncodingRegistry() {
  if (arrayLength(kTargetFormatEncodings) !=
      arrayLength(kTargetFormatEngines) * arrayLength(kLogicalFormats))
    return false;

  for (const TargetFormatEncodingRecord &record : kTargetFormatEncodings)
    if (record.dataFormatCode != *findDataFormatCode(record.format))
      return false;

  for (size_t index = 0; index < arrayLength(kTargetFormatEncodings); ++index) {
    const TargetFormatEncodingRecord &record = kTargetFormatEncodings[index];
    for (size_t other = index + 1; other < arrayLength(kTargetFormatEncodings);
         ++other) {
      const TargetFormatEncodingRecord &candidate =
          kTargetFormatEncodings[other];
      if (candidate.engine == record.engine &&
          candidate.dataFormatCode == record.dataFormatCode)
        return false;
    }
  }

  for (TargetFormatEngine engine : kTargetFormatEngines) {
    for (const LogicalFormatDescriptor &format : kLogicalFormats) {
      size_t count = 0;
      for (const TargetFormatEncodingRecord &record : kTargetFormatEncodings)
        if (record.engine == engine &&
            record.format == format.format)
          ++count;
      if (count != 1)
        return false;
    }
  }
  return true;
}

constexpr bool hasCompleteAndUniqueConvertRegistry() {
  if (arrayLength(kTargetConvertRoutes) != 36)
    return false;
  for (size_t index = 0; index < arrayLength(kTargetConvertRoutes); ++index) {
    const TargetConvertRoute &route = kTargetConvertRoutes[index];
    if (route.opcode != 139 + index ||
        route.source == route.destination)
      return false;
    for (size_t other = index + 1; other < arrayLength(kTargetConvertRoutes);
         ++other) {
      const TargetConvertRoute &candidate = kTargetConvertRoutes[other];
      if (route.opcode == candidate.opcode ||
          (route.source == candidate.source &&
           route.destination == candidate.destination))
        return false;
    }
  }
  return true;
}

static_assert(arrayLength(kLogicalFormats) == 13,
              "logical storage registry must enumerate 13 formats");
static_assert(hasCompleteAndValidLogicalFormatRegistry(),
              "logical format metadata must be complete, unique, and "
              "internally consistent");
static_assert(hasCompleteAndUniqueDataFormatCodeRegistry(),
              "target Data_Format registry must contain one unique code for "
              "each logical format");
static_assert(arrayLength(kTargetFormatEngines) == 5,
              "target format engine registry must remain explicit");
static_assert(hasCompleteAndConsistentEncodingRegistry(),
              "target format encoding matrix must contain one explicit row "
              "for every engine-format pair");
static_assert(hasCompleteAndUniqueConvertRegistry(),
              "target convert registry must exactly enumerate opcodes "
              "139..174");

} // namespace

llvm::ArrayRef<LogicalFormatDescriptor> getLogicalFormatDescriptors() {
  return kLogicalFormats;
}

const LogicalFormatDescriptor *
findLogicalFormatDescriptor(LogicalFormat format) {
  for (const LogicalFormatDescriptor &descriptor : kLogicalFormats)
    if (descriptor.format == format)
      return &descriptor;
  return nullptr;
}

llvm::ArrayRef<TargetDataFormatCodeRecord> getTargetDataFormatCodeRecords() {
  return kTargetDataFormatCodes;
}

const TargetDataFormatCodeRecord *
findTargetDataFormatCode(LogicalFormat format) {
  for (const TargetDataFormatCodeRecord &record : kTargetDataFormatCodes)
    if (record.format == format)
      return &record;
  return nullptr;
}

llvm::Expected<LogicalFormat>
parseLogicalFormat(llvm::StringRef canonicalSpelling) {
  for (const LogicalFormatDescriptor &descriptor : kLogicalFormats)
    if (descriptor.canonicalSpelling == canonicalSpelling)
      return descriptor.format;
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown logical format '%s'",
                                 canonicalSpelling.str().c_str());
}

llvm::StringRef stringifyLogicalFormat(LogicalFormat format) {
  if (const LogicalFormatDescriptor *descriptor =
          findLogicalFormatDescriptor(format))
    return descriptor->canonicalSpelling;
  llvm_unreachable("logical format is not registered");
}

llvm::ArrayRef<TargetFormatEngine> getTargetFormatEngines() {
  return kTargetFormatEngines;
}

llvm::Expected<TargetFormatEngine>
parseTargetFormatEngine(llvm::StringRef canonicalSpelling) {
  for (TargetFormatEngine engine : kTargetFormatEngines)
    if (stringifyTargetFormatEngine(engine) == canonicalSpelling)
      return engine;
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown target format engine '%s'",
                                 canonicalSpelling.str().c_str());
}

llvm::StringRef stringifyTargetFormatEngine(TargetFormatEngine engine) {
  switch (engine) {
  case Engine::RDMA:
    return "rdma";
  case Engine::WDMA:
    return "wdma";
  case Engine::TDMA:
    return "tdma";
  case Engine::CT:
    return "ct";
  case Engine::NE:
    return "ne";
  }
  llvm_unreachable("target format engine is not registered");
}

llvm::ArrayRef<TargetFormatEncodingRecord> getTargetFormatEncodingRecords() {
  return kTargetFormatEncodings;
}

const TargetFormatEncodingRecord *
findTargetFormatEncoding(TargetFormatEngine engine, LogicalFormat format) {
  for (const TargetFormatEncodingRecord &record : kTargetFormatEncodings)
    if (record.engine == engine &&
        record.format == format)
      return &record;
  return nullptr;
}

llvm::Expected<LogicalFormat> decodeTargetFormat(TargetFormatEngine engine,
                                                 uint32_t dataFormatCode) {
  const TargetFormatEncodingRecord *match = nullptr;
  for (const TargetFormatEncodingRecord &record :
       getTargetFormatEncodingRecords()) {
    if (record.engine != engine || record.dataFormatCode != dataFormatCode)
      continue;
    if (match)
      return llvm::createStringError(
          "target format code is not unique for the current target and "
          "engine");
    match = &record;
  }
  if (!match)
    return llvm::createStringError(
        "target format code is unsupported for the current target and "
        "engine");
  return match->format;
}

llvm::StringRef
stringifyTargetFormatConstraint(TargetFormatConstraint constraint) {
  switch (constraint) {
  case Constraint::None:
    return "none";
  case Constraint::BitpackedLayout:
    return "bitpacked-layout";
  case Constraint::BitpackedDMA:
    return "bitpacked-dma";
  }
  llvm_unreachable("target format constraint is not registered");
}

llvm::ArrayRef<TargetConvertRoute> getTargetConvertRoutes() {
  return kTargetConvertRoutes;
}

const TargetConvertRoute *findTargetConvertRoute(uint16_t opcode) {
  for (const TargetConvertRoute &route : kTargetConvertRoutes)
    if (route.opcode == opcode)
      return &route;
  return nullptr;
}

const TargetConvertRoute *findTargetConvertRoute(LogicalFormat source,
                                                 LogicalFormat destination) {
  for (const TargetConvertRoute &route : kTargetConvertRoutes)
    if (route.source == source &&
        route.destination == destination)
      return &route;
  return nullptr;
}

const TargetConvertRoute *
findTargetConvertRoute(llvm::StringRef canonicalSpelling) {
  for (const TargetConvertRoute &route : kTargetConvertRoutes)
    if (route.canonicalSpelling == canonicalSpelling)
      return &route;
  return nullptr;
}

llvm::StringRef
stringifyTargetConvertParameterKind(TargetConvertParameterKind kind) {
  switch (kind) {
  case Parameter::None:
    return "none";
  case Parameter::RoundingMode:
    return "rounding-mode";
  case Parameter::ZeroPoint:
    return "zero-point";
  }
  llvm_unreachable("target convert parameter kind is not registered");
}

} // namespace wafer
