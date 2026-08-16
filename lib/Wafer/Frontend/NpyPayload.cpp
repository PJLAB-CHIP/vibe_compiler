//===- NpyPayload.cpp - Frontend NPY payload codec ----------------------===//

#include "ProgramInternal.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace wafer::frontend::program_detail {

namespace {

/// Upper bound for a parsed NPY header. Version 2 headers can legally declare
/// larger lengths, but no current Wafer producer needs them and unbounded
/// header reads would break the bounded-read contract.
constexpr uint64_t kMaximumNpyHeaderBytes = 1 << 20;

struct NpyPayloadMetadata {
  uint64_t fileSize = 0;
  uint64_t dataOffset = 0;
  std::string descr;
  bool fortranOrder = false;
  std::vector<int64_t> shape;
};

std::optional<llvm::StringRef> findNpyFieldValue(llvm::StringRef header,
                                                 llvm::StringRef field) {
  std::string singleQuoted = (Twine("'") + field + "'").str();
  std::string doubleQuoted = (Twine("\"") + field + "\"").str();
  size_t key = header.find(singleQuoted);
  if (key == llvm::StringRef::npos)
    key = header.find(doubleQuoted);
  if (key == llvm::StringRef::npos)
    return std::nullopt;

  size_t colon = header.find(':', key);
  if (colon == llvm::StringRef::npos)
    return std::nullopt;
  return header.drop_front(colon + 1).ltrim();
}

std::optional<std::string> parseNpyStringField(llvm::StringRef header,
                                               llvm::StringRef field) {
  std::optional<llvm::StringRef> value = findNpyFieldValue(header, field);
  if (!value || value->empty())
    return std::nullopt;
  char quote = value->front();
  if (quote != '\'' && quote != '"')
    return std::nullopt;
  llvm::StringRef rest = value->drop_front();
  size_t end = rest.find(quote);
  if (end == llvm::StringRef::npos)
    return std::nullopt;
  return rest.take_front(end).str();
}

std::optional<bool> parseNpyBoolField(llvm::StringRef header,
                                      llvm::StringRef field) {
  std::optional<llvm::StringRef> value = findNpyFieldValue(header, field);
  if (!value)
    return std::nullopt;
  if (value->starts_with("False") || value->starts_with("false"))
    return false;
  if (value->starts_with("True") || value->starts_with("true"))
    return true;
  return std::nullopt;
}

std::optional<std::vector<int64_t>> parseNpyShapeField(llvm::StringRef header) {
  std::optional<llvm::StringRef> value = findNpyFieldValue(header, "shape");
  if (!value)
    return std::nullopt;

  size_t open = value->find('(');
  size_t close = value->find(')');
  if (open == llvm::StringRef::npos || close == llvm::StringRef::npos ||
      close < open)
    return std::nullopt;

  std::vector<int64_t> shape;
  llvm::StringRef body = value->slice(open + 1, close);
  while (!body.empty()) {
    auto split = body.split(',');
    llvm::StringRef token = split.first.trim();
    if (!token.empty()) {
      int64_t dim = 0;
      if (token.getAsInteger(10, dim) || dim < 0)
        return std::nullopt;
      shape.push_back(dim);
    }
    body = split.second;
  }
  return shape;
}

FailureOr<NpyPayloadMetadata>
readNpyPayloadMetadata(llvm::StringRef path, llvm::StringRef displayName,
                       llvm::raw_ostream &diagnostics) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError) {
    rejectProgramDirectory(
        ("failed to read npy payload file: " + displayName).str(), diagnostics);
    return failure();
  }

  llvm::StringRef bytes = (*bufferOrError)->getBuffer();
  if (bytes.size() < 10 ||
      bytes.take_front(6) != llvm::StringRef("\x93NUMPY", 6)) {
    rejectProgramDirectory(
        ("npy payload is missing magic: " + displayName).str(), diagnostics);
    return failure();
  }

  auto byte = [&](size_t index) -> uint64_t {
    return static_cast<unsigned char>(bytes[index]);
  };

  uint64_t major = byte(6);
  uint64_t headerLen = 0;
  uint64_t headerOffset = 0;
  if (major == 1) {
    headerOffset = 10;
    headerLen = byte(8) | (byte(9) << 8);
  } else if (major == 2) {
    if (bytes.size() < 12) {
      rejectProgramDirectory(("truncated npy v2 header: " + displayName).str(),
                             diagnostics);
      return failure();
    }
    headerOffset = 12;
    headerLen = byte(8) | (byte(9) << 8) | (byte(10) << 16) | (byte(11) << 24);
  } else {
    rejectProgramDirectory(
        ("unsupported npy payload version: " + displayName).str(), diagnostics);
    return failure();
  }

  if (bytes.size() < headerOffset + headerLen) {
    rejectProgramDirectory(
        ("truncated npy payload header: " + displayName).str(), diagnostics);
    return failure();
  }

  llvm::StringRef header = bytes.slice(headerOffset, headerOffset + headerLen);
  std::optional<std::string> descr = parseNpyStringField(header, "descr");
  std::optional<bool> fortranOrder = parseNpyBoolField(header, "fortran_order");
  std::optional<std::vector<int64_t>> shape = parseNpyShapeField(header);
  if (!descr || !fortranOrder || !shape) {
    rejectProgramDirectory(("invalid npy payload header: " + displayName).str(),
                           diagnostics);
    return failure();
  }

  NpyPayloadMetadata metadata;
  metadata.fileSize = bytes.size();
  metadata.dataOffset = headerOffset + headerLen;
  metadata.descr = std::move(*descr);
  metadata.fortranOrder = *fortranOrder;
  metadata.shape = std::move(*shape);
  return metadata;
}


std::optional<std::pair<llvm::StringRef, uint64_t>>
decodeNpyDescr(llvm::StringRef descr) {
  // ProgramTensor owns canonical little-endian compact bytes. NumPy '=' means
  // native endian, so it is equivalent to '<' only on a little-endian host;
  // rejecting it elsewhere prevents downstream numeric consumers from
  // interpreting native big-endian payload bytes as little-endian values.
  const bool nativeIsLittle =
      llvm::endianness::native == llvm::endianness::little;
  if (descr == "<f4" || (nativeIsLittle && descr == "=f4"))
    return std::pair<llvm::StringRef, uint64_t>{"f32", 4};
  if (descr == "<f8" || (nativeIsLittle && descr == "=f8"))
    return std::pair<llvm::StringRef, uint64_t>{"f64", 8};
  if (descr == "<f2" || (nativeIsLittle && descr == "=f2"))
    return std::pair<llvm::StringRef, uint64_t>{"f16", 2};
  if (descr == "|V2")
    return std::pair<llvm::StringRef, uint64_t>{"bf16", 2};
  if (descr == "|b1")
    return std::pair<llvm::StringRef, uint64_t>{"i1", 1};
  if (descr == "|i1")
    return std::pair<llvm::StringRef, uint64_t>{"i8", 1};
  if (descr == "<i2" || (nativeIsLittle && descr == "=i2"))
    return std::pair<llvm::StringRef, uint64_t>{"i16", 2};
  if (descr == "<i4" || (nativeIsLittle && descr == "=i4"))
    return std::pair<llvm::StringRef, uint64_t>{"i32", 4};
  if (descr == "<i8" || (nativeIsLittle && descr == "=i8"))
    return std::pair<llvm::StringRef, uint64_t>{"i64", 8};
  return std::nullopt;
}

bool npyDescrMatchesDtype(llvm::StringRef descr, Type elementType) {
  // NumPy's one-byte signed-integer descriptor was historically accepted for
  // i1 program payloads as well as i8. Preserve that verifier compatibility;
  // the generic payload loader decodes the unambiguous storage dtype as i8.
  if (elementType.isInteger(1) && descr == "|i1")
    return true;
  auto decoded = decodeNpyDescr(descr);
  return decoded && decoded->first == dtypeString(elementType);
}

} // namespace

FailureOr<NpyPayloadMetadata>
readNpyPayloadMetadataFromSource(const ProgramPayloadSource &source,
                                 llvm::StringRef displayName,
                                 llvm::raw_ostream &diagnostics) {
  const uint64_t fileSize = source.getPayloadFileSize();
  auto readBytes = [&](uint64_t offset,
                       llvm::MutableArrayRef<uint8_t> out) -> bool {
    if (offset > fileSize || out.size() > fileSize - offset) {
      rejectProgramDirectory(
          ("truncated npy payload: " + displayName).str(), diagnostics);
      return true;
    }
    if (llvm::Error error = source.readPayloadBytes(offset, out)) {
      rejectProgramDirectory(
          ("failed to read npy payload source: " + displayName).str(),
          diagnostics);
      return true;
    }
    return false;
  };

  llvm::SmallVector<uint8_t, 12> prefix;
  prefix.resize(10);
  if (readBytes(0, prefix))
    return failure();
  if (llvm::ArrayRef<uint8_t>(prefix).take_front(6) !=
      llvm::ArrayRef<uint8_t>(
          reinterpret_cast<const uint8_t *>("\x93NUMPY"), 6)) {
    rejectProgramDirectory(
        ("npy payload is missing magic: " + displayName).str(), diagnostics);
    return failure();
  }

  uint64_t major = prefix[6];
  uint64_t headerLen = 0;
  uint64_t headerOffset = 0;
  if (major == 1) {
    headerOffset = 10;
    headerLen = prefix[8] | (prefix[9] << 8);
  } else if (major == 2) {
    headerOffset = 12;
    llvm::SmallVector<uint8_t, 4> lengthBytes;
    lengthBytes.resize(4);
    if (readBytes(8, lengthBytes))
      return failure();
    headerLen = static_cast<uint64_t>(lengthBytes[0]) |
                (static_cast<uint64_t>(lengthBytes[1]) << 8) |
                (static_cast<uint64_t>(lengthBytes[2]) << 16) |
                (static_cast<uint64_t>(lengthBytes[3]) << 24);
  } else {
    rejectProgramDirectory(
        ("unsupported npy payload version: " + displayName).str(), diagnostics);
    return failure();
  }

  if (headerLen > kMaximumNpyHeaderBytes || headerOffset > fileSize ||
      headerLen > fileSize - headerOffset) {
    rejectProgramDirectory(
        ("npy payload header is truncated or unreasonably large: " +
         displayName)
            .str(),
        diagnostics);
    return failure();
  }

  std::vector<uint8_t> headerStorage(static_cast<size_t>(headerLen));
  if (readBytes(headerOffset, headerStorage))
    return failure();
  llvm::StringRef header(reinterpret_cast<const char *>(headerStorage.data()),
                         headerStorage.size());
  std::optional<std::string> descr = parseNpyStringField(header, "descr");
  std::optional<bool> fortranOrder = parseNpyBoolField(header, "fortran_order");
  std::optional<std::vector<int64_t>> shape = parseNpyShapeField(header);
  if (!descr || !fortranOrder || !shape) {
    rejectProgramDirectory(("invalid npy payload header: " + displayName).str(),
                           diagnostics);
    return failure();
  }

  NpyPayloadMetadata metadata;
  metadata.fileSize = fileSize;
  metadata.dataOffset = headerOffset + headerLen;
  metadata.descr = std::move(*descr);
  metadata.fortranOrder = *fortranOrder;
  metadata.shape = std::move(*shape);
  return metadata;
}

bool verifyNpyTensorPayloadFromSource(const ProgramPayloadSource &source,
                                      llvm::StringRef displayName,
                                      llvm::ArrayRef<int64_t> expectedShape,
                                      Type elementType,
                                      llvm::raw_ostream &diagnostics) {
  FailureOr<NpyPayloadMetadata> metadata =
      readNpyPayloadMetadataFromSource(source, displayName, diagnostics);
  if (failed(metadata))
    return true;

  if (metadata->fortranOrder)
    return rejectProgramDirectory(
        ("npy payload must be row-major: " + displayName).str(), diagnostics);
  if (!llvm::equal(metadata->shape, expectedShape))
    return rejectProgramDirectory(
        ("npy payload shape does not match tensor: " + displayName).str(),
        diagnostics);
  if (!npyDescrMatchesDtype(metadata->descr, elementType))
    return rejectProgramDirectory(
        ("npy payload dtype does not match tensor: " + displayName).str(),
        diagnostics);

  std::optional<uint64_t> expectedRawBytes =
      checkedRawByteSize(expectedShape, elementType);
  if (!expectedRawBytes)
    return rejectProgramDirectory(
        ("npy tensor byte size is not representable: " + displayName).str(),
        diagnostics);
  if (metadata->dataOffset > metadata->fileSize ||
      *expectedRawBytes > metadata->fileSize - metadata->dataOffset)
    return rejectProgramDirectory(
        ("npy payload file is smaller than tensor payload: " + displayName)
            .str(),
        diagnostics);

  return false;
}

bool verifyNpyTensorPayloadFile(llvm::StringRef path,
                                llvm::StringRef displayName,
                                llvm::ArrayRef<int64_t> expectedShape,
                                Type elementType,
                                llvm::raw_ostream &diagnostics) {
  FailureOr<NpyPayloadMetadata> metadata =
      readNpyPayloadMetadata(path, displayName, diagnostics);
  if (failed(metadata))
    return true;

  if (metadata->fortranOrder)
    return rejectProgramDirectory(
        ("npy payload must be row-major: " + displayName).str(), diagnostics);
  if (!llvm::equal(metadata->shape, expectedShape))
    return rejectProgramDirectory(
        ("npy payload shape does not match tensor: " + displayName).str(),
        diagnostics);
  if (!npyDescrMatchesDtype(metadata->descr, elementType))
    return rejectProgramDirectory(
        ("npy payload dtype does not match tensor: " + displayName).str(),
        diagnostics);

  std::optional<uint64_t> expectedRawBytes =
      checkedRawByteSize(expectedShape, elementType);
  if (!expectedRawBytes)
    return rejectProgramDirectory(
        ("npy tensor byte size is not representable: " + displayName).str(),
        diagnostics);
  if (metadata->dataOffset > metadata->fileSize ||
      *expectedRawBytes > metadata->fileSize - metadata->dataOffset)
    return rejectProgramDirectory(
        ("npy payload file is smaller than tensor payload: " + displayName)
            .str(),
        diagnostics);

  return false;
}

} // namespace wafer::frontend::program_detail

namespace wafer::frontend {

using namespace program_detail;

llvm::Expected<NpyTensorPayload> loadNpyTensorPayload(llvm::StringRef path) {
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  FailureOr<NpyPayloadMetadata> metadata =
      readNpyPayloadMetadata(path, path, diagnostics);
  if (failed(metadata))
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   diagnostics.str().c_str());
  if (metadata->fortranOrder)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "npy payload must be row-major: %s",
                                   path.str().c_str());
  auto decoded = decodeNpyDescr(metadata->descr);
  if (!decoded)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "npy payload has unsupported dtype: %s",
                                   path.str().c_str());

  uint64_t rawBytes = decoded->second;
  for (int64_t dim : metadata->shape) {
    uint64_t next = 0;
    if (dim < 0 ||
        !checkedMulUint64(rawBytes, static_cast<uint64_t>(dim), next))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "npy payload byte count is not representable: %s",
          path.str().c_str());
    rawBytes = next;
  }
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to reopen npy payload: %s",
                                   path.str().c_str());
  llvm::StringRef fileBytes = (*buffer)->getBuffer();
  if (metadata->dataOffset > fileBytes.size() ||
      rawBytes > fileBytes.size() - metadata->dataOffset)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "npy tensor payload is truncated: %s",
                                   path.str().c_str());
  llvm::ArrayRef<uint8_t> payload(
      reinterpret_cast<const uint8_t *>(fileBytes.data()) +
          metadata->dataOffset,
      static_cast<size_t>(rawBytes));
  return NpyTensorPayload{decoded->first.str(), metadata->shape,
                          std::vector<uint8_t>(payload.begin(), payload.end())};
}

llvm::Expected<NpyPayloadHeader>
parseNpyPayloadHeader(const ProgramPayloadSource &source,
                      llvm::StringRef displayName) {
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  FailureOr<NpyPayloadMetadata> metadata =
      readNpyPayloadMetadataFromSource(source, displayName, diagnostics);
  if (failed(metadata))
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   diagnostics.str().c_str());
  return NpyPayloadHeader{metadata->fileSize, metadata->dataOffset,
                          std::move(metadata->descr), metadata->fortranOrder,
                          std::move(metadata->shape)};
}

std::optional<std::pair<llvm::StringRef, uint64_t>>
decodeProgramNpyDescr(llvm::StringRef descr) {
  return decodeNpyDescr(descr);
}

} // namespace wafer::frontend
