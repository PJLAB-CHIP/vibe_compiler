//===- ParameterShards.cpp - Parameter shard validation -------------===//

#include "ProgramInternal.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace wafer::frontend::program_detail {

namespace {

bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

FailureOr<llvm::json::Value> parseJsonFile(llvm::StringRef path,
                                           llvm::raw_ostream &diagnostics) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError) {
    rejectProgramDirectory("failed to read '" + path.str() + "'", diagnostics);
    return failure();
  }

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*bufferOrError)->getBuffer());
  if (!parsed) {
    std::string message = llvm::toString(parsed.takeError());
    rejectProgramDirectory("invalid JSON: " + message, diagnostics);
    return failure();
  }
  return std::move(*parsed);
}

bool isSafeRelativePath(llvm::StringRef path) {
  return !path.empty() && !llvm::sys::path::is_absolute(path) &&
         !path.contains("..");
}

struct ParameterShardSlice {
  int64_t rank = -1;
  int64_t replicaId = -1;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  std::string path;
  std::string relativePath;
};

wafer::frontend::ProgramRankSlice
getVerifiedRankSlice(const ParameterShardSlice &slice) {
  return {slice.rank,  slice.replicaId, slice.offsets,
          slice.sizes, slice.strides,   slice.relativePath};
}

bool verifyShardEntry(const llvm::json::Object &object,
                      int64_t logicalRankCount,
                      llvm::ArrayRef<int64_t> globalShape,
                      llvm::ArrayRef<int64_t> localShape, Type elementType,
                      llvm::StringRef parameterName, llvm::StringRef programDir,
                      llvm::StringRef distribution,
                      std::vector<bool> &seenRanks,
                      std::vector<bool> &seenReplicaIds,
                      std::vector<ParameterShardSlice> &verifiedSlices,
                      llvm::raw_ostream &diagnostics) {
  int64_t rank = -1;
  int64_t replicaId = -1;
  std::string file;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  if (readIntegerField(object, "rank", rank, diagnostics) ||
      readIntegerField(object, "replica_id", replicaId, diagnostics) ||
      readStringField(object, "file", file, diagnostics) ||
      readIntegerArrayField(object, "offsets", offsets, diagnostics) ||
      readIntegerArrayField(object, "sizes", sizes, diagnostics) ||
      readIntegerArrayField(object, "strides", strides, diagnostics,
                            /*requirePositive=*/true))
    return true;

  if (rank < 0 || rank >= logicalRankCount)
    return rejectProgramDirectory("parameter shard rank is out of range",
                                  diagnostics);
  if (seenRanks[rank])
    return rejectProgramDirectory("duplicate parameter shard rank",
                                  diagnostics);
  seenRanks[rank] = true;

  if (replicaId < 0)
    return rejectProgramDirectory(
        "parameter shard replica_id must be non-negative", diagnostics);
  if (distribution == "partitioned" && replicaId != 0)
    return rejectProgramDirectory(
        "partitioned parameter shard replica_id must be 0", diagnostics);
  if (distribution == "replicated") {
    if (replicaId >= logicalRankCount)
      return rejectProgramDirectory(
          "replicated parameter shard replica_id is out of range", diagnostics);
    if (seenReplicaIds[replicaId])
      return rejectProgramDirectory(
          "duplicate replicated parameter shard replica_id", diagnostics);
    seenReplicaIds[replicaId] = true;
  }

  size_t rankSize = globalShape.size();
  if (offsets.size() != rankSize || sizes.size() != rankSize ||
      strides.size() != rankSize)
    return rejectProgramDirectory(
        "parameter shard rank does not match global shape", diagnostics);

  for (auto [dim, offset] : llvm::enumerate(offsets)) {
    int64_t size = sizes[dim];
    if (offset > globalShape[dim] || size > globalShape[dim] - offset)
      return rejectProgramDirectory(
          "parameter shard slice exceeds global shape", diagnostics);
    if (dim < localShape.size() && size > localShape[dim])
      return rejectProgramDirectory(
          "parameter shard size exceeds local tensor shape", diagnostics);
    if (strides[dim] != 1)
      return rejectProgramDirectory(
          "parameter shard stride must be 1 for coverage verification",
          diagnostics);
  }

  if (!isSafeRelativePath(file))
    return rejectProgramDirectory(
        "parameter shard file must be a safe relative path", diagnostics);
  std::string expectedPrefix =
      (Twine("parameter_shards/") + parameterName + "/").str();
  if (!llvm::StringRef(file).starts_with(expectedPrefix))
    return rejectProgramDirectory(
        "parameter shard file path does not match parameter", diagnostics);

  std::string path = programPath(programDir, {file});
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(path, status))
    return rejectProgramDirectory("parameter shard file is missing: " + file,
                                  diagnostics);
  if (!llvm::sys::fs::is_regular_file(status))
    return rejectProgramDirectory(
        "parameter shard path is not a regular file: " + file, diagnostics);

  if (verifyNpyTensorPayloadFile(path, file, sizes, elementType, diagnostics))
    return true;
  verifiedSlices.push_back(ParameterShardSlice{
      rank, replicaId, std::move(offsets), std::move(sizes), std::move(strides),
      std::move(path), std::move(file)});
  return false;
}

bool verifyParameterShardCoverage(llvm::ArrayRef<ParameterShardSlice> slices,
                                  llvm::ArrayRef<int64_t> globalShape,
                                  llvm::StringRef distribution,
                                  llvm::raw_ostream &diagnostics) {
  if (distribution == "replicated") {
    std::unique_ptr<llvm::MemoryBuffer> canonicalPayload;
    for (const ParameterShardSlice &slice : slices) {
      if (!llvm::all_of(slice.offsets,
                        [](int64_t offset) { return offset == 0; }) ||
          !llvm::equal(slice.sizes, globalShape))
        return rejectProgramDirectory(
            "replicated parameter shard must contain the full global tensor",
            diagnostics);

      auto payload = llvm::MemoryBuffer::getFile(slice.path);
      if (!payload)
        return rejectProgramDirectory(
            "failed to reread replicated parameter shard payload", diagnostics);
      if (!canonicalPayload) {
        canonicalPayload = std::move(*payload);
      } else if (canonicalPayload->getBuffer() != (*payload)->getBuffer()) {
        return rejectProgramDirectory(
            "replicated parameter shard payloads must be byte-identical",
            diagnostics);
      }
    }
    return false;
  }

  if (distribution != "partitioned")
    return rejectProgramDirectory(
        "parameter shard distribution must be 'replicated' or 'partitioned'",
        diagnostics);

  int64_t globalElements = 1;
  for (int64_t dim : globalShape) {
    if (!checkedMul(globalElements, dim, globalElements))
      return rejectProgramDirectory(
          "parameter shard global coverage size overflows int64", diagnostics);
  }

  int64_t coveredElements = 0;
  for (const ParameterShardSlice &slice : slices) {
    int64_t sliceElements = 1;
    for (int64_t size : slice.sizes) {
      if (!checkedMul(sliceElements, size, sliceElements))
        return rejectProgramDirectory(
            "parameter shard coverage size overflows int64", diagnostics);
    }
    if (!checkedAdd(coveredElements, sliceElements, coveredElements))
      return rejectProgramDirectory(
          "parameter shard coverage size overflows int64", diagnostics);
  }

  for (size_t lhsIndex = 0; lhsIndex < slices.size(); ++lhsIndex) {
    for (size_t rhsIndex = lhsIndex + 1; rhsIndex < slices.size(); ++rhsIndex) {
      const ParameterShardSlice &lhs = slices[lhsIndex];
      const ParameterShardSlice &rhs = slices[rhsIndex];
      bool overlaps = true;
      for (size_t dim = 0; dim < globalShape.size(); ++dim) {
        int64_t lhsEnd = lhs.offsets[dim] + lhs.sizes[dim];
        int64_t rhsEnd = rhs.offsets[dim] + rhs.sizes[dim];
        if (lhsEnd <= rhs.offsets[dim] || rhsEnd <= lhs.offsets[dim]) {
          overlaps = false;
          break;
        }
      }
      if (overlaps)
        return rejectProgramDirectory("parameter shard slices overlap",
                                      diagnostics);
    }
  }

  if (coveredElements != globalElements)
    return rejectProgramDirectory(
        "parameter shard slices do not cover the global tensor", diagnostics);
  return false;
}

bool verifyParameterShardMetadata(
    const llvm::json::Object &object, const ProgramMetadata &meta,
    FunctionType functionType, int64_t logicalRankCount,
    std::vector<bool> &seenParameterArgs, llvm::StringRef programDir,
    llvm::raw_ostream &diagnostics,
    wafer::frontend::ProgramParameterBinding *verifiedBinding) {
  int64_t argumentIndex = -1;
  std::string name;
  std::string dtype;
  std::string distribution;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;

  if (readIntegerField(object, "argument_index", argumentIndex, diagnostics) ||
      readStringField(object, "name", name, diagnostics) ||
      readStringField(object, "dtype", dtype, diagnostics) ||
      readStringField(object, "distribution", distribution, diagnostics) ||
      readIntegerArrayField(object, "global_shape", globalShape, diagnostics) ||
      readIntegerArrayField(object, "local_shape", localShape, diagnostics))
    return true;

  if (argumentIndex < 0 ||
      argumentIndex >= static_cast<int64_t>(functionType.getNumInputs()))
    return rejectProgramDirectory(
        "parameter shard argument_index is out of range", diagnostics);

  const ProgramInputLocation &location = meta.inputLocations[argumentIndex];
  if (location.type != "parameter")
    return rejectProgramDirectory(
        "parameter shard argument_index does not refer to a "
        "parameter input",
        diagnostics);
  if (name != location.name)
    return rejectProgramDirectory(
        "parameter shard name does not match input location", diagnostics);
  if (seenParameterArgs[argumentIndex])
    return rejectProgramDirectory(
        "duplicate parameter shard metadata record for parameter argument",
        diagnostics);

  Type inputType = functionType.getInput(argumentIndex);
  auto tensorType = dyn_cast<RankedTensorType>(inputType);
  if (!tensorType || !tensorType.hasStaticShape())
    return rejectProgramDirectory(
        "parameter shard argument must be a static ranked tensor", diagnostics);
  if (!llvm::equal(tensorType.getShape(), localShape))
    return rejectProgramDirectory(
        "parameter shard local_shape does not match func.func "
        "argument type",
        diagnostics);
  if (globalShape.size() != static_cast<size_t>(tensorType.getRank()))
    return rejectProgramDirectory(
        "parameter shard global_shape rank does not match "
        "func.func argument type",
        diagnostics);
  if (dtypeString(tensorType.getElementType()) != normalizeProgramDtype(dtype))
    return rejectProgramDirectory(
        "parameter shard dtype does not match func.func "
        "argument type",
        diagnostics);

  std::optional<uint64_t> expectedGlobalBytes =
      checkedRawByteSize(globalShape, tensorType.getElementType());
  if (!expectedGlobalBytes)
    return rejectProgramDirectory(
        "parameter shard global tensor byte size is not "
        "representable",
        diagnostics);

  const llvm::json::Array *shards = object.getArray("shards");
  if (!shards)
    return rejectProgramDirectory("expected array field 'shards'", diagnostics);
  if (shards->size() != static_cast<size_t>(logicalRankCount))
    return rejectProgramDirectory("parameter shard count does not match "
                                  "logical_rank_count",
                                  diagnostics);

  std::vector<bool> seenRanks(logicalRankCount, false);
  std::vector<bool> seenReplicaIds(logicalRankCount, false);
  std::vector<ParameterShardSlice> verifiedSlices;
  verifiedSlices.reserve(shards->size());
  for (const llvm::json::Value &value : *shards) {
    const llvm::json::Object *shardObject = value.getAsObject();
    if (!shardObject)
      return rejectProgramDirectory("parameter shard entries must be objects",
                                    diagnostics);
    if (verifyShardEntry(*shardObject, logicalRankCount, globalShape,
                         localShape, tensorType.getElementType(), name,
                         programDir, distribution, seenRanks, seenReplicaIds,
                         verifiedSlices, diagnostics))
      return true;
  }
  if (distribution == "replicated" &&
      llvm::any_of(seenReplicaIds, [](bool seen) { return !seen; }))
    return rejectProgramDirectory(
        "replicated parameter shard replica_id domain is incomplete",
        diagnostics);
  if (verifyParameterShardCoverage(verifiedSlices, globalShape, distribution,
                                   diagnostics))
    return true;

  seenParameterArgs[argumentIndex] = true;
  if (verifiedBinding) {
    verifiedBinding->argumentIndex = argumentIndex;
    verifiedBinding->name = std::move(name);
    verifiedBinding->distribution = getVerifiedDistributionKind(distribution);
    verifiedBinding->globalShape = std::move(globalShape);
    verifiedBinding->localShape = std::move(localShape);
    verifiedBinding->dtype = normalizeProgramDtype(dtype);
    verifiedBinding->rankSlices.reserve(verifiedSlices.size());
    for (const ParameterShardSlice &slice : verifiedSlices)
      verifiedBinding->rankSlices.push_back(getVerifiedRankSlice(slice));
  }
  return false;
}

} // namespace

bool verifyParameterShards(
    ModuleOp module, llvm::StringRef programDir, const ProgramMetadata &meta,
    func::FuncOp func, llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result) {
  std::string path =
      programPath(programDir, {"functions", "forward.parameter_shards.json"});
  if (!fileExists(path)) {
    if (hasSpmdParameterShardings(module))
      return rejectProgramDirectory(
          "partitioned StableHLO program directory is missing parameter "
          "shards",
          diagnostics);
    return false;
  }

  FailureOr<llvm::json::Value> parsed = parseJsonFile(path, diagnostics);
  if (failed(parsed))
    return true;

  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return rejectProgramDirectory(
        "parameter shard metadata root must be an object", diagnostics);

  int64_t version = 0;
  std::string function;
  int64_t logicalRankCount = 0;
  if (readIntegerField(*root, "parameter_shards_version", version,
                       diagnostics) ||
      readStringField(*root, "function", function, diagnostics) ||
      readIntegerField(*root, "logical_rank_count", logicalRankCount,
                       diagnostics))
    return true;
  if (version != 3)
    return rejectProgramDirectory(
        "unsupported parameter shard metadata version", diagnostics);
  if (function != meta.name)
    return rejectProgramDirectory(
        "parameter shard function does not match program directory meta",
        diagnostics);
  if (logicalRankCount <= 0)
    return rejectProgramDirectory("logical_rank_count must be positive",
                                  diagnostics);

  FailureOr<int64_t> meshRankCount =
      getSingleExecutionMeshRankCount(module, diagnostics);
  if (failed(meshRankCount))
    return true;
  if (*meshRankCount != logicalRankCount)
    return rejectProgramDirectory(
        "parameter shard logical_rank_count does not match execution mesh "
        "rank count",
        diagnostics);

  const llvm::json::Array *parameters = root->getArray("parameters");
  if (!parameters)
    return rejectProgramDirectory("expected array field 'parameters'",
                                  diagnostics);

  FunctionType functionType = func.getFunctionType();
  std::vector<bool> seenParameterArgs(functionType.getNumInputs(), false);
  unsigned parameterShardCount = 0;
  for (const llvm::json::Value &value : *parameters) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return rejectProgramDirectory(
          "parameter shard metadata entries must be objects", diagnostics);
    wafer::frontend::ProgramParameterBinding verifiedBinding;
    if (verifyParameterShardMetadata(
            *object, meta, functionType, logicalRankCount, seenParameterArgs,
            programDir, diagnostics, result ? &verifiedBinding : nullptr))
      return true;
    if (result)
      result->parameters.push_back(std::move(verifiedBinding));
    ++parameterShardCount;
  }

  for (auto [index, location] : llvm::enumerate(meta.inputLocations)) {
    if (location.type == "parameter" && !seenParameterArgs[index])
      return rejectProgramDirectory(
          "parameter input is missing shard metadata: " + location.name,
          diagnostics);
  }

  if (result)
    result->programParameterShardCount = parameterShardCount;
  return false;
}

} // namespace wafer::frontend::program_detail
