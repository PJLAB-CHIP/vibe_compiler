//===- OptimizationAdoption.cpp - Canonical adoption inventory ----------===//

#include "Wafer/Support/OptimizationAdoption.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>

namespace wafer {
namespace {

enum TypeTag : uint8_t {
  Bool = 0x01,
  U16 = 0x02,
  U32 = 0x03,
  U64 = 0x04,
  ClosedEnum = 0x05,
  TypedId = 0x06,
  Digest32 = 0x07,
  Bytes = 0x08,
  Record = 0x09,
  Sequence = 0x0a,
  SortedSet = 0x0b,
  Optional = 0x0c,
};

static void appendU16(std::vector<uint8_t> &bytes, uint16_t value) {
  bytes.push_back(static_cast<uint8_t>(value >> 8));
  bytes.push_back(static_cast<uint8_t>(value));
}

static void appendU32(std::vector<uint8_t> &bytes, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

static void appendU64(std::vector<uint8_t> &bytes, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

static void appendBytes(std::vector<uint8_t> &bytes, llvm::StringRef value) {
  appendU32(bytes, value.size());
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
}

static void appendField(std::vector<uint8_t> &body, uint16_t number,
                        TypeTag tag, llvm::ArrayRef<uint8_t> payload) {
  appendU16(body, number);
  body.push_back(tag);
  appendU32(body, payload.size());
  body.insert(body.end(), payload.begin(), payload.end());
}

static std::vector<uint8_t> u16(uint16_t value) {
  std::vector<uint8_t> bytes;
  appendU16(bytes, value);
  return bytes;
}

static std::vector<uint8_t> u32(uint32_t value) {
  std::vector<uint8_t> bytes;
  appendU32(bytes, value);
  return bytes;
}

static AdoptionDigest digest(llvm::ArrayRef<uint8_t> bytes) {
  llvm::SHA256 hash;
  hash.update(bytes);
  return hash.final();
}

static AdoptionDigest digestDefinition(llvm::StringRef definition) {
  return digest(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(definition.data()),
      definition.size()));
}

static std::vector<uint8_t> encodeOptionalDigest(
    const std::optional<AdoptionDigest> &value) {
  if (!value)
    return {0};
  std::vector<uint8_t> bytes = {1, Digest32};
  appendU32(bytes, value->size());
  bytes.insert(bytes.end(), value->begin(), value->end());
  return bytes;
}

static std::vector<uint8_t> encodeProvider(const AdoptionSpec &spec) {
  std::vector<uint8_t> body;
  std::vector<uint8_t> id;
  appendBytes(id, spec.providerId);
  appendField(body, 1, TypedId, id);
  std::vector<uint8_t> revision;
  appendBytes(revision, spec.providerRevision);
  appendField(body, 2, Bytes, revision);
  std::vector<uint8_t> content =
      encodeOptionalDigest(spec.providerContentDigest);
  appendField(body, 3, Optional, content);
  return body;
}

static std::vector<uint8_t> encodeContract(uint32_t kind, uint32_t id) {
  std::vector<uint8_t> body;
  appendField(body, 1, ClosedEnum, u32(kind));
  appendField(body, 2, U32, u32(id));
  appendField(body, 3, U16, u16(1));
  AdoptionDigest definition = digestDefinition(
      "wafer.optimization-contract-v1:" + std::to_string(kind) + ":" +
      std::to_string(id));
  appendField(body, 4, Digest32, definition);
  return body;
}

static std::vector<uint8_t> encodeWorkPolicy(uint32_t kind) {
  std::vector<uint8_t> body;
  appendField(body, 1, ClosedEnum, u32(kind));
  appendField(body, 2, U16, u16(1));
  AdoptionDigest definition = digestDefinition(
      "wafer.optimization-work-policy-v1:" + std::to_string(kind));
  appendField(body, 3, Digest32, definition);
  return body;
}

static std::vector<uint8_t> encodeOwner(uint32_t owner) {
  std::vector<uint8_t> body;
  appendField(body, 1, U32, u32(owner));
  return body;
}

template <typename Enum>
static std::vector<uint8_t> encodeEnumSet(std::vector<Enum> values) {
  std::sort(values.begin(), values.end(), [](Enum lhs, Enum rhs) {
    return static_cast<uint32_t>(lhs) < static_cast<uint32_t>(rhs);
  });
  values.erase(std::unique(values.begin(), values.end()), values.end());
  std::vector<uint8_t> bytes;
  appendU32(bytes, values.size());
  for (Enum value : values) {
    std::vector<uint8_t> element = u32(static_cast<uint32_t>(value));
    appendU32(bytes, element.size());
    bytes.insert(bytes.end(), element.begin(), element.end());
  }
  return bytes;
}

static OperationDomain ir(std::initializer_list<uint32_t> families,
                          std::initializer_list<uint32_t> interfaces,
                          std::initializer_list<uint32_t> dialects) {
  return {OperationDomainKind::IR, families, interfaces, dialects, {}};
}

static OperationDomain action(std::initializer_list<uint32_t> executors,
                              std::initializer_list<uint32_t> actions,
                              std::initializer_list<uint32_t> inputs,
                              std::initializer_list<uint32_t> outputs) {
  return {OperationDomainKind::ArtifactAction, executors, actions, inputs,
          outputs};
}

static AdoptionSpec makeSpec(MechanismDescriptor descriptor,
                             ProviderOrigin origin, llvm::StringRef provider,
                             llvm::StringRef revision, OperationDomain domain,
                             std::vector<Exposure> exposure, AdoptionMode mode,
                             uint32_t contractBase, uint32_t workPolicy,
                             uint32_t owner) {
  AdoptionSpec spec;
  spec.mechanismKey = descriptor.key;
  spec.providerOrigin = origin;
  spec.providerId = provider.str();
  spec.providerRevision = revision.str();
  spec.cutPoint = descriptor.cutPoint;
  spec.operationDomain = std::move(domain);
  spec.availability = Availability::Resolved;
  spec.exposureSet = std::move(exposure);
  spec.adoptionMode = mode;
  spec.preconditionContractId = contractBase;
  spec.numericContractId = contractBase + 1;
  spec.effectContractId = contractBase + 2;
  spec.workPolicyKind = workPolicy;
  spec.ownerId = owner;
  spec.evidenceKind = descriptor.evidenceKind;
  return spec;
}

static std::vector<AdoptionSpec> buildSpecs() {
  constexpr llvm::StringLiteral llvmRevision =
      "f0b3287297aeeddcf030e3c1b08d05a69ad465aa";
  constexpr llvm::StringLiteral stablehloRevision =
      "e51fd95e5b2c28861f22dc9d609fb2a7f002124e";
  std::vector<AdoptionSpec> specs;
  specs.reserve(31);
  auto add = [&](MechanismKey key, ProviderOrigin origin,
                 llvm::StringRef provider, llvm::StringRef revision,
                 OperationDomain domain, std::vector<Exposure> exposure,
                 AdoptionMode mode, uint32_t workPolicy = 1) {
    MechanismDescriptor descriptor = *lookupMechanismDescriptor(key);
    uint32_t base = 1000 + descriptor.key.semanticId * 10;
    specs.push_back(makeSpec(descriptor, origin, provider, revision,
                             std::move(domain), std::move(exposure), mode,
                             base, workPolicy, 5));
  };

  add(mechanism::FrontendProgramImport, ProviderOrigin::Internal,
      "wafer.frontend", "source-v1", action({0}, {0}, {0}, {1}),
      {Exposure::LibraryCallable}, AdoptionMode::RequiredNormalization);
  add(mechanism::SpmdPartition, ProviderOrigin::Vendored, "wafer.xla-spmd",
      "pinned-helper-v1", action({1}, {1}, {1}, {2}),
      {Exposure::BackendExecutable}, AdoptionMode::RequiredNormalization);
  add(mechanism::StablehloCollectiveNormalization, ProviderOrigin::Internal,
      "wafer.stablehlo-collective", "source-v1", ir({7}, {1}, {1, 2}),
      {Exposure::LibraryCallable}, AdoptionMode::RequiredNormalization);
  add(mechanism::StablehloStructuredLegalization, ProviderOrigin::Upstream,
      "stablehlo.legalize-to-linalg", stablehloRevision,
      ir({0, 1, 2, 3, 4, 5, 6, 7}, {1}, {1, 3, 4, 5, 6}),
      {Exposure::LibraryCallable}, AdoptionMode::RequiredNormalization);
  add(mechanism::RequiredTensorNormalization, ProviderOrigin::Internal,
      "wafer.required-tensor-normalization", "source-v1",
      ir({0, 1, 2, 3, 4}, {1, 2}, {3, 4, 5, 6}),
      {Exposure::LibraryCallable}, AdoptionMode::RequiredNormalization, 2);

  for (MechanismKey key :
       {mechanism::StablehloCleanup, mechanism::StructuredTensorCleanup,
        mechanism::CandidateCommitCleanup,
        mechanism::PreBufferizationCleanup})
    add(key, ProviderOrigin::Upstream, "mlir.canonicalizer", llvmRevision,
        ir({0, 4, 5, 6, 8}, {0}, {0, 3, 4, 5, 6, 7}),
        {Exposure::DebugRegistered, Exposure::LibraryCallable},
        AdoptionMode::None);

  add(mechanism::FunctionBoundaryBufferization, ProviderOrigin::Upstream,
      "mlir.one-shot-bufferize", llvmRevision, ir({5}, {2}, {7, 8}),
      {Exposure::DebugRegistered, Exposure::LibraryCallable},
      AdoptionMode::RequiredNormalization);
  for (MechanismKey key : {mechanism::PostBufferizationCleanup,
                           mechanism::PostMemoryPlanningCleanup})
    add(key, ProviderOrigin::Upstream, "mlir.canonicalizer", llvmRevision,
        ir({0, 4, 5, 6, 8}, {0}, {0, 3, 4, 5, 6, 7}),
        {Exposure::DebugRegistered, Exposure::LibraryCallable},
        AdoptionMode::None);
  for (MechanismKey key :
       {mechanism::StructuredTilingInterface,
        mechanism::TiledShapeConstruction,
        mechanism::ProducerSliceFusion,
        mechanism::TileDataflowMaterialization})
    add(key, key == mechanism::TileDataflowMaterialization
                 ? ProviderOrigin::Internal
                 : ProviderOrigin::Upstream,
        key == mechanism::TileDataflowMaterialization
            ? "wafer.tile-dataflow-materialization"
            : "mlir.structured-tiling-utility",
        key == mechanism::TileDataflowMaterialization ? "source-v1"
                                                       : llvmRevision,
        ir({0, 1, 2, 3, 4}, {1, 2}, {3, 4, 5, 6}),
        {Exposure::LibraryCallable}, AdoptionMode::CandidateLocal);
  add(mechanism::InstructionLowering, ProviderOrigin::Internal,
      "wafer.instruction-lowering", "source-v1", ir({5, 6, 7}, {3}, {0}),
      {Exposure::DebugRegistered, Exposure::LibraryCallable},
      AdoptionMode::TargetSpecific);
  add(mechanism::TargetLLVMConversion, ProviderOrigin::Internal,
      "wafer.target-llvm-conversion", "source-v1", ir({8, 9}, {3}, {0, 9}),
      {Exposure::LibraryCallable}, AdoptionMode::TargetSpecific);
  for (MechanismKey key :
       {mechanism::DeviceObjectCompilation,
        mechanism::DeviceRuntimeCompilation,
        mechanism::DeviceGarbageCollectionLink})
    add(key, ProviderOrigin::Toolchain, "tx8.device-toolchain",
        "configured-toolchain-v1", action({2, 3}, {2, 4}, {3}, {4}),
        {Exposure::BackendExecutable}, AdoptionMode::TargetBackend);

  add(mechanism::ScalarCommonSubexpressionElimination,
      ProviderOrigin::Upstream, "mlir.cse", llvmRevision,
      ir({0, 4}, {0}, {3, 4, 5, 6}),
      {Exposure::DebugRegistered, Exposure::LibraryCallable},
      AdoptionMode::CandidateLocal);
  add(mechanism::SparseConditionalConstantPropagation,
      ProviderOrigin::Upstream, "mlir.sccp", llvmRevision,
      ir({0, 4}, {0}, {3, 4, 5, 6}),
      {Exposure::DebugRegistered, Exposure::LibraryCallable},
      AdoptionMode::None);
  for (MechanismKey key :
       {mechanism::LinalgTransformFamily, mechanism::TensorTransformFamily,
        mechanism::ScfTransformFamily,
        mechanism::BufferizationTransformFamily,
        mechanism::ArithTransformFamily})
    add(key, ProviderOrigin::Upstream, "mlir.debug-transform-family",
        llvmRevision, ir({0, 4, 5}, {0}, {3, 4, 5, 6, 7, 8}),
        {Exposure::DebugRegistered, Exposure::LibraryCallable},
        AdoptionMode::None);
  add(mechanism::SelectedPayloadNormalization, ProviderOrigin::Internal,
      "wafer.selected-payload-normalization", "source-v1",
      ir({4, 5, 6, 7}, {3, 4}, {0, 5, 7, 8}),
      {Exposure::LibraryCallable}, AdoptionMode::RequiredNormalization, 3);
  for (MechanismKey key :
       {mechanism::PostLegalizationCanonicalization,
        mechanism::StructuredTensorCanonicalization})
    add(key, ProviderOrigin::Upstream, "mlir.canonicalizer", llvmRevision,
        ir({0, 4, 5, 6, 8}, {0}, {0, 3, 4, 5, 6, 7}),
        {Exposure::LibraryCallable}, AdoptionMode::RequiredNormalization);
  return specs;
}

static const std::vector<AdoptionSpec> &specs() {
  static const std::vector<AdoptionSpec> value = buildSpecs();
  return value;
}

static bool validTypedId(llvm::StringRef id) {
  if (id.empty() || id.size() > 128 || !std::islower(id.front()) &&
                                          !std::isdigit(id.front()))
    return false;
  return llvm::all_of(id, [](unsigned char character) {
    return std::islower(character) || std::isdigit(character) ||
           character == '.' || character == '_' || character == '-';
  });
}

static bool sortedUniqueNonempty(llvm::ArrayRef<uint32_t> values) {
  return !values.empty() &&
         std::adjacent_find(values.begin(), values.end(),
                            std::greater_equal<uint32_t>()) == values.end();
}

static bool validateSpecValue(const AdoptionSpec &spec,
                              std::string *diagnostic) {
  auto fail = [&](llvm::StringRef message) {
    if (diagnostic)
      *diagnostic = message.str();
    return false;
  };
  if (spec.schemaVersion != 1)
    return fail("unsupported adoption spec schema");
  if (!validTypedId(spec.providerId))
    return fail("invalid provider typed id");
  if (spec.providerRevision.empty())
    return fail("empty provider revision");
  if (!sortedUniqueNonempty(spec.operationDomain.semanticFamilies) ||
      !sortedUniqueNonempty(spec.operationDomain.requiredInterfaces) ||
      !sortedUniqueNonempty(spec.operationDomain.allowedDialects))
    return fail("operation domain sets are not canonical nonempty sets");
  if (spec.operationDomain.kind == OperationDomainKind::ArtifactAction &&
      !sortedUniqueNonempty(spec.operationDomain.outputArtifacts))
    return fail("artifact action output set is not canonical nonempty");
  if (spec.operationDomain.kind == OperationDomainKind::IR &&
      !spec.operationDomain.outputArtifacts.empty())
    return fail("IR domain carried artifact output kinds");
  return true;
}

} // namespace

std::vector<AdoptionSpec> getAllAdoptionSpecs() { return specs(); }

std::optional<AdoptionSpec> lookupAdoptionSpec(MechanismKey key) {
  auto it = std::lower_bound(specs().begin(), specs().end(), key,
                             [](const AdoptionSpec &spec, MechanismKey query) {
                               return spec.mechanismKey < query;
                             });
  if (it == specs().end() || it->mechanismKey != key)
    return std::nullopt;
  return *it;
}

std::vector<uint8_t> encodeMechanismKeyV1(MechanismKey key) {
  constexpr char domain[] = "wafer.mechanism-key";
  std::vector<uint8_t> bytes(domain, domain + sizeof(domain));
  appendU16(bytes, 1);
  appendU32(bytes, key.semanticId);
  return bytes;
}

std::vector<uint8_t> encodeOperationDomainV1(const OperationDomain &domain) {
  constexpr char owner[] = "wafer.operation-domain";
  std::vector<uint8_t> bytes(owner, owner + sizeof(owner));
  appendU16(bytes, 1);
  bytes.push_back(static_cast<uint8_t>(domain.kind));
  auto appendSet = [&](llvm::ArrayRef<uint32_t> values) {
    appendU32(bytes, values.size());
    for (uint32_t value : values)
      appendU32(bytes, value);
  };
  appendSet(domain.semanticFamilies);
  appendSet(domain.requiredInterfaces);
  appendSet(domain.allowedDialects);
  if (domain.kind == OperationDomainKind::ArtifactAction)
    appendSet(domain.outputArtifacts);
  return bytes;
}

std::vector<uint8_t> encodeAdoptionSpecV1(const AdoptionSpec &spec) {
  std::string diagnostic;
  if (!validateSpecValue(spec, &diagnostic))
    return {};
  std::vector<uint8_t> body;
  appendField(body, 1, U16, u16(spec.schemaVersion));
  appendField(body, 2, Record, encodeMechanismKeyV1(spec.mechanismKey));
  appendField(body, 3, ClosedEnum,
              u32(static_cast<uint32_t>(spec.providerOrigin)));
  appendField(body, 4, Record, encodeProvider(spec));
  appendField(body, 5, ClosedEnum,
              u32(static_cast<uint32_t>(spec.cutPoint)));
  appendField(body, 6, Record,
              encodeOperationDomainV1(spec.operationDomain));
  appendField(body, 7, ClosedEnum,
              u32(static_cast<uint32_t>(spec.availability)));
  appendField(body, 8, SortedSet, encodeEnumSet(spec.exposureSet));
  appendField(body, 9, ClosedEnum,
              u32(static_cast<uint32_t>(spec.adoptionMode)));
  appendField(body, 10, Record,
              encodeContract(/*precondition=*/0,
                             spec.preconditionContractId));
  appendField(body, 11, Record,
              encodeContract(/*numeric=*/1, spec.numericContractId));
  appendField(body, 12, Record,
              encodeContract(/*effect=*/2, spec.effectContractId));
  appendField(body, 13, Record, encodeWorkPolicy(spec.workPolicyKind));
  appendField(body, 14, Record, encodeOwner(spec.ownerId));
  appendField(body, 15, ClosedEnum,
              u32(static_cast<uint32_t>(spec.evidenceKind)));

  constexpr char owner[] = "wafer.optimization-adoption-spec";
  std::vector<uint8_t> bytes(owner, owner + sizeof(owner));
  appendU16(bytes, spec.schemaVersion);
  appendU32(bytes, body.size());
  bytes.insert(bytes.end(), body.begin(), body.end());
  return bytes;
}

AdoptionDigest digestAdoptionSpecV1(const AdoptionSpec &spec) {
  return digest(encodeAdoptionSpecV1(spec));
}

AdoptionDigest digestAdoptionWorkPolicyV1(uint32_t workPolicyKind) {
  return digestDefinition("wafer.optimization-work-policy-v1:" +
                          std::to_string(workPolicyKind));
}

bool validateCanonicalAdoptionSpecV1(const std::vector<uint8_t> &bytes,
                                     std::string *diagnostic) {
  auto fail = [&](llvm::StringRef message) {
    if (diagnostic)
      *diagnostic = message.str();
    return false;
  };
  constexpr char owner[] = "wafer.optimization-adoption-spec";
  constexpr size_t prefix = sizeof(owner) + 2 + 4;
  if (bytes.size() < prefix ||
      !std::equal(bytes.begin(), bytes.begin() + sizeof(owner), owner))
    return fail("invalid adoption spec domain separator");
  auto readU16 = [&](size_t offset) {
    return static_cast<uint16_t>((bytes[offset] << 8) | bytes[offset + 1]);
  };
  auto readU32 = [&](size_t offset) {
    uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes[offset + index];
    return value;
  };
  if (readU16(sizeof(owner)) != 1 ||
      readU32(sizeof(owner) + 2) != bytes.size() - prefix)
    return fail("invalid adoption spec envelope size or schema");

  size_t cursor = prefix;
  std::array<TypeTag, 15> expected = {
      U16,       Record,     ClosedEnum, Record, ClosedEnum,
      Record,    ClosedEnum, SortedSet,  ClosedEnum, Record,
      Record,    Record,     Record,     Record, ClosedEnum};
  MechanismKey key;
  for (uint16_t field = 1; field <= 15; ++field) {
    if (cursor + 7 > bytes.size())
      return fail("missing adoption spec field");
    uint16_t number = readU16(cursor);
    TypeTag tag = static_cast<TypeTag>(bytes[cursor + 2]);
    uint32_t size = readU32(cursor + 3);
    cursor += 7;
    if (number != field || tag != expected[field - 1] ||
        size > bytes.size() - cursor)
      return fail("unknown, reordered, mistyped or truncated adoption field");
    if (field == 2) {
      constexpr char keyOwner[] = "wafer.mechanism-key";
      if (size != sizeof(keyOwner) + 2 + 4 ||
          !std::equal(bytes.begin() + cursor,
                      bytes.begin() + cursor + sizeof(keyOwner), keyOwner))
        return fail("invalid mechanism key record");
      size_t keyOffset = cursor + sizeof(keyOwner) + 2;
      key.semanticId = readU32(keyOffset);
    }
    cursor += size;
  }
  if (cursor != bytes.size())
    return fail("extra adoption spec field or trailing bytes");
  std::optional<AdoptionSpec> spec = lookupAdoptionSpec(key);
  if (!spec)
    return fail("unknown adoption mechanism key");
  if (encodeAdoptionSpecV1(*spec) != bytes)
    return fail("adoption spec is not the immutable canonical registry row");
  return true;
}

std::vector<std::string> auditOptimizationAdoptionInventory() {
  std::vector<std::string> diagnostics;
  std::vector<MechanismDescriptor> descriptors = getAllMechanismDescriptors();
  if (descriptors.size() != specs().size())
    diagnostics.push_back("mechanism descriptor/spec cardinality mismatch");
  uint32_t previous = 0;
  for (const AdoptionSpec &spec : specs()) {
    if (spec.mechanismKey.semanticId <= previous)
      diagnostics.push_back("adoption specs are not sorted by unique key");
    previous = spec.mechanismKey.semanticId;
    std::optional<MechanismDescriptor> descriptor =
        lookupMechanismDescriptor(spec.mechanismKey);
    if (!descriptor) {
      diagnostics.push_back("adoption spec has no mechanism descriptor");
      continue;
    }
    if (descriptor->cutPoint != spec.cutPoint)
      diagnostics.push_back("descriptor/spec cut point mismatch for key " +
                            std::to_string(spec.mechanismKey.semanticId));
    if (descriptor->evidenceKind != spec.evidenceKind)
      diagnostics.push_back("descriptor/spec evidence mismatch for key " +
                            std::to_string(spec.mechanismKey.semanticId));
    std::string reason;
    std::vector<uint8_t> bytes = encodeAdoptionSpecV1(spec);
    if (bytes.empty() || !validateCanonicalAdoptionSpecV1(bytes, &reason))
      diagnostics.push_back("invalid canonical spec for key " +
                            std::to_string(spec.mechanismKey.semanticId) +
                            ": " + reason);
  }
  std::sort(diagnostics.begin(), diagnostics.end());
  return diagnostics;
}

std::string toHex(const AdoptionDigest &digestValue) {
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (uint8_t byte : digestValue) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0xf]);
  }
  return result;
}

} // namespace wafer
