//===- InlineConstantData.cpp - Own current tensor literals ---------------===//
#include "InlineConstantData.h"
#include "Wafer/Analysis/Module/ExecutableCallClosure.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <optional>

namespace wafer::compiler::detail {
namespace {
struct Encoding {
  ProgramElementType type;
  llvm::StringRef descr;
  unsigned bytes;
};
std::optional<Encoding> encoding(mlir::Type type) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  if (integer && !integer.isSignless())
    return std::nullopt;
  if (type.isInteger(1))
    return Encoding{ProgramElementType::Bool, "|b1", 1};
  if (type.isInteger(8))
    return Encoding{ProgramElementType::I8, "|i1", 1};
  if (type.isInteger(16))
    return Encoding{ProgramElementType::I16, "<i2", 2};
  if (type.isInteger(32))
    return Encoding{ProgramElementType::I32, "<i4", 4};
  if (type.isInteger(64))
    return Encoding{ProgramElementType::I64, "<i8", 8};
  if (type.isF16())
    return Encoding{ProgramElementType::F16, "<f2", 2};
  if (type.isBF16())
    return Encoding{ProgramElementType::BF16, "|V2", 2};
  if (type.isF32())
    return Encoding{ProgramElementType::F32, "<f4", 4};
  return std::nullopt;
}

llvm::Expected<SourceDataId> establishLiteral(mlir::DenseElementsAttr value,
                                              const Encoding &format,
                                              llvm::StringRef locator,
                                              ProgramDataHandoff &data) {
  int fd;
  llvm::SmallString<128> path;
  if (auto error =
          llvm::sys::fs::createTemporaryFile("wafer-constant", "npy", fd, path))
    return llvm::errorCodeToError(error);
  llvm::FileRemover remove(path);
  {
    llvm::raw_fd_ostream stream(fd, true);
    std::string header;
    llvm::raw_string_ostream text(header);
    text << "{'descr': '" << format.descr
         << "', 'fortran_order': False, 'shape': (";
    for (int64_t size : value.getType().getShape())
      text << size << ", ";
    text << "), }";
    text.flush();
    header.append((64 - ((10 + header.size() + 1) % 64)) % 64, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<uint16_t>::max())
      return llvm::createStringError("inline constant NPY header is too large");
    stream.write("\x93NUMPY\x01\x00", 8);
    stream << static_cast<char>(header.size() & 255)
           << static_cast<char>((header.size() >> 8) & 255);
    stream << header;
    llvm::SmallVector<char, 4096> window;
    for (auto element : value.getValues<mlir::Attribute>()) {
      llvm::APInt bits = mlir::isa<mlir::IntegerAttr>(element)
                             ? mlir::cast<mlir::IntegerAttr>(element).getValue()
                             : mlir::cast<mlir::FloatAttr>(element)
                                   .getValue()
                                   .bitcastToAPInt();
      uint64_t raw = bits.getZExtValue();
      for (unsigned byte = 0; byte < format.bytes; ++byte)
        window.push_back(static_cast<char>((raw >> (8 * byte)) & 255));
      if (window.size() >= 4096) {
        stream.write(window.data(), window.size());
        window.clear();
      }
    }
    stream.write(window.data(), window.size());
    stream.flush();
    if (stream.has_error()) {
      stream.clear_error();
      return llvm::createStringError("cannot write inline constant payload");
    }
  }
  ProgramDataFailure failure;
  return data.establishSource(path, locator, &failure);
}
} // namespace

llvm::Expected<std::vector<std::vector<ProgramResourceBinding>>>
bindInlineConstantBuffers(llvm::ArrayRef<mlir::ModuleOp> modules,
                          ProgramDataHandoff &data) {
  struct Literal {
    mlir::DenseElementsAttr value;
    mlir::MemRefType type;
    Encoding format;
  };
  llvm::SmallVector<Literal> literals;
  llvm::DenseMap<mlir::Attribute, unsigned> indices;
  llvm::SmallVector<mlir::func::FuncOp> entries;
  llvm::SmallVector<
      llvm::SmallVector<std::pair<mlir::memref::GetGlobalOp, unsigned>>>
      reads;
  // Preflight every current use before changing the module set.
  for (mlir::ModuleOp module : modules) {
    auto closure = analyzeExecutableCallClosure(module);
    if (!closure)
      return closure.takeError();
    entries.push_back(closure->entry);
    reads.emplace_back();
    bool invalid = false;
    module.walk([&](mlir::memref::GetGlobalOp read) {
      auto global =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
              read, read.getNameAttr());
      if (!global || !global.getInitialValue())
        return; // External resource declarations have their own binding.
      auto value =
          mlir::dyn_cast<mlir::DenseElementsAttr>(*global.getInitialValue());
      auto type = read.getType();
      auto memory = mlir::dyn_cast_or_null<MemoryAttr>(type.getMemorySpace());
      auto format = encoding(type.getElementType());
      if (!global.getConstant() || !value || !format || !memory ||
          memory.getSpace() != MemorySpace::DDR ||
          memory.getLayout() != MemLayout::Tensor ||
          !type.getLayout().isIdentity() ||
          read->getParentOfType<mlir::func::FuncOp>() != closure->entry) {
        invalid = true;
        return;
      }
      auto inserted = indices.try_emplace(value, literals.size());
      if (inserted.second)
        literals.push_back({value, type, *format});
      else if (literals[inserted.first->second].type != type)
        invalid = true;
      reads.back().emplace_back(read, inserted.first->second);
    });
    if (invalid)
      return llvm::createStringError(
          "inline buffer constant requires an immutable Tensor DDR literal in "
          "the entry");
  }
  std::vector<ProgramResourceBinding> bindings;
  int64_t next = 0;
  for (const auto &range : data.getRanges())
    if (range.getTensorId().role == ProgramResourceRole::Constant)
      next = std::max(next, range.getTensorId().roleIndex + 1);
  for (const Literal &literal : literals) {
    std::string identity;
    llvm::raw_string_ostream stream(identity);
    literal.value.getType().print(stream);
    stream.write(literal.value.getRawData().data(),
                 literal.value.getRawData().size());
    llvm::SHA256 hash;
    hash.update(stream.str());
    std::string locator = "constants/local_" + llvm::toHex(hash.final(), true);
    const ProgramDataRange *existing = nullptr;
    for (const auto &range : data.getRanges())
      if (range.getTensorId().role == ProgramResourceRole::Constant &&
          data.findByLocator(locator) == &data.getSource(range.getSourceId())) {
        existing = &range;
        break;
      }
    ProgramTensorId id{ProgramResourceRole::Constant,
                       existing ? existing->getTensorId().roleIndex : next++};
    auto shape = literal.type.getShape();
    llvm::SmallVector<int64_t> zeros(shape.size(), 0), ones(shape.size(), 1);
    if (!existing) {
      auto source =
          establishLiteral(literal.value, literal.format, locator, data);
      if (!source)
        return source.takeError();
      auto range = ProgramDataRange::create(
          id, literal.format.type, shape, shape,
          frontend::ProgramDistributionKind::Replicated,
          ProgramDataRangeOrigin::OriginalSource, zeros, shape, ones, *source,
          data.getSource(*source), nullptr);
      if (!range)
        return range.takeError();
      if (auto error = data.addRange(std::move(*range)))
        return std::move(error);
    }
    frontend::ProgramPartitionSlice slice;
    slice.partitionId = 0;
    slice.replicaId = 0;
    slice.offsets.assign(zeros.begin(), zeros.end());
    slice.sizes.assign(shape.begin(), shape.end());
    slice.strides.assign(ones.begin(), ones.end());
    slice.payloadPath = locator;
    bindings.push_back({ProgramResourceRole::Constant,
                        id,
                        -1,
                        id.roleIndex,
                        {},
                        literal.format.type,
                        frontend::ProgramDistributionKind::Replicated,
                        {shape.begin(), shape.end()},
                        {shape.begin(), shape.end()},
                        std::move(slice)});
  }
  std::vector<std::vector<ProgramResourceBinding>> result(modules.size(),
                                                          bindings);
  if (modules.empty())
    return result;
  mlir::IRRewriter rewriter(modules.front()->getContext());
  for (auto [tile, entry] : llvm::enumerate(entries)) {
    llvm::SmallVector<mlir::Value> arguments;
    for (auto [i, literal] : llvm::enumerate(literals)) {
      unsigned index = entry.getNumArguments();
      entry.insertArgument(index, literal.type, mlir::DictionaryAttr{},
                           entry.getLoc());
      arguments.push_back(entry.getArgument(index));
      result[tile][i].index = index;
    }
    llvm::DenseMap<std::pair<mlir::Operation *, unsigned>, mlir::Value>
        regionInputs;
    for (auto [read, index] : reads[tile]) {
      llvm::SmallVector<TileRegionOp> scopes;
      for (auto *parent = read->getParentOp(); parent != entry;
           parent = parent->getParentOp())
        if (auto region = mlir::dyn_cast<TileRegionOp>(parent))
          scopes.push_back(region);
      mlir::Value value = arguments[index];
      for (auto region : llvm::reverse(scopes)) {
        auto key = std::make_pair(region.getOperation(), index);
        auto found = regionInputs.find(key);
        if (found == regionInputs.end()) {
          region.getInputsMutable().append(value);
          value = region.getBody().front().addArgument(value.getType(),
                                                       read.getLoc());
          regionInputs.try_emplace(key, value);
        } else {
          value = found->second;
        }
      }
      rewriter.replaceOp(read, value);
    }
    if (mlir::failed(mlir::verify(modules[tile])))
      return llvm::createStringError(
          "inline buffer constant binding produced invalid IR");
  }
  return result;
}

llvm::Error
outlineInlineConstantData(mlir::ModuleOp module,
                          frontend::FrontendProgramVerificationResult &program,
                          ProgramDataHandoff &data) {
  mlir::func::FuncOp function;
  for (auto candidate : module.getOps<mlir::func::FuncOp>())
    if (!candidate.isExternal()) {
      if (function)
        return llvm::createStringError(
            "inline constants require one program function");
      function = candidate;
    }
  if (!function || !function.getBody().hasOneBlock())
    return llvm::createStringError(
        "inline constants require a single-block program");
  llvm::SmallVector<mlir::arith::ConstantOp> constants;
  for (auto constant : function.getOps<mlir::arith::ConstantOp>()) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(constant.getType());
    auto value = mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
    if (tensor && tensor.hasStaticShape() && value && !value.isSplat() &&
        !constant->use_empty())
      constants.push_back(constant);
  }
  int64_t next = 0;
  for (const auto &constant : program.constants)
    next = std::max(next, constant.position + 1);
  llvm::DenseMap<mlir::Attribute, mlir::Value> arguments;
  mlir::IRRewriter rewriter(module.getContext());
  for (auto constant : constants) {
    auto value = mlir::cast<mlir::DenseElementsAttr>(constant.getValue());
    if (auto found = arguments.find(value); found != arguments.end()) {
      rewriter.replaceOp(constant, found->second);
      continue;
    }
    auto type = mlir::cast<mlir::RankedTensorType>(constant.getType());
    auto format = encoding(type.getElementType());
    if (!format)
      return llvm::createStringError(
          "inline constant element type has no program encoding");
    std::string locator = "constants/inline_" + std::to_string(next);
    auto source = establishLiteral(value, *format, locator, data);
    if (!source)
      return source.takeError();
    llvm::SmallVector<int64_t> zeros(type.getRank(), 0),
        ones(type.getRank(), 1);
    auto range = ProgramDataRange::create(
        {ProgramResourceRole::Constant, next}, format->type, type.getShape(),
        type.getShape(), frontend::ProgramDistributionKind::Replicated,
        ProgramDataRangeOrigin::OriginalSource, zeros, type.getShape(), ones,
        *source, data.getSource(*source), nullptr);
    if (!range)
      return range.takeError();
    if (auto error = data.addRange(std::move(*range)))
      return error;
    unsigned index = function.getNumArguments();
    function.insertArgument(index, type, mlir::DictionaryAttr{},
                            constant.getLoc());
    auto argument = function.getArgument(index);
    program.constants.push_back(
        {static_cast<int64_t>(index), next++,
         std::vector<int64_t>(type.getShape().begin(), type.getShape().end()),
         format->type, locator});
    ++program.programConstantCount;
    arguments[value] = argument;
    rewriter.replaceOp(constant, argument);
  }
  if (mlir::failed(mlir::verify(module)))
    return llvm::createStringError(
        "inline constant data binding produced invalid IR");
  return llvm::Error::success();
}
} // namespace wafer::compiler::detail
