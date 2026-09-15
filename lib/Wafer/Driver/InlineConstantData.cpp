//===- InlineConstantData.cpp - Own current tensor literals ---------------===//
#include "InlineConstantData.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
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
