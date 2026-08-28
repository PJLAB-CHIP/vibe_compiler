//===- TargetFormatTest.cpp - Closed target format registry tests --------===//

#include "Wafer/Target/Core/TargetFormat.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Conversion/InstrToLLVM/InstrToLLVM.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>

namespace {

using wafer::LogicalFormat;
using wafer::LogicalFormatCategory;
using wafer::TargetConvertParameterKind;
using wafer::TargetFormatConstraint;
using wafer::TargetFormatEngine;

TEST(TargetFormatTest, LogicalDescriptorsExactlyCoverStorageFormats) {
  llvm::ArrayRef<wafer::LogicalFormatDescriptor> descriptors =
      wafer::getLogicalFormatDescriptors();
  ASSERT_EQ(descriptors.size(), 13u);

  std::set<LogicalFormat> formats;
  std::set<std::string> spellings;
  for (const wafer::LogicalFormatDescriptor &descriptor : descriptors) {
    EXPECT_TRUE(formats.insert(descriptor.format).second);
    EXPECT_TRUE(spellings.insert(descriptor.canonicalSpelling.str()).second);
    EXPECT_EQ(wafer::findLogicalFormatDescriptor(descriptor.format),
              &descriptor);
    EXPECT_EQ(wafer::stringifyLogicalFormat(descriptor.format),
              descriptor.canonicalSpelling);

    llvm::Expected<LogicalFormat> parsed =
        wafer::parseLogicalFormat(descriptor.canonicalSpelling);
    ASSERT_TRUE(static_cast<bool>(parsed))
        << llvm::toString(parsed.takeError());
    EXPECT_EQ(*parsed, descriptor.format);
  }

  const wafer::LogicalFormatDescriptor *tf32 =
      wafer::findLogicalFormatDescriptor(LogicalFormat::TF32);
  ASSERT_NE(tf32, nullptr);
  EXPECT_EQ(tf32->storageBits, 32);
  EXPECT_EQ(tf32->semanticBits, 19);
  EXPECT_EQ(tf32->category, LogicalFormatCategory::BinaryFloatingPoint);
  EXPECT_EQ(tf32->exponentBits, 8);
  EXPECT_EQ(tf32->precisionBits, 11);
  EXPECT_EQ(tf32->canonicalMask, UINT64_C(0xffffe000));
  EXPECT_FALSE(tf32->bitpacked);
  EXPECT_TRUE(tf32->hasInfinity);
  EXPECT_TRUE(tf32->hasNaN);
  EXPECT_TRUE(tf32->hasSubnormal);

  const wafer::LogicalFormatDescriptor *boolean =
      wafer::findLogicalFormatDescriptor(LogicalFormat::Bool);
  ASSERT_NE(boolean, nullptr);
  EXPECT_EQ(boolean->storageBits, 1);
  EXPECT_EQ(boolean->semanticBits, 1);
  EXPECT_EQ(boolean->category, LogicalFormatCategory::Boolean);
  EXPECT_EQ(boolean->canonicalMask, UINT64_C(1));
  EXPECT_TRUE(boolean->bitpacked);

  for (const wafer::LogicalFormatDescriptor &descriptor : descriptors) {
    if (descriptor.format != LogicalFormat::Bool) {
      EXPECT_FALSE(descriptor.bitpacked);
    }
    if (descriptor.format != LogicalFormat::TF32) {
      EXPECT_EQ(descriptor.storageBits, descriptor.semanticBits);
    }
    if (descriptor.category == LogicalFormatCategory::BinaryFloatingPoint) {
      EXPECT_GT(descriptor.exponentBits, 0);
      EXPECT_GT(descriptor.precisionBits, 1);
      EXPECT_TRUE(descriptor.hasInfinity);
      EXPECT_TRUE(descriptor.hasNaN);
      EXPECT_TRUE(descriptor.hasSubnormal);
    } else {
      EXPECT_EQ(descriptor.exponentBits, 0);
      EXPECT_EQ(descriptor.precisionBits, 0);
      EXPECT_FALSE(descriptor.hasInfinity);
      EXPECT_FALSE(descriptor.hasNaN);
      EXPECT_FALSE(descriptor.hasSubnormal);
    }
  }
}

TEST(TargetFormatTest, LogicalAndEngineLookupsHaveNoUnknownFallback) {
  EXPECT_EQ(
      wafer::findLogicalFormatDescriptor(static_cast<LogicalFormat>(UINT8_MAX)),
      nullptr);

  for (llvm::StringRef spelling : {llvm::StringRef(), llvm::StringRef("I8"),
                                   llvm::StringRef("unknown-format")}) {
    llvm::Expected<LogicalFormat> parsed = wafer::parseLogicalFormat(spelling);
    ASSERT_FALSE(static_cast<bool>(parsed));
    EXPECT_NE(llvm::toString(parsed.takeError()).find("unknown logical format"),
              std::string::npos);
  }

  llvm::ArrayRef<TargetFormatEngine> engines = wafer::getTargetFormatEngines();
  ASSERT_EQ(engines.size(), 5u);
  std::set<TargetFormatEngine> uniqueEngines;
  std::set<std::string> uniqueSpellings;
  for (TargetFormatEngine engine : engines) {
    EXPECT_TRUE(uniqueEngines.insert(engine).second);
    llvm::StringRef spelling = wafer::stringifyTargetFormatEngine(engine);
    EXPECT_TRUE(uniqueSpellings.insert(spelling.str()).second);
    llvm::Expected<TargetFormatEngine> parsed =
        wafer::parseTargetFormatEngine(spelling);
    ASSERT_TRUE(static_cast<bool>(parsed))
        << llvm::toString(parsed.takeError());
    EXPECT_EQ(*parsed, engine);
  }

  llvm::Expected<TargetFormatEngine> unknown =
      wafer::parseTargetFormatEngine("dte");
  ASSERT_FALSE(static_cast<bool>(unknown));
  EXPECT_NE(
      llvm::toString(unknown.takeError()).find("unknown target format engine"),
      std::string::npos);
}

TEST(TargetFormatTest, DataFormatCodeRegistryExactlyCoversVendorEnum) {
  llvm::ArrayRef<wafer::LogicalFormatDescriptor> formats =
      wafer::getLogicalFormatDescriptors();
  llvm::ArrayRef<wafer::TargetDataFormatCodeRecord> codes =
      wafer::getTargetDataFormatCodeRecords();
  ASSERT_EQ(codes.size(), 13u);
  ASSERT_EQ(codes.size(), formats.size());

  std::set<LogicalFormat> uniqueFormats;
  std::set<uint8_t> uniqueCodes;
  for (size_t index = 0; index < codes.size(); ++index) {
    const wafer::TargetDataFormatCodeRecord &code = codes[index];
    EXPECT_EQ(code.format, formats[index].format);
    EXPECT_EQ(code.dataFormatCode, index);
    EXPECT_TRUE(uniqueFormats.insert(code.format).second);
    EXPECT_TRUE(uniqueCodes.insert(code.dataFormatCode).second);
    EXPECT_EQ(wafer::findTargetDataFormatCode(code.format), &code);
  }
  EXPECT_EQ(
      wafer::findTargetDataFormatCode(static_cast<LogicalFormat>(UINT8_MAX)),
      nullptr);
}

TEST(TargetFormatTest, EncodingMatrixExactlyCoversEveryEngineAndFormat) {
  llvm::ArrayRef<TargetFormatEngine> engines = wafer::getTargetFormatEngines();
  llvm::ArrayRef<wafer::LogicalFormatDescriptor> formats =
      wafer::getLogicalFormatDescriptors();
  llvm::ArrayRef<wafer::TargetFormatEncodingRecord> records =
      wafer::getTargetFormatEncodingRecords();
  ASSERT_EQ(records.size(), 65u);
  ASSERT_EQ(records.size(), engines.size() * formats.size());

  std::set<std::tuple<unsigned, unsigned>> uniqueRows;
  for (size_t engineIndex = 0; engineIndex < engines.size(); ++engineIndex) {
    std::set<uint8_t> uniqueEngineCodes;
    for (size_t formatIndex = 0; formatIndex < formats.size(); ++formatIndex) {
      TargetFormatEngine engine = engines[engineIndex];
      LogicalFormat format = formats[formatIndex].format;
      const wafer::TargetFormatEncodingRecord *record =
          wafer::findTargetFormatEncoding(engine, format);
      ASSERT_NE(record, nullptr);
      EXPECT_TRUE(uniqueRows
                      .insert({static_cast<unsigned>(engine),
                               static_cast<unsigned>(format)})
                      .second);
      const wafer::TargetDataFormatCodeRecord *code =
          wafer::findTargetDataFormatCode(format);
      ASSERT_NE(code, nullptr);
      EXPECT_EQ(record->dataFormatCode, code->dataFormatCode);
      EXPECT_TRUE(uniqueEngineCodes.insert(record->dataFormatCode).second);
    }
  }
  EXPECT_EQ(uniqueRows.size(), 65u);

  auto expectConstraint = [](TargetFormatEngine engine,
                             TargetFormatConstraint constraint) {
    const wafer::TargetFormatEncodingRecord *record =
        wafer::findTargetFormatEncoding(engine, LogicalFormat::Bool);
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->constraint, constraint);
  };
  expectConstraint(TargetFormatEngine::RDMA,
                   TargetFormatConstraint::BitpackedDMA);
  expectConstraint(TargetFormatEngine::WDMA,
                   TargetFormatConstraint::BitpackedDMA);
  expectConstraint(TargetFormatEngine::CT,
                   TargetFormatConstraint::BitpackedLayout);
  expectConstraint(TargetFormatEngine::TDMA,
                   TargetFormatConstraint::BitpackedLayout);

  EXPECT_EQ(wafer::findTargetFormatEncoding(
                static_cast<TargetFormatEngine>(UINT8_MAX), LogicalFormat::I8),
            nullptr);
  EXPECT_EQ(
      wafer::findTargetFormatEncoding(TargetFormatEngine::RDMA,
                                      static_cast<LogicalFormat>(UINT8_MAX)),
      nullptr);
}

TEST(TargetFormatTest, SupportedEngineCodesRoundTripThroughTypedDecoder) {
  for (const wafer::TargetFormatEncodingRecord &record :
       wafer::getTargetFormatEncodingRecords()) {
    llvm::Expected<wafer::LogicalFormat> decoded =
        wafer::decodeTargetFormat(record.engine, record.dataFormatCode);
    ASSERT_TRUE(static_cast<bool>(decoded))
        << llvm::toString(decoded.takeError());
    EXPECT_EQ(*decoded, record.format);
  }

  auto unsupported =
      wafer::decodeTargetFormat(wafer::TargetFormatEngine::CT, 255);
  ASSERT_FALSE(static_cast<bool>(unsupported));
  EXPECT_NE(llvm::toString(unsupported.takeError()).find("unsupported"),
            std::string::npos);
}

struct ExpectedConvertRoute {
  uint16_t opcode;
  llvm::StringLiteral spelling;
  LogicalFormat source;
  LogicalFormat destination;
  TargetConvertParameterKind parameter;
};

constexpr ExpectedConvertRoute kExpectedConvertRoutes[] = {
    {139, "int8_fp16", LogicalFormat::I8, LogicalFormat::F16,
     TargetConvertParameterKind::ZeroPoint},
    {140, "int8_bf16", LogicalFormat::I8, LogicalFormat::BF16,
     TargetConvertParameterKind::ZeroPoint},
    {141, "int8_fp32", LogicalFormat::I8, LogicalFormat::F32,
     TargetConvertParameterKind::ZeroPoint},
    {142, "int8_tf32", LogicalFormat::I8, LogicalFormat::TF32,
     TargetConvertParameterKind::ZeroPoint},
    {143, "int16_fp16", LogicalFormat::I16, LogicalFormat::F16,
     TargetConvertParameterKind::None},
    {144, "int16_bf16", LogicalFormat::I16, LogicalFormat::BF16,
     TargetConvertParameterKind::RoundingMode},
    {145, "int16_fp32", LogicalFormat::I16, LogicalFormat::F32,
     TargetConvertParameterKind::RoundingMode},
    {146, "int16_tf32", LogicalFormat::I16, LogicalFormat::TF32,
     TargetConvertParameterKind::RoundingMode},
    {147, "int32_fp16", LogicalFormat::I32, LogicalFormat::F16,
     TargetConvertParameterKind::RoundingMode},
    {148, "int32_bf16", LogicalFormat::I32, LogicalFormat::BF16,
     TargetConvertParameterKind::RoundingMode},
    {149, "int32_fp32", LogicalFormat::I32, LogicalFormat::F32,
     TargetConvertParameterKind::RoundingMode},
    {150, "int32_tf32", LogicalFormat::I32, LogicalFormat::TF32,
     TargetConvertParameterKind::RoundingMode},
    {151, "bf16_int8", LogicalFormat::BF16, LogicalFormat::I8,
     TargetConvertParameterKind::None},
    {152, "bf16_int16", LogicalFormat::BF16, LogicalFormat::I16,
     TargetConvertParameterKind::RoundingMode},
    {153, "bf16_int32", LogicalFormat::BF16, LogicalFormat::I32,
     TargetConvertParameterKind::RoundingMode},
    {154, "bf16_fp16", LogicalFormat::BF16, LogicalFormat::F16,
     TargetConvertParameterKind::None},
    {155, "bf16_fp32", LogicalFormat::BF16, LogicalFormat::F32,
     TargetConvertParameterKind::None},
    {156, "bf16_tf32", LogicalFormat::BF16, LogicalFormat::TF32,
     TargetConvertParameterKind::None},
    {157, "fp16_int8", LogicalFormat::F16, LogicalFormat::I8,
     TargetConvertParameterKind::RoundingMode},
    {158, "fp16_int16", LogicalFormat::F16, LogicalFormat::I16,
     TargetConvertParameterKind::RoundingMode},
    {159, "fp16_int32", LogicalFormat::F16, LogicalFormat::I32,
     TargetConvertParameterKind::RoundingMode},
    {160, "fp16_bf16", LogicalFormat::F16, LogicalFormat::BF16,
     TargetConvertParameterKind::RoundingMode},
    {161, "fp16_fp32", LogicalFormat::F16, LogicalFormat::F32,
     TargetConvertParameterKind::None},
    {162, "fp16_tf32", LogicalFormat::F16, LogicalFormat::TF32,
     TargetConvertParameterKind::None},
    {163, "fp32_int8", LogicalFormat::F32, LogicalFormat::I8,
     TargetConvertParameterKind::RoundingMode},
    {164, "fp32_int16", LogicalFormat::F32, LogicalFormat::I16,
     TargetConvertParameterKind::RoundingMode},
    {165, "fp32_int32", LogicalFormat::F32, LogicalFormat::I32,
     TargetConvertParameterKind::RoundingMode},
    {166, "fp32_fp16", LogicalFormat::F32, LogicalFormat::F16,
     TargetConvertParameterKind::RoundingMode},
    {167, "fp32_bf16", LogicalFormat::F32, LogicalFormat::BF16,
     TargetConvertParameterKind::RoundingMode},
    {168, "fp32_tf32", LogicalFormat::F32, LogicalFormat::TF32,
     TargetConvertParameterKind::RoundingMode},
    {169, "tf32_int8", LogicalFormat::TF32, LogicalFormat::I8,
     TargetConvertParameterKind::RoundingMode},
    {170, "tf32_int16", LogicalFormat::TF32, LogicalFormat::I16,
     TargetConvertParameterKind::RoundingMode},
    {171, "tf32_int32", LogicalFormat::TF32, LogicalFormat::I32,
     TargetConvertParameterKind::RoundingMode},
    {172, "tf32_fp16", LogicalFormat::TF32, LogicalFormat::F16,
     TargetConvertParameterKind::None},
    {173, "tf32_bf16", LogicalFormat::TF32, LogicalFormat::BF16,
     TargetConvertParameterKind::RoundingMode},
    {174, "tf32_fp32", LogicalFormat::TF32, LogicalFormat::F32,
     TargetConvertParameterKind::None},
};

TEST(TargetFormatTest, TypedConvertRoutesExactlyMatchOpcodeContract) {
  llvm::ArrayRef<wafer::TargetConvertRoute> routes =
      wafer::getTargetConvertRoutes();
  ASSERT_EQ(routes.size(), std::size(kExpectedConvertRoutes));
  ASSERT_EQ(routes.size(), 36u);

  std::set<uint16_t> opcodes;
  std::set<std::string> spellings;
  std::set<std::tuple<LogicalFormat, LogicalFormat>> typePairs;
  size_t zeroPointCount = 0;
  size_t roundingCount = 0;
  size_t parameterlessCount = 0;

  for (size_t index = 0; index < routes.size(); ++index) {
    const wafer::TargetConvertRoute &route = routes[index];
    const ExpectedConvertRoute &expected = kExpectedConvertRoutes[index];
    EXPECT_EQ(route.opcode, expected.opcode);
    EXPECT_EQ(route.canonicalSpelling, expected.spelling);
    EXPECT_EQ(route.source, expected.source);
    EXPECT_EQ(route.destination, expected.destination);
    EXPECT_EQ(route.parameterKind, expected.parameter);
    EXPECT_TRUE(opcodes.insert(route.opcode).second);
    EXPECT_TRUE(spellings.insert(route.canonicalSpelling.str()).second);
    EXPECT_TRUE(typePairs.insert({route.source, route.destination}).second);

    EXPECT_EQ(wafer::findTargetConvertRoute(route.opcode), &route);
    EXPECT_EQ(wafer::findTargetConvertRoute(route.source, route.destination),
              &route);
    EXPECT_EQ(wafer::findTargetConvertRoute(route.canonicalSpelling), &route);
    EXPECT_FALSE(wafer::stringifyTargetConvertParameterKind(route.parameterKind)
                     .empty());

    switch (route.parameterKind) {
    case TargetConvertParameterKind::ZeroPoint:
      ++zeroPointCount;
      break;
    case TargetConvertParameterKind::RoundingMode:
      ++roundingCount;
      break;
    case TargetConvertParameterKind::None:
      ++parameterlessCount;
      break;
    }
  }

  EXPECT_EQ(zeroPointCount, 4u);
  EXPECT_EQ(roundingCount, 23u);
  EXPECT_EQ(parameterlessCount, 9u);
  EXPECT_EQ(opcodes.size(), 36u);
  EXPECT_EQ(spellings.size(), 36u);
  EXPECT_EQ(typePairs.size(), 36u);
}

TEST(TargetFormatTest, TypedConvertWhitelistDoesNotOpenGenericCTRows) {
  EXPECT_EQ(wafer::findTargetConvertRoute(138), nullptr);
  EXPECT_EQ(wafer::findTargetConvertRoute(175), nullptr);
  EXPECT_EQ(wafer::findTargetConvertRoute("unknown_convert"), nullptr);

  for (const wafer::LogicalFormatDescriptor &format :
       wafer::getLogicalFormatDescriptors())
    EXPECT_EQ(wafer::findTargetConvertRoute(format.format, format.format),
              nullptr);

  for (LogicalFormat excluded :
       {LogicalFormat::Bool, LogicalFormat::U8, LogicalFormat::U16,
        LogicalFormat::U32, LogicalFormat::I64, LogicalFormat::U64}) {
    EXPECT_EQ(wafer::findTargetConvertRoute(excluded, LogicalFormat::F32),
              nullptr);
    EXPECT_EQ(wafer::findTargetConvertRoute(LogicalFormat::F32, excluded),
              nullptr);
  }

  for (LogicalFormat genericFormat :
       {LogicalFormat::I16, LogicalFormat::I32, LogicalFormat::TF32}) {
    const wafer::TargetFormatEncodingRecord *generic =
        wafer::findTargetFormatEncoding(TargetFormatEngine::CT, genericFormat);
    ASSERT_NE(generic, nullptr);
  }
  EXPECT_NE(
      wafer::findTargetConvertRoute(LogicalFormat::I16, LogicalFormat::F16),
      nullptr);
  EXPECT_NE(
      wafer::findTargetConvertRoute(LogicalFormat::I32, LogicalFormat::F32),
      nullptr);
  EXPECT_NE(
      wafer::findTargetConvertRoute(LogicalFormat::TF32, LogicalFormat::F32),
      nullptr);
}

TEST(TargetFormatTest, EveryTypedConvertRoutePassesTargetFormatVerification) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  for (const wafer::TargetConvertRoute &route :
       wafer::getTargetConvertRoutes()) {
    SCOPED_TRACE(route.canonicalSpelling.str());
    std::string source;
    llvm::raw_string_ostream os(source);
    os << "module {\n"
          "  func.func @main() {\n"
          "    %source = memref.alloc() {wafer.spm.offset = "
          "#wafer.spm_offset<65536>} : memref<1x"
       << wafer::stringifyLogicalFormat(route.source)
       << ", #wafer.memory<spm, tensor>>\n"
          "    %dest = memref.alloc() {wafer.spm.offset = "
          "#wafer.spm_offset<65792>} : memref<1x"
       << wafer::stringifyLogicalFormat(route.destination)
       << ", #wafer.memory<spm, tensor>>\n"
          "    wafer.instr.convert #wafer.instr_convert_kind<"
       << route.canonicalSpelling << "> %source into %dest";
    switch (route.parameterKind) {
    case TargetConvertParameterKind::None:
      break;
    case TargetConvertParameterKind::RoundingMode:
      os << " {rounding_mode = 0 : i64}";
      break;
    case TargetConvertParameterKind::ZeroPoint:
      os << " {zero_point = 0 : i64}";
      break;
    }
    os << " : memref<1x" << wafer::stringifyLogicalFormat(route.source)
       << ", #wafer.memory<spm, tensor>> to memref<1x"
       << wafer::stringifyLogicalFormat(route.destination)
       << ", #wafer.memory<spm, tensor>>\n"
          "    return\n"
          "  }\n"
          "}\n";
    os.flush();

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(source,
                                                mlir::ParserConfig(&context));
    ASSERT_TRUE(module) << source;
    mlir::PassManager manager(&context);
    wafer::TargetConversionRequest request{};
    manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
    ASSERT_TRUE(mlir::succeeded(manager.run(*module))) << source;

    bool hasWaferOperation = false;
    module->walk([&](mlir::Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "wafer")
        hasWaferOperation = true;
    });
    EXPECT_FALSE(hasWaferOperation);
  }
}

static mlir::OwningOpRef<mlir::ModuleOp>
parseNCCJoinModule(mlir::MLIRContext &context, uint64_t operationCount) {
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n  func.func @main() {\n";
  for (uint64_t index = 0; index < operationCount; ++index)
    os << "    wafer.instr.ncc_join [0]\n";
  os << "    return\n  }\n}\n";
  os.flush();
  return mlir::parseSourceString<mlir::ModuleOp>(source,
                                                 mlir::ParserConfig(&context));
}

TEST(TargetFormatTest, DirectTargetAcceptsExactly4096TerminalOperations) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module = parseNCCJoinModule(context, 4096);
  ASSERT_TRUE(module);
  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
  ASSERT_TRUE(mlir::succeeded(manager.run(*module)));
  EXPECT_EQ(module->lookupSymbol("main")->getName().getStringRef(),
            "llvm.func");
}

TEST(TargetFormatTest, DirectTargetAccepts4097ExecutableOperations) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module = parseNCCJoinModule(context, 4097);
  ASSERT_TRUE(module);
  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
  ASSERT_TRUE(mlir::succeeded(manager.run(*module)));
  EXPECT_EQ(module->lookupSymbol("main")->getName().getStringRef(),
            "llvm.func");
}

TEST(TargetFormatTest,
     TargetFormatVerificationRejectsResidualReduceInitAtomically) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %input = memref.alloc() : memref<2x3xf16, #wafer.memory<spm, cx>>
    %dest = memref.alloc() : memref<2xf16, #wafer.memory<spm, cx>>
    wafer.instr.reduce #wafer.instr_reduce_kind<sum> %input into %dest
        {dim = 0 : i64}
        : memref<2x3xf16, #wafer.memory<spm, cx>>
      into memref<2xf16, #wafer.memory<spm, cx>>
    return
  }
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  wafer::InstrReduceOp reduce;
  module->walk([&](wafer::InstrReduceOp op) { reduce = op; });
  ASSERT_TRUE(reduce);
  reduce->setAttr("init_value",
                  mlir::FloatAttr::get(mlir::Float16Type::get(&context), 0.0));

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        return mlir::success();
      });
  mlir::PassManager manager(&context);
  manager.enableVerifier(false);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
  EXPECT_TRUE(mlir::failed(manager.run(*module)));
  EXPECT_NE(diagnostics.find(
                "does not accept schema-free semantic attribute 'init_value'"),
            std::string::npos)
      << diagnostics;
  EXPECT_TRUE(reduce->hasAttr("init_value"));
  EXPECT_EQ(module->lookupSymbol("main")->getName().getStringRef(),
            "func.func");
}

} // namespace
