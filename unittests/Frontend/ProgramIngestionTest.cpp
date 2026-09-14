//===- ProgramIngestionTest.cpp - StableHLO source boundary tests --------===//

#include "Wafer/Frontend/StableHLO/ProgramIngestion.h"
#include "Wafer/Conversion/Passes.h"

#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/reference/Api.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <string>

namespace {

class SourceMathIngestionTest : public testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<mlir::func::FuncDialect,
                        mlir::stablehlo::StablehloDialect>();
    ASSERT_FALSE(
        llvm::sys::fs::createUniqueDirectory("wafer-source-math", root));
    artifact = root;
    llvm::sys::path::append(artifact, "functions");
    ASSERT_FALSE(llvm::sys::fs::create_directory(artifact));
    llvm::sys::path::append(artifact, "forward.stablehlo.bc");
  }

  void TearDown() override {
    if (!root.empty()) {
      EXPECT_FALSE(llvm::sys::fs::remove_directories(root));
    }
  }

  void writeSource(llvm::StringRef text) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    std::error_code error;
    llvm::raw_fd_ostream output(artifact, error);
    ASSERT_FALSE(error);
    ASSERT_TRUE(mlir::succeeded(mlir::stablehlo::serializePortableArtifact(
        *module, mlir::vhlo::Version::getCurrentVersion().toString(), output)));
  }

  std::string erfSource(llvm::StringRef type, llvm::StringRef attributes,
                        llvm::StringRef target = "mhlo.erf",
                        bool precedingValidCall = false) {
    std::string source;
    llvm::raw_string_ostream out(source);
    out << "module { func.func @main(%arg: " << type << ") -> " << type
        << " {\n";
    if (precedingValidCall)
      out << "%first = stablehlo.custom_call @mhlo.erf(%arg) "
             "{mhlo.version = 1 : i64, mhlo.attributes = {}} : ("
          << type << ") -> " << type << "\n";
    out << "%result = stablehlo.custom_call @" << target << "("
        << (precedingValidCall ? "%first" : "%arg") << ") {" << attributes
        << "} : (" << type << ") -> " << type << "\nreturn %result : " << type
        << "\n}}";
    return source;
  }

  mlir::MLIRContext context;
  llvm::SmallString<256> root, artifact;
};

TEST_F(SourceMathIngestionTest, LegalizesPortableErfToVerifiedPureStableHLO) {
  for (llvm::StringRef element : {"f16", "bf16", "f32", "f64"}) {
    for (int64_t extent : {1024, 1025, 1031}) {
      std::string type =
          "tensor<1x" + std::to_string(extent) + "x1x" + element.str() + ">";
      SCOPED_TRACE(type);
      writeSource(
          erfSource(type, "mhlo.version = 1 : i64, mhlo.attributes = {}"));
      ASSERT_FALSE(HasFatalFailure());
      std::string diagnostics;
      llvm::raw_string_ostream stream(diagnostics);
      auto module = wafer::frontend::deserializeStableHLOProgramDirectory(
          root, context, stream);
      ASSERT_TRUE(mlir::succeeded(module)) << diagnostics;
      ASSERT_TRUE(mlir::succeeded(
          wafer::frontend::verifyStableHLOSourceModule(**module, stream)))
          << diagnostics;
      unsigned arithmetic = 0;
      (*module)->walk([&](mlir::Operation *op) {
        EXPECT_FALSE(mlir::isa<mlir::stablehlo::CustomCallOp>(op));
        EXPECT_NE(op->getName().getDialectNamespace(), "chlo");
        arithmetic += mlir::isa<mlir::stablehlo::MulOp>(op);
      });
      EXPECT_GT(arithmetic, 0u);
      auto entry = (*module)->lookupSymbol<mlir::func::FuncOp>("main");
      ASSERT_TRUE(entry);
      EXPECT_EQ(entry.getArgument(0).getType(), entry.getResultTypes()[0]);
      auto tensorType =
          mlir::cast<mlir::RankedTensorType>(entry.getArgument(0).getType());
      EXPECT_EQ(tensorType.getShape(), (llvm::ArrayRef<int64_t>{1, extent, 1}));

      // Independent libm oracle over the actual normalized module. Repeated
      // values include negative/positive saturation, signed zero, and NaN/Inf.
      const double samples[] = {-12.,
                                -3.,
                                -1.,
                                -.125,
                                -0.,
                                0.,
                                .125,
                                1.,
                                3.,
                                12.,
                                -std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::quiet_NaN()};
      auto floatType = mlir::cast<mlir::FloatType>(tensorType.getElementType());
      llvm::SmallVector<llvm::APFloat> values;
      for (int64_t i = 0; i < extent; ++i) {
        llvm::APFloat value(samples[i % std::size(samples)]);
        bool losesInfo;
        value.convert(floatType.getFloatSemantics(),
                      llvm::APFloat::rmNearestTiesToEven, &losesInfo);
        values.push_back(value);
      }
      auto input = mlir::DenseElementsAttr::get(tensorType, values);
      auto result = mlir::stablehlo::evalModule(
          **module, llvm::ArrayRef<mlir::DenseElementsAttr>{input}, {});
      ASSERT_TRUE(mlir::succeeded(result));
      ASSERT_EQ(result->size(), 1u);
      EXPECT_EQ(result->front().getType(), tensorType);
      int64_t index = 0;
      for (llvm::APFloat actual : result->front().getValues<llvm::APFloat>()) {
        double expected = std::erf(values[index++].convertToDouble());
        double got = actual.convertToDouble();
        if (std::isnan(expected))
          EXPECT_TRUE(std::isnan(got));
        else {
          double tolerance = element == "f64"   ? 1e-14
                             : element == "f32" ? 2e-7
                             : element == "f16" ? 1e-3
                                                : 1.6e-2;
          double absolute = element == "f64" ? 1e-14 : 1e-7;
          EXPECT_NEAR(got, expected, absolute + tolerance * std::abs(expected));
          if (expected == 0.) {
            EXPECT_EQ(std::signbit(got), std::signbit(expected));
          }
        }
      }
      mlir::PassManager downstream(&context);
      downstream.addPass(wafer::createLegalizeStablehloToLinalgPass());
      ASSERT_TRUE(mlir::succeeded(downstream.run(**module)));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(**module)));
      unsigned structured = 0;
      (*module)->walk([&](mlir::Operation *op) {
        EXPECT_NE(op->getName().getDialectNamespace(), "stablehlo");
        EXPECT_NE(op->getName().getDialectNamespace(), "chlo");
        structured += mlir::isa<mlir::linalg::GenericOp>(op);
      });
      EXPECT_GT(structured, 0u);
      auto loweredEntry = (*module)->lookupSymbol<mlir::func::FuncOp>("main");
      ASSERT_TRUE(loweredEntry);
      EXPECT_EQ(loweredEntry.getResultTypes()[0], tensorType);
    }
  }
}

TEST_F(SourceMathIngestionTest,
       RejectsUnknownContractsWithoutPublishingAModule) {
  for (llvm::StringRef attributes :
       {"mhlo.version = 2 : i64, mhlo.attributes = {}",
        "mhlo.version = 1 : i32, mhlo.attributes = {}", "mhlo.attributes = {}",
        "mhlo.version = 1 : i64, mhlo.attributes = {unknown = 1 : i64}",
        "mhlo.version = 1 : i64, mhlo.attributes = {}, has_side_effect = true",
        "mhlo.version = 1 : i64, mhlo.attributes = {}, api_version = 4 : i32",
        "mhlo.version = 1 : i64, mhlo.attributes = {}, backend_config = "
        "\"configured\"",
        "mhlo.version = 1 : i64, mhlo.attributes = {}, operand_layouts = "
        "[dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, "
        "0]> : tensor<3xindex>]",
        "mhlo.version = 1 : i64, mhlo.attributes = {}, output_operand_aliases "
        "= [#stablehlo.output_operand_alias<output_tuple_indices = [], "
        "operand_index = 0, operand_tuple_indices = []>]",
        "mhlo.version = 1 : i64, mhlo.attributes = {}, unknown = true"}) {
    SCOPED_TRACE(attributes.str());
    writeSource(
        erfSource("tensor<1x1025x1xf16>", attributes, "mhlo.erf", true));
    ASSERT_FALSE(HasFatalFailure());
    auto before = llvm::MemoryBuffer::getFile(artifact);
    ASSERT_TRUE(before);
    std::string diagnostics;
    llvm::raw_string_ostream stream(diagnostics);
    EXPECT_TRUE(
        mlir::failed(wafer::frontend::deserializeStableHLOProgramDirectory(
            root, context, stream)));
    EXPECT_NE(
        diagnostics.find("unsupported stablehlo.custom_call source contract"),
        std::string::npos)
        << diagnostics;
    auto after = llvm::MemoryBuffer::getFile(artifact);
    ASSERT_TRUE(after);
    EXPECT_EQ((*before)->getBuffer(), (*after)->getBuffer());
  }
  for (llvm::StringRef type : {"tensor<1x1024x1xi32>", "tensor<1x?x1xf32>"}) {
    writeSource(
        erfSource(type, "mhlo.version = 1 : i64, mhlo.attributes = {}"));
    ASSERT_FALSE(HasFatalFailure());
    std::string diagnostics;
    llvm::raw_string_ostream stream(diagnostics);
    EXPECT_TRUE(
        mlir::failed(wafer::frontend::deserializeStableHLOProgramDirectory(
            root, context, stream)));
  }
  writeSource(erfSource("tensor<1x1031x1xf32>",
                        "mhlo.version = 1 : i64, mhlo.attributes = {}",
                        "opaque"));
  ASSERT_FALSE(HasFatalFailure());
  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  EXPECT_TRUE(
      mlir::failed(wafer::frontend::deserializeStableHLOProgramDirectory(
          root, context, stream)));

  writeSource(R"mlir(module {
    func.func @main(%arg: tensor<1x1024x1xf32>) -> tensor<1x1025x1xf32> {
      %r = stablehlo.custom_call @mhlo.erf(%arg)
        {mhlo.version = 1 : i64, mhlo.attributes = {}} :
        (tensor<1x1024x1xf32>) -> tensor<1x1025x1xf32>
      return %r : tensor<1x1025x1xf32>
    }
  })mlir");
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_TRUE(
      mlir::failed(wafer::frontend::deserializeStableHLOProgramDirectory(
          root, context, stream)));
}

TEST_F(SourceMathIngestionTest,
       LeavesOrdinarySourceUnchangedAndVerifierStrict) {
  writeSource(R"mlir(module {
    func.func @main(%arg: tensor<1x1031x16xf16>) -> tensor<1x1031x16xf16> {
      %a = stablehlo.add %arg, %arg : tensor<1x1031x16xf16>
      %b = stablehlo.multiply %a, %arg : tensor<1x1031x16xf16>
      return %b : tensor<1x1031x16xf16>
    }
  })mlir");
  ASSERT_FALSE(HasFatalFailure());
  auto bytes = llvm::MemoryBuffer::getFile(artifact);
  ASSERT_TRUE(bytes);
  auto original = mlir::stablehlo::deserializePortableArtifact(
      (*bytes)->getBuffer(), &context);
  ASSERT_TRUE(original);
  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  auto result = wafer::frontend::deserializeStableHLOProgramDirectory(
      root, context, stream);
  ASSERT_TRUE(mlir::succeeded(result)) << diagnostics;
  std::string before, after;
  llvm::raw_string_ostream beforeStream(before), afterStream(after);
  original->print(beforeStream);
  (*result)->print(afterStream);
  EXPECT_EQ(before, after);

  auto opaque = mlir::parseSourceString<mlir::ModuleOp>(
      erfSource("tensor<1x1024x1xf16>",
                "mhlo.version = 1 : i64, mhlo.attributes = {}"),
      &context);
  ASSERT_TRUE(opaque);
  EXPECT_TRUE(mlir::failed(
      wafer::frontend::verifyStableHLOSourceModule(*opaque, stream)));
}

TEST(ProgramIngestionTest, AcceptsOnePublicEntryWithPrivatePureHelper) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func private @helper(%arg: tensor<2xf32>) -> tensor<2xf32> {
        return %arg : tensor<2xf32>
      }
      func.func @forward(%arg: tensor<2xf32>) -> tensor<2xf32> {
        %result = func.call @helper(%arg) : (tensor<2xf32>) -> tensor<2xf32>
        return %result : tensor<2xf32>
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  EXPECT_TRUE(mlir::succeeded(
      wafer::frontend::verifyStableHLOSourceModule(*module, stream)))
      << diagnostics;
}

TEST(ProgramIngestionTest, RejectsRecursivePrivateHelpers) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func private @first() {
        func.call @second() : () -> ()
        return
      }
      func.func private @second() {
        func.call @first() : () -> ()
        return
      }
      func.func @forward() {
        func.call @first() : () -> ()
        return
      }
    }
  )mlir",
                                                          &context);
  ASSERT_TRUE(module);
  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  EXPECT_TRUE(mlir::failed(
      wafer::frontend::verifyStableHLOSourceModule(*module, stream)));
}

} // namespace
