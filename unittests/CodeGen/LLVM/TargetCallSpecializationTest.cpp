#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <set>
#include <string>
#include <vector>

namespace {

using wafer::TargetCallScalarType;

TEST(TargetCallSpecializationTest, AllNCCFamiliesPreserveArgumentsInCompiledC) {
  // Bounded ABI oracle, not a numerical instruction test. Actual rank-three
  // main/tail workloads separately traverse the production package pipeline.
  llvm::LLVMContext context;
  llvm::Module module("fixed-arguments", context);
  llvm::IRBuilder<> builder(context);
  auto *entryType = llvm::FunctionType::get(
      builder.getVoidTy(), {builder.getInt64Ty(), builder.getInt32Ty()}, false);
  auto *entry = llvm::Function::Create(
      entryType, llvm::GlobalValue::ExternalLinkage, "entry", module);
  builder.SetInsertPoint(llvm::BasicBlock::Create(context, "body", entry));
  std::string stub, main;
  llvm::raw_string_ostream stubOut(stub), mainOut(main);
  stubOut << "#include <assert.h>\n#include <stdint.h>\n"
             "uint64_t expected[40];\nunsigned calls, kind;\n";
  struct Invocation {
    unsigned kind;
    std::vector<uint64_t> values;
    std::vector<bool> dynamic;
  };
  std::vector<Invocation> invocations;
  unsigned kind = 0;
  size_t ordinaryFamilies = 0;
  for (const auto &descriptor : wafer::getTargetCallDescriptors()) {
    if (!descriptor.issueDomain || !descriptor.issueDomain->nccWorkerArgument)
      continue;
    ++ordinaryFamilies;
    llvm::SmallVector<llvm::Type *> types;
    stubOut << "void " << descriptor.symbol << '(';
    for (size_t index = 0; index < descriptor.arguments.size(); ++index) {
      bool wide = descriptor.arguments[index] == TargetCallScalarType::I64;
      types.push_back(wide ? builder.getInt64Ty() : builder.getInt32Ty());
      stubOut << (index ? ", " : "") << (wide ? "uint64_t" : "uint32_t") << " a"
              << index;
    }
    stubOut << ") { assert(kind == " << kind << "); ++calls;\n";
    for (size_t index = 0; index < types.size(); ++index)
      stubOut << "assert(a" << index << " == expected[" << index << "]);\n";
    stubOut << "}\n";
    auto *type = llvm::FunctionType::get(builder.getVoidTy(), types, false);
    auto callee = module.getOrInsertFunction(descriptor.symbol, type);
    for (uint32_t extent : {1024U, 1025U, 1031U})
      for (bool partial : {false, true}) {
        llvm::SmallVector<llvm::Value *> arguments;
        Invocation invocation{kind, {}, {}};
        bool madeDynamic = false;
        for (size_t index = 0; index < types.size(); ++index) {
          bool wide = types[index]->isIntegerTy(64);
          bool dynamic = wide || (partial && !madeDynamic);
          if (!wide && dynamic)
            madeDynamic = true;
          uint64_t value = wide      ? UINT64_C(0x123456789abc)
                           : dynamic ? UINT32_C(0xfedcba98)
                                     : extent + index;
          if (index == *descriptor.issueDomain->nccWorkerArgument) {
            value = extent == 1024 ? 0 : extent == 1025 ? 1 : 2;
            dynamic = false;
          }
          llvm::Value *argument =
              dynamic ? static_cast<llvm::Value *>(entry->getArg(wide ? 0 : 1))
                      : builder.getInt32(value);
          arguments.push_back(argument);
          invocation.values.push_back(value);
          invocation.dynamic.push_back(dynamic);
        }
        // Duplicate calls must share code, while runtime operands are passed
        // on every invocation. This is independent of a particular operator.
        builder.CreateCall(callee, arguments);
        builder.CreateCall(callee, arguments);
        invocations.push_back(invocation);
        invocations.push_back(invocation);
      }
    ++kind;
  }
  builder.CreateRetVoid();
  auto source = wafer::compiler::detail::specializeTargetCalls(module);
  ASSERT_TRUE(static_cast<bool>(source))
      << (source ? "" : llvm::toString(source.takeError()));
  ASSERT_FALSE(llvm::verifyModule(module));
  std::set<llvm::Function *> callees;
  mainOut << "int main(void) {\n";
  size_t index = 0;
  for (llvm::Instruction &instruction : entry->getEntryBlock()) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
    if (!call)
      continue;
    ASSERT_LT(index, invocations.size());
    const auto &invocation = invocations[index++];
    callees.insert(call->getCalledFunction());
    mainOut << "kind = " << invocation.kind << ";\n";
    unsigned dynamicIndex = 0;
    for (size_t arg = 0; arg < invocation.values.size(); ++arg) {
      mainOut << "expected[" << arg << "] = UINT64_C(" << invocation.values[arg]
              << ");\n";
      if (invocation.dynamic[arg]) {
        bool wide = invocation.values[arg] == UINT64_C(0x123456789abc);
        EXPECT_EQ(call->getArgOperand(dynamicIndex++),
                  entry->getArg(wide ? 0 : 1));
      }
    }
    EXPECT_EQ(call->arg_size(), dynamicIndex);
    mainOut << call->getCalledFunction()->getName() << '(';
    bool first = true;
    for (size_t arg = 0; arg < invocation.values.size(); ++arg)
      if (invocation.dynamic[arg]) {
        mainOut << (first ? "" : ", ") << "UINT64_C(" << invocation.values[arg]
                << ')';
        first = false;
      }
    mainOut << ");\n";
  }
  EXPECT_EQ(index, invocations.size());
  EXPECT_EQ(callees.size(), ordinaryFamilies * 6);
  mainOut << "assert(calls == " << index << "); return 0; }\n";

  llvm::SmallString<256> directory;
  ASSERT_FALSE(
      llvm::sys::fs::createUniqueDirectory("wafer-crt-arguments", directory));
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(directory); });
  auto write = [&](llvm::StringRef name, llvm::StringRef contents) {
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, name);
    std::error_code error;
    llvm::raw_fd_ostream output(path, error);
    EXPECT_FALSE(error);
    output << contents;
    return path.str().str();
  };
  write("wafer_tx81_crt.c", stub);
  std::string cFile = write("oracle.c", *source + main);
  std::string binary = (directory + "/oracle").str();
  std::string includeRoot = "-I" WAFER_TEST_WAFER_INCLUDE_DIR;
  std::string includeCRT = "-I" WAFER_TEST_WAFER_CRT_INCLUDE_DIR;
  llvm::SmallVector<llvm::StringRef> args = {
      WAFER_TEST_LLVM_CLANGXX,
      "-DWAFER_CRT_SOURCE=\"wafer_tx81_crt.c\"",
      "-x",
      "c",
      "-std=c11",
      "-O2",
      includeRoot,
      includeCRT,
      cFile,
      "-o",
      binary};
  ASSERT_EQ(llvm::sys::ExecuteAndWait(WAFER_TEST_LLVM_CLANGXX, args), 0);
  ASSERT_EQ(llvm::sys::ExecuteAndWait(binary, {binary}), 0);
}

TEST(TargetCallSpecializationTest,
     DynamicArgumentsAndSynchronizationStayIntact) {
  llvm::LLVMContext context;
  llvm::Module module("dynamic-arguments", context);
  llvm::IRBuilder<> builder(context);
  auto *entry = llvm::Function::Create(
      llvm::FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty()},
                              false),
      llvm::GlobalValue::ExternalLinkage, "entry", module);
  builder.SetInsertPoint(llvm::BasicBlock::Create(context, "body", entry));
  const auto &descriptor =
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::GatherScatter);
  llvm::SmallVector<llvm::Type *> types;
  llvm::SmallVector<llvm::Value *> arguments;
  for (auto type : descriptor.arguments) {
    bool wide = type == TargetCallScalarType::I64;
    types.push_back(wide ? builder.getInt64Ty() : builder.getInt32Ty());
    arguments.push_back(wide
                            ? static_cast<llvm::Value *>(builder.getInt64(4096))
                            : entry->getArg(0));
  }
  auto callee = module.getOrInsertFunction(
      descriptor.symbol,
      llvm::FunctionType::get(builder.getVoidTy(), types, false));
  auto *dynamic = builder.CreateCall(callee, arguments);
  arguments[2] = builder.getInt32(1024);
  builder.CreateCall(callee, arguments);
  arguments[0] = builder.getInt64(8192);
  builder.CreateCall(callee, arguments);
  auto joinCallee = module.getOrInsertFunction(
      "wafer_tx81_ncc_join",
      llvm::FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty()},
                              false));
  auto *join = builder.CreateCall(joinCallee, builder.getInt32(3));
  builder.CreateRetVoid();
  auto result = wafer::compiler::detail::specializeTargetCalls(module);
  ASSERT_TRUE(static_cast<bool>(result))
      << (result ? "" : llvm::toString(result.takeError()));
  EXPECT_EQ(dynamic->getCalledFunction()->getName(), descriptor.symbol);
  EXPECT_EQ(join->getCalledFunction()->getName(), "wafer_tx81_ncc_join");
  auto *first = llvm::cast<llvm::CallInst>(dynamic->getNextNode());
  auto *second = llvm::cast<llvm::CallInst>(first->getNextNode());
  EXPECT_EQ(first->getCalledFunction(), second->getCalledFunction());
  EXPECT_EQ(
      llvm::cast<llvm::ConstantInt>(first->getArgOperand(0))->getZExtValue(),
      4096U);
  EXPECT_EQ(
      llvm::cast<llvm::ConstantInt>(second->getArgOperand(0))->getZExtValue(),
      8192U);
  EXPECT_EQ(first->arg_size(), descriptor.arguments.size() - 1);
  EXPECT_EQ(module.size(), 4U);
}

TEST(TargetCallSpecializationTest, InvalidRegistrySignatureDoesNotMutateIR) {
  llvm::LLVMContext context;
  llvm::Module module("invalid-signature", context);
  llvm::IRBuilder<> builder(context);
  auto *entry = llvm::Function::Create(
      llvm::FunctionType::get(builder.getVoidTy(), false),
      llvm::GlobalValue::ExternalLinkage, "entry", module);
  builder.SetInsertPoint(llvm::BasicBlock::Create(context, "body", entry));
  auto callee = module.getOrInsertFunction(
      "wafer_tx81_gather_scatter",
      llvm::FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty()},
                              false));
  auto *call = builder.CreateCall(callee, builder.getInt32(1025));
  builder.CreateRetVoid();
  ASSERT_FALSE(llvm::verifyModule(module));
  auto result = wafer::compiler::detail::specializeTargetCalls(module);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("registered signature"),
            std::string::npos);
  EXPECT_EQ(call->getCalledFunction()->getName(), "wafer_tx81_gather_scatter");
  EXPECT_EQ(module.size(), 2U);
}

} // namespace
