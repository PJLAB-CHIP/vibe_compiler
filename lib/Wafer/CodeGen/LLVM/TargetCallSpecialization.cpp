//===- TargetCallSpecialization.cpp - Fixed CRT arguments ----------------===//

#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace wafer::compiler::detail {
namespace {

struct Specialization {
  const TargetCallDescriptor *descriptor;
  std::vector<std::optional<uint32_t>> constants;
  std::vector<llvm::CallInst *> calls;
};

llvm::StringRef cType(TargetCallScalarType type) {
  return type == TargetCallScalarType::I32 ? "uint32_t" : "uint64_t";
}

llvm::Error invalidCall(llvm::StringRef symbol) {
  return llvm::createStringError(
      llvm::errc::invalid_argument,
      "fixed CRT argument preparation requires an ordinary void C call with "
      "the registered signature: %s",
      symbol.str().c_str());
}

} // namespace

llvm::Expected<std::string> specializeTargetCalls(llvm::Module &module) {
  if (llvm::verifyModule(module))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "invalid LLVM before CRT specialization");

  // All analysis and potentially failing call checks precede mutation. The
  // semantic registry is authoritative; no instruction family is inferred
  // from a symbol prefix or an address value.
  std::map<std::string, Specialization> groups;
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
        if (!callee)
          continue;
        const auto *descriptor = findTargetCallDescriptor(callee->getName());
        if (!descriptor || !descriptor->issueDomain ||
            !descriptor->issueDomain->nccWorkerArgument)
          continue;
        auto *ordinary = llvm::dyn_cast<llvm::CallInst>(call);
        if (!ordinary || ordinary->isMustTailCall() ||
            ordinary->hasOperandBundles() || !callee->isDeclaration() ||
            call->getCallingConv() != llvm::CallingConv::C ||
            callee->getCallingConv() != llvm::CallingConv::C ||
            !call->getType()->isVoidTy() ||
            call->getFunctionType()->isVarArg() ||
            call->arg_size() != descriptor->arguments.size())
          return invalidCall(descriptor->symbol);
        std::vector<std::optional<uint32_t>> constants;
        std::string key = descriptor->symbol;
        bool hasConstant = false;
        for (auto [index, type] : llvm::enumerate(descriptor->arguments)) {
          llvm::Value *argument = call->getArgOperand(index);
          unsigned width = type == TargetCallScalarType::I32 ? 32 : 64;
          if (!argument->getType()->isIntegerTy(width))
            return invalidCall(descriptor->symbol);
          std::optional<uint32_t> constant;
          if (width == 32)
            if (auto *value = llvm::dyn_cast<llvm::ConstantInt>(argument)) {
              constant = value->getZExtValue();
              hasConstant = true;
            }
          constants.push_back(constant);
          key += constant ? ":" + std::to_string(*constant) : ":dynamic";
        }
        if (hasConstant) {
          auto [entry, inserted] = groups.try_emplace(
              key, Specialization{descriptor, std::move(constants), {}});
          entry->second.calls.push_back(ordinary);
        }
      }

  for (size_t index = 0; index < groups.size(); ++index)
    if (module.getNamedValue("__wafer_crt_fixed_" + std::to_string(index)))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "generated CRT symbol collision");

  std::string source;
  llvm::raw_string_ostream output(source);
  output << "/* Generated from actual LLVM calls; compiled with the unique "
            "CRT implementation. */\n#include \"wafer_tx81_crt.h\"\n";
  std::set<std::string> declared;
  for (const auto &[key, group] : groups) {
    if (!declared.insert(group.descriptor->symbol).second)
      continue;
    output << "extern inline __attribute__((always_inline)) void "
           << group.descriptor->symbol << '(';
    for (auto [index, type] : llvm::enumerate(group.descriptor->arguments))
      output << (index ? ", " : "") << cType(type);
    output << ");\n";
  }
  // Device-link supplies the configured source path. This is the original CRT
  // translation unit, not a second implementation of SDK packet encoding.
  output << "#include WAFER_CRT_SOURCE\n";

  size_t ordinal = 0;
  for (const auto &[key, group] : groups) {
    const std::string name = "__wafer_crt_fixed_" + std::to_string(ordinal++);
    llvm::SmallVector<llvm::Type *> types;
    output << "__attribute__((visibility(\"hidden\"))) void " << name << '(';
    for (auto [index, type] : llvm::enumerate(group.descriptor->arguments)) {
      if (group.constants[index])
        continue;
      output << (types.empty() ? "" : ", ") << cType(type) << " a" << index;
      types.push_back(group.calls.front()->getArgOperand(index)->getType());
    }
    if (types.empty())
      output << "void";
    output << ") {\n  " << group.descriptor->symbol << '(';
    for (size_t index = 0; index < group.constants.size(); ++index) {
      output << (index ? ", " : "");
      if (group.constants[index])
        output << "UINT32_C(" << *group.constants[index] << ')';
      else
        output << 'a' << index;
    }
    output << ");\n}\n";

    auto *type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(module.getContext()), types, false);
    auto *specialized = llvm::Function::Create(
        type, llvm::GlobalValue::ExternalLinkage, name, module);
    specialized->setVisibility(llvm::GlobalValue::HiddenVisibility);
    for (llvm::CallInst *call : group.calls) {
      llvm::SmallVector<llvm::Value *> arguments;
      llvm::SmallVector<llvm::AttributeSet> attributes;
      for (size_t index = 0; index < group.constants.size(); ++index)
        if (!group.constants[index]) {
          arguments.push_back(call->getArgOperand(index));
          attributes.push_back(call->getAttributes().getParamAttrs(index));
        }
      llvm::IRBuilder<> builder(call);
      auto *replacement = builder.CreateCall(specialized, arguments);
      replacement->setAttributes(llvm::AttributeList::get(
          module.getContext(), call->getAttributes().getFnAttrs(),
          call->getAttributes().getRetAttrs(), attributes));
      replacement->setDebugLoc(call->getDebugLoc());
      call->eraseFromParent();
    }
  }
  if (llvm::verifyModule(module))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "invalid LLVM after CRT specialization");
  return source;
}

} // namespace wafer::compiler::detail
