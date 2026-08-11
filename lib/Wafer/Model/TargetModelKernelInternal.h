//===- TargetModelKernelInternal.h - Plain kernel private API -*- C++ -*-===//
#pragma once

#include "Wafer/Model/TargetModelKernel.h"

#include "llvm/ADT/Twine.h"

#include <array>
#include <cstdint>
#include <vector>

namespace wafer::model::kernel_detail {

llvm::Error kernelError(TargetModelKernelErrorCode code,
                        const llvm::Twine &detail);

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result);

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result);

llvm::Expected<uint64_t>
getDescriptorSegmentCount(const std::array<uint32_t, 3> &iterations);

llvm::Expected<std::vector<uint8_t>>
readSnapshot(const InvocationMemoryRegistry &memory, int64_t launchSlot,
             TargetModelAddressSpace space, uint64_t address, uint64_t bytes,
             uint64_t alignment = 1);

llvm::Expected<TargetModelCommandEffect>
executeMovement(const compiler::TargetTransaction &transaction,
                const compiler::TargetStridedDMATransaction &value,
                const InvocationMemoryRegistry &memory,
                TargetModelKernelBudget budget);

llvm::Expected<TargetModelCommandEffect>
executeGatherScatter(const compiler::TargetTransaction &transaction,
                     const compiler::TargetGatherScatterTransaction &value,
                     const InvocationMemoryRegistry &memory,
                     TargetModelKernelBudget budget);

llvm::Expected<TargetModelCommandEffect>
executeElementwise(const compiler::TargetTransaction &transaction,
                   const compiler::TargetElementwiseTransaction &value,
                   const InvocationMemoryRegistry &memory,
                   TargetModelKernelBudget budget,
                   TargetModelExecutionPolicy policy);

llvm::Expected<TargetModelCommandEffect>
executeConvert(const compiler::TargetTransaction &transaction,
               const compiler::TargetConvertTransaction &value,
               const InvocationMemoryRegistry &memory,
               TargetModelKernelBudget budget,
               TargetModelExecutionPolicy policy);

llvm::Expected<TargetModelCommandEffect>
executeReduce(const compiler::TargetTransaction &transaction,
              const compiler::TargetReduceTransaction &value,
              const InvocationMemoryRegistry &memory,
              TargetModelKernelBudget budget,
              TargetModelExecutionPolicy policy);

llvm::Expected<TargetModelCommandEffect>
executeGemm(const compiler::TargetTransaction &transaction,
            const compiler::TargetGemmTransaction &value,
            const InvocationMemoryRegistry &memory,
            TargetModelKernelBudget budget, TargetModelExecutionPolicy policy);

llvm::Expected<TargetModelCommandEffect>
executeMemset(const compiler::TargetTransaction &transaction,
              const compiler::TargetMemsetTransaction &value,
              const InvocationMemoryRegistry &memory);

TargetModelControlAction
getControlAction(const compiler::TargetTransactionPayload &payload);

llvm::Error
validateControlAddresses(const compiler::TargetTransaction &transaction,
                         const InvocationAddressPlan &plan);

} // namespace wafer::model::kernel_detail
