//===- TargetModelKernelInternal.h - Plain kernel private API -*- C++ -*-===//
#pragma once

#include "Wafer/Model/Core/TargetModelKernel.h"

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
executeMovement(const compiler::TargetCommand &command,
                const target::TargetStridedDMACommand &value,
                const InvocationMemoryRegistry &memory,
                TargetModelKernelBudget budget);

llvm::Expected<TargetModelCommandEffect>
executeGatherScatter(const compiler::TargetCommand &command,
                     const target::TargetGatherScatterCommand &value,
                     const InvocationMemoryRegistry &memory,
                     TargetModelKernelBudget budget);

llvm::Expected<TargetModelCommandEffect>
executeElementwise(const compiler::TargetCommand &command,
                   const target::TargetElementwiseCommand &value,
                   const InvocationMemoryRegistry &memory,
                   TargetModelKernelBudget budget,
                   TargetModelExecutionPolicy policy);

llvm::Expected<TargetModelCommandEffect>
executeConvert(const compiler::TargetCommand &command,
               const target::TargetConvertCommand &value,
               const InvocationMemoryRegistry &memory,
               TargetModelKernelBudget budget,
               TargetModelExecutionPolicy policy);

llvm::Expected<TargetModelCommandEffect>
executeReduce(const compiler::TargetCommand &command,
              const target::TargetReduceCommand &value,
              const InvocationMemoryRegistry &memory,
              TargetModelKernelBudget budget,
              TargetModelExecutionPolicy policy);

llvm::Expected<TargetModelCommandEffect>
executeGemm(const compiler::TargetCommand &command,
            const target::TargetGemmCommand &value,
            const InvocationMemoryRegistry &memory,
            TargetModelKernelBudget budget, TargetModelExecutionPolicy policy);

llvm::Expected<TargetModelCommandEffect>
executeMemset(const compiler::TargetCommand &command,
              const target::TargetMemsetCommand &value,
              const InvocationMemoryRegistry &memory);

TargetModelControlAction
getControlAction(const target::TargetCommandPayload &payload);

llvm::Error validateControlAddresses(const compiler::TargetCommand &command,
                                     const InvocationAddressPlan &plan);

} // namespace wafer::model::kernel_detail
