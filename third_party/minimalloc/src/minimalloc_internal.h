/*
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modified by the Wafer project: checked arithmetic and typed validation.
*/

#ifndef WAFER_THIRD_PARTY_MINIMALLOC_INTERNAL_H
#define WAFER_THIRD_PARTY_MINIMALLOC_INTERNAL_H

#include "wafer_third_party/minimalloc/minimalloc.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer_third_party {
namespace minimalloc {
namespace internal {

enum class ValidationStatus { kValid, kProvenInfeasible, kInvalid };

struct ValidationResult {
  ValidationStatus status = ValidationStatus::kInvalid;
  std::string message;
};

bool CheckedAdd(std::int64_t lhs, std::int64_t rhs, std::int64_t *result);
bool CheckedSubtract(std::int64_t lhs, std::int64_t rhs, std::int64_t *result);
bool CheckedMultiply(std::int64_t lhs, std::int64_t rhs, std::int64_t *result);
ValidationResult Validate(const Problem &problem, const SolveOptions &options);
Area BufferArea(const Buffer &buffer);
std::optional<std::int64_t> EffectiveSize(const Buffer &lower,
                                          const Buffer &upper);

} // namespace internal
} // namespace minimalloc
} // namespace wafer_third_party

#endif // WAFER_THIRD_PARTY_MINIMALLOC_INTERNAL_H
