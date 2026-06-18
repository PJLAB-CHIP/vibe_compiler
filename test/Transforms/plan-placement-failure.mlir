// RUN: not wafer-opt --wafer-plan-placement='logical-rank-count=3 card-y-count=1 card-x-count=1 tile-y-count=1 tile-x-count=2 bad-tile-ids=1' %s 2>&1 | FileCheck %s
// RUN: not wafer-opt --wafer-plan-placement='bad-tile-ids=1,,2' %s 2>&1 | FileCheck --check-prefix=BAD-IDS %s

module {
}

// CHECK: placement_failure: logical rank count 3 exceeds available good tile count 1
// BAD-IDS: invalid empty entry in bad-tile-ids
