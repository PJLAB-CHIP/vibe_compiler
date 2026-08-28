#include "Wafer/Planning/PhysicalDataflow/ExactDemand.h"

int main() {
  wafer::analysis::ExactIndexSet set;
  return set.getRank() == 0 ? 0 : 1;
}
