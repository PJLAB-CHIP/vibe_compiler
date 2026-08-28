#include "Wafer/Driver/ProgramData/ProgramData.h"

int main() {
  wafer::compiler::ProgramDataHandoff handoff;
  return handoff.getSourceCount() == 0 ? 0 : 1;
}
