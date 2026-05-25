//===- wafer-import-model.cpp - Wafer importer placeholder ---------------===//

#include "llvm/Support/raw_ostream.h"

#include <string>

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    llvm::outs() << "wafer-import-model\n"
                 << "  importer dependencies are disabled in this build\n";
    return 0;
  }

  llvm::errs() << "wafer-import-model: importer dependencies are disabled in "
                  "this build\n";
  return 1;
}
