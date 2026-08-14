#include "Wafer/Runtime/PackageManifest.h"

int main() {
  return wafer::runtime::stringifyPackageResourceRole(
             wafer::runtime::PackageResourceRole::UserInput) == "user_input"
             ? 0
             : 1;
}
