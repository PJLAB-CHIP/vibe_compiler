//===- AccessReuseInputs.h - Real-size read reuse mechanism inputs ------===//
#ifndef WAFER_TESTSUPPORT_TRANSFORMS_ACCESSREUSEINPUTS_H
#define WAFER_TESTSUPPORT_TRANSFORMS_ACCESSREUSEINPUTS_H
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>
namespace wafer::testing {
inline std::string makeAccessReuseInput(int64_t count, int64_t extent,
                                        llvm::StringRef dtype,
                                        bool sliding = false) {
  std::string text;
  llvm::raw_string_ostream out(text);
  auto tensor = [&](int64_t batches, int64_t rows, llvm::StringRef space) {
    return "memref<" + std::to_string(batches) + "x" + std::to_string(rows) +
           "x64x" + dtype.str() + ", #wafer.memory<" + space.str() +
           ", tensor>>";
  };
  const int64_t inputRows = sliding ? extent + 7 : extent;
  const int64_t outputBatches = count * (sliding ? extent : 16);
  const int64_t outputRows = sliding ? 8 : extent;
  auto input = tensor(1, inputRows, "ddr");
  auto output = tensor(outputBatches, outputRows, "ddr");
  out << "module { wafer.execution.mesh @mesh {axes = [\"card\"], shape = "
         "array<i64: 1>} "
         "wafer.target.topology @topology {card_grid = array<i64: 1, 1>, "
         "card_interconnect = \"mesh\", tile_grid = array<i64: 4, 4>, "
         "unavailable_tiles = array<i64>} ";
  for (int64_t tile = 0; tile < count; ++tile) {
    out << "wafer.tile.module card_id = 0 tile_id = " << tile
        << " { func.func @entry(%src: " << input
        << " {wafer.program_argument = #wafer.program_argument<0>}) -> "
        << output << " { %dst = memref.alloc() : " << output
        << " wafer.tile.region(%src, %dst : " << input << ", " << output
        << ") -> () { ^bb0(%input: " << input << ", %output: " << output
        << "): "
        << "%c0 = arith.constant 0 : index %c1 = arith.constant 1 : index "
        << "%c4 = arith.constant 4 : index %step = arith.constant 256 : index "
        << "%tile = arith.constant " << tile * (sliding ? extent : 16)
        << " : index ";
    auto emit = [&](int64_t rows, llvm::StringRef sourceRow,
                    llvm::StringRef batch, llvm::StringRef outputRow,
                    llvm::StringRef tag) {
      auto local = tensor(1, rows, "spm");
      auto view = [&](int64_t stride) {
        return "memref<1x" + std::to_string(rows) + "x64x" + dtype.str() +
               ", strided<[" + std::to_string(stride * 64) +
               ", 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>";
      };
      // Explicit affine zero keeps source and destination view types uniform.
      out << "%read" << tag << " = memref.subview %input[%c0, " << sourceRow
          << ", 0] [1, " << rows << ", 64] [1, 1, 1] : " << input << " to "
          << view(inputRows) << " %local" << tag
          << " = memref.alloc() : " << local << " wafer.tile.load %read" << tag
          << " into %local" << tag << " : " << view(inputRows) << " into "
          << local << " %write" << tag << " = memref.subview %output[" << batch
          << ", " << outputRow << ", 0] [1, " << rows
          << ", 64] [1, 1, 1] : " << output << " to " << view(outputRows)
          << " wafer.tile.store %local" << tag << ", %write" << tag << " : "
          << local << " -> " << view(outputRows) << " ";
    };
    if (sliding) {
      out << "%end = arith.constant " << extent << " : index "
          << "scf.for %row = %c0 to %end step %c1 { "
          << "%batch = arith.addi %row, %tile : index ";
      emit(8, "%row", "%batch", "%c0", "window");
      out << "} ";
    } else {
      out << "%end = arith.constant " << (extent / 256) * 256 << " : index "
          << "scf.for %repeat = %c0 to %c4 step %c1 { "
          << "%base = arith.muli %repeat, %c4 : index "
          << "%tilebase = arith.addi %base, %tile : index "
          << "scf.for %row = %c0 to %end step %step { "
          << "scf.for %again = %c0 to %c4 step %c1 { "
          << "%batch = arith.addi %tilebase, %again : index ";
      emit(256, "%row", "%batch", "%row", "main");
      out << "} } ";
      if (extent % 256) {
        out << "scf.for %again = %c0 to %c4 step %c1 { "
            << "%batch = arith.addi %tilebase, %again : index ";
        emit(extent % 256, "%end", "%batch", "%end", "tail");
        out << "} ";
      }
      out << "} ";
    }
    out << "wafer.tile.yield } return %dst : " << output << " } } ";
  }
  out << "}";
  return text;
}

} // namespace wafer::testing
#endif
