//===- WaferPhysicalEncodingInterfaces.td - Physical encoding interface --===//

#ifndef WAFER_IR_WAFERPHYSICALENCODINGINTERFACES_TD
#define WAFER_IR_WAFERPHYSICALENCODINGINTERFACES_TD

include "mlir/IR/OpBase.td"

def WaferPhysicalEncodingAttrInterface
    : AttrInterface<"WaferPhysicalEncodingAttrInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface implemented by Wafer physical memory encodings. All answers are
    derived from the encoding attribute and the queried memref type; accepted
    offsets, planner choices, runtime handles, and movement routes are outside
    this interface. Physical layout pieces must have pairwise-disjoint logical
    domains, cover the complete valid logical domain, and map valid elements to
    non-overlapping physical bit spans inside the reported footprint.
  }];
  let methods = [
    InterfaceMethod<
      "Return the complete physical storage footprint in bytes.",
      "::mlir::FailureOr<int64_t>", "getPhysicalFootprintBytes",
      (ins "::mlir::MemRefType":$type)>,
    InterfaceMethod<
      "Return the encoding's minimum natural byte alignment.",
      "::mlir::FailureOr<int64_t>", "getMinimumAlignmentBytes",
      (ins "::mlir::MemRefType":$type)>,
    InterfaceMethod<
      "Return the number of elements in the logical valid domain.",
      "::mlir::FailureOr<int64_t>", "getValidElementCount",
      (ins "::mlir::MemRefType":$type)>,
    InterfaceMethod<
      "Return the number of physical elements outside the logical valid domain.",
      "::mlir::FailureOr<int64_t>", "getPaddingElementCount",
      (ins "::mlir::MemRefType":$type)>,
    InterfaceMethod<
      "Return the physical bit width of one valid logical element.",
      "::mlir::FailureOr<int64_t>", "getPhysicalElementBitWidth",
      (ins "::mlir::MemRefType":$type)>,
    InterfaceMethod<
      "Return exact encoding-owned logical-to-physical bit-offset pieces.",
      "::mlir::FailureOr<::llvm::SmallVector<::wafer::WaferPhysicalLayoutPiece, 2>>",
      "getPhysicalLayoutPieces",
      (ins "::mlir::MemRefType":$type)>,
    InterfaceMethod<
      "Return the physical bit segment occupied by one valid logical element.",
      "::mlir::FailureOr<::wafer::WaferPhysicalElementSpan>",
      "getPhysicalElementSpan",
      (ins "::mlir::MemRefType":$type,
           "::llvm::ArrayRef<int64_t>":$logicalIndices)>
  ];
}

#endif // WAFER_IR_WAFERPHYSICALENCODINGINTERFACES_TD
