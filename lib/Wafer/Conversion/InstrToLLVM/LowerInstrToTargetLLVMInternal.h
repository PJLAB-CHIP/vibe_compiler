//===- LowerInstrToTargetLLVMInternal.h - Private target LLVM lowering ----===//

#ifndef WAFER_TRANSFORMS_TARGET_LOWERINSTRTOTARGETLLVMINTERNAL_H
#define WAFER_TRANSFORMS_TARGET_LOWERINSTRTOTARGETLLVMINTERNAL_H

#include "Wafer/Analysis/Module/DirectCallGraphAnalysis.h"
#include "Wafer/IR/Topology/TargetTopology.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/TopologyIds.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Conversion/InstrToLLVM/InstrToLLVM.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace wafer::target_llvm_detail {

struct AddressValue {
  mlir::Value dynamicBase;
  int64_t staticOffset = 0;
};

struct CalleeSignature {
  mlir::LLVM::LLVMFunctionType type;
};

struct DirectDTEEndpointDomain {
  CardId cardId = CardId(-1);
  TileId tileId = TileId(-1);
  llvm::SmallVector<TileId, 16> availableTileIds;
};

struct DynamicSubviewAddressPlan {
  int64_t staticByteOffset = 0;
  llvm::SmallVector<int64_t, 4> dynamicByteStrides;
};

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result);
bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result);
bool isWaferInstruction(mlir::Operation *op);

mlir::Value resolveTileRegionBoundaryValue(mlir::Value value);
mlir::Value getRootViewSource(mlir::Value value);
mlir::Value resolveReturnedMemRefRoot(mlir::Value value);
mlir::FailureOr<int64_t> getStaticElementCount(mlir::Operation *op,
                                               mlir::MemRefType type,
                                               llvm::StringRef role);
mlir::FailureOr<int64_t> getPhysicalTraversalElementCount(mlir::Operation *op,
                                                          mlir::MemRefType type,
                                                          llvm::StringRef role);
mlir::FailureOr<int64_t> getStaticViewOffsetBytes(mlir::Operation *op,
                                                  mlir::MemRefType viewType,
                                                  llvm::StringRef role);
mlir::FailureOr<DynamicSubviewAddressPlan>
analyzeDynamicTensorSubviewAddressing(mlir::memref::SubViewOp subviewOp);
mlir::FailureOr<int64_t> getStaticUInt32SPMAddress(mlir::Operation *op,
                                                   mlir::Value value,
                                                   llvm::StringRef role);
mlir::FailureOr<int64_t> getStaticSPMAddress(mlir::Operation *op,
                                             mlir::Value value,
                                             llvm::StringRef role);
int64_t getIntegerAttrValue(mlir::IntegerAttr attr);
int64_t getOptionalIntegerAttrValue(mlir::IntegerAttr attr, int64_t fallback);
mlir::FailureOr<int64_t> getConstantScalarValue(mlir::Operation *op,
                                                mlir::Value value);
bool isTargetRelationElementwiseKind(InstrElementwiseKind kind);
mlir::FailureOr<int64_t>
getDataFormatCode(mlir::Operation *op, mlir::Value value, llvm::StringRef role);
mlir::LogicalResult verifyTargetInstructionFormats(mlir::ModuleOp moduleOp);
mlir::LogicalResult verifyTargetSubviewAddresses(mlir::ModuleOp moduleOp);
mlir::FailureOr<DirectDTEEndpointDomain> resolveDirectDTEEndpointDomain(
    const TargetTopology &topology, mlir::ModuleOp diagnosticModule,
    CardId cardId, TileId tileId);

struct FunctionLowering {
  mlir::OpBuilder &builder;
  mlir::MLIRContext *context;
  mlir::Type i64Type;
  mlir::Type i32Type;
  llvm::DenseMap<mlir::Value, mlir::Value> convertedValues;

  FunctionLowering(mlir::MLIRContext *context, mlir::OpBuilder &builder);

  mlir::Value constantI64(mlir::Location loc, int64_t value);
  mlir::Value constantI32(mlir::Location loc, int64_t value);
  void appendI32(mlir::Location loc, llvm::SmallVectorImpl<mlir::Value> &out,
                 int64_t value);
  void appendArrayI32(mlir::Location loc,
                      llvm::SmallVectorImpl<mlir::Value> &out,
                      llvm::ArrayRef<int64_t> values);
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
  getNHWCShape(mlir::Operation *op, mlir::MemRefType type,
               llvm::StringRef role);
  mlir::FailureOr<AddressValue>
  addStaticOffset(mlir::Operation *op, AddressValue address, int64_t offset);
  mlir::Value materializeAddress(mlir::Location loc, AddressValue address);
  mlir::FailureOr<AddressValue>
  resolveAddress(mlir::Operation *op, mlir::Value value, llvm::StringRef role);
  mlir::FailureOr<mlir::Value> materializeAddress(mlir::Operation *op,
                                                  mlir::Value value,
                                                  llvm::StringRef role);
  void verifyCallSignature(const TargetCallDescriptor &descriptor,
                           mlir::ValueRange args,
                           TargetCallResultType result) const;
  void emitCall(mlir::Location loc, const TargetCallDescriptor &descriptor,
                mlir::ValueRange args);
  void emitNCCCall(mlir::Location loc, const TargetCallDescriptor &descriptor,
                   mlir::ValueRange args, NCCWorker worker);
  mlir::Value emitI64Call(mlir::Location loc,
                          const TargetCallDescriptor &descriptor,
                          mlir::ValueRange args);

  mlir::FailureOr<mlir::Value>
  lowerDTESend(InstrDTESendOp op, const DirectDTEEndpointDomain &domain);
  mlir::FailureOr<mlir::Value>
  lowerDTERecv(InstrDTERecvOp op, const DirectDTEEndpointDomain &domain);
  mlir::LogicalResult lowerDTEWait(InstrDTEWaitOp op,
                                   llvm::ArrayRef<mlir::Value> events);
  mlir::LogicalResult lowerRDMA(InstrRDMAOp op);
  mlir::LogicalResult lowerWDMA(InstrWDMAOp op);
  mlir::LogicalResult lowerGatherScatter(InstrGatherScatterOp op);
  mlir::LogicalResult lowerFill(InstrFillOp op);
  mlir::LogicalResult lowerElementwise(InstrElementwiseOp op);
  mlir::LogicalResult lowerBit2Fp(InstrBit2FpOp op);
  mlir::LogicalResult lowerMaskMove(InstrMaskMoveOp op);
  mlir::LogicalResult lowerReduce(InstrReduceOp op);
  mlir::LogicalResult lowerConvert(InstrConvertOp op);
  mlir::LogicalResult lowerGemm(InstrGemmOp op);
  mlir::LogicalResult lowerConv(InstrConvOp op);
  mlir::LogicalResult lowerPool(InstrPoolOp op);
  mlir::LogicalResult lowerUnpool(InstrUnpoolOp op);
  bool isTransformLikeTDMA(InstrDataMoveKind kind);
  mlir::LogicalResult lowerTDMADataMove(InstrTDMADataMoveOp op);
  mlir::LogicalResult lowerPeripheral(InstrPeripheralOp op);
  mlir::LogicalResult lowerNCCJoin(SyncNCCJoinOp op);
  mlir::LogicalResult lowerInstruction(mlir::Operation *op);
};

using AliasSummary = llvm::SmallVector<unsigned, 4>;

mlir::LogicalResult flattenTileRegions(mlir::ModuleOp moduleOp);
mlir::LogicalResult
validateDirectCallsForTarget(const analysis::DirectCallGraphAnalysis &graph,
                             int64_t defaultDDRArenaArgumentIndex,
                             int64_t transportStatusArgumentIndex,
                             int64_t profileRecordArgumentIndex);
mlir::FailureOr<mlir::func::FuncOp>
findUniqueRootFunction(const analysis::DirectCallGraphAnalysis &graph);
mlir::LogicalResult analyzeDDRAliasContracts(
    mlir::ModuleOp moduleOp, const analysis::DirectCallGraphAnalysis &graph,
    llvm::DenseMap<mlir::Operation *, AliasSummary> &summaries);
void dropRootAliasResults(mlir::ModuleOp moduleOp,
                          const analysis::DirectCallGraphAnalysis &graph);
void eraseTargetMetadata(mlir::ModuleOp moduleOp);
mlir::LogicalResult lowerSCFToControlFlow(mlir::ModuleOp moduleOp);

void populateTargetLLVMStructureConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    int64_t defaultDDRArenaArgumentIndex);
void populateTargetInstructionConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    const DirectDTEEndpointDomain *dteDomain);

mlir::LogicalResult injectDirectDTEStatusLifecycle(
    mlir::ModuleOp moduleOp, llvm::StringRef entrySymbol,
    int64_t statusArgumentIndex, int64_t participantCount,
    TargetCallBuiltin beginBuiltin,
    llvm::StringMap<CalleeSignature> &usedCallees);

mlir::LogicalResult
lowerModuleInPlace(mlir::ModuleOp moduleOp, bool transportPreparedBeforeEntry,
                   int64_t defaultDDRArenaArgumentIndex, int64_t cardId,
                   int64_t tileId, int64_t transportStatusArgumentIndex,
                   int64_t profileRecordArgumentIndex);

} // namespace wafer::target_llvm_detail

#endif // WAFER_TRANSFORMS_TARGET_LOWERINSTRTOTARGETLLVMINTERNAL_H
