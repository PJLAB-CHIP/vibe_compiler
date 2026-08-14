//===- TargetCallIRAdapter.cpp - Instr IR to target protocol -----------===//

#include "Target/TargetCallIRAdapter.h"

#include "llvm/Support/ErrorHandling.h"

namespace wafer {
namespace {

NumericElementwiseOperation
mapElementwiseOperation(InstrElementwiseKind operation) {
#define WAFER_MAP_ELEMENTWISE(NAME)                                           \
  case InstrElementwiseKind::NAME:                                            \
    return NumericElementwiseOperation::NAME
  switch (operation) {
    WAFER_MAP_ELEMENTWISE(Abs);
    WAFER_MAP_ELEMENTWISE(Recip);
    WAFER_MAP_ELEMENTWISE(Square);
    WAFER_MAP_ELEMENTWISE(Sqrt);
    WAFER_MAP_ELEMENTWISE(Rsqrt);
    WAFER_MAP_ELEMENTWISE(Neg);
    WAFER_MAP_ELEMENTWISE(Max);
    WAFER_MAP_ELEMENTWISE(Min);
    WAFER_MAP_ELEMENTWISE(Add);
    WAFER_MAP_ELEMENTWISE(Sub);
    WAFER_MAP_ELEMENTWISE(Mul);
    WAFER_MAP_ELEMENTWISE(Div);
    WAFER_MAP_ELEMENTWISE(Eq);
    WAFER_MAP_ELEMENTWISE(Ne);
    WAFER_MAP_ELEMENTWISE(Ge);
    WAFER_MAP_ELEMENTWISE(Gt);
    WAFER_MAP_ELEMENTWISE(Le);
    WAFER_MAP_ELEMENTWISE(Lt);
    WAFER_MAP_ELEMENTWISE(LogicNot);
    WAFER_MAP_ELEMENTWISE(LogicAnd);
    WAFER_MAP_ELEMENTWISE(LogicOr);
    WAFER_MAP_ELEMENTWISE(LogicXor);
    WAFER_MAP_ELEMENTWISE(Log2);
    WAFER_MAP_ELEMENTWISE(Ln);
    WAFER_MAP_ELEMENTWISE(Pow2);
    WAFER_MAP_ELEMENTWISE(Exp);
    WAFER_MAP_ELEMENTWISE(ExpLp);
    WAFER_MAP_ELEMENTWISE(Sin);
    WAFER_MAP_ELEMENTWISE(Cos);
    WAFER_MAP_ELEMENTWISE(Tanh);
    WAFER_MAP_ELEMENTWISE(Sigmoid);
    WAFER_MAP_ELEMENTWISE(Relu);
    WAFER_MAP_ELEMENTWISE(SatRelu);
    WAFER_MAP_ELEMENTWISE(LeakyRelu);
    WAFER_MAP_ELEMENTWISE(Softplus);
  }
#undef WAFER_MAP_ELEMENTWISE
  llvm_unreachable("unknown Instr elementwise operation");
}

NumericReduceOperation mapReduceOperation(InstrReduceKind operation) {
  switch (operation) {
  case InstrReduceKind::Sum:
    return NumericReduceOperation::Sum;
  case InstrReduceKind::Max:
    return NumericReduceOperation::Max;
  case InstrReduceKind::Min:
    return NumericReduceOperation::Min;
  case InstrReduceKind::Avg:
    return NumericReduceOperation::Avg;
  }
  llvm_unreachable("unknown Instr reduction operation");
}

TargetConvolutionOperation mapConvolutionOperation(InstrConvKind operation) {
  switch (operation) {
  case InstrConvKind::Conv:
    return TargetConvolutionOperation::Convolution;
  case InstrConvKind::Depthwise:
    return TargetConvolutionOperation::DepthwiseConvolution;
  case InstrConvKind::BackwardConv:
    return TargetConvolutionOperation::BackwardConvolution;
  }
  llvm_unreachable("unknown Instr convolution operation");
}

TargetPoolingOperation mapPoolingOperation(InstrPoolKind operation) {
  switch (operation) {
  case InstrPoolKind::Avg:
    return TargetPoolingOperation::Average;
  case InstrPoolKind::Sum:
    return TargetPoolingOperation::Sum;
  case InstrPoolKind::Max:
    return TargetPoolingOperation::Maximum;
  case InstrPoolKind::IndexedMax:
    return TargetPoolingOperation::IndexedMaximum;
  case InstrPoolKind::Min:
    return TargetPoolingOperation::Minimum;
  case InstrPoolKind::IndexedMin:
    return TargetPoolingOperation::IndexedMinimum;
  }
  llvm_unreachable("unknown Instr pooling operation");
}

TargetUnpoolingOperation mapUnpoolingOperation(InstrUnpoolKind operation) {
  switch (operation) {
  case InstrUnpoolKind::Unpool:
    return TargetUnpoolingOperation::Unpool;
  case InstrUnpoolKind::Avg:
    return TargetUnpoolingOperation::Average;
  case InstrUnpoolKind::Mask:
    return TargetUnpoolingOperation::Mask;
  }
  llvm_unreachable("unknown Instr unpooling operation");
}

TargetPeripheralOperation mapPeripheralOperation(InstrPeripheralKind operation) {
  switch (operation) {
  case InstrPeripheralKind::Count:
    return TargetPeripheralOperation::Count;
  case InstrPeripheralKind::ArgMax:
    return TargetPeripheralOperation::ArgMaximum;
  case InstrPeripheralKind::ArgMin:
    return TargetPeripheralOperation::ArgMinimum;
  case InstrPeripheralKind::Factorize:
    return TargetPeripheralOperation::Factorize;
  case InstrPeripheralKind::Bilinear:
    return TargetPeripheralOperation::Bilinear;
  case InstrPeripheralKind::Lut16:
    return TargetPeripheralOperation::LookupTable16;
  case InstrPeripheralKind::Lut32:
    return TargetPeripheralOperation::LookupTable32;
  case InstrPeripheralKind::RandGen:
    return TargetPeripheralOperation::Random;
  case InstrPeripheralKind::ElemMask:
    return TargetPeripheralOperation::ElementMask;
  }
  llvm_unreachable("unknown Instr peripheral operation");
}

} // namespace

const TargetCallDescriptor &
getTargetCallDescriptor(InstrElementwiseKind operation) {
  return getTargetCallDescriptor(mapElementwiseOperation(operation));
}

const TargetCallDescriptor &getTargetCallDescriptor(InstrReduceKind operation) {
  return getTargetCallDescriptor(mapReduceOperation(operation));
}

const TargetCallDescriptor &
getTargetCallDescriptor(InstrConvertKind operation) {
  auto targetOperation = TargetConvertOperation::create(
      static_cast<uint16_t>(operation));
  if (!targetOperation) {
    std::string diagnostic = llvm::toString(targetOperation.takeError());
    llvm::report_fatal_error(llvm::StringRef(diagnostic));
  }
  return getTargetCallDescriptor(*targetOperation);
}

const TargetCallDescriptor &getTargetCallDescriptor(InstrConvKind operation) {
  return getTargetCallDescriptor(mapConvolutionOperation(operation));
}

const TargetCallDescriptor &getTargetCallDescriptor(InstrPoolKind operation) {
  return getTargetCallDescriptor(mapPoolingOperation(operation));
}

const TargetCallDescriptor &getTargetCallDescriptor(InstrUnpoolKind operation) {
  return getTargetCallDescriptor(mapUnpoolingOperation(operation));
}

const TargetCallDescriptor &
getTargetCallDescriptor(InstrPeripheralKind operation) {
  return getTargetCallDescriptor(mapPeripheralOperation(operation));
}

} // namespace wafer
