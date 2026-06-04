//===- AbiOps.td - Wafer Abi operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_ABIOPS_TD
#define WAFER_IR_OPS_ABIOPS_TD

def Wafer_AbiRdmaOp : Wafer_Op<"abi.rdma", [
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_DDRResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "C ABI DDR-to-SPM DMA issue";
  let arguments = (ins
    Wafer_AbiWaitPolicyAttr:$wait_policy,
    AnyType:$source,
    I64Attr:$bytes
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$wait_policy $source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_AbiWdmaOp : Wafer_Op<"abi.wdma", [
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_DDRResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "C ABI SPM-to-DDR DMA issue";
  let arguments = (ins
    Wafer_AbiWaitPolicyAttr:$wait_policy,
    AnyType:$source,
    AnyType:$dest,
    I64Attr:$bytes
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$wait_policy $source `,` $dest attr-dict `:` type($source) `->` type($dest)";
}

def Wafer_AbiGemmOp : Wafer_Op<"abi.gemm", [
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "C ABI GEMM issue";
  let arguments = (ins
    Wafer_AbiWaitPolicyAttr:$wait_policy,
    AnyType:$lhs,
    AnyType:$rhs,
    I64Attr:$m,
    I64Attr:$k,
    I64Attr:$n
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$wait_policy $lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def Wafer_AbiElementwiseOp : Wafer_Op<"abi.elementwise", [
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "C ABI elementwise issue";
  let arguments = (ins
    Wafer_AbiWaitPolicyAttr:$wait_policy,
    Wafer_ComputeElementwiseKindAttr:$kind,
    Variadic<AnyType>:$inputs
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$wait_policy $kind $inputs attr-dict `:` functional-type(operands, results)";
}

def Wafer_AbiReduceOp : Wafer_Op<"abi.reduce", [
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "C ABI local reduce issue";
  let description = [{
    `wafer.abi.reduce` is the wrapper-facing local reduce issue point. It
    keeps reduce dimensions and the scalar init value explicit so the C ABI
    layer can validate units and preserve reduction semantics.
  }];

  let arguments = (ins
    Wafer_AbiWaitPolicyAttr:$wait_policy,
    Wafer_ComputeReduceKindAttr:$kind,
    AnyType:$input
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$wait_policy $kind $input attr-dict `:` functional-type(operands, results)";
}

class Wafer_AbiDteP2POp<string mnemonic> : Wafer_Op<mnemonic, [
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let arguments = (ins
    AnyType:$buffer,
    I64Attr:$peer,
    I64Attr:$bytes,
    I64Attr:$fsm_id,
    I64Attr:$packet_id,
    I64Attr:$stream_id
  );
  let results = (outs AnyType:$token);
  let hasVerifier = 1;
  let assemblyFormat =
      "$buffer attr-dict `:` type($buffer) `->` type($token)";
}

def Wafer_AbiDteSendOp : Wafer_AbiDteP2POp<"abi.dte_send"> {
  let summary = "C ABI Direct DTE unicast send issue";
}

def Wafer_AbiDteRecvOp : Wafer_AbiDteP2POp<"abi.dte_recv"> {
  let summary = "C ABI Direct DTE unicast receive issue";
}

def Wafer_AbiDteWaitOp : Wafer_Op<"abi.dte_wait", [
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_CommunicationResource>]>
  ]> {
  let summary = "C ABI Direct DTE completion wait";
  let arguments = (ins Variadic<AnyType>:$tokens);
  let hasVerifier = 1;
  let assemblyFormat = "$tokens attr-dict `:` type($tokens)";
}

#endif // WAFER_IR_OPS_ABIOPS_TD
