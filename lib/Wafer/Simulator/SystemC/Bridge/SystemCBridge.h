//===- SystemCBridge.h - RTTI-isolated SystemC process bridge -*- C++ -*-===//

#ifndef WAFER_SIMULATOR_SYSTEMC_BRIDGE_SYSTEMCBRIDGE_H
#define WAFER_SIMULATOR_SYSTEMC_BRIDGE_SYSTEMCBRIDGE_H

#include <cstdint>

namespace wafer::model::detail {

struct SystemCRunner;
struct SystemCEvent;

using SystemCTileEntry = void (*)(void *owner, int64_t launchSlot);

bool isSystemCInitialElaboration();
const char *getSystemCBridgeDiagnostic();

SystemCRunner *createSystemCRunner(uint64_t tileCount,
                                   SystemCTileEntry tileEntry, void *owner);
void destroySystemCRunner(SystemCRunner *runner);
void startSystemCSimulation();
uint64_t getSystemCDeltaCount();
uint64_t getSystemCThreadProcessCount(const SystemCRunner *runner);
const char *getSystemCVersion();

SystemCEvent *createSystemCEvent();
void destroySystemCEvent(SystemCEvent *event);
void notifySystemCEvent(SystemCEvent *event);
void waitSystemCEvent(SystemCEvent *event);
void waitSystemCDelta();

} // namespace wafer::model::detail

#endif // WAFER_SIMULATOR_SYSTEMC_BRIDGE_SYSTEMCBRIDGE_H
