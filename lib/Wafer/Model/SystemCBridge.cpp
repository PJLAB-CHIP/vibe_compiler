//===- SystemCBridge.cpp - RTTI-isolated SystemC process bridge ---------===//

#define SC_INCLUDE_DYNAMIC_PROCESSES
#include <systemc>

#include "SystemCBridge.h"

#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace wafer::model::detail {
namespace {

thread_local std::string bridgeDiagnostic;

void recordException(const char *fallback) {
  try {
    throw;
  } catch (const std::exception &error) {
    bridgeDiagnostic = error.what();
  } catch (...) {
    bridgeDiagnostic = fallback;
  }
}

} // namespace

struct SystemCEvent {
  sc_core::sc_event event;
};

struct SystemCRunner final : public sc_core::sc_module {
  SystemCRunner(sc_core::sc_module_name name, uint64_t rankCount,
                SystemCRankEntry rankEntry, void *owner)
      : sc_core::sc_module(name), rankEntry(rankEntry), owner(owner) {
    processes.reserve(static_cast<size_t>(rankCount + 1));
    for (uint64_t rank = 0; rank < rankCount; ++rank) {
      sc_core::sc_spawn_options options;
      processes.push_back(sc_core::sc_spawn(
          sc_core::sc_bind(&SystemCRunner::runRank, this,
                           static_cast<int64_t>(rank)),
          sc_core::sc_gen_unique_name("wafer_rank_thread"), &options));
    }
    sc_core::sc_spawn_options options;
    processes.push_back(sc_core::sc_spawn(
        sc_core::sc_bind(&SystemCRunner::controlThread, this),
        sc_core::sc_gen_unique_name("wafer_control_thread"), &options));
  }

  void runRank(int64_t logicalRank) {
    ++threadProcessCount;
    if (sc_core::sc_get_current_process_handle().proc_kind() !=
        sc_core::SC_THREAD_PROC_) {
      bridgeDiagnostic = "rank process is not a SystemC thread";
      return;
    }
    rankEntry(owner, logicalRank);
  }

  void controlThread() {
    ++threadProcessCount;
    if (sc_core::sc_get_current_process_handle().proc_kind() !=
        sc_core::SC_THREAD_PROC_) {
      bridgeDiagnostic = "control process is not a SystemC thread";
      return;
    }
    sc_core::wait(controlEvent);
  }

  SystemCRankEntry rankEntry;
  void *owner;
  std::vector<sc_core::sc_process_handle> processes;
  sc_core::sc_event controlEvent;
  uint64_t threadProcessCount = 0;
};

bool isSystemCInitialElaboration() {
  return sc_core::sc_get_status() == sc_core::SC_ELABORATION;
}

const char *getSystemCBridgeDiagnostic() { return bridgeDiagnostic.c_str(); }

SystemCRunner *createSystemCRunner(uint64_t rankCount,
                                   SystemCRankEntry rankEntry, void *owner) {
  bridgeDiagnostic.clear();
  if (rankCount == 0 || !rankEntry || !owner) {
    bridgeDiagnostic = "SystemC runner requires ranks, callback, and owner";
    return nullptr;
  }
  try {
    return new SystemCRunner(sc_core::sc_gen_unique_name("wafer_target_model"),
                             rankCount, rankEntry, owner);
  } catch (...) {
    recordException("SystemC runner construction failed");
    return nullptr;
  }
}

void destroySystemCRunner(SystemCRunner *runner) { delete runner; }

void startSystemCSimulation() {
  try {
    sc_core::sc_start();
  } catch (...) {
    recordException("SystemC simulation failed");
  }
}

uint64_t getSystemCDeltaCount() { return sc_core::sc_delta_count(); }

uint64_t getSystemCThreadProcessCount(const SystemCRunner *runner) {
  return runner ? runner->threadProcessCount : 0;
}

const char *getSystemCVersion() { return sc_core::sc_version(); }

SystemCEvent *createSystemCEvent() {
  try {
    return new SystemCEvent;
  } catch (...) {
    recordException("SystemC event allocation failed");
    return nullptr;
  }
}

void destroySystemCEvent(SystemCEvent *event) { delete event; }

void notifySystemCEvent(SystemCEvent *event) {
  if (!event)
    return;
  try {
    event->event.notify(sc_core::SC_ZERO_TIME);
  } catch (...) {
    recordException("SystemC event notification failed");
  }
}

void waitSystemCEvent(SystemCEvent *event) {
  if (!event) {
    bridgeDiagnostic = "cannot wait on a null SystemC event";
    return;
  }
  try {
    sc_core::wait(event->event);
  } catch (...) {
    recordException("SystemC event wait failed");
  }
}

void waitSystemCDelta() {
  try {
    sc_core::wait(sc_core::SC_ZERO_TIME);
  } catch (...) {
    recordException("SystemC delta wait failed");
  }
}

} // namespace wafer::model::detail
