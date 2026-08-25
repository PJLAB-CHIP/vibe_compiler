//===- CompileTiming.h - Invocation-local compile timing -------*- C++ -*-===//

#ifndef WAFER_SUPPORT_COMPILETIMING_H
#define WAFER_SUPPORT_COMPILETIMING_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <time.h>

namespace wafer::support {

/// Thread-safe, invocation-local timing owner shared by compiler orchestration
/// and its bounded workers. Short events are aggregated in memory. A low-rate
/// active-work report keeps an externally timed-out compilation diagnosable
/// without adding per-candidate I/O to the measured path.
class CompileTimingSession {
public:
  using Clock = std::chrono::steady_clock;

  struct Token {
    Token() = default;
    Token(Token &&) = default;
    Token &operator=(Token &&) = default;
    Token(const Token &) = delete;
    Token &operator=(const Token &) = delete;

  private:
    friend class CompileTimingSession;
    uint64_t id = 0;
    uint64_t threadId = 0;
    size_t shardIndex = 0;
    Clock::time_point wallStart;
    uint64_t cpuStartNs = 0;
    std::string kind;
    std::string pipeline;
    std::string name;
    std::string detail;
  };

  explicit CompileTimingSession(llvm::raw_ostream &diagnostics)
      : diagnostics(diagnostics), sessionStart(Clock::now()),
        monitor([this] { monitorActiveWork(); }) {}

  CompileTimingSession(const CompileTimingSession &) = delete;
  CompileTimingSession &operator=(const CompileTimingSession &) = delete;

  ~CompileTimingSession() { stopMonitor(); }

  Token begin(llvm::StringRef kind, llvm::StringRef pipeline,
              llvm::StringRef name, llvm::StringRef detail = {}) {
    Token token;
    token.id = nextId.fetch_add(1, std::memory_order_relaxed);
    token.threadId = llvm::get_threadid();
    token.shardIndex = token.threadId % kTimingShardCount;
    token.wallStart = Clock::now();
    token.cpuStartNs = readThreadCpuNanoseconds();
    token.kind = kind.str();
    token.pipeline = pipeline.str();
    token.name = name.str();
    token.detail = detail.str();
    TimingShard &shard = timingShards[token.shardIndex];
    std::lock_guard<std::mutex> lock(shard.mutex);
    shard.active.emplace(token.id,
                         ActiveRecord{token.id, token.threadId, token.wallStart,
                                      token.kind, token.pipeline, token.name,
                                      token.detail});
    return token;
  }

  void end(Token &token, bool failed = false) {
    if (token.id == 0)
      return;
    const Clock::time_point wallEnd = Clock::now();
    const uint64_t cpuEndNs = readThreadCpuNanoseconds();
    const uint64_t wallUs = elapsedMicroseconds(token.wallStart, wallEnd);
    const uint64_t cpuUs =
        cpuEndNs >= token.cpuStartNs ? (cpuEndNs - token.cpuStartNs) / 1000 : 0;
    TimingShard &shard = timingShards[token.shardIndex];
    std::lock_guard<std::mutex> lock(shard.mutex);
    shard.active.erase(token.id);
    Summary &summary =
        shard.summaries[{token.kind, token.pipeline, token.name}];
    ++summary.calls;
    summary.wallUs += wallUs;
    summary.cpuUs += cpuUs;
    summary.maxWallUs = std::max(summary.maxWallUs, wallUs);
    summary.failures += failed;
    token.id = 0;
  }

  /// Stops active-work reporting and emits one stable, wall-descending table.
  /// It is safe to call this more than once; only the first call prints.
  void finishAndPrintSummary() {
    bool expected = false;
    if (!finished.compare_exchange_strong(expected, true,
                                          std::memory_order_acq_rel))
      return;
    stopMonitor();

    std::map<SummaryKey, Summary> combined;
    for (TimingShard &shard : timingShards) {
      std::lock_guard<std::mutex> lock(shard.mutex);
      mergeSummaries(combined, shard.summaries);
    }
    std::vector<std::pair<SummaryKey, Summary>> rows(combined.begin(),
                                                     combined.end());
    llvm::stable_sort(rows, [](const auto &lhs, const auto &rhs) {
      if (lhs.second.wallUs != rhs.second.wallUs)
        return lhs.second.wallUs > rhs.second.wallUs;
      return lhs.first < rhs.first;
    });

    std::string report;
    llvm::raw_string_ostream os(report);
    os << "wafer-compile: compile-timing-summary-begin\n";
    os << "wafer-compile: | kind | pipeline | item | calls | cumulative "
          "wall ms | cumulative CPU ms | average wall ms | max wall ms | "
          "failures |\n";
    os << "wafer-compile: | --- | --- | --- | ---: | ---: | ---: | ---: | "
          "---: | ---: |\n";
    for (const auto &[key, summary] : rows) {
      const double wallMs = microsecondsToMilliseconds(summary.wallUs);
      const double cpuMs = microsecondsToMilliseconds(summary.cpuUs);
      const double averageMs =
          summary.calls ? wallMs / static_cast<double>(summary.calls) : 0.0;
      os << "wafer-compile: | " << key.kind << " | " << key.pipeline << " | "
         << key.name << " | " << summary.calls << " | "
         << llvm::formatv("{0:F3}", wallMs) << " | "
         << llvm::formatv("{0:F3}", cpuMs) << " | "
         << llvm::formatv("{0:F3}", averageMs) << " | "
         << llvm::formatv("{0:F3}",
                          microsecondsToMilliseconds(summary.maxWallUs))
         << " | " << summary.failures << " |\n";
    }
    os << "wafer-compile: compile-timing-summary-end transaction_wall_ms="
       << llvm::formatv("{0:F3}",
                        microsecondsToMilliseconds(
                            elapsedMicroseconds(sessionStart, Clock::now())))
       << "\n";
    os.flush();
    writeDiagnostic(report);
  }

private:
  struct ActiveRecord {
    uint64_t id;
    uint64_t threadId;
    Clock::time_point wallStart;
    std::string kind;
    std::string pipeline;
    std::string name;
    std::string detail;
  };

  struct SummaryKey {
    std::string kind;
    std::string pipeline;
    std::string name;

    bool operator<(const SummaryKey &other) const {
      return std::tie(kind, pipeline, name) <
             std::tie(other.kind, other.pipeline, other.name);
    }
  };

  struct Summary {
    uint64_t calls = 0;
    uint64_t wallUs = 0;
    uint64_t cpuUs = 0;
    uint64_t maxWallUs = 0;
    uint64_t failures = 0;
  };

  struct TimingShard {
    std::mutex mutex;
    std::map<uint64_t, ActiveRecord> active;
    std::map<SummaryKey, Summary> summaries;
  };

  static constexpr size_t kTimingShardCount = 256;

  static void mergeSummary(Summary &destination, const Summary &source) {
    destination.calls += source.calls;
    destination.wallUs += source.wallUs;
    destination.cpuUs += source.cpuUs;
    destination.maxWallUs = std::max(destination.maxWallUs, source.maxWallUs);
    destination.failures += source.failures;
  }

  static void mergeSummaries(std::map<SummaryKey, Summary> &destination,
                             const std::map<SummaryKey, Summary> &source) {
    for (const auto &[key, summary] : source)
      mergeSummary(destination[key], summary);
  }

  static uint64_t elapsedMicroseconds(Clock::time_point start,
                                      Clock::time_point end) {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    return elapsed > 0 ? static_cast<uint64_t>(elapsed) : 0;
  }

  static double microsecondsToMilliseconds(uint64_t microseconds) {
    return static_cast<double>(microseconds) / 1000.0;
  }

  static uint64_t readThreadCpuNanoseconds() {
#ifdef CLOCK_THREAD_CPUTIME_ID
    struct timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0)
      return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
             static_cast<uint64_t>(value.tv_nsec);
#endif
    return 0;
  }

  void stopMonitor() {
    {
      std::lock_guard<std::mutex> lock(monitorMutex);
      monitorStopping = true;
    }
    monitorWake.notify_all();
    if (monitor.joinable())
      monitor.join();
  }

  void monitorActiveWork() {
    constexpr auto interval = std::chrono::seconds(10);
    while (true) {
      {
        std::unique_lock<std::mutex> lock(monitorMutex);
        if (monitorWake.wait_for(lock, interval,
                                 [&] { return monitorStopping; }))
          return;
      }

      const Clock::time_point now = Clock::now();
      std::vector<std::pair<uint64_t, ActiveRecord>> snapshot;
      std::map<SummaryKey, Summary> progress;
      size_t activeCount = 0;
      std::map<uint64_t, ActiveRecord> leafByThread;
      for (TimingShard &shard : timingShards) {
        std::lock_guard<std::mutex> lock(shard.mutex);
        activeCount += shard.active.size();
        for (const auto &[id, record] : shard.active) {
          auto found = leafByThread.find(record.threadId);
          if (found == leafByThread.end() ||
              record.wallStart > found->second.wallStart ||
              (record.wallStart == found->second.wallStart &&
               record.id > found->second.id))
            leafByThread.insert_or_assign(record.threadId, record);
        }
        mergeSummaries(progress, shard.summaries);
      }
      snapshot.reserve(leafByThread.size());
      for (const auto &[threadId, record] : leafByThread)
        snapshot.emplace_back(elapsedMicroseconds(record.wallStart, now),
                              record);
      std::vector<std::pair<SummaryKey, Summary>> progressRows(progress.begin(),
                                                               progress.end());
      llvm::stable_sort(snapshot, [](const auto &lhs, const auto &rhs) {
        if (lhs.first != rhs.first)
          return lhs.first > rhs.first;
        return lhs.second.id < rhs.second.id;
      });
      llvm::stable_sort(progressRows, [](const auto &lhs, const auto &rhs) {
        if (lhs.second.wallUs != rhs.second.wallUs)
          return lhs.second.wallUs > rhs.second.wallUs;
        return lhs.first < rhs.first;
      });

      constexpr size_t activeReportLimit = 20;
      constexpr size_t progressReportLimit = 10;
      std::string report;
      llvm::raw_string_ostream os(report);
      os << "wafer-compile: compile-timing-active-count=" << activeCount
         << " leaf-count=" << snapshot.size()
         << " longest=" << std::min(snapshot.size(), activeReportLimit) << "\n";
      for (const auto &[elapsedUs, record] :
           llvm::ArrayRef(snapshot).take_front(activeReportLimit)) {
        os << "wafer-compile: compile-timing-active elapsed_ms="
           << elapsedUs / 1000 << " thread=" << record.threadId
           << " kind=" << record.kind << " pipeline=" << record.pipeline
           << " item=" << record.name;
        if (!record.detail.empty())
          os << " detail=" << record.detail;
        os << "\n";
      }
      os << "wafer-compile: compile-timing-progress-count="
         << progressRows.size()
         << " longest=" << std::min(progressRows.size(), progressReportLimit)
         << "\n";
      for (const auto &[key, summary] :
           llvm::ArrayRef(progressRows).take_front(progressReportLimit)) {
        os << "wafer-compile: compile-timing-progress cumulative_wall_ms="
           << llvm::formatv("{0:F3}",
                            microsecondsToMilliseconds(summary.wallUs))
           << " cumulative_cpu_ms="
           << llvm::formatv("{0:F3}", microsecondsToMilliseconds(summary.cpuUs))
           << " calls=" << summary.calls << " max_wall_ms="
           << llvm::formatv("{0:F3}",
                            microsecondsToMilliseconds(summary.maxWallUs))
           << " failures=" << summary.failures << " kind=" << key.kind
           << " pipeline=" << key.pipeline << " item=" << key.name << "\n";
      }
      os.flush();
      writeDiagnostic(report);
    }
  }

  void writeDiagnostic(llvm::StringRef text) {
    std::lock_guard<std::mutex> lock(outputMutex);
    diagnostics.write(text.data(), text.size());
    diagnostics.flush();
  }

  llvm::raw_ostream &diagnostics;
  const Clock::time_point sessionStart;
  std::atomic<uint64_t> nextId{1};
  std::atomic<bool> finished{false};
  std::array<TimingShard, kTimingShardCount> timingShards;
  std::mutex outputMutex;
  std::mutex monitorMutex;
  std::condition_variable monitorWake;
  bool monitorStopping = false;
  std::thread monitor;
};

inline thread_local std::shared_ptr<CompileTimingSession>
    activeCompileTimingSession;

inline std::shared_ptr<CompileTimingSession> getActiveCompileTimingSession() {
  return activeCompileTimingSession;
}

/// Exact operation inventory for one explicitly profiled IR boundary. The
/// inventory is invocation-local diagnostic data; it is never consulted by
/// compilation decisions or serialized into IR/package state.
class CompileIRInventory {
public:
  void record(mlir::Operation *root) {
    if (!root)
      return;
    root->walk([&](mlir::Operation *operation) {
      ++totalOperations;
      ++operationsByName[operation->getName().getStringRef().str()];
    });
  }

  void merge(const CompileIRInventory &other) {
    totalOperations += other.totalOperations;
    for (const auto &[name, count] : other.operationsByName)
      operationsByName[name] += count;
  }

  void print(llvm::StringRef stage, llvm::raw_ostream &output) const {
    output << "wafer-compile: ir-inventory stage=" << stage
           << " total_operations=" << totalOperations
           << " distinct_operations=" << operationsByName.size() << '\n';
    for (const auto &[name, count] : operationsByName)
      output << "wafer-compile: ir-inventory-op stage=" << stage
             << " op=" << name << " count=" << count << '\n';
  }

  uint64_t getTotalOperations() const { return totalOperations; }

private:
  uint64_t totalOperations = 0;
  std::map<std::string, uint64_t> operationsByName;
};

/// Installs an invocation timing session on the current thread. Worker owners
/// explicitly propagate the shared session when they create bounded threads.
class ScopedCompileTimingActivation {
public:
  explicit ScopedCompileTimingActivation(
      std::shared_ptr<CompileTimingSession> session)
      : previous(std::move(activeCompileTimingSession)) {
    activeCompileTimingSession = std::move(session);
  }

  ScopedCompileTimingActivation(const ScopedCompileTimingActivation &) = delete;
  ScopedCompileTimingActivation &
  operator=(const ScopedCompileTimingActivation &) = delete;

  ~ScopedCompileTimingActivation() {
    activeCompileTimingSession = std::move(previous);
  }

private:
  std::shared_ptr<CompileTimingSession> previous;
};

/// RAII measurement for compiler-owned stages that are not themselves MLIR
/// passes. The stable pipeline/item pair is aggregated; detail is shown only
/// while the item is active so per-request indices do not explode the table.
class ScopedCompileTimingSpan {
public:
  ScopedCompileTimingSpan(llvm::StringRef kind, llvm::StringRef pipeline,
                          llvm::StringRef name, llvm::StringRef detail = {})
      : session(getActiveCompileTimingSession()) {
    if (session)
      token.emplace(session->begin(kind, pipeline, name, detail));
  }

  ScopedCompileTimingSpan(const ScopedCompileTimingSpan &) = delete;
  ScopedCompileTimingSpan &operator=(const ScopedCompileTimingSpan &) = delete;

  ~ScopedCompileTimingSpan() {
    if (session && token)
      session->end(*token, failed);
  }

  void markFailed() { failed = true; }

private:
  std::shared_ptr<CompileTimingSession> session;
  std::optional<CompileTimingSession::Token> token;
  bool failed = false;
};

namespace detail {

class DetailedPassTimingInstrumentation final
    : public mlir::PassInstrumentation {
public:
  DetailedPassTimingInstrumentation(
      std::shared_ptr<CompileTimingSession> session, llvm::StringRef pipeline)
      : session(std::move(session)), pipeline(pipeline.str()) {}

  void runBeforePass(mlir::Pass *pass, mlir::Operation *operation) override {
    ActivePass activePass;
    activePass.pass = pass;
    activePass.operation = operation;
    llvm::StringRef name = pass->getArgument();
    if (name.empty())
      name = pass->getName();
    std::string operationDetail =
        (llvm::Twine("op=") + operation->getName().getStringRef()).str();
    activePass.token = session->begin("pass", pipeline, name, operationDetail);
    std::lock_guard<std::mutex> lock(mutex);
    activePasses[llvm::get_threadid()].push_back(std::move(activePass));
  }

  void runAfterPass(mlir::Pass *pass, mlir::Operation *operation) override {
    finishPass(pass, operation, /*failed=*/false);
  }

  void runAfterPassFailed(mlir::Pass *pass,
                          mlir::Operation *operation) override {
    finishPass(pass, operation, /*failed=*/true);
  }

  void runBeforeAnalysis(llvm::StringRef name, mlir::TypeID,
                         mlir::Operation *operation) override {
    ActiveAnalysis activeAnalysis;
    activeAnalysis.name = name.str();
    activeAnalysis.operation = operation;
    std::string operationDetail =
        (llvm::Twine("op=") + operation->getName().getStringRef()).str();
    activeAnalysis.token =
        session->begin("analysis", pipeline, name, operationDetail);
    std::lock_guard<std::mutex> lock(mutex);
    activeAnalyses[llvm::get_threadid()].push_back(std::move(activeAnalysis));
  }

  void runAfterAnalysis(llvm::StringRef name, mlir::TypeID,
                        mlir::Operation *operation) override {
    std::optional<CompileTimingSession::Token> token;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto found = activeAnalyses.find(llvm::get_threadid());
      if (found == activeAnalyses.end())
        return;
      auto &stack = found->second;
      auto match = std::find_if(
          stack.rbegin(), stack.rend(), [&](const ActiveAnalysis &active) {
            return active.name == name && active.operation == operation;
          });
      if (match == stack.rend())
        return;
      auto forward = std::next(match).base();
      token.emplace(std::move(forward->token));
      stack.erase(forward);
      if (stack.empty())
        activeAnalyses.erase(found);
    }
    session->end(*token);
  }

private:
  struct ActivePass {
    mlir::Pass *pass = nullptr;
    mlir::Operation *operation = nullptr;
    CompileTimingSession::Token token;
  };

  struct ActiveAnalysis {
    std::string name;
    mlir::Operation *operation = nullptr;
    CompileTimingSession::Token token;
  };

  void finishPass(mlir::Pass *pass, mlir::Operation *operation, bool failed) {
    std::optional<CompileTimingSession::Token> token;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto found = activePasses.find(llvm::get_threadid());
      if (found == activePasses.end())
        return;
      auto &stack = found->second;
      auto match = std::find_if(
          stack.rbegin(), stack.rend(), [&](const ActivePass &active) {
            return active.pass == pass && active.operation == operation;
          });
      if (match == stack.rend())
        return;
      auto forward = std::next(match).base();
      token.emplace(std::move(forward->token));
      stack.erase(forward);
      if (stack.empty())
        activePasses.erase(found);
    }
    session->end(*token, failed);
  }

  std::shared_ptr<CompileTimingSession> session;
  std::string pipeline;
  std::mutex mutex;
  std::unordered_map<uint64_t, std::vector<ActivePass>> activePasses;
  std::unordered_map<uint64_t, std::vector<ActiveAnalysis>> activeAnalyses;
};

} // namespace detail

/// Attaches detailed pass/analysis timing when the current thread belongs to
/// an explicitly timed compiler invocation. With timing disabled this is a
/// single thread-local pointer check.
inline void attachCompileTiming(mlir::PassManager &manager,
                                llvm::StringRef pipeline) {
  std::shared_ptr<CompileTimingSession> session =
      getActiveCompileTimingSession();
  if (session)
    manager.addInstrumentation(
        std::make_unique<detail::DetailedPassTimingInstrumentation>(
            std::move(session), pipeline));
}

} // namespace wafer::support

#endif // WAFER_SUPPORT_COMPILETIMING_H
