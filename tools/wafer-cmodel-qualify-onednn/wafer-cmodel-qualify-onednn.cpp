//===- wafer-cmodel-qualify-onednn.cpp - Offline onednn qualification
//-------===//

#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Model/Core/TargetModelInvocation.h"
#include "Wafer/Program/ProgramInvocation.h"
#include "Wafer/Target/Layout/TargetTensorMaterialization.h"
#include "Wafer/Target/Numeric/Qualification/OneDNNQualification.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

enum class Mode { SourceSpec, Calibrate, Freeze, Validate };

struct Options {
  std::optional<Mode> mode;
  std::string spec;
  std::string calibration;
  std::string heldOutSpec;
  std::string policy;
  std::string output;
  std::string format;
  std::string lhsNpy;
  std::string rhsNpy;
  std::optional<uint64_t> m;
  std::optional<uint64_t> k;
  std::optional<uint64_t> n;
  std::optional<uint64_t> batchCount;
  std::optional<uint64_t> seed;
  std::optional<uint64_t> formalScalarBudget;
  std::optional<uint64_t> formalFMABudget;
  std::optional<uint64_t> maximumTotalBytes;
  std::optional<uint64_t> maximumScratchpadBytes;
  std::optional<uint64_t> maximumReorderBytes;
  std::optional<double> maximumAbsoluteError;
  std::optional<double> maximumRelativeError;
};

void printUsage() {
  llvm::outs()
      << "usage:\n"
         "  wafer-cmodel-qualify-onednn --mode source-spec --format "
         "<f16|bf16|f32> --m <count> --k <count> --n <count> "
         "--batch-count <count> --seed <identity> --lhs-npy <path> "
         "--rhs-npy <path> --output <canonical-json>\n"
         "  wafer-cmodel-qualify-onednn --mode calibrate --spec "
         "<canonical-json> "
         "--output <path> <budgets>\n"
         "  wafer-cmodel-qualify-onednn --mode freeze --calibration <path> "
         "--held-out-spec <canonical-json> --maximum-absolute-error <value> "
         "--maximum-relative-error <value> --output <path>\n"
         "  wafer-cmodel-qualify-onednn --mode validate --policy <path> "
         "--output <path> <budgets>\n"
         "budgets: --formal-scalar-budget <count> --formal-fma-budget <count> "
         "--maximum-total-bytes <bytes> --maximum-scratchpad-bytes <bytes> "
         "--maximum-reorder-bytes <bytes>\n";
}

llvm::Expected<uint64_t> parseUnsigned(llvm::StringRef text,
                                       llvm::StringRef option) {
  uint64_t value = 0;
  if (text.empty() || text.getAsInteger(10, value))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   option + " requires an unsigned integer");
  return value;
}

llvm::Expected<double> parseNonnegativeDouble(llvm::StringRef text,
                                              llvm::StringRef option) {
  std::string storage = text.str();
  errno = 0;
  char *end = nullptr;
  double value = std::strtod(storage.c_str(), &end);
  if (errno != 0 || end != storage.c_str() + storage.size() ||
      !std::isfinite(value) || value < 0.0)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   option +
                                       " requires a finite nonnegative number");
  return value;
}

llvm::Expected<Options> parseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      printUsage();
      std::exit(0);
    }
    auto value = [&]() -> llvm::Expected<llvm::StringRef> {
      if (index + 1 >= argc)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "missing value for " + argument);
      return llvm::StringRef(argv[++index]);
    };
    if (argument == "--mode") {
      llvm::Expected<llvm::StringRef> text = value();
      if (!text)
        return text.takeError();
      if (*text == "source-spec")
        options.mode = Mode::SourceSpec;
      else if (*text == "calibrate")
        options.mode = Mode::Calibrate;
      else if (*text == "freeze")
        options.mode = Mode::Freeze;
      else if (*text == "validate")
        options.mode = Mode::Validate;
      else
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "unknown --mode " + *text);
      continue;
    }
    auto setString = [&](std::string &slot) -> llvm::Error {
      if (!slot.empty())
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "duplicate option " + argument);
      llvm::Expected<llvm::StringRef> text = value();
      if (!text)
        return text.takeError();
      if (text->empty())
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       argument + " must not be empty");
      slot = text->str();
      return llvm::Error::success();
    };
    if (argument == "--spec") {
      if (llvm::Error error = setString(options.spec))
        return std::move(error);
      continue;
    }
    if (argument == "--calibration") {
      if (llvm::Error error = setString(options.calibration))
        return std::move(error);
      continue;
    }
    if (argument == "--held-out-spec") {
      if (llvm::Error error = setString(options.heldOutSpec))
        return std::move(error);
      continue;
    }
    if (argument == "--policy") {
      if (llvm::Error error = setString(options.policy))
        return std::move(error);
      continue;
    }
    if (argument == "--output") {
      if (llvm::Error error = setString(options.output))
        return std::move(error);
      continue;
    }
    if (argument == "--format") {
      if (llvm::Error error = setString(options.format))
        return std::move(error);
      continue;
    }
    if (argument == "--lhs-npy") {
      if (llvm::Error error = setString(options.lhsNpy))
        return std::move(error);
      continue;
    }
    if (argument == "--rhs-npy") {
      if (llvm::Error error = setString(options.rhsNpy))
        return std::move(error);
      continue;
    }
    auto setUnsigned = [&](std::optional<uint64_t> &slot) -> llvm::Error {
      if (slot)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "duplicate option " + argument);
      llvm::Expected<llvm::StringRef> text = value();
      if (!text)
        return text.takeError();
      llvm::Expected<uint64_t> parsed = parseUnsigned(*text, argument);
      if (!parsed)
        return parsed.takeError();
      slot = *parsed;
      return llvm::Error::success();
    };
    if (argument == "--formal-scalar-budget") {
      if (llvm::Error error = setUnsigned(options.formalScalarBudget))
        return std::move(error);
      continue;
    }
    if (argument == "--m") {
      if (llvm::Error error = setUnsigned(options.m))
        return std::move(error);
      continue;
    }
    if (argument == "--k") {
      if (llvm::Error error = setUnsigned(options.k))
        return std::move(error);
      continue;
    }
    if (argument == "--n") {
      if (llvm::Error error = setUnsigned(options.n))
        return std::move(error);
      continue;
    }
    if (argument == "--batch-count") {
      if (llvm::Error error = setUnsigned(options.batchCount))
        return std::move(error);
      continue;
    }
    if (argument == "--seed") {
      if (llvm::Error error = setUnsigned(options.seed))
        return std::move(error);
      continue;
    }
    if (argument == "--formal-fma-budget") {
      if (llvm::Error error = setUnsigned(options.formalFMABudget))
        return std::move(error);
      continue;
    }
    if (argument == "--maximum-total-bytes") {
      if (llvm::Error error = setUnsigned(options.maximumTotalBytes))
        return std::move(error);
      continue;
    }
    if (argument == "--maximum-scratchpad-bytes") {
      if (llvm::Error error = setUnsigned(options.maximumScratchpadBytes))
        return std::move(error);
      continue;
    }
    if (argument == "--maximum-reorder-bytes") {
      if (llvm::Error error = setUnsigned(options.maximumReorderBytes))
        return std::move(error);
      continue;
    }
    auto setDouble = [&](std::optional<double> &slot) -> llvm::Error {
      if (slot)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "duplicate option " + argument);
      llvm::Expected<llvm::StringRef> text = value();
      if (!text)
        return text.takeError();
      llvm::Expected<double> parsed = parseNonnegativeDouble(*text, argument);
      if (!parsed)
        return parsed.takeError();
      slot = *parsed;
      return llvm::Error::success();
    };
    if (argument == "--maximum-absolute-error") {
      if (llvm::Error error = setDouble(options.maximumAbsoluteError))
        return std::move(error);
      continue;
    }
    if (argument == "--maximum-relative-error") {
      if (llvm::Error error = setDouble(options.maximumRelativeError))
        return std::move(error);
      continue;
    }
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown option " + argument);
  }
  if (!options.mode)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "--mode is required");
  if (options.output.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "--output is required");
  return options;
}

llvm::Error requireBudgets(const Options &options) {
  if (!options.formalScalarBudget || !options.formalFMABudget ||
      !options.maximumTotalBytes || !options.maximumScratchpadBytes ||
      !options.maximumReorderBytes)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "calibrate/validate requires every formal and onednn budget");
  return llvm::Error::success();
}

int fail(llvm::Error error) {
  llvm::errs() << "wafer-cmodel-qualify-onednn: "
               << llvm::toString(std::move(error)) << '\n';
  return 1;
}

llvm::Error writeSourceSpec(const Options &options) {
  if (options.format.empty() || options.lhsNpy.empty() ||
      options.rhsNpy.empty() || options.output.empty() || !options.m ||
      !options.k || !options.n || !options.batchCount || !options.seed)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "source-spec requires format, dimensions, batch count, seed and both "
        "NPY inputs plus an output path");
  if (*options.m == 0 || *options.k == 0 || *options.n == 0 ||
      *options.batchCount == 0 ||
      *options.m > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      *options.k > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      *options.n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      *options.batchCount >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "source-spec dimensions must be positive signed 64-bit values");
  llvm::Expected<wafer::LogicalFormat> format =
      wafer::parseLogicalFormat(options.format);
  if (!format)
    return format.takeError();
  const wafer::PhysicalTensorLayout layout =
      *options.batchCount == 1 ? wafer::PhysicalTensorLayout::Cx
                               : wafer::PhysicalTensorLayout::NCx;
  const wafer::MemLayout abiLayout =
      *options.batchCount == 1 ? wafer::MemLayout::Cx : wafer::MemLayout::NCx;
  std::vector<int64_t> lhsShape;
  std::vector<int64_t> rhsShape;
  std::vector<int64_t> destinationShape;
  if (*options.batchCount > 1) {
    lhsShape.push_back(static_cast<int64_t>(*options.batchCount));
    rhsShape.push_back(static_cast<int64_t>(*options.batchCount));
    destinationShape.push_back(static_cast<int64_t>(*options.batchCount));
  }
  lhsShape.insert(lhsShape.end(), {static_cast<int64_t>(*options.m),
                                   static_cast<int64_t>(*options.k)});
  rhsShape.insert(rhsShape.end(), {static_cast<int64_t>(*options.k),
                                   static_cast<int64_t>(*options.n)});
  destinationShape.insert(
      destinationShape.end(),
      {static_cast<int64_t>(*options.m), static_cast<int64_t>(*options.n)});
  auto makeKey = [&](llvm::ArrayRef<int64_t> shape) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(shape.size());
    for (int64_t dimension : shape)
      dimensions.push_back(static_cast<uint64_t>(dimension));
    return wafer::PhysicalTensorDescriptor::create(*format, layout,
                                                   std::move(dimensions));
  };
  llvm::Expected<wafer::PhysicalTensorDescriptor> lhsKey = makeKey(lhsShape);
  llvm::Expected<wafer::PhysicalTensorDescriptor> rhsKey = makeKey(rhsShape);
  llvm::Expected<wafer::PhysicalTensorDescriptor> destinationKey =
      makeKey(destinationShape);
  if (!lhsKey)
    return lhsKey.takeError();
  if (!rhsKey)
    return rhsKey.takeError();
  if (!destinationKey)
    return destinationKey.takeError();
  llvm::Expected<uint64_t> lhsBytes =
      wafer::getOneDNNTensorPhysicalBytes(*lhsKey);
  llvm::Expected<uint64_t> rhsBytes =
      wafer::getOneDNNTensorPhysicalBytes(*rhsKey);
  llvm::Expected<uint64_t> destinationBytes =
      wafer::getOneDNNTensorPhysicalBytes(*destinationKey);
  if (!lhsBytes)
    return lhsBytes.takeError();
  if (!rhsBytes)
    return rhsBytes.takeError();
  if (!destinationBytes)
    return destinationBytes.takeError();
  if (*lhsBytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      *rhsBytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      *destinationBytes >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return llvm::createStringError(
        llvm::errc::file_too_large,
        "source-spec physical tensor size exceeds the Kernel ABI domain");
  llvm::Expected<wafer::compiler::ProgramTensor> lhs =
      wafer::compiler::ProgramTensor::loadNpy(options.lhsNpy);
  llvm::Expected<wafer::compiler::ProgramTensor> rhs =
      wafer::compiler::ProgramTensor::loadNpy(options.rhsNpy);
  if (!lhs)
    return lhs.takeError();
  if (!rhs)
    return rhs.takeError();
  const wafer::LogicalFormatDescriptor *descriptor =
      wafer::findLogicalFormatDescriptor(*format);
  if (!descriptor || descriptor->bitpacked || descriptor->storageBits % 8 != 0)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "source-spec format is not byte-oriented");
  if (destinationKey->getElementCount() >
      std::numeric_limits<size_t>::max() / (descriptor->storageBits / 8))
    return llvm::createStringError(llvm::errc::file_too_large,
                                   "source-spec destination is too large");
  std::vector<uint8_t> destinationCompact(
      static_cast<size_t>(destinationKey->getElementCount() *
                          (descriptor->storageBits / 8)),
      0);
  llvm::Expected<wafer::compiler::ProgramTensor> destination =
      wafer::compiler::ProgramTensor::create(
          wafer::getProgramElementType(*format), destinationShape,
          destinationCompact);
  if (!destination)
    return destination.takeError();
  auto makeSlot = [&](llvm::ArrayRef<int64_t> shape, uint64_t physicalBytes) {
    return wafer::compiler::TileEntryArgument{
        0,
        wafer::compiler::TileEntryArgumentKind::ExternalInput,
        0,
        "tensor",
        *format,
        abiLayout,
        std::vector<int64_t>(shape.begin(), shape.end()),
        static_cast<int64_t>(physicalBytes),
        64,
        wafer::compiler::TileEntryArgumentAccess::ReadOnly};
  };
  auto lhsSlot = makeSlot(lhsShape, *lhsBytes);
  auto rhsSlot = makeSlot(rhsShape, *rhsBytes);
  auto destinationSlot = makeSlot(destinationShape, *destinationBytes);
  llvm::Expected<std::vector<uint8_t>> lhsPhysical =
      wafer::model::encodeTargetModelProgramTensor(*lhs, lhsSlot);
  llvm::Expected<std::vector<uint8_t>> rhsPhysical =
      wafer::model::encodeTargetModelProgramTensor(*rhs, rhsSlot);
  llvm::Expected<std::vector<uint8_t>> destinationPhysical =
      wafer::model::encodeTargetModelProgramTensor(*destination,
                                                   destinationSlot);
  if (!lhsPhysical)
    return lhsPhysical.takeError();
  if (!rhsPhysical)
    return rhsPhysical.takeError();
  if (!destinationPhysical)
    return destinationPhysical.takeError();
  llvm::Expected<wafer::OneDNNQualificationSpec> spec =
      wafer::OneDNNQualificationSpec::createWithPhysicalPayload(
          *format, *options.m, *options.k, *options.n, *options.batchCount,
          layout, layout, layout, *options.seed, std::move(*lhsPhysical),
          std::move(*rhsPhysical), std::move(*destinationPhysical));
  if (!spec)
    return spec.takeError();
  return wafer::writeOneDNNQualificationSpec(*spec, options.output);
}

} // namespace

int main(int argc, char **argv) {
  llvm::Expected<Options> options = parseOptions(argc, argv);
  if (!options)
    return fail(options.takeError());
  if (*options->mode == Mode::SourceSpec) {
    if (llvm::Error error = writeSourceSpec(*options))
      return fail(std::move(error));
    llvm::outs() << "onednn source qualification spec written: "
                 << options->output << '\n';
    return 0;
  }
  if (*options->mode == Mode::Freeze) {
    if (options->calibration.empty() || options->heldOutSpec.empty() ||
        !options->maximumAbsoluteError || !options->maximumRelativeError)
      return fail(llvm::createStringError(
          llvm::errc::invalid_argument,
          "freeze requires calibration, held-out spec and both tolerances"));
    if (llvm::Error error = wafer::freezeOneDNNBackendPolicy(
            options->calibration, options->heldOutSpec, options->output,
            {*options->maximumAbsoluteError, *options->maximumRelativeError}))
      return fail(std::move(error));
    llvm::outs() << "onednn qualification policy frozen: " << options->output
                 << '\n';
    return 0;
  }
  if (llvm::Error error = requireBudgets(*options))
    return fail(std::move(error));
  llvm::Expected<wafer::OneDNNExecutionEnvironment> environment =
      wafer::createManagedOneDNNExecutionEnvironment();
  if (!environment)
    return fail(environment.takeError());
  wafer::FormalNumericWorkBudget formal =
      wafer::FormalNumericWorkBudget::create(*options->formalScalarBudget,
                                             *options->formalFMABudget);
  wafer::OneDNNNumericWorkBudget onednn =
      wafer::OneDNNNumericWorkBudget::create(*options->maximumTotalBytes,
                                             *options->maximumScratchpadBytes,
                                             *options->maximumReorderBytes);
  if (*options->mode == Mode::Calibrate) {
    if (options->spec.empty())
      return fail(llvm::createStringError(llvm::errc::invalid_argument,
                                          "calibrate requires --spec"));
    if (llvm::Error error = wafer::calibrateOneDNNBackend(
            *environment, options->spec, options->output, formal, onednn))
      return fail(std::move(error));
    llvm::outs() << "onednn calibration written: " << options->output << '\n';
    return 0;
  }
  if (options->policy.empty())
    return fail(llvm::createStringError(llvm::errc::invalid_argument,
                                        "validate requires --policy"));
  if (llvm::Error error = wafer::validateOneDNNBackend(
          *environment, options->policy, options->output, formal, onednn))
    return fail(std::move(error));
  llvm::Expected<wafer::VerifiedOneDNNQualificationRecord> record =
      wafer::loadVerifiedOneDNNQualificationRecord(options->output);
  if (!record)
    return fail(record.takeError());
  llvm::outs() << "onednn qualification validated: " << options->output
               << " digest=" << record->getRecordDigest() << '\n';
  return 0;
}
