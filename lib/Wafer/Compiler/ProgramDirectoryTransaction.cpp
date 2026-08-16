//===- ProgramDirectoryTransaction.cpp - Program directory transaction ---===//

#include "CompilationInternal.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <string>
#include <system_error>

#ifdef __linux__
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace wafer::compiler::detail {

bool reject(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << "wafer-compile: " << message << "\n";
  return true;
}

bool pathEntryExists(llvm::StringRef path) {
  llvm::sys::fs::file_type type =
      llvm::sys::fs::get_file_type(path, /*Follow=*/false);
  return type != llvm::sys::fs::file_type::file_not_found &&
         type != llvm::sys::fs::file_type::status_error;
}

bool isDirectory(llvm::StringRef path) {
  llvm::sys::fs::file_status status;
  return !llvm::sys::fs::status(path, status) &&
         llvm::sys::fs::is_directory(status);
}

bool isRegularFile(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::regular_file;
}

bool createDirectory(llvm::StringRef path, llvm::raw_ostream &diagnostics) {
  if (std::error_code error = llvm::sys::fs::create_directories(path))
    return reject(diagnostics, "failed to create directory '" + path.str() +
                                   "': " + error.message());
  return false;
}

std::string programFile(llvm::StringRef programDirectory,
                        llvm::ArrayRef<llvm::StringRef> components) {
  llvm::SmallString<256> path(programDirectory);
  for (llvm::StringRef component : components)
    llvm::sys::path::append(path, component);
  return path.str().str();
}

bool copyDirectory(llvm::StringRef sourceDirectory,
                   llvm::StringRef destinationDirectory,
                   llvm::raw_ostream &diagnostics,
                   llvm::ArrayRef<llvm::StringRef> skipMembers) {
  if (!isDirectory(sourceDirectory))
    return reject(diagnostics,
                  "source program directory is not a directory: '" +
                      sourceDirectory.str() + "'");
  if (createDirectory(destinationDirectory, diagnostics))
    return true;

  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(sourceDirectory, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return reject(diagnostics, "failed to walk directory '" +
                                     sourceDirectory.str() +
                                     "': " + error.message());

    llvm::StringRef source = iterator->path();
    if (iterator->type() == llvm::sys::fs::file_type::symlink_file)
      return reject(diagnostics,
                    "program directory contains unsupported symbolic link: '" +
                        source.str() + "'");
    llvm::StringRef relative = source;
    if (!relative.consume_front(sourceDirectory))
      return reject(diagnostics, "failed to derive source program member path");
    if (relative.starts_with(llvm::sys::path::get_separator()))
      relative = relative.drop_front();
    if (llvm::is_contained(skipMembers, relative)) {
      iterator.no_push();
      continue;
    }

    llvm::SmallString<256> destination(destinationDirectory);
    llvm::sys::path::append(destination, relative);

    llvm::sys::fs::file_status status;
    if (std::error_code statusError = llvm::sys::fs::status(source, status))
      return reject(diagnostics, "failed to stat '" + source.str() +
                                     "': " + statusError.message());

    if (llvm::sys::fs::is_directory(status)) {
      if (createDirectory(destination, diagnostics))
        return true;
      continue;
    }
    if (!llvm::sys::fs::is_regular_file(status))
      return reject(
          diagnostics,
          "program directory contains unsupported non-regular member: '" +
              source.str() + "'");

    llvm::SmallString<256> parent(destination);
    llvm::sys::path::remove_filename(parent);
    if (createDirectory(parent, diagnostics))
      return true;
    if (std::error_code copyError =
            llvm::sys::fs::copy_file(source, destination))
      return reject(diagnostics, "failed to copy '" + source.str() +
                                     "': " + copyError.message());
  }

  if (error)
    return reject(diagnostics, "failed to walk directory '" +
                                   sourceDirectory.str() +
                                   "': " + error.message());
  return false;
}

bool validateRegularDirectoryTree(llvm::StringRef root,
                                  llvm::raw_ostream &diagnostics) {
  if (llvm::sys::fs::get_file_type(root, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return reject(diagnostics, "helper output is not a real directory: '" +
                                   root.str() + "'");

  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(root, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return reject(diagnostics, "failed to walk helper output directory: " +
                                     error.message());
    llvm::sys::fs::file_type type = iterator->type();
    if (type != llvm::sys::fs::file_type::directory_file &&
        type != llvm::sys::fs::file_type::regular_file)
      return reject(diagnostics,
                    "helper output contains unsupported non-regular member: '" +
                        iterator->path() + "'");
  }
  if (error)
    return reject(diagnostics,
                  "failed to walk helper output directory: " + error.message());
  return false;
}

bool mergeMissingProgramMembers(llvm::StringRef sourceDirectory,
                                llvm::StringRef destinationDirectory,
                                llvm::raw_ostream &diagnostics) {
  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(sourceDirectory, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return reject(diagnostics, "failed to walk source program members: " +
                                     error.message());

    llvm::StringRef source = iterator->path();
    llvm::StringRef relative = source;
    if (!relative.consume_front(sourceDirectory))
      return reject(diagnostics, "failed to derive source program member path");
    if (relative.starts_with(llvm::sys::path::get_separator()))
      relative = relative.drop_front();

    llvm::SmallString<256> destination(destinationDirectory);
    llvm::sys::path::append(destination, relative);
    llvm::sys::fs::file_type sourceType = iterator->type();
    llvm::sys::fs::file_status destinationStatus;
    std::error_code destinationStatusError =
        llvm::sys::fs::status(destination, destinationStatus, /*follow=*/false);
    bool destinationMissing = false;
    if (destinationStatusError) {
      if (destinationStatusError == std::errc::no_such_file_or_directory) {
        destinationMissing = true;
      } else {
        return reject(diagnostics, "failed to inspect helper output member '" +
                                       relative.str() + "': " +
                                       destinationStatusError.message());
      }
    }
    llvm::sys::fs::file_type destinationType = destinationStatus.type();

    if (sourceType == llvm::sys::fs::file_type::directory_file) {
      if (destinationMissing) {
        if (createDirectory(destination, diagnostics))
          return true;
      } else if (destinationType != llvm::sys::fs::file_type::directory_file) {
        return reject(
            diagnostics,
            "helper output changes a source directory into a file: '" +
                relative.str() + "'");
      }
      continue;
    }
    if (sourceType != llvm::sys::fs::file_type::regular_file)
      return reject(
          diagnostics,
          "source program contains unsupported non-regular member: '" +
              source.str() + "'");
    if (!destinationMissing) {
      if (destinationType != llvm::sys::fs::file_type::regular_file)
        return reject(
            diagnostics,
            "helper output changes a source file into a directory: '" +
                relative.str() + "'");
      continue;
    }
    llvm::SmallString<256> parent(destination);
    llvm::sys::path::remove_filename(parent);
    if (createDirectory(parent, diagnostics))
      return true;
    if (std::error_code copyError =
            llvm::sys::fs::copy_file(source, destination))
      return reject(diagnostics, "failed to preserve source program member '" +
                                     relative.str() +
                                     "': " + copyError.message());
  }
  if (error)
    return reject(diagnostics,
                  "failed to walk source program members: " + error.message());
  return false;
}

bool writeProgramModule(mlir::ModuleOp module, llvm::StringRef programDirectory,
                        llvm::raw_ostream &diagnostics) {
  std::string functionsDirectory =
      programFile(programDirectory, {llvm::StringRef("functions")});
  if (createDirectory(functionsDirectory, diagnostics))
    return true;

  std::string modulePath =
      programFile(programDirectory, {llvm::StringRef("functions"),
                                     llvm::StringRef("forward.mlir")});
  std::error_code error;
  llvm::raw_fd_ostream stream(modulePath, error, llvm::sys::fs::OF_Text);
  if (error)
    return reject(diagnostics,
                  "failed to write '" + modulePath + "': " + error.message());
  module->print(stream);
  stream << "\n";
  stream.close();
  if (stream.has_error())
    return reject(diagnostics, "failed to close '" + modulePath + "'");
  return false;
}

bool makeAbsoluteNormalizedPath(llvm::StringRef path,
                                llvm::SmallVectorImpl<char> &storage,
                                llvm::raw_ostream &diagnostics) {
  storage.assign(path.begin(), path.end());
  if (std::error_code error = llvm::sys::fs::make_absolute(storage))
    return reject(diagnostics,
                  "failed to make path absolute: " + error.message());
  llvm::sys::path::remove_dots(storage, /*remove_dot_dot=*/true);
  return false;
}

bool resolveThroughExistingAncestor(llvm::StringRef path,
                                    llvm::SmallVectorImpl<char> &storage,
                                    llvm::raw_ostream &diagnostics) {
  llvm::SmallString<256> existing(path);
  llvm::SmallVector<std::string, 8> missingComponents;
  while (!pathEntryExists(existing)) {
    llvm::StringRef filename = llvm::sys::path::filename(existing);
    if (filename.empty())
      return reject(diagnostics,
                    "failed to locate an existing output path ancestor");
    missingComponents.push_back(filename.str());
    llvm::sys::path::remove_filename(existing);
  }

  llvm::SmallString<256> canonicalExisting;
  if (std::error_code error =
          llvm::sys::fs::real_path(existing, canonicalExisting))
    return reject(diagnostics, "failed to resolve output path ancestor '" +
                                   existing.str().str() +
                                   "': " + error.message());
  for (const std::string &component : llvm::reverse(missingComponents))
    llvm::sys::path::append(canonicalExisting, component);
  storage.assign(canonicalExisting.begin(), canonicalExisting.end());
  return false;
}

bool pathIsWithin(llvm::StringRef path, llvm::StringRef directory) {
  if (path == directory)
    return true;
  if (!path.starts_with(directory) || path.size() <= directory.size())
    return false;
  return llvm::sys::path::is_separator(path[directory.size()]);
}

bool renameDirectoryNoReplace(llvm::StringRef source,
                              llvm::StringRef destination,
                              llvm::raw_ostream &diagnostics) {
#ifdef __linux__
  std::string sourceStorage = source.str();
  std::string destinationStorage = destination.str();
  if (::syscall(SYS_renameat2, AT_FDCWD, sourceStorage.c_str(), AT_FDCWD,
                destinationStorage.c_str(), RENAME_NOREPLACE) == 0)
    return false;

  int errorNumber = errno;
  if (errorNumber == EEXIST)
    return reject(diagnostics,
                  "output program directory appeared before rename; "
                  "refusing to replace it");
  return reject(
      diagnostics,
      "failed to rename output program directory "
      "without replacement: " +
          std::error_code(errorNumber, std::generic_category()).message());
#else
  (void)source;
  (void)destination;
  return reject(diagnostics,
                "no-replace directory rename is unsupported on this host");
#endif
}

} // namespace wafer::compiler::detail
