//===- NumericDependencyManifest.cpp - Numeric dependency manifest parsing ===//

#include "NumericDependencyConformanceInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"

#include <cctype>
#include <optional>
#include <string>
#include <vector>

namespace wafer::numeric_dependency_conformance_internal {

bool isLowerSHA256(llvm::StringRef digest) {
  return digest.size() == 64 && llvm::all_of(digest, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

llvm::Error requireExactKeys(const llvm::json::Object &object,
                             std::initializer_list<llvm::StringRef> expected,
                             const llvm::Twine &label) {
  llvm::StringSet<> allowed;
  for (llvm::StringRef key : expected)
    allowed.insert(key);
  for (const auto &entry : object)
    if (!allowed.contains(entry.first))
      return invalid(ErrorCode::UnknownField,
                     label + " has unknown field '" + entry.first.str() + "'");
  for (llvm::StringRef key : expected)
    if (!object.get(key))
      return invalid(ErrorCode::MissingField,
                     label + " is missing field '" + key + "'");
  return llvm::Error::success();
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Value &value, const llvm::Twine &label) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid(ErrorCode::TypeMismatch, label + " must be an object");
  return object;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Value &value, const llvm::Twine &label) {
  const llvm::json::Array *array = value.getAsArray();
  if (!array)
    return invalid(ErrorCode::TypeMismatch, label + " must be an array");
  return array;
}

llvm::Expected<std::string> requireString(const llvm::json::Value &value,
                                          const llvm::Twine &label) {
  std::optional<llvm::StringRef> string = value.getAsString();
  if (!string)
    return invalid(ErrorCode::TypeMismatch, label + " must be a string");
  return string->str();
}

llvm::Expected<uint64_t> requireUnsigned(const llvm::json::Value &value,
                                         const llvm::Twine &label) {
  std::optional<int64_t> integer = value.getAsInteger();
  if (!integer || *integer < 0)
    return invalid(ErrorCode::TypeMismatch,
                   label + " must be a non-negative integer");
  return static_cast<uint64_t>(*integer);
}

llvm::Expected<std::vector<std::string>>
requireStringArray(const llvm::json::Value &value, const llvm::Twine &label) {
  llvm::Expected<const llvm::json::Array *> array = requireArray(value, label);
  if (!array)
    return array.takeError();
  std::vector<std::string> result;
  result.reserve((*array)->size());
  for (size_t index = 0; index < (*array)->size(); ++index) {
    llvm::Expected<std::string> item =
        requireString((**array)[index], label + " item");
    if (!item)
      return item.takeError();
    result.push_back(std::move(*item));
  }
  return result;
}

llvm::Expected<std::optional<std::string>>
requireNullableString(const llvm::json::Value &value,
                      const llvm::Twine &label) {
  if (value.getAsNull())
    return std::optional<std::string>();
  llvm::Expected<std::string> string = requireString(value, label);
  if (!string)
    return string.takeError();
  return std::optional<std::string>(std::move(*string));
}

llvm::Error expectString(llvm::StringRef actual, llvm::StringRef expected,
                         const llvm::Twine &label) {
  if (actual != expected)
    return invalid(ErrorCode::PolicyMismatch,
                   label + " does not match frozen policy");
  return llvm::Error::success();
}

llvm::Error expectStringArray(llvm::ArrayRef<std::string> actual,
                              std::initializer_list<llvm::StringRef> expected,
                              const llvm::Twine &label) {
  if (actual.size() != expected.size())
    return invalid(ErrorCode::PolicyMismatch,
                   label + " does not match frozen policy");
  size_t index = 0;
  for (llvm::StringRef item : expected)
    if (actual[index++] != item)
      return invalid(ErrorCode::PolicyMismatch,
                     label + " does not match frozen policy");
  return llvm::Error::success();
}

/// LLVM's JSON object intentionally keeps the last value for a repeated key.
/// The managed record format is fail-closed, so scan the already
/// syntax-validated document and reject duplicate decoded keys at every depth.
class DuplicateKeyScanner {
public:
  explicit DuplicateKeyScanner(llvm::StringRef input) : input(input) {}

  llvm::Error scan() {
    if (llvm::Error error = parseValue(/*depth=*/0))
      return error;
    skipWhitespace();
    if (offset != input.size())
      return invalid(ErrorCode::JSONSyntax,
                     "record contains trailing JSON data");
    return llvm::Error::success();
  }

private:
  void skipWhitespace() {
    while (offset < input.size() &&
           std::isspace(static_cast<unsigned char>(input[offset])))
      ++offset;
  }

  llvm::Error expect(char character) {
    skipWhitespace();
    if (offset >= input.size() || input[offset] != character)
      return invalid(ErrorCode::JSONSyntax, "record JSON token scan failed");
    ++offset;
    return llvm::Error::success();
  }

  llvm::Expected<std::string> parseString() {
    skipWhitespace();
    if (offset >= input.size() || input[offset] != '"')
      return invalid(ErrorCode::JSONSyntax, "record JSON string scan failed");
    const size_t start = offset++;
    bool escaped = false;
    while (offset < input.size()) {
      const char character = input[offset++];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (character == '\\') {
        escaped = true;
        continue;
      }
      if (character == '"') {
        llvm::Expected<llvm::json::Value> decoded =
            llvm::json::parse(input.slice(start, offset));
        if (!decoded) {
          llvm::consumeError(decoded.takeError());
          return invalid(ErrorCode::JSONSyntax,
                         "record JSON string cannot be decoded");
        }
        std::optional<llvm::StringRef> value = decoded->getAsString();
        if (!value)
          return invalid(ErrorCode::JSONSyntax,
                         "record JSON key is not a string");
        return value->str();
      }
    }
    return invalid(ErrorCode::JSONSyntax, "record JSON string is unterminated");
  }

  llvm::Error parseObject(unsigned depth) {
    if (llvm::Error error = expect('{'))
      return error;
    llvm::StringSet<> keys;
    skipWhitespace();
    if (offset < input.size() && input[offset] == '}') {
      ++offset;
      return llvm::Error::success();
    }
    while (true) {
      llvm::Expected<std::string> key = parseString();
      if (!key)
        return key.takeError();
      if (!keys.insert(*key).second)
        return invalid(ErrorCode::DuplicateField,
                       "record contains duplicate field '" + *key + "'");
      if (llvm::Error error = expect(':'))
        return error;
      if (llvm::Error error = parseValue(depth + 1))
        return error;
      skipWhitespace();
      if (offset < input.size() && input[offset] == '}') {
        ++offset;
        return llvm::Error::success();
      }
      if (llvm::Error error = expect(','))
        return error;
    }
  }

  llvm::Error parseArray(unsigned depth) {
    if (llvm::Error error = expect('['))
      return error;
    skipWhitespace();
    if (offset < input.size() && input[offset] == ']') {
      ++offset;
      return llvm::Error::success();
    }
    while (true) {
      if (llvm::Error error = parseValue(depth + 1))
        return error;
      skipWhitespace();
      if (offset < input.size() && input[offset] == ']') {
        ++offset;
        return llvm::Error::success();
      }
      if (llvm::Error error = expect(','))
        return error;
    }
  }

  llvm::Error parseValue(unsigned depth) {
    if (depth > 128)
      return invalid(ErrorCode::ResourceLimit,
                     "record JSON nesting exceeds limit");
    skipWhitespace();
    if (offset >= input.size())
      return invalid(ErrorCode::JSONSyntax, "record JSON value is missing");
    if (input[offset] == '{')
      return parseObject(depth);
    if (input[offset] == '[')
      return parseArray(depth);
    if (input[offset] == '"') {
      llvm::Expected<std::string> ignored = parseString();
      if (!ignored)
        return ignored.takeError();
      return llvm::Error::success();
    }

    const size_t start = offset;
    while (offset < input.size() && input[offset] != ',' &&
           input[offset] != '}' && input[offset] != ']' &&
           !std::isspace(static_cast<unsigned char>(input[offset])))
      ++offset;
    if (offset == start)
      return invalid(ErrorCode::JSONSyntax,
                     "record JSON primitive scan failed");
    return llvm::Error::success();
  }

  llvm::StringRef input;
  size_t offset = 0;
};

llvm::Error scanDuplicateJSONKeys(llvm::StringRef input) {
  DuplicateKeyScanner scanner(input);
  return scanner.scan();
}

} // namespace wafer::numeric_dependency_conformance_internal
