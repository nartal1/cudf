/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "VariantUtilsNative.hpp"
#include "variant_path.hpp"
#include "variant_reader.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cudf::jni {
namespace {
namespace reader = cudf::io::parquet::experimental::detail::variant_reader;
using op_status = reader::op_status;
using basic_type = reader::basic_type;
using primitive_type = reader::primitive_type;

// Strict standard UTF8, including embedded NUL, excluding overlong encodings,
// surrogate codepoints and values above U+10FFFF. Never reads past the span.
__device__ bool next_codepoint(device_span<uint8_t const> bytes,
                               std::size_t& position,
                               uint32_t& codepoint)
{
  if (position == bytes.size()) { return false; }
  auto const lead = bytes[position++];
  if (lead < 0x80) {
    codepoint = lead;
    return true;
  }
  int count;
  uint32_t minimum;
  if (lead >= 0xC2 && lead <= 0xDF) {
    count = 1;
    codepoint = lead & 0x1F;
    minimum = 0x80;
  } else if (lead >= 0xE0 && lead <= 0xEF) {
    count = 2;
    codepoint = lead & 0x0F;
    minimum = 0x800;
  } else if (lead >= 0xF0 && lead <= 0xF4) {
    count = 3;
    codepoint = lead & 0x07;
    minimum = 0x10000;
  } else {
    return false;
  }
  if (static_cast<std::size_t>(count) > bytes.size() - position) { return false; }
  for (int index = 0; index < count; ++index) {
    auto const byte = bytes[position++];
    if ((byte & 0xC0) != 0x80) { return false; }
    codepoint = (codepoint << 6) | (byte & 0x3F);
  }
  return codepoint >= minimum && codepoint <= 0x10FFFF &&
         !(codepoint >= 0xD800 && codepoint <= 0xDFFF);
}

__device__ bool valid_utf8(device_span<uint8_t const> bytes)
{
  std::size_t position = 0;
  uint32_t codepoint;
  while (position < bytes.size()) {
    if (!next_codepoint(bytes, position, codepoint)) { return false; }
  }
  return true;
}

__device__ device_span<uint8_t const> string_bytes(cudf::string_view name)
{
  return {reinterpret_cast<uint8_t const*>(name.data()),
          static_cast<std::size_t>(name.size_bytes())};
}

struct utf16_units {
  device_span<uint8_t const> bytes;
  std::size_t position{0};
  uint16_t pending{0};
  bool has_pending{false};

  __device__ bool done() const { return position == bytes.size() && !has_pending; }

  __device__ bool next(uint16_t& unit)
  {
    if (has_pending) {
      unit = pending;
      has_pending = false;
      return true;
    }
    uint32_t codepoint;
    if (!next_codepoint(bytes, position, codepoint)) { return false; }
    if (codepoint <= 0xFFFF) {
      unit = static_cast<uint16_t>(codepoint);
    } else {
      codepoint -= 0x10000;
      unit = static_cast<uint16_t>(0xD800 + (codepoint >> 10));
      pending = static_cast<uint16_t>(0xDC00 + (codepoint & 0x3FF));
      has_pending = true;
    }
    return true;
  }
};

// Java String.compareTo compares UTF16 code units, unlike native UTF8 order.
__device__ bool utf16_less(cudf::string_view left, cudf::string_view right)
{
  utf16_units a{string_bytes(left)};
  utf16_units b{string_bytes(right)};
  while (!a.done() && !b.done()) {
    uint16_t x, y;
    if (!a.next(x) || !b.next(y)) { return false; }
    if (x != y) { return x < y; }
  }
  return a.done() && !b.done();
}

__device__ bool validate_metadata(device_span<uint8_t const> bytes,
                                  reader::metadata_dictionary& dictionary)
{
  auto const [parsed, status] = reader::parse_metadata_dictionary(bytes);
  if (status != op_status::SUCCESS || (bytes[0] & 0x20) != 0) { return false; }
  dictionary = parsed;
  if (static_cast<std::size_t>(parsed.strings_base) + parsed.strings_declared != bytes.size()) {
    return false;
  }
  // No dictionary ordering assumption: Spark assigns IDs in insertion order.
  for (size_type id = 0; id < parsed.num_entries; ++id) {
    auto const name = reader::name_for_id(parsed, bytes, id);
    if (!name.has_value() || !valid_utf8(string_bytes(*name))) { return false; }
  }
  return true;
}

// The unchanged SDK length reader narrows table positions before loading the
// sentinel. Prove its complete table fits first, using only bounded uint64 math,
// so even hostile counts can never reach a signed pos + width overflow.
__device__ bool safe_length_input(device_span<uint8_t const> bytes)
{
  if (bytes.empty()) { return false; }
  auto const type = reader::decode_basic_type(bytes[0]);
  if (type != basic_type::OBJECT && type != basic_type::ARRAY) { return true; }
  auto const widths = reader::decode_object_array_header(
    reader::variant_value_header(bytes[0]), type == basic_type::OBJECT);
  auto const count = reader::read_uint64(bytes, 1, widths.num_elements_size);
  if (!count.has_value()) { return false; }
  auto const table_end = uint64_t{1} + widths.num_elements_size +
    *count * widths.field_id_size + (*count + 1) * widths.field_offset_size;
  return table_end <= bytes.size();
}

__device__ bool validate_node(device_span<uint8_t const> bytes)
{
  if (bytes.empty()) { return false; }
  auto const type = reader::decode_basic_type(bytes[0]);
  auto const header = reader::variant_value_header(bytes[0]);
  if (type == basic_type::PRIMITIVE) {
    // Common Spark400/411/420 IDs. Core IDs17..19 are newer physical types,
    // and must not be silently classified as scalar-missing on Spark400.
    if (!(header <= 16 || header == 20)) { return false; }
  } else if (type == basic_type::OBJECT) {
    if ((header & 0x20) != 0) { return false; }
  } else if (type == basic_type::ARRAY) {
    if ((header & 0x38) != 0) { return false; }
  }
  if (!safe_length_input(bytes)) { return false; }
  auto const length = reader::variant_value_length(bytes);
  return length.has_value() && *length == bytes.size();
}

__device__ bool validate_object(reader::metadata_dictionary const& dictionary,
                                device_span<uint8_t const> metadata,
                                device_span<uint8_t const> value,
                                cudf::string_view key)
{
  auto const widths = reader::decode_object_array_header(
    reader::variant_value_header(value[0]), true);
  auto const count = reader::read_uint64(value, 1, widths.num_elements_size);
  if (!count.has_value()) { return false; }
  auto const ids_start = uint64_t{1} + widths.num_elements_size;
  auto const offsets_start = ids_start + *count * widths.field_id_size;
  auto const values_base = offsets_start + (*count + 1) * widths.field_offset_size;
  // validate_node bounded the complete table and sentinel in 64-bit arithmetic.
  if (values_base > value.size()) { return false; }
  auto const declared = value.size() - values_base;
  cudf::string_view previous;
  for (uint64_t index = 0; index < *count; ++index) {
    auto const id = reader::narrow_cast(reader::read_uint64(
      value, static_cast<size_type>(ids_start + index * widths.field_id_size), widths.field_id_size));
    if (!id.has_value()) { return false; }
    auto const name = reader::name_for_id(dictionary, metadata, *id);
    if (!name.has_value()) { return false; }
    if (index != 0 && (previous.compare(*name) >= 0 || !utf16_less(previous, *name))) {
      return false;
    }
    previous = *name;
    auto const offset = reader::read_uint64(value,
      static_cast<size_type>(offsets_start + index * widths.field_offset_size),
      widths.field_offset_size);
    if (!offset.has_value() || *offset > declared) { return false; }
    if (name->compare(key) == 0 &&
        !safe_length_input(value.subspan(values_base + *offset))) {
      return false;
    }
  }
  // Object values may be physically out of key order. Only selected spans are
  // inspected, exactly as both readers do; this is not full-payload validation.
  return true;
}

__device__ bool validate_array(device_span<uint8_t const> value)
{
  auto const widths = reader::decode_object_array_header(
    reader::variant_value_header(value[0]), false);
  auto const count = reader::read_uint64(value, 1, widths.num_elements_size);
  if (!count.has_value()) { return false; }
  auto const offsets_start = uint64_t{1} + widths.num_elements_size;
  auto const values_base = offsets_start + (*count + 1) * widths.field_offset_size;
  if (values_base > value.size()) { return false; }
  auto const declared = value.size() - values_base;
  uint64_t previous = 0;
  for (uint64_t index = 0; index <= *count; ++index) {
    auto const offset = reader::read_uint64(value,
      static_cast<size_type>(offsets_start + index * widths.field_offset_size),
      widths.field_offset_size);
    if (!offset.has_value() || *offset < previous || *offset > declared ||
        (index == 0 && *offset != 0)) {
      return false;
    }
    previous = *offset;
  }
  return true;
}

__device__ bool exact_terminal(device_span<uint8_t const> value)
{
  auto const type = reader::decode_basic_type(value[0]);
  auto const header = reader::variant_value_header(value[0]);
  if (type == basic_type::SHORT_STRING) { return valid_utf8(value.subspan(1)); }
  if (type != basic_type::PRIMITIVE) { return false; }
  if (header == static_cast<uint8_t>(primitive_type::LONG_STRING)) {
    return valid_utf8(value.subspan(5));
  }
  return (header >= static_cast<uint8_t>(primitive_type::BOOLEAN_TRUE) &&
          header <= static_cast<uint8_t>(primitive_type::FLOAT64)) ||
         header == static_cast<uint8_t>(primitive_type::FLOAT32);
}

__device__ cuda::std::optional<op_status> certified_outcome(
  device_span<uint8_t const> metadata,
  device_span<uint8_t const> value,
  column_device_view path)
{
  reader::metadata_dictionary dictionary;
  if (!validate_metadata(metadata, dictionary)) { return cuda::std::nullopt; }
  auto current = value;
  for (size_type index = 0; index < path.size(); ++index) {
    if (!validate_node(current)) { return cuda::std::nullopt; }
    auto const step = path.element<cudf::string_view>(index);
    auto const type = reader::decode_basic_type(current[0]);
    cuda::std::pair<device_span<uint8_t const>, op_status> next;
    if (step.data()[0] == '[') {
      if (type != basic_type::ARRAY) { return op_status::MISSING_PATH; }
      if (!validate_array(current)) { return cuda::std::nullopt; }
      auto const position = reader::parse_index_step(step);
      if (!position.has_value()) { return cuda::std::nullopt; }
      next = reader::locate_array_element(current, *position);
    } else {
      if (type != basic_type::OBJECT) { return op_status::MISSING_PATH; }
      if (!validate_object(dictionary, metadata, current, step)) { return cuda::std::nullopt; }
      next = reader::locate_object_field(dictionary, metadata, current, step);
    }
    if (next.second == op_status::MISSING_PATH) { return op_status::MISSING_PATH; }
    if (next.second != op_status::SUCCESS) { return cuda::std::nullopt; }
    current = next.first;
    if (!validate_node(current)) { return cuda::std::nullopt; }
    if (index + 1 < path.size() && reader::is_variant_null(current)) {
      return op_status::MISSING_PATH;
    }
  }
  if (reader::is_variant_null(current)) { return op_status::VARIANT_NULL; }
  if (!exact_terminal(current)) { return cuda::std::nullopt; }
  return op_status::SUCCESS;
}

__device__ cuda::std::optional<device_span<uint8_t const>> row_bytes(
  lists_column_device_view const& values, size_type row, size_type limit)
{
  // Spark binary reads must not silently consume nullable byte-child poison.
  // A mask is conservatively UNKNOWN, even if every child bit happens to be valid.
  if (values.is_null(row) || values.child().nullable()) { return cuda::std::nullopt; }
  auto const begin = values.offset_at(row);
  auto const end = values.offset_at(row + 1);
  if (begin < 0 || end <= begin || end > values.child().size() || end - begin > limit) {
    return cuda::std::nullopt;
  }
  return device_span<uint8_t const>{values.child().data<uint8_t>() + begin,
                                   static_cast<std::size_t>(end - begin)};
}

bool supported_path(std::string_view path, std::vector<std::string>& steps)
{
  if (path.size() < 3 || path[0] != '$' || (path[1] != '.' && path[1] != '[')) {
    return false;
  }
  try {
    steps = cudf::io::parquet::experimental::detail::parse_variant_path_steps(path);
  } catch (std::invalid_argument const&) {
    return false;
  }
  auto identifier_start = [](char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
  };
  for (auto const& step : steps) {
    if (step.empty()) { return false; }
    if (step[0] == '[') { continue; }  // Shared parser already checked all digits/Int bounds.
    if (!identifier_start(step[0])) { return false; }
    for (auto character : step) {
      if (!identifier_start(character) && !(character >= '0' && character <= '9')) {
        return false;
      }
    }
  }
  return !steps.empty();
}

std::unique_ptr<column> path_column(std::vector<std::string> const& steps,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  std::string characters;
  std::vector<size_type> offsets(steps.size() + 1);
  for (std::size_t index = 0; index < steps.size(); ++index) {
    CUDF_EXPECTS(characters.size() <= std::numeric_limits<size_type>::max(),
                 "Variant path is too large", std::invalid_argument);
    offsets[index] = static_cast<size_type>(characters.size());
    characters.append(steps[index]);
  }
  CUDF_EXPECTS(characters.size() <= std::numeric_limits<size_type>::max() &&
                 offsets.size() <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "Variant path is too large", std::invalid_argument);
  offsets.back() = static_cast<size_type>(characters.size());
  auto device_offsets = cudf::detail::make_device_uvector_async(offsets, stream, mr);
  auto offsets_column = std::make_unique<column>(data_type{type_id::INT32},
    static_cast<size_type>(offsets.size()), device_offsets.release(),
    cudf::create_null_mask(0, mask_state::UNALLOCATED, stream, mr), 0);
  auto device_characters = cudf::detail::make_device_uvector(
    host_span<char const>{characters.data(), characters.size()}, stream, mr);
  return cudf::make_strings_column(static_cast<size_type>(steps.size()),
    std::move(offsets_column), device_characters.release(), 0,
    cudf::create_null_mask(0, mask_state::UNALLOCATED, stream, mr));
}
}  // namespace

bool all_variant_rows_covered_with_path_validation(
  column_view const& variants,
  std::string_view path,
  column_view const& extraction,
  std::vector<column_view> const& decoded,
  size_type limit,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(variants.type().id() == type_id::STRUCT && variants.num_children() == 2,
               "Variant must be STRUCT(metadata, value)", std::invalid_argument);
  CUDF_EXPECTS(limit > 0 && !decoded.empty(),
               "Variant size limit and decoder count must be positive", std::invalid_argument);
  for (auto const& child : {variants.child(0), variants.child(1)}) {
    CUDF_EXPECTS(child.type().id() == type_id::LIST &&
                   lists_column_view{child}.child().type().id() == type_id::UINT8,
                 "Variant children must be LIST<UINT8>", std::invalid_argument);
    CUDF_EXPECTS(static_cast<int64_t>(variants.offset()) + variants.size() <= child.size(),
                 "Variant children do not cover the parent view", std::invalid_argument);
  }
  std::vector<column_view> statuses{extraction};
  statuses.insert(statuses.end(), decoded.begin(), decoded.end());
  for (auto const& status : statuses) {
    CUDF_EXPECTS(status.type().id() == type_id::UINT8 && !status.nullable() &&
                   status.size() == variants.size(), "Invalid Variant status", std::invalid_argument);
  }
  std::vector<std::string> steps;
  if (!supported_path(path, steps)) { return false; }
  if (variants.size() == 0) { return true; }
  auto path_storage = path_column(steps, stream, mr);
  auto path_device = column_device_view::create(path_storage->view(), stream);
  structs_column_view raw{variants};
  std::vector<column_view> columns{variants, raw.get_sliced_child(0, stream),
                                 raw.get_sliced_child(1, stream)};
  columns.insert(columns.end(), statuses.begin(), statuses.end());
  auto device = table_device_view::create(table_view{columns}, stream);
  auto const view = *device;
  auto const path_view = *path_device;
  auto const begin = thrust::make_counting_iterator<size_type>(0);
  return thrust::transform_reduce(rmm::exec_policy(stream, mr), begin, begin + variants.size(),
    [view, path_view, limit] __device__(size_type row) -> bool {
      auto const original = static_cast<op_status>(view.column(3).element<uint8_t>(row));
      bool success = false;
      for (size_type index = 4; index < view.num_columns(); ++index) {
        auto const status = static_cast<op_status>(view.column(index).element<uint8_t>(row));
        if (original == op_status::SUCCESS) {
          if (status != op_status::SUCCESS && status != op_status::TYPE_MISMATCH) { return false; }
          success |= status == op_status::SUCCESS;
        } else if (status != original) {
          return false;
        }
      }
      if (view.column(0).is_null(row)) { return original == op_status::ROW_NULL; }
      if (original != op_status::MISSING_PATH && original != op_status::VARIANT_NULL &&
          !(original == op_status::SUCCESS && success)) {
        return false;
      }
      auto const metadata = row_bytes(lists_column_device_view{view.column(1)}, row, limit);
      auto const value = row_bytes(lists_column_device_view{view.column(2)}, row, limit);
      if (!metadata.has_value() || !value.has_value()) { return false; }
      auto const outcome = certified_outcome(*metadata, *value, path_view);
      return outcome.has_value() && *outcome == original;
    }, true, cuda::std::logical_and<bool>{});
}
}  // namespace cudf::jni
