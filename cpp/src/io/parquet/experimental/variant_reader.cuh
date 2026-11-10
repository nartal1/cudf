/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Shared private encoding/path readers. Function bodies are retained from d984f0a2;
// additional Spark-equivalence validation belongs in the JNI certificate consumer.
#include <cudf/column/column_device_view.cuh>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/span.hpp>

#include <cuda/numeric>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/utility>

#include <cstdint>

namespace cudf::io::parquet::experimental::detail::variant_reader {

constexpr int variant_version_v1 = 1;

// Bytes consumed by the leading metadata byte common to every Variant value.
constexpr size_type variant_header_bytes = 1;

// Low 2 bits of a value's metadata byte: the basic type.
using basic_type = variant_basic_type;

// For a primitive value, the value_header is the physical type id of the payload.
using primitive_type = variant_primitive_type;

// The status of a VARIANT operation.
using op_status = variant_operation_status;

__device__ inline cuda::std::optional<uint64_t> read_uint64(device_span<uint8_t const> data,
                                                     size_type pos,
                                                     int width)
{
  if (cuda::std::cmp_greater(pos + width, data.size())) { return cuda::std::nullopt; }
  uint64_t v = 0;
  cuda::std::memcpy(&v, data.data() + pos, width);
  return v;
}

__device__ inline cuda::std::optional<size_type> narrow_cast(cuda::std::optional<uint64_t> value)
{
  if (!value.has_value() ||
      cuda::std::cmp_greater(value.value(), cuda::std::numeric_limits<size_type>::max())) {
    return cuda::std::nullopt;
  }
  return static_cast<size_type>(value.value());
}

__device__ inline basic_type decode_basic_type(uint8_t value_metadata)
{
  return static_cast<basic_type>(value_metadata & 0x03);
}

__device__ inline uint8_t variant_value_header(uint8_t value_metadata)
{
  return (value_metadata >> 2) & 0x3F;
}

struct object_array_header {
  int field_offset_size;  // bytes per field_offset entry
  int field_id_size;      // bytes per field_id (0 for arrays)
  int num_elements_size;  // bytes holding num_elements
};

/**
 * @brief Decode the size fields packed into an object/array value header.
 *
 * For object and array values, the 6-bit value header (bits 2..7 of the value metadata byte)
 * encodes the widths used by the rest of the value. The layout differs between the two:
 *
 *   object value_header bits:  | is_large (1) | field_id_size-1 (2) | field_offset_size-1 (2) |
 *   array  value_header bits:  |          is_large (1)             | field_offset_size-1 (2) |
 *
 * where each `*_size-1` field stores (width in bytes - 1), so the decoded width is the field + 1
 * (1..4 bytes), and `is_large` selects the width of the `num_elements` field: 4 bytes if set,
 * else 1 byte. Arrays have no field ids, so `field_id_size` is 0.
 *
 * @param value_header The 6-bit value header (see variant_value_header)
 * @param is_object True for object values, false for array values
 * @return The decoded byte widths
 */
__device__ inline object_array_header decode_object_array_header(uint8_t value_header, bool is_object)
{
  auto const large_bit = is_object ? 4 : 2;
  bool const is_large  = (value_header >> large_bit) & 0x01;

  return {.field_offset_size = (value_header & 0x03) + 1,
          .field_id_size     = is_object ? ((value_header >> 2) & 0x03) + 1 : 0,
          .num_elements_size = is_large ? 4 : 1};
}

/**
 * @brief Compute the total encoded byte length of a single VARIANT value.
 *
 * Every value starts with a 1-byte value metadata header (`basic_type` in bits 0..1, `value_header`
 * in bits 2..7); the bytes that follow depend on the basic type:
 *
 *   - primitive (0): header + a fixed payload keyed by the primitive type id. Binary/long_string
 *     carry a 4-byte little-endian length prefix followed by that many payload bytes.
 *   - short_string (1): header + `value_header` payload bytes (the header is the string length).
 *   - object/array (2/3): header + num_elements + field-id list + field-offset list + values; the
 *     total values-region size is read from the trailing field_offset (the "sentinel" at index
 *     num_elements). See decode_object_array_header / locate_object_field for the sub-layout.
 *
 * @param enc The encoded value bytes (must begin at the value metadata byte)
 * @return The total length in bytes of the value, or nullopt if `enc` is empty/malformed or the
 *         type id is unrecognized
 */
__device__ inline cuda::std::optional<uint64_t> variant_value_length(device_span<uint8_t const> enc)
{
  if (enc.size() < 1) { return cuda::std::nullopt; }
  auto const value_metadata = enc[0];
  auto const btype          = decode_basic_type(value_metadata);
  auto const value_header   = variant_value_header(value_metadata);

  if (btype == basic_type::PRIMITIVE) {
    uint64_t payload = 0;
    switch (static_cast<primitive_type>(value_header)) {
      case primitive_type::NULLVAL:
      case primitive_type::BOOLEAN_TRUE:
      case primitive_type::BOOLEAN_FALSE: break;  // no payload
      case primitive_type::INT8: payload = 1; break;
      case primitive_type::INT16: payload = 2; break;
      case primitive_type::INT32:
      case primitive_type::DATE:
      case primitive_type::FLOAT32: payload = 4; break;
      case primitive_type::INT64:
      case primitive_type::FLOAT64:
      case primitive_type::TIMESTAMP_MICROS:
      case primitive_type::TIMESTAMP_NTZ_MICROS:
      case primitive_type::TIME_NTZ_MICROS:
      case primitive_type::TIMESTAMP_NANOS:
      case primitive_type::TIMESTAMP_NTZ_NANOS: payload = 8; break;
      case primitive_type::DECIMAL4: payload = 1 + 4; break;    // scale + int32
      case primitive_type::DECIMAL8: payload = 1 + 8; break;    // scale + int64
      case primitive_type::DECIMAL16: payload = 1 + 16; break;  // scale + int128
      case primitive_type::UUID: payload = 16; break;
      case primitive_type::BINARY:
      case primitive_type::LONG_STRING: {
        constexpr int length_prefix_bytes = 4;
        auto const len = read_uint64(enc, variant_header_bytes, length_prefix_bytes);
        if (!len.has_value()) { return cuda::std::nullopt; }
        payload = length_prefix_bytes + len.value();
        break;
      }
      default: return cuda::std::nullopt;
    }
    return variant_header_bytes + payload;
  }

  if (btype == basic_type::SHORT_STRING) {
    // The value header is the payload length, following the header byte.
    return variant_header_bytes + static_cast<uint64_t>(value_header);
  }

  // Object / array: the encoded size is the header bytes (metadata byte, element count, optional
  // field-id list, and offset list)
  bool const is_object = btype == basic_type::OBJECT;
  auto const [offset_size, id_size, num_elements_size] =
    decode_object_array_header(value_header, is_object);

  auto const num_elements = read_uint64(enc, variant_header_bytes, num_elements_size);
  if (!num_elements.has_value()) { return cuda::std::nullopt; }
  auto const n = num_elements.value();

  auto const offsets_start = variant_header_bytes + num_elements_size + n * id_size;
  auto const values_base   = offsets_start + (n + 1) * offset_size;
  // Sentinel offset (entry n) holds the total size of the values region.
  auto const sentinel_pos = narrow_cast(offsets_start + n * offset_size);
  if (!sentinel_pos.has_value()) { return cuda::std::nullopt; }
  auto const sentinel = read_uint64(enc, sentinel_pos.value(), offset_size);
  if (!sentinel.has_value()) { return cuda::std::nullopt; }
  return values_base + sentinel.value();
}

// Parsed offset table of a metadata dictionary blob, produced once per row by
// `parse_metadata_dictionary` and reused for every `name_for_id` lookup along a row's path (a
// path may step through several objects, each needing id-to-name resolution against the same
// dictionary) instead of re-parsing and re-validating `meta` on every step.
struct metadata_dictionary {
  size_type offset_size;
  size_type offsets_start;
  size_type strings_base;
  size_type strings_declared;
  size_type num_entries;
};

/**
 * @brief Parse and validate a metadata dictionary blob's offset table.
 *
 * @param meta The metadata blob for this row
 * @return The parsed offset table, or a failure status if `meta` is malformed
 */
__device__ inline cuda::std::pair<metadata_dictionary, op_status> parse_metadata_dictionary(
  device_span<uint8_t const> meta)
{
  auto const meta_len = static_cast<size_type>(meta.size());
  if (meta_len < 1) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const meta_header = meta[0];
  if ((meta_header & 0x0F) != variant_version_v1) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const meta_offset_size = ((meta_header >> 6) & 0x03) + 1;

  size_type meta_pos          = 1;
  auto const num_meta_entries = narrow_cast(read_uint64(meta, meta_pos, meta_offset_size));
  if (!num_meta_entries.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
  meta_pos += meta_offset_size;

  auto const meta_offsets_start = meta_pos;
  auto const meta_offsets_bytes =
    (static_cast<uint64_t>(num_meta_entries.value()) + 1) * meta_offset_size;
  if (cuda::std::cmp_greater(meta_offsets_bytes, meta_len - meta_offsets_start)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const meta_strings_base   = meta_offsets_start + static_cast<size_type>(meta_offsets_bytes);
  auto const meta_strings_extent = meta_len - meta_strings_base;

  // Parquet VARIANT spec requires offsets[0] == 0; any other value is malformed.
  auto const first_off = read_uint64(meta, meta_offsets_start, meta_offset_size);
  if (!first_off.has_value() || first_off.value() != 0) {
    return {{}, op_status::MALFORMED_VARIANT};
  }

  // Read the terminal offset offsets[num_entries] up front: it is the authoritative size of the
  // string-data region, so every individual entry's end offset must be bounded by it below, not
  // just by the physical buffer remainder (`meta_strings_extent`) -- the same distinction the
  // object value's own sentinel/values_region check makes for field values.
  auto const terminal_off_pos =
    meta_offsets_start + static_cast<size_type>(num_meta_entries.value()) * meta_offset_size;
  auto const terminal_off = read_uint64(meta, terminal_off_pos, meta_offset_size);
  if (!terminal_off.has_value() ||
      cuda::std::cmp_greater(terminal_off.value(), meta_strings_extent)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }

  return {metadata_dictionary{.offset_size      = meta_offset_size,
                              .offsets_start    = meta_offsets_start,
                              .strings_base     = meta_strings_base,
                              .strings_declared = static_cast<size_type>(terminal_off.value()),
                              .num_entries      = num_meta_entries.value()},
          op_status::SUCCESS};
}

// O(1) name lookup by id: two offset reads into the metadata table. `field_id` may come directly
// from untrusted object data (an out-of-range dictionary index), so it is bounds checked against
// the metadata dictionary size before use, and the offset positions are computed in 64-bit
// arithmetic to avoid overflowing `size_type` for large ids.
__device__ inline cuda::std::optional<cudf::string_view> name_for_id(metadata_dictionary const& dict,
                                                              device_span<uint8_t const> meta,
                                                              size_type field_id)
{
  if (field_id < 0 || cuda::std::cmp_greater_equal(field_id, dict.num_entries)) {
    return cuda::std::nullopt;
  }
  auto const start_pos =
    static_cast<uint64_t>(dict.offsets_start) + static_cast<uint64_t>(field_id) * dict.offset_size;
  auto const end_pos = start_pos + static_cast<uint64_t>(dict.offset_size);
  if (cuda::std::cmp_greater(end_pos, meta.size())) { return cuda::std::nullopt; }
  auto const s = read_uint64(meta, static_cast<size_type>(start_pos), dict.offset_size);
  auto const e = read_uint64(meta, static_cast<size_type>(end_pos), dict.offset_size);
  if (!s.has_value() || !e.has_value()) { return cuda::std::nullopt; }
  if (e.value() < s.value() || cuda::std::cmp_greater(e.value(), dict.strings_declared)) {
    return cuda::std::nullopt;
  }
  return cudf::string_view{
    reinterpret_cast<char const*>(meta.data() + dict.strings_base + s.value()),
    static_cast<size_type>(e.value() - s.value())};
}

/**
 * @brief Locate the encoded bytes of a single field within an object value by field name.
 *
 * Object value layout, following the 1-byte value metadata header (basic_type=object in the low 2
 * bits; value_header in the high 6 bits, see decode_object_array_header):
 *
 *   bytes 1..:     num_elements   (num_elements_size bytes) = number of fields N
 *   next N*field_id_size bytes:        field_ids[0..N-1]   (sorted by field name)
 *   next (N+1)*field_offset_size bytes: field_offsets[0..N] (relative to values_base)
 *   remaining bytes (values_base..):   the concatenated field values
 *
 * `num_elements_size`, `field_id_size`, and `field_offset_size` come from the value header (see
 * decode_object_array_header). The trailing offset `field_offsets[N]` is the total size of the
 * values region.
 *
 * Per the spec, `field_ids[0..N-1]` are ordered by the corresponding field name
 * (lexicographically), not by the numeric id value, and the values themselves may be in any order,
 * so `field_offsets` are not necessarily monotonic -- hence the value length is taken from each
 * field's own header rather than from offset deltas.
 *
 * `field_ids[0..N-1]` is ordered by name (a per-object invariant, independent of whether the
 * metadata dictionary itself happens to be sorted), so it can be binary searched directly against
 * `key` without first resolving `key` to a dictionary id: each probe turns `field_ids[mid]` into
 * its dictionary name via an O(1) lookup in `meta` (the metadata offset table is indexed directly
 * by id) and compares that name against `key`, giving O(log N) with no separate name-to-id lookup
 * over the (potentially much larger) dictionary.
 *
 * Not using thrust::lower_bound since it does not propagate entry read failures.
 *
 * @param dict The already-parsed metadata dictionary offset table for `meta` (see
 *             parse_metadata_dictionary), reused across every path step so it is not re-parsed
 *             and re-validated once per object
 * @param meta The metadata blob for this row, used to resolve field ids to names
 * @param val The object value bytes
 * @param key The name of the field to locate
 * @return The encoded bytes of the field value, or an empty span if `val` is not an object, the
 *         field is absent, or either blob is malformed
 */
__device__ inline cuda::std::pair<device_span<uint8_t const>, op_status> locate_object_field(
  metadata_dictionary const& dict,
  device_span<uint8_t const> meta,
  device_span<uint8_t const> val,
  cudf::string_view key)
{
  auto const val_len = static_cast<size_type>(val.size());
  if (val_len < 1) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const value_metadata = val[0];
  if (decode_basic_type(value_metadata) != basic_type::OBJECT) {
    return {{}, op_status::MISSING_PATH};
  }

  auto const [offset_size, id_size, num_elements_size] =
    decode_object_array_header(variant_value_header(value_metadata), true);

  size_type pos         = 1;
  auto const num_fields = narrow_cast(read_uint64(val, pos, num_elements_size));
  if (!num_fields.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
  if (num_fields.value() == 0) { return {{}, op_status::MISSING_PATH}; }
  pos += num_elements_size;

  auto const ids_start = pos;
  auto const ids_bytes = static_cast<uint64_t>(num_fields.value()) * id_size;
  if (ids_bytes > val_len - ids_start) { return {{}, op_status::MALFORMED_VARIANT}; }

  auto const offsets_start = ids_start + static_cast<size_type>(ids_bytes);
  auto const offsets_bytes = (static_cast<uint64_t>(num_fields.value()) + 1) * offset_size;
  if (offsets_bytes > val_len - offsets_start) { return {{}, op_status::MALFORMED_VARIANT}; }

  auto const values_base   = offsets_start + static_cast<size_type>(offsets_bytes);
  auto const values_extent = val_len - values_base;

  // Read the sentinel (terminal offset at index num_fields) to get the authoritative end of the
  // values region.  Using the physical remainder (values_extent) would allow a malformed object
  // to reference bytes beyond the sentinel, passing validation despite corrupt data.
  auto const sentinel_raw =
    read_uint64(val, offsets_start + num_fields.value() * offset_size, offset_size);
  if (!sentinel_raw.has_value() || sentinel_raw.value() > static_cast<uint64_t>(values_extent)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const values_region = static_cast<size_type>(sentinel_raw.value());

  // Binary search field_ids[0..N-1] by resolving each probe to its name and comparing against
  // `key` directly
  bool found           = false;
  uint64_t match_start = 0;
  size_type lo         = 0;
  size_type hi         = num_fields.value();
  while (lo < hi) {
    size_type const mid = lo + (hi - lo) / 2;
    auto const probe_id = narrow_cast(read_uint64(val, ids_start + mid * id_size, id_size));
    if (!probe_id.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
    auto const probe_name = name_for_id(dict, meta, probe_id.value());
    if (!probe_name.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
    int const cmp = probe_name.value().compare(key);
    if (cmp == 0) {
      auto const match_offset = read_uint64(val, offsets_start + mid * offset_size, offset_size);
      if (!match_offset.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
      if (match_offset.value() > static_cast<uint64_t>(values_region)) {
        return {{}, op_status::MALFORMED_VARIANT};
      }
      match_start = match_offset.value();
      found       = true;
      break;
    }
    if (cmp < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (!found) { return {{}, op_status::MISSING_PATH}; }

  auto const value     = val.subspan(values_base + match_start);
  auto const value_len = variant_value_length(value);
  if (!value_len.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const match_end = match_start + value_len.value();
  if (match_end > static_cast<uint64_t>(values_region)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  return {val.subspan(values_base + match_start, value_len.value()), op_status::SUCCESS};
}

// Parse an array value header and return the sub-span of the element at `index` (0-based) within
// `val`. Returns an empty span if `val` is not an array (`basic_type != array`), if `index` is out
// of bounds, or if the encoded data is truncated.
//
// Array layout per the Variant spec:
//   byte 0: header (basic_type=array in low 2 bits; value_header in high 6 bits)
//     value_header bits: (offset_size - 1) in bits 0-1, is_large in bit 2, bits 3-5 unused
//   num_elements: 1 byte if !is_large else 4 bytes (little-endian)
//   offsets:      (num_elements + 1) entries, each `offset_size` bytes, relative to the end of
//                 offsets
//   values:       concatenated element blobs
//
// Array element offsets are monotonically increasing, so the element length is taken directly from
// the offset delta (o1 - o0) rather than from the element's own header.
__device__ inline cuda::std::pair<device_span<uint8_t const>, op_status> locate_array_element(
  device_span<uint8_t const> value, size_type index)
{
  if (index < 0) { return {{}, op_status::MISSING_PATH}; }

  auto const value_size = static_cast<size_type>(value.size());
  if (value_size < 1) { return {{}, op_status::MALFORMED_VARIANT}; }
  uint8_t const value_metadata = value[0];
  if (decode_basic_type(value_metadata) != basic_type::ARRAY) {
    return {{}, op_status::MISSING_PATH};
  }

  int const value_header = variant_value_header(value_metadata);
  [[maybe_unused]] auto const [offset_size, _, num_elements_size] =
    decode_object_array_header(value_header, false);

  size_type position            = 1;
  auto const num_elements_value = narrow_cast(read_uint64(value, position, num_elements_size));
  if (!num_elements_value.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const num_elements = num_elements_value.value();
  if (index >= num_elements) { return {{}, op_status::MISSING_PATH}; }
  position += num_elements_size;

  size_type const offsets_start = position;

  // Computed in 64-bit because (num_elements + 1) * offset_size can exceed the signed `size_type`
  // range (which would be UB); the check below then rejects any array that overruns the value blob.
  auto const offsets_bytes = (static_cast<uint64_t>(num_elements) + 1) * offset_size;
  if (cuda::std::cmp_greater(offsets_bytes, value_size - offsets_start)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  size_type const values_base = offsets_start + static_cast<size_type>(offsets_bytes);
  auto const values_extent    = value_size - values_base;
  // Read the terminal offset offsets[num_elements]; it is the spec-declared bound on the
  // values region and must be used instead of the physical extent so that an element whose
  // offset escapes the declared boundary is caught as malformed even when physical bytes
  // are present beyond it.
  auto const terminal_off_pos = offsets_start + static_cast<uint64_t>(num_elements) * offset_size;
  auto const terminal_off     = read_uint64(value, terminal_off_pos, offset_size);
  if (!terminal_off.has_value() || cuda::std::cmp_greater(*terminal_off, values_extent)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  // The spec requires offsets[0] == 0; a nonzero first offset silently skips leading
  // value bytes and can return a plausible result from a malformed array.
  auto const first_off = read_uint64(value, offsets_start, offset_size);
  if (!first_off.has_value() || *first_off != 0) { return {{}, op_status::MALFORMED_VARIANT}; }

  auto const start_offset_pos = offsets_start + static_cast<uint64_t>(index) * offset_size;
  auto const end_offset_pos   = offsets_start + (static_cast<uint64_t>(index) + 1) * offset_size;
  if (cuda::std::cmp_greater(end_offset_pos + offset_size, value_size)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const start_offset = read_uint64(value, start_offset_pos, offset_size);
  auto const end_offset   = read_uint64(value, end_offset_pos, offset_size);
  if (!start_offset.has_value() || !end_offset.has_value()) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const element_start = *start_offset;
  auto const element_end   = *end_offset;
  if (element_end < element_start || cuda::std::cmp_greater(element_end, *terminal_off)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  return {value.subspan(values_base + element_start, element_end - element_start),
          op_status::SUCCESS};
}

__device__ inline bool is_variant_null(device_span<uint8_t const> enc)
{
  if (enc.empty()) { return false; }
  auto const vm = enc[0];
  return decode_basic_type(vm) == basic_type::PRIMITIVE &&
         variant_value_header(vm) == static_cast<uint8_t>(primitive_type::NULLVAL);
}

// Parse an array-index step token of the form "[<N>]" into its zero-based index. Returns nullopt
// for any malformed token or an index that does not fit in `size_type` (such an index is out of
// range for any array, so the caller treats it as a missing element).
__device__ inline cuda::std::optional<size_type> parse_index_step(cudf::string_view step)
{
  auto const step_size  = step.size_bytes();
  auto const* step_data = step.data();
  if (step_size < 3 || step_data[0] != '[' || step_data[step_size - 1] != ']') {
    return cuda::std::nullopt;
  }

  // Accumulate directly in `size_type`; the checked-arithmetic helpers reject the token if the
  // running value overflows, which means the index is out of range for any array and the caller
  // treats it as a missing element.
  size_type index = 0;
  for (size_type k = 1; k < step_size - 1; ++k) {
    char const c = step_data[k];
    if (c < '0' || c > '9') { return cuda::std::nullopt; }
    if (cuda::mul_overflow(index, index, size_type{10}) ||
        cuda::add_overflow(index, index, static_cast<size_type>(c - '0'))) {
      return cuda::std::nullopt;
    }
  }
  return index;
}

// Walk a path of object-key or array-index steps level by level starting at `val` and return
// the span of the final value (subspan of `val`). Returns an empty span on failure.
//
// Each path step is encoded in the `path` strings column as either:
//   - "<name>"  -> descend into an object by dictionary key, or
//   - "[<N>]"   -> descend into an array by zero-based integer index.
// The step kind is inferred from the first byte (`'['` means index).
__device__ inline cuda::std::pair<device_span<uint8_t const>, op_status> resolve_path(
  device_span<uint8_t const> meta, device_span<uint8_t const> val, column_device_view path)
{
  device_span<uint8_t const> sub_val = val;
  // The metadata dictionary is a per-row property shared by every object-key step of the path, so
  // it is parsed and validated at most once per row (lazily, on the first object-key step) rather
  // than being re-parsed on each step -- a path with only array-index steps never touches it.
  cuda::std::optional<metadata_dictionary> dict;
  for (size_type i = 0; i < path.size(); ++i) {
    auto const step = path.element<cudf::string_view>(i);
    if (step.size_bytes() >= 1 && step.data()[0] == '[') {
      auto const index = parse_index_step(step);
      if (!index.has_value()) { return {{}, op_status::MISSING_PATH}; }
      auto const [span, st] = locate_array_element(sub_val, index.value());
      if (st != op_status::SUCCESS) { return {{}, st}; }
      sub_val = span;
    } else {
      if (!dict.has_value()) {
        auto [parsed, st] = parse_metadata_dictionary(meta);
        if (st != op_status::SUCCESS) { return {{}, st}; }
        dict = parsed;
      }
      auto const [span, st] = locate_object_field(*dict, meta, sub_val, step);
      if (st != op_status::SUCCESS) { return {{}, st}; }
      sub_val = span;
    }

    // VARIANT null before the end of the path is missing_path per spec.
    if (i + 1 < path.size() && is_variant_null(sub_val)) { return {{}, op_status::MISSING_PATH}; }
    // A zero-length resolved value is not decodable; the value-only path drops the row.
    if (sub_val.empty()) { return {{}, op_status::MALFORMED_VARIANT}; }
  }

  // Terminal VARIANT null: return the bytes with variant_null status.
  if (is_variant_null(sub_val)) { return {sub_val, op_status::VARIANT_NULL}; }
  return {sub_val, op_status::SUCCESS};
}

}  // namespace cudf::io::parquet::experimental::detail::variant_reader
