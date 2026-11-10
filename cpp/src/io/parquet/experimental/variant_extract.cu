/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/block_utils.cuh"
#include "variant_path.hpp"
#include "variant_reader.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/numeric>
#include <cuda/std/concepts>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace cudf {
namespace io::parquet::experimental {
namespace {

using namespace detail::variant_reader;

// The fixed-width signed integers a VARIANT value can be cast to: INT{8,16,32,64}.  Matches the
// exact width types (not e.g. __int128) since those are the only variant primitive int headers.
template <typename T>
concept variant_int = cuda::std::signed_integral<T> && !cuda::std::same_as<T, __int128_t>;

// The microsecond/nanosecond timestamps a VARIANT value can be decoded into.
template <typename T>
concept variant_timestamp =
  cuda::std::same_as<T, cudf::timestamp_us> || cuda::std::same_as<T, cudf::timestamp_ns>;

// The fixed-width primitive types (signed integers, floats, timestamps) a VARIANT value can be
// decoded into.
template <typename T>
concept variant_fixed_width =
  variant_int<T> || cuda::std::floating_point<T> || variant_timestamp<T>;

// The output types a VARIANT value can be cast to: the fixed-width signed integers, floats,
// timestamps, decimals, bool, and strings.
template <typename T>
concept variant_castable = variant_fixed_width<T> || cudf::concepts::fixed_point<T> ||
                           cudf::concepts::boolean<T> || cudf::concepts::string_view<T>;

// Whether a VARIANT primitive type header encodes `T`.
template <variant_fixed_width T>
__device__ constexpr bool matches_primitive_type(primitive_type ptype)
{
  if constexpr (cuda::std::is_same_v<T, int8_t>) {
    return ptype == primitive_type::INT8;
  } else if constexpr (cuda::std::is_same_v<T, int16_t>) {
    return ptype == primitive_type::INT16;
  } else if constexpr (cuda::std::is_same_v<T, int32_t>) {
    return ptype == primitive_type::INT32;
  } else if constexpr (cuda::std::is_same_v<T, int64_t>) {
    return ptype == primitive_type::INT64;
  } else if constexpr (cuda::std::is_same_v<T, float>) {
    return ptype == primitive_type::FLOAT32;
  } else if constexpr (cuda::std::is_same_v<T, double>) {
    return ptype == primitive_type::FLOAT64;
  }
  // cuDF timestamps carry no timezone, so a timestamp target also accepts the matching NTZ
  // encoding; Spark's TIMESTAMP and TIMESTAMP_NTZ both map to TIMESTAMP_MICROSECONDS.
  else if constexpr (cuda::std::is_same_v<T, cudf::timestamp_us>) {
    return ptype == primitive_type::TIMESTAMP_MICROS ||
           ptype == primitive_type::TIMESTAMP_NTZ_MICROS;
  } else {
    static_assert(cuda::std::is_same_v<T, cudf::timestamp_ns>,
                  "T is not a supported variant primitive type");
    return ptype == primitive_type::TIMESTAMP_NANOS || ptype == primitive_type::TIMESTAMP_NTZ_NANOS;
  }
}

/**
 * @brief Decode a single VARIANT value blob into a fixed-width primitive of type `T`.
 *
 * Requires `basic_type == primitive` and a value header accepted by `matches_primitive_type<T>`.
 */
template <typename T>
__device__ inline cuda::std::optional<T> decode_primitive(device_span<uint8_t const> enc)
{
  if (cuda::std::cmp_less(enc.size(), 1 + sizeof(T))) { return cuda::std::nullopt; }

  uint8_t const value_metadata = enc[0];
  if (decode_basic_type(value_metadata) != basic_type::PRIMITIVE ||
      !matches_primitive_type<T>(
        static_cast<primitive_type>(variant_value_header(value_metadata)))) {
    return cuda::std::nullopt;
  }
  return cudf::io::unaligned_load<T>(enc.data() + 1);
}

/**
 * @brief Decode a single VARIANT value blob into a bool.
 *
 * Boolean values carry no payload: the distinction between true and false is encoded entirely in
 * the primitive type header (`boolean_true` vs `boolean_false`).
 */
__device__ inline cuda::std::optional<bool> decode_bool(device_span<uint8_t const> enc)
{
  if (enc.empty()) { return cuda::std::nullopt; }
  uint8_t const value_metadata = enc[0];
  if (decode_basic_type(value_metadata) != basic_type::PRIMITIVE) { return cuda::std::nullopt; }
  auto const value_header = variant_value_header(value_metadata);
  if (value_header == static_cast<uint8_t>(primitive_type::BOOLEAN_TRUE)) { return true; }
  if (value_header == static_cast<uint8_t>(primitive_type::BOOLEAN_FALSE)) { return false; }
  return cuda::std::nullopt;
}

__device__ cuda::std::optional<device_span<uint8_t const>> decode_string(
  device_span<uint8_t const> enc)
{
  auto const len = enc.size();
  if (len < 1) { return cuda::std::nullopt; }
  uint8_t const value_metadata = enc[0];
  auto const btype             = decode_basic_type(value_metadata);
  auto const value_header      = variant_value_header(value_metadata);

  if (btype == basic_type::SHORT_STRING) {
    // Short string: value_header = length
    std::size_t const str_len = value_header;
    if (1 + str_len > len) { return cuda::std::nullopt; }
    return enc.subspan(1, str_len);
  }
  if (btype == basic_type::PRIMITIVE &&
      value_header == static_cast<uint8_t>(primitive_type::LONG_STRING)) {
    // Long string: 1-byte header + 4-byte LE length + char bytes
    constexpr std::size_t long_string_prefix_bytes = 1 + sizeof(uint32_t);
    if (len < long_string_prefix_bytes) { return cuda::std::nullopt; }
    auto const str_len = cudf::io::unaligned_load<uint32_t>(enc.data() + 1);
    // Encoded length claims more char bytes than the buffer holds: truncated/malformed blob
    if (long_string_prefix_bytes + str_len > len) { return cuda::std::nullopt; }
    return enc.subspan(long_string_prefix_bytes, str_len);
  }
  return cuda::std::nullopt;
}

__device__ device_span<uint8_t const> list_row_span(cudf::lists_column_device_view const& col,
                                                    size_type row)
{
  auto const begin = col.offset_at(row);
  auto const end   = col.offset_at(row + 1);
  return {col.child().data<uint8_t>() + begin, static_cast<std::size_t>(end - begin)};
}

__device__ cuda::std::pair<device_span<uint8_t const>, device_span<uint8_t const>>
metadata_and_value_at(cudf::lists_column_device_view const& metadata,
                      cudf::lists_column_device_view const& values,
                      size_type row)
{
  return {list_row_span(metadata, row), list_row_span(values, row)};
}

constexpr int block_size = 256;

/**
 * @brief Resolves `path` in each VARIANT row and record the located field's size and source offset.
 *
 * For each non-null row, walks `path` to the target value and writes its byte length to
 * `d_sizes[row]` and its offset within the row's value blob to `d_src_offsets[row]`. Rows that are
 * null, or whose path does not resolve, are marked null in `d_null_mask` with a size of 0.
 */
CUDF_KERNEL __launch_bounds__(block_size) void locate_variant_fields_kernel(
  cudf::lists_column_device_view metadata,
  cudf::lists_column_device_view values,
  column_device_view path,
  device_span<size_type> d_sizes,
  device_span<size_type> d_src_offsets,
  bitmask_type* d_null_mask,
  device_span<op_status> d_status)  // empty when no status was requested
{
  auto const num_rows = static_cast<size_type>(d_sizes.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (!cudf::bit_is_set(d_null_mask, row)) {
      d_sizes[row]       = 0;
      d_src_offsets[row] = 0;
      if (!d_status.empty()) { d_status[row] = op_status::ROW_NULL; }
      continue;
    }

    auto const [meta, val] = metadata_and_value_at(metadata, values, row);
    auto const [field, st] = resolve_path(meta, val, path);

    if (!d_status.empty()) { d_status[row] = st; }

    if (field.empty()) {
      d_sizes[row]       = 0;
      d_src_offsets[row] = 0;
      cudf::clear_bit(d_null_mask, row);
    } else {
      d_sizes[row]       = static_cast<size_type>(field.size());
      d_src_offsets[row] = static_cast<size_type>(field.data() - val.data());
    }
  }
}

// Returns true for every primitive_type ID that variant_value_length maps to a known payload
// size in its `basic_type::PRIMITIVE` switch, i.e. every ID other than its `default` case.
__device__ bool is_recognized_primitive_type(primitive_type ptype)
{
  switch (ptype) {
    case primitive_type::NULLVAL:
    case primitive_type::BOOLEAN_TRUE:
    case primitive_type::BOOLEAN_FALSE:
    case primitive_type::INT8:
    case primitive_type::INT16:
    case primitive_type::INT32:
    case primitive_type::INT64:
    case primitive_type::FLOAT64:
    case primitive_type::DECIMAL4:
    case primitive_type::DECIMAL8:
    case primitive_type::DECIMAL16:
    case primitive_type::DATE:
    case primitive_type::TIMESTAMP_MICROS:
    case primitive_type::TIMESTAMP_NTZ_MICROS:
    case primitive_type::FLOAT32:
    case primitive_type::BINARY:
    case primitive_type::LONG_STRING:
    case primitive_type::TIME_NTZ_MICROS:
    case primitive_type::TIMESTAMP_NANOS:
    case primitive_type::TIMESTAMP_NTZ_NANOS:
    case primitive_type::UUID: return true;
    default: return false;
  }
}

/**
 * @brief Status helper for fixed-width primitive targets: classifies why `decode_primitive<T>`
 * failed to decode `val`, per `variant_operation_status` semantics.
 */
template <variant_fixed_width T>
__device__ op_status cast_status_for_primitive(device_span<uint8_t const> val)
{
  if (val.empty()) { return op_status::MALFORMED_VARIANT; }
  if (is_variant_null(val)) { return op_status::VARIANT_NULL; }
  if (decode_primitive<T>(val).has_value()) { return op_status::SUCCESS; }
  if (decode_basic_type(val[0]) != basic_type::PRIMITIVE) { return op_status::TYPE_MISMATCH; }
  auto const ptype = static_cast<primitive_type>(variant_value_header(val[0]));
  if (matches_primitive_type<T>(ptype)) { return op_status::MALFORMED_VARIANT; }
  return is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                             : op_status::MALFORMED_VARIANT;
}

// The spec allows a scale in [0, 38] for every decimal width.
constexpr int variant_decimal_max_scale = 38;

// The largest power of ten each representation holds; no value fits past it.
constexpr int max_int128_pow10 = cuda::std::numeric_limits<__int128_t>::digits10;
constexpr int max_int64_pow10  = cuda::std::numeric_limits<int64_t>::digits10;

// Multiply `value` by 10^exp, or return nullopt if the result does not fit in `__int128_t`.
__device__ cuda::std::optional<__int128_t> constexpr multiply_pow10(__int128_t value, int64_t exp)
{
  // Zero is representable at every scale, while any other value overflows past 10^38.
  if (value == 0) { return 0; }
  if (exp > max_int128_pow10) { return cuda::std::nullopt; }

  constexpr __int128_t max_over_10 = cuda::std::numeric_limits<__int128_t>::max() / 10;
  constexpr __int128_t min_over_10 = cuda::std::numeric_limits<__int128_t>::min() / 10;
  for (int64_t i = 0; i < exp; ++i) {
    if (value > max_over_10 || value < min_over_10) { return cuda::std::nullopt; }
    value *= 10;
  }
  return value;
}

// Divide `value` by 10^exp, truncating toward zero.
__device__ __int128_t constexpr divide_pow10(__int128_t value, int64_t exp)
{
  using numeric::detail::ipow;

  // Any `__int128_t` is smaller than 10^39, so a larger divisor truncates it away
  if (exp > max_int128_pow10) { return 0; }
  auto const exponent = static_cast<int32_t>(exp);

  // 128-bit division is a slow software sequence; use the 64-bit one when both operands fit.
  constexpr __int128_t i64_max = cuda::std::numeric_limits<int64_t>::max();
  constexpr __int128_t i64_min = cuda::std::numeric_limits<int64_t>::min();
  if (exponent <= max_int64_pow10 && value <= i64_max && value >= i64_min) {
    return static_cast<int64_t>(value) / ipow<int64_t, numeric::Radix::BASE_10>(exponent);
  }
  return value / ipow<__int128_t, numeric::Radix::BASE_10>(exponent);
}

__device__ int constexpr variant_decimal_unscaled_width(primitive_type ptype)
{
  switch (ptype) {
    case primitive_type::DECIMAL4: return 4;
    case primitive_type::DECIMAL8: return 8;
    case primitive_type::DECIMAL16: return 16;
    default: return 0;
  }
}

/**
 * @brief Decode a single VARIANT decimal value blob into the representation of a cuDF fixed-point
 * type, rescaled to `desired_scale`.
 *
 * Encoded as the value metadata byte, a one-byte scale (fractional digit count), then the
 * little-endian two's-complement unscaled integer. Any of the three widths decodes into any `Rep`
 * whose range fits the rescaled value; digits below `desired_scale` truncate toward zero.
 *
 * @return The rescaled representation, valid only when the returned status is `SUCCESS`
 */
template <typename Rep>
__device__ cuda::std::pair<Rep, op_status> decode_decimal(device_span<uint8_t const> enc,
                                                          int desired_scale)
{
  auto const fail = [](op_status status) { return cuda::std::pair<Rep, op_status>{Rep{}, status}; };

  if (enc.empty()) { return fail(op_status::MALFORMED_VARIANT); }
  if (is_variant_null(enc)) { return fail(op_status::VARIANT_NULL); }
  if (decode_basic_type(enc[0]) != basic_type::PRIMITIVE) { return fail(op_status::TYPE_MISMATCH); }

  auto const ptype = static_cast<primitive_type>(variant_value_header(enc[0]));
  auto const width = variant_decimal_unscaled_width(ptype);
  if (width == 0) {
    return fail(is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                                    : op_status::MALFORMED_VARIANT);
  }

  constexpr size_type scale_bytes = 1;
  if (cuda::std::cmp_less(enc.size(), variant_header_bytes + scale_bytes + width)) {
    return fail(op_status::MALFORMED_VARIANT);
  }
  int const encoded_scale = enc[variant_header_bytes];
  if (encoded_scale > variant_decimal_max_scale) { return fail(op_status::MALFORMED_VARIANT); }

  auto const* unscaled_data = enc.data() + variant_header_bytes + scale_bytes;
  auto const unscaled       = [&]() -> __int128_t {
    switch (width) {
      case 4: return cudf::io::unaligned_load<int32_t>(unscaled_data);
      case 8: return cudf::io::unaligned_load<int64_t>(unscaled_data);
      case 16: return cudf::io::unaligned_load<__int128_t>(unscaled_data);
      default: return 0;  // should be unreachable
    }
  }();

  // The encoded value is `unscaled * 10^-encoded_scale`; the output is `rep * 10^desired_scale`.
  auto const shift = -static_cast<int64_t>(encoded_scale) - desired_scale;
  __int128_t rescaled{};
  if (shift >= 0) {
    auto const scaled = multiply_pow10(unscaled, shift);
    if (!scaled.has_value()) { return fail(op_status::OVERFLOW); }
    rescaled = scaled.value();
  } else {
    rescaled = divide_pow10(unscaled, -shift);
  }

  if constexpr (!cuda::std::is_same_v<Rep, __int128_t>) {
    if (rescaled < static_cast<__int128_t>(cuda::std::numeric_limits<Rep>::min()) ||
        rescaled > static_cast<__int128_t>(cuda::std::numeric_limits<Rep>::max())) {
      return fail(op_status::OVERFLOW);
    }
  }
  return {static_cast<Rep>(rescaled), op_status::SUCCESS};
}

/**
 * @brief Decides whether a row should be decoded, shared by every cast path.
 *
 * A skipped row has its null bit cleared and its status recorded here. Zeroing its output element
 * is left to the caller, since the element type is not known here.
 *
 * @return True when the row's value blob should be decoded
 */
__device__ bool should_decode_row(size_type row, bitmask_type* d_null_mask, op_status* d_status)
{
  auto const is_valid = cudf::bit_is_set(d_null_mask, row);

  if (d_status == nullptr) { return is_valid; }

  if (d_status[row] != op_status::SUCCESS) {
    if (is_valid) { cudf::clear_bit(d_null_mask, row); }
    return false;
  }
  // Status column is always non-nullable, so ROW_NULL is what reports the cleared null bit.
  if (!is_valid) {
    d_status[row] = op_status::ROW_NULL;
    return false;
  }
  return true;
}

/**
 * @brief Per-row kernel: decode each VARIANT value blob into a fixed-width primitive of type `T`.
 *
 * Writes the decoded value to `d_output[row]` for non-null rows whose blob is a variant primitive
 * whose physical type id matches `T` exactly (e.g. an int16 value does not decode into an int32
 * output, and a float32 value does not decode into a float64 output; there is no widening). Rows
 * that are null, or whose value is not an exact-width match for `T`, are marked null in
 * `d_null_mask` with an output of 0.
 */
// `d_status`, when present, is an in-out buffer: it is read as incoming status from a prior
// `get_variant_field` call (rows already marked non-success are propagated without decoding), then
// overwritten in place with the final per-row status. Callers with no real incoming status must
// pre-fill every row with `op_status::SUCCESS` before calling.
template <typename T>
CUDF_KERNEL __launch_bounds__(block_size) void cast_variant_primitive_kernel(
  cudf::lists_column_device_view values,
  device_span<T> d_output,
  bitmask_type* d_null_mask,
  op_status* d_status)  // nullptr when no status was requested
{
  auto const num_rows = static_cast<size_type>(d_output.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (!should_decode_row(row, d_null_mask, d_status)) {
      d_output[row] = T{};
      continue;
    }

    auto const val     = list_row_span(values, row);
    auto const decoded = decode_primitive<T>(val);
    if (decoded.has_value()) {
      d_output[row] = *decoded;
      if (d_status != nullptr) { d_status[row] = op_status::SUCCESS; }
    } else {
      d_output[row] = T{};
      cudf::clear_bit(d_null_mask, row);
      if (d_status != nullptr) { d_status[row] = cast_status_for_primitive<T>(val); }
    }
  }
}

/**
 * @brief Per-row kernel: decode each VARIANT decimal value blob into a fixed-point representation
 * of type `Rep`, rescaled to `desired_scale`. Same null and status protocol as
 * `cast_variant_primitive_kernel`.
 */
template <typename Rep>
CUDF_KERNEL __launch_bounds__(block_size) void cast_variant_decimal_kernel(
  cudf::lists_column_device_view values,
  device_span<Rep> d_output,
  int desired_scale,
  bitmask_type* d_null_mask,
  op_status* d_status)  // nullptr when no status was requested
{
  auto const num_rows = static_cast<size_type>(d_output.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (!should_decode_row(row, d_null_mask, d_status)) {
      d_output[row] = Rep{};
      continue;
    }

    auto const [value, status] = decode_decimal<Rep>(list_row_span(values, row), desired_scale);
    if (status == op_status::SUCCESS) {
      d_output[row] = value;
    } else {
      d_output[row] = Rep{};
      cudf::clear_bit(d_null_mask, row);
    }
    if (d_status != nullptr) { d_status[row] = status; }
  }
}

__device__ op_status cast_status_for_bool(device_span<uint8_t const> val)
{
  if (val.empty()) { return op_status::MALFORMED_VARIANT; }
  if (is_variant_null(val)) { return op_status::VARIANT_NULL; }
  if (decode_bool(val).has_value()) { return op_status::SUCCESS; }
  if (decode_basic_type(val[0]) != basic_type::PRIMITIVE) { return op_status::TYPE_MISMATCH; }
  // Boolean values carry no payload, so a BOOLEAN_TRUE/FALSE header can never be truncated;
  // decode_bool would have succeeded above.  Any remaining primitive ID is a type mismatch when
  // recognised, or malformed when not.
  auto const ptype = static_cast<primitive_type>(variant_value_header(val[0]));
  return is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                             : op_status::MALFORMED_VARIANT;
}

__device__ op_status cast_status_for_string(device_span<uint8_t const> val)
{
  if (val.empty()) { return op_status::MALFORMED_VARIANT; }
  if (is_variant_null(val)) { return op_status::VARIANT_NULL; }
  if (decode_string(val).has_value()) { return op_status::SUCCESS; }
  auto const btype = decode_basic_type(val[0]);
  if (btype == basic_type::SHORT_STRING) { return op_status::MALFORMED_VARIANT; }
  if (btype == basic_type::PRIMITIVE) {
    auto const ptype = static_cast<primitive_type>(variant_value_header(val[0]));
    // LONG_STRING is a recognized string type whose payload was truncated.
    if (ptype == primitive_type::LONG_STRING) { return op_status::MALFORMED_VARIANT; }
    return is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                               : op_status::MALFORMED_VARIANT;
  }
  // OBJECT, ARRAY, or other non-primitive basic types: well-formed, just not a string.
  return op_status::TYPE_MISMATCH;
}

/**
 * @brief Strings-children functor: decode each VARIANT value blob into a string.
 *
 * Used with `make_strings_children`, so it runs in two passes. On the sizing pass (`d_chars ==
 * nullptr`) it writes each decoded string's length to `d_sizes[row]`; on the write pass it copies
 * the decoded bytes to `d_chars` at `d_offsets[row]`. Rows that are null, or whose value does not
 * decode to a string, are marked null in `d_null_mask` with size 0.
 */
struct cast_variant_string_fn {
  cudf::lists_column_device_view d_values;
  bitmask_type* d_null_mask;
  size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;
  // In-out status tracking (optional: d_status non-null to enable; status is always non-nullable).
  // Read as incoming status from a prior `get_variant_field` call, then overwritten in place with
  // the final status on the sizing pass.
  op_status* d_status{nullptr};

  __device__ void operator()(size_type row)
  {
    // Status is only written on the sizing pass (d_chars == nullptr). On the writing pass the
    // null mask may already be cleared from the sizing pass, so we must not re-inspect it to
    // write status (that would misidentify a decode-failed row as a SQL-null row).
    bool const is_sizing_pass = (d_chars == nullptr);

    if (d_status) {
      // Status column is always non-nullable; row_null replaces the null bit.
      auto const s = d_status[row];
      if (s != op_status::SUCCESS) {
        if (is_sizing_pass) { d_sizes[row] = 0; }
        if (cudf::bit_is_set(d_null_mask, row)) { cudf::clear_bit(d_null_mask, row); }
        return;
      }
      if (!cudf::bit_is_set(d_null_mask, row)) {
        if (is_sizing_pass) {
          d_sizes[row]  = 0;
          d_status[row] = op_status::ROW_NULL;
        }
        return;
      }
    } else {
      if (!cudf::bit_is_set(d_null_mask, row)) {
        if (is_sizing_pass) { d_sizes[row] = 0; }
        return;
      }
    }

    auto const val = list_row_span(d_values, row);

    auto const str = decode_string(val);
    if (!str) {
      if (is_sizing_pass) { d_sizes[row] = 0; }
      cudf::clear_bit(d_null_mask, row);
      if (is_sizing_pass && d_status) { d_status[row] = cast_status_for_string(val); }
      return;
    }

    if (is_sizing_pass) {
      d_sizes[row] = str->size();
    } else {
      cuda::std::memcpy(d_chars + d_offsets[row], str->data(), str->size());
    }
    if (is_sizing_pass && d_status) { d_status[row] = op_status::SUCCESS; }
  }
};

void validate_variant_child(column_view const& child)
{
  CUDF_EXPECTS(child.type().id() == type_id::LIST,
               "VARIANT metadata/value column must be a list",
               std::invalid_argument);
  CUDF_EXPECTS(lists_column_view{child}.child().type().id() == type_id::UINT8,
               "VARIANT metadata/value column must be list<uint8>",
               std::invalid_argument);
}

struct cast_variant_fn {
  cudf::lists_column_device_view values;
  size_type num_rows;
  data_type desired_type;
  bitmask_type* d_null_mask;
  cuda::device_buffer<cudf::bitmask_type> null_mask{
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED)};
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;
  // In-out status tracking; null when no status was requested.
  op_status* d_status{nullptr};

  template <variant_fixed_width T>
  std::unique_ptr<column> operator()()
  {
    rmm::device_buffer data{num_rows * sizeof(T), stream, mr};
    auto const grid = cudf::detail::grid_1d{num_rows, block_size};
    auto const d_out =
      device_span<T>{static_cast<T*>(data.data()), static_cast<std::size_t>(num_rows)};
    cast_variant_primitive_kernel<T>
      <<<grid.num_blocks, block_size, 0, stream.get()>>>(values, d_out, d_null_mask, d_status);
    CUDF_CUDA_TRY(cudaGetLastError());

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0
                                      ? std::move(null_mask)
                                      : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                    null_count);
  }

  template <cudf::concepts::fixed_point T>
  std::unique_ptr<column> operator()()
  {
    using Rep = typename T::rep;
    rmm::device_buffer data{num_rows * sizeof(Rep), stream, mr};
    auto const grid = cudf::detail::grid_1d{num_rows, block_size};
    auto const d_out =
      device_span<Rep>{static_cast<Rep*>(data.data()), static_cast<std::size_t>(num_rows)};
    cast_variant_decimal_kernel<Rep><<<grid.num_blocks, block_size, 0, stream.get()>>>(
      values, d_out, desired_type.scale(), d_null_mask, d_status);
    CUDF_CUDA_TRY(cudaGetLastError());

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0
                                      ? std::move(null_mask)
                                      : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                    null_count);
  }

  template <cudf::concepts::boolean T>
  std::unique_ptr<column> operator()()
  {
    rmm::device_buffer data{num_rows * sizeof(bool), stream, mr};

    auto* dp_s = d_status;

    thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>(0),
                     cuda::counting_iterator<size_type>(num_rows),
                     [vals  = this->values,
                      d_out = static_cast<bool*>(data.data()),
                      dnm   = this->d_null_mask,
                      dp_s] __device__(size_type row) {
                       if (!should_decode_row(row, dnm, dp_s)) {
                         d_out[row] = false;
                         return;
                       }
                       // The row's null bit is still set here, so a failure can clear it unguarded.
                       auto const fail = [&](op_status s) {
                         d_out[row] = false;
                         cudf::clear_bit(dnm, row);
                         if (dp_s) { dp_s[row] = s; }
                       };
                       auto const val     = list_row_span(vals, row);
                       auto const decoded = decode_bool(val);
                       if (!decoded) { return fail(cast_status_for_bool(val)); }
                       d_out[row] = *decoded;
                       if (dp_s) { dp_s[row] = op_status::SUCCESS; }
                     });

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0
                                      ? std::move(null_mask)
                                      : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                    null_count);
  }

  template <cudf::concepts::string_view T>
  std::unique_ptr<column> operator()()
  {
    cast_variant_string_fn fn{values, d_null_mask, nullptr, nullptr, {}, d_status};
    auto [offsets_column, chars] =
      cudf::strings::detail::make_strings_children(fn, num_rows, stream, mr);

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return make_strings_column(num_rows,
                               std::move(offsets_column),
                               chars.release(),
                               null_count,
                               null_count > 0
                                 ? std::move(null_mask)
                                 : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  }

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(not variant_castable<T>)
  {
    CUDF_FAIL("unsupported type for variant cast", std::invalid_argument);
  }
};

/**
 * @brief Classifies only the first (value_metadata) byte of enc; does not validate the remaining
 * payload. A recognized header returns its logical type even when the payload is truncated. Returns
 * nullopt for an empty blob or an unrecognized primitive type ID.
 */
__device__ cuda::std::optional<variant_logical_type> logical_type_of(device_span<uint8_t const> enc)
{
  if (enc.empty()) { return cuda::std::nullopt; }
  auto const value_metadata = enc[0];
  auto const btype          = decode_basic_type(value_metadata);

  if (btype == basic_type::SHORT_STRING) { return variant_logical_type::STRING; }
  if (btype == basic_type::OBJECT) { return variant_logical_type::OBJECT; }
  if (btype == basic_type::ARRAY) { return variant_logical_type::ARRAY; }

  switch (static_cast<primitive_type>(variant_value_header(value_metadata))) {
    case primitive_type::NULLVAL: return variant_logical_type::NULL_VALUE;
    case primitive_type::BOOLEAN_TRUE:
    case primitive_type::BOOLEAN_FALSE: return variant_logical_type::BOOLEAN;
    case primitive_type::INT8:
    case primitive_type::INT16:
    case primitive_type::INT32:
    case primitive_type::INT64: return variant_logical_type::LONG_VALUE;
    case primitive_type::FLOAT64: return variant_logical_type::DOUBLE_VALUE;
    case primitive_type::DECIMAL4:
    case primitive_type::DECIMAL8:
    case primitive_type::DECIMAL16: return variant_logical_type::DECIMAL;
    case primitive_type::DATE: return variant_logical_type::DATE;
    case primitive_type::TIMESTAMP_MICROS:
    case primitive_type::TIMESTAMP_NANOS: return variant_logical_type::TIMESTAMP;
    case primitive_type::TIMESTAMP_NTZ_MICROS:
    case primitive_type::TIMESTAMP_NTZ_NANOS: return variant_logical_type::TIMESTAMP_NTZ;
    case primitive_type::FLOAT32: return variant_logical_type::FLOAT_VALUE;
    case primitive_type::BINARY: return variant_logical_type::BINARY;
    case primitive_type::LONG_STRING: return variant_logical_type::STRING;
    case primitive_type::TIME_NTZ_MICROS: return variant_logical_type::TIME_NTZ;
    case primitive_type::UUID: return variant_logical_type::UUID;
    default: return cuda::std::nullopt;
  }
}

std::unique_ptr<column> build_path_column(cudf::host_span<std::string const> steps,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const depth = steps.size();

  std::string host_chars;
  std::vector<size_type> host_offsets(depth + 1);
  for (size_t i = 0; i < depth; ++i) {
    host_offsets[i] = static_cast<size_type>(host_chars.size());
    host_chars.append(steps[i]);
  }
  host_offsets[depth] = host_chars.size();

  auto d_offsets = cudf::detail::make_device_uvector_async(host_offsets, stream, mr);
  auto offsets_col =
    std::make_unique<column>(data_type{type_id::INT32},
                             static_cast<size_type>(host_offsets.size()),
                             d_offsets.release(),
                             cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                             0);

  auto d_chars = cudf::detail::make_device_uvector(
    host_span<char const>{host_chars.data(), host_chars.size()}, stream, mr);
  return cudf::make_strings_column(depth,
                                   std::move(offsets_col),
                                   d_chars.release(),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

}  // namespace

namespace detail {

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          std::optional<mutable_column_view> status,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(variant_column.type().id() == type_id::STRUCT,
               "VARIANT column must be struct type",
               std::invalid_argument);
  CUDF_EXPECTS(variant_column.num_children() >= 2,
               "VARIANT struct must have at least two children",
               std::invalid_argument);
  validate_variant_child(variant_column.child(0));
  validate_variant_child(variant_column.child(1));

  // Validate the path even for empty input columns
  auto const steps = parse_variant_path(path);

  auto const num_rows = variant_column.size();

  if (status.has_value()) {
    CUDF_EXPECTS(!status->nullable(),
                 "status column must not be nullable; use row_null for SQL-null rows",
                 std::invalid_argument);
    CUDF_EXPECTS(
      status->type().id() == type_id::UINT8, "status column must be UINT8", std::invalid_argument);
    CUDF_EXPECTS(status->size() == num_rows,
                 "status column must have the same number of rows as variant_column",
                 std::invalid_argument);
  }

  if (num_rows == 0) {
    return cudf::make_lists_column(0,
                                   make_empty_column(type_id::INT32),
                                   make_empty_column(type_id::UINT8),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();

  auto path_column      = build_path_column(steps, stream, temp_mr);
  auto path_device_view = column_device_view::create(path_column->view(), stream);

  // Resolve children with respect to any slice/offset on the parent struct
  structs_column_view const variant_struct{variant_column};
  auto const meta_view = variant_struct.get_sliced_child(0, stream);
  auto const val_view  = variant_struct.get_sliced_child(1, stream);

  auto meta_device_view = column_device_view::create(meta_view, stream);
  auto val_device_view  = column_device_view::create(val_view, stream);
  cudf::lists_column_device_view meta_lists_device_view(*meta_device_view);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  cuda::device_buffer<size_type> d_sizes(stream, temp_mr, num_rows, cuda::no_init);
  // Caches the per-row intra-value byte offset
  cuda::device_buffer<size_type> d_src_offsets(stream, temp_mr, num_rows, cuda::no_init);
  auto null_mask =
    variant_column.nullable()
      ? cudf::detail::copy_bitmask(variant_column, stream, mr)
      : cudf::create_null_mask(variant_column.size(), mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = null_mask.data();

  auto grid = cudf::detail::grid_1d{num_rows, block_size};

  auto const d_status =
    status.has_value()
      ? device_span<op_status>{reinterpret_cast<op_status*>(status->data<uint8_t>()),
                               static_cast<std::size_t>(num_rows)}
      : device_span<op_status>{};
  locate_variant_fields_kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(
    meta_lists_device_view,
    val_lists_device_view,
    *path_device_view,
    d_sizes,
    d_src_offsets,
    d_null_mask,
    d_status);
  CUDF_CUDA_TRY(cudaGetLastError());

  auto [offsets_column, total_bytes] =
    cudf::strings::detail::make_offsets_child_column(d_sizes, stream, mr);
  CUDF_EXPECTS(total_bytes <= std::numeric_limits<size_type>::max(),
               "VARIANT extracted bytes exceed cudf size_type limit",
               std::overflow_error);
  device_span<size_type const> d_offsets{offsets_column->view().data<size_type>(),
                                         static_cast<std::size_t>(num_rows + 1)};

  auto val_child = make_numeric_column(
    data_type{type_id::UINT8}, total_bytes, mask_state::UNALLOCATED, stream, mr);
  if (total_bytes > 0) {
    auto const out_base = val_child->mutable_view().data<uint8_t>();
    auto src_iter       = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t const*>(
        [vlv   = val_lists_device_view,
         d_src = d_src_offsets.data()] __device__(size_type row) -> uint8_t const* {
          return vlv.child().template data<uint8_t>() + vlv.offset_at(row) + d_src[row];
        }));
    auto dst_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t*>(
        [out_base, d_off = d_offsets.data()] __device__(size_type row) -> uint8_t* {
          return out_base + d_off[row];
        }));
    cudf::detail::batched_memcpy_async(src_iter, dst_iter, d_sizes.begin(), num_rows, stream);
  }

  auto const null_count = num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
  return make_lists_column(num_rows,
                           std::move(offsets_column),
                           std::move(val_child),
                           null_count,
                           null_count > 0
                             ? std::move(null_mask)
                             : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

std::unique_ptr<column> cast_variant(column_view const& values,
                                     data_type desired_type,
                                     std::optional<mutable_column_view> status,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  validate_variant_child(values);

  switch (desired_type.id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64:
    case type_id::FLOAT32:
    case type_id::FLOAT64:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS:
    case type_id::BOOL8:
    case type_id::STRING:
    case type_id::DECIMAL32:
    case type_id::DECIMAL64:
    case type_id::DECIMAL128: break;
    default: CUDF_FAIL("unsupported type for variant cast", std::invalid_argument);
  }

  size_type const num_rows = values.size();

  // Validate status before the empty-values fast path so callers always get
  // std::invalid_argument for a malformed status column, even when values is empty.
  if (status.has_value()) {
    CUDF_EXPECTS(!status->nullable(),
                 "status column must not be nullable; use row_null for SQL-null rows",
                 std::invalid_argument);
    CUDF_EXPECTS(
      status->type().id() == type_id::UINT8, "status column must be UINT8", std::invalid_argument);
    CUDF_EXPECTS(status->size() == num_rows,
                 "status column must have the same number of rows as the values column",
                 std::invalid_argument);
  }

  if (num_rows == 0) { return make_empty_column(desired_type); }

  auto val_device_view = column_device_view::create(values, stream);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  auto null_mask    = values.nullable()
                        ? cudf::detail::copy_bitmask(values, stream, mr)
                        : cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = null_mask.data();

  return cudf::type_dispatcher(
    desired_type,
    cast_variant_fn{
      val_lists_device_view,
      num_rows,
      desired_type,
      d_null_mask,
      std::move(null_mask),
      stream,
      mr,
      status.has_value() ? reinterpret_cast<op_status*>(status->data<uint8_t>()) : nullptr});
}

std::unique_ptr<column> get_variant_type_id(column_view const& values,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  validate_variant_child(values);
  size_type const num_rows = values.size();
  if (num_rows == 0) { return make_empty_column(data_type{type_id::UINT8}); }

  auto val_device_view = column_device_view::create(values, stream);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  auto null_mask    = values.nullable()
                        ? cudf::detail::copy_bitmask(values, stream, mr)
                        : cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = null_mask.data();

  rmm::device_buffer data{static_cast<std::size_t>(num_rows) * sizeof(uint8_t), stream, mr};

  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>(0),
    cuda::counting_iterator<size_type>(num_rows),
    static_cast<uint8_t*>(data.data()),
    [values = val_lists_device_view, d_null_mask] __device__(size_type row) -> uint8_t {
      if (!cudf::bit_is_set(d_null_mask, row)) { return 0; }
      auto const ltype = logical_type_of(list_row_span(values, row));
      if (ltype.has_value()) { return static_cast<uint8_t>(ltype.value()); }
      cudf::clear_bit(d_null_mask, row);
      return 0;
    });

  auto const null_count = num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
  return std::make_unique<column>(data_type{type_id::UINT8},
                                  num_rows,
                                  std::move(data),
                                  null_count > 0
                                    ? std::move(null_mask)
                                    : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                  null_count);
}

}  // namespace detail

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          std::optional<mutable_column_view> status,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_variant_field(variant_column, path, status, stream, mr);
}

std::unique_ptr<column> cast_variant(column_view const& values,
                                     data_type desired_type,
                                     std::optional<mutable_column_view> status,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::cast_variant(values, desired_type, status, stream, mr);
}

std::unique_ptr<column> get_variant_type_id(column_view const& values,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_variant_type_id(values, stream, mr);
}

std::unique_ptr<column> extract_variant_field(column_view const& variant_column,
                                              std::string_view path,
                                              data_type desired_type,
                                              std::optional<mutable_column_view> status,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const temp_mr = cudf::get_current_device_resource_ref();

  if (status.has_value()) {
    // `status` is filled by `get_variant_field`, then read back by `cast_variant` as incoming
    // status and overwritten in place with the final per-row status.
    auto value = detail::get_variant_field(variant_column, path, status, stream, temp_mr);
    return detail::cast_variant(value->view(), desired_type, status, stream, mr);
  }

  auto value = detail::get_variant_field(variant_column, path, std::nullopt, stream, temp_mr);
  return detail::cast_variant(value->view(), desired_type, std::nullopt, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
