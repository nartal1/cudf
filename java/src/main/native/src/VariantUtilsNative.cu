/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "VariantUtilsNative.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/functional>
#include <cuda_runtime_api.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <stdexcept>

namespace cudf::jni {
namespace {
using op_status          = cudf::io::parquet::experimental::variant_operation_status;
using primitive_type     = cudf::io::parquet::experimental::variant_primitive_type;
constexpr int block_size = 256;

void validate_status(column_view const& status, size_type rows)
{
  CUDF_EXPECTS(
    status.type().id() == type_id::UINT8, "Variant status must be UINT8", std::invalid_argument);
  CUDF_EXPECTS(
    !status.nullable(), "Variant status must not have a validity buffer", std::invalid_argument);
  CUDF_EXPECTS(status.size() == rows, "Variant status row count must match", std::invalid_argument);
}

// Keep the recognized primitive IDs aligned with cast_status_for_primitive in variant_extract.cu.
// A known noninteger type is a mismatch even if its own payload is truncated; the integer
// decoder does not validate another physical type's payload.
__device__ bool recognized_noninteger(primitive_type type)
{
  switch (type) {
    case primitive_type::BOOLEAN_TRUE:
    case primitive_type::BOOLEAN_FALSE:
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

// Payloads are little endian and may be unaligned. Byte accumulation avoids unaligned native
// loads. Sign extension uses arithmetic on in-range INT64 values, including the INT64 minimum.
__device__ int64_t decode_signed_integer(uint8_t const* data, int width)
{
  int64_t value = data[width - 1];
  if (value >= 128) { value -= 256; }
  for (int byte = width - 2; byte >= 0; --byte) {
    value = value * 256 + data[byte];
  }
  return value;
}

CUDF_KERNEL void cast_integer_kernel(lists_column_device_view values,
                                     int64_t* output,
                                     bitmask_type* validity,
                                     uint8_t* statuses,
                                     size_type rows)
{
  auto const tid    = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto row = tid; row < rows; row += stride) {
    output[row] = 0;
    auto status = static_cast<op_status>(statuses[row]);
    if (status == op_status::SUCCESS) {
      if (values.is_null(row)) {
        status = op_status::ROW_NULL;
      } else {
        auto const begin = values.offset_at(row);
        auto const size  = values.offset_at(row + 1) - begin;
        if (size == 0) {
          status = op_status::MALFORMED_VARIANT;
        } else {
          auto const* bytes = values.child().data<uint8_t>() + begin;
          auto const header = bytes[0];
          auto const type   = static_cast<primitive_type>(header >> 2);
          if ((header & 3) != 0) {
            status = op_status::TYPE_MISMATCH;
          } else if (type == primitive_type::NULLVAL) {
            status = op_status::VARIANT_NULL;
          } else {
            int width = 0;
            switch (type) {
              case primitive_type::INT8: width = 1; break;
              case primitive_type::INT16: width = 2; break;
              case primitive_type::INT32: width = 4; break;
              case primitive_type::INT64: width = 8; break;
              default: break;
            }
            if (width != 0) {
              if (size >= 1 + width) {
                output[row] = decode_signed_integer(bytes + 1, width);
              } else {
                status = op_status::MALFORMED_VARIANT;
              }
            } else {
              status = recognized_noninteger(type) ? op_status::TYPE_MISMATCH
                                                   : op_status::MALFORMED_VARIANT;
            }
          }
        }
      }
    }
    statuses[row] = static_cast<uint8_t>(status);
    if (status != op_status::SUCCESS) { cudf::clear_bit(validity, row); }
  }
}
}  // namespace

bool all_variant_rows_covered(column_view const& extraction_status,
                              std::vector<column_view> const& decoded_statuses,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  auto const rows = extraction_status.size();
  validate_status(extraction_status, rows);
  for (auto const& decoded : decoded_statuses) {
    validate_status(decoded, rows);
  }
  if (rows == 0) { return true; }

  std::vector<column_view> statuses{extraction_status};
  statuses.insert(statuses.end(), decoded_statuses.begin(), decoded_statuses.end());
  auto device_statuses = table_device_view::create(table_view{statuses}, stream);
  auto const view      = *device_statuses;
  auto const begin     = thrust::make_counting_iterator<size_type>(0);
  return thrust::transform_reduce(
    rmm::exec_policy(stream, mr),
    begin,
    begin + rows,
    [view] __device__(size_type row) -> bool {
      if (view.column(0).element<uint8_t>(row) == static_cast<uint8_t>(op_status::ROW_NULL)) {
        return true;
      }
      for (size_type col = 1; col < view.num_columns(); ++col) {
        if (view.column(col).element<uint8_t>(row) == static_cast<uint8_t>(op_status::SUCCESS)) {
          return true;
        }
      }
      return false;
    },
    true,
    cuda::std::logical_and<bool>{});
}

std::unique_ptr<column> cast_variant_integer_to_long(column_view const& values,
                                                     mutable_column_view status,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(values.type().id() == type_id::LIST,
               "Variant values must be LIST<UINT8>",
               std::invalid_argument);
  CUDF_EXPECTS(lists_column_view{values}.child().type().id() == type_id::UINT8,
               "Variant values must be LIST<UINT8>",
               std::invalid_argument);
  validate_status(status, values.size());
  if (values.size() == 0) { return make_empty_column(data_type{type_id::INT64}); }

  auto output = make_numeric_column(
    data_type{type_id::INT64}, values.size(), mask_state::ALL_VALID, stream, mr);
  auto output_view   = output->mutable_view();
  auto values_device = column_device_view::create(values, stream);
  auto const grid    = cudf::detail::grid_1d{values.size(), block_size};
  cast_integer_kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(
    lists_column_device_view{*values_device},
    output_view.data<int64_t>(),
    output_view.null_mask(),
    status.data<uint8_t>(),
    values.size());
  CUDF_CUDA_TRY(cudaGetLastError());
  auto const null_count = cudf::null_count(output_view.null_mask(), 0, values.size(), stream);
  if (null_count == 0) {
    output->set_null_mask(cudf::create_null_mask(0, mask_state::UNALLOCATED, stream, mr), 0);
  } else {
    output->set_null_count(null_count);
  }
  return output;
}
}  // namespace cudf::jni
