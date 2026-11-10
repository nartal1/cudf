/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>
#include <vector>

namespace cudf::jni {

bool all_variant_rows_covered(column_view const& extraction_status,
                              std::vector<column_view> const& decoded_statuses,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

std::unique_ptr<column> cast_variant_integer_to_long(column_view const& values,
                                                     mutable_column_view status,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr);

}  // namespace cudf::jni
