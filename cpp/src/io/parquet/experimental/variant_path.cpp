/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "variant_path.hpp"

namespace cudf::io::parquet::experimental::detail {

std::vector<std::string> parse_variant_path(std::string_view path)
{
  return parse_variant_path_steps(path);
}

}  // namespace cudf::io::parquet::experimental::detail
