#ifndef GDF_GDF_H
#define GDF_GDF_H

#include <cstdlib>
#include <cstdint>
#include "types.h"
#include "io_types.h"
#include "convert_types.h"
#include "io_readers.hpp"

constexpr size_t GDF_VALID_BITSIZE{(sizeof(gdf_valid_type) * 8)};

extern "C" {
#include "functions.h"
#include "io_functions.h"
}

#endif /* GDF_GDF_H */
