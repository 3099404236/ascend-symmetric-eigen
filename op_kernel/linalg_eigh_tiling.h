#pragma once
#include <cstdint>

struct LinalgEighTilingData {
    uint32_t n;
    uint32_t batch_size;
    uint32_t is_upper;
};
