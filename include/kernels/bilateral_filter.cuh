#pragma once
#include <cstdint>

constexpr int TILE_W = 16;
constexpr int TILE_H = 16;
constexpr int RADIUS = 5;
constexpr int TWO_RADIUS = RADIUS * 2;

__global__ void bilateral_filter(
    const float *d_input,
    float *d_output,
    const int width,
    const int height,
    const float sigma_s,
    const float sigma_r);
