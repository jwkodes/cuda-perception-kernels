#include "kernels/bilateral_filter.cuh"

__device__ float unnormalised_gaussian_1d(float dx, float sigma_inv_sq)
{
    return expf(-dx * dx * 0.5f * sigma_inv_sq);
}

__device__ float unnormalised_gaussian_2d(float dx, float dy, float sigma_inv_sq)
{
    return expf(-(dx * dx + dy * dy) * 0.5f * sigma_inv_sq);
}

__global__ void bilateral_filter(
    const float *d_input,
    float *d_output,
    const int width,
    const int height,
    const float sigma_s,
    const float sigma_r)
{
    __shared__ float tile[TILE_H + TWO_RADIUS][TILE_W + TWO_RADIUS];

    // Global coordinates of this block's tile
    int tile_start_x = blockIdx.x * blockDim.x - RADIUS;
    int tile_start_y = blockIdx.y * blockDim.y - RADIUS;

    // Each thread loads elements into the tile
    for (int dy = threadIdx.y; dy < TILE_H + TWO_RADIUS; dy += blockDim.y)
    {
        for (int dx = threadIdx.x; dx < TILE_W + TWO_RADIUS; dx += blockDim.x)
        {
            int gy = tile_start_y + dy;
            int gx = tile_start_x + dx;

            // Clamp to image boundaries
            // Also means border is replicated (BORDER_REPLICATE in OpenCV)
            gy = max(0, min(gy, height - 1));
            gx = max(0, min(gx, width - 1));

            tile[dy][dx] = d_input[gy * width + gx];
        }
    }

    // Wait for tile to be fully populated
    __syncthreads();

    // Coordinates of the pixel to be filtered, in tile frame
    int tile_x = threadIdx.x + RADIUS;
    int tile_y = threadIdx.y + RADIUS;

    // Value of the pixel to be filtered
    float center_val = tile[tile_y][tile_x];
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    float sigma_s_inv_sq = 1.0f / (sigma_s * sigma_s);
    float sigma_r_inv_sq = 1.0f / (sigma_r * sigma_r);
    for (int ty = -RADIUS; ty < RADIUS + 1; ty++)
    {
        for (int tx = -RADIUS; tx < RADIUS + 1; tx++)
        {
            float neighbour_val = tile[tile_y + ty][tile_x + tx];
            float weight = unnormalised_gaussian_2d(ty, tx, sigma_s_inv_sq) * unnormalised_gaussian_1d(neighbour_val - center_val, sigma_r_inv_sq);
            weight_sum += weight;
            weighted_sum += (weight * neighbour_val);
        }
    }

    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gx < width && gy < height)
        d_output[gy * width + gx] = weighted_sum / weight_sum;
}
