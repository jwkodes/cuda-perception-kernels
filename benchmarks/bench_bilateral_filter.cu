#include <benchmark/benchmark.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "kernels/bilateral_filter.cuh"

constexpr int WIDTH = 1920;
constexpr int HEIGHT = 1080;
constexpr int LENGTH = WIDTH * HEIGHT;
constexpr float SIGMA_S = 3.0f;
constexpr float SIGMA_R = 0.5f;
constexpr int BLOCK_DIM = 16;
constexpr int GRID_DIM_Y = (HEIGHT + BLOCK_DIM - 1) / BLOCK_DIM;
constexpr int GRID_DIM_X = (WIDTH + BLOCK_DIM - 1) / BLOCK_DIM;

static void BM_BilateralFilterCUDA(benchmark::State &state)
{
    // Setup
    std::vector<float> image = std::vector<float>(LENGTH, 1.0f);
    const float *h_image = &image[0];

    // Prepare device variables
    float *d_image = nullptr;
    cudaMalloc(&d_image, LENGTH * sizeof(float));
    cudaMemcpy(d_image, h_image, LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    float *d_filtered = nullptr;
    cudaMalloc(&d_filtered, LENGTH * sizeof(float));

    // Filter, warm up
    dim3 grid(GRID_DIM_X, GRID_DIM_Y);
    bilateral_filter<<<grid, dim3(BLOCK_DIM, BLOCK_DIM)>>>(d_image, d_filtered, WIDTH, HEIGHT, SIGMA_S, SIGMA_R);
    cudaDeviceSynchronize(); // ensure memcpy does not begin after processing is done

    for (auto _ : state)
    {
        bilateral_filter<<<grid, dim3(BLOCK_DIM, BLOCK_DIM)>>>(d_image, d_filtered, WIDTH, HEIGHT, SIGMA_S, SIGMA_R);
        cudaDeviceSynchronize(); // ensure memcpy does not begin after processing is done

        // // Copy result to cpu and free device memory
        // std::vector<float> filtered;
        // filtered.resize(LENGTH);
        // cudaMemcpy(&filtered[0], d_filtered, LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_image);
    cudaFree(d_filtered);
}

static void BM_BilateralFilterOpenCV(benchmark::State &state)
{
    // Setup
    std::vector<float> image(LENGTH, 1.0f);
    cv::Mat image_mat(HEIGHT, WIDTH, CV_32FC1, image.data()); // shallow copy
    cv::Mat filtered_mat;

    for (auto _ : state)
    {
        cv::bilateralFilter(image_mat, filtered_mat, RADIUS * 2 + 1, SIGMA_R, SIGMA_S, cv::BORDER_REPLICATE);
    }
}

BENCHMARK(BM_BilateralFilterCUDA);
BENCHMARK(BM_BilateralFilterOpenCV);
BENCHMARK_MAIN();