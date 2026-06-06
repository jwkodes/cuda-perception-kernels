#include <gtest/gtest.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "kernels/bilateral_filter.cuh"

constexpr int WIDTH = 64;
constexpr int HEIGHT = 64;
constexpr int LENGTH = WIDTH * HEIGHT;
constexpr float SIGMA_S = 3.0f;
constexpr float SIGMA_R = 0.5f;
constexpr int BLOCK_DIM = 16;
constexpr int GRID_DIM_Y = (HEIGHT + BLOCK_DIM - 1) / BLOCK_DIM;
constexpr int GRID_DIM_X = (WIDTH + BLOCK_DIM - 1) / BLOCK_DIM;

std::vector<float> apply_bilateral_filter(const std::vector<float> &image)
{
    const float *h_image = &image[0];

    // Prepare device variables
    float *d_image = nullptr;
    cudaMalloc(&d_image, LENGTH * sizeof(float));
    cudaMemcpy(d_image, h_image, LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    float *d_filtered = nullptr;
    cudaMalloc(&d_filtered, LENGTH * sizeof(float));

    // Filter
    dim3 grid(GRID_DIM_X, GRID_DIM_Y);
    bilateral_filter<<<grid, dim3(BLOCK_DIM, BLOCK_DIM)>>>(d_image, d_filtered, WIDTH, HEIGHT, SIGMA_S, SIGMA_R);
    cudaDeviceSynchronize(); // ensure memcpy does not begin after processing is done

    // Copy result to cpu and free device memory
    std::vector<float> filtered;
    filtered.resize(LENGTH);
    cudaMemcpy(&filtered[0], d_filtered, LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    cudaFree(d_filtered);

    return filtered;
}

TEST(BilateralFilterTest, HandlesHardEdge)
{
    // Prepare image with horizontal discontinuity in the middle
    std::vector<float> image(LENGTH, 1.0f);
    std::fill(image.begin() + LENGTH / 2, image.end(), 2.0f);

    std::vector<float> filtered = apply_bilateral_filter(image);

    // Perform the same filtering with OpenCV
    cv::Mat image_mat(HEIGHT, WIDTH, CV_32FC1, image.data()); // shallow copy
    cv::Mat filtered_mat;
    cv::bilateralFilter(image_mat, filtered_mat, RADIUS * 2 + 1, SIGMA_R, SIGMA_S, cv::BORDER_REPLICATE);
    float *filtered_mat_ptr = filtered_mat.ptr<float>(0);

    // Compare original image with filtered image
    for (int i = 0; i < LENGTH; i++)
    {
        ASSERT_NEAR(filtered_mat_ptr[i], filtered[i], 1e-2f);
    }
}

TEST(BilateralFilterTest, HandlesFlatImage)
{
    // Prepare image
    // Image of length 64x64 with constant value of 1.0f
    std::vector<float> image(LENGTH, 1.0f);

    std::vector<float> filtered = apply_bilateral_filter(image);

    // Compare original image with filtered image
    for (int i = 0; i < LENGTH; i++)
    {
        ASSERT_NEAR(image[i], filtered[i], 1e-2f);
    }
}