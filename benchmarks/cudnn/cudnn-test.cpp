#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>

// nvcc cudnn.cpp -lcudnn && ./a.out

#define CHECK_CUDNN(expression)                             \
  {                                                         \
    cudnnStatus_t status = (expression);                    \
    if (status != CUDNN_STATUS_SUCCESS) {                   \
      std::cerr << "Error on line " << __LINE__ << ": "     \
                << cudnnGetErrorString(status) << std::endl;\
      std::exit(EXIT_FAILURE);                              \
    }                                                       \
  }

void *alloc(size_t bytes) {
  void *ret = nullptr;
  if (!bytes)
    return ret;
  if (0 != cudaMalloc(&ret, bytes))
    ret = nullptr;
  return ret;
}

int double_size(std::vector<int> data) {
  int ret = sizeof(double);
  for (int i = 0; i < data.size(); ++i)
    ret *= data[i];
  return ret;
}

int *stride(std::vector<int> data) {
  static int array[1024];
  array[data.size() - 1] = 1;
  for (int i = data.size() - 2; i >= 0; --i)
    array[i] = array[i + 1] * data[i + 1];
  return array;
}

int main() {
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));

// #define CUDNN_DATA_DOUBLE CUDNN_DATA_FLOAT

    std::vector<int> image = {1, 1, 432000, 1}, kernel = {1, 1, 11, 1};

    CHECK_CUDNN(cudnnSetTensorNdDescriptor(input_descriptor,
                                           CUDNN_DATA_DOUBLE,
                                           image.size(),
                                           image.data(),
                                           stride(image)));

    cudnnFilterDescriptor_t filter_descriptor;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(filter_descriptor,
                                           CUDNN_DATA_DOUBLE,
                                           CUDNN_TENSOR_NCHW,
                                           kernel.size(),
                                           kernel.data()));

    std::vector<int> pad = {0, 0, 0, 0, 0}, dial = {1, 1, 1, 1, 1}, st = {1, 1, 1, 1, 1};

    cudnnConvolutionDescriptor_t convolution_descriptor;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(convolution_descriptor,
                                                kernel.size() - 2,
                                                pad.data(),
                                                dial.data(),
                                                st.data(),
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_DOUBLE));

    std::vector<int> output_dim(image.size());
    CHECK_CUDNN(cudnnGetConvolutionNdForwardOutputDim(convolution_descriptor,
                                                      input_descriptor,
                                                      filter_descriptor,
                                                      output_dim.size(),
                                                      output_dim.data()));

    CHECK_CUDNN(cudnnSetTensorNdDescriptor(output_descriptor,
                                           CUDNN_DATA_DOUBLE,
                                           output_dim.size(),
                                           output_dim.data(),
                                           stride(output_dim)));

    double alpha = 1.0f, beta = 0.0f;

    static const char *algo_names[] = {
      "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
      "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
      "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
      "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
      "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
      "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
      "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
      "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
      "(null)"
    };

    for (int conv_algo = 0; conv_algo < CUDNN_CONVOLUTION_FWD_ALGO_COUNT; ++conv_algo) {
      size_t workspace_bytes{0};
      cudnnStatus_t ret = cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                          input_descriptor,
                                                          filter_descriptor,
                                                          convolution_descriptor,
                                                          output_descriptor,
                                                          (cudnnConvolutionFwdAlgo_t)conv_algo,
                                                          &workspace_bytes);
      if (ret == CUDNN_STATUS_NOT_SUPPORTED)
        printf("algo-%d: CUDNN_STATUS_NOT_SUPPORTED\n", conv_algo);
      else if (ret != CUDNN_STATUS_SUCCESS)
        printf("algo-%d: CUDNN_STATUS_ERROR\n", conv_algo);
      if (ret != CUDNN_STATUS_SUCCESS)
        continue;
      printf("algo-%d: CUDNN_STATUS_SUCCESS (workspace = %lld, name = %s)\n", conv_algo, (long long)workspace_bytes, algo_names[conv_algo]);

      void *in_workspace = alloc(workspace_bytes);
      void *in_image = alloc(double_size(image));
      void *in_kernel = alloc(double_size(kernel));
      void *out_image = alloc(double_size(output_dim));

      cudaEvent_t hStart, hStop;
      cudaEventCreate(&hStart);
      cudaEventCreate(&hStop);

      if ((workspace_bytes > 0 && !in_workspace) || !in_image || !in_kernel || !out_image)
        printf("  OUT_OF_MEMORY\n");
      else {
        auto run = [&]() {
          CHECK_CUDNN(cudnnConvolutionForward(cudnn,
                                              &alpha,
                                              input_descriptor,
                                              in_image,
                                              filter_descriptor,
                                              in_kernel,
                                              convolution_descriptor,
                                              (cudnnConvolutionFwdAlgo_t)conv_algo,
                                              in_workspace,
                                              workspace_bytes,
                                              &beta,
                                              output_descriptor,
                                              out_image));
        };

        for (int i = 0; i < 2; ++i)
          run();

        assert(0 == cudaDeviceSynchronize());
        assert(0 == cudaEventRecord(hStart));

        const int RUNS = 100;
        for (int i = 0; i < RUNS; ++i)
          run();
        assert(0 == cudaEventRecord(hStop));
        assert(0 == cudaDeviceSynchronize());

        float ms = -1;
        assert(0 == cudaEventElapsedTime(&ms, hStart, hStop));
        printf(" COST = %.3f msec\n", ms / RUNS);
      }
      cudaFree(in_workspace);
      cudaFree(in_image);
      cudaFree(in_kernel);
      cudaFree(out_image);

    }

    printf("\nOutput Shape = [");
    for (auto it: output_dim)
      printf("%d, ", it);
    puts("]");

}
