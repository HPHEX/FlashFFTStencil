#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cufft.h>
#include <vector>
#include <algorithm>

#include "check_correct.hpp"
#include "helper_cuda/helper_cuda.h"

#define rfft_size (64)
#define fft_size (128)
#define pfa_size (56)

#define row 8
#define col 7
#define nwarp_in_block 1

#include "rfftstencil_pfa/rfft_8_pfa_fastcomplex.cu"


void printHelp()
{
    const char *helpMessage =
        "Program name: FlashFFTStencil-1D\n"
        "Usage: a.out [stencil-shape] [input_size] [time_step] \n"
        "Stencil-shape: Heat-1D / 1D5P / 1D7P \n";
    printf("%s\n", helpMessage);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printHelp();
        return 1;
    }

    int KERNEL_SIZE = -1;
    const int time = std::stoi(argv[3]);
    int after_fusion_time = -1;
    
    std::string kernel_shape = argv[1];
    if (kernel_shape == "Heat-1D")
    {
        KERNEL_SIZE = 29;
        after_fusion_time = time / 14;
    }
    else if (kernel_shape == "1D5P")
    {
        KERNEL_SIZE = 29;
        after_fusion_time = time / 7;
    }
    else if (kernel_shape == "1D7P")
    {
        KERNEL_SIZE = 31;
        after_fusion_time = time / 5;
    }
    else 
    {
        std::cout << "FlashFFTStencil can support stencil calculations for more shapes, but we haven't implemented it yet. "
                  << "We will consider implementing more in the future." << std::endl;
        return 1;
    }

    const long long INPUT_SIZE = std::stoi(argv[2]);
    
    const bool is_print_data = false;

    const int sub_input_size = pfa_size - (KERNEL_SIZE - 1); // TODO : 54,
    // const int sub_input_size = pfa_size;
    const int OVERLAP_SIZE = KERNEL_SIZE - 1;

    if (INPUT_SIZE % sub_input_size != 0)
    {

        std::cerr << "Please re-enter the input size" << std::endl;
        std::cerr << "input_size % subinput_size != 0" << std::endl;
        std::cerr << "subinput_size = " << sub_input_size << std::endl;
        std::cerr << "input_size = " << INPUT_SIZE << std::endl;
        return 0.0;
    }
    else
    {
        std::cout << "INFO: stencil kernel shape = " << kernel_shape << std::endl;
        std::cout << "INFO: input size = " << INPUT_SIZE << std::endl;
        std::cout << "INFO: times step = " << time << std::endl;
    }

    const int rfft_allnum = INPUT_SIZE / sub_input_size;

    const int fft_allnum = rfft_allnum / 2;
    
    const int block_num = fft_allnum / nwarp_in_block;

    // pfa_size = 56
    // sub_input_size = 56 - 2 = 54
    // rfft_size = 64
    const int gpu_input_size = rfft_allnum * rfft_size;

    // malloc
    size_t mem_size_input_gpu = gpu_input_size * sizeof(double);
    size_t mem_size_output = INPUT_SIZE * sizeof(double);

    double *h_input_gpu = (double *)calloc(gpu_input_size, sizeof(double));

    double *h_input_cpu = (double *)calloc(INPUT_SIZE, sizeof(double));

    double *h_output = (double *)calloc(INPUT_SIZE, sizeof(double));

    std::vector<double> h_kernel(KERNEL_SIZE);

    // 初始化输入数据
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        if (is_print_data)
        {
            h_input_cpu[i] = static_cast<double>(i + 1);
            std::cout << "INPUT data " << i << " : " << h_input_cpu[i] << std::endl;
        }
        else
        {
            h_input_cpu[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        }

        int index_for_inputgpu = (i / sub_input_size) * rfft_size + i % sub_input_size;
        h_input_gpu[index_for_inputgpu] = h_input_cpu[i];
    }
    for (int i = 0; i < KERNEL_SIZE; i++)
    {
        if (is_print_data)
        {
            h_kernel[i] = static_cast<double>((i + 1));
            std::cout << "KERNEL data " << i << " : " << h_kernel[i] << std::endl;
        }
        else
        {
            h_kernel[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        }
    }

    CreatePlan(h_kernel.data(), KERNEL_SIZE, false);

    // rdftstencil_handle handle;

    // compute_handle(handle, false);

    // malloc device memory
    double *d_input;
    checkCudaErrors(cudaMalloc((void **)&d_input, mem_size_input_gpu));
    double *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_output, mem_size_output));

    checkCudaErrors(cudaMemcpy(d_input, h_input_gpu, mem_size_input_gpu, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));

    for (int i= 0; i < after_fusion_time; i++)
    {

    rfft_pfa_stencil_8_nwarp<row, col, nwarp_in_block><<<block_num, nwarp_in_block * WARP_SIZE, 
                    (nwarp_in_block * 2 * 64) * sizeof(double)>>>(
                // (64 * 4 + nwarp_in_block * 2 * 64) * sizeof(double)>>>(
        d_input,
        sub_input_size,
        OVERLAP_SIZE,
        fft_allnum - 1,
        // handle,
        d_output);
        
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    // compute time
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Time = " << elapsedTime << "[ms]" << std::endl;
    printf("GStencil/s = %f\n\n", (INPUT_SIZE * ((KERNEL_SIZE -1) / 2)) * after_fusion_time / elapsedTime / 1e6);

    cudaMemcpy(h_output, d_output, mem_size_output, cudaMemcpyDeviceToHost);

    if (is_print_data)
    {
        std::cout << std::endl;
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            printf("%f\t", h_output[i]);
            // std::cout << "REF data " << i << " : " << ref_result[i] << std::endl;
        }
        // for (int i = 0; i < 8; ++i)
        // {

        //     for (int j = 0; j < 8; ++j)
        //     {
        //         printf("%f\t", h_output[8 * i + j]);
        //     }
        //     printf("\n");
        // }
        std::cout << std::endl;
    }

    // // 计算CPU结果
    // std::reverse(h_kernel.begin(), h_kernel.end());
    // std::vector<double> ref_result(INPUT_SIZE);
    // stencil1D(h_input_cpu, INPUT_SIZE, h_kernel.data(), KERNEL_SIZE, ref_result.data());
    // std::rotate(ref_result.rbegin(), ref_result.rbegin() + (KERNEL_SIZE / 2), ref_result.rend());

    // if (is_print_data)
    // {
    //     for (size_t i = 0; i < INPUT_SIZE; i++)
    //     {
    //         printf("%f\t", ref_result[i]);
    //     }
    // }

    // only works in 1 time step
    // if (areArraysEqual(h_output, ref_result.data(), INPUT_SIZE, 1e-7))
    // {
    //     std::cout << "Check correct!" << std::endl;
    // }
    // else
    // {
    //     std::cout << "Error: result wrong!" << std::endl;
    // }

    free(h_input_cpu);
    free(h_input_gpu);
    free(h_output);

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}
