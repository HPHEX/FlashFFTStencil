#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cufft.h>
#include "./helper_cuda/helper_cuda.h"

__constant__ double dft_matrix_real_1[fft_size];
__constant__ double dft_matrix_imag_1[fft_size];

__constant__ double dft_matrix_real_2[fft_size];
__constant__ double dft_matrix_imag_2[fft_size];

__constant__ double kernel_fft_real[fft_size];
__constant__ double kernel_fft_imag[fft_size];

// __constant__ int reindex_lookup[fft_size];

void CreatePlan(
    double *k,
    const int KERNEL_SIZE,
    bool print_matrix)
{
    // row > col
    // fft_size = row * row
    // dim 是 8 的倍数
    const int dim = row;
    // const int fft_size = dim * dim;
    // const int pfa_size = row * col;

    double *host_dft_matrix_real_1 = (double *)malloc(fft_size * sizeof(double));
    double *host_dft_matrix_imag_1 = (double *)malloc(fft_size * sizeof(double));

    double *host_dft_matrix_real_2 = (double *)malloc(fft_size * sizeof(double));
    double *host_dft_matrix_imag_2 = (double *)malloc(fft_size * sizeof(double));

    double *host_idft_matrix_real_1 = (double *)malloc(fft_size * sizeof(double));
    double *host_idft_matrix_imag_1 = (double *)malloc(fft_size * sizeof(double));

    double *host_idft_matrix_real_2 = (double *)malloc(fft_size * sizeof(double));
    double *host_idft_matrix_imag_2 = (double *)malloc(fft_size * sizeof(double));

    for (int i = 0; i < dim; ++i)
    {

        int yushu = i % 8;
        int beishu = i / 8;
        int row_index = beishu * 8 + yushu % 2 * 4 + yushu / 2;
        // int col_index = beishu * 8 + yushu % 2 * 4 + yushu / 2;

        for (int j = 0; j < dim; ++j)
        {
            host_dft_matrix_real_1[dim * i + j] = cos(2 * M_PI * i * j / row);
            host_dft_matrix_imag_1[dim * i + j] = -sin(2 * M_PI * i * j / row);

            host_idft_matrix_real_1[dim * i + j] = cos(2 * M_PI * i * j / row);
            host_idft_matrix_imag_1[dim * i + j] = sin(2 * M_PI * i * j / row);

            if (i == dim - 1 || j == dim - 1)
            {
                host_dft_matrix_real_2[dim * row_index + j] = 0.0;
                host_dft_matrix_imag_2[dim * row_index + j] = 0.0;

                host_idft_matrix_real_2[dim * row_index + j] = 0.0;
                host_idft_matrix_imag_2[dim * row_index + j] = 0.0;
            }
            else
            {
                host_dft_matrix_real_2[dim * row_index + j] = cos(2 * M_PI * i * j / col);
                host_dft_matrix_imag_2[dim * row_index + j] = -sin(2 * M_PI * i * j / col);

                host_idft_matrix_real_2[dim * row_index + j] = cos(2 * M_PI * i * j / col);
                host_idft_matrix_imag_2[dim * row_index + j] = sin(2 * M_PI * i * j / col);
            }
        }
    }

    if (print_matrix)
    {
        printf("DFT row Matrix 1\n");
        for (int i = 0; i < dim; ++i)
        {

            for (int j = 0; j < dim; ++j)
            {
                printf("(%f, %f)\t", host_dft_matrix_real_1[dim * i + j], host_dft_matrix_imag_1[dim * i + j]);
            }
            printf("\n");
        }
        std::cout << std::endl;

        printf("DFT col Matrix 2\n");
        for (int i = 0; i < dim; ++i)
        {

            for (int j = 0; j < dim; ++j)
            {
                printf("(%f, %f)\t", host_dft_matrix_real_2[dim * i + j], host_dft_matrix_imag_2[dim * i + j]);
            }
            printf("\n");
        }
        std::cout << std::endl;

        printf("IDFT row Matrix 1\n");
        for (int i = 0; i < dim; ++i)
        {

            for (int j = 0; j < dim; ++j)
            {
                printf("(%f, %f)\t", host_idft_matrix_real_1[dim * i + j], host_idft_matrix_imag_1[dim * i + j]);
            }
            printf("\n");
        }
        std::cout << std::endl;

        printf("IDFT col Matrix 2\n");
        for (int i = 0; i < dim; ++i)
        {

            for (int j = 0; j < dim; ++j)
            {
                printf("(%f, %f)\t", host_idft_matrix_real_2[dim * i + j], host_idft_matrix_imag_2[dim * i + j]);
            }
            printf("\n");
        }
        std::cout << std::endl;
    }

    // cuda malloc
    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_real_1, host_dft_matrix_real_1, fft_size * sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_imag_1, host_dft_matrix_imag_1, fft_size * sizeof(double)));

    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_real_2, host_dft_matrix_real_2, fft_size * sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_imag_2, host_dft_matrix_imag_2, fft_size * sizeof(double)));

    // checkCudaErrors(cudaMemcpyToSymbol(idft_matrix_real_1, host_idft_matrix_real_1, fft_size * sizeof(double)));
    // checkCudaErrors(cudaMemcpyToSymbol(idft_matrix_imag_1, host_idft_matrix_imag_1, fft_size * sizeof(double)));

    // checkCudaErrors(cudaMemcpyToSymbol(idft_matrix_real_2, host_idft_matrix_real_2, fft_size * sizeof(double)));
    // checkCudaErrors(cudaMemcpyToSymbol(idft_matrix_imag_2, host_idft_matrix_imag_2, fft_size * sizeof(double)));

    // part 2: create FFT kernel
    double2 *h_kernel = (double2 *)malloc(KERNEL_SIZE * sizeof(double2));
    double2 *h_fft_kernel = (double2 *)malloc(pfa_size * sizeof(double2));
    for (size_t i = 0; i < KERNEL_SIZE; i++)
    {
        h_kernel[i] = make_double2(k[i], 0.0);
    }

    double2 *d_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_kernel, KERNEL_SIZE * sizeof(double2)));
    double2 *d_fft_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_fft_kernel, pfa_size * sizeof(double2)));

    checkCudaErrors(cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE * sizeof(double2), cudaMemcpyHostToDevice));

    cufftHandle plan_1d_fft;
    checkCudaErrors(cufftPlan1d(&plan_1d_fft, pfa_size, CUFFT_Z2Z, 1));
    checkCudaErrors(cufftExecZ2Z(plan_1d_fft, (cufftDoubleComplex *)d_kernel, (cufftDoubleComplex *)d_fft_kernel, CUFFT_FORWARD));

    checkCudaErrors(cudaMemcpy(h_fft_kernel, d_fft_kernel, pfa_size * sizeof(double2), cudaMemcpyDeviceToHost));

    double *h_kernel_fft_real = (double *)malloc(fft_size * sizeof(double));
    double *h_kernel_fft_imag = (double *)malloc(fft_size * sizeof(double));
    // reindex
    for (int i = 0; i < fft_size; ++i)
    {
        h_kernel_fft_real[i] = 0.0;
        h_kernel_fft_imag[i] = 0.0;
    }
    for (int i = 0; i < pfa_size; ++i)
    {
        int row_index = ((-i) % row + row) % row;
        int col_index = i % col;

        h_kernel_fft_real[row_index * row + col_index] = h_fft_kernel[i].x / pfa_size;
        h_kernel_fft_imag[row_index * row + col_index] = h_fft_kernel[i].y / pfa_size;
    }

    if (print_matrix)
    {
        printf("Kernel_FFT Reindex Matrix\n");
        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                printf("(%f, %f) \t", h_kernel_fft_real[dim * i + j], h_kernel_fft_imag[dim * i + j]);
            }
            printf("\n");
        }
        std::cout << std::endl;
    }

    checkCudaErrors(cudaMemcpyToSymbol(kernel_fft_real, h_kernel_fft_real, fft_size * sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(kernel_fft_imag, h_kernel_fft_imag, fft_size * sizeof(double)));


    // // part 3: create reindex lookup table
    // int *host_reindex_lookup = (int *)malloc(fft_size * sizeof(int));
    // for (int i = 0; i < fft_size; ++i)
    // {
    //     if (i < pfa_size)
    //     {
    //         int row_index = i % row;
    //         int col_index = i % col;
    //         host_reindex_lookup[i] = row_index * row + col_index;
    //     }
    //     else
    //     {
    //         host_reindex_lookup[i] = 0;
    //     }
    // }


    // checkCudaErrors(cudaMemcpyToSymbol(reindex_lookup, host_reindex_lookup, fft_size * sizeof(int)));

    // free(host_dft_matrix_real_1);
    // free(host_dft_matrix_imag_1);

    // free(host_dft_matrix_real_2);
    // free(host_dft_matrix_real_2);

    // free(host_idft_matrix_real_1);
    // free(host_idft_matrix_imag_1);

    // free(host_idft_matrix_real_2);
    // free(host_idft_matrix_imag_2);
}