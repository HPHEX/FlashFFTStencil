// 8 * 7
// fast complex

#include <mma.h>

#include "../create_fft_pfa_plan.cu"
#include "../elementwise_mul.cuh"

#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 4

#define WARP_SIZE 32

using namespace nvcuda;

template <int ROW_IN_BLOCK, int COL_IN_BLOCK, int N_WARP>
__global__ void rfft_pfa_stencil_8_nwarp(
    const double *__restrict__ input,

    const int SUB_INPUT_SIZE,
    const int OVERLAP_SIZE,
    const int fft_n_max,

    double *output)
{
    extern __shared__ double sharedmem[];

    const int bid = blockIdx.x;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int fft_id = bid * N_WARP + warp_id; // 0 - fft_n_max

    const int laneId = threadIdx.x & 0x1f;

    double *reindex_input = &sharedmem[0 + warp_id * rfft_size];
    double *temp_shared = &sharedmem[rfft_size * N_WARP + warp_id * rfft_size];

    // reindex input
#pragma unroll
    for (size_t i = laneId; i < pfa_size; i += WARP_SIZE)
    {
        const int sharedmem_index = (i % ROW_IN_BLOCK) * ROW_IN_BLOCK + (i % COL_IN_BLOCK);

        reindex_input[sharedmem_index] = input[fft_id * fft_size + i];
        temp_shared[sharedmem_index] = input[fft_id * fft_size + rfft_size + i];
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag_real[2];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag_imag[2];

    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag_real[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag_imag[2];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_1;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_1;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> A;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> B;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> C;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> new_A;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> new_B;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> new_C;

    /**
     * @brief f f t
     *
     */
    wmma::fill_fragment(A, 0.0);
    wmma::fill_fragment(B, 0.0);
    wmma::fill_fragment(C, 0.0);
    // 8 * 8 @ 8 * 8（7）
    // dht_matrix_row @ reindex_input
#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        wmma::load_matrix_sync(a_frag_real[i], dft_matrix_real_1 + WMMA_K * i, ROW_IN_BLOCK); // a
        wmma::load_matrix_sync(a_frag_imag[i], dft_matrix_imag_1 + WMMA_K * i, ROW_IN_BLOCK); // b

        wmma::load_matrix_sync(b_frag_real[i], reindex_input + i * WMMA_K * ROW_IN_BLOCK, ROW_IN_BLOCK); // c
        wmma::load_matrix_sync(b_frag_imag[i], temp_shared + i * WMMA_K * ROW_IN_BLOCK, ROW_IN_BLOCK); // d

        a_1.x[0] = a_frag_real[i].x[0] + a_frag_imag[i].x[0];

        wmma::mma_sync(A, a_1, b_frag_real[i], A); // (a+b) * c

        a_1.x[0] = a_frag_imag[i].x[0] - a_frag_real[i].x[0];

        wmma::mma_sync(C, a_1, b_frag_imag[i], C); // (b-a) * d

        b_1.x[0] = b_frag_real[i].x[0] + b_frag_imag[i].x[0];

        wmma::mma_sync(B, a_frag_imag[i], b_1, B); // b * (c+d)
    }
#pragma unroll
    for (int i = 0; i < A.num_elements; i++)
    {
        new_A.x[i] = A.x[i] - C.x[i];
        new_B.x[i] = B.x[i] - C.x[i];
        new_C.x[i] = 2 * B.x[i] - A.x[i] - C.x[i];
    }

    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    wmma::fill_fragment(A, 0.0);
    wmma::fill_fragment(B, 0.0);
    wmma::fill_fragment(C, 0.0);
    // 8 * 8（7） @  8 * 8
    // reindex_input @ dht_matrix_col
#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        wmma::load_matrix_sync(b_frag_real[i], dft_matrix_real_2 + i * WMMA_K * ROW_IN_BLOCK, ROW_IN_BLOCK); // c
        wmma::load_matrix_sync(b_frag_imag[i], dft_matrix_imag_2 + i * WMMA_K * ROW_IN_BLOCK, ROW_IN_BLOCK); // d

        a_1.x[0] = new_A.x[i];
        wmma::mma_sync(A, a_1, b_frag_real[i], A); // (a+b) * c

        a_1.x[0] = new_C.x[i];
        wmma::mma_sync(C, a_1, b_frag_imag[i], C); // (b-a) * d

        a_1.x[0] = new_B.x[i];
        b_1.x[0] = b_frag_real[i].x[0] + b_frag_imag[i].x[0];
        wmma::mma_sync(B, a_1, b_1, B); // b * (c+d)
    }

    // --------------------------------------------------------------------------------------------------------------
    // -------------------------------kernel multiplication----------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

#pragma unroll
    for (int i = 0; i < A.num_elements; i++)
    {
        new_A.x[i] = A.x[i] - B.x[i]; // real
        new_B.x[i] = B.x[i] - C.x[i]; // imag
    }

    // element-wise complex multiplication between fft & ifft
    // for k_f
    wmma::fill_fragment(A, 0.0);
    wmma::fill_fragment(B, 0.0);
    wmma::load_matrix_sync(A, kernel_fft_real, ROW_IN_BLOCK, wmma::mem_row_major);
    wmma::load_matrix_sync(B, kernel_fft_imag, ROW_IN_BLOCK, wmma::mem_row_major);
    // Compute the complex multiplication
#pragma unroll
    for (int i = 0; i < A.num_elements; i++)
    {
        complexMul(new_A.x[i], new_B.x[i], A.x[i], B.x[i],
                   &new_A.x[i], &new_B.x[i]);

        A.x[i] = new_A.x[i] + new_B.x[i];
        B.x[i] = new_B.x[i];
        C.x[i] = new_B.x[i] - new_A.x[i];
    }

    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    /**
     * @brief i f h t
     *
     */
    // 8 * 8（7） @  8 * 8
    // reindex_input @ idht_matrix_col
    wmma::fill_fragment(new_A, 0.0);
    wmma::fill_fragment(new_B, 0.0);
    wmma::fill_fragment(new_C, 0.0);
#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        // wmma::load_matrix_sync(b_frag_real[i], idft_matrix_real_2 + i * WMMA_K * ROW_IN_BLOCK, ROW_IN_BLOCK); // c
        // wmma::load_matrix_sync(b_frag_imag[i], idft_matrix_imag_2 + i * WMMA_K * ROW_IN_BLOCK, ROW_IN_BLOCK); // d

        a_1.x[0] = A.x[i];
        wmma::mma_sync(new_A, a_1, b_frag_real[i], new_A); // (a+b) * c

        a_1.x[0] = C.x[i];
        b_frag_imag[i].x[0] = -b_frag_imag[i].x[0];
        wmma::mma_sync(new_C, a_1, b_frag_imag[i], new_C); // (b-a) * d

        a_1.x[0] = B.x[i];
        b_1.x[0] = b_frag_real[i].x[0] + b_frag_imag[i].x[0];
        wmma::mma_sync(new_B, a_1, b_1, new_B); // b * (c+d)
    }
#pragma unroll
    for (int i = 0; i < A.num_elements; i++)
    {
        A.x[i] = new_A.x[i] - new_B.x[i]; // real
        B.x[i] = new_B.x[i] - new_C.x[i]; // imag
    }

    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    wmma::store_matrix_sync(reindex_input, A, ROW_IN_BLOCK, wmma::mem_row_major);
    wmma::store_matrix_sync(temp_shared, B, ROW_IN_BLOCK, wmma::mem_row_major);

    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    wmma::fill_fragment(A, 0.0);
    wmma::fill_fragment(B, 0.0);
    wmma::fill_fragment(C, 0.0);
    // 8 * 8 @ 8 * 8（7）
    // idht_matrix_row @ reindex_input
#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        // wmma::load_matrix_sync(a_frag_real, idft_matrix_real_1 + WMMA_K * i, ROW_IN_BLOCK); // a
        // wmma::load_matrix_sync(a_frag_imag, idft_matrix_imag_1 + WMMA_K * i, ROW_IN_BLOCK); // b

        a_frag_imag[i].x[0] = - a_frag_imag[i].x[0];

        wmma::load_matrix_sync(b_frag_real[i], reindex_input + i * WMMA_K * ROW_IN_BLOCK, ROW_IN_BLOCK); // c
        wmma::load_matrix_sync(b_frag_imag[i], temp_shared + i * WMMA_K * ROW_IN_BLOCK, ROW_IN_BLOCK);   // d

        a_1.x[0] = a_frag_real[i].x[0] + a_frag_imag[i].x[0];

        wmma::mma_sync(A, a_1, b_frag_real[i], A); // (a+b) * c

        a_1.x[0] = a_frag_imag[i].x[0] - a_frag_real[i].x[0];

        wmma::mma_sync(C, a_1, b_frag_imag[i], C); // (b-a) * d

        b_1.x[0] = b_frag_real[i].x[0] + b_frag_imag[i].x[0];

        wmma::mma_sync(B, a_frag_imag[i], b_1, B); // b * (c+d)
    }
#pragma unroll
    for (int i = 0; i < A.num_elements; i++)
    {

        new_A.x[i] = A.x[i] - B.x[i]; // real
        new_B.x[i] = B.x[i] - C.x[i]; // imag
    }

    wmma::store_matrix_sync(reindex_input, new_A, ROW_IN_BLOCK, wmma::mem_row_major);
    wmma::store_matrix_sync(temp_shared, new_B, ROW_IN_BLOCK, wmma::mem_row_major);


    if (laneId < OVERLAP_SIZE)
    {
        const int sharedmem_index1 = (laneId % ROW_IN_BLOCK) * ROW_IN_BLOCK + (laneId % COL_IN_BLOCK);
        int i = laneId + pfa_size - OVERLAP_SIZE;
        const int sharedmem_index2 = (i % ROW_IN_BLOCK) * ROW_IN_BLOCK + (i % COL_IN_BLOCK);

        reindex_input[sharedmem_index2] += temp_shared[sharedmem_index1];
    }


    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    if (bid == 0) // the first one
    {
        for (size_t i = laneId; i < pfa_size; i += WARP_SIZE)
        {
            const int sharedmem_index = (i % ROW_IN_BLOCK) * ROW_IN_BLOCK + (i % COL_IN_BLOCK);

            if (i < OVERLAP_SIZE)
            {
                atomicAdd(output + fft_id * 2 * SUB_INPUT_SIZE + i, reindex_input[sharedmem_index]);
                // atomicAdd(output + (fft_id * 2 + 1) * SUB_INPUT_SIZE + i, temp_shared[sharedmem_index]);
            }
            else if (i < SUB_INPUT_SIZE)
            {
                output[fft_id * 2 * SUB_INPUT_SIZE + i] = reindex_input[sharedmem_index];
                output[(fft_id * 2 + 1) * SUB_INPUT_SIZE + i] = temp_shared[sharedmem_index];
            }
            else if (i < pfa_size)
            {
                // atomicAdd(output + fft_id * 2 * SUB_INPUT_SIZE + i, reindex_input[sharedmem_index]);
                output[fft_id * 2 * SUB_INPUT_SIZE + i] = reindex_input[sharedmem_index];
                atomicAdd(output + (fft_id * 2 + 1) * SUB_INPUT_SIZE + i, temp_shared[sharedmem_index]);
            }
        }
    }
    else if (bid == gridDim.x - 1) // the last one
    {
        for (size_t i = laneId; i < pfa_size; i += WARP_SIZE)
        {
            const int sharedmem_index = (i % ROW_IN_BLOCK) * ROW_IN_BLOCK + (i % COL_IN_BLOCK);
            if (i < OVERLAP_SIZE)
            {
                atomicAdd(output + fft_id * 2 * SUB_INPUT_SIZE + i, reindex_input[sharedmem_index]);
                // atomicAdd(output + (fft_id * 2 + 1) * SUB_INPUT_SIZE + i, temp_shared[sharedmem_index]);
            }
            else if (i < SUB_INPUT_SIZE)
            {
                output[fft_id * 2 * SUB_INPUT_SIZE + i] = reindex_input[sharedmem_index];
                output[(fft_id * 2 + 1) * SUB_INPUT_SIZE + i] = temp_shared[sharedmem_index];
            }
            else if (i < pfa_size)
            {
                // TODO
                if (fft_id == fft_n_max)
                {
                    // atomicAdd(output + fft_id * 2 * SUB_INPUT_SIZE + i, reindex_input[sharedmem_index]);
                    output[fft_id * 2 * SUB_INPUT_SIZE + i] = reindex_input[sharedmem_index];
                    atomicAdd(output + i - SUB_INPUT_SIZE, temp_shared[sharedmem_index]);
                }
                else
                {
                    // atomicAdd(output + fft_id * 2 * SUB_INPUT_SIZE + i, reindex_input[sharedmem_index]);
                    output[fft_id * 2 * SUB_INPUT_SIZE + i] = reindex_input[sharedmem_index];
                    atomicAdd(output + (fft_id * 2 + 1) * SUB_INPUT_SIZE + i, temp_shared[sharedmem_index]);
                }
            }
        }
    }
    else
    {
        for (size_t i = laneId; i < pfa_size; i += WARP_SIZE)
        {
            const int sharedmem_index = (i % ROW_IN_BLOCK) * ROW_IN_BLOCK + (i % COL_IN_BLOCK);
            if (i < OVERLAP_SIZE)
            {
                atomicAdd(output + fft_id * 2 * SUB_INPUT_SIZE + i, reindex_input[sharedmem_index]);
                // atomicAdd(output + (fft_id * 2 + 1) * SUB_INPUT_SIZE + i, temp_shared[sharedmem_index]);
            }
            else if (i < SUB_INPUT_SIZE)
            {
                output[fft_id * 2 * SUB_INPUT_SIZE + i] = reindex_input[sharedmem_index];
                output[(fft_id * 2 + 1) * SUB_INPUT_SIZE + i] = temp_shared[sharedmem_index];
            }
            else if (i < pfa_size)
            {
                // atomicAdd(output + fft_id * 2 * SUB_INPUT_SIZE + i, reindex_input[sharedmem_index]);
                output[fft_id * 2 * SUB_INPUT_SIZE + i] = reindex_input[sharedmem_index];
                atomicAdd(output + (fft_id * 2 + 1) * SUB_INPUT_SIZE + i, temp_shared[sharedmem_index]);
            }
        }
    }
}