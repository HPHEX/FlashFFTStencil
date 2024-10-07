#include <cmath>

bool areArraysEqual(const double *array1, const double *array2, const int length, const double epsilon)
{
    for (int i = 0; i < length; ++i)
    {
        if (std::abs(array1[i] - array2[i]) > epsilon)
        {
            return false; // 如果差异超过 epsilon，认为不相等
        }
    }
    return true; // 所有元素都在误差范围内相等
}

void stencil1D(const double *input, const int input_size, const double *kernel, const int kernel_size, double *output)
{
    int half_kernel = kernel_size / 2;

    for (int i = 0; i < input_size; ++i)
    {
        output[i] = 0.0;
        for (int k = -half_kernel; k <= half_kernel; k++)
        {
            int idx = i + k;
            if (idx >= 0 && idx < input_size)
            {
                output[i] += input[idx] * kernel[k + half_kernel];
            }
            else if (idx < 0)
            {
                output[i] += input[idx + input_size] * kernel[k + half_kernel];
            }
            else if (idx >= input_size)
            {
                output[i] += input[idx - input_size] * kernel[k + half_kernel];
            }
        }
    }
}