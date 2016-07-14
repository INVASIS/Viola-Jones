extern "C"
__global__ void blur_filter(float** globalInputData, int height, float* globalOutputData)
{
    const unsigned int tidX = threadIdx.x + 1;

    for (int i = 1; i < height - 1; i++)
    {
        globalOutputData[tidX * height + i] =
            (globalInputData[tidX - 1][i - 1] + globalInputData[tidX][i - 1] + globalInputData[tidX + 1][i - 1] +
            globalInputData[tidX - 1][i] + globalInputData[tidX][i] + globalInputData[tidX + 1][i] +
            globalInputData[tidX - 1][i + 1] + globalInputData[tidX][i + 1] + globalInputData[tidX + 1][i + 1]) / 9;
    }

    __syncthreads();
}
