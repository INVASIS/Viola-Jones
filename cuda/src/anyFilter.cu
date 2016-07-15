extern "C"
__global__ void any_filter(float** globalInputData, int width, int height, float** filter, float* globalOutputData)
{
    // Get an "unique id" of the thread that correcpond to one pixel
    const unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;

    if (tidX < width * height - 1)
    {
        const unsigned int x = tidX / height + 1;
        const unsigned int y = tidX % height + 1;

        globalOutputData[tidX] =
            filter[0][0] * globalInputData[x - 1][y - 1] + filter[1][0] * globalInputData[x][y - 1] + filter[2][0] * globalInputData[x + 1][y - 1] +
             filter[0][1] * globalInputData[x - 1][y] + filter[1][1] * globalInputData[x][y] + filter[2][1] * globalInputData[x + 1][y] +
             filter[0][2] * globalInputData[x - 1][y + 1] + filter[1][2] * globalInputData[x][y + 1] + filter[2][2] * globalInputData[x + 1][y + 1];

    }

    __syncthreads();
}
