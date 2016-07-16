extern "C"
__global__ void haar_type_A(int** globalInputData, int posx, int posy, int sizex, int sizey, int* globalOutputData)
{
    // Get an "unique id" of the thread that correspond to one pixel
    const unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int x = tidX / 24;
    const unsigned int y = tidX % 24;

    if (tidX < (24 - sizex + 1) * (24 - sizey + 1))
    {
        int a = 0;
        for (int i = 0; i < sizex / 2; i++)
            for (int j = 0; j < sizey; j++)
                a += globalInputData[posx + x + i][posy + y + j];

        int b = 0;
        for (int i = sizex / 2; i < sizex; i++)
            for (int j = 0; j < sizey; j++)
                b += globalInputData[posx + x + i][posy + y + j];

        globalOutputData[tidX] = a - b;
    }

    __syncthreads();
}
