__device__
int rectanglesSum(int** integralImage, int x, int y, int w, int h)
{
    int A = x > 0 && y > 0 ? integralImage[x - 1][y - 1] : 0;
    int B = x + w > 0 && y > 0 ? integralImage[x + w - 1][y - 1] : 0;
    int C = x > 0 && y + h > 0 ? integralImage[x - 1][y + h - 1] : 0;
    int D = x + w > 0 && y + h > 0 ? integralImage[x + w - 1][y + h - 1] : 0;

    return A + D - B - C;
}

extern "C"
__global__ void haar_type_A(int** integralImage, int* allRectangles, int numRectangles, float coeff, int* haarFeatures)
{
    // Get an "unique id" of the thread that correspond to one pixel
    const unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;

    if (tidX < numRectangles)
    {

        int x = (int) (allRectangles[tidX * 4] * coeff);
        int y = (int) (allRectangles[tidX * 4 + 1] * coeff);
        int w = (int) (allRectangles[tidX * 4 + 2] * coeff);
        int h = (int) (allRectangles[tidX * 4 + 3] * coeff);

        int mid = w / 2;

        int r1 = rectanglesSum(integralImage, x, y, mid, h);
        int r2 = rectanglesSum(integralImage, x + mid, y, mid, h);

        haarFeatures[tidX] = r1 - r2;
    }

    __syncthreads();
}
