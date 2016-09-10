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
	__global__
void computeWindowFeatures(int** integralImage, int* features, int totalNumFeatures, float* window, int* haarFeatures)
{
	// Get an "unique id" of the thread
	const unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;

	if (tidX < totalNumFeatures)
	{
		int type = features[threadIdx.x * 5];
		int x = features[threadIdx.x * 5 + 1] + (int)window[blockIdx.x * 2];
		int y = features[threadIdx.x * 5 + 2] + (int)window[blockIdx.x * 2 + 1];
		int w = (int) (((float) (features[threadIdx.x * 5 + 3])) * window[blockIdx.x * 2 + 2]);
		int h = (int) (((float) (features[threadIdx.x * 5 + 4])) * window[blockIdx.x * 2 + 2]);

		if (type == 1)
		{
			int mid = w / 2;
			int r1 = rectanglesSum(integralImage, x, y, mid, h);
			int r2 = rectanglesSum(integralImage, x + mid, y, mid, h);
			haarFeatures[tidX] = r1 - r2;
		}
		else if (type == 2)
		{
			int mid = w / 3;

			int r1 = rectanglesSum(integralImage, x, y, mid, h);
			int r2 = rectanglesSum(integralImage, x + mid, y, mid, h);
			int r3 = rectanglesSum(integralImage, x + 2 * mid, y, mid, h);

			haarFeatures[tidX] = r1 - r2 + r3;
		}
		else if (type == 3)
		{
			int mid = h / 2;
			int r1 = rectanglesSum(integralImage, x, y, w, mid);
			int r2 = rectanglesSum(integralImage, x, y + mid, w, mid);
			haarFeatures[tidX] = r2 - r1;
		}
		else if (type == 4)
		{
			int mid = h / 3;

			int r1 = rectanglesSum(integralImage, x, y, w, mid);
			int r2 = rectanglesSum(integralImage, x, y + mid, w, mid);
			int r3 = rectanglesSum(integralImage, x, y + 2 * mid, w, mid);

			haarFeatures[tidX] = r1 - r2 + r3;
		}
		else if (type == 5)
		{
			int mid_w = w / 2;
			int mid_h = h / 2;

			int r1 = rectanglesSum(integralImage, x, y, mid_w, mid_h);
			int r2 = rectanglesSum(integralImage, x + mid_w, y, mid_w, mid_h);
			int r3 = rectanglesSum(integralImage, x, y + mid_h, mid_w, mid_h);
			int r4 = rectanglesSum(integralImage, x + mid_w, y + mid_h, mid_w, mid_h);

			haarFeatures[tidX] = r1 - r2 - r3 + r4;
		}
	}

	__syncthreads();
}
