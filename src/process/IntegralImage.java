package process;

import cuda.CudaUtils;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

public class IntegralImage {

    public static int[][] summedAreaTableCPU(int[][] image, int width, int height) {
        int[][] result = new int[width][height];

        // Array copy
        for (int x = 0; x < width; x++)
            System.arraycopy(image[x], 0, result[x], 0, height);


        // Top border
        for (int x = 1; x < width; x++)
            result[x][0] = result[x][0] + result[x - 1][0];

        // Left border
        for (int y = 1; y < height; y++)
            result[0][y]  = result[0][y] + result[0][y - 1];

        // Remaining pixels
        for (int x = 1; x < width; x++)
            for (int y = 1; y < height; y++)
                result[x][y] = result[x][y] + result[x - 1][y] + result[x][y - 1] - result[x - 1][y - 1];

        return result;
    }

    public static int[][] summedAreaTableGPU(int[][] image, int width, int height) {

        CUmodule module = CudaUtils.initCuda("IntegralImage");
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "integral_image");

        CUdeviceptr srcPtr = new CUdeviceptr();
        CUdeviceptr dstPtr = new CUdeviceptr();
        CUdeviceptr[] tmpDataPtrSrc = new CUdeviceptr[width];
        CUdeviceptr[] tmpDataPtrDst = new CUdeviceptr[width];

        // Allocate input data to CUDA memory
        CudaUtils.newArray2D(image, width, height, tmpDataPtrSrc, srcPtr);
        CudaUtils.newArray2D(image, width, height, tmpDataPtrDst, dstPtr);

        Pointer kernelParams = Pointer.to(
                Pointer.to(srcPtr),
                Pointer.to(new int[]{height}),
                Pointer.to(new int[]{width}),
                Pointer.to(dstPtr)
        );

        // Call the kernel function.
        cuLaunchKernel(
                function, // CUDA function to be called
                1, 1, 1, // 3D (x, y, z) grid of block
                1, 1, 1, // 3D (x, y, z) grid of threads
                0, // sharedMemBytes sets the amount of dynamic shared memory that will be available to each thread block.
                null, // can optionally be associated to a stream by passing a non-zero hStream argument.
                kernelParams, // Array of params to be passed to the function
                null // extra parameters
        );
        cuCtxSynchronize();

        int[][] result = CudaUtils.memCpyArray2D(dstPtr, width, height);

        // Free output that will chang for each haar feature
        CudaUtils.freeArray2D(tmpDataPtrSrc, srcPtr, width);
        CudaUtils.freeArray2D(tmpDataPtrDst, dstPtr, width);
//        cuMemFree(dstPtr);

        return result;
    }

    public static int[][] summedAreaTable(int[][] image, int width, int height) {
        if (Conf.USE_CUDA)
            return summedAreaTableGPU(image, width, height);
        else
            return summedAreaTableCPU(image, width, height);
    }

    // Warning : this does not compute the mean of the image, just the sum of pixels
    // To have the mean you must divide by the number of pixels in your rectangle
    public static int rectangleSum(int[][] summedAeraTable, int x, int y, int width, int height) {

        int A = x > 0 && y > 0 ? summedAeraTable[x - 1][y - 1] : 0;
        int B = x + width > 0 && y > 0 ? summedAeraTable[x + width - 1][y - 1] : 0;
        int C = x > 0 && y + height > 0 ? summedAeraTable[x - 1][y + height - 1] : 0;
        int D = x + width > 0 && y + height > 0 ? summedAeraTable[x + width - 1][y + height - 1] : 0;

        return A + D - B - C;
    }

    public static int rectangleMean(int[][] summedAeraTable, int x, int y, int width, int height) {
        int sum = rectangleSum(summedAeraTable, x, y, width, height);
        int size = (width - x) * (height - y);
        return sum / size;
    }
}
