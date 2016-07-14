package cuda;

import GUI.Display;
import GUI.ImageHandler;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import java.io.IOException;

import static jcuda.driver.JCudaDriver.*;

public class BlurFilter {

    public static boolean process(int[][] image, int width, int height) {

        JCudaDriver.setExceptionsEnabled(true);

        // Create the PTX file by calling the NVCC
        String ptxFileName = null;
        try {
            ptxFileName = CudaUtils.compileCuda("BlurFilter");
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUcontext pctx = new CUcontext();
        CUdevice dev = new CUdevice();
        cuDeviceGet(dev, 0);
        cuCtxCreate(pctx, 0, dev);

        // Load the file containing the kernels
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Perform the tests

        return compute(module, image, width, height);
    }


    private static boolean compute(CUmodule module, int[][] data, int width, int height)
    {
        // Obtain a function pointer to the "blur_filter" function.
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "blur_filter");

        float hostInput[][] = new float[width][height];
        for(int i = 0; i < width; i++)
        {
            for (int j=0; j<height; j++)
            {
                hostInput[i][j] = (float)data[i][j];
            }
        }

        CUdeviceptr hostDevicePtr[] = new CUdeviceptr[width];
        for (int i = 0; i < width; i++) {
            hostDevicePtr[i] = new CUdeviceptr();
            cuMemAlloc(hostDevicePtr[i], height * Sizeof.FLOAT);
        }

        for (int i = 0; i < width; i++) {
            cuMemcpyHtoD(hostDevicePtr[i], Pointer.to(hostInput[i]), height * Sizeof.FLOAT);
        }


        CUdeviceptr deviceInput = new CUdeviceptr();
        cuMemAlloc(deviceInput, width * Sizeof.POINTER);
        cuMemcpyHtoD(deviceInput, Pointer.to(hostDevicePtr), width * Sizeof.POINTER);

        // Allocate device output memory: A single column with
        // height 'numThreads'.
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, width * height * Sizeof.FLOAT);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParams = Pointer.to(
                Pointer.to(deviceInput),
                Pointer.to(new int[]{height}),
                Pointer.to(deviceOutput)
        );

        // Call the kernel function.
        cuLaunchKernel(function,
                1, 1, 1,           // Grid dimension
                width - 2, 1, 1,  // Block dimension (I access in the .cu to 8 pixels around so I need to - 2 to avoid outOfBounds)
                0, null,           // Shared memory size and stream
                kernelParams, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        float hostOutput[] = new float[width * height];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, height * width * Sizeof.FLOAT);

        int outputedImg[][] = new int[width][height];
        for(int i = 0; i < width; i++)
            for (int j=0; j<height; j++) {
                outputedImg[i][j] = (int) hostOutput[j + i * height];
            }


        ImageHandler newImg = new ImageHandler(outputedImg, width, height);
        Display.drawImage(newImg.getBufferedImage());

        // Clean up.
        for(int i = 0; i < width; i++)
        {
            cuMemFree(hostDevicePtr[i]);
        }
        cuMemFree(deviceInput);
        cuMemFree(deviceOutput);

        return true;
    }

}
