package cuda;

import GUI.Display;
import GUI.ImageHandler;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import static jcuda.driver.JCudaDriver.*;

public class AnyFilter {

    public static final int THREADS_IN_BLOCK = 1024;
    public static final String CUDA_FILENAME = "anyFilter";
    public static final String KERNEL_NAME = "any_filter";

    private final int[][] data;
    private final int width;
    private final int height;
    private final CUmodule module;

    private float[][] filter = {{0, 1, 0},
                                {1, -4, 1},
                                {0, 1, 0}};

    // Identity filter
    /*
    private float[][] filter = {{0, 0, 0},
                                {0, 1, 0},
                                {0, 0, 0}};
    */

    // Blur Filter
    /*
    private float coeff = 1f/9f;
    private float[][] filter = {{coeff, coeff, coeff},
            {coeff, coeff, coeff},
            {coeff, coeff, coeff}};
    */

    public AnyFilter(int width, int height, int[][] image) {
        this.width = width;
        this.height = height;
        this.data = image;

        this.module = CudaUtils.getModule(CUDA_FILENAME);
    }

    public AnyFilter(int width, int height, int[][] image, float[][] filter) {
        this(width, height, image);
        this.filter = filter;
    }

    public void compute() {

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, KERNEL_NAME);

        // Initialisation of the input data that will be passed to Cuda
        // The image is larger to compute the filter without bounds checking
        float hostInput[][] = new float[width + 2][height + 2];

        for (int i = 1; i < width + 1; i++) {
            for (int j = 1; j < height + 1; j++) {
                hostInput[i][j] = data[i - 1][j - 1];
            }
            hostInput[i][0] = 0;
            hostInput[i][height + 1] = 0;
        }
        for (int i = 0; i < height + 2; i++) {
            hostInput[0][i] = 0;
            hostInput[width + 1][i] = 0;
        }

        // Allocate input memory
        CUdeviceptr hostDevicePtr[] = new CUdeviceptr[width + 2];
        for (int i = 0; i < width + 2; i++) {
            hostDevicePtr[i] = new CUdeviceptr();
            cuMemAlloc(hostDevicePtr[i], (height + 2) * Sizeof.FLOAT);
            cuMemcpyHtoD(hostDevicePtr[i], Pointer.to(hostInput[i]), (height + 2) * Sizeof.FLOAT);
        }

        CUdeviceptr deviceInput = new CUdeviceptr();
        cuMemAlloc(deviceInput, (width + 2) * Sizeof.POINTER);
        cuMemcpyHtoD(deviceInput, Pointer.to(hostDevicePtr), (width + 2) * Sizeof.POINTER);


        // Allocate filter memory
        CUdeviceptr filterPtr[] = new CUdeviceptr[3];
        for (int i = 0; i < 3; i++) {
            filterPtr[i] = new CUdeviceptr();
            cuMemAlloc(filterPtr[i], 3 * Sizeof.FLOAT);
            cuMemcpyHtoD(filterPtr[i], Pointer.to(filter[i]), 3 * Sizeof.FLOAT);
        }

        CUdeviceptr filterInput = new CUdeviceptr();
        cuMemAlloc(filterInput, 3 * Sizeof.POINTER);
        cuMemcpyHtoD(filterInput, Pointer.to(filterPtr), 3 * Sizeof.POINTER);


        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, width * height * Sizeof.FLOAT);

        // Set up the kernel parameters
        Pointer kernelParams = Pointer.to(
                Pointer.to(deviceInput), // globalInputData = 2D array =
                Pointer.to(new int[]{width}), // int width
                Pointer.to(new int[]{height}), // int height
                Pointer.to(filterInput),
                Pointer.to(deviceOutput)
        );

        int nb_pixels = width * height;
        int nb_blocks = nb_pixels / THREADS_IN_BLOCK + ((nb_pixels % THREADS_IN_BLOCK) == 0 ? 0 : 1);

        // Call the kernel function.
        cuLaunchKernel(function,
                nb_blocks, 1, 1,    // Grid dimension : The number of blocks in the grid
                THREADS_IN_BLOCK, 1, 1,
                // Block dimension
                // Number of thread (in 3 dim)
                0, null,            // Shared memory size and stream
                kernelParams, null  // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        float hostOutput[] = new float[width * height];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, height * width * Sizeof.FLOAT);

        int outputedImg[][] = new int[width][height];
        for (int i = 0; i < width; i++)
            for (int j = 0; j < height; j++) {
                outputedImg[i][j] = (int) hostOutput[j + i * height];
            }


        ImageHandler newImg = new ImageHandler(outputedImg, width, height);
        Display.drawImage(newImg.getBufferedImage());

        // Clean up.
        for (int i = 0; i < width; i++) {
            cuMemFree(hostDevicePtr[i]);
        }
        cuMemFree(deviceInput);
        cuMemFree(deviceOutput);

    }

}
