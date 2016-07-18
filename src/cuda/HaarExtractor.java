package cuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import java.util.ArrayList;

import static jcuda.driver.JCudaDriver.*;

public class HaarExtractor {

    public static final int THREADS_IN_BLOCK = 1024;
    public static final String CUDA_FILENAME = "HaarType";
    public static final String KERNEL_NAME = "haar_type_";

    public static final int NUM_FEATURES_A = 43200;
    public static final int NUM_FEATURES_B = 27600;
    public static final int NUM_FEATURES_C = 43200;
    public static final int NUM_FEATURES_D = 27600;
    public static final int NUM_FEATURES_E = 20736;

    private int[][] data;
    private int[][] integral;
    private int width;
    private int height;

    private ArrayList<Integer> featuresA;
    private ArrayList<Integer> featuresB;
    private ArrayList<Integer> featuresC;
    private ArrayList<Integer> featuresD;
    private ArrayList<Integer> featuresE;

    private CUmodule module;

    private CUdeviceptr deviceInput;
    private CUdeviceptr hostDevicePtr[];
    private CUdeviceptr deviceOutput;

    // La haar feature A : |0 1| taille 2w * 1h (width et height) scalable : de 0 à w = 0 de w à 2w = 1

    public HaarExtractor(int[][] data, int[][] integral, int width, int height) {
        this.data = data;
        this.integral = integral;
        this.width = width;
        this.height = height;

        this.featuresA = new ArrayList<>();
    }

    // Haar value of sizex * sizey dim in a 24 * 24 rectangle
    // in a rectangle 24 * 24 that starts at (posx, posy)
    private void typeA(int posx, int posy, int sizex, int sizey) {

        int nb_features = (24 - sizex + 1) * (24 - sizey + 1);

        this.module = CudaUtils.initCuda(CUDA_FILENAME + "A");

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, KERNEL_NAME + "A");

        // Allocate device output memory
        this.deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, nb_features * Sizeof.INT);

        // Set up the kernel parameters
        Pointer kernelParams = Pointer.to(
                Pointer.to(deviceInput),
                Pointer.to(new int[]{posx}),
                Pointer.to(new int[]{posy}),
                Pointer.to(new int[]{sizex}),
                Pointer.to(new int[]{sizey}),
                Pointer.to(deviceOutput)
        );

        int nb_blocks = nb_features / THREADS_IN_BLOCK + ((nb_features % THREADS_IN_BLOCK) == 0 ? 0 : 1);

        // Call the kernel function.
        cuLaunchKernel(function,
                nb_blocks, 1, 1,
                THREADS_IN_BLOCK, 1, 1,
                0, null,
                kernelParams, null
        );
        cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        int hostOutput[] = new int[nb_features];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, nb_features * Sizeof.INT);

        // TODO : Need to do better
        for (int index = 0; index < nb_features; index++) {
            featuresA.add(hostOutput[index]);
        }

        // Free output that will chang for each haar feature
        cuMemFree(deviceOutput);
    }

    public void compute() {

        // Initialisation of the input data that will be passed to Cuda
        // The image is larger to compute the filter without bounds checking

        // Allocate input data memory
        this.hostDevicePtr = new CUdeviceptr[width];
        for (int i = 0; i < width; i++) {
            hostDevicePtr[i] = new CUdeviceptr();
            cuMemAlloc(hostDevicePtr[i], height * Sizeof.INT);
            cuMemcpyHtoD(hostDevicePtr[i], Pointer.to(data[i]), height * Sizeof.INT);
        }

        this.deviceInput = new CUdeviceptr();
        cuMemAlloc(deviceInput, width * Sizeof.POINTER);
        cuMemcpyHtoD(deviceInput, Pointer.to(hostDevicePtr), width * Sizeof.POINTER);

        // Need to compute it on all boxes of 24 * 24 in the image
         typeA(0, 0, 2, 1);
         typeA(width - 24, height - 24, 2, 1);

        //typeB();
        //typeC();
        //typeD();
        //typeE();

        cleanUp();

    }

    private void cleanUp() {
        for (int i = 0; i < width; i++) {
            cuMemFree(hostDevicePtr[i]);
        }
        cuMemFree(deviceInput);
    }
}
