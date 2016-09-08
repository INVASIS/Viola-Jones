package cuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import process.features.FeatureExtractor;
import process.features.Rectangle;

import java.util.ArrayList;
import java.util.HashMap;

import static jcuda.driver.JCudaDriver.*;

public class HaarExtractor extends HaarBase {

    public HaarExtractor() {
        super();
    }

    @Override
    public void setUp(int width, int height) {
        super.setUp(width, height);

        // TODO : we will only need certains haar-feature, not all, so why not compute only those needed ?
        listAllTypeN(this.NUM_FEATURES_A, FeatureExtractor.widthTypeA, FeatureExtractor.heightTypeA, 'A');
        listAllTypeN(this.NUM_FEATURES_B, FeatureExtractor.widthTypeB, FeatureExtractor.heightTypeB, 'B');
        listAllTypeN(this.NUM_FEATURES_C, FeatureExtractor.widthTypeC, FeatureExtractor.heightTypeC, 'C');
        listAllTypeN(this.NUM_FEATURES_D, FeatureExtractor.widthTypeD, FeatureExtractor.heightTypeD, 'D');
        listAllTypeN(this.NUM_FEATURES_E, FeatureExtractor.widthTypeE, FeatureExtractor.heightTypeE, 'E');
    }

    private void listAllTypeN(long numFeatures, int width, int height, char type) {

        long size_output = 4 * numFeatures;

        ArrayList<Rectangle> typeN = FeatureExtractor.listFeaturePositions(width, height, this.width, this.height);

        int[] arrayTypeN = new int[(int) size_output];

        int j = 0;
        for (int i = 0; i < numFeatures; i++) {
            arrayTypeN[j] = typeN.get(i).getX();
            arrayTypeN[j + 1] = typeN.get(i).getY();
            arrayTypeN[j + 2] = typeN.get(i).getWidth();
            arrayTypeN[j + 3] = typeN.get(i).getHeight();

            j += 4;
        }

        CUdeviceptr tmp_ptr = null;
        switch (type) {
            case 'A':
                tmp_ptr = this.allRectanglesA;
                break;
            case 'B':
                tmp_ptr = this.allRectanglesB;
                break;
            case 'C':
                tmp_ptr = this.allRectanglesC;
                break;
            case 'D':
                tmp_ptr = this.allRectanglesD;
                break;
            case 'E':
                tmp_ptr = this.allRectanglesE;
                break;
        }

        cuMemAlloc(tmp_ptr, size_output * Sizeof.INT);
        cuMemcpyHtoD(tmp_ptr, Pointer.to(arrayTypeN), size_output * Sizeof.INT);
    }

    private void computeTypeN(long numFeatures, char type) {

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, modules.get(type), KERNEL_NAME + type);
        //cuModuleGetFunction(function, CudaUtils.initCuda(CUDA_FILENAME + type), KERNEL_NAME + type);


        // Allocate device output memory
        // dstPtr will contain the results
        this.dstPtr = new CUdeviceptr();
        cuMemAlloc(dstPtr, numFeatures * Sizeof.INT);

        CUdeviceptr tmp_ptr = null;
        ArrayList<Integer> tmp_list = null;
        switch (type) {
            case 'A':
                tmp_ptr = this.allRectanglesA;
                tmp_list = this.featuresA;
                break;
            case 'B':
                tmp_ptr = this.allRectanglesB;
                tmp_list = this.featuresB;
                break;
            case 'C':
                tmp_ptr = this.allRectanglesC;
                tmp_list = this.featuresC;
                break;
            case 'D':
                tmp_ptr = this.allRectanglesD;
                tmp_list = this.featuresD;
                break;
            case 'E':
                tmp_ptr = this.allRectanglesE;
                tmp_list = this.featuresE;
                break;
        }

        // Set up the kernel parameters
        Pointer kernelParams = Pointer.to(
                Pointer.to(srcPtr),
                Pointer.to(tmp_ptr),
                Pointer.to(new int[]{(int) numFeatures}),
                Pointer.to(new float[]{1}),
                Pointer.to(dstPtr)
        );

        int nb_blocks = (int) (numFeatures / THREADS_IN_BLOCK + ((numFeatures % THREADS_IN_BLOCK) == 0 ? 0 : 1));

        // Call the kernel function.
        cuLaunchKernel(
                function, // CUDA function to be called
                nb_blocks, 1, 1, // 3D (x, y, z) grid of block
                THREADS_IN_BLOCK, 1, 1, // 3D (x, y, z) grid of threads
                0, // sharedMemBytes sets the amount of dynamic shared memory that will be available to each thread block.
                null, // can optionally be associated to a stream by passing a non-zero hStream argument.
                kernelParams, // Array of params to be passed to the function
                null // extra parameters
        );
        cuCtxSynchronize();

        int hostOutput[] = new int[(int) numFeatures];
        cuMemcpyDtoH(Pointer.to(hostOutput), dstPtr, numFeatures * Sizeof.INT);

        // TODO : Need to do better - Opti by returning only an int[]
        for (int index = 0; index < numFeatures; index++) {
            tmp_list.add(hostOutput[index]);
        }

        cuMemFree(dstPtr);
    }

    public void compute() {
        if (this.tmpDataPtr == null) {
            System.err.println("ERROR HaarExtractor not init - Aborting");
            System.exit(42);
        }
        // Initialisation of the input data that will be passed to Cuda
        // The image is larger to compute the filter without bounds checking

        // Allocate input data to CUDA memory
        // Pas utiliser ça...
        srcPtr = new CUdeviceptr();
        CudaUtils.newArray2D(integral, width, height, tmpDataPtr, srcPtr);

        computeTypeN(this.NUM_FEATURES_A, 'A');
        computeTypeN(this.NUM_FEATURES_B, 'B');
        computeTypeN(this.NUM_FEATURES_C, 'C');
        computeTypeN(this.NUM_FEATURES_D, 'D');
        computeTypeN(this.NUM_FEATURES_E, 'E');

        CudaUtils.freeArray2D(tmpDataPtr, srcPtr, width);

    }
}
