package cuda;

import GUI.ImageHandler;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import process.features.FeatureExtractor;
import process.features.Rectangle;

import java.util.ArrayList;

import static jcuda.driver.JCudaDriver.*;

// Toutes haar features dechaque type
// calculer tous les rectangles possibles pour chaque feature.

public class HaarExtractor {

    public static final int THREADS_IN_BLOCK = 1024;
    public static final String CUDA_FILENAME = "HaarType";
    public static final String KERNEL_NAME = "haar_type_";

    private final long NUM_FEATURES_A;
    private final long NUM_FEATURES_B;
    private final long NUM_FEATURES_C;
    private final long NUM_FEATURES_D;
    private final long NUM_FEATURES_E;

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

    private CUdeviceptr srcPtr;
    private CUdeviceptr dstPtr;
    private CUdeviceptr tmpDataPtr[];

    private CUdeviceptr allRectanglesA;
    private CUdeviceptr allRectanglesB;
    private CUdeviceptr allRectanglesC;
    private CUdeviceptr allRectanglesD;
    private CUdeviceptr allRectanglesE;


    public HaarExtractor(ImageHandler image) {
        this.data = image.getGrayImage();
        this.integral = image.getIntegralImage();
        this.width = image.getWidth();
        this.height = image.getHeight();

        this.NUM_FEATURES_A = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeA, FeatureExtractor.heightTypeA, width, height);
        this.NUM_FEATURES_B = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeB, FeatureExtractor.heightTypeB, width, height);
        this.NUM_FEATURES_C = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeC, FeatureExtractor.heightTypeC, width, height);
        this.NUM_FEATURES_D = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeD, FeatureExtractor.heightTypeD, width, height);
        this.NUM_FEATURES_E = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeE, FeatureExtractor.heightTypeE, width, height);

        this.tmpDataPtr = new CUdeviceptr[width];

        this.allRectanglesA = new CUdeviceptr();
        this.allRectanglesB = new CUdeviceptr();
        this.allRectanglesC = new CUdeviceptr();
        this.allRectanglesD = new CUdeviceptr();
        this.allRectanglesE = new CUdeviceptr();

        this.featuresA = new ArrayList<>();
        this.featuresB = new ArrayList<>();

        listAllTypeN(this.NUM_FEATURES_A, FeatureExtractor.widthTypeA, FeatureExtractor.heightTypeA, 'A');
        listAllTypeN(this.NUM_FEATURES_B, FeatureExtractor.widthTypeB, FeatureExtractor.heightTypeB, 'B');
    }

    // Free Cuda !
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

        this.module = CudaUtils.initCuda(CUDA_FILENAME + type);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, KERNEL_NAME + type);

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

        // Initialisation of the input data that will be passed to Cuda
        // The image is larger to compute the filter without bounds checking

        // Allocate input data to CUDA memory
        // Pas utiliser ça...
        srcPtr = new CUdeviceptr();
        CudaUtils.newArray2D(integral, width, height, tmpDataPtr, srcPtr);

        computeTypeN(this.NUM_FEATURES_A, 'A');
        computeTypeN(this.NUM_FEATURES_B, 'B');

        // Free intergralImg and typeABCDE

        cuMemFree(allRectanglesA);
        cuMemFree(allRectanglesB);
        cuMemFree(allRectanglesC);
        cuMemFree(allRectanglesD);
        cuMemFree(allRectanglesE);

        CudaUtils.freeArray2D(tmpDataPtr, srcPtr, width);

        /*
        System.out.println(featuresA);
        System.out.println("Features B :" + System.lineSeparator() + System.lineSeparator() + System.lineSeparator());
        System.out.println(featuresB);
        */

    }
}
