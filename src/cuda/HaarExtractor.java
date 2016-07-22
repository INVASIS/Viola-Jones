package cuda;

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

    // La haar feature A : |0 1| taille 2w * 1h (width et height) scalable : de 0 à w = 0 de w à 2w = 1

    public HaarExtractor(int[][] data, int[][] integral, int width, int height) {
        this.data = data;
        this.integral = integral;
        this.width = width;
        this.height = height;

        this.NUM_FEATURES_A = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeA, FeatureExtractor.heightTypeA, 19, 19);
        this.NUM_FEATURES_B = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeB, FeatureExtractor.heightTypeB, 19, 19);
        this.NUM_FEATURES_C = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeC, FeatureExtractor.heightTypeC, 19, 19);
        this.NUM_FEATURES_D = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeD, FeatureExtractor.heightTypeD, 19, 19);
        this.NUM_FEATURES_E = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeE, FeatureExtractor.heightTypeE, 19, 19);

        this.tmpDataPtr = new CUdeviceptr[width];

        this.featuresA = new ArrayList<>();

        listAllTypeA();
    }

    // Pas oublier de free en CUDA !
    private void listAllTypeA() {

        long size_output = 4 * this.NUM_FEATURES_A;

        ArrayList<Rectangle> typeA = FeatureExtractor.listFeaturePositions(FeatureExtractor.widthTypeA, FeatureExtractor.heightTypeA, 19, 19);

        int[] arrayTypeA = new int[(int) size_output];

        int j = 0;
        for (int i = 0; i < NUM_FEATURES_A; i++) {
            arrayTypeA[j] = typeA.get(i).getX();
            arrayTypeA[j + 1] = typeA.get(i).getY();
            arrayTypeA[j + 2] = typeA.get(i).getWidth();
            arrayTypeA[j + 3] = typeA.get(i).getHeight();

            j += 4;
        }

        this.allRectanglesA = new CUdeviceptr();
        cuMemAlloc(allRectanglesA, size_output * Sizeof.INT);

        cuMemcpyHtoD(allRectanglesA, Pointer.to(arrayTypeA), size_output * Sizeof.INT);

    }

    // Haar value of sizex * sizey dim in a 24 * 24 rectangle
    // in a rectangle 24 * 24 that starts at (posx, posy)
    private void computeTypeA() {

        this.module = CudaUtils.initCuda(CUDA_FILENAME + "A");

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, KERNEL_NAME + "A");

        // Allocate device output memory
        // dstPtr will contain the results
        this.dstPtr = new CUdeviceptr();
        cuMemAlloc(dstPtr, NUM_FEATURES_A * Sizeof.INT);

        // Set up the kernel parameters
        Pointer kernelParams = Pointer.to(
                Pointer.to(srcPtr),
                Pointer.to(allRectanglesA),
                Pointer.to(new int[]{(int) NUM_FEATURES_A}),
                Pointer.to(dstPtr)
        );

        int nb_blocks = (int) (NUM_FEATURES_A / THREADS_IN_BLOCK + ((NUM_FEATURES_A % THREADS_IN_BLOCK) == 0 ? 0 : 1));

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

        int hostOutput[] = new int[(int) NUM_FEATURES_A];
        cuMemcpyDtoH(Pointer.to(hostOutput), dstPtr, NUM_FEATURES_A * Sizeof.INT);

        // TODO : Need to do better - Opti by returning only an int[]
        for (int index = 0; index < NUM_FEATURES_A; index++) {
            featuresA.add(hostOutput[index]);
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

        computeTypeA();

        // Free intergralImg and typeA
        CudaUtils.freeArray2D(tmpDataPtr, srcPtr, width);

        System.out.println(featuresA);

    }
}
