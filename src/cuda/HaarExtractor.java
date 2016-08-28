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

// TODO : singleton ?
public class HaarExtractor implements AutoCloseable {

    private static final int THREADS_IN_BLOCK = 1024;
    private static final String CUDA_FILENAME = "HaarType";
    private static final String KERNEL_NAME = "haar_type_";

    private long NUM_FEATURES_A;
    private long NUM_FEATURES_B;
    private long NUM_FEATURES_C;
    private long NUM_FEATURES_D;
    private long NUM_FEATURES_E;

    private int[][] integral;
    private int width;
    private int height;

    private ArrayList<Integer> featuresA;
    private ArrayList<Integer> featuresB;
    private ArrayList<Integer> featuresC;
    private ArrayList<Integer> featuresD;
    private ArrayList<Integer> featuresE;

    private HashMap<Character, CUmodule> modules;

    private CUdeviceptr srcPtr;
    private CUdeviceptr dstPtr;
    private CUdeviceptr tmpDataPtr[];

    private CUdeviceptr allRectanglesA;
    private CUdeviceptr allRectanglesB;
    private CUdeviceptr allRectanglesC;
    private CUdeviceptr allRectanglesD;
    private CUdeviceptr allRectanglesE;


    public HaarExtractor() {
        this.modules = new HashMap<>();
        this.modules.put('A', CudaUtils.getModule(CUDA_FILENAME + "A"));
        this.modules.put('B', CudaUtils.getModule(CUDA_FILENAME + "B"));
        this.modules.put('C', CudaUtils.getModule(CUDA_FILENAME + "C"));
        this.modules.put('D', CudaUtils.getModule(CUDA_FILENAME + "D"));
        this.modules.put('E', CudaUtils.getModule(CUDA_FILENAME + "E"));

        this.allRectanglesA = new CUdeviceptr();
        this.allRectanglesB = new CUdeviceptr();
        this.allRectanglesC = new CUdeviceptr();
        this.allRectanglesD = new CUdeviceptr();
        this.allRectanglesE = new CUdeviceptr();

        this.featuresA = new ArrayList<>();
        this.featuresB = new ArrayList<>();
        this.featuresC = new ArrayList<>();
        this.featuresD = new ArrayList<>();
        this.featuresE = new ArrayList<>();

    }

    public void setUp(int width, int height) {
        this.integral = null;
        this.width = width;
        this.height = height;

        this.NUM_FEATURES_A = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeA, FeatureExtractor.heightTypeA, width, height);
        this.NUM_FEATURES_B = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeB, FeatureExtractor.heightTypeB, width, height);
        this.NUM_FEATURES_C = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeC, FeatureExtractor.heightTypeC, width, height);
        this.NUM_FEATURES_D = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeD, FeatureExtractor.heightTypeD, width, height);
        this.NUM_FEATURES_E = FeatureExtractor.countFeatures(FeatureExtractor.widthTypeE, FeatureExtractor.heightTypeE, width, height);

        this.tmpDataPtr = new CUdeviceptr[width];

        // TODO: possible optimization: do not use all rectangles, as we train on simple x*x squares already centered on faces, we don't need all rectangles.
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

    // Change the image to avoid recomputing all init stuff - to be used only for training purposes
    public void updateImage(int[][] newIntegral) {
        this.integral = newIntegral;
        this.featuresA.clear();
        this.featuresB.clear();
        this.featuresC.clear();
        this.featuresD.clear();
        this.featuresE.clear();
    }

    public ArrayList<Integer> getFeaturesA() {
        return featuresA;
    }

    public ArrayList<Integer> getFeaturesB() {
        return featuresB;
    }

    public ArrayList<Integer> getFeaturesC() {
        return featuresC;
    }

    public ArrayList<Integer> getFeaturesD() {
        return featuresD;
    }

    public ArrayList<Integer> getFeaturesE() {
        return featuresE;
    }

    @Override
    public void close() throws Exception {
        // Free CUDA
        System.out.println("Freeing CUDA memory...");
        cuMemFree(allRectanglesA);
        cuMemFree(allRectanglesB);
        cuMemFree(allRectanglesC);
        cuMemFree(allRectanglesD);
        cuMemFree(allRectanglesE);
    }
}
