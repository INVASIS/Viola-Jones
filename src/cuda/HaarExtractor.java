package cuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import process.Conf;
import process.features.FeatureExtractor;
import process.features.Rectangle;

import java.util.ArrayList;
import java.util.HashMap;

import static jcuda.driver.JCudaDriver.*;

public class HaarExtractor extends HaarBase {

    protected static final String CUDA_FILENAME = "HaarType";
    protected static final String KERNEL_NAME = "haar_type_";

    protected long NUM_FEATURES_A;
    protected long NUM_FEATURES_B;
    protected long NUM_FEATURES_C;
    protected long NUM_FEATURES_D;
    protected long NUM_FEATURES_E;
    protected long NUM_TOTAL_FEATURES;

    protected int[] featuresA;
    protected int[] featuresB;
    protected int[] featuresC;
    protected int[] featuresD;
    protected int[] featuresE;

    protected CUdeviceptr allRectanglesA;
    protected CUdeviceptr allRectanglesB;
    protected CUdeviceptr allRectanglesC;
    protected CUdeviceptr allRectanglesD;
    protected CUdeviceptr allRectanglesE;

    protected HashMap<Character, CUmodule> modules;

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
        this.NUM_TOTAL_FEATURES = NUM_FEATURES_A + NUM_FEATURES_B + NUM_FEATURES_C + NUM_FEATURES_D + NUM_FEATURES_E;
        this.tmpDataPtr = new CUdeviceptr[width];

        // TODO : we will only need certains haar-feature, not all, so why not compute only those needed ?
        listAllTypeN(this.NUM_FEATURES_A, FeatureExtractor.widthTypeA, FeatureExtractor.heightTypeA, 'A');
        listAllTypeN(this.NUM_FEATURES_B, FeatureExtractor.widthTypeB, FeatureExtractor.heightTypeB, 'B');
        listAllTypeN(this.NUM_FEATURES_C, FeatureExtractor.widthTypeC, FeatureExtractor.heightTypeC, 'C');
        listAllTypeN(this.NUM_FEATURES_D, FeatureExtractor.widthTypeD, FeatureExtractor.heightTypeD, 'D');
        listAllTypeN(this.NUM_FEATURES_E, FeatureExtractor.widthTypeE, FeatureExtractor.heightTypeE, 'E');
    }

    // Change the image to avoid recomputing all init stuff - to be used only for training purposes
    public void updateImage(int[][] newIntegral) {
        this.integral = newIntegral;
        this.featuresA = new int[(int) NUM_FEATURES_A];
        this.featuresB = new int[(int) NUM_FEATURES_B];
        this.featuresC = new int[(int) NUM_FEATURES_C];
        this.featuresD = new int[(int) NUM_FEATURES_D];
        this.featuresE = new int[(int) NUM_FEATURES_E];
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
                tmp_ptr = this.allRectanglesE;        break;
        }

        // Set up the kernel parameters
        Pointer kernelParams = Pointer.to(
                Pointer.to(srcPtr),
                Pointer.to(tmp_ptr),
                Pointer.to(new int[]{(int) numFeatures}),
                Pointer.to(new float[]{1}),
                Pointer.to(dstPtr)
        );

        int nb_blocks = (int) (numFeatures / Conf.maxThreadsPerBlock + ((numFeatures % Conf.maxThreadsPerBlock) == 0 ? 0 : 1));

        // Call the kernel function.
        cuLaunchKernel(
                function, // CUDA function to be called
                nb_blocks, 1, 1, // 3D (x, y, z) grid of block
                Conf.maxThreadsPerBlock, 1, 1, // 3D (x, y, z) grid of threads
                0, // sharedMemBytes sets the amount of dynamic shared memory that will be available to each thread block.
                null, // can optionally be associated to a stream by passing a non-zero hStream argument.
                kernelParams, // Array of params to be passed to the function
                null // extra parameters
        );
        cuCtxSynchronize();

        int hostOutput[] = new int[(int) numFeatures];
        cuMemcpyDtoH(Pointer.to(hostOutput), dstPtr, numFeatures * Sizeof.INT);

        switch (type) {
            case 'A':
               this.featuresA = hostOutput;
                break;
            case 'B':
                this.featuresB = hostOutput;
                break;
            case 'C':
                this.featuresC = hostOutput;
                break;
            case 'D':
                this.featuresD = hostOutput;
                break;
            case 'E':
                this.featuresE = hostOutput;
                break;
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

    public int[] getFeaturesA() {
        return featuresA;
    }

    public int[] getFeaturesB() {
        return featuresB;
    }

    public int[] getFeaturesC() {
        return featuresC;
    }

    public int[] getFeaturesD() {
        return featuresD;
    }

    public int[] getFeaturesE() {
        return featuresE;
    }

    public long getNUM_TOTAL_FEATURES() {
        return NUM_TOTAL_FEATURES;
    }

    public long getNUM_FEATURES_A() {
        return NUM_FEATURES_A;
    }

    public long getNUM_FEATURES_B() {
        return NUM_FEATURES_B;
    }

    public long getNUM_FEATURES_C() {
        return NUM_FEATURES_C;
    }

    public long getNUM_FEATURES_D() {
        return NUM_FEATURES_D;
    }

    public long getNUM_FEATURES_E() {
        return NUM_FEATURES_E;
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
