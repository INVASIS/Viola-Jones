package cuda;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUmodule;
import process.features.FeatureExtractor;

import java.util.ArrayList;
import java.util.HashMap;

import static jcuda.driver.JCudaDriver.cuMemFree;

public abstract class HaarBase implements AutoCloseable {

    protected static final int THREADS_IN_BLOCK = 1024;
    protected static final String CUDA_FILENAME = "HaarType";
    protected static final String KERNEL_NAME = "haar_type_";

    protected long NUM_FEATURES_A;
    protected long NUM_FEATURES_B;
    protected long NUM_FEATURES_C;
    protected long NUM_FEATURES_D;
    protected long NUM_FEATURES_E;
    protected long NUM_TOTAL_FEATURES;

    protected int[][] integral;
    protected int width;
    protected int height;

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

    protected CUdeviceptr srcPtr;
    protected CUdeviceptr dstPtr;
    protected CUdeviceptr tmpDataPtr[];

    // TODO REFACTOR !!!

    public HaarBase() {
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
}
