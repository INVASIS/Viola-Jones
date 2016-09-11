package cuda;

import GUI.ImageHandler;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import process.Conf;
import process.features.Feature;
import process.features.FeatureExtractor;
import process.features.Rectangle;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import static jcuda.driver.JCudaDriver.*;
import static process.features.FeatureExtractor.streamFeaturesByType;


public class HaarDetector extends HaarBase {
    private static final int valuesByFeature = 5;
    private static final int valuesByWindow = 3;

    protected static final String DETECTOR_CUDA_FILENAME = "ComputeWindowFeatures";
    protected static final String DETECTOR_KERNEL_NAME = "computeWindowFeatures";
    private HashMap<Integer, Integer> neededHaarValues;

    private int[] neededFeatures;
    private float[] slidingWindows;

    private CUdeviceptr neededFeaturesPtr;
    private CUdeviceptr slidingWindowsPtr;

    private int neededFeaturesSize;
    private int slidingWindowsSize;
    private int outputSize;

    private CUmodule moduleDetector;
    private int[] allFeatures;
    private int baseSize;

    public HaarDetector(HashMap<Integer, Integer> neededHaarValues, int baseSize) {

        this.neededHaarValues = neededHaarValues;
        this.neededFeaturesPtr = new CUdeviceptr();
        this.slidingWindowsPtr = new CUdeviceptr();
        this.moduleDetector = CudaUtils.getModule(DETECTOR_CUDA_FILENAME);
        this.baseSize = baseSize;
    }

    public void setUp(int width, int height, ArrayList<Rectangle> windows) {
        this.integral = null;
        this.width = width;
        this.height = height;
        this.tmpDataPtr = new CUdeviceptr[width];


        neededFeaturesSize = neededHaarValues.size();
        slidingWindowsSize = windows.size();

        // Check limitations
        if (neededFeaturesSize > Conf.maxThreadsPerBlock
                || (((long)(neededFeaturesSize)) * ((long)(slidingWindowsSize))) >= Integer.MAX_VALUE // For dstPtr
                || (((long)(valuesByFeature)) * ((long)(neededFeaturesSize))) >= Integer.MAX_VALUE // For neededFeaturesPtr
                || (((long)(valuesByWindow)) * ((long)(slidingWindowsSize))) >= Integer.MAX_VALUE) // For slidingWindowsPtr
        {
            System.err.println("Error with values neededHaarValues and number of windows: ");
            System.err.println("neededHaarValues: " + neededHaarValues.size() + " num of windows: " + windows.size());
            System.err.println("Max values are: " + Conf.maxThreadsPerBlock + " and 65535");
            throw new InvalidParameterException("Invalid number of thread or block needed for CUDA");
        }

        outputSize = slidingWindowsSize * neededFeaturesSize;

        int cpt = 0;
        ArrayList<Feature> ft = new ArrayList<>();
        // Get features that correspond to given indexes
        {
            Feature feattt[] = new Feature[neededFeaturesSize];
            for (ArrayList<Feature> lf : streamFeaturesByType(new ImageHandler(new int[19][19], 19, 19))) {
                for (Feature f : lf) {
                    if (neededHaarValues.containsKey(cpt))
                        feattt[neededHaarValues.get(cpt)] = f;
                    cpt++;
                }
            }
            Collections.addAll(ft, feattt);

            if (ft.size() != neededFeaturesSize) {
                System.err.println("Error in computing neededFeaturesSize");
                System.exit(1);
            }
        }

        // Alloc memory for neededFeaturesPtr
        {
            neededFeatures = new int[valuesByFeature * neededFeaturesSize];
            cpt = 0;
            for (Feature f : ft) {
                neededFeatures[(cpt * valuesByFeature)] = f.getType();
                neededFeatures[(cpt * valuesByFeature + 1)] = f.getRectangle().getX();
                neededFeatures[(cpt * valuesByFeature + 2)] = f.getRectangle().getY();
                neededFeatures[(cpt * valuesByFeature + 3)] = f.getRectangle().getWidth();
                neededFeatures[(cpt * valuesByFeature + 4)] = f.getRectangle().getHeight();
                cpt++;
            }

            cuMemAlloc(neededFeaturesPtr, valuesByFeature * neededFeaturesSize * Sizeof.INT);
            cuMemcpyHtoD(neededFeaturesPtr, Pointer.to(neededFeatures), valuesByFeature * neededFeaturesSize * Sizeof.INT);
        }

        // Alloc memory for slidingWindowsPtr
        {
            slidingWindows = new float[valuesByWindow * slidingWindowsSize];
            cpt = 0;
            for (Rectangle rectangle : windows) {
                slidingWindows[cpt * valuesByWindow] = rectangle.getX();
                slidingWindows[cpt * valuesByWindow + 1] = rectangle.getY();
                slidingWindows[cpt * valuesByWindow + 2] = (float) rectangle.getHeight() / (float) baseSize;
                cpt++;
            }

            cuMemAlloc(slidingWindowsPtr, valuesByWindow * slidingWindowsSize * Sizeof.FLOAT);
            cuMemcpyHtoD(slidingWindowsPtr, Pointer.to(slidingWindows), valuesByWindow * slidingWindowsSize * Sizeof.FLOAT);
        }

        this.dstPtr = new CUdeviceptr();
        cuMemAlloc(dstPtr, outputSize * Sizeof.INT);
        allFeatures = new int[outputSize];
    }

    private void launchKernel() {

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, moduleDetector, DETECTOR_KERNEL_NAME);



        Pointer kernelParams = Pointer.to(
                Pointer.to(srcPtr),
                Pointer.to(neededFeaturesPtr),
                Pointer.to(new int[]{outputSize}),
                Pointer.to(slidingWindowsPtr),
                Pointer.to(dstPtr)
        );

        int nbBlocksX;
        int nbBlocksY;
        if (slidingWindowsSize > Conf.maxBlocksByDim) {
            nbBlocksX = Conf.maxBlocksByDim;
            nbBlocksY = (int) Math.ceil( (double)slidingWindowsSize / (double)Conf.maxBlocksByDim);
        }
        else {
            nbBlocksX = slidingWindowsSize;
            nbBlocksY = 1;
        }

        cuLaunchKernel(
                function, // CUDA function to be called
                nbBlocksX, nbBlocksY, 1, // 3D (x, y, z) grid of block
                neededFeaturesSize, 1, 1, // 3D (x, y, z) grid of threads
                0, // sharedMemBytes sets the amount of dynamic shared memory that will be available to each thread block.
                null, // can optionally be associated to a stream by passing a non-zero hStream argument.
                kernelParams, // Array of params to be passed to the function
                null // extra parameters
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(allFeatures), dstPtr, outputSize * Sizeof.INT);
    }


    public int[] compute() {
        if (this.tmpDataPtr == null) {
            System.err.println("ERROR HaarDetector not init - Aborting");
            System.exit(42);
        }

        srcPtr = new CUdeviceptr();
        CudaUtils.newArray2D(integral, width, height, tmpDataPtr, srcPtr);

        long milliseconds = System.currentTimeMillis();
        launchKernel();
        System.out.println("haar time: " + (System.currentTimeMillis() - milliseconds) + " ms");

        CudaUtils.freeArray2D(tmpDataPtr, srcPtr, width);

        return allFeatures;
    }

    // TODO : make it compute the image also ?
    public void updateImage(int[][] newIntegral, int width, int height, ArrayList<Rectangle> windows) {
        if (!(this.width == width && this.height == height)) {
            cuMemFree(dstPtr);
            this.setUp(width, height, windows);
        }
        this.integral = newIntegral;
        this.tmpDataPtr = new CUdeviceptr[width];
        this.width = width;
        this.height = height;
    }

    @Override
    public void close() throws Exception {
        // Free CUDA
        System.out.println("Freeing CUDA memory for detector...");
        cuMemFree(this.neededFeaturesPtr);
        cuMemFree(dstPtr);
    }

}

