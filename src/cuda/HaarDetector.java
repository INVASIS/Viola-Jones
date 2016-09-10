package cuda;

import GUI.ImageHandler;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import process.features.Feature;
import process.features.Rectangle;

import java.util.ArrayList;
import java.util.HashMap;

import static jcuda.driver.JCudaDriver.*;
import static process.features.FeatureExtractor.streamFeaturesByType;


// TODO : all cuda should return int[] instead of ArrayList<Integer>
// FIXME : make it work!
public class HaarDetector extends HaarBase {

    protected static final String DETECTOR_CUDA_FILENAME = "ComputeWindowFeatures";
    protected static final String DETECTOR_KERNEL_NAME = "computeWindowFeatures";
    private HashMap<Integer, Integer> neededHaarValues;

    private int[] neededFeatures;
    private float[] slidingWindows;

    private CUdeviceptr neededFeaturesPtr;
    private CUdeviceptr slidingWindowsPtr;

    private int neededFeaturesSize;
    private int slidingWindowsSize;

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

        {
            ArrayList<Feature> ft = new ArrayList<>();
            int cpt = 0;
            for (ArrayList<Feature> lf : streamFeaturesByType(new ImageHandler(new int[19][19], 19, 19))) {
                for (Feature f : lf) {
                    if (neededHaarValues.containsKey(cpt))
                        ft.add(f);
                    cpt++;
                }
            }

            neededFeaturesSize = ft.size();
            if (neededFeaturesSize != neededHaarValues.size())
                System.err.println("Error in computing neededFeaturesSize");

            neededFeatures = new int[5 * neededFeaturesSize];
            cpt = 0;
            for (Feature f : ft) {
                neededFeatures[(cpt * 5)] = f.getType();
                neededFeatures[(cpt * 5 + 1)] = f.getRectangle().getX();
                neededFeatures[(cpt * 5 + 2)] = f.getRectangle().getY();
                neededFeatures[(cpt * 5 + 3)] = f.getRectangle().getWidth();
                neededFeatures[(cpt * 5 + 4)] = f.getRectangle().getHeight();
                cpt++;
            }

            cuMemAlloc(neededFeaturesPtr, 5 * neededFeaturesSize * Sizeof.INT);
            cuMemcpyHtoD(neededFeaturesPtr, Pointer.to(neededFeatures), 5 * neededFeaturesSize * Sizeof.INT);
        }

        // Compute windows where we slide in*
        slidingWindowsSize = windows.size();
        slidingWindows = new float[slidingWindowsSize * 3];
        int i = 0;
        for (Rectangle rectangle : windows) {
            slidingWindows[i++] = rectangle.getX();
            slidingWindows[i++] = rectangle.getY();
            slidingWindows[i++] = (float) rectangle.getHeight() / (float)baseSize;
        }
        cuMemAlloc(slidingWindowsPtr, slidingWindowsSize * 3 * Sizeof.INT);
        cuMemcpyHtoD(slidingWindowsPtr, Pointer.to(slidingWindows), slidingWindowsSize * 3 * Sizeof.INT);
    }

    private void launchKernel(long outputSize) {

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, moduleDetector, DETECTOR_KERNEL_NAME);


        this.dstPtr = new CUdeviceptr();
        cuMemAlloc(dstPtr, outputSize * Sizeof.INT);

        Pointer kernelParams = Pointer.to(
                Pointer.to(srcPtr),
                Pointer.to(neededFeaturesPtr),
                Pointer.to(new int[]{(int) outputSize}),
                Pointer.to(slidingWindowsPtr),
                Pointer.to(dstPtr)
        );

        // Nb block limit : 65535 per dimension per grid
        //int nb_blocks = (int) (outputSize / THREADS_IN_BLOCK + ((outputSize % THREADS_IN_BLOCK) == 0 ? 0 : 1));

        cuLaunchKernel(
                function, // CUDA function to be called
                slidingWindowsSize, 1, 1, // 3D (x, y, z) grid of block
                neededFeaturesSize, 1, 1, // 3D (x, y, z) grid of threads
                0, // sharedMemBytes sets the amount of dynamic shared memory that will be available to each thread block.
                null, // can optionally be associated to a stream by passing a non-zero hStream argument.
                kernelParams, // Array of params to be passed to the function
                null // extra parameters
        );
        cuCtxSynchronize();

        int hostOutput[] = new int[(int) outputSize];
        cuMemcpyDtoH(Pointer.to(hostOutput), dstPtr, outputSize * Sizeof.INT);

        allFeatures = hostOutput;
        cuMemFree(dstPtr);
    }


    public int[] compute() {
        if (this.tmpDataPtr == null) {
            System.err.println("ERROR HaarDetector not init - Aborting");
            System.exit(42);
        }

        srcPtr = new CUdeviceptr();
        CudaUtils.newArray2D(integral, width, height, tmpDataPtr, srcPtr);

        launchKernel(slidingWindowsSize * neededFeaturesSize);

        CudaUtils.freeArray2D(tmpDataPtr, srcPtr, width);

        return allFeatures;
    }

    // TODO : make it compute the image also ?
    public void updateImage(int[][] newIntegral, int width, int height, ArrayList<Rectangle> windows) {
        if (!(this.width == width && this.height == height)) {
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
    }

}

