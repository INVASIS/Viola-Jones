package cuda;

import GUI.ImageHandler;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import process.features.Feature;
import process.features.FeatureExtractor;

import java.util.ArrayList;
import java.util.HashMap;

import static jcuda.driver.JCudaDriver.*;
import static process.features.FeatureExtractor.streamFeaturesByType;


// TODO : all cuda should return int[] instead of ArrayList<Integer>
// FIXME : make it work!
public class HaarDetector extends HaarBase {

    private HashMap<Integer, Integer> neededHaarValues;

    public HaarDetector(HashMap<Integer, Integer> neededHaarValues) {
        super();

        this.neededHaarValues = neededHaarValues;
    }

    @Override
    public void setUp(int width, int height) {
        super.setUp(width, height);

        // TODO: possible optimization: do not use all rectangles, as we train on simple x*x squares already centered on faces, we don't need all rectangles.
        // TODO: + we will only need certains haar-feature, not all, so why not compute only those needed ?
        listNeededFeatures();
    }

    private void listNeededFeatures() {

        this.NUM_FEATURES_A = 0;
        this.NUM_FEATURES_B = 0;
        this.NUM_FEATURES_C = 0;
        this.NUM_FEATURES_D = 0;
        this.NUM_FEATURES_E = 0;

        ArrayList<Feature> ft = new ArrayList<>();
        int cpt = 0;
        for (ArrayList<Feature> lf : streamFeaturesByType(new ImageHandler(new int[19][19], 19, 19))) {
            for (Feature f : lf) {
                if (neededHaarValues.containsKey(cpt))
                    ft.add(f);
                cpt++;
            }
        }

        int[] neededFeatures = new int[5 * ft.size()];
        cpt = 0;
        for (Feature f : ft) {
            neededFeatures[(cpt * 5)] = f.getType();
            if (f.getType() == FeatureExtractor.typeA)
                NUM_FEATURES_A++;
            else if (f.getType() == FeatureExtractor.typeB)
                NUM_FEATURES_B++;
            else if (f.getType() == FeatureExtractor.typeC)
                NUM_FEATURES_C++;
            else if (f.getType() == FeatureExtractor.typeD)
                NUM_FEATURES_D++;
            else
                NUM_FEATURES_E++;

            neededFeatures[(cpt * 5 + 1)] = f.getRectangle().getX();
            neededFeatures[(cpt * 5 + 2)] = f.getRectangle().getY();
            neededFeatures[(cpt * 5 + 3)] = f.getRectangle().getWidth();
            neededFeatures[(cpt * 5 + 4)] = f.getRectangle().getHeight();
            cpt++;
        }

        long size_outputA = 4 * NUM_FEATURES_A;
        long size_outputB = 4 * NUM_FEATURES_B;
        long size_outputC = 4 * NUM_FEATURES_C;
        long size_outputD = 4 * NUM_FEATURES_D;
        long size_outputE = 4 * NUM_FEATURES_E;

        int[] arrayTypeA = new int[(int) size_outputA];
        int[] arrayTypeB = new int[(int) size_outputB];
        int[] arrayTypeC = new int[(int) size_outputC];
        int[] arrayTypeD = new int[(int) size_outputD];
        int[] arrayTypeE = new int[(int) size_outputE];

        int a = 0;
        int b = 0;
        int c = 0;
        int d = 0;
        int e = 0;
        for (int i = 0; i < 5 * ft.size(); i += 5) {
            if (neededFeatures[i] == FeatureExtractor.typeA) {
                arrayTypeA[a] = neededFeatures[i + 1];
                arrayTypeA[a + 1] = neededFeatures[i + 2];
                arrayTypeA[a + 2] = neededFeatures[i + 3];
                arrayTypeA[a + 3] = neededFeatures[i + 4];

                a += 4;
            } else if (neededFeatures[i] == FeatureExtractor.typeB) {
                arrayTypeB[b] = neededFeatures[i + 1];
                arrayTypeB[b + 1] = neededFeatures[i + 2];
                arrayTypeB[b + 2] = neededFeatures[i + 3];
                arrayTypeB[b + 3] = neededFeatures[i + 4];

                b += 4;
            } else if (neededFeatures[i] == FeatureExtractor.typeC) {
                arrayTypeC[c] = neededFeatures[i + 1];
                arrayTypeC[c + 1] = neededFeatures[i + 2];
                arrayTypeC[c + 2] = neededFeatures[i + 3];
                arrayTypeC[c + 3] = neededFeatures[i + 4];

                c += 4;
            } else if (neededFeatures[i] == FeatureExtractor.typeD) {
                arrayTypeD[d] = neededFeatures[i + 1];
                arrayTypeD[d + 1] = neededFeatures[i + 2];
                arrayTypeD[d + 2] = neededFeatures[i + 3];
                arrayTypeD[d + 3] = neededFeatures[i + 4];

                d += 4;
            } else {
                arrayTypeE[e] = neededFeatures[i + 1];
                arrayTypeE[e + 1] = neededFeatures[i + 2];
                arrayTypeE[e + 2] = neededFeatures[i + 3];
                arrayTypeE[e + 3] = neededFeatures[i + 4];

                e += 4;
            }
        }

        cuMemAlloc(this.allRectanglesA, NUM_FEATURES_A * Sizeof.INT);
        cuMemcpyHtoD(this.allRectanglesA, Pointer.to(arrayTypeA), NUM_FEATURES_A * Sizeof.INT);

        cuMemAlloc(this.allRectanglesB, NUM_FEATURES_B * Sizeof.INT);
        cuMemcpyHtoD(this.allRectanglesB, Pointer.to(arrayTypeB), NUM_FEATURES_B * Sizeof.INT);

        cuMemAlloc(this.allRectanglesC, NUM_FEATURES_C * Sizeof.INT);
        cuMemcpyHtoD(this.allRectanglesC, Pointer.to(arrayTypeC), NUM_FEATURES_C * Sizeof.INT);

        cuMemAlloc(this.allRectanglesD, NUM_FEATURES_D * Sizeof.INT);
        cuMemcpyHtoD(this.allRectanglesD, Pointer.to(arrayTypeD), NUM_FEATURES_D * Sizeof.INT);

        cuMemAlloc(this.allRectanglesE, NUM_FEATURES_E * Sizeof.INT);
        cuMemcpyHtoD(this.allRectanglesE, Pointer.to(arrayTypeE), NUM_FEATURES_E * Sizeof.INT);
    }


    private void computeTypeN(long numFeatures, char type, float coeff) {

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, modules.get(type), KERNEL_NAME + type);


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
                Pointer.to(new float[]{coeff}),
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


    public void compute(float coeff) {
        if (this.tmpDataPtr == null) {
            System.err.println("ERROR HaarDetector not init - Aborting");
            System.exit(42);
        }
        // Initialisation of the input data that will be passed to Cuda
        // The image is larger to compute the filter without bounds checking

        // Allocate input data to CUDA memory
        // Pas utiliser ça...
        srcPtr = new CUdeviceptr();
        CudaUtils.newArray2D(integral, width, height, tmpDataPtr, srcPtr);

        computeTypeN(this.NUM_FEATURES_A, 'A', coeff);
        computeTypeN(this.NUM_FEATURES_B, 'B', coeff);
        computeTypeN(this.NUM_FEATURES_C, 'C', coeff);
        computeTypeN(this.NUM_FEATURES_D, 'D', coeff);
        computeTypeN(this.NUM_FEATURES_E, 'E', coeff);

        CudaUtils.freeArray2D(tmpDataPtr, srcPtr, width);

    }

    public void updateImage(int[][] newIntegral, int width, int height) {
        super.updateImage(newIntegral);

        this.tmpDataPtr = new CUdeviceptr[width];
        this.width = width;
        this.height = height;
    }

}

