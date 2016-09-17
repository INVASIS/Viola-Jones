package process;

import cuda.CudaUtils;
import cuda.HaarExtractor;
import jcuda.runtime.cudaDeviceProp;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import static utils.Utils.listFiles;

public class Conf {

    public static HaarExtractor haarExtractor;

    public static boolean USE_CUDA = isCUDAAvailable();
    public static int maxThreadsPerBlock;
    public static int multiProcessorCount;
    public static int maxThreadsPerMultiProcessor;
    public static long maxGPUMemory;
    public static final int maxBlocksByDim = 65535;
    public final static String TMP_DIR = "tmp";
    public final static String LIB_DIR = "libs";
    public final static String TRAIN_DIR = TMP_DIR + "/training";
    public final static String TEST_DIR = TMP_DIR + "/test";
    public final static String FACES = "/faces";
    public final static String NONFACES = "/non-faces";
    public final static String IMAGES_FEATURES_TRAIN = TRAIN_DIR + "/imagesFeatures.data";
    public final static String IMAGES_FEATURES_TEST = TEST_DIR + "/imagesFeatures.data";
    public final static String ORGANIZED_FEATURES = TRAIN_DIR + "/organizedFeatures.data";
    public final static String ORGANIZED_SAMPLE = TRAIN_DIR + "/organizedSample.data";
    public final static String TRAIN_FEATURES = TRAIN_DIR + "/featuresValues.data";
    public final static int TRAIN_MAX_CONCURENT_PROCESSES = 20;
    public final static boolean PATH_CREATED = createPaths();
    public final static int TRAIN_MAX_ROUNDS = 20;
    public final static String FEATURE_EXTENSION = ".haar";
    public final static String IMAGES_EXTENSION = ".png";
    public final static int CUDA_DEVICE_ID = 0;



    public static boolean isCUDAAvailable() {
        try {
            CudaUtils.initCuda();
            System.out.println("CUDA available!");
            System.out.println("  - maxThreadsPerBlock: " + maxThreadsPerBlock);
            System.out.println("  - multiProcessorCount: " + multiProcessorCount);
            System.out.println("  - maxThreadsPerMultiProcessor: " + maxThreadsPerMultiProcessor);
            System.out.println("  - maxGPUMemory: " + (((maxGPUMemory/1024)/1024)/1024) + "Gio");
            haarExtractor = new HaarExtractor();
            return true;
        }
        catch (Throwable t) {
            System.out.println("CUDA not available!");
            return false;
        }
    }

    public static boolean createPaths() {
        try {
            Files.createDirectories(Paths.get(TMP_DIR));
            Files.createDirectories(Paths.get(TRAIN_DIR));
            Files.createDirectories(Paths.get(TEST_DIR));
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static void loadLibraries() {
        String lib_ext;
        String jcuda;
        if (System.getProperty("os.name").contains("Windows")) {
            jcuda = "\\JCuda-All-0.7.5b-bin-windows-x86_64";
            lib_ext = "dll";
        } else {
            jcuda = "/JCuda-All-0.7.5b-bin-linux-x86_64";
            lib_ext = "so";
        }

        ArrayList<String> files = new ArrayList<>();
        files.addAll(listFiles(Conf.LIB_DIR + jcuda + "", lib_ext));
        for (String f : files) {
            String abspath = new File(f).getAbsolutePath();
            System.out.println("Loading external library: " + abspath);
            System.load(abspath);
        }
    }
}
