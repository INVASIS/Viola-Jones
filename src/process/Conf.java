package process;

import cuda.CudaUtils;
import cuda.HaarExtractor;
import jcuda.jcublas.JCublas;

import java.io.IOException;
import java.io.StreamCorruptedException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Conf {

    public static HaarExtractor haarExtractor;

    public final static boolean USE_CUDA = isCUDAAvailable();
    public final static String TMP_DIR = "tmp";
    public final static String TRAIN_DIR = TMP_DIR + "/training";
    public final static String ORGANIZED_FEATURES = TRAIN_DIR + "/organizedFeatures";
    public final static String ORGANIZED_SAMPLE = TRAIN_DIR + "/organizedSample";
    public final static String TRAIN_FEATURES = TRAIN_DIR + "/featuresValues.data";
    public final static int TRAIN_MAX_CONCURENT_PROCESSES = 1; // FIXME
    public final static boolean PATH_CREATED = createPaths();
    public final static int TRAIN_MAX_ROUNDS = 20;
    public final static String FEATURE_EXTENSION = ".haar";



    public static boolean isCUDAAvailable() {
        try {
            CudaUtils.initCuda();
            System.out.println("CUDA available!");
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
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }
}
