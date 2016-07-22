package process;

import jcuda.jcublas.JCublas;

public class Conf {
    public final static boolean USE_CUDA = isCUDAAvailable();
    public final static String TMP_DIR = "tmp";
    public final static String TRAIN_DIR = TMP_DIR + "/training";
    public final static String TRAIN_FEATURES = TRAIN_DIR + "/featuresValues.data";
    public final static int TRAIN_MAX_CONCURENT_PROCESSES = 20;

    public static boolean isCUDAAvailable() {
        try {
            JCublas.cublasInit();
            System.out.println("CUDA available!");
            return true;
        }
        catch (Throwable t) {
            System.out.println("CUDA not available!");
            return false;
        }
    }
}
