package process;

import jcuda.jcublas.JCublas;

public class Conf {
    public final static boolean USE_CUDA = isCUDAAvailable();

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
