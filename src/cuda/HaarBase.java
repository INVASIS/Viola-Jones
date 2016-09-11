package cuda;

import jcuda.driver.CUdeviceptr;

public abstract class HaarBase implements AutoCloseable {
    protected int[][] integral;
    protected int width;
    protected int height;

    protected CUdeviceptr srcPtr;
    protected CUdeviceptr dstPtr;
    protected CUdeviceptr tmpDataPtr[];
}
