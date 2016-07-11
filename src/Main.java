import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

import java.util.Random;

/**
 * this is a main class
 */
public class Main {
    public static void main(String[] args) {
        int n = 1024;
        int loops = 1;

        float alpha = 0.3f;
        float beta = 0.7f;
        int nn = n * n;

        System.out.println("Creating input data...");
        float h_A[];
        float h_B[];
        float h_C[];

        System.out.println("Performing Sgemm with JCublas...");
        long cu = System.currentTimeMillis();
        for (int i = 0; i < loops; i++) {
            h_A = createRandomFloatData(nn);
            h_B = createRandomFloatData(nn);
            h_C = createRandomFloatData(nn);
            sgemmJCublas(n, alpha, h_A, h_B, beta, h_C);
        }
        cu = System.currentTimeMillis() - cu;

        System.out.println("Cuda time spent : " + cu);
    }

    private static void sgemmJCublas(int n, float alpha, float A[], float B[],
                                     float beta, float C[])
    {
        int nn = n * n;

        // Initialize JCublas
        JCublas.cublasInit();

        // Allocate memory on the device
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        JCublas.cublasAlloc(nn, Sizeof.FLOAT, d_A);
        JCublas.cublasAlloc(nn, Sizeof.FLOAT, d_B);
        JCublas.cublasAlloc(nn, Sizeof.FLOAT, d_C);

        // Copy the memory from the host to the device
        JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(A), 1, d_A, 1);
        JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(B), 1, d_B, 1);
        JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(C), 1, d_C, 1);

        // Execute sgemm
        JCublas.cublasSgemm(
                'n', 'n', n, n, n, alpha, d_A, n, d_B, n, beta, d_C, n);

        // Copy the result from the device to the host
        JCublas.cublasGetVector(nn, Sizeof.FLOAT, d_C, 1, Pointer.to(C), 1);

        //System.out.println("Preforming...");

        // Clean up
        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();
    }

    private static float[] createRandomFloatData(int n)
    {
        Random random = new Random();
        float x[] = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = random.nextFloat();
        }
        return x;
    }

}
