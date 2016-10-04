package utils;

import process.Conf;
import process.StumpRule;

import java.io.*;
import java.nio.file.FileAlreadyExistsException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;

import static utils.Utils.fileExists;


public class Serializer {
    // Stores feature values for each example: xxxImagesFeatures[i] contains all feature values of example i
    private static int[][] trainImagesFeatures = null;
    private static int[][] testImagesFeatures = null;
    private static int[][] organizedFeatures = null;
    private static int[][] organizedSamples = null;

    // Map a filepath to the index of xxxImagesFeatures
    private static ConcurrentHashMap<String, Integer> fileIndex = new ConcurrentHashMap<>();

    // If String matches a training example, then it returns 1
    // If String matches a validation example, then it returns 0
    private static ConcurrentHashMap<String, Integer> fileTraining = new ConcurrentHashMap<>();

    private static boolean inMemory = false;
    public static long featureCount;


    public static boolean isInMemory() {
        return inMemory;
    }

    public static void clearTrainImagesFeatures() {
        trainImagesFeatures = null;
        System.gc();
    }

    public static void loadOrganizedInMemory(String features, String samples, int trainN) {
        if (organizedFeatures != null) {
            System.err.println("Already loaded organizedFeatures in memory...");
            System.exit(1);
        }
        organizedFeatures = new int[(int)featureCount][trainN];
        organizedSamples = new int[(int)featureCount][trainN];
        for (int i = 0; i < featureCount; i++) {
            organizedFeatures[i] = readArrayFromDisk(features, i * trainN, trainN * (i + 1));
            organizedSamples[i] = readArrayFromDisk(samples, i * trainN, trainN * (i + 1));
        }
    }

    private static void skipBytesLong(DataInputStream dis, long skip) throws IOException {
        long total = 0;
        long cur;

        while ((total < skip) && ((cur = dis.skip(skip - total)) > 0)) {
            total += cur;
        }
    }

    public static void appendArrayToDisk(String filePath, int[] values, long size) {
        DataOutputStream os;
        try {
            os = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath, fileExists(filePath))));
            for (long i = 0; i < size; i++)
                os.writeInt(values[(int) i]);
            os.close();
        } catch (IOException e) {
            System.err.println("Could not write to " + filePath);
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void appendArrayToDisk(String filePath, ArrayList<Integer> values) {
        DataOutputStream os;
        try {
            os = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath, fileExists(filePath))));
            for (int i : values)
                os.writeInt(i);
            os.close();
        } catch (IOException e) {
            System.err.println("Could not write to " + filePath);
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void writeArrayToDisk(String filePath, int[] values, long size) {
        if (fileExists(filePath)) {
            new FileAlreadyExistsException(filePath).printStackTrace();
            System.exit(1);
        }
        appendArrayToDisk(filePath, values, size);
    }

    public static void writeArrayToDisk(String filePath, ArrayList<Integer> values) {
        if (fileExists(filePath)) {
            new FileAlreadyExistsException(filePath).printStackTrace();
            System.exit(1);
        }
        appendArrayToDisk(filePath, values);
    }

    public static int[] readArrayFromDisk(String filePath, long expectedSize) {
        int[] result = new int[(int) expectedSize];
        DataInputStream os;
        try {
            long c = 0;
            os = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)));
            while (true) {
                if (c == expectedSize)
                    break;
                try {
                    result[(int)c] = os.readInt();
                    c++;
                } catch (EOFException e) {
                    break;
                }
            }
            os.close();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        return result;
    }

    public static int[] readFeatures(String filePath) {
        if (inMemory && fileIndex.containsKey(filePath)) {
        if (fileTraining.get(filePath) == 1)
            return trainImagesFeatures[fileIndex.get(filePath)];
        else
            return testImagesFeatures[fileIndex.get(filePath)];
    }
//        System.out.println("readFeatures(" + filePath + ") not in memory...");
        return readArrayFromDisk(filePath, featureCount);
}

    public static int[] featureExamplesIndexes(String filePath, long featureIndex, int trainN) {
        if (organizedSamples != null)
            return organizedSamples[(int)featureIndex];
        return readArrayFromDisk(filePath, featureIndex * trainN, trainN * (featureIndex + 1));
    }

    public static int[] featureValues(String filePath, long featureIndex, int trainN) {
        if (organizedFeatures != null)
            return organizedFeatures[(int)featureIndex];
        return readArrayFromDisk(filePath, featureIndex * trainN, trainN * (featureIndex + 1));
    }

    public static int[] readArrayFromDisk(String filePath, long fromIndex, long toIndex) {
        int[] result = new int[(int) (toIndex-fromIndex)];

        DataInputStream os;
        try {
            os = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)));
            skipBytesLong(os, Integer.BYTES * fromIndex);
            long i = 0;
            while (true) {
                if (i == (toIndex-fromIndex))
                    break;
                try {
                    result[(int) i] = os.readInt();
                    i++;
                } catch (EOFException e) {
                    break;
                }
                if (i == (toIndex - fromIndex))
                    break;
            }
            os.close();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        return result;
    }

    /**
     * Checks if filePath contains exactly expectedSize values.
     */
    public static boolean validSizeOfArray(String filePath, long expectedSize) {
        DataInputStream os;

        try {
            os = new DataInputStream(new FileInputStream(filePath));
            skipBytesLong(os, Integer.BYTES * (expectedSize - 1));
            os.readInt(); // Read the last expected int
            try {
                os.readInt(); // Should raise EOFException if expectedSize is right
                os.close();
            } catch (EOFException e) {
                return true;
            }
        } catch (EOFException e) {
            return false;
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        return false;
    }

    public static int readIntFromDisk(String filePath, long valueIndex) {
        DataInputStream os;

        int i = 0;
        try {
            os = new DataInputStream(new FileInputStream(filePath));
            skipBytesLong(os, Integer.BYTES * valueIndex);
            i = os.readInt();
            os.close();
        } catch (IOException e) {
            System.err.println("Could not read int from " + filePath + "!");
            e.printStackTrace();
            System.exit(1);
        }

        return i;
    }

    public static int readIntFromMemory(String filePath, long featureIndex) {
        if (!inMemory)
            return readIntFromDisk(filePath, featureIndex);

        if (fileTraining.get(filePath) == 1)
            return trainImagesFeatures[fileIndex.get(filePath)][(int) featureIndex];
        else
            return testImagesFeatures[fileIndex.get(filePath)][(int) featureIndex];
    }

    public static void buildImagesFeatures(ArrayList<String> faces, ArrayList<String> nonfaces, boolean trainingSet) {
        // Build xxxImagesFeatures arrays for O(1) access to feature values

        System.out.println("Caching images features values for " + (trainingSet ? "training" : "validation") + " set:");
        int posN = faces.size();
        int negN = nonfaces.size();
        int N = posN + negN;

        long presumableFreeMemory = Runtime.getRuntime().maxMemory() - (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
        long neededMemory = featureCount * Integer.BYTES * N;
        System.out.println("  - Needed memory: " + neededMemory + " (presumable free memory: " + presumableFreeMemory + ")");
        if (!(presumableFreeMemory > neededMemory)) {
            System.out.println("    - Could not store in memory");
            inMemory = false;
            return;
        }

        if (featureCount > Integer.MAX_VALUE) {
            System.out.println("Size exceeds Integer.MAX_VALUE");
            System.exit(1);
        }

        System.out.println("  - Initializing int[" + N + "][" + featureCount + "]...");
        if (trainingSet)
            trainImagesFeatures = new int[N][(int) featureCount];
        else
            testImagesFeatures = new int[N][(int) featureCount];


        System.out.println("  - Reading all values from disk to memory...");
        for (int i = 0; i < N; i++) {
            String filePath = (i < posN ? faces.get(i) : nonfaces.get(i - posN)) + Conf.FEATURE_EXTENSION;
            fileIndex.putIfAbsent(filePath, i);
            fileTraining.putIfAbsent(filePath, trainingSet ? 1 : 0);

            if (trainingSet) {
                trainImagesFeatures[i] = Arrays.copyOf(readFeatures(filePath), (int) featureCount);
            }
            else {
                testImagesFeatures[i] = Arrays.copyOf(readFeatures(filePath), (int) featureCount);
            }

        }
        
        if (trainingSet)
            inMemory = true;
    }
}