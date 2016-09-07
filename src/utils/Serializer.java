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
    private static int[][] trainImagesFeatures = null;
    private static int[][] testImagesFeatures = null;
    private static boolean inMemory = false;
    public static long featureCount;

    private static ConcurrentHashMap<String, Integer> fileIndex = new ConcurrentHashMap<>();
    private static ConcurrentHashMap<String, Integer> fileTraining = new ConcurrentHashMap<>();

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

    public static int[] readFeaturesFromDisk(String filePath) {
        if (inMemory && fileIndex.containsKey(fileIndex)) {
            if (fileTraining.get(filePath) == 1)
                return trainImagesFeatures[fileIndex.get(filePath)];
            else
                return testImagesFeatures[fileIndex.get(filePath)];
        }

        return readArrayFromDisk(filePath, featureCount);
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

    public static void writeRule(ArrayList<StumpRule> committee, boolean firstRound, String fileName) {
        try {

            if (firstRound && Utils.fileExists(fileName))
                Utils.deleteFile(fileName);

            PrintWriter writer = new PrintWriter(new FileWriter(fileName, true));

            if (firstRound)
                writer.println("double stumps[][4]=");

            for (StumpRule stumpRule : committee) {
                writer.println(stumpRule.featureIndex + ";" + stumpRule.error + ";"
                        + stumpRule.threshold + ";" + stumpRule.toggle);
            }

            writer.flush();
            writer.close();

        } catch (IOException e) {
            System.err.println("Error : Could Not Write committee, aborting");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static ArrayList<StumpRule> readRule(String fileName) {
        ArrayList<StumpRule> result = new ArrayList<>();

        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            String line = br.readLine();

            while (line != null) {
                line = br.readLine();
                if (line == null || line.equals(""))
                    break;

                String[] parts = line.split(";");

                StumpRule stumpRule = new StumpRule(Long.parseLong(parts[0]),
                                                    Double.parseDouble(parts[1]),
                                                    Double.parseDouble(parts[2]),
                                                    -1,
                                                    Integer.parseInt(parts[3]));

                result.add(stumpRule);
            }

            br.close();

        } catch (IOException e) {
            System.err.println("Error while reading the list of decisionStumps");
            e.printStackTrace();
            System.exit(1);
        }
        return result;
    }

    public static void writeLayerMemory(ArrayList<Integer> layerMemory, ArrayList<Float> tweaks, String fileName) {
        try {

            int layerCount = layerMemory.size();
            PrintWriter writer = new PrintWriter(new FileWriter(fileName, true));

            writer.println(System.lineSeparator());
            writer.println("int layerCount=" + layerCount);
            writer.println("int layerCommitteeSize[]=");

            layerMemory.forEach(writer::println);

            writer.println("float tweaks[]=");

            tweaks.forEach(writer::println);
            /*
            for (int i = 0; i < layerCount; i++) {
                writer.println(tweaks.get(i));
            }
            */

            writer.close();

        } catch (IOException e) {
            System.err.println("Error : Could Not Write layer Memory or tweaks, aborting");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void buildImagesFeatures(ArrayList<String> faces, ArrayList<String> nonfaces, boolean training) {
        System.out.println("Caching images features values for " + (training ? "training" : "validation") + " set:");
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
        if (training)
            trainImagesFeatures = new int[N][(int) featureCount];
        else
            testImagesFeatures = new int[N][(int) featureCount];


        System.out.println("  - Reading all values from disk to memory...");
        for (int i = 0; i < N; i++) {
            String filePath = (i < posN ? faces.get(i) : nonfaces.get(i - posN)) + Conf.FEATURE_EXTENSION;
            fileIndex.putIfAbsent(filePath, i);
            fileTraining.putIfAbsent(filePath, training ? 1 : 0);

            if (training) {
                trainImagesFeatures[i] = Arrays.copyOf(readFeaturesFromDisk(filePath), (int) featureCount);
            }
            else {
                testImagesFeatures[i] = Arrays.copyOf(readFeaturesFromDisk(filePath), (int) featureCount);
            }

        }
        inMemory = true;


//        try {
//            if (fileExists(imagesFeatureFilepath)) {
//                if (force)
//                    deleteFile(imagesFeatureFilepath);
//                else {
//                    FileChannel fc = new RandomAccessFile(imagesFeatureFilepath, "r").getChannel();
//                    if (training)
//                        trainImagesFeatures = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
//                    else
//                        testImagesFeatures = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
//                    return;
//                }
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//            System.exit(1);
//        }

//        ArrayList<Integer> tmp = readArrayFromDisk(faces.get(0) + Conf.FEATURE_EXTENSION);
//
//        long length = Integer.BYTES * tmp.size() * (long)(faces.size() + nonfaces.size());
//        System.out.println("Length: " + length);

//        MappedByteBuffer out;
//        try {
//            int c = 0;
//            out = new RandomAccessFile(imagesFeatureFilepath, "rw").getChannel().map(FileChannel.MapMode.READ_WRITE, 0, length);
//            for (String face : faces){
//                tmp = readArrayFromDisk(face + Conf.FEATURE_EXTENSION);
//                for (long i = 0; i < tmp.size(); i++)
//                    out.putInt(tmp.get((int) i));
//                c++;
//            }
//            for (String nonface : nonfaces){
//                tmp = readArrayFromDisk(nonface + Conf.FEATURE_EXTENSION);
//                for (long i = 0; i < tmp.size();  i++)
//                    out.putInt(tmp.get((int) i));
//                c++;
//            }
//            if (c != faces.size() + nonfaces.size()) {
//                System.out.println("Sizes doesn't match! (c=" + c + " while expecting " + (faces.size() + nonfaces.size()) + ")");
//                System.exit(1);
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//            System.exit(1);
//        }
    }

    /**
     *
     * @return the layerCount
     * @param fileName the path to the file with the data needed
     * @param layerCommitteeSize ArrayList in which the size of each committe will be stored
     * @param tweaks ArrayList in which the tweaks will be stored
     */
    public static int readLayerMemory(String fileName, ArrayList<Integer> layerCommitteeSize, ArrayList<Float> tweaks) {

        int layercount = 0;

        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            String line = br.readLine();

            while (line != null) {
                line = br.readLine();
                if (line.equals("")) {
                    line = br.readLine();
                    break;
                }
            }

            line = br.readLine();
            String[] parts = line.split("=");
            layercount = Integer.parseInt(parts[1]);
            line = br.readLine();

            for (int i = 0; i < layercount; i++) {
                line = br.readLine();
                layerCommitteeSize.add(Integer.parseInt(line));
            }

            line = br.readLine();

            for (int i = 0; i < layercount; i++) {
                line = br.readLine();
                tweaks.add(Float.parseFloat(line));
            }

            br.close();

        } catch (IOException e) {
            System.err.println("Error while reading the layer memory");
            e.printStackTrace();
            System.exit(1);
        }

        return layercount;
    }
}