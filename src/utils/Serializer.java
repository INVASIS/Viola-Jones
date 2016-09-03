package utils;

import process.DecisionStump;

import java.io.*;
import java.math.BigDecimal;
import java.nio.file.FileAlreadyExistsException;
import java.text.DecimalFormat;
import java.util.ArrayList;

import static utils.Utils.fileExists;


public class Serializer {
    private static void skipBytesLong(DataInputStream dis, long skip) throws IOException {
        long total = 0;
        long cur = 0;

        while ((total < skip) && ((cur = dis.skip(skip - total)) > 0)) {
            total += cur;
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

    public static void writeArrayToDisk(String filePath, ArrayList<Integer> values) {
        if (fileExists(filePath)) {
            new FileAlreadyExistsException(filePath).printStackTrace();
            System.exit(1);
        }
        appendArrayToDisk(filePath, values);
    }

    public static ArrayList<Integer> readArrayFromDisk(String filePath) {
        ArrayList<Integer> result = new ArrayList<>();

        DataInputStream os;
        try {
            os = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)));
            while (true) {
                try {
                    result.add(os.readInt());
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

    public static ArrayList<Integer> readArrayFromDisk(String filePath, long fromIndex, long toIndex) {
        ArrayList<Integer> result = new ArrayList<>();

        DataInputStream os;
        try {
            os = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)));
            skipBytesLong(os, Integer.BYTES * fromIndex);
            long i = 0;
            while (true) {
                try {
                    result.add(os.readInt());
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
        assert result.size() == (toIndex - fromIndex);
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

    public static void writeRule(ArrayList<DecisionStump> committee, boolean firstRound, String fileName) {
        try {

            if (firstRound && Utils.fileExists(fileName))
                Utils.deleteFile(fileName);

            PrintWriter writer = new PrintWriter(new FileWriter(fileName, true));
            int memberCount = committee.size();

            if (firstRound)
                writer.println("double stumps[][4]=");

            for (int i = 0; i < memberCount; i++) {
                DecisionStump decisionStump = committee.get(i);
                writer.println(decisionStump.featureIndex + ";" + decisionStump.error + ";"
                        + decisionStump.threshold + ";" + decisionStump.toggle);
            }

            writer.flush();
            writer.close();

        } catch (IOException e) {
            System.err.println("Error : Could Not Write committe, aborting");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static ArrayList<DecisionStump> readRule(String fileName) {
        ArrayList<DecisionStump> result = new ArrayList<>();

        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            String line = br.readLine();

            while (line != null) {
                line = br.readLine();
                if (line == null || line.equals(""))
                    break;

                String[] parts = line.split(";");

                BigDecimal error = new BigDecimal(parts[1]);
                DecisionStump decisionStump = new DecisionStump(Long.parseLong(parts[0]), error,
                        Double.parseDouble(parts[2]), -1, Integer.parseInt(parts[3]));

                result.add(decisionStump);
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

            for (int i = 0; i < layerCount; i++) {
                writer.println(layerMemory.get(i) + ";");
            }

            writer.println("float tweaks[]=");
            for (int i = 0; i < layerCount; i++) {
                writer.println(tweaks.get(i) + ";");
            }

            writer.close();

        } catch (IOException e) {
            System.err.println("Error : Could Not Write layer Memory or tweaks, aborting");
            e.printStackTrace();
            System.exit(1);
        }
    }
}