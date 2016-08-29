package utils;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import static javafx.application.Platform.exit;


public class Serializer {
    public static void appendArrayToDisk(String filePath, ArrayList<Integer> values) {
        /**
         * This writer considers that, for a single file, length of values array will always be the same.
         * The first int written in the file is the length of values for subsequent uses. Then comes values.
         */
        DataOutputStream os;
        try {
            boolean b = Files.exists(Paths.get(filePath));
            if (!b) {
                os = new DataOutputStream(new FileOutputStream(filePath, false));
                os.writeInt(values.size()); // First int will always be the size of the array!
            }
            else {
                os = new DataOutputStream(new FileOutputStream(filePath, true));
            }
            for (Integer i : values)
                os.writeInt(i);
            os.close();
        } catch (IOException e) {
            System.err.println("Could not write to " + filePath);
            e.printStackTrace();
            exit();
        }
    }

    public static void appendArrayOfArrayToDisk(String filePath, ArrayList<ArrayList<Integer>> values) {
        DataOutputStream os;
        try {
            boolean b = Files.exists(Paths.get(filePath));
            if (!b) {
                os = new DataOutputStream(new FileOutputStream(filePath, false));
                os.writeInt(values.get(0).size()); // First int will always be the size of the array!
            }
            else {
                os = new DataOutputStream(new FileOutputStream(filePath, true));
            }
            for (ArrayList<Integer> a : values)
                for (Integer i : a)
                    os.writeInt(i);
            os.close();
        } catch (IOException e) {
            System.err.println("Could not write to " + filePath);
            e.printStackTrace();
            exit();
        }
    }

    public static ArrayList<Integer> readArrayFromDisk(String filePath, int index) throws IOException {
        ArrayList<Integer> result = new ArrayList<>();

        DataInputStream os;
        os = new DataInputStream(new FileInputStream(filePath));
        int elementsByArray = os.readInt();
        os.skipBytes(Integer.BYTES * index * elementsByArray);
        for (int i = 0; i < elementsByArray; i++)
            result.add(os.readInt());
        os.close();
        return result;
    }

    public static ArrayList<ArrayList<Integer>> readArrayOfArrayFromDisk(String filePath) {
        boolean b = true;
        int i = 0;

        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        while (b) {
            try {
                result.add(readArrayFromDisk(filePath, i));
                i++;
            } catch (IOException e) {
                b = false;
            }
        }
        return result;
    }

    public static int readIntFromDisk(String filePath, int arrayIndex, int valueIndex) {
        DataInputStream os;

        int i = 0;
        try {
            os = new DataInputStream(new FileInputStream(filePath));
            int elementsByArray = os.readInt();
            os.skipBytes(Integer.BYTES * arrayIndex * elementsByArray + Integer.BYTES * valueIndex);
            i = os.readInt();
            os.close();
        } catch (IOException e) {
            System.err.println("Could not read int from " + filePath + "!");
            e.printStackTrace();
            exit();
        }

        return i;
    }

    private static void skipBytesLong(DataInputStream dis, long skip) throws IOException {
        int total = 0;
        int cur = 0;

        while ((total<skip) && ((cur = (int) dis.skip(skip-total)) > 0)) {
            total += cur;
        }
    }

    public static int readIntFromDisk(String filePath, long valueIndex) {
        DataInputStream os;

        int i = 0;
        try {
            os = new DataInputStream(new FileInputStream(filePath));
            os.readInt();
            skipBytesLong(os, Integer.BYTES * valueIndex);
            i = os.readInt();
            os.close();
        } catch (IOException e) {
            System.err.println("Could not read int from " + filePath + "!");
            e.printStackTrace();
            exit();
        }

        return i;
    }
}