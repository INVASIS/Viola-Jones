package utils;
import java.io.*;
import java.nio.file.FileAlreadyExistsException;
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

    public static void writeArrayOfArrayToDisk(String filePath, ArrayList<ArrayList<Integer>> values) {
        System.out.println("writeArrayOfArrayToDisk(" + filePath + ")");
        if (Files.exists(Paths.get(filePath))) {
            new FileAlreadyExistsException(filePath).printStackTrace();
            exit();
        }
        
        DataOutputStream os;
        try {
            int j = 0;
            os = new DataOutputStream(new FileOutputStream(filePath, false));
            for (ArrayList<Integer> a : values) {
                os.writeInt(values.get(j++).size()); // First int will always be the size of the array! - Nope ! arrays are not always the same length :)
                for (Integer i : a)
                    os.writeInt(i);
            }
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
        int i = 0;

        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        try {
            DataInputStream os = new DataInputStream(new FileInputStream(filePath));
            while (true) {
                try {
                    int elementsByArray = os.readInt();
                    ArrayList<Integer> tmp = new ArrayList<>();
                    for (int j = 0; j < elementsByArray; j++)
                        tmp.add(os.readInt());
                    result.add(tmp);
                    i++;
                } catch (EOFException e) {
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            exit();
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