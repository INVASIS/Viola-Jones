package utils;
import java.io.*;
import java.nio.file.FileAlreadyExistsException;
import java.util.ArrayList;

import static javafx.application.Platform.exit;
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
            exit();
        }
    }

    public static void writeArrayToDisk(String filePath, ArrayList<Integer> values) {
        if (fileExists(filePath)) {
            new FileAlreadyExistsException(filePath).printStackTrace();
            exit();
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
            exit();
        }
        return result;
    }

    public static boolean validSizeOfArray(String filePath, long expectedSize) {
        DataInputStream os;

        try {
            os = new DataInputStream(new FileInputStream(filePath));
            skipBytesLong(os, Integer.BYTES *(expectedSize - 1));
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
            exit();
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
            exit();
        }

        return i;
    }
}