package process.features;

import process.Conf;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

@Deprecated
public class FeaturesSerializer {
    public static void toDisk(HashMap<String, ArrayList<Integer>> result, String filePath) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filePath), "utf-8"));
            for (Map.Entry<String, ArrayList<Integer>> entry : result.entrySet()) {
                writer.write(entry.getKey() + ":");
                for (Integer i : entry.getValue())
                    writer.write(i + ';');
                writer.write('\n');
            }
        } catch (IOException ex) {
            System.err.println("Could not write feature values to " + Conf.TRAIN_FEATURES);
        } finally {
            try {
                if (writer != null) {
                    writer.close();
                }
            } catch (Exception ex) {/*ignore*/}
        }
    }
    public static HashMap<String, ArrayList<Integer>> fromDisk(String filePath) {
        HashMap<String, ArrayList<Integer>> result = new HashMap<>();

        if (Files.exists(Paths.get(filePath)))
            return result;

        final Scanner s = new Scanner(filePath);
        while(s.hasNextLine()) {
            final String line = s.nextLine();
            if (line.equals(""))
                continue;

            String[] parts = line.split(":");

            ArrayList<Integer> values = new ArrayList<>();
            int i = 0;
            for (String val : parts[1].split(";")) {
                values.add(i, Integer.parseInt(val));
                i++;
            }
            result.put(parts[0], values);
        }
        return result;
    }
}
