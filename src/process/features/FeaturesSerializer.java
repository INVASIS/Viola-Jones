package process.features;

import process.Conf;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;


public class FeaturesSerializer {

    public static void imageFeaturesToDisk(String filePath, ArrayList<ArrayList<Integer>> features) {
        PrintWriter writer = null;
        try {
            writer = new PrintWriter(filePath, "UTF-8");

            for (ArrayList<Integer> featuresOfType : features) {
                for (Integer i : featuresOfType)
                    writer.write(i + ";");
                writer.write(System.lineSeparator());
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

    public static ArrayList<ArrayList<Integer>> imageFeaturesFromDisk(String filePath) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(filePath));
            String line = br.readLine();
            while (line != null) {
                ArrayList<Integer> values = new ArrayList<>();
                for (String val : line.split(";")) {
                    values.add(Integer.parseInt(val));
                }
                res.add(values);
                line = br.readLine();
            }
            br.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

        return res;
    }
}
