package process.features;

import org.junit.Test;
import process.Conf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import static junit.framework.TestCase.assertEquals;
import static process.features.FeatureExtractor.computeImageFeatures;
import static utils.Serializer.*;
import static utils.Serializer.readArrayOfArrayFromDisk;


public class TestSerializer {


    // FIXME : J'ai fix l'autre test mais ça fait rater celui la
    // car j'ai modifié readArrayOfArrasFromDisk pour le fix.
    // Je laisse comme ça que tu change en fonction de ce qu'il te convient le mieux
    @Test
    public void writeReadArray() {
        String filePath = Conf.TEST_DIR + "/writeReadArray.data";
        ArrayList<Integer> r1 = new ArrayList<>();
        r1.add(1);
        r1.add(2);
        r1.add(3);
        r1.add(5);
        r1.add(7);
        appendArrayToDisk(filePath, r1);
        ArrayList<Integer> r2 = new ArrayList<>();
        r2.add(8);
        r2.add(9);
        r2.add(1);
        r2.add(6);
        r2.add(10);
        appendArrayToDisk(filePath, r2);
        ArrayList<ArrayList<Integer>> result = readArrayOfArrayFromDisk(filePath);
        assertEquals(result.get(0).get(4), new Integer(7));
        assertEquals(result.get(1).get(4), new Integer(10));
        assertEquals(result.get(0).get(0), new Integer(1));
        assertEquals(readIntFromDisk(filePath, 1, 3), 6);
        assertEquals(readIntFromDisk(filePath, 4+4), 6);
        try {
            Files.delete(Paths.get(filePath));
        } catch (IOException e) {/*ignored*/}
        writeArrayOfArrayToDisk(filePath, result);
        result = readArrayOfArrayFromDisk(filePath);
        assertEquals(result.get(0).get(4), new Integer(7));
        assertEquals(result.get(1).get(4), new Integer(10));
        assertEquals(result.get(0).get(0), new Integer(1));
        assertEquals(readIntFromDisk(filePath, 1, 3), 6);
        assertEquals(readIntFromDisk(filePath, 4+4), 6);
        try {
            Files.delete(Paths.get(filePath));
        } catch (IOException e) {/*ignored*/}
    }

    @Test
    public void computeWriteAndRead() {
        Conf.haarExtractor.setUp(19, 19);
        String img = "data/trainset/faces/face00001.png";
        String haar = img + Conf.FEATURE_EXTENSION;

        if (Files.exists(Paths.get(haar))) {
            try {
                Files.delete(Paths.get(haar));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        ArrayList<ArrayList<Integer>> correctValues = computeImageFeatures(img, true);
        ArrayList<ArrayList<Integer>> writtenValues = readArrayOfArrayFromDisk(haar);

        for (int i = 0; i < correctValues.size(); i++)
            for (int j = 0; j < correctValues.get(i).size(); j++)
                assertEquals(writtenValues.get(i).get(j), correctValues.get(i).get(j));
    }
}
