package process.features;

import org.junit.Test;
import process.Conf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import static junit.framework.TestCase.assertEquals;
import static utils.Serializer.appendArrayToDisk;
import static utils.Serializer.readArrayOfArrayFromDisk;
import static utils.Serializer.readIntFromDisk;


public class TestSerializer {
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
        try {
            Files.delete(Paths.get(filePath));
        } catch (IOException e) {/*ignored*/}
    }
}
