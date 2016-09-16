package process.features;

import org.junit.Assert;
import org.junit.Test;
import process.Conf;
import process.StumpRule;
import utils.Serializer;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import static junit.framework.TestCase.assertEquals;
import static process.features.FeatureExtractor.computeImageFeatures;
import static process.features.FeatureExtractor.countAllFeatures;
import static utils.Serializer.*;


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
        int[] result = readArrayFromDisk(filePath, 10);
        assertEquals(result[0], 1);
        assertEquals(result[1], 2);
        assertEquals(result[2], 3);
        assertEquals(result[3], 5);
        assertEquals(result[4], 7);
        assertEquals(result[5], 8);
        assertEquals(result[6], 9);
        assertEquals(result[7], 1);
        assertEquals(result[8], 6);
        assertEquals(result[9], 10);
        assertEquals(readIntFromDisk(filePath, 8), 6);

        try {
            Files.delete(Paths.get(filePath));
        } catch (IOException e) {
            assertEquals(false, true);
        }

        writeArrayToDisk(filePath, result, 10);
        result = readArrayFromDisk(filePath, 10);
        assertEquals(result[0], 1);
        assertEquals(result[1], 2);
        assertEquals(result[2], 3);
        assertEquals(result[3], 5);
        assertEquals(result[4], 7);
        assertEquals(result[5], 8);
        assertEquals(result[6], 9);
        assertEquals(result[7], 1);
        assertEquals(result[8], 6);
        assertEquals(result[9], 10);
        assertEquals(readIntFromDisk(filePath, 8), 6);

        result = readArrayFromDisk(filePath, 3, 6);
        assertEquals(result[0], 5);
        assertEquals(result[1], 7);
        assertEquals(result[2], 8);
        //assertEquals(result.get(3), new Integer(9));

        try {
            Files.delete(Paths.get(filePath));
        } catch (IOException e) {
            assertEquals(false, true);
        }
    }

    @Test
    public void computeWriteAndRead() {
        Serializer.featureCount = countAllFeatures(19, 19);
        if (Conf.USE_CUDA)
            Conf.haarExtractor.setUp(19, 19);
        String img = "data/trainset/faces/face00001.png";
        String haar = img + Conf.FEATURE_EXTENSION;

        if (Files.exists(Paths.get(haar))) {
            try {
                Files.delete(Paths.get(haar));
            } catch (IOException e) {
                e.printStackTrace();
                assertEquals(false, true);
            }
        }

        int[] correctValues = computeImageFeatures(img, true);
        int[] writtenValues = readArrayFromDisk(haar, countAllFeatures(19, 19));


        for (int i = 0; i < countAllFeatures(19, 19); i++)
            assertEquals(writtenValues[i], correctValues[i]);
    }

    @Test
    public void printRuleTest() {
        String tmp_file = "tmp/test/featuresValues.data";

        ArrayList<StumpRule> committee = new ArrayList<>();

        StumpRule stumpRule = new StumpRule(1, 1, 1, 1, 1);
        committee.add(stumpRule);

        Serializer.writeRule(committee, true, tmp_file);

        StumpRule stumpRule2 = new StumpRule(2, 2, 2, 2, -1);
        committee.add(stumpRule2);

        Serializer.writeRule(committee, false, tmp_file);

        ArrayList<StumpRule> read = Serializer.readRule(tmp_file);

        assertEquals(committee.size() + 1, read.size());

        assertEquals(committee.get(0).error, read.get(0).error);
        assertEquals(committee.get(0).featureIndex, read.get(0).featureIndex);
        assertEquals(committee.get(0).threshold, read.get(0).threshold);
        assertEquals(committee.get(0).toggle, read.get(0).toggle);

        assertEquals(committee.get(0).error, read.get(1).error);
        assertEquals(committee.get(0).featureIndex, read.get(1).featureIndex);
        assertEquals(committee.get(0).threshold, read.get(1).threshold);
        assertEquals(committee.get(0).toggle, read.get(1).toggle);

        assertEquals(committee.get(1).error, read.get(2).error);
        assertEquals(committee.get(1).featureIndex, read.get(2).featureIndex);
        assertEquals(committee.get(1).threshold, read.get(2).threshold);
        assertEquals(committee.get(1).toggle, read.get(2).toggle);
    }

    @Test
    public void printLayerMemoryTest() {
        String tmp_file = "tmp/test/featuresValues.data";

        ArrayList<StumpRule> committee = new ArrayList<>();

        StumpRule stumpRule = new StumpRule(1, 1, 1, 1, 1);
        StumpRule stumpRule2 = new StumpRule(2, 2, 2, 2, -1);
        StumpRule stumpRule3 = new StumpRule(3, 3, 3, 3, 1);
        StumpRule stumpRule4 = new StumpRule(4, 4, 4, 4, -1);
        StumpRule stumpRule5 = new StumpRule(5, 5, 5, 5, 1);
        committee.add(stumpRule);
        committee.add(stumpRule2);
        committee.add(stumpRule3);
        committee.add(stumpRule4);
        committee.add(stumpRule5);

        Serializer.writeRule(committee, true, tmp_file);

        ArrayList<Integer> layerMemory = new ArrayList<>();
        layerMemory.add(2);
        layerMemory.add(2);
        layerMemory.add(1);

        ArrayList<Float> tweaks = new ArrayList<>();
        tweaks.add(0.000122f);
        tweaks.add(0.101010f);
        tweaks.add(0.424242f);

        Serializer.writeLayerMemory(layerMemory, tweaks, tmp_file);

        ArrayList<Integer> layerCommitteSize = new ArrayList<>();
        ArrayList<Float> readTweaks = new ArrayList<>();

        int layerCount = Serializer.readLayerMemory(tmp_file, layerCommitteSize, readTweaks);

        Assert.assertEquals(3, layerCount);
        Assert.assertEquals(3, layerCommitteSize.size());
        Assert.assertEquals(3, readTweaks.size());

        for (int i = 0; i < 3; i++) {
            Assert.assertEquals(layerMemory.get(i), layerCommitteSize.get(i));
            Assert.assertEquals(tweaks.get(i), readTweaks.get(i));
        }
    }
}
