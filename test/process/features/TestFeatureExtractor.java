package process.features;

import GUI.ImageHandler;
import org.junit.Assert;
import org.junit.Test;
import process.Conf;
import utils.Serializer;
import utils.Utils;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;
import static process.features.FeatureExtractor.*;

public class TestFeatureExtractor {

    public final static String TRAIN_DIR = "tmp/test/";
    public final static String ORGANIZED_FEATURES = TRAIN_DIR + "/organizedFeatures.data";
    public final static String ORGANIZED_SAMPLE = TRAIN_DIR + "/organizedSample.data";


    @Test
    public void featuresChecker() {
        int[][] tmp = {{0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}};
        ImageHandler image = new ImageHandler(tmp, 4, 4);

        assertEquals(40, FeatureExtractor.listAllTypeA(image).size()); // 2*1 -> 40
        assertEquals(20, FeatureExtractor.listAllTypeB(image).size()); // 3*1 -> 20
        assertEquals(40, FeatureExtractor.listAllTypeC(image).size()); // 1*2 -> 40
        assertEquals(20, FeatureExtractor.listAllTypeD(image).size()); // 1*3 -> 20
        assertEquals(16, FeatureExtractor.listAllTypeE(image).size()); // 2*2 -> 16

        ArrayList<Feature> tmp_lf = new ArrayList<>();

        FeatureExtractor.streamAllTypeA(image).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(40, tmp_lf.size()); // 2*1 -> 40

        tmp_lf.clear();

        FeatureExtractor.streamAllTypeB(image).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(20, tmp_lf.size()); // 3*1 -> 20

        tmp_lf.clear();

        FeatureExtractor.streamAllTypeC(image).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(40, tmp_lf.size()); // 1*2 -> 40

        tmp_lf.clear();

        FeatureExtractor.streamAllTypeD(image).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(20, tmp_lf.size()); // 1*3 -> 20

        tmp_lf.clear();

        FeatureExtractor.streamAllTypeE(image).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(16, tmp_lf.size()); // 2*2 -> 16
    }

    @Test
    public void countFeaturesTest() {
        // Manually count: http://stackoverflow.com/a/1711158/3157230

        assertEquals(136L, FeatureExtractor.countAllFeatures(4, 4));
        assertEquals(162336L, FeatureExtractor.countAllFeatures(24, 24));
        assertEquals(29979041500L, FeatureExtractor.countAllFeatures(500, 500));
        assertEquals(8 + 6 + 4 + 2 + 9 + 3 + 6 + 2 + 3 + 1 + 4 + 3 + 2 + 1 + 6 + 4 + 2 + 6 + 2, FeatureExtractor.countAllFeatures(4, 3));
    }

    @Test
    public void getExampleIndexAllRAMTest() {
        getExampleIndexTest(false);
    }

    @Test
    public void getExampleIndexForceDiskTest() {
        getExampleIndexTest(true);
    }

    private void getExampleIndexTest(boolean forceDisk) {
        if (Conf.USE_CUDA)
            Conf.haarExtractor.setUp(19, 19);

        String img1 = "data/trainset/faces/face00001.png";
        String img2 = "data/trainset/faces/face00002.png";
        String img3 = "data/trainset/faces/face00003.png";
        String img4 = "data/trainset/faces/face00004.png";

        ArrayList<String> files = new ArrayList();
        files.add(img1);
        files.add(img2);
        files.add(img3);
        files.add(img4);

        ArrayList<Integer> features1;
        if (Utils.fileExists(img1))
            features1 = Serializer.readArrayFromDisk(img1 + Conf.FEATURE_EXTENSION);
        else
            features1 = computeImageFeatures(img1, true);

        ArrayList<Integer> features2;
        if (Utils.fileExists(img2))
            features2 = Serializer.readArrayFromDisk(img2 + Conf.FEATURE_EXTENSION);
        else
            features2 = computeImageFeatures(img2, true);

        ArrayList<Integer> features3;
        if (Utils.fileExists(img3))
            features3 = Serializer.readArrayFromDisk(img3 + Conf.FEATURE_EXTENSION);
        else
            features3 = computeImageFeatures(img3, true);

        ArrayList<Integer> features4;
        if (Utils.fileExists(img4))
            features4 = Serializer.readArrayFromDisk(img4 + Conf.FEATURE_EXTENSION);
        else
            features4 = computeImageFeatures(img4, true);


        ArrayList<ArrayList<Integer>> all = new ArrayList<>();
        all.add(features1);
        all.add(features2);
        all.add(features3);
        all.add(features4);

        if (Utils.fileExists(ORGANIZED_SAMPLE))
            Utils.deleteFile(ORGANIZED_SAMPLE);
        if (Utils.fileExists(ORGANIZED_FEATURES))
            Utils.deleteFile(ORGANIZED_FEATURES);

        organizeFeatures(features1.size(), files, ORGANIZED_FEATURES, ORGANIZED_SAMPLE, forceDisk);

        for (int i = 0; i < features1.size(); i++) {
            for (int j = 0; j < 4; j++) {
                int index = getExampleIndex(i, j, 4, ORGANIZED_SAMPLE);

                Assert.assertTrue(index < 4);
                Assert.assertTrue(index >= 0);

                int feat = getExampleFeature(i, j, 4, ORGANIZED_FEATURES);

                assertEquals(Long.valueOf(all.get(index).get(i)), Long.valueOf(feat));

            }
        }
    }
}
