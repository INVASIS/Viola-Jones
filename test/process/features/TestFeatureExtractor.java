package process.features;

import GUI.ImageHandler;
import org.junit.Test;

import java.util.ArrayList;

import static junit.framework.TestCase.assertEquals;

public class TestFeatureExtractor {
    @Test
    public void featuresChecker() {
        int[][] tmp = {{0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}};
        ImageHandler image = new ImageHandler(tmp, 4, 4);

        assertEquals(40, FeatureExtractor.listAllTypeA(image, 4, 4).size()); // 2*1 -> 40
        assertEquals(20, FeatureExtractor.listAllTypeB(image, 4, 4).size()); // 3*1 -> 20
        assertEquals(40, FeatureExtractor.listAllTypeC(image, 4, 4).size()); // 1*2 -> 40
        assertEquals(20, FeatureExtractor.listAllTypeD(image, 4, 4).size()); // 1*3 -> 20
        assertEquals(16, FeatureExtractor.listAllTypeE(image, 4, 4).size()); // 2*2 -> 16

        ArrayList<Feature> tmp_lf = new ArrayList<>();

        FeatureExtractor.streamAllTypeA(image, 4, 4).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(40, tmp_lf.size()); // 2*1 -> 40

        tmp_lf.clear();

        FeatureExtractor.streamAllTypeB(image, 4, 4).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(20, tmp_lf.size()); // 3*1 -> 20

        tmp_lf.clear();

        FeatureExtractor.streamAllTypeC(image, 4, 4).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(40, tmp_lf.size()); // 1*2 -> 40

        tmp_lf.clear();

        FeatureExtractor.streamAllTypeD(image, 4, 4).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(20, tmp_lf.size()); // 1*3 -> 20

        tmp_lf.clear();

        FeatureExtractor.streamAllTypeE(image, 4, 4).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(16, tmp_lf.size()); // 2*2 -> 16
    }
}
