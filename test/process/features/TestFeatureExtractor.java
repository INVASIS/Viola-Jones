package process.features;

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
        FeatureExtractor fe = new FeatureExtractor();
        assertEquals(40, fe.listAllTypeA(tmp, 4, 4).size()); // 2*1 -> 40
        assertEquals(20, fe.listAllTypeB(tmp, 4, 4).size()); // 3*1 -> 20
        assertEquals(40, fe.listAllTypeC(tmp, 4, 4).size()); // 1*2 -> 40
        assertEquals(20, fe.listAllTypeD(tmp, 4, 4).size()); // 1*3 -> 20
        assertEquals(16, fe.listAllTypeE(tmp, 4, 4).size()); // 2*2 -> 16

        ArrayList<Feature> tmp_lf = new ArrayList<>();

        fe.streamAllTypeA(tmp, 4, 4).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(40, tmp_lf.size()); // 2*1 -> 40

        tmp_lf.clear();

        fe.streamAllTypeB(tmp, 4, 4).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(20, fe.listAllTypeB(tmp, 4, 4).size()); // 3*1 -> 20

        tmp_lf.clear();

        fe.streamAllTypeC(tmp, 4, 4).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(40, fe.listAllTypeC(tmp, 4, 4).size()); // 1*2 -> 40

        tmp_lf.clear();

        fe.streamAllTypeD(tmp, 4, 4).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(20, fe.listAllTypeD(tmp, 4, 4).size()); // 1*3 -> 20

        tmp_lf.clear();

        fe.streamAllTypeE(tmp, 4, 4).iterator().forEachRemaining(tmp_lf::add);
        assertEquals(16, fe.listAllTypeE(tmp, 4, 4).size()); // 2*2 -> 16
    }
}
