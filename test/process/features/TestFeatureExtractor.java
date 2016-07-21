package process.features;

import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class TestFeatureExtractor {
    @Test
    public void featuresChecker() {
        int[][] tmp = {{0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}};
        FeatureExtractor fe = new FeatureExtractor();
        assertEquals(40, fe.getAllTypeA(tmp, 4, 4).size()); // 2*1 -> 40
        assertEquals(20, fe.getAllTypeB(tmp, 4, 4).size()); // 3*1 -> 20
        assertEquals(40, fe.getAllTypeC(tmp, 4, 4).size()); // 1*2 -> 40
        assertEquals(20, fe.getAllTypeD(tmp, 4, 4).size()); // 1*3 -> 20
        assertEquals(16, fe.getAllTypeE(tmp, 4, 4).size()); // 2*2 -> 16
    }
}
