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
        FeatureExtractor fe = new FeatureExtractor(tmp, 4, 4);
        assertEquals(40, fe.getFeaturesTypeA().size()); // 2*1 -> 40
        assertEquals(20, fe.getFeaturesTypeB().size()); // 3*1 -> 20
        assertEquals(40, fe.getFeaturesTypeC().size()); // 1*2 -> 40
        assertEquals(20, fe.getFeaturesTypeD().size()); // 1*3 -> 20
        assertEquals(16, fe.getFeaturesTypeE().size()); // 2*2 -> 16
    }
}
