package process;

import GUI.ImageHandler;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestFeaturesExtractor {

    private ImageHandler getBIExample() {
        int[][] tmp = {{5, 3, 5, 3},
                       {2, 6, 2, 6},
                       {5, 3, 5, 3},
                       {2, 6, 2, 6}};

        return new ImageHandler(tmp, 4, 4);
    }

    @Test
    public void summedAreaTableTest() {
        ImageHandler imageHandler = getBIExample();
        int[][] sat = imageHandler.getIntegralImage();
        assertEquals(5, sat[0][0]);
        assertEquals(8, sat[0][1]);
        assertEquals(13, sat[0][2]);
        assertEquals(16, sat[0][3]);
        assertEquals(7, sat[1][0]);
        assertEquals(16, sat[1][1]);
        assertEquals(23, sat[1][2]);
        assertEquals(32, sat[1][3]);
        assertEquals(12, sat[2][0]);
        assertEquals(24, sat[2][1]);
        assertEquals(36, sat[2][2]);
        assertEquals(48, sat[2][3]);
        assertEquals(14, sat[3][0]);
        assertEquals(32, sat[3][1]);
        assertEquals(46, sat[3][2]);
        assertEquals(64, sat[3][3]);
    }

    @Test
    public void rectangleMeanTest() {

        ImageHandler imageHandler = getBIExample();

        int res = FeaturesExtractor.rectangleMean(imageHandler.getIntegralImage(), 2, 2, 2, 2);
        assertEquals(16, res);
    }
}
