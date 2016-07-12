package process;

import org.junit.Test;

import java.awt.image.BufferedImage;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Dubrzr on 12/07/2016.
 */
public class TestFeaturesExtractor {
    private BufferedImage getBIExample() {
        BufferedImage bi = new BufferedImage(4, 4, BufferedImage.TYPE_BYTE_GRAY);
        bi.setRGB(0, 0, 5);
        bi.setRGB(0, 1, 3);
        bi.setRGB(0, 2, 5);
        bi.setRGB(0, 3, 3);
        bi.setRGB(1, 0, 2);
        bi.setRGB(1, 1, 6);
        bi.setRGB(1, 2, 2);
        bi.setRGB(1, 3, 6);
        bi.setRGB(2, 0, 5);
        bi.setRGB(2, 1, 3);
        bi.setRGB(2, 2, 5);
        bi.setRGB(2, 3, 3);
        bi.setRGB(3, 0, 2);
        bi.setRGB(3, 1, 6);
        bi.setRGB(3, 2, 2);
        bi.setRGB(3, 3, 6);
        return bi;
    }

    @Test
    public void summedAreaTableTest() {
        BufferedImage bi = getBIExample();
        BufferedImage sat = FeaturesExtractor.summedAreaTable(bi);
        assertEquals(5, sat.getRGB(0, 0));
        assertEquals(8, sat.getRGB(0, 1));
        assertEquals(13, sat.getRGB(0, 2));
        assertEquals(16, sat.getRGB(0, 3));
        assertEquals(7, sat.getRGB(1, 0));
        assertEquals(16, sat.getRGB(1, 1));
        assertEquals(23, sat.getRGB(1, 2));
        assertEquals(32, sat.getRGB(1, 3));
        assertEquals(12, sat.getRGB(2, 0));
        assertEquals(24, sat.getRGB(2, 1));
        assertEquals(36, sat.getRGB(2, 2));
        assertEquals(48, sat.getRGB(2, 3));
        assertEquals(14, sat.getRGB(3, 0));
        assertEquals(32, sat.getRGB(3, 1));
        assertEquals(46, sat.getRGB(3, 2));
        assertEquals(64, sat.getRGB(3, 3));
        assertTrue(false);
    }

    @Test
    public void rectangleMeanTest() {
        BufferedImage bi = new BufferedImage(4, 4, BufferedImage.TYPE_BYTE_GRAY);
        int res = FeaturesExtractor.rectangleMean(bi, 2, 2, 2, 2);
        assertEquals(16, res);
    }
}
