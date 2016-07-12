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
        assertEquals(sat.getRGB(0, 0), 5);
        assertEquals(sat.getRGB(0, 1), 8);
        assertEquals(sat.getRGB(0, 2), 13);
        assertEquals(sat.getRGB(0, 3), 16);
        assertEquals(sat.getRGB(1, 0), 7);
        assertEquals(sat.getRGB(1, 1), 16);
        assertEquals(sat.getRGB(1, 2), 23);
        assertEquals(sat.getRGB(1, 3), 32);
        assertEquals(sat.getRGB(2, 0), 12);
        assertEquals(sat.getRGB(2, 1), 24);
        assertEquals(sat.getRGB(2, 2), 36);
        assertEquals(sat.getRGB(2, 3), 48);
        assertEquals(sat.getRGB(3, 0), 14);
        assertEquals(sat.getRGB(3, 1), 32);
        assertEquals(sat.getRGB(3, 2), 46);
        assertEquals(sat.getRGB(3, 3), 64);
        assertTrue(false);
    }

    @Test
    public void rectangleMeanTest() {
        BufferedImage bi = new BufferedImage(4, 4, BufferedImage.TYPE_BYTE_GRAY);

    }
}
