package process;

import GUI.ImageHandler;
import org.junit.Test;

import java.awt.image.BufferedImage;

import static org.junit.Assert.assertEquals;

public class TestFeaturesExtractor {

    private ImageHandler getBIExample() {
        BufferedImage bi = new BufferedImage(4, 4, BufferedImage.TYPE_BYTE_GRAY);
        ImageHandler imageHandler = new ImageHandler(bi);
        imageHandler.setPixelValue(0, 0, 5);
        imageHandler.setPixelValue(0, 1, 3);
        imageHandler.setPixelValue(0, 2, 5);
        imageHandler.setPixelValue(0, 3, 3);
        imageHandler.setPixelValue(1, 0, 2);
        imageHandler.setPixelValue(1, 1, 6);
        imageHandler.setPixelValue(1, 2, 2);
        imageHandler.setPixelValue(1, 3, 6);
        imageHandler.setPixelValue(2, 0, 5);
        imageHandler.setPixelValue(2, 1, 3);
        imageHandler.setPixelValue(2, 2, 5);
        imageHandler.setPixelValue(2, 3, 3);
        imageHandler.setPixelValue(3, 0, 2);
        imageHandler.setPixelValue(3, 1, 6);
        imageHandler.setPixelValue(3, 2, 2);
        imageHandler.setPixelValue(3, 3, 6);

        imageHandler.setBufferedImageFromPixels();
        return imageHandler;
    }

    @Test
    public void summedAreaTableTest() {
        ImageHandler imageHandler = getBIExample();
        ImageHandler sat = FeaturesExtractor.summedAreaTable(imageHandler);
        assertEquals(5, sat.getPixelValue(0, 0));
        assertEquals(8, sat.getPixelValue(0, 1));
        assertEquals(13, sat.getPixelValue(0, 2));
        assertEquals(16, sat.getPixelValue(0, 3));
        assertEquals(7, sat.getPixelValue(1, 0));
        assertEquals(16, sat.getPixelValue(1, 1));
        assertEquals(23, sat.getPixelValue(1, 2));
        assertEquals(32, sat.getPixelValue(1, 3));
        assertEquals(12, sat.getPixelValue(2, 0));
        assertEquals(24, sat.getPixelValue(2, 1));
        assertEquals(36, sat.getPixelValue(2, 2));
        assertEquals(48, sat.getPixelValue(2, 3));
        assertEquals(14, sat.getPixelValue(3, 0));
        assertEquals(32, sat.getPixelValue(3, 1));
        assertEquals(46, sat.getPixelValue(3, 2));
        assertEquals(64, sat.getPixelValue(3, 3));
    }

    @Test
    public void rectangleMeanTest() {

        ImageHandler imageHandler = getBIExample();

        int res = FeaturesExtractor.rectangleMean(FeaturesExtractor.summedAreaTable(imageHandler), 2, 2, 2, 2);
        assertEquals(16, res);
    }
}
