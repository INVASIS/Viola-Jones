package process;

import GUI.ImageHandler;

public class FeaturesExtractor {

    public static ImageHandler summedAreaTable(ImageHandler image) {
        ImageHandler result = new ImageHandler(image.getBufferedImage());
        result.setPixels(image.getPixels());

        for (int x = 1; x < image.getWidth(); x++) {

            int i = result.getPixelValue(x, 0) + result.getPixelValue(x - 1, 0);
            result.setPixelValue(x, 0, i);
        }

        for (int y = 1; y < image.getHeight(); y++) {

            int i = result.getPixelValue(0, y) + result.getPixelValue(0, y - 1);
            result.setPixelValue(0, y, i);
        }

        for (int x = 1; x < image.getWidth(); x++) {
            for (int y = 1; y < image.getHeight(); y++) {
                int i = image.getPixelValue(x, y) + result.getPixelValue(x - 1, y) + result.getPixelValue(x, y - 1) - result.getPixelValue(x - 1, y - 1);
                result.setPixelValue(x, y, i);
            }
        }

        result.setBufferedImageFromPixels(); // A verif !
        return result;
    }

    public static int rectangleMean(ImageHandler summedAeraTable, int x, int y, int width, int height) {

        int A = x > 0 && y > 0 ? summedAeraTable.getPixelValue(x - 1, y - 1) : 0;
        int B = x + width > 0 && y > 0 ? summedAeraTable.getPixelValue(x + width - 1, y - 1) : 0;
        int C = x > 0 && y + height > 0 ? summedAeraTable.getPixelValue(x - 1, y + height - 1) : 0;
        int D = x + width > 0 && y + height > 0 ? summedAeraTable.getPixelValue(x + width - 1, y + height - 1) : 0;

        return A + D - B - C;
    }

}
