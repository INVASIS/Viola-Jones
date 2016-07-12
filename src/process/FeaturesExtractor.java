package process;

import java.awt.image.BufferedImage;

import static java.awt.image.BufferedImage.TYPE_USHORT_GRAY;

/**
 * Created by Dubrzr on 11/07/2016.
 */
public class FeaturesExtractor {
    public static BufferedImage summedAreaTable(BufferedImage image) {
        // The image must be greyscale (8bit)
        BufferedImage result = new BufferedImage(image.getWidth(), image.getHeight(), TYPE_USHORT_GRAY);
        result.setRGB(0, 0, image.getRGB(0, 0));

        for (int x = 1; x < image.getWidth(); x++) {
            int i = image.getRGB(x, 0) + result.getRGB(x-1, 0);
            result.setRGB(x, 0, i);
        }

        for (int y = 1; y < image.getHeight(); y++) {
            int i = image.getRGB(0, y) + result.getRGB(0, y-1);
            result.setRGB(0, y, i);
        }

        for (int x = 1; x < image.getWidth(); x++) {
            for (int y = 1; y < image.getHeight(); y++) {
                int i = image.getRGB(x, y) + result.getRGB(x-1, y) + result.getRGB(x, y-1) - result.getRGB(x-1, y-1);
                result.setRGB(x, y, i);
            }
        }
        return result;
    }

    public static int rectangleMean(BufferedImage summedAeraTable, int x, int y, int width, int height) {
        int A = x > 0 && y > 0 ? summedAeraTable.getRGB(x-1, y-1) : 0;
        int B = x + width > 0 && y > 0 ? summedAeraTable.getRGB(x+width-1, y-1) : 0;
        int C = x > 0 && y + height > 0 ? summedAeraTable.getRGB(x-1, y+height-1) : 0;
        int D = x + width > 0 && y + height > 0 ? summedAeraTable.getRGB(x+width-1, y+height-1) : 0;
        return A + D - B - C;
    }

}
