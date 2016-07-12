package process;

import java.awt.image.BufferedImage;

/**
 * Created by Dubrzr on 12/07/2016.
 */
public class Filters {
    public static int[][] greyscale(BufferedImage bi) {
        int[][] result = new int[bi.getWidth()][bi.getHeight()];

        for (int x = 0; x < bi.getWidth(); x++) {
            for (int y = 0; y < bi.getHeight(); y++) {
                int v = bi.getRGB(x, y);

                int r = (int) (((v>>16)&0xFF) * 0.299);
                int g = (int) (((v>>8)&0xFF) * 0.587);
                int b = (int) (((v>>0)&0xFF) * 0.114);

                result[x][y] = (r + g + b) / 3;
            }
        }
        return result;
    }
}
