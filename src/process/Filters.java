package process;

import java.awt.*;
import java.awt.image.BufferedImage;

public class Filters {
    public static int[][] grayscale(BufferedImage bi) {
        int[][] result = new int[bi.getWidth()][bi.getHeight()];

        for (int x = 0; x < bi.getWidth(); x++) {
            for (int y = 0; y < bi.getHeight(); y++) {

                Color c = new Color(bi.getRGB(x, y));
                int med = (int) (c.getRed() * 0.299 + c.getBlue() * 0.587 + c.getGreen() * 0.114);
                result[x][y] = med;

            }
        }
        return result;
    }
}
