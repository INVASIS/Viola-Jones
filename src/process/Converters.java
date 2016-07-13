package process;

import java.awt.*;
import java.awt.image.BufferedImage;

public class Converters {

    public static BufferedImage intArrayToBufferedImage(int[][] arrayImage, int width, int height) {

        BufferedImage res = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int med = arrayImage[i][j];
                Color color = new Color(med, med, med);
                res.setRGB(i, j, color.getRGB());
            }
        }

        return res;
    }

}
