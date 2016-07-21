package utils;

import java.awt.*;
import java.awt.image.BufferedImage;

public class Converters {

    public static BufferedImage intArrayToBufferedImage(int[][] arrayImage, int width, int height) {

        BufferedImage res = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int med = arrayImage[i][j];
                if (med < 0)
                    med = 0;
                if (med > 255)
                    med = 255;
                Color color = new Color(med, med, med);
                res.setRGB(i, j, color.getRGB());
            }
        }

        return res;
    }

}
