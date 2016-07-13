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

    public static int[][] crGrayscale(BufferedImage bi) {
        int[][] result = Filters.grayscale(bi);
        double nb_pixels = bi.getWidth() * bi.getHeight();

        // Center in 0
        for (int x = 0; x < bi.getWidth(); x++)
            for (int y = 0; y < bi.getHeight(); y++)
                result[x][y] = result[x][y] - 256/2;

        // Compute mean
        double sum = 0;
        for (int x = 0; x < bi.getWidth(); x++)
            for (int y = 0; y < bi.getHeight(); y++)
                sum += result[x][y];
        int mean = (int)(sum / nb_pixels);
        System.out.println("Mean: " + mean);

        // Compute variance
        double variance = 0;
        for (int x = 0; x < bi.getWidth(); x++)
            for (int y = 0; y < bi.getHeight(); y++)
                variance += Math.pow(result[x][y] - mean, 2);
        variance /= nb_pixels;
        double sd = Math.sqrt(variance);

        if (variance <= 1) {
            System.out.println("FAIL, variance should be more than 1, image is not good to be used for Viola-Jones");
            System.out.println("Variance is: " + variance);
        } else {
            System.out.println("Variance is correct: " + variance);
            System.out.println("Standard deviation is: " + sd);
        }

        for (int x = 0; x < bi.getWidth(); x++)
            for (int y = 0; y < bi.getHeight(); y++)
                result[x][y] = (int) (((result[x][y] - mean) / sd) + 128);

        return result;
    }
}