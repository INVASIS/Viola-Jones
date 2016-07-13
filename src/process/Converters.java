package process;

import GUI.ImageHandler;

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

    public static double[][] centerReduceImage(ImageHandler imageHandler) {

        double sum = FeaturesExtractor.rectangleSum(imageHandler.getIntegralImage(), 0, 0, imageHandler.getWidth(), imageHandler.getHeight());
        double nb_pixels = imageHandler.getWidth() * imageHandler.getHeight();
        double mean =  sum / nb_pixels;
        System.out.println("sum is : " + sum + " nb_pixels : " + nb_pixels + " Mean is : " + mean);

        double variance = 0;

        for (int x = 0; x < imageHandler.getWidth(); x++) {
            for (int y = 0; y < imageHandler.getHeight(); y++) {
                variance += (imageHandler.getGrayImage()[x][y] - mean) * (imageHandler.getGrayImage()[x][y] - mean) / nb_pixels;
            }
        }

        if (variance <= 1) {
            System.out.println("FAIL, variance should be more than 1, image is not good to be used for Viola-Jones");
            System.out.println("Variance is : " + variance);
        } else {
            System.out.println("Variance is correct : " + variance);
        }

        double[][] res = new double[imageHandler.getWidth()][imageHandler.getHeight()];

        for (int x = 0; x < imageHandler.getWidth(); x++) {
            for (int y = 0; y < imageHandler.getHeight(); y++) {
                res[x][y] = (imageHandler.getGrayImage()[x][y] - mean) / variance;
            }
        }

        return res;
    }

}
