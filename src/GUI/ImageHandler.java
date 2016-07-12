package GUI;

import process.Converters;
import process.FeaturesExtractor;
import process.Filters;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageHandler {

    private BufferedImage bufferedImage;
    private int width;
    private int height;
    private int[][] grayImage;
    private int[][] integralImage;

    public ImageHandler(BufferedImage bufferedImage) {
        this.bufferedImage = bufferedImage;
        this.width = bufferedImage.getWidth();
        this.height = bufferedImage.getHeight();

        this.grayImage = Filters.grayscale(this.bufferedImage);
        this.integralImage = FeaturesExtractor.summedAreaTable(this.grayImage, this.width, this.height);
    }

    public ImageHandler(String filePath) {
        try {
            BufferedImage bufferedImage = ImageIO.read(new File(filePath));

            this.bufferedImage = bufferedImage;
            this.width = bufferedImage.getWidth();
            this.height = bufferedImage.getHeight();

            this.grayImage = Filters.grayscale(this.bufferedImage);
            this.integralImage = FeaturesExtractor.summedAreaTable(this.grayImage, this.width, this.height);

        } catch (IOException e) {
            System.err.println("ERROR ! Cannot open file : " + filePath);
            e.printStackTrace();
        }
    }

    public ImageHandler(int[][] grayImage, int width, int height) {
        this.width = width;
        this.height = height;

        this.grayImage = new int[width][height];

        for (int x = 0; x < width; x++) {
            System.arraycopy(grayImage[x], 0, this.grayImage[x], 0, height);
        }

        this.integralImage = FeaturesExtractor.summedAreaTable(this.grayImage, this.width, this.height);

        this.bufferedImage = Converters.intArrayToBufferedImage(this.grayImage, this.width, this.height);

    }

    public BufferedImage getGrayBufferedImage() {
        return Converters.intArrayToBufferedImage(this.grayImage, this.width, this.height);
    }

    public BufferedImage getBufferedImage() {
        return bufferedImage;
    }

    public int[][] getGrayImage() {
        return grayImage;
    }

    public int[][] getIntegralImage() {
        return integralImage;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int getGrayValue(int x, int y) {
        return this.grayImage[x][y];
    }
}
