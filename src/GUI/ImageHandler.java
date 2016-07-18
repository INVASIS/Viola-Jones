package GUI;

import process.Converters;
import process.Filters;
import process.IntegralImage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageHandler {

    private BufferedImage bufferedImage;
    private int width;
    private int height;
    private int[][] crGrayImage;// Centered & reduced gray image
    private int[][] integralImage;

    private void init() {
        this.crGrayImage = Filters.crGrayscale(this.bufferedImage);
        this.integralImage = IntegralImage.summedAreaTable(this.crGrayImage, this.width, this.height);
    }

    public ImageHandler(BufferedImage bufferedImage) {
        this.bufferedImage = bufferedImage;
        this.width = bufferedImage.getWidth();
        this.height = bufferedImage.getHeight();

        this.init();
    }

    public ImageHandler(String filePath) {
        try {
            BufferedImage bufferedImage = ImageIO.read(new File(filePath));

            this.bufferedImage = bufferedImage;
            this.width = bufferedImage.getWidth();
            this.height = bufferedImage.getHeight();

            this.init();

        } catch (IOException e) {
            System.err.println("ERROR ! Cannot open file : " + filePath);
            e.printStackTrace();
        }
    }

    public ImageHandler(int[][] grayImage, int width, int height) {
        this.width = width;
        this.height = height;

        this.crGrayImage = new int[width][height];

        for (int x = 0; x < width; x++)
            System.arraycopy(grayImage[x], 0, this.crGrayImage[x], 0, height);

        this.integralImage = IntegralImage.summedAreaTable(this.crGrayImage, this.width, this.height);
        this.bufferedImage = Converters.intArrayToBufferedImage(this.crGrayImage, this.width, this.height);
    }

    public BufferedImage getBufferedImage() {
        return this.bufferedImage;
    }

    public int[][] getGrayImage() {
        return this.crGrayImage;
    }

    public BufferedImage getGrayBufferedImage() {
        return Converters.intArrayToBufferedImage(this.crGrayImage, this.width, this.height);
    }

    public int[][] getIntegralImage() {
        return this.integralImage;
    }

    public int getWidth() {
        return this.width;
    }

    public int getHeight() {
        return this.height;
    }
}
