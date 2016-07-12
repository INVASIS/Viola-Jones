package GUI;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageHandler {

    private BufferedImage bufferedImage;
    private int width;
    private int height;
    private int[][] pixels;

    public ImageHandler(BufferedImage bufferedImage) {
        this.bufferedImage = bufferedImage;
        this.width = bufferedImage.getWidth();
        this.height = bufferedImage.getHeight();

        this.pixels = new int[this.width][this.height];

        setPixelsFromBufferedImage();
    }

    public ImageHandler(String filePath) {
        try {
            BufferedImage bufferedImage = ImageIO.read(new File(filePath));

            this.bufferedImage = bufferedImage;
            this.width = bufferedImage.getWidth();
            this.height = bufferedImage.getHeight();

            this.pixels = new int[this.width][this.height];

            setPixelsFromBufferedImage();

        } catch (IOException e) {
            System.err.println("ERROR ! Cannot open file : " + filePath);
            e.printStackTrace();
        }
    }

    public BufferedImage getBufferedImage() {
        return bufferedImage;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int[][] getPixels() {
        return pixels;
    }

    public void setBufferedImageFromPixels() {
        for (int x = 0; x < this.width; x++) {
            for (int y = 0; y < this.height; y++) {
                int med = pixels[x][y];
                Color c = new Color(med, med, med);
                this.bufferedImage.setRGB(x, y, c.getRGB());
            }
        }
    }

    public void setPixelsFromBufferedImage() {
        for (int x = 0; x < this.width; x++) {
            for (int y = 0; y < this.height; y++) {
                Color c = new Color(this.bufferedImage.getRGB(x, y));
                int med = (c.getBlue() + c.getGreen() + c.getRed()) / 3; // Should already be grayscale, but just in case...
                this.pixels[x][y] = med;
            }
        }
    }
}
