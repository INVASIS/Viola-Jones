package GUI;

import process.Conf;
import utils.Converters;
import process.Filters;
import process.IntegralImage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static javafx.application.Platform.exit;
import static process.features.FeatureExtractor.computeImageFeatures;
import static utils.Serializer.readArrayFromDisk;
import static utils.Utils.fileExists;

public class ImageHandler {

    private BufferedImage bufferedImage;
    private int width;
    private int height;
    private int[][] crGrayImage;// Centered & reduced gray image
    private int[][] integralImage;
    private final String filePath;

    private void init() {
        this.crGrayImage = Filters.crGrayscale(this.bufferedImage);
        this.integralImage = IntegralImage.summedAreaTable(this.crGrayImage, this.width, this.height);
    }

    public ImageHandler(BufferedImage bufferedImage) {
        this.bufferedImage = bufferedImage;
        this.width = bufferedImage.getWidth();
        this.height = bufferedImage.getHeight();

        this.filePath = null;

        this.init();
    }

    public ImageHandler(String filePath) {
        BufferedImage bufferedImage = null;
        try {
            bufferedImage = ImageIO.read(new File(filePath));
        } catch (IOException e) {
            e.printStackTrace();
        }

        assert bufferedImage != null;

        this.bufferedImage = bufferedImage;
        this.width = bufferedImage.getWidth();
        this.height = bufferedImage.getHeight();

        this.filePath = filePath;

        this.init();
    }

    public ImageHandler(int[][] grayImage, int width, int height) {
        this.width = width;
        this.height = height;

        this.filePath = null;

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

    public String getFilePath() {
        if (this.filePath == null) {
            System.err.println("Requesting filePath of an ImageHandler which has not been initialized from an image file.");
            exit();
        }
        return filePath;
    }

    public ArrayList<Integer> getFeatures() {
        String haarFilePath = filePath + Conf.FEATURE_EXTENSION;

        if (fileExists(haarFilePath))
            return readArrayFromDisk(haarFilePath);
        else
            return computeImageFeatures(filePath, true);
    }
}
