package GUI;

import cuda.HaarExtractor;
import process.Conf;
import utils.Converters;
import process.Filters;
import process.IntegralImage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static javafx.application.Platform.exit;

public class ImageHandler {

    private BufferedImage bufferedImage;
    private int width;
    private int height;
    private int[][] crGrayImage;// Centered & reduced gray image
    private int[][] integralImage;
    private final UUID uid = UUID.randomUUID();
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

    public ArrayList<ArrayList<Integer>> getFeatures() {
        if (!Files.exists(Paths.get(filePath + Conf.FEATURE_EXTENSION)))
            return computeFeatures();
        else {
            ArrayList<ArrayList<Integer>> res = new ArrayList<>();
            try {
                BufferedReader br = new BufferedReader(new FileReader(filePath + Conf.FEATURE_EXTENSION));
                String line = br.readLine();
                while (line != null) {
                    ArrayList<Integer> values = new ArrayList<>();
                    for (String val : line.split(";")) {
                        values.add(Integer.parseInt(val));
                    }
                    res.add(values);
                    line = br.readLine();
                }
                br.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }

            return res;
        }
    }

    public ArrayList<ArrayList<Integer>> computeFeatures() {
        Conf.haarExtractor.updateImage(integralImage);
        Conf.haarExtractor.compute();

        ArrayList<ArrayList<Integer>> res = new ArrayList<>();


        try {
            PrintWriter writer = new PrintWriter(filePath + Conf.FEATURE_EXTENSION, "UTF-8");

            for (Integer i : Conf.haarExtractor.getFeaturesA())
                    writer.write(i + ";");
            writer.write(System.lineSeparator());

            for (Integer i : Conf.haarExtractor.getFeaturesB())
                writer.write(i + ";");
            writer.write(System.lineSeparator());

            for (Integer i : Conf.haarExtractor.getFeaturesC())
                writer.write(i + ";");
            writer.write(System.lineSeparator());

            for (Integer i : Conf.haarExtractor.getFeaturesD())
                writer.write(i + ";");
            writer.write(System.lineSeparator());

            for (Integer i : Conf.haarExtractor.getFeaturesE())
                writer.write(i + ";");

            writer.close();

        } catch (IOException ex) {
            System.err.println("Could not write feature values to " + Conf.TRAIN_FEATURES);
        }

        res.add(Conf.haarExtractor.getFeaturesA());
        res.add(Conf.haarExtractor.getFeaturesB());
        res.add(Conf.haarExtractor.getFeaturesC());
        res.add(Conf.haarExtractor.getFeaturesD());
        res.add(Conf.haarExtractor.getFeaturesE());

        return res;
    }

    public UUID getUid() {
        return uid;
    }
}
