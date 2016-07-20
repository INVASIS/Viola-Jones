package process.features;

import GUI.ImageHandler;
import process.Conf;
import utils.yield.Yielderable;

import java.util.ArrayList;
import java.util.stream.Collectors;

import static process.IntegralImage.rectangleSum;


public class FeatureExtractor {
    private int frameWidth;
    private int frameHeight;
    private int[][] integralImage;

    public FeatureExtractor(int[][] integralImage, int width, int height) {
        init(integralImage, width, height);
    }

    public FeatureExtractor(ImageHandler ih) {
        init(ih.getIntegralImage(), ih.getWidth(), ih.getHeight());
    }

    private void init(int[][] integralImage, int width, int height) {
        this.frameWidth = width;
        this.frameHeight = height;
        this.integralImage = integralImage;
    }

    private final int widthTypeA = 2;
    private final int heightTypeA = 1;
    public Feature computeTypeA(Rectangle r) {
        final int type = 1;
        /**
         * a ------- b ------- c
         * -         -         -
         * -   R1    -    R2   -
         * -         -         -
         * d ------- e ------- f
         */

        int w = r.getWidth() / widthTypeA;
        int h = r.getHeight();
        int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
        int r2 = rectangleSum(this.integralImage, r.getX() + w, r.getY(), w, h);

        return new Feature(r, type, r1 - r2);
    }

    public ArrayList<Feature> computeAllTypeA() {
        return listFeaturePositions(widthTypeA, heightTypeA).stream().map(this::computeTypeA).collect(Collectors.toCollection(ArrayList::new));
    }

    private final int widthTypeB = 3;
    private final int heightTypeB = 1;
    public Feature computeTypeB(Rectangle r) {
        final int type = 2;
        /**
         * a ------- b ------- c ------- d
         * -         -         -         -
         * -   R1    -    R2   -    R3   -
         * -         -         -         -
         * e ------- f ------- g ------- h
         */

        int w = r.getWidth() / widthTypeA;
        int h = r.getHeight();
        int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
        int r2 = rectangleSum(this.integralImage, r.getX() + w, r.getY(), w, h);
        int r3 = rectangleSum(this.integralImage, r.getX() + w + w, r.getY(), w, h);

        return new Feature(r, type, r1 - r2 + r3);
    }

    public ArrayList<Feature> computeAllTypeB() {
        return listFeaturePositions(widthTypeB, heightTypeB).stream().map(this::computeTypeA).collect(Collectors.toCollection(ArrayList::new));
    }

    private final int widthTypeC = 1;
    private final int heightTypeC = 2;
    public Feature computeTypeC(Rectangle r) {
        final int type = 3;
        /**
         * a ------- b
         * -         -
         * -   R1    -
         * -         -
         * c ------- d
         * -         -
         * -   R2    -
         * -         -
         * e ------- f
         */

        int w = r.getWidth();
        int h = r.getHeight() / heightTypeC;
        int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
        int r2 = rectangleSum(this.integralImage, r.getX(), r.getY() + h, w, h);

        return new Feature(r, type, r2 - r1);
    }

    public ArrayList<Feature> computeAllTypeC() {
        return listFeaturePositions(widthTypeC, heightTypeC).stream().map(this::computeTypeA).collect(Collectors.toCollection(ArrayList::new));
    }

    private final int widthTypeD = 1;
    private final int heightTypeD = 3;
    public Feature computeTypeD(Rectangle r) {
        final int type = 4;
        /**
         * a ------- b
         * -         -
         * -   R1    -
         * -         -
         * c ------- d
         * -         -
         * -   R2    -
         * -         -
         * e ------- f
         * -         -
         * -   R3    -
         * -         -
         * g ------- h
         */

        int w = r.getWidth();
        int h = r.getHeight() / heightTypeD;
        int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
        int r2 = rectangleSum(this.integralImage, r.getX(), r.getY() + h, w, h);
        int r3 = rectangleSum(this.integralImage, r.getX(), r.getY() + h + h, w, h);

        return new Feature(r, type, r1 - r2 + r3);
    }

    public ArrayList<Feature> computeAllTypeD() {
        return listFeaturePositions(widthTypeD, heightTypeD).stream().map(this::computeTypeA).collect(Collectors.toCollection(ArrayList::new));
    }

    private final int widthTypeE = 2;
    private final int heightTypeE = 2;
    public Feature computeTypeE(Rectangle r) {
        final int type = 5;
        /**
         * a ------- b ------- c
         * -         -         -
         * -   R1    -    R2   -
         * -         -         -
         * d ------- e ------- f
         * -         -         -
         * -   R3    -    R4   -
         * -         -         -
         * g ------- h ------- i
         */

        int w = r.getWidth() / widthTypeE;
        int h = r.getHeight() / heightTypeE;
        int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
        int r2 = rectangleSum(this.integralImage, r.getX() + w, r.getY(), w, h);
        int r3 = rectangleSum(this.integralImage, r.getX(), r.getY() + h, w, h);
        int r4 = rectangleSum(this.integralImage, r.getX() + w, r.getY() + h, w, h);

        return new Feature(r, type, r1 - r2 - r3 + r4);
    }

    public ArrayList<Feature> computeAllTypeE() {
        return listFeaturePositions(widthTypeE, heightTypeE).stream().map(this::computeTypeA).collect(Collectors.toCollection(ArrayList::new));
    }

    public Yielderable<Rectangle> streamFeaturePositions(int sizeX, int sizeY) {
        return yield -> {
            for (int w = sizeX; w <= this.frameWidth; w += sizeX) {
                for (int h = sizeY; h <= this.frameHeight; h += sizeY) {
                    for (int x = 0; x <= this.frameWidth - w; x++) {
                        for (int y = 0; y <= this.frameHeight - h; y++) {
                            yield.returning(new Rectangle(x, y, w, h));
                        }
                    }
                }
            }
        };
    }

    private ArrayList<Rectangle> listFeaturePositions(int sizeX, int sizeY) {
        ArrayList<Rectangle> rectangles = new ArrayList<>();
        for (Rectangle r : streamFeaturePositions(sizeX, sizeY))
            rectangles.add(r);
        return rectangles;
    }
}
