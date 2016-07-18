package process.features;

import GUI.ImageHandler;
import process.Conf;

import java.util.ArrayList;

import static process.IntegralImage.rectangleSum;


public class FeatureExtractor {
    private int frameWidth;
    private int frameHeight;
    private int[][] integralImage;

    private ArrayList<Feature> featuresTypeA = new ArrayList<>();
    private ArrayList<Feature> featuresTypeB = new ArrayList<>();
    private ArrayList<Feature> featuresTypeC = new ArrayList<>();
    private ArrayList<Feature> featuresTypeD = new ArrayList<>();
    private ArrayList<Feature> featuresTypeE = new ArrayList<>();

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

        // Compute all
        this.computeTypeA();
        System.out.println("Found " + this.featuresTypeA.size() + " features of type A.");
        this.computeTypeB();
        System.out.println("Found " + this.featuresTypeB.size() + " features of type B.");
        this.computeTypeC();
        System.out.println("Found " + this.featuresTypeC.size() + " features of type C.");
        this.computeTypeD();
        System.out.println("Found " + this.featuresTypeD.size() + " features of type D.");
        this.computeTypeE();
        System.out.println("Found " + this.featuresTypeE.size() + " features of type E.");
    }

    public ArrayList<Feature> getAllFeatures() {
        ArrayList<Feature> result = new ArrayList<>();
        result.addAll(this.featuresTypeA);
        result.addAll(this.featuresTypeB);
        result.addAll(this.featuresTypeC);
        result.addAll(this.featuresTypeD);
        result.addAll(this.featuresTypeE);
        return result;
    }

    private void computeTypeA() {
        final int type = 1;
        final int width = 2;
        final int height = 1;
        /**
         * a ------- b ------- c
         * -         -         -
         * -   R1    -    R2   -
         * -         -         -
         * d ------- e ------- f
         */

        for (Rectangle r: listFeaturePositions(width, height)) {
            int w = r.getWidth() / width;
            int h = r.getHeight();
            int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
            int r2 = rectangleSum(this.integralImage, r.getX() + w, r.getY(), w, h);

            this.featuresTypeA.add(new Feature(r, type, r1 - r2));
        }
    }

    private void computeTypeB() {
        final int type = 2;
        final int width = 3;
        final int height = 1;
        /**
         * a ------- b ------- c ------- d
         * -         -         -         -
         * -   R1    -    R2   -    R3   -
         * -         -         -         -
         * e ------- f ------- g ------- h
         */

        for (Rectangle r: listFeaturePositions(width, height)) {
            int w = r.getWidth() / width;
            int h = r.getHeight();
            int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
            int r2 = rectangleSum(this.integralImage, r.getX() + w, r.getY(), w, h);
            int r3 = rectangleSum(this.integralImage, r.getX() + w + w, r.getY(), w, h);

            this.featuresTypeB.add(new Feature(r, type, r1 - r2 + r3));
        }
    }

    private void computeTypeC() {
        final int type = 3;
        final int width = 1;
        final int height = 2;
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

        for (Rectangle r: listFeaturePositions(width, height)) {
            int w = r.getWidth();
            int h = r.getHeight() / height;
            int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
            int r2 = rectangleSum(this.integralImage, r.getX(), r.getY() + h, w, h);

            this.featuresTypeC.add(new Feature(r, type, r2 - r1));
        }
    }

    private void computeTypeD() {
        final int type = 4;
        final int width = 1;
        final int height = 3;
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

        for (Rectangle r: listFeaturePositions(width, height)) {
            int w = r.getWidth();
            int h = r.getHeight() / height;
            int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
            int r2 = rectangleSum(this.integralImage, r.getX(), r.getY() + h, w, h);
            int r3 = rectangleSum(this.integralImage, r.getX(), r.getY() + h + h, w, h);

            this.featuresTypeD.add(new Feature(r, type, r1 - r2 + r3));
        }
    }

    private void computeTypeE() {
        final int type = 5;
        final int width = 2;
        final int height = 2;
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
        int lol = 0;
        for (Rectangle r: listFeaturePositions(width, height)) {
            int w = r.getWidth() / width;
            int h = r.getHeight() / height;
            int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
            int r2 = rectangleSum(this.integralImage, r.getX() + w, r.getY(), w, h);
            int r3 = rectangleSum(this.integralImage, r.getX(), r.getY() + h, w, h);
            int r4 = rectangleSum(this.integralImage, r.getX() + w, r.getY() + h, w, h);
            lol += 1;
            this.featuresTypeE.add(new Feature(r, type, r1 - r2 - r3 + r4));
        }
    }

    private ArrayList<Rectangle> listFeaturePositions(int sizeX, int sizeY) {
        long startTime = System.currentTimeMillis();

        ArrayList<Rectangle> rectangles = new ArrayList<>();

//        if (Conf.USE_CUDA) {
//
//        }
//        else {
            for (int w = sizeX; w <= this.frameWidth; w += sizeX) {
                for (int h = sizeY; h <= this.frameHeight; h += sizeY) {
                    for (int x = 0; x <= this.frameWidth - w; x++) {
                        for (int y = 0; y <= this.frameHeight - h; y++) {
                            rectangles.add(new Rectangle(x, y, w, h));
                        }
                    }
                }
            }
//        }


        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("Computing rectangle positions: found " + rectangles.size() + " in " + estimatedTime + "ms");
        return rectangles;
    }

    public ArrayList<Feature> getFeaturesTypeA() {
        return featuresTypeA;
    }

    public ArrayList<Feature> getFeaturesTypeB() {
        return featuresTypeB;
    }

    public ArrayList<Feature> getFeaturesTypeC() {
        return featuresTypeC;
    }

    public ArrayList<Feature> getFeaturesTypeD() {
        return featuresTypeD;
    }

    public ArrayList<Feature> getFeaturesTypeE() {
        return featuresTypeE;
    }
}
