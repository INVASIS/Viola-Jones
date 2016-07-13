package process.features;

import java.util.ArrayList;

import static process.FeaturesExtractor.rectangleSum;


public class FeatureCompute {
    private final int width;
    private final int height;
    private final int[][] integralImage;

    private ArrayList<Feature> featuresTypeA = new ArrayList<>();
    private ArrayList<Feature> featuresTypeB = new ArrayList<>();
    private ArrayList<Feature> featuresTypeC = new ArrayList<>();
    private ArrayList<Feature> featuresTypeD = new ArrayList<>();
    private ArrayList<Feature> featuresTypeE = new ArrayList<>();

    public FeatureCompute(int[][] integralImage, int width, int height) {
        this.width = width;
        this.height = height;
        this.integralImage = integralImage;

        // Compute all
        this.computeTypeA();
        this.computeTypeB();
        this.computeTypeC();
        this.computeTypeD();
        this.computeTypeE();
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
            int w = r.getWidth() / 2; // TODO: is it correct? -> What if r.getWidth() is odd?
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
            int w = r.getWidth() / 3;
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
            int h = r.getHeight() / 2;
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
            int h = r.getHeight() / 3;
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

        for (Rectangle r: listFeaturePositions(width, height)) {
            int w = r.getWidth() / 2;
            int h = r.getHeight() / 2;
            int r1 = rectangleSum(this.integralImage, r.getX(), r.getY(), w, h);
            int r2 = rectangleSum(this.integralImage, r.getX() + w, r.getY(), w, h);
            int r3 = rectangleSum(this.integralImage, r.getX(), r.getY() + h, w, h);
            int r4 = rectangleSum(this.integralImage, r.getX() + w, r.getY() + h, w, h);

            this.featuresTypeE.add(new Feature(r, type, r1 - r2 - r3 + r4));
        }
    }

    private ArrayList<Rectangle> listFeaturePositions(int minWidth, int minHeight) {
        ArrayList<Rectangle> rectangles = new ArrayList<>();

        int maxX = this.width - minWidth;
        int maxY = this.height - minHeight;
        for (int x = 0; x <= maxX; x++) {
            for (int y = 0; y <= maxY; y++) {
                int maxWidth = this.width - x;
                for (int w = minWidth; w < maxWidth; w += minWidth) {
                    int maxHeight = this.height - y;
                    for (int h = minHeight; h < maxHeight; h += minHeight) {
                        rectangles.add(new Rectangle(x, y, w, h));
                    }
                }
            }
        }

        return rectangles;
    }
}
