package process.features;

import GUI.ImageHandler;
import process.Conf;
import utils.yield.Yielderable;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.stream.Collectors;

import static process.IntegralImage.rectangleSum;
import static utils.Serializer.writeArrayToDisk;
import static utils.Utils.fileExists;
import static utils.Utils.streamFiles;


public class FeatureExtractor {

    public static final int typeA = 1;
    public static final int widthTypeA = 2;
    public static final int heightTypeA = 1;

    public static int computeTypeA(ImageHandler image, Rectangle r) {
        /**
         * a ------- b ------- c
         * -         -         -
         * -   R1    -    R2   -
         * -         -         -
         * d ------- e ------- f
         */

        int w = r.getWidth() / widthTypeA;
        int h = r.getHeight();
        int r1 = rectangleSum(image, r.getX(), r.getY(), w, h);
        int r2 = rectangleSum(image, r.getX() + w, r.getY(), w, h);

        return r1 - r2;
    }

    public static Yielderable<Feature> streamAllTypeA(ImageHandler image) {
        return yield -> {
            for (Rectangle r : listFeaturePositions(widthTypeA, heightTypeA, image.getWidth(), image.getHeight()))
                yield.returning(new Feature(r, typeA, image));
        };
    }

    public static ArrayList<Feature> listAllTypeA(ImageHandler image) {
        return listFeaturePositions(widthTypeA, heightTypeA, image.getWidth(), image.getHeight())
                .stream()
                .map(r -> new Feature(r, typeA, image))
                .collect(Collectors.toCollection(ArrayList::new));
    }

    public static final int typeB = 2;
    public static final int widthTypeB = 3;
    public static final int heightTypeB = 1;

    public static int computeTypeB(ImageHandler image, Rectangle r) {
        /**
         * a ------- b ------- c ------- d
         * -         -         -         -
         * -   R1    -    R2   -    R3   -
         * -         -         -         -
         * e ------- f ------- g ------- h
         */

        int w = r.getWidth() / widthTypeB;
        int h = r.getHeight();
        int r1 = rectangleSum(image, r.getX(), r.getY(), w, h);
        int r2 = rectangleSum(image, r.getX() + w, r.getY(), w, h);
        int r3 = rectangleSum(image, r.getX() + w + w, r.getY(), w, h);

        return r1 - r2 + r3;
    }

    public static Yielderable<Feature> streamAllTypeB(ImageHandler image) {
        return yield -> {
            for (Rectangle r : listFeaturePositions(widthTypeB, heightTypeB, image.getWidth(), image.getHeight()))
                yield.returning(new Feature(r, typeB, image));
        };
    }

    public static ArrayList<Feature> listAllTypeB(ImageHandler image) {
        return listFeaturePositions(widthTypeB, heightTypeB, image.getWidth(), image.getHeight())
                .stream()
                .map(r -> new Feature(r, typeB, image))
                .collect(Collectors.toCollection(ArrayList::new));
    }

    public static final int typeC = 3;
    public static final int widthTypeC = 1;
    public static final int heightTypeC = 2;

    public static int computeTypeC(ImageHandler image, Rectangle r) {
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
        int r1 = rectangleSum(image, r.getX(), r.getY(), w, h);
        int r2 = rectangleSum(image, r.getX(), r.getY() + h, w, h);

        return r2 - r1;
    }

    public static Yielderable<Feature> streamAllTypeC(ImageHandler image) {
        return yield -> {
            for (Rectangle r : listFeaturePositions(widthTypeC, heightTypeC, image.getWidth(), image.getHeight()))
                yield.returning(new Feature(r, typeC, image));
        };
    }

    public static ArrayList<Feature> listAllTypeC(ImageHandler image) {
        return listFeaturePositions(widthTypeC, heightTypeC, image.getWidth(), image.getHeight())
                .stream()
                .map(r -> new Feature(r, typeC, image))
                .collect(Collectors.toCollection(ArrayList::new));
    }

    public static final int typeD = 4;
    public static final int widthTypeD = 1;
    public static final int heightTypeD = 3;

    public static int computeTypeD(ImageHandler image, Rectangle r) {
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
        int r1 = rectangleSum(image, r.getX(), r.getY(), w, h);
        int r2 = rectangleSum(image, r.getX(), r.getY() + h, w, h);
        int r3 = rectangleSum(image, r.getX(), r.getY() + h + h, w, h);

        return r1 - r2 + r3;
    }

    public static Yielderable<Feature> streamAllTypeD(ImageHandler image) {
        return yield -> {
            for (Rectangle r : listFeaturePositions(widthTypeD, heightTypeD, image.getWidth(), image.getHeight()))
                yield.returning(new Feature(r, typeD, image));
        };
    }

    public static ArrayList<Feature> listAllTypeD(ImageHandler image) {
        return listFeaturePositions(widthTypeD, heightTypeD, image.getWidth(), image.getHeight())
                .stream()
                .map(r -> new Feature(r, typeD, image))
                .collect(Collectors.toCollection(ArrayList::new));
    }

    public static final int typeE = 5;
    public static final int widthTypeE = 2;
    public static final int heightTypeE = 2;

    public static int computeTypeE(ImageHandler image, Rectangle r) {
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
        int r1 = rectangleSum(image, r.getX(), r.getY(), w, h);
        int r2 = rectangleSum(image, r.getX() + w, r.getY(), w, h);
        int r3 = rectangleSum(image, r.getX(), r.getY() + h, w, h);
        int r4 = rectangleSum(image, r.getX() + w, r.getY() + h, w, h);

        return r1 - r2 - r3 + r4;
    }

    public static Yielderable<Feature> streamAllTypeE(ImageHandler image) {
        return yield -> {
            for (Rectangle r : listFeaturePositions(widthTypeE, heightTypeE, image.getWidth(), image.getHeight()))
                yield.returning(new Feature(r, typeE, image));
        };
    }

    public static ArrayList<Feature> listAllTypeE(ImageHandler image) {
        return listFeaturePositions(widthTypeE, heightTypeE, image.getWidth(), image.getHeight())
                .stream()
                .map(r -> new Feature(r, typeE, image))
                .collect(Collectors.toCollection(ArrayList::new));
    }

    public static Yielderable<Rectangle> streamFeaturePositions(int featureWidth, int featureHeight, int frameWidth, int frameHeight) {
        return yield -> {
            for (int w = featureWidth; w <= frameWidth; w += featureWidth) {
                for (int h = featureHeight; h <= frameHeight; h += featureHeight) {
                    for (int x = 0; x <= frameWidth - w; x++) {
                        for (int y = 0; y <= frameHeight - h; y++) {
                            yield.returning(new Rectangle(x, y, w, h));
                        }
                    }
                }
            }
        };
    }

    public static ArrayList<Rectangle> listFeaturePositions(int featureWidth, int featureHeight, int frameWidth, int frameHeight) {
        ArrayList<Rectangle> rectangles = new ArrayList<>();
        for (Rectangle r : streamFeaturePositions(featureWidth, featureHeight, frameWidth, frameHeight))
            rectangles.add(r);
        return rectangles;
    }

    public static Yielderable<ArrayList<Feature>> streamFeaturesByType(ImageHandler image) {
        return yield -> {
            yield.returning(listAllTypeA(image));
            yield.returning(listAllTypeB(image));
            yield.returning(listAllTypeC(image));
            yield.returning(listAllTypeD(image));
            yield.returning(listAllTypeE(image));
        };
    }

    public static long countFeatures(int featureWidth, int featureHeight, int frameWidth, int frameHeight) {
        // TODO: Use CUDA? (It could be very long on large frames)
        long count = 0;
        for (int w = featureWidth; w <= frameWidth; w += featureWidth)
            for (int h = featureHeight; h <= frameHeight; h += featureHeight)
                for (int x = 0; x <= frameWidth - w; x++)
                    for (int y = 0; y <= frameHeight - h; y++)
                        count++;
        return count;
    }

    public static long countAllFeatures(int width, int height) {
        long count = 0;

        long typeA = countFeatures(widthTypeA, heightTypeA, width, height);
        long typeB = countFeatures(widthTypeB, heightTypeB, width, height);
        long typeC = countFeatures(widthTypeC, heightTypeC, width, height);
        long typeD = countFeatures(widthTypeD, heightTypeD, width, height);
        long typeE = countFeatures(widthTypeE, heightTypeE, width, height);

        count += typeA;
        count += typeB;
        count += typeC;
        count += typeD;
        count += typeE;

        return count;
    }

    // Warning: Need to train and evaluate on the same features : only on GPU or only on CPU
    public static ArrayList<Integer> computeImageFeatures(String imagePath, boolean writeToDisk) {
        ImageHandler image = new ImageHandler(imagePath);

        ArrayList<Integer> result = new ArrayList<>();
        if (Conf.USE_CUDA) {
            Conf.haarExtractor.updateImage(image.getIntegralImage());
            Conf.haarExtractor.compute();
            result.addAll(Conf.haarExtractor.getFeaturesA());
            result.addAll(Conf.haarExtractor.getFeaturesB());
            result.addAll(Conf.haarExtractor.getFeaturesC());
            result.addAll(Conf.haarExtractor.getFeaturesD());
            result.addAll(Conf.haarExtractor.getFeaturesE());
        }
        else
            for (ArrayList<Feature> features : FeatureExtractor.streamFeaturesByType(image))
                for (Feature f : features)
                    result.add(f.getValue());

        if (writeToDisk)
            writeArrayToDisk(imagePath + Conf.FEATURE_EXTENSION, result);

        return result;
    }

    private static int computeImagesFeatures(String dir, boolean writeToDisk) {
        /**
         * returns: the number of features computed
         */

        int count = 0;
        for (String imagePath : streamFiles(dir, Conf.IMAGES_EXTENSION)) {
            if (!fileExists(imagePath + Conf.FEATURE_EXTENSION))
            {
                computeImageFeatures(imagePath, writeToDisk);
                count++;
            }
        }
        return count;
    }

    public static int computeSetFeatures(String faces_dir, String nonfaces_dir, boolean writeToDisk) { // Set = faces + non-faces
        /**
         * returns: the number of features computed
         */

        int count = 0;
        count += computeImagesFeatures(faces_dir, writeToDisk);
        count += computeImagesFeatures(nonfaces_dir, writeToDisk);
        return count;
    }
}
