package process.features;

import GUI.ImageHandler;
import cuda.HaarDetector;
import javafx.util.Pair;
import process.Conf;
import utils.Serializer;
import utils.yield.Yielderable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.stream.Collectors;

import static process.IntegralImage.rectangleSum;
import static utils.Serializer.*;
import static utils.Utils.*;


public class FeatureExtractor {
    public static final int typeA = 1;
    public static final int widthTypeA = 2;
    public static final int heightTypeA = 1;

    /**
     * a ------- b ------- c
     * -         -         -
     * -   R1    -    R2   -
     * -         -         -
     * d ------- e ------- f
     */
    public static int computeTypeA(ImageHandler image, Rectangle r) {

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

    /**
     * a ------- b ------- c ------- d
     * -         -         -         -
     * -   R1    -    R2   -    R3   -
     * -         -         -         -
     * e ------- f ------- g ------- h
     */
    public static int computeTypeB(ImageHandler image, Rectangle r) {

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
    public static int computeTypeC(ImageHandler image, Rectangle r) {

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
    public static int computeTypeD(ImageHandler image, Rectangle r) {

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
    public static int computeTypeE(ImageHandler image, Rectangle r) {

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
    public static int[] computeImageFeatures(String imagePath, boolean writeToDisk) {
        ImageHandler image = new ImageHandler(imagePath);

        return computeImageFeatures(image, writeToDisk, imagePath);
    }

    public static int[] computeImageFeatures(ImageHandler image, boolean writeToDisk, String imagePath) {

        int[] result = new int[(int) Serializer.featureCount];
        if (Conf.USE_CUDA) {
            Conf.haarExtractor.updateImage(image.getIntegralImage());
            Conf.haarExtractor.compute();
            int offset = 0;
            System.arraycopy(Conf.haarExtractor.getFeaturesA(), 0, result, offset, (int) Conf.haarExtractor.getNUM_FEATURES_A());
            offset += (int) Conf.haarExtractor.getNUM_FEATURES_A();
            System.arraycopy(Conf.haarExtractor.getFeaturesB(), 0, result, offset, (int) Conf.haarExtractor.getNUM_FEATURES_B());
            offset += (int) Conf.haarExtractor.getNUM_FEATURES_B();
            System.arraycopy(Conf.haarExtractor.getFeaturesC(), 0, result, offset, (int) Conf.haarExtractor.getNUM_FEATURES_C());
            offset += (int) Conf.haarExtractor.getNUM_FEATURES_C();
            System.arraycopy(Conf.haarExtractor.getFeaturesD(), 0, result, offset, (int) Conf.haarExtractor.getNUM_FEATURES_D());
            offset += (int) Conf.haarExtractor.getNUM_FEATURES_D();
            System.arraycopy(Conf.haarExtractor.getFeaturesE(), 0, result, offset, (int) Conf.haarExtractor.getNUM_FEATURES_E());
        } else {
            int cpt = 0;
            for (ArrayList<Feature> features : FeatureExtractor.streamFeaturesByType(image))
                for (Feature f : features)
                    result[cpt++] = f.getValue();
        }
        if (writeToDisk)
            writeArrayToDisk(imagePath + Conf.FEATURE_EXTENSION, result, Serializer.featureCount);

        return result;
    }

    /**
     * returns: the number of features computed
     */
    private static int computeImagesFeatures(String dir, boolean writeToDisk) {
        int count = 0;
        for (String imagePath : streamFiles(dir, Conf.IMAGES_EXTENSION)) {
            if (!fileExists(imagePath + Conf.FEATURE_EXTENSION)) {
                computeImageFeatures(imagePath, writeToDisk);
                count++;
            }
        }
        return count;
    }

    /**
     * returns: the number of features computed
     */
    public static int computeSetFeatures(String faces_dir, String nonfaces_dir, boolean writeToDisk) { // Set = faces + non-faces
        int count = 0;
        count += computeImagesFeatures(faces_dir, writeToDisk);
        count += computeImagesFeatures(nonfaces_dir, writeToDisk);
        return count;
    }

    public static void computeFeaturesTimed(String path, String imagesFeatureFilepath, boolean training) {
        System.out.println("Computing features for:");
        System.out.println("  - " + path);
        int count = 0;
        long startTime = System.currentTimeMillis();
        count += computeSetFeatures(path + Conf.FACES, path + Conf.NONFACES, true);
        if (count > 0) {
            long elapsedTimeMS = (new Date()).getTime() - startTime;
            System.out.println("  Statistics:");
            System.out.println("    - Elapsed time: " + elapsedTimeMS / 1000 + "s");
            System.out.println("    - Images computed: " + count);
            System.out.println("    - image/seconds: " + count / (elapsedTimeMS / 1000));
        } else
            System.out.println("  - All features already computed!");
    }

    /**
     * Pour chaque feature:
     *      vector<pair<valeur-de-la-feature, l'index de l'exemple (image)>> ascendingFeatures;
     *      Pour chaque exemple:
     *          ascendingFeatures.add(<valeur-de-cette-feature-pour-cet-example, index-de-l'exemple>)
     *          trier ascendingFeatures en fonction de pair.first
     *          Write sur disque:
     *              * OrganizedFeatures (à l'index de la feature actuelle le ascendingFeatures.first en entier) tmp/training
     *              * OrganizedSample (à l'index de la feature actuelle le ascendingFeatures.second en entier)
     * <p>
     * Le résultat est le suivant:
     *      * OrganizedFeatures : (une ligne = une feature | chaque colonne dans cette ligne est la valeur de cette feature pour une image)
     *      * OrganizedSample   : (une ligne = une feature | chaque colonne dans cette ligne est l'index de l'image correspondante)
     * <p>
     * organizeFeatures works in-memory only if possible (enough heap memory), else on-disk (it could be extremely slow).
     */
    public static void organizeFeatures(long featureCount, ArrayList<String> examples, String feature, String sample) {
        System.out.println("Organizing features...");
        long startTime = System.currentTimeMillis();

        int trainN = examples.size();

        if (fileExists(feature)) {
            if (!validSizeOfArray(feature, trainN * featureCount))
                deleteFile(feature);
        }
        if (fileExists(sample)) {
            if (!validSizeOfArray(sample, trainN * featureCount))
                deleteFile(sample);
        }
        if (fileExists(feature) && fileExists(sample)) { // Already exist & both good!
            System.out.println("  - Already computed!");
            return;
        }

        assert examples.size() == trainN;

        for (long featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            // <exampleIndex, value>
            ArrayList<Pair<Integer, Integer>> ascendingFeatures = new ArrayList<>();

            for (int exampleIndex = 0; exampleIndex < trainN; exampleIndex++)
                ascendingFeatures.add(new Pair<>(exampleIndex, readIntFromMemory(examples.get(exampleIndex) + Conf.FEATURE_EXTENSION, featureIndex)));

            Collections.sort(ascendingFeatures, (o1, o2) -> o1.getValue().compareTo(o2.getValue()));

            ArrayList<Integer> permutedSamples = new ArrayList<>(trainN);
            ArrayList<Integer> permutedFeatures = new ArrayList<>(trainN);

            for (int k = 0; k < trainN; k++) {
                permutedSamples.add(ascendingFeatures.get(k).getKey());
                permutedFeatures.add(ascendingFeatures.get(k).getValue());
            }

            appendArrayToDisk(sample, permutedSamples);
            appendArrayToDisk(feature, permutedFeatures);
        }
        long elapsedTimeMS = (new Date()).getTime() - startTime;
        System.out.println("  - Done in " + (elapsedTimeMS/1000) + "s");
    }

    /**
     * Call with organizedSample
     */
    public static int getExampleIndex(long featureIndex, int iterator, int trainN, String organizedSample) {
        return readIntFromDisk(organizedSample, featureIndex * trainN + iterator);
    }
    public static int getExampleIndex(long featureIndex, int iterator, int trainN) {
        return getExampleIndex(featureIndex, iterator, trainN, Conf.ORGANIZED_SAMPLE);
    }
    public static int[] getFeatureExamplesIndexes(long featureIndex, int trainN) {
        return readArrayFromDisk(Conf.ORGANIZED_SAMPLE, featureIndex * trainN, trainN * (featureIndex + 1));
    }

    /**
     * Call with organizedFeature
     */
    public static int getExampleFeature(long featureIndex, int iterator, int trainN, String organizedFeatures) {
        return readIntFromDisk(organizedFeatures, featureIndex * trainN + iterator);
    }
    public static int getExampleFeature(long featureIndex, int iterator, int trainN) {
        return getExampleFeature(featureIndex, iterator, trainN, Conf.ORGANIZED_FEATURES);
    }
    public static int[] getFeatureValues(long featureIndex, int trainN) {
        return readArrayFromDisk(Conf.ORGANIZED_FEATURES, featureIndex * trainN, trainN * (featureIndex + 1));
    }
}
