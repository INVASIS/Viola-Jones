package process;

import GUI.ImageHandler;
import cuda.AnyFilter;
import process.features.Rectangle;
import utils.Serializer;
import utils.Utils;

import java.util.ArrayList;

import static process.features.FeatureExtractor.computeImageFeatures;
import static utils.Serializer.readFeaturesFromDisk;

public class EvaluateImage {

    public static float SCALE_COEFF = 1.25f;

    private int baseWidth;
    private int baseHeight;

    private int countTestPos;
    private int countTestNeg;
    private int testN;

    private String directory;

    private ArrayList<StumpRule> rules;
    private ArrayList<Integer> layerCommitteeSize;
    private ArrayList<Float> tweaks;
    private int layerCount;

    private ArrayList<StumpRule>[] cascade;

    public EvaluateImage(int countTestPos, int countTestNeg, String directory, int baseWidth, int baseHeight) {
        this.countTestPos = countTestPos;
        this.countTestNeg = countTestNeg;
        this.testN = countTestNeg + countTestPos;
        this.directory = directory;
        this.baseHeight = baseHeight;
        this.baseWidth = baseWidth;

        init();
    }

    private void init() {
        this.rules = Serializer.readRule(Conf.TRAIN_FEATURES);

        this.layerCommitteeSize = new ArrayList<>();
        this.tweaks = new ArrayList<>();
        this.layerCount = Serializer.readLayerMemory(Conf.TRAIN_FEATURES, this.layerCommitteeSize, this.tweaks);

        cascade = new ArrayList[this.layerCount];

        int committeeStart = 0;
        for (int i = 0; i < this.layerCount; i++) {
            cascade[i] = new ArrayList<>();
            for (int committeeIndex = committeeStart; committeeIndex < this.layerCommitteeSize.get(i) + committeeStart; committeeIndex++) {
                cascade[i].add(rules.get(committeeIndex));
            }
            committeeStart += this.layerCommitteeSize.get(i);
        }
    }

    // TODO : remove unnecessary har features to compute only those needed
    // TODO : test if this is really more efficient to go through cuda to compute those haar features needed...

    public boolean guess(String fileName) {

        // Handle the case this is not a haar file
        int[] haar;
        if (fileName.endsWith(Conf.IMAGES_EXTENSION) || !Utils.fileExists(fileName)) {
            haar = computeImageFeatures(fileName, false).stream().mapToInt(i -> i).toArray();
            ;
        } else {
            haar = readFeaturesFromDisk(fileName);
        }

        return Classifier.isFace(this.cascade, this.tweaks, haar, this.layerCount);

    }

    public ArrayList<Rectangle> getFaces(String fileName) {
        ImageHandler imageHandler = new ImageHandler(fileName);

        return getFaces(imageHandler);
    }

    public ArrayList<Rectangle> getFaces(ImageHandler imageHandler) {

        ArrayList<Rectangle> allRectangles = getAllRectangles(imageHandler);
        ArrayList<Rectangle> res = new ArrayList<>();


        int[] haar;

        return null;
    }

    public ArrayList<Rectangle> getAllRectangles(ImageHandler imageHandler) {
        return getAllRectangles(imageHandler.getWidth(), imageHandler.getHeight(), SCALE_COEFF);
    }

    public ArrayList<Rectangle> getAllRectangles(int imageWidth, int imageHeight, float coeff) {

        if (coeff <= 1) {
            System.err.println("Error for coeff in getAllRectanges, coeff should be > 1. coeff=" + coeff + " Aborting now!");
            System.exit(1);
        }

        ArrayList<Rectangle> rectangles = new ArrayList<>();

        int minDim = Math.min(imageHeight, imageWidth);
        int frameSize = Math.max(baseWidth, baseHeight);

        // FIXME : ne pas avancer pixel par pixel mais plutôt 3-4 pixels à la fois
        for (int frame = frameSize; frame <= minDim; frame *= coeff) {
            for (int x = 0; x <= imageWidth - frame; x++) {
                for (int y = 0; y <= imageHeight - frame; y++) {
                    rectangles.add(new Rectangle(x, y, frame - 1, frame - 1));
                }
            }
        }

        return rectangles;
    }

    public ImageHandler downsamplingImage(ImageHandler input) {

        /*
        // STANDARD DEVIATION
        double sum = 0;
        double sumSum = 0;
        for (Integer exampleFeatureValue : exampleFeatureValues) {
            sum += exampleFeatureValue;
            sumSum += exampleFeatureValue * exampleFeatureValue;
        }
        // standardDeviation = SQRT(VAR(X))
        double standardDeviation = Math.sqrt((sumSum/Math.pow((double)featureCount, 2)) - (Math.pow(sum/Math.pow((float)featureCount, 2), 2)));
         */


        // FIXME : compute a better gaussian blur to improve accuracy
        // Approximation of gaussian blur :
        float[][] blurMatrix = {
                {1 / 16f, 1 / 8f, 1 / 16f},
                {1 / 8f, 1 / 4f, 1 / 8f},
                {1 / 16f, 1 / 8f, 1 / 16f}
        };

        AnyFilter blurFilter = new AnyFilter(input.getWidth(), input.getHeight(), input.getGrayImage(), blurMatrix);

        ImageHandler blured = blurFilter.compute();

        int[][] grayImage = blured.getGrayImage();

        int[][] newImg = new int[baseWidth][baseHeight];

        for (int i = 0; i < baseHeight * baseWidth; i++) {

            int row = i / baseHeight;
            int col = i % baseWidth;

            float rowPos = (float)(blured.getWidth() - 1) / (float)(baseWidth + 1) * (float)(row + 1);
            float colPos = (float)(blured.getHeight() - 1) / (float)(baseHeight + 1) * (float)(col + 1);

            int lowRow = Math.max((int)Math.floor(rowPos), 0);
            int upRow = Math.min(lowRow + 1, blured.getWidth() - 1);

            int lowCol = Math.max((int)Math.floor(colPos), 0);
            int upCol = Math.min(lowCol + 1, blured.getHeight() - 1);

            newImg[row][col] = (grayImage[lowRow][lowCol] + grayImage[lowRow][upCol] + grayImage[upRow][lowCol] + grayImage[upRow][upCol]) / 4;

        }

        return new ImageHandler(newImg, baseWidth, baseHeight);

    }
}
