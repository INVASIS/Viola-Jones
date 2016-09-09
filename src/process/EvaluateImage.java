package process;

import GUI.ImageHandler;
import cuda.AnyFilter;
import cuda.HaarDetector;
import javafx.util.Pair;
import org.jgrapht.DirectedGraph;
import org.jgrapht.ListenableGraph;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.alg.ConnectivityInspector;
import org.jgrapht.alg.StrongConnectivityInspector;
import org.jgrapht.alg.interfaces.StrongConnectivityAlgorithm;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.ListenableDirectedGraph;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.jgrapht.graph.SimpleGraph;
import process.features.Face;
import process.features.Rectangle;
import utils.Serializer;
import utils.Utils;

import java.util.*;

import static process.features.FeatureExtractor.computeImageFeatures;
import static utils.Serializer.readFeatures;

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

    private HashMap<Integer, Integer> neededHaarValues;
    private HaarDetector haarDetector;

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

        neededHaarValues = new HashMap<>();

        int i = 0;
        for (int layer = 0; layer < layerCount; layer++) {
            for (StumpRule rule : cascade[layer]) {
                if (!neededHaarValues.containsKey((int) rule.featureIndex)) {
                    neededHaarValues.put((int) rule.featureIndex, i);
                    i++;
                }
            }
        }
        System.out.println("Found " + i + " different indexes");
        this.haarDetector = new HaarDetector(neededHaarValues);
        this.haarDetector.setUp(baseWidth, baseHeight);

    }

    // TODO : remove unnecessary har features to compute only those needed
    // TODO : test if this is really more efficient to go through cuda to compute those haar features needed...

    public boolean guess(String fileName) {

        // Handle the case this is not a haar file
        int[] haar;
        if (fileName.endsWith(Conf.IMAGES_EXTENSION) || !Utils.fileExists(fileName)) {
            haar = computeImageFeatures(fileName, false);
        } else {
            haar = readFeatures(fileName);
        }

        return Classifier.isFace(this.cascade, this.tweaks, haar, this.layerCount) > 0;

    }

    public ArrayList<Face> getFaces(String fileName) {
        ImageHandler imageHandler = new ImageHandler(fileName);

        return getFaces(imageHandler);
    }

    // TODO : centrer-reduire les rectangles
    public ArrayList<Face> getFaces(ImageHandler imageHandler) {

        ArrayList<Rectangle> allRectangles = getAllRectangles(imageHandler);
        ArrayList<Face> res = new ArrayList<>();

        int[] haar;
        for (Rectangle rectangle : allRectangles) {

            int[][] tmpImage = new int[rectangle.getWidth()][rectangle.getHeight()];

            // TODO : improve copy or optimize this !
            for (int x = rectangle.getX(); x < rectangle.getWidth() + rectangle.getX(); x++)
                System.arraycopy(imageHandler.getGrayImage()[x], rectangle.getY(), tmpImage[x - rectangle.getX()], 0, rectangle.getY() + rectangle.getHeight() - rectangle.getY());

            ImageHandler tmpImageHandler = new ImageHandler(tmpImage, rectangle.getWidth(), rectangle.getHeight());


            // TODO : make it work (or find another solution)
            //haar = computeImageFeaturesDetector(imageHandler, haarDetector, (float) (rectangle.getHeight()) / (float) baseHeight);

            haar = computeImageFeatures(downsamplingImage(tmpImageHandler), false, null);

            double confidence = Classifier.isFace(cascade, tweaks, haar, layerCount);
            if (confidence > 0) {
                res.add(new Face(rectangle, confidence));
            }
        }

        res = postProcessing(res);
        // TODO : call post-processing to remove unnecessary rectangles
        return res;
    }

    public ArrayList<Rectangle> getAllRectangles(ImageHandler imageHandler) {
        return getAllRectangles(imageHandler.getWidth(), imageHandler.getHeight(), SCALE_COEFF);
    }

    public ArrayList<Rectangle> getAllRectangles(int imageWidth, int imageHeight, float coeff) {

        if (coeff <= 1) {
            System.err.println("Error for coeff in getAllRectanges, coeff should be > 1. coeff=" + coeff + " Aborting now!");
            System.exit(1);
        }

        // Quick and dirty way to reduce the number of rectangles
        // TODO : needs to be improved!
        int xDisplacer = imageWidth / 100;
        if (xDisplacer > 10)
            xDisplacer = 10;
        if (xDisplacer < 1)
            xDisplacer = 1;

        int yDisplacer = imageHeight / 100;
        if (yDisplacer > 10)
            yDisplacer = 10;
        if (yDisplacer < 1)
            yDisplacer = 1;

        ArrayList<Rectangle> rectangles = new ArrayList<>();

        int minDim = Math.min(imageHeight, imageWidth);
        int frameSize = Math.max(baseWidth, baseHeight);

        // FIXME : ne pas avancer pixel par pixel mais plutôt 3-4 pixels à la fois
        for (int frame = frameSize; frame <= minDim; frame *= coeff) {
            for (int x = 0; x <= imageWidth - frame; x += xDisplacer) {
                for (int y = 0; y <= imageHeight - frame; y += yDisplacer) {
                    rectangles.add(new Rectangle(x, y, frame, frame));
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

            float rowPos = (float) (blured.getWidth() - 1) / (float) (baseWidth + 1) * (float) (row + 1);
            float colPos = (float) (blured.getHeight() - 1) / (float) (baseHeight + 1) * (float) (col + 1);

            int lowRow = Math.max((int) Math.floor(rowPos), 0);
            int upRow = Math.min(lowRow + 1, blured.getWidth() - 1);

            int lowCol = Math.max((int) Math.floor(colPos), 0);
            int upCol = Math.min(lowCol + 1, blured.getHeight() - 1);

            newImg[row][col] = (grayImage[lowRow][lowCol] + grayImage[lowRow][upCol] + grayImage[upRow][lowCol] + grayImage[upRow][upCol]) / 4;

        }

        return new ImageHandler(newImg, baseWidth, baseHeight);

    }

    // TODO : can improve perf here !
    // TODO : Improve by discarging rectangles with not enouth red on the original face (on the image)
    private ArrayList<Face> postProcessing(ArrayList<Face> allFaces) {
        ArrayList<Face> result = new ArrayList<>();
        ArrayList<Pair<Integer, Integer>> centers = new ArrayList<>();

        UndirectedGraph g = new SimpleGraph(DefaultEdge.class);
        for (Face face : allFaces) {
            int x = (face.getX() * 2 + face.getWidth()) / 2;
            int y = (face.getY() * 2 + face.getHeight()) / 2;

            Pair<Integer, Integer> center = new Pair<>(x, y);
            centers.add(center);
            g.addVertex(center);
            g.addVertex(face);
        }

        for (Face face : allFaces)
            centers.stream().filter(center -> face.conrains(center.getKey(), center.getValue())).forEach(center -> g.addEdge(face, center));

        ConnectivityInspector inspector = new ConnectivityInspector(g);

        ArrayList list = (ArrayList) inspector.connectedSets();
        for (Object aList : list) {
            HashSet set = (HashSet) aList;

            Face face = new Face(new Rectangle(0, 0, 0, 0), -1);
            for (Object o : set) {
                if (o instanceof Face) {
                    Face candidate = (Face) o;
                    if (candidate.getConfidence() > face.getConfidence())
                        face = candidate;
                }
            }
            if (face.getConfidence() > 0)
                result.add(face);
        }

        return result;
    }

    public HashMap<Integer, Integer> getNeededHaarValues() {
        return neededHaarValues;
    }
}
