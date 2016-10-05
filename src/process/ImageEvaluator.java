package process;

import GUI.ImageHandler;
import cuda.AnyFilter;
import cuda.HaarDetector;
import javafx.util.Pair;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.alg.ConnectivityInspector;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;
import process.features.Face;
import process.features.Rectangle;
import utils.CascadeSerializer;
import utils.Serializer;

import java.util.*;

import static process.Test.isFace;

public class ImageEvaluator {

    private static final float SCALE_COEFF = 1.25f;

    private int trainWidth;
    private int trainHeight;

    private int layerCount;

    private ArrayList<ArrayList<StumpRule>> cascade;
    private ArrayList<Float> tweaks;

    private HashMap<Integer, Integer> neededHaarValues;
    public HaarDetector haarDetector;
    public ArrayList<Rectangle> slidingWindows;

    public long computingTimeMS;

    public ImageEvaluator(int trainWidth, int trainHeight, int imgWidth, int imgHeight,
                          int xDisplacer, int yDisplacer, int minSlidingSize, int maxSlidingSize, ArrayList<ArrayList<StumpRule>> cascade, ArrayList<Float> tweaks) {
        this(trainWidth, trainHeight, imgWidth, imgHeight, xDisplacer, yDisplacer, minSlidingSize, maxSlidingSize, SCALE_COEFF, cascade, tweaks);
    }

    public ImageEvaluator(int trainWidth, int trainHeight, int imgWidth, int imgHeight,
                          int xDisplacer, int yDisplacer, int minSlidingSize, int maxSlidingSize, float coeff, ArrayList<ArrayList<StumpRule>> cascade, ArrayList<Float> tweaks) {
        this.trainHeight = trainHeight;
        this.trainWidth = trainWidth;

        this.computingTimeMS = 0;

        this.tweaks = tweaks;
        //this.cascade = CascadeSerializer.readLayerMemory(Conf.TRAIN_FEATURES, this.tweaks, tmp);
        this.cascade = cascade;
        this.layerCount = cascade.size();


        // Define new indexes for wanted haar features
        {
            this.neededHaarValues = new HashMap<>();
            int i = 0;
            for (int layer = 0; layer < layerCount; layer++) {
                for (StumpRule rule : cascade.get(layer)) {
                    if (!neededHaarValues.containsKey((int) rule.featureIndex)) {
                        neededHaarValues.put((int) rule.featureIndex, i);
                        i++;
                    }
                }
            }
            System.out.println("Found " + i + " different indexes");
        }

        if(coeff < 1.08 && coeff > 1.5) { // TODO
            System.err.println("WARNING : SCALE_COEFF out of bounds [1.08 ; 1.5] - SCALE_COEFF used : " + SCALE_COEFF);
            this.slidingWindows = getAllRectangles(imgWidth, imgHeight, SCALE_COEFF, xDisplacer, yDisplacer, minSlidingSize, maxSlidingSize);
        }
        else
            this.slidingWindows = getAllRectangles(imgWidth, imgHeight, coeff, xDisplacer, yDisplacer, minSlidingSize, maxSlidingSize);
        this.haarDetector = new HaarDetector(neededHaarValues, trainHeight, imgWidth, imgHeight, slidingWindows);
    }

    public ArrayList<Face> getFaces(String fileName, boolean postProcess) {
        ImageHandler imageHandler = new ImageHandler(fileName);
        return getFaces(imageHandler, postProcess);
    }

    /**
     * TODO : centrer-reduire les rectangles
     * TODO : make it parallel ??
     * Algorithm 7 from the original paper
     */
    public ArrayList<Face> getFaces(ImageHandler image, boolean postProcess) {
        ArrayList<Face> res = new ArrayList<>();


        long cudaMilliseconds = System.currentTimeMillis();
        int[] haar = haarDetector.computeImage(image);
        computingTimeMS += System.currentTimeMillis() - cudaMilliseconds;

        int offset = 0;
        int haarSize = neededHaarValues.size();
        for (Rectangle rectangle : slidingWindows) {
            // Get features for that rectangle
            int tmpHaar[] = new int[haarSize];
            System.arraycopy(haar, offset, tmpHaar, 0, haarSize);
            offset += haarSize;

            double confidence = isFace(cascade, tweaks, tmpHaar, layerCount, neededHaarValues);
            if (confidence > 0) {
                res.add(new Face(rectangle, confidence));
            }
        }

        if (postProcess)
            res = postProcessing(res);
        return res;
    }

    private ArrayList<Rectangle> getAllRectangles(ImageHandler imageHandler) {

        int minDim = Math.min(imageHandler.getHeight(), imageHandler.getWidth());
        int frameSize = Math.max(trainWidth, trainHeight);

        // TODO : needs to be improved!
        int xDisplacer = imageHandler.getWidth() / 100;
        if (xDisplacer > 10)
            xDisplacer = 10;
        if (xDisplacer < 1)
            xDisplacer = 1;

        int yDisplacer = imageHandler.getHeight() / 100;
        if (yDisplacer > 10)
            yDisplacer = 10;
        if (yDisplacer < 1)
            yDisplacer = 1;

        return getAllRectangles(imageHandler.getWidth(), imageHandler.getHeight(), SCALE_COEFF, xDisplacer, yDisplacer, frameSize, minDim);
    }

    public static ArrayList<Rectangle> getAllRectangles(int imageWidth, int imageHeight, float coeff, int xDisplacer, int yDisplacer, int minSlidingWindowSize, int maxSlidingWindowSize) {

        if (coeff <= 1) {
            System.err.println("Error for coeff in getAllRectanges, coeff should be > 1. coeff=" + coeff + " Aborting now!");
            System.exit(1);
        }

        if (minSlidingWindowSize * coeff == minSlidingWindowSize) {
            System.err.println("Coeff too small, cannot scale minwindow. Aborting now!");
            System.exit(1);
        }

        ArrayList<Rectangle> rectangles = new ArrayList<>();
        for (int frame = minSlidingWindowSize; frame <= maxSlidingWindowSize; frame *= coeff) {
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

        int[][] newImg = new int[trainWidth][trainHeight];

        for (int i = 0; i < trainHeight * trainWidth; i++) {

            int row = i / trainHeight;
            int col = i % trainWidth;

            float rowPos = (float) (blured.getWidth() - 1) / (float) (trainWidth + 1) * (float) (row + 1);
            float colPos = (float) (blured.getHeight() - 1) / (float) (trainHeight + 1) * (float) (col + 1);

            int lowRow = Math.max((int) Math.floor(rowPos), 0);
            int upRow = Math.min(lowRow + 1, blured.getWidth() - 1);

            int lowCol = Math.max((int) Math.floor(colPos), 0);
            int upCol = Math.min(lowCol + 1, blured.getHeight() - 1);

            newImg[row][col] = (grayImage[lowRow][lowCol] + grayImage[lowRow][upCol] + grayImage[upRow][lowCol] + grayImage[upRow][upCol]) / 4;

        }

        return new ImageHandler(newImg, trainWidth, trainHeight);

    }

    // TODO : can improve perf  + res here !
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
            float cumulatedConfidence = 0;
            int lap = 0;
            for (Object o : set) {
                if (o instanceof Face) {
                    Face candidate = (Face) o;
                    cumulatedConfidence += candidate.getConfidence();
                    lap++;
                    if (candidate.getConfidence() > face.getConfidence())
                        face = candidate;
                }
            }
            if (lap >= 3 && cumulatedConfidence > 18 && face.getConfidence() > 0)
                result.add(face);
        }

        return result;
    }

    public HashMap<Integer, Integer> getNeededHaarValues() {
        return neededHaarValues;
    }
}
