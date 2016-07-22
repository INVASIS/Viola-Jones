import GUI.ImageHandler;
import cuda.AnyFilter;
import cuda.HaarExtractor;
import process.training.Classifier;


/**
 * this is a main class
 */
public class Main {
    public static void main(String[] args) {

        //ImageHandler imageHandler = new ImageHandler("data/face.jpg");
        ImageHandler imageHandler = new ImageHandler("data/testset-19x19/face-png/face00001.png");

//        Display.drawImage(imageHandler.getBufferedImage());
//        Display.drawImage(imageHandler.getGrayBufferedImage());

        AnyFilter filter = new AnyFilter(imageHandler.getWidth(), imageHandler.getHeight(), imageHandler.getGrayImage());
        filter.compute();

        HaarExtractor haarExtractor = new HaarExtractor(imageHandler.getGrayImage(), imageHandler.getIntegralImage(), imageHandler.getWidth(), imageHandler.getHeight());
        haarExtractor.compute();

//        FeatureExtractor fc = new FeatureExtractor(imageHandler);
//        ArrayList<Feature> features = fc.getAllFeatures();

        Classifier.train("data/testset-19x19/face-png", "data/testset-19x19/non-face-png", 19, 19);

    }
}
