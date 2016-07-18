import GUI.Display;
import GUI.ImageHandler;
import cuda.AnyFilter;
import cuda.HaarExtractor;
import jcuda.jcublas.JCublas;
import process.features.Feature;
import process.features.FeatureExtractor;

import java.util.ArrayList;

/**
 * this is a main class
 */
public class Main {
    public static void main(String[] args) {

        ImageHandler imageHandler = new ImageHandler("data/face.jpg");

        Display.drawImage(imageHandler.getBufferedImage());
        Display.drawImage(imageHandler.getGrayBufferedImage());

        AnyFilter filter = new AnyFilter(imageHandler.getWidth(), imageHandler.getHeight(), imageHandler.getGrayImage());
        filter.compute();

//        HaarExtractor haarExtractor = new HaarExtractor(imageHandler.getGrayImage(), imageHandler.getIntegralImage(), imageHandler.getWidth(), imageHandler.getHeight());
//        haarExtractor.compute();

        FeatureExtractor fc = new FeatureExtractor(imageHandler);
        ArrayList<Feature> features = fc.getAllFeatures();
    }
}
