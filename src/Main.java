import GUI.ImageHandler;
import cuda.AnyFilter;
import cuda.CudaUtils;
import cuda.HaarExtractor;
import process.Conf;
import process.training.Classifier;

import java.util.List;


/**
 * this is a main class
 */
public class Main {
    public static void main(String[] args) {

        CudaUtils.initCuda();
        if (Conf.USE_CUDA) {
            Conf.haarExtractor.setUp(19, 19);
        }


        //ImageHandler imageHandler = new ImageHandler("data/face.jpg");
        ImageHandler imageHandler = new ImageHandler("data/testset-19x19/face-png/face00001.png");

//        Display.drawImage(imageHandler.getBufferedImage());
//        Display.drawImage(imageHandler.getGrayBufferedImage());

//        AnyFilter filter = new AnyFilter(imageHandler.getWidth(), imageHandler.getHeight(), imageHandler.getGrayImage());
//        filter.compute();
//
        //HaarExtractor haarExtractor = new HaarExtractor(imageHandler.getWidth(), imageHandler.getHeight());
        //haarExtractor.updateImage(imageHandler.getIntegralImage());
        //haarExtractor.compute();

//        FeatureExtractor fc = new FeatureExtractor(imageHandler);
//        ArrayList<Feature> features = fc.getAllFeatures();

        // Verif Should be all true
        int c = 0;
        for (List<Integer> i: imageHandler.getFeatures())
        {
            System.out.println(i.containsAll(imageHandler.computeFeatures().get(c++)));

        }
        //Classifier.train("data/testset-19x19/face-png", "data/testset-19x19/non-face-png", 19, 19);

        if (Conf.haarExtractor != null)
            Conf.haarExtractor.freeCuda();

    }
}
