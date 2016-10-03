import Statistics.Perfs;
import process.Classifier;
import process.Conf;
import utils.Serializer;

import static process.features.FeatureExtractor.countAllFeatures;


public class Main {
    public static void main(String[] args) {
        // Training image size
        int width = 19;
        int height = 19;

        // TODO : TO CONSTANTS
        float overallTargetDetectionRate = 0.985f;
        float overallTargetFalsePositiveRate = 0.0000005f;
        float targetDetectionRate = 0.995f;
        float targetFalsePositiveRate = 0.48f;

        Serializer.featureCount = countAllFeatures(width, height);

        if (Conf.USE_CUDA)
            Conf.haarExtractor.setUp(width, height);

        Conf.loadLibraries();

        System.out.println("JDK Version: " + System.getProperty("java.specification.version"));
        System.out.println("Max memory: " + Runtime.getRuntime().maxMemory());
        System.out.println("Free memory: " + Runtime.getRuntime().freeMemory());
        System.out.println("Total memory: " + Runtime.getRuntime().totalMemory());
        System.out.println("Available Processors (num of max threads) : " + Runtime.getRuntime().availableProcessors());

        Classifier classifier = new Classifier(width, height);
        classifier.train("data2/trainset", "data2/testset", 0.5f, overallTargetDetectionRate, overallTargetFalsePositiveRate, targetFalsePositiveRate, true);
        classifier.test("data/testset");

//        Perfs.benchmarksTrainFeatures();
//        Perfs.benchmarkDetect();
//        Perfs.compareDetectFacesTime(width, height);
    }
}
