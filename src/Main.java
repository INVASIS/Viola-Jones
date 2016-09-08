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
        float overallTargetDetectionRate = 0.999f;
        float overallTargetFalsePositiveRate = 0.000001f;
        float targetDetectionRate = 0.995f;
        float targetFalsePositiveRate = 0.5f;

        if (Conf.USE_CUDA)
            Conf.haarExtractor.setUp(width, height);

        System.out.println("Max memory : " + Runtime.getRuntime().maxMemory());
        System.out.println("Free memory : " + Runtime.getRuntime().freeMemory());
        System.out.println("Total memory : " + Runtime.getRuntime().totalMemory());
        System.out.println("Available Processors (num of max threads) : " + Runtime.getRuntime().availableProcessors());

        Serializer.featureCount = countAllFeatures(width, height);

        Classifier classifier = new Classifier(width, height);
        //classifier.train("data/trainset", "data/testset", 0.5f, overallTargetDetectionRate, overallTargetFalsePositiveRate, targetFalsePositiveRate, true);
        classifier.test("data/testset");
    }
}
