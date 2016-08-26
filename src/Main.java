import process.Conf;
import process.training.Classifier;


public class Main {
    public static void main(String[] args) {

        // TODO : TO CONSTANTS
        float overallTargetDetectionRate = 0.80f;
        float overallTargetFalsePositiveRate = 0.000001f;
        float targetDetectionRate = 0.995f;
        float targetFalsePositiveRate = 0.5f;

        if (Conf.USE_CUDA)
            Conf.haarExtractor.setUp(19, 19);

        if (Conf.haarExtractor != null)
            Conf.haarExtractor.freeCuda();

        Classifier classifier = new Classifier("data/trainset", "data/testset", 19, 19);
        classifier.train(overallTargetDetectionRate, overallTargetFalsePositiveRate, targetDetectionRate, targetFalsePositiveRate);
    }
}
