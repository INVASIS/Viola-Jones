import process.Classifier;
import process.Conf;


public class Main {
    public static void main(String[] args) {
        Conf.haarExtractor.setUp(19, 19);

        // TODO : TO CONSTANTS
        float overallTargetDetectionRate = 0.80f;
        float overallTargetFalsePositiveRate = 0.000001f;
        float targetDetectionRate = 0.995f;
        float targetFalsePositiveRate = 0.5f;

        if (Conf.USE_CUDA)
            Conf.haarExtractor.setUp(19, 19);

        System.out.println("Max memory : " + Runtime.getRuntime().maxMemory());
        System.out.println("Free memory : " + Runtime.getRuntime().freeMemory());
        System.out.println("Total memory : " + Runtime.getRuntime().totalMemory());



        Classifier classifier = new Classifier(19, 19);
        classifier.train("data/trainset", 0.5f, overallTargetDetectionRate, overallTargetFalsePositiveRate, targetDetectionRate, targetFalsePositiveRate);
        classifier.test("data/testset");
    }
}
