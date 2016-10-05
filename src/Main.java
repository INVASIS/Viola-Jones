import Statistics.Perfs;
import process.Classifier;
import process.Conf;
import process.StumpRule;
import process.Test;
import utils.CascadeSerializer;
import utils.Serializer;

import java.util.ArrayList;

import static process.features.FeatureExtractor.countAllFeatures;


public class Main {
    public static void main(String[] args) {
        // Training image size
        int width = 19;
        int height = 19;

        // TODO : TO CONSTANTS
        float cascadeTargetFPR = 0.0000005f;
        float cascadeTargetAccuracy = 0.985f;
        float layerTargetFPR = 0.50f;

        Serializer.featureCount = countAllFeatures(width, height);

        if (Conf.USE_CUDA)
            Conf.haarExtractor.setUp(width, height);

        Conf.loadLibraries();

        System.out.println("JDK Version: " + System.getProperty("java.specification.version"));
        System.out.println("Max memory: " + Runtime.getRuntime().maxMemory());
        System.out.println("Free memory: " + Runtime.getRuntime().freeMemory());
        System.out.println("Total memory: " + Runtime.getRuntime().totalMemory());
        System.out.println("Available Processors (num of max threads) : " + Runtime.getRuntime().availableProcessors());

        Serializer.featureCount = countAllFeatures(width, height);


        Classifier classifier = new Classifier(width, height);
        classifier.train("data/trainset", "data/testset", 0.5f, cascadeTargetAccuracy, cascadeTargetFPR, layerTargetFPR, true);


//        ArrayList<Float> tweaks = new ArrayList<>();
//        ArrayList<ArrayList<StumpRule>> cascade = CascadeSerializer.loadCascadeFromXML(Conf.TRAIN_DIR + "/cascade-2016-10-05-14-39-01.data", tweaks);
//        classifier.test("data/testset", cascade, tweaks);
//        Test.soutenance(19, 19, cascade, tweaks);

//        Perfs.benchmarksTrainFeatures();
//        Perfs.benchmarkDetect(cascade, tweaks);
//        Perfs.compareDetectFacesTime(width, height);
    }
}
