package Statistics;

import GUI.ImageHandler;
import process.Conf;
import process.ImageEvaluator;
import process.features.Face;
import process.features.Feature;
import process.features.FeatureExtractor;
import utils.Utils;

import java.util.ArrayList;

import static utils.Utils.streamFiles;

public class Perfs {

    public static void benchmarksTrainFeatures() {

        int[] result = new int[(int) Conf.haarExtractor.getNUM_TOTAL_FEATURES()];
        long cudaMilliseconds = 0;
        long cudaTotalTime = 0;

        long cpuTotalTime = 0;
        long cpuMilliseconds = 0;

        int maxIter = 1000;

        int i = 0;
        for (String path : Utils.streamFiles("data/trainset/faces", ".png")) {

            ImageHandler image = new ImageHandler(path);

            cudaMilliseconds = System.currentTimeMillis();
            Conf.haarExtractor.updateImage(image.getIntegralImage());
            Conf.haarExtractor.compute();
            int offset = 0;
            System.arraycopy(Conf.haarExtractor.getFeaturesA(), 0, result, offset, (int) Conf.haarExtractor.getNUM_FEATURES_A());
            offset += (int) Conf.haarExtractor.getNUM_FEATURES_A();
            System.arraycopy(Conf.haarExtractor.getFeaturesB(), 0, result, offset, (int) Conf.haarExtractor.getNUM_FEATURES_B());
            offset += (int) Conf.haarExtractor.getNUM_FEATURES_B();
            System.arraycopy(Conf.haarExtractor.getFeaturesC(), 0, result, offset, (int) Conf.haarExtractor.getNUM_FEATURES_C());
            offset += (int) Conf.haarExtractor.getNUM_FEATURES_C();
            System.arraycopy(Conf.haarExtractor.getFeaturesD(), 0, result, offset, (int) Conf.haarExtractor.getNUM_FEATURES_D());
            offset += (int) Conf.haarExtractor.getNUM_FEATURES_D();
            System.arraycopy(Conf.haarExtractor.getFeaturesE(), 0, result, offset, (int) Conf.haarExtractor.getNUM_FEATURES_E());

            cudaTotalTime += System.currentTimeMillis() - cudaMilliseconds;

            int cpt = 0;
            cpuMilliseconds = System.currentTimeMillis();
            for (ArrayList<Feature> features : FeatureExtractor.streamFeaturesByType(image))
                for (Feature f : features)
                    result[cpt++] = f.getValue();

            cpuTotalTime += System.currentTimeMillis() - cpuMilliseconds;

            i++;
            if (i == maxIter)
                break;
        }

        System.out.println(result[2344]);
        System.out.println("------ TEST 1 ------");
        System.out.println("BENCHMARK TRAIN HAAR");
        System.out.println("NUMBER OF ITERATIONS:" + maxIter);
        System.out.println("CUDA  : total time: " + cudaTotalTime);
        System.out.println("CPU   : total time: " + cpuTotalTime);
        System.out.println("RATIO : CPU/CUDA  : " + (float) cpuTotalTime / (float) cudaTotalTime);
    }

    public static void benchmarkBestStump() {

        // TODO : benchmark for bestStump!

        System.out.println("------ TEST 2 ------");
        System.out.println("BENCHMARK BEST STUMP");

    }

    public static void benchmarkDetect() {
        ImageEvaluator imageEvaluator;

        System.out.println("------ TEST 3 ------");
        System.out.println("  BENCHMARK DETECT");

        int r = 0;
        // On test set
        {
            Conf.USE_CUDA = true;
            imageEvaluator = new ImageEvaluator(19, 19, 19, 19, 1, 1, 19, 19);

            for (String listTestFace : streamFiles("data/testset" + Conf.FACES, Conf.IMAGES_EXTENSION)) {
                r += imageEvaluator.getFaces(listTestFace, false).size();
            }

            for (String listTestFace : streamFiles("data/testset" + Conf.NONFACES, Conf.IMAGES_EXTENSION)) {
                r += imageEvaluator.getFaces(listTestFace, false).size();
            }

            System.out.println("Total computing time for HaarDetector (GPU): " + imageEvaluator.computingTimeMS + "ms for 24044 images 19x19 (" + r + " faces detected)");


            Conf.USE_CUDA = false;
            imageEvaluator = new ImageEvaluator(19, 19, 19, 19, 1, 1, 19, 19);

            for (String listTestFace : streamFiles("data/testset" + Conf.FACES, Conf.IMAGES_EXTENSION)) {
                r = imageEvaluator.getFaces(listTestFace, false).size();
            }
            for (String listTestFace : streamFiles("data/testset" + Conf.NONFACES, Conf.IMAGES_EXTENSION)) {
                r += imageEvaluator.getFaces(listTestFace, false).size();
            }

            System.out.println("Total computing time for HaarDetector (CPU): " + imageEvaluator.computingTimeMS + "ms for 24044 images 19x19 (" + r + " faces detected)");
        }
        {
            Conf.USE_CUDA = true;
            imageEvaluator = new ImageEvaluator(19, 19, 2048, 1536, 3, 3, 100, 300);

            int i = 0;
            for (String listTestFace : streamFiles("data/high-res", ".jpg")) {
                r += imageEvaluator.getFaces(listTestFace, false).size();
                i++;
                System.out.println("i=" + i);
                if (i > 20)
                    break;
            }

            System.out.println("Total computing time for HaarDetector (GPU): " + imageEvaluator.computingTimeMS + "ms for " + i + " images 2048x1536 (" + r + " faces detected)");


            i = 0;
            Conf.USE_CUDA = false;
            imageEvaluator = new ImageEvaluator(19, 19, 2048, 1536, 3, 3, 100, 300);

            for (String listTestFace : streamFiles("data/high-res", ".jpg")) {
                r = imageEvaluator.getFaces(listTestFace, false).size();
                i++;
                System.out.println("i=" + i);
                if (i > 20)
                    break;
            }

            System.out.println("Total computing time for HaarDetector (CPU): " + imageEvaluator.computingTimeMS + "ms for " + i + " images 2048x1536 (" + r + " faces detected)");
        }
    }
}
