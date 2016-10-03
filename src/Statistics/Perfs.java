package Statistics;

import GUI.ImageHandler;
import process.Conf;
import process.ImageEvaluator;
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

    private static int computeImageEval(String path, String ext, ImageEvaluator imageEvaluator) {
        int faces = 0;
        for (String listTestFace : streamFiles(path, ext)) {
            faces += imageEvaluator.getFaces(listTestFace, false).size();
        }
        return faces;
    }

    public static void benchmarkDetect() {
        ImageEvaluator imageEvaluator;

        System.out.println("------ TEST 3 ------");
        System.out.println("  BENCHMARK DETECT");

        int r;
        // On test set
        {
            Conf.USE_CUDA = true;
            imageEvaluator = new ImageEvaluator(19, 19, 19, 19, 1, 1, 19, 19);
            r = computeImageEval("data/testset", Conf.IMAGES_EXTENSION, imageEvaluator);
            System.out.println("Total computing time for HaarDetector (GPU): " + imageEvaluator.computingTimeMS + "ms for 24044 images 19x19 (" + r + " faces detected - " + imageEvaluator.haarDetector.slidingWindowsSize + " sliding windows)");
            imageEvaluator.haarDetector.close();

            Conf.USE_CUDA = false;
            imageEvaluator = new ImageEvaluator(19, 19, 19, 19, 1, 1, 19, 19);
            r = computeImageEval("data/testset", Conf.IMAGES_EXTENSION, imageEvaluator);
            System.out.println("Total computing time for HaarDetector (CPU): " + imageEvaluator.computingTimeMS + "ms for 24044 images 19x19 (" + r + " faces detected - " + imageEvaluator.haarDetector.slidingWindowsSize + " sliding windows)");
            imageEvaluator.haarDetector.close();

            Conf.USE_CUDA = true;
            imageEvaluator = new ImageEvaluator(19, 19, 19, 19, 1, 1, 10, 19);
            r = computeImageEval("data/testset", Conf.IMAGES_EXTENSION, imageEvaluator);
            System.out.println("Total computing time for HaarDetector (GPU): " + imageEvaluator.computingTimeMS + "ms for 24044 images 19x19 (" + r + " faces detected - " + imageEvaluator.haarDetector.slidingWindowsSize + " sliding windows)");
            imageEvaluator.haarDetector.close();

            Conf.USE_CUDA = false;
            imageEvaluator = new ImageEvaluator(19, 19, 19, 19, 1, 1, 10, 19);
            r = computeImageEval("data/testset", Conf.IMAGES_EXTENSION, imageEvaluator);
            System.out.println("Total computing time for HaarDetector (CPU): " + imageEvaluator.computingTimeMS + "ms for 24044 images 19x19 (" + r + " faces detected - " + imageEvaluator.haarDetector.slidingWindowsSize + " sliding windows)");
            imageEvaluator.haarDetector.close();

            Conf.USE_CUDA = true;
            imageEvaluator = new ImageEvaluator(19, 19, 19, 19, 1, 1, 5, 19);
            r = computeImageEval("data/testset", Conf.IMAGES_EXTENSION, imageEvaluator);
            System.out.println("Total computing time for HaarDetector (GPU): " + imageEvaluator.computingTimeMS + "ms for 24044 images 19x19 (" + r + " faces detected - " + imageEvaluator.haarDetector.slidingWindowsSize + " sliding windows)");
            imageEvaluator.haarDetector.close();

            Conf.USE_CUDA = false;
            imageEvaluator = new ImageEvaluator(19, 19, 19, 19, 1, 1, 5, 19);
            r = computeImageEval("data/testset", Conf.IMAGES_EXTENSION, imageEvaluator);
            System.out.println("Total computing time for HaarDetector (CPU): " + imageEvaluator.computingTimeMS + "ms for 24044 images 19x19 (" + r + " faces detected - " + imageEvaluator.haarDetector.slidingWindowsSize + " sliding windows)");
            imageEvaluator.haarDetector.close();
        }
        {
            Conf.USE_CUDA = true;
            imageEvaluator = new ImageEvaluator(19, 19, 2048, 1536, 3, 3, 100, 300);

            int i = 0;
            for (String listTestFace : streamFiles("data/high-res", ".jpg")) {
                r += imageEvaluator.getFaces(listTestFace, false).size();
                i++;
                if (i > 20)
                    break;
            }

            System.out.println("Total computing time for HaarDetector (GPU): " + imageEvaluator.computingTimeMS + "ms for " + i + " images 2048x1536 (" + r + " faces detected - " + imageEvaluator.haarDetector.slidingWindowsSize + " sliding windows)");


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

            System.out.println("Total computing time for HaarDetector (CPU): " + imageEvaluator.computingTimeMS + "ms for " + i + " images 2048x1536 (" + r + " faces detected - " + imageEvaluator.haarDetector.slidingWindowsSize + " sliding windows)");
        }
    }

    public static void compareDetectFacesTime(int width, int height) {

        String images[] = {"face1.jpg", "got.jpeg", "face5.jpg", "groupe2.jpg", "groupe.jpg", "hardcore.jpg"};
        Conf.USE_CUDA =true;

        long timeCuda = 0;
        long timeCPU = 0;
        int nbSlidingWindowsCuda = 0;
        int nbSlidingWindowsCPU = 0;
        int nbFacesFoundCuda = 0;
        int nbFacesFoundCPU = 0;

        for (String img : images) {
            ImageHandler image = new ImageHandler("data/" + img);
            int maxDim = Math.max(image.getHeight(), image.getWidth());
            int minDim = Math.min(image.getHeight(), image.getWidth());
            int displacer = maxDim / 500;
            if (displacer < 1)
                displacer = 1;

            Conf.USE_CUDA =true;
            ImageEvaluator imageEvaluatorCUDA = new ImageEvaluator(width, height, image.getWidth(), image.getHeight(), displacer, displacer, 19, minDim, 1.25f);
            nbFacesFoundCuda = imageEvaluatorCUDA.getFaces(image, false).size();
            timeCuda = imageEvaluatorCUDA.computingTimeMS;
            nbSlidingWindowsCuda = imageEvaluatorCUDA.slidingWindows.size();
            imageEvaluatorCUDA.haarDetector.close();

            Conf.USE_CUDA =false;
            ImageEvaluator imageEvaluatorCPU = new ImageEvaluator(width, height, image.getWidth(), image.getHeight(), displacer, displacer, 19, minDim, 1.25f);
            nbFacesFoundCPU = imageEvaluatorCPU.getFaces(image, false).size();
            timeCPU = imageEvaluatorCPU.computingTimeMS;
            nbSlidingWindowsCPU = imageEvaluatorCPU.slidingWindows.size();
            imageEvaluatorCUDA.haarDetector.close();


            System.out.println("Size image: " + image.getWidth() + "*" + image.getHeight() + " ; CUDA time: " + timeCuda + "ms ; CUDA nb rectangles found: " + nbFacesFoundCuda +
                    " ; CPU time: " + timeCPU + "ms ; CPU rectangles found: " + nbFacesFoundCPU + " ; sliding windows: " + (nbSlidingWindowsCPU == nbSlidingWindowsCuda ? nbSlidingWindowsCPU : "error not equal!"));
        }

    }
}
