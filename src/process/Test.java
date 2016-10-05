package process;


import GUI.Display;
import GUI.ImageHandler;
import process.features.Face;

import java.util.ArrayList;
import java.util.HashMap;

import static java.lang.Math.log;

public class Test {
    public static double isFace(ArrayList<StumpRule>[] cascade, ArrayList<Float> tweaks, int[] exampleFeatureValues, int defaultLayerNumber, HashMap<Integer, Integer> neededHaarValues) {
        // Everything is a face if no layer is involved
        if (defaultLayerNumber == 0) {
            return 1;
        }
        int layerCount = defaultLayerNumber < 0 ? tweaks.size() : defaultLayerNumber;
        double confidence = 0;
        for(int layer = 0; layer < layerCount; layer++){
            double prediction = 0;
            int committeeSize = cascade[layer].size();
            for(int ruleIndex = 0; ruleIndex < committeeSize; ruleIndex++){
                StumpRule rule = cascade[layer].get(ruleIndex);
                int ftIndex = (int) rule.featureIndex;
                if (neededHaarValues != null) {
                    ftIndex = neededHaarValues.get(ftIndex);
                }
                double featureValue = (double)exampleFeatureValues[ftIndex];
                double vote = (featureValue > rule.threshold ? 1 : -1) * rule.toggle + tweaks.get(layer);
                if (rule.error == 0) {
                    if (ruleIndex == 0)
                        return vote;
                    else {
                        System.err.println("Find an invalid rule!");
                        System.exit(1);
                    }
                }
                prediction += vote * log((1.0d/rule.error) - 1);
            }
            confidence += prediction;
            if (prediction < 0)
                return prediction;
        }
        return confidence;
    }

    public static double isFace(ArrayList<ArrayList<StumpRule>> cascade, ArrayList<Float> tweaks, int[] exampleFeatureValues, int defaultLayerNumber, HashMap<Integer, Integer> neededHaarValues) {
        // Everything is a face if no layer is involved
        if (defaultLayerNumber == 0) {
            return 1;
        }
        int layerCount = defaultLayerNumber < 0 ? tweaks.size() : defaultLayerNumber;
        double confidence = 0;
        for(int layer = 0; layer < layerCount; layer++){
            double prediction = 0;
            int committeeSize = cascade.get(layer).size();
            for(int ruleIndex = 0; ruleIndex < committeeSize; ruleIndex++){
                StumpRule rule = cascade.get(layer).get(ruleIndex);
                int ftIndex = (int) rule.featureIndex;
                if (neededHaarValues != null) {
                    ftIndex = neededHaarValues.get(ftIndex);
                }
                double featureValue = (double)exampleFeatureValues[ftIndex];
                double vote = (featureValue > rule.threshold ? 1 : -1) * rule.toggle + tweaks.get(layer);
                if (rule.error == 0) {
                    if (ruleIndex == 0)
                        return vote;
                    else {
                        System.err.println("Find an invalid rule!");
                        System.exit(1);
                    }
                }
                prediction += vote * log((1.0d/rule.error) - 1);
            }
            confidence += prediction;
            if (prediction < 0)
                return prediction;
        }
        return confidence;
    }

    public static double isFace(ArrayList<ArrayList<StumpRule>> cascade, ArrayList<Float> tweaks, int[] exampleFeatureValues, int defaultLayerNumber) {
        return isFace(cascade, tweaks, exampleFeatureValues, defaultLayerNumber, null);
    }

    public static double isFace(ArrayList<StumpRule>[] cascade, ArrayList<Float> tweaks, int[] exampleFeatureValues, int defaultLayerNumber) {
        return isFace(cascade, tweaks, exampleFeatureValues, defaultLayerNumber, null);
    }

    public static void evaluateImage(String img, ImageEvaluator imageEvaluator, boolean postProcess) {
        imageEvaluator.computingTimeMS = 0;
        ImageHandler image = new ImageHandler(img);
        ArrayList<Face> rectangles = imageEvaluator.getFaces(image, postProcess);
        System.out.println("Found " + rectangles.size() + " faces rectangle that contains a face");
        System.out.println("Time spent for this image: " + imageEvaluator.computingTimeMS + "ms");
        System.out.println("Sliding Windows: " + imageEvaluator.slidingWindows.size());
        image.drawFaces(rectangles);
        Display.drawImage(image.getBufferedImage());
    }

    public static void soutenance(int w, int h, ArrayList<ArrayList<StumpRule>> cascade, ArrayList<Float> tweaks) {
        ImageEvaluator evaluatorGOT = new ImageEvaluator(w, h, 200, 200, 1, 1, 16, 30, 1.21f, cascade, tweaks);
//        ImageEvaluator evaluatorBE = new ImageEvaluator(w, h, 200, 200, 2, 2, 40, 41, 1.25f, cascade, tweaks);
//        ImageEvaluator evaluatorGOT2 = new ImageEvaluator(w, h, 600, 600, 10, 10, 85, 86, 1.25f, cascade, tweaks);
        ImageEvaluator evaluator100 = new ImageEvaluator(w, h, 100, 100, 1, 1, 30, 60, 1.25f, cascade, tweaks);
        ImageEvaluator evaluator300 = new ImageEvaluator(w, h, 300, 300, 4, 4, 120, 121, 1.25f, cascade, tweaks);
        ImageEvaluator evaluator640 = new ImageEvaluator(w, h, 640, 436, 3, 3, 28, 35, 1.25f, cascade, tweaks);
        ImageEvaluator evaluator500 = new ImageEvaluator(w, h, 500, 281, 4, 4, 65, 78, 1.25f, cascade, tweaks);

        evaluateImage("data/got.jpeg", evaluatorGOT, true);
//        evaluateImage("data/serie.jpg", evaluatorGOT, true);
//        evaluateImage("data/beatles.jpg", evaluatorBE, true);
//        evaluateImage("data/got2.jpg", evaluatorGOT2, true);
        evaluateImage("data/baelish.jpg", evaluator100, true);
        evaluateImage("data/face1.jpg", evaluator100, true);
        evaluateImage("data/tesla.jpg", evaluator100, true);
        evaluateImage("data/man.jpg", evaluator100, true);
        evaluateImage("data/land.jpeg", evaluator100, true);
        evaluateImage("data/face5.jpg", evaluator300, true);
        evaluateImage("data/fusia.jpg", evaluator640, true);
        evaluateImage("data/groupe2.jpg", evaluator500, true);
    }
}
