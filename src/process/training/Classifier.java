package process.training;

import GUI.ImageHandler;
import jeigen.DenseMatrix;
import process.Conf;
import process.DecisionStump;
import process.features.FeatureExtractor;
import process.features.FeaturesSerializer;

import java.util.*;

import static java.lang.Math.floor;
import static java.lang.Math.log;
import static javafx.application.Platform.exit;
import static process.features.FeatureExtractor.computeFeatures;
import static process.features.FeatureExtractor.computeFeaturesImages;
import static utils.Utils.countFiles;
import static utils.Utils.streamImageHandler;

public class Classifier {
    /**
     * A Classifier maps an observation to a label valued in a finite set.
     * f(HaarFeaturesOfTheImage) = -1 or 1
     * <p>
     * CAUTION: the training only works on same-sized images!
     * <p>
     * This Classifier uses what is know as Strong & Weak classifiers
     * A Weak classifier is just a simple basic classifier, it could be Binary, Naive Bayes or anything else.
     * The only need for a weak classifier is to return results with a success rate > 0.5, which is actually better than random.
     * The combination of all these Weak classifier create of very good classifier, which is called the Strong classifier.
     */

    private static final int POSITIVE = 0;        //some convention
    private static final int NEGATIVE = 1;    //some convention
    private static final float TWEAK_UNIT = 1e-2f;    //initial tweak unit
    private static final double MIN_TWEAK = 1e-5;    //tweak unit cannot go lower than this
    private static final double GOAL = 1e-7;

    // Make it static ??
    private static void predictLabel(ArrayList<DecisionStump> committee, int n, float decisionTweak, DenseMatrix prediction) {
        /**
         * prediction = Vector (Matrix< 1,n >)
         */
        int committeeSize = committee.size();
        DenseMatrix memberVerdict = new DenseMatrix(committeeSize, n);
        DenseMatrix memberWeight = new DenseMatrix(1, committeeSize);

        for (int member = committeeSize - 1; member < committeeSize; member++) {
            if (committee.get(member).getError() == 0 && member != 0) {
                System.err.println("Boosting Error Occurred!");
                exit();
            }

            // 0.5 does not count here
            // if member's weightedError is zero, member weight is nan, but it won't be used anyway
            memberWeight.set(member, log((1.0 / committee.get(member).getError()) - 1));
            UUID featureId = committee.get(member).getFeatureId();
            for (int i = 0; i < n; i++) {
                // TODO
//                int exampleIndex = getExampleIndex(featureId, i);
//                memberVerdict.set(member, exampleIndex, (getExampleFeature(featureId, i) > committee.get(member).getThreshold() ? 1 : -1) * committee.get(member).getToggle()) + decisionTweak;
            }
        }

        DenseMatrix finalVerdict = memberWeight.mmul(memberVerdict);
        for (int exampleIndex = 0; exampleIndex < n; exampleIndex++)
            prediction.set(1, exampleIndex, finalVerdict.get(1, exampleIndex) > 0 ? 1 : -1);

    }

    // MAke it static ?
    private static ArrayList<DecisionStump> adaboost(int N) {
        ArrayList<DecisionStump> committee = new ArrayList<>();
        ArrayList<Long> weights = new ArrayList<>();

        // TODO
//            DecisionStump bestDS = DecisionStump.bestStump();
//            committee.add(bestDS);


        DenseMatrix prediction = new DenseMatrix(1, N);
        predictLabel(committee, N, 0, prediction);


        boolean werror = false;


        if (werror) {
            // Update weights

        } else {
            // Training ends, just return

        }

        return committee;
    }

    public static void cascadeClassify(String faces_dir, String nonfaces_dir, int width, int height) {
        /**
         * In order to avoid excessive memory usage, this training temporary stores metadata on disk.
         *
         * width: Width of all training images
         * height : Height of all training images
         */
        Iterable<ImageHandler> positives = streamImageHandler(faces_dir);
        Iterable<ImageHandler> negatives = streamImageHandler(nonfaces_dir);

        int countPos = countFiles(faces_dir);
        int countNeg = countFiles(nonfaces_dir);
        int N = countPos + countNeg;

        double averageWeightPos = 0.5 / countPos;
        double averageWeightNeg = 0.5 / countNeg;

        DenseMatrix weights = new DenseMatrix(N, 1); // weight vector for all examples involved
        DenseMatrix labels = new DenseMatrix(N, 1); // -1 = negative | 1 = positive example

        // Init weights & labels
        for (int i = 0; i < N; i++) {
            labels.set(i, 0, i < countPos ? 1 : -1); // labels = [positives then negatives] = [1 1 ..., -1 -1 ...]
            weights.set(i, 0, i < countPos ? averageWeightPos : averageWeightNeg);
        }

        long featuresCount = FeatureExtractor.countAllFeatures(width, height);

        // TODO: à mettre ds une fonction
        // Pour chaque feature (60 000)
        //   vector<pair<valeur-de-la-feature, l'index de l'exemple (image)>> ascendingFeatures;
        //   Pour chaque exemple
        //     ascendingFeatures.add(<valeur-de-cette-feature-pour-cet-example, index-de-l'exemple>)
        //   trier ascendingFeatures en fonction de pair.first
        //   Write sur disque:
        //      * OrganizedFeatures (à l'index de la feature actuelle le ascendingFeatures.first en entier)
        //      * OrganizedSample (à l'index de la feature actuelle le ascendingFeatures.second en entier)

        // Get already computed feature values if any
        //HashMap<String, ArrayList<Integer>> result = FeaturesSerializer.fromDisk(Conf.TRAIN_FEATURES);

        /*for (ImageHandler image : positives) {
            if (image.getWidth() == width && image.getHeight() == height) {
                result.putIfAbsent(image.getFilePath(), computeFeatures(image));
            }
        }*/
        /*for (ImageHandler image : negatives) {
            if (image.getWidth() == width && image.getHeight() == height) {
                result.putIfAbsent(image.getFilePath(), computeFeatures(image));
            }
        }*/
        //FeatureExtractor.haarExtractor.freeCuda();
        // Compute Haar-features of all examples
        //computeFeaturesImages(positives, width, height, result);
        //computeFeaturesImages(negatives, width, height, result);

        //FeaturesSerializer.toDisk(result, Conf.TRAIN_FEATURES);
    }

    public static void train(String train_dir, String test_dir, int width, int height,
                             float overallTargetDetectionRate, float overallTargetFalsePositiveRate, float targetDetectionRate, float targetFalsePositiveRate) {

        int countPos = countFiles(train_dir + "/face-png");
        int countNeg = countFiles(train_dir + "/non-face-png");
        int N = countPos + countNeg;

        ArrayList<Integer> layerMemory = new ArrayList<>();

        int boostingRounds = (int) (Math.ceil(Math.log(overallTargetFalsePositiveRate) / Math.log(targetFalsePositiveRate)) + 20);
        System.out.println("Boosting ropunds : " + boostingRounds);

        ArrayList<DecisionStump>[] cascade = new ArrayList[boostingRounds];
        float[] tweaks = new float[boostingRounds];

        double accumulatedFalsePositive = 1;

        DenseMatrix weightsTrain[] = new DenseMatrix[boostingRounds];
        DenseMatrix weightsTest[] = new DenseMatrix[boostingRounds];
        DenseMatrix labelsTrain[] = new DenseMatrix[boostingRounds];
        DenseMatrix labelsTest[] = new DenseMatrix[boostingRounds];
        for (int i = 0; i < boostingRounds; i++) {
            weightsTest[i] = new DenseMatrix(N, 1);
            labelsTest[i] = new DenseMatrix(N, 1);
            weightsTrain[i] = new DenseMatrix(N, 1);
            labelsTrain[i] = new DenseMatrix(N, 1);
        }

        for (int round = 0; round < boostingRounds && accumulatedFalsePositive > GOAL; round++) {

            cascadeClassify(train_dir + "/face-png", train_dir + "/non-face-png", width, height);
            int committeeSizeGuide = Math.min(20 + round * 10, 200);
            boolean layerMissionAccomplished = false;

            while (!layerMissionAccomplished) {
                cascade[round] = adaboost(N);

                boolean overSized = cascade[round].size() > committeeSizeGuide ? true : false;
                boolean finalTweak = overSized;

                int tweakCounter = 0;

                int[] oscillationObserver = new int[2];
                float tweak = 0;
                if (finalTweak)
                    tweak = -1;
                float tweakUnit = TWEAK_UNIT;
                float ctrlFalsePositive, ctrlDetectionRate, falsePositive, detectionRate;

                while (Math.abs(tweak) < 1.1) {
                    tweaks[round] = tweak;

                    // Update the 4 floats : ctrlFalsePositive, ctrlDetectionRate, falsePositive, detectionRate
                    float tmp[] = calcEmpiricalError(cascade[round], tweaks, round + 1, N, countPos, labelsTest[round]);
                    ctrlFalsePositive = tmp[0];
                    ctrlDetectionRate = tmp[1];

                    tmp = calcEmpiricalError(cascade[round], tweaks, round + 1, N, countPos, labelsTrain[round]);
                    falsePositive = tmp[0];
                    detectionRate = tmp[1];


                    float worstFalsePositive = Math.max(falsePositive, ctrlFalsePositive);
                    float worstDetectionRate = Math.min(detectionRate, ctrlDetectionRate);

                    if (finalTweak) {
                        if (worstDetectionRate >= 0.99) {
                            System.out.println(" final tweak settles to " + tweak);
                            break;
                        } else {
                            tweak += TWEAK_UNIT;
                            continue;
                        }
                    }

                    if (worstDetectionRate >= overallTargetDetectionRate && worstFalsePositive <= overallTargetFalsePositiveRate) {
                        layerMissionAccomplished = true;
                        break;
                    } else if (worstDetectionRate >= overallTargetDetectionRate && worstFalsePositive > overallTargetFalsePositiveRate) {
                        tweak -= tweakUnit;
                        tweakCounter++;
                        oscillationObserver[tweakCounter % 2] = -1;
                    } else if (worstDetectionRate < overallTargetDetectionRate && worstFalsePositive <= overallTargetFalsePositiveRate) {
                        tweak += tweakUnit;
                        tweakCounter++;
                        oscillationObserver[tweakCounter % 2] = 1;
                    } else {
                        finalTweak = true;
                        System.out.println("INFO: no way out at this point. tweak goes from " + tweak);
                        continue;
                    }

                    if (!finalTweak && tweakCounter > 1 && oscillationObserver[0] + oscillationObserver[1] == 0) {
                        tweakUnit /= 2;

                        System.out.println("backtracked at " + tweakCounter + " ! Modify tweakUnit to " + tweakUnit);

                        if (tweakUnit < MIN_TWEAK) {
                            finalTweak = true;
                            System.out.println("tweakUnit too small. tweak goes from " + tweak);
                        }
                    }
                }
                if (overSized)
                    break;
            }
            System.out.println("Layer " + round + 1 + " done!");

            //layerMemory.add(trainSet.committee.size());
            layerMemory.add(cascade[round].size());
            System.out.println("The committee size is " + cascade[round].size());

            float detectionRate, falsePositive;
            float[] tmp = calcEmpiricalError(cascade[round], tweaks, round + 1, N, countPos, labelsTrain[round]);
            falsePositive = tmp[0];
            detectionRate = tmp[1];


            calcEmpiricalError(cascade, tweaks, round + 1, falsePositive, detectionRate, true);
            calcEmpiricalError(cascade, tweaks, round + 1, falsePositive, detectionRate, true);
            System.out.println("The current tweak " + tweaks[round] + " has falsePositive " + falsePositive + " and detectionRate " + detectionRate + " on the validation examples.");
            System.out.println("Target:  falsePositiveTarget " + targetFalsePositiveRate + " detectionRateTarget " + targetDetectionRate);

            accumulatedFalsePositive *= falsePositive;
            System.out.println("Accumulated False Positive Rate is around " + accumulatedFalsePositive);

        }
    }

    // p141 in paper?
    private static float[] calcEmpiricalError(ArrayList<DecisionStump> committee,
                                       float[] tweaks, int layerCount, int N, int countPos,
                                       DenseMatrix labels) {

        float res[] = new float[2];

        int nFalsePositive = 0;
        int nFalseNegative = 0;

        DenseMatrix verdicts = DenseMatrix.ones(1, N);
        DenseMatrix layerPrediction = DenseMatrix.zeros(1, N);

        for (int layer = 0; layer < layerCount; layer++) {
            predictLabel(committee, N, tweaks[layer], layerPrediction);
            verdicts = verdicts.min(layerPrediction);
        }

        DenseMatrix agree = labels.mmul(verdicts.t());
        for (int exampleIndex = 0; exampleIndex < N; exampleIndex++) {
            if (agree.get(0, exampleIndex) < 0) {
                if (exampleIndex < countPos) {
                    nFalseNegative += 1;
                } else {
                    nFalsePositive += 1;
                }
            }
        }

        res[0] = nFalsePositive / (float) (N - countPos);
        res[1] = 1 - nFalseNegative / (float) countPos;

        return res;
    }

}