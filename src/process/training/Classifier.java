package process.training;

import GUI.ImageHandler;
import javafx.util.Pair;
import jeigen.DenseMatrix;
import process.Conf;
import process.DecisionStump;
import process.features.FeatureExtractor;
import utils.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import static java.lang.Math.log;
import static javafx.application.Platform.exit;
import static utils.Utils.countFiles;
import static utils.Utils.streamImageHandler;

public class Classifier {
    /**
     * A Classifier maps an observation to a label valued in a finite set.
     * f(HaarFeaturesOfTheImage) = -1 or 1
     * CAUTION: the training only works on same-sized images!
     * This Classifier uses what is know as Strong & Weak classifiers
     * A Weak classifier is just a simple basic classifier, it could be Binary, Naive Bayes or anything else.
     * The only need for a weak classifier is to return results with a success rate > 0.5, which is actually better than random.
     * The combination of all these Weak classifier create of very good classifier, which is called the Strong classifier.
     *
     * This is what we call Adaboosting.
     *
     */

    /* --- CONSTANTS FOR TRAINING --- */
    private static final int POSITIVE = 0;              // some convention
    private static final int NEGATIVE = 1;              // some convention
    private static final float TWEAK_UNIT = 1e-2f;      // initial tweak unit
    private static final double MIN_TWEAK = 1e-5;       // tweak unit cannot go lower than this
    private static final double GOAL = 1e-7;

    private static int adaboostPasses = 0;
    /* --- CLASS VARIABLES --- */
    private final int countTrainPos;
    private final int countTrainNeg;
    private final int trainN;

    private final int countTestPos;
    private final int countTestNeg;
    private final int testN;

    private final String train_dir;
    private final String test_dir;

    private final int width;
    private final int height;

    private boolean computed = false;

    private ArrayList<Integer> layerMemory;
    private ArrayList<DecisionStump>[] cascade;
    private ArrayList<Float> tweaks;

    private DenseMatrix weightsTrain[];
    private DenseMatrix weightsTest[];
    private DenseMatrix labelsTrain[];
    private DenseMatrix labelsTest[];

    public Classifier(String train_dir, String test_dir, int width, int height) {
        this.train_dir = train_dir;
        this.test_dir = test_dir;

        this.width = width;
        this.height = height;

        countTrainPos = countFiles(train_dir + "/faces", ".png");
        countTrainNeg = countFiles(train_dir + "/non-faces", ".png");
        trainN = countTrainNeg + countTrainNeg;

        countTestPos = countFiles(test_dir + "/faces", ".png");
        countTestNeg = countFiles(test_dir + "/non-faces", ".png");
        testN = countTestPos + countTestNeg;
    }

    private void predictLabel(int round, int N, float decisionTweak, DenseMatrix prediction, boolean onlyMostRecent) {
        /**
         * prediction = Vector (Matrix< 1,n >)
         */
        int committeeSize = cascade[round].size();
        DenseMatrix memberVerdict = new DenseMatrix(committeeSize, N);
        DenseMatrix memberWeight = new DenseMatrix(1, committeeSize);

        int start = onlyMostRecent ? committeeSize - 1 : 0;

        for (int member = start; member < committeeSize; member++) {
            if (cascade[round].get(member).getError() == 0 && member != 0) {
                System.err.println("Boosting Error Occurred!");
                exit();
            }

            // 0.5 does not count here
            // if member's weightedError is zero, member weight is nan, but it won't be used anyway
            memberWeight.set(member, log((1.0 / cascade[round].get(member).getError()) - 1));
            int featureId = cascade[round].get(member).getFeatureIndex();
            for (int i = 0; i < N; i++) {
                // TODO
//                int exampleIndex = getExampleIndex(featureId, i);
//                memberVerdict.set(member, exampleIndex, (getExampleFeature(featureId, i) > committee.get(member).getThreshold() ? 1 : -1) * committee.get(member).getToggle()) + decisionTweak;
            }
        }

        DenseMatrix finalVerdict = memberWeight.mmul(memberVerdict);
        for (int exampleIndex = 0; exampleIndex < N; exampleIndex++)
            prediction.set(1, exampleIndex, finalVerdict.get(1, exampleIndex) > 0 ? 1 : -1);

    }


    private ArrayList<DecisionStump> adaboost(int round, int N) {
        /**
         * Strong classifier based on multiple weak classifiers.
         * Here, weak classifier are called "Stumps", see: https://en.wikipedia.org/wiki/Decision_stump
         */

        ArrayList<DecisionStump> committee = new ArrayList<>();

        // TODO : when we have the list of list of features and the weights
        //DecisionStump bestDS = DecisionStump.bestStump(features, weights);
        //committee.add(bestDS);

        DenseMatrix prediction = new DenseMatrix(1, N);
        predictLabel(round, N, 0, prediction, true);
        adaboostPasses++;

        boolean werror = false;

        if (werror) {
            // Update weights
            // Update pos and neg weights
            // Update positiveTotalWeights

        } else {
            // Training ends, just return

        }

        System.out.println("Adaboost passes : " + adaboostPasses);
        return committee;
    }

    // TODO: à mettre ds une fonction
    // Pour chaque feature (60 000)
    //   vector<pair<valeur-de-la-feature, l'index de l'exemple (image)>> ascendingFeatures;
    //   Pour chaque exemple
    //     ascendingFeatures.add(<valeur-de-cette-feature-pour-cet-example, index-de-l'exemple>)
    //   trier ascendingFeatures en fonction de pair.first
    //   Write sur disque:
    //      * OrganizedFeatures (à l'index de la feature actuelle le ascendingFeatures.first en entier) tmp/training
    //      * OrganizedSample (à l'index de la feature actuelle le ascendingFeatures.second en entier)

    // fileList computed in main
    public static void organizeFeatures(String[] fileList) {
        ImageHandler imageHandler = new ImageHandler("data/testset/faces/face00001.png");
        ArrayList<ArrayList<Integer>> features = imageHandler.getFeatures();

        int nb_features = 0;
        for (ArrayList<Integer> list : features)
            nb_features += list.size();

        for (int i = 0; i < nb_features; i++) {
            ArrayList<Pair<Integer, Integer>> ascendingFeatures = new ArrayList<>();
            int j = 0;
            for (String s : fileList) {
                ImageHandler tmp_image = new ImageHandler(s);
                ArrayList<ArrayList<Integer>> tmp_features = tmp_image.getFeatures();
                ArrayList<Integer> allFeat = tmp_features.get(0);
                allFeat.addAll(tmp_features.get(1));
                allFeat.addAll(tmp_features.get(2));
                allFeat.addAll(tmp_features.get(3));
                allFeat.addAll(tmp_features.get(4));
                ascendingFeatures.add(new Pair<>(allFeat.get(i), j));
            }

            // Sort ascending features by first arg of the pair
            ascendingFeatures.sort((o1, o2) -> o1.getKey() < o2.getKey() ? -1 : o1.getKey() > o2.getKey() ? 1 : 0);

            // Write on disk but does not write over the file (if recomputed the file needs to be erased by hand)
            try {
                // true to write after the file
                FileWriter writer1 = new FileWriter(Conf.ORGANIZED_FEATURES, true);
                FileWriter writer2 = new FileWriter(Conf.ORGANIZED_SAMPLE, true);
                for (Pair pair : ascendingFeatures) {
                    writer1.write(pair.getKey().toString() + ";");
                    writer2.write(pair.getValue().toString() + ";");
                }
                writer1.write(System.lineSeparator());
                writer2.write(System.lineSeparator());

                writer1.close();
                writer2.close();

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void computeFeatures(String faces_dir, String nonfaces_dir, int countPos, int countNeg, int N, int width, int height) {
        /**
         * In order to avoid excessive memory usage, this training temporary stores metadata on disk.
         *
         * width: Width of all training images
         * height : Height of all training images
         */
        System.out.println("Pre-computing features for:");
        System.out.println("  - " + faces_dir);
        System.out.println("  - " + nonfaces_dir);

        long startTime = System.currentTimeMillis();
        int count = 0;
        count += Utils.computeHaar(new File(faces_dir));
        count += Utils.computeHaar(new File(nonfaces_dir));
        long elapsedTimeMS = (new Date()).getTime() - startTime;
        System.out.println("Statistics:");
        System.out.println("  - Elapsed time: " + elapsedTimeMS / 1000 + "s");
        System.out.println("  - Images computed: " + count);
        System.out.println("  - image/seconds: " + count / (elapsedTimeMS / 1000));


        // FIXME: Do we need all that code?
//        Iterable<ImageHandler> positives = streamImageHandler(faces_dir, ".png");
//        Iterable<ImageHandler> negatives = streamImageHandler(nonfaces_dir, ".png");
//
//        double averageWeightPos = 0.5 / countPos;
//        double averageWeightNeg = 0.5 / countNeg;
//
//        DenseMatrix weights = new DenseMatrix(N, 1); // weight vector for all examples involved
//        DenseMatrix labels = new DenseMatrix(N, 1); // -1 = negative | 1 = positive example
//
//        // Init weights & labels
//        for (int i = 0; i < N; i++) {
//            labels.set(i, 0, i < countPos ? 1 : -1); // labels = [positives then negatives] = [1 1 ..., -1 -1 ...]
//            weights.set(i, 0, i < countPos ? averageWeightPos : averageWeightNeg);
//        }
//
//        long featuresCount = FeatureExtractor.countAllFeatures(width, height);
    }

    // p141 in paper?
    private float[] calcEmpiricalError(int round, int N, int countPos, DenseMatrix labels) {
        // STATE: OK & CHECKED 16/26/08
        float res[] = new float[2];

        int nFalsePositive = 0;
        int nFalseNegative = 0;

        DenseMatrix verdicts = DenseMatrix.ones(1, N);
        DenseMatrix layerPrediction = DenseMatrix.zeros(1, N);

        for (int layer = 0; layer < round + 1; layer++) {
            predictLabel(round, N, tweaks.get(round), layerPrediction, false);
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

    /**
     * Algorithm 10 from the original paper
     */
    public void attentionalCascade(int round, int committeeSizeGuide,
                                   float overallTargetDetectionRate, float overallTargetFalsePositiveRate) {
        // STATE: OK & CHECKED 16/26/08
        boolean layerMissionAccomplished = false;

        while (!layerMissionAccomplished) {

            // Run algorithm N°6 to produce a classifier
            cascade[round] = adaboost(round, trainN);

            boolean overSized = cascade[round].size() > committeeSizeGuide;
            boolean finalTweak = overSized;

            int tweakCounter = 0;

            int[] oscillationObserver = new int[2];
            float tweak = 0;
            if (finalTweak)
                tweak = -1;
            float tweakUnit = TWEAK_UNIT;
            float ctrlFalsePositive, ctrlDetectionRate, falsePositive, detectionRate;

            while (Math.abs(tweak) < 1.1) {
                tweaks.set(round, tweak);

                float tmp[] = calcEmpiricalError(round, trainN, countTrainPos, labelsTrain[round]);
                ctrlFalsePositive = tmp[0];
                ctrlDetectionRate = tmp[1];

                tmp = calcEmpiricalError(round, testN, countTestPos, labelsTest[round]);
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
    }

    public void train(float overallTargetDetectionRate, float overallTargetFalsePositiveRate,
                      float targetDetectionRate, float targetFalsePositiveRate) {

        // TODO : mettre a jour positiveTotalWeights de DecisionStump

        if (computed) {
            System.out.println("Training already done!");
            return;
        }

        // FIXME: What is that?
        layerMemory = new ArrayList<>();

        // Compute all features for train & test set
        //computeFeatures(train_dir + "/faces", train_dir + "/non-faces", countTrainPos, countTrainNeg, trainN, width, height);
        //computeFeatures(test_dir + "/faces", test_dir + "/non-faces", countTestPos, countTestNeg, testN, width, height);

        // Estimated number of rounds needed
        int boostingRounds = (int) (Math.ceil(Math.log(overallTargetFalsePositiveRate) / Math.log(targetFalsePositiveRate)) + 20);
        System.out.println("Boosting rounds : " + boostingRounds);

        // Initialization
        tweaks = new ArrayList<>(boostingRounds);
        cascade = new ArrayList[boostingRounds];
        for (int i = 0; i < boostingRounds; i++)
            cascade[i] = new ArrayList<>();

        // Init weights & labels
        weightsTrain = new DenseMatrix[boostingRounds];
        weightsTest = new DenseMatrix[boostingRounds];
        labelsTrain = new DenseMatrix[boostingRounds];
        labelsTest = new DenseMatrix[boostingRounds];
        for (int i = 0; i < boostingRounds; i++) {
            weightsTest[i] = new DenseMatrix(testN, 1);
            labelsTest[i] = new DenseMatrix(testN, 1);
            weightsTrain[i] = new DenseMatrix(trainN, 1);
            labelsTrain[i] = new DenseMatrix(trainN, 1);
        }

        double accumulatedFalsePositive = 1;

        // Training: run Cascade until we arrive to a certain wanted rate of success
        for (int round = 0; round < boostingRounds && accumulatedFalsePositive > GOAL; round++) {

            int committeeSizeGuide = Math.min(20 + round * 10, 200);
            System.out.println("CommitteeSizeGuide = " + committeeSizeGuide);

            attentionalCascade(round, committeeSizeGuide, overallTargetDetectionRate, overallTargetFalsePositiveRate);


            // -- Display results for this round --

            System.out.println("Layer " + round + 1 + " done!");

            //layerMemory.add(trainSet.committee.size());
            layerMemory.add(cascade[round].size());
            System.out.println("The committee size is " + cascade[round].size());

            float detectionRate, falsePositive;
            float[] tmp = calcEmpiricalError(round, trainN, countTrainPos, labelsTrain[round]);
            falsePositive = tmp[0];
            detectionRate = tmp[1];
            System.out.println("The current tweak " + tweaks.get(round) + " has falsePositive " + falsePositive + " and detectionRate " + detectionRate + " on the training examples.");

            tmp = calcEmpiricalError(round, testN, countTestPos, labelsTest[round]);
            falsePositive = tmp[0];
            detectionRate = tmp[1];
            System.out.println("The current tweak " + tweaks.get(round) + " has falsePositive " + falsePositive + " and detectionRate " + detectionRate + " on the validation examples.");

            accumulatedFalsePositive *= falsePositive;
            System.out.println("Accumulated False Positive Rate is around " + accumulatedFalsePositive);

            //record the boosted rule into a target file
            // TODO: recordRule(target.toFile, cascade[round], round == 0, round == boostingRounds - 1 || accumulatedFalsePositive < GOAL);
        }
        // TODO: record layerMemory


        // Serialize training
        computed = true;
    }
}