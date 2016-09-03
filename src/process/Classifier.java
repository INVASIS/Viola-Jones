package process;

import jeigen.DenseMatrix;
import utils.Serializer;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.*;

import static java.lang.Math.log;
import static java.lang.Math.negateExact;
import static process.features.FeatureExtractor.*;
import static utils.Utils.*;

public class Classifier {
    /**
     * A Classifier maps an observation to a label valued in a finite set.
     * f(HaarFeaturesOfTheImage) = -1 or 1
     * CAUTION: the training only works on same-sized images!
     * This Classifier uses what is know as Strong & Weak classifiers
     * A Weak classifier is just a simple basic classifier, it could be Binary, Naive Bayes or anything else.
     * The only need for a weak classifier is to return results with a success rate > 0.5, which is actually better than random.
     * The combination of all these Weak classifier create of very good classifier, which is called the Strong classifier.
     * <p>
     * This is what we call Adaboosting.
     */

    /* --- CONSTANTS FOR TRAINING --- */
    private static final int POSITIVE = 0;              // some convention
    private static final int NEGATIVE = 1;              // some convention
    private static final float TWEAK_UNIT = 1e-2f;      // initial tweak unit
    private static final double MIN_TWEAK = 1e-5;       // tweak unit cannot go lower than this
    private static final double GOAL = 1e-7;

    private static int adaboostPasses = 0;

    /* --- CLASS VARIABLES --- */
    private int countTrainPos;
    private int countTrainNeg;
    private int trainN;

    private long featureCount;

    private int countTestPos;
    private int countTestNeg;
    private int testN;

    private String train_dir;
    private ArrayList<String> train_faces;
    private ArrayList<String> train_nonfaces;
    private String test_dir;

    private final int width;
    private final int height;

    private boolean computed = false;

    private ArrayList<Integer> layerMemory;
    private ArrayList<DecisionStump>[] cascade;
    private ArrayList<Float> tweaks;

    private ArrayList<BigDecimal> weightsTrain;
    private ArrayList<BigDecimal> weightsTest;
    private DenseMatrix labelsTrain;
    private DenseMatrix labelsTest;

    BigDecimal totalWeightPos; // total weight received by positive examples currently
    BigDecimal totalWeightNeg; // total weight received by negative examples currently

    BigDecimal minWeight; // minimum weight among all weights currently
    BigDecimal maxWeight; // maximum weight among all weights currently

    public Classifier(int width, int height) {
        this.width = width;
        this.height = height;

        this.featureCount = countAllFeatures(width, height);
        System.out.println("Feature count for " + width + "x" + height + ": " + featureCount);
    }

    private void predictLabel(int round, int N, float decisionTweak, DenseMatrix prediction, boolean onlyMostRecent) {
        // prediction = Matrix<int, 1,n > -> To be filled here

        int committeeSize = cascade[round].size();
        DenseMatrix memberVerdict = new DenseMatrix(committeeSize, N);
        DenseMatrix memberWeight = new DenseMatrix(1, committeeSize);

        onlyMostRecent = committeeSize == 1 || onlyMostRecent;

        int start = onlyMostRecent ? committeeSize - 1 : 0;

        for (int member = start; member < committeeSize; member++) {
            if (cascade[round].get(member).error.compareTo(new BigDecimal(0)) == 0 && member != 0) { // commitee[member].error == 0
                System.err.println("Boosting Error Occurred!");
                System.exit(1);
            }

            // 0.5 does not count here
            // if member's weightedError is zero, member weight is nan, but it won't be used anyway
            if (!Objects.equals(cascade[round].get(member).error, BigDecimal.ZERO))
                memberWeight.set(member, log((BigDecimal.ONE.divide(cascade[round].get(member).error, MathContext.DECIMAL128)).subtract(new BigDecimal(1)).doubleValue())); // log((1.0d / commitee[member].error) - 1) // FIXME: .doubleValue() ok or not?
            else
                memberWeight.set(member, Double.MAX_VALUE - 1);
            long featureIndex = cascade[round].get(member).featureIndex;
            for (int i = 0; i < N; i++) {
                int exampleIndex = getExampleIndex(featureIndex, i, N);
                memberVerdict.set(member, exampleIndex, ((getExampleFeature(featureIndex, i, N) > cascade[round].get(member).threshold ? 1 : -1) * cascade[round].get(member).toggle) + decisionTweak);
            }
        }
        if (!onlyMostRecent) {
            DenseMatrix finalVerdict = memberWeight.mul(memberVerdict);
            for (int i = 0; i < N; i++)
                prediction.set(0, i, finalVerdict.get(1, i) > 0 ? 1 : -1);
        }
        else {
            for (int i = 0; i < N; i++)
                prediction.set(0, i, memberVerdict.get(start, i) > 0 ? 1 : -1);
        }

    }

    /**
     * Strong classifier based on multiple weak classifiers.
     * Here, weak classifier are called "Stumps", see: https://en.wikipedia.org/wiki/Decision_stump
     */
    private void adaboost(int round) {
        System.out.println();
        // STATE: OK & CHECKED 16/31/08

        // The result to be filled & returned
        ArrayList<DecisionStump> committee = new ArrayList<>();

        DecisionStump bestDS = DecisionStump.bestStump(labelsTrain, weightsTrain, featureCount, trainN, totalWeightPos, totalWeightNeg, minWeight);
        committee.add(bestDS);
        cascade[round] = committee;
        adaboostPasses++;

        DenseMatrix prediction = new DenseMatrix(1, trainN);
        predictLabel(round, trainN, 0, prediction, true);

        DenseMatrix agree = labelsTrain.mul(prediction); // FIXME
        ArrayList<BigDecimal> weightUpdate = new ArrayList<>(trainN);

        boolean werror = false;

        for (int i = 0; i < trainN; i++) {
            if (agree.get(0, i) < 0) {
                weightUpdate.add(new BigDecimal(1).divide(bestDS.error, MathContext.DECIMAL128).subtract(new BigDecimal(1))); // 1.0d / bestDS.error - 1
                werror = true;
            } else {
                weightUpdate.add(new BigDecimal(1));
            }
        }

        //update weights only if there is an error
        if (werror) {
            for (int i = 0; i < trainN; i++)
                weightsTrain.set(i, weightsTrain.get(i).multiply(weightUpdate.get(i))); // <=> weightsTrain = weightsTrain.mul(weightUpdate)
            BigDecimal sum = new BigDecimal(0);
            for (int i = 0; i < trainN; i++)
                sum = sum.add(weightsTrain.get(i));
            BigDecimal sumPos = new BigDecimal(0);
            for (int i = 0; i < trainN; i++) {
                BigDecimal newVal = weightsTrain.get(i).divide(sum, MathContext.DECIMAL128);
                weightsTrain.set(i, newVal);
                sumPos = sumPos.add(newVal);
            }
            totalWeightPos = sumPos;
            totalWeightNeg = new BigDecimal(1).subtract(sumPos); // 1 - sumPos

            minWeight = weightsTrain.get(0);
            maxWeight = weightsTrain.get(0);
            for (int i = 1; i < trainN; i++) {
                BigDecimal currentVal = weightsTrain.get(i);
                if (minWeight.compareTo(currentVal) == 1) // minWeight > currentVal
                    minWeight = currentVal;
                if (maxWeight.compareTo(currentVal) == -1) // maxWeight < currentVal
                    maxWeight = currentVal;
            }
        }

        System.out.println("Adaboost passes : " + adaboostPasses);
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

        DenseMatrix agree = labels.mul(verdicts);
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
    private void attentionalCascade(int round, float overallTargetDetectionRate, float overallTargetFalsePositiveRate) {
        // STATE: OK & CHECKED 16/31/08

        int committeeSizeGuide = Math.min(20 + round * 10, 200);
        System.out.println("    - CommitteeSizeGuide = " + committeeSizeGuide);

        boolean layerMissionAccomplished = false;
        while (!layerMissionAccomplished) {

            // Run algorithm N°6 (adaboost) to produce a classifier (which is in fact a committee == ArrayList<DecisionStump>)
            adaboost(round);

            boolean overSized = cascade[round].size() > committeeSizeGuide;
            boolean finalTweak = overSized;

            int tweakCounter = 0;

            int[] oscillationObserver = new int[2];
            float tweak = 0;
            if (finalTweak)
                tweak = -1;
            float tweakUnit = TWEAK_UNIT;
            float falsePositive, detectionRate;

            while (Math.abs(tweak) < 1.1) {
                tweaks.set(round, tweak);

                float tmp[] = calcEmpiricalError(round, trainN, countTrainPos, labelsTrain);
                falsePositive = tmp[0];
                detectionRate = tmp[1];

                /*
                FIXME : train without using the test values... maybe less detection rate doing that ?
                tmp = calcEmpiricalError(round, testN, countTestPos, labelsTest);
                ctrlFalsePositive = tmp[0];
                ctrlDetectionRate = tmp[1];


                float worstFalsePositive = Math.max(falsePositive, ctrlFalsePositive);
                float worstDetectionRate = Math.min(detectionRate, ctrlDetectionRate);
                */

                float worstFalsePositive = falsePositive;
                float worstDetectionRate = detectionRate;

                if (finalTweak) {
                    if (worstDetectionRate >= 0.99) {
                        System.out.println("    - Final tweak settles to " + tweak);
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
                    System.out.println("    - No way out at this point. tweak goes from " + tweak);
                    continue;
                }

                // It is possible that tweak vacillates
                if (!finalTweak && tweakCounter > 1 && (oscillationObserver[0] + oscillationObserver[1]) == 0) {
                    // One solution is to reduce tweakUnit
                    tweakUnit /= 2;
                    tweak += oscillationObserver[tweakCounter % 2] == 1 ? -1 * tweakUnit : tweakUnit;

                    System.out.println("    - Backtracked at " + tweakCounter + "! Modify tweakUnit to " + tweakUnit);

                    if (tweakUnit < MIN_TWEAK) {
                        finalTweak = true;
                        System.out.println("    - TweakUnit too small. Tweak goes from " + tweak);
                    }
                }
            }
            if (overSized)
                break;
        }
    }

    private ArrayList<String> orderedExamples() {
        ArrayList<String> examples = new ArrayList<>(trainN);
        examples.addAll(train_faces);
        examples.addAll(train_nonfaces);
        return examples;
    }

    public void train(String dir, float initialPositiveWeight, float overallTargetDetectionRate, float overallTargetFalsePositiveRate,
                      float targetDetectionRate, float targetFalsePositiveRate) {
        if (computed) {
            System.out.println("Training already done!");
            return;
        }

        train_dir = dir;
        countTrainPos = countFiles(train_dir + "/faces", Conf.IMAGES_EXTENSION);
        countTrainNeg = countFiles(train_dir + "/non-faces", Conf.IMAGES_EXTENSION);
        trainN = countTrainPos + countTrainNeg;
        System.out.println("Total number of training images: " + trainN + " (pos: " + countTrainPos + ", neg: " + countTrainNeg + ")");

        // FIXME : alwys in the same order ??? Could be a problem ??
        train_faces = listFiles(train_dir + "/faces", Conf.IMAGES_EXTENSION);
        train_nonfaces = listFiles(train_dir + "/non-faces", Conf.IMAGES_EXTENSION);


        layerMemory = new ArrayList<>();

        // Compute all features for train & test set
        computeFeaturesTimed(train_dir);

        // Now organize all those features, so that it is easier to make requests on it
        organizeFeatures(featureCount, orderedExamples(), Conf.ORGANIZED_FEATURES, Conf.ORGANIZED_SAMPLE, false);


        System.out.println("Training classifier:");

        // Estimated number of rounds needed
        int boostingRounds = (int) (Math.ceil(Math.log(overallTargetFalsePositiveRate) / Math.log(targetFalsePositiveRate)) + 20);
        System.out.println("  - Estimated needed boosting rounds: " + boostingRounds);

        // Initialization
        tweaks = new ArrayList<>(boostingRounds);
        for (int i = 0; i < boostingRounds; i++)
            tweaks.add(0f);
        cascade = new ArrayList[boostingRounds];

        // Updating weights
        totalWeightPos = new BigDecimal(initialPositiveWeight);
        totalWeightNeg = new BigDecimal(1 - initialPositiveWeight);
        BigDecimal averageWeightPos = totalWeightPos.divide(new BigDecimal(countTrainPos), MathContext.DECIMAL128);
        BigDecimal averageWeightNeg = totalWeightNeg.divide(new BigDecimal(countTrainNeg), MathContext.DECIMAL128);
        minWeight = averageWeightPos.min(averageWeightNeg);
        maxWeight = averageWeightPos.max(averageWeightNeg);

        // Init labels & weights
        labelsTrain = new DenseMatrix(1, trainN);
        weightsTrain = new ArrayList<>(trainN);
        for (int i = 0; i < trainN; i++) {
            labelsTrain.set(0, i, i < countTrainPos ? 1 : -1);
            weightsTrain.add(i < countTrainPos ? averageWeightPos : averageWeightNeg);
        }

        double accumulatedFalsePositive = 1;

        // Training: run Cascade until we arrive to a certain wanted rate of success
        for (int round = 0; round < boostingRounds && accumulatedFalsePositive > GOAL; round++) {
            System.out.println("  - Round N." + round + ":");

            attentionalCascade(round, overallTargetDetectionRate, overallTargetFalsePositiveRate);
            System.out.println("    - Attentional Cascade computed!");

            // -- Display results for this round --

            //layerMemory.add(trainSet.committee.size());
            layerMemory.add(cascade[round].size());
            System.out.println("    - The committee size is " + cascade[round].size());

            float detectionRate, falsePositive;
            float[] tmp = calcEmpiricalError(round, trainN, countTrainPos, labelsTrain);
            falsePositive = tmp[0];
            detectionRate = tmp[1];
            System.out.println("    - The current tweak " + tweaks.get(round) + " has falsePositive " + falsePositive + " and detectionRate " + detectionRate + " on the training examples.");

            /*
            tmp = calcEmpiricalError(round, testN, countTestPos, labelsTest);
            falsePositive = tmp[0];
            detectionRate = tmp[1];
            System.out.println("The current tweak " + tweaks.get(round) + " has falsePositive " + falsePositive + " and detectionRate " + detectionRate + " on the validation examples.");
            */
            accumulatedFalsePositive *= falsePositive;
            System.out.println("    - Accumulated False Positive Rate is around " + accumulatedFalsePositive);

            // TODO : blackList ??

            //record the boosted rule into a target file
            Serializer.writeRule(cascade[round], round == 0, Conf.TRAIN_FEATURES);

        }

        // Serialize training
        Serializer.writeLayerMemory(this.layerMemory, this.tweaks, Conf.TRAIN_FEATURES);

        computed = true;
    }

    public float test(String dir) {
        if (!computed) {
            System.err.println("Train the classifier before testing it!");
            System.exit(1);
        }

        test_dir = dir;
        countTestPos = countFiles(test_dir + "/faces", Conf.IMAGES_EXTENSION);
        countTestNeg = countFiles(test_dir + "/non-faces", Conf.IMAGES_EXTENSION);
        testN = countTestPos + countTestNeg;

        computeFeaturesTimed(test_dir);

        // TODO: after the training has been done, we can test on a new set of images.

        return 0;
    }
}

