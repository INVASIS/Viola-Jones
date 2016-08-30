package process;

import javafx.util.Pair;
import jeigen.DenseMatrix;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static java.lang.Math.log;
import static java.util.Comparator.comparing;
import static javafx.application.Platform.exit;
import static process.features.FeatureExtractor.computeSetFeatures;
import static process.features.FeatureExtractor.countAllFeatures;
import static utils.Serializer.appendArrayToDisk;
import static utils.Serializer.readArrayFromDisk;
import static utils.Serializer.readIntFromDisk;
import static utils.Utils.countFiles;
import static utils.Utils.listFiles;

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

    // FIXME : the [] was removed
    private DenseMatrix weightsTrain;
    private DenseMatrix weightsTest;
    private DenseMatrix labelsTrain;
    private DenseMatrix labelsTest;

    public Classifier(int width, int height) {
        this.width = width;
        this.height = height;

        this.featureCount = countAllFeatures(width, height);
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
            int featureIndex = cascade[round].get(member).getFeatureIndex();
            for (int i = 0; i < N; i++) {
                // TODO
//                int exampleIndex = getExampleIndex(featureId, i);
//                memberVerdict.set(member, exampleIndex, (getExampleFeature(featureId, i) > committee.get(member).getThreshold() ? 1 : -1) * committee.get(member).getToggle()) + decisionTweak;
            }
        }

        DenseMatrix finalVerdict = memberWeight.mul(memberVerdict);
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
        //DecisionStump bestDS = DecisionStump.bestStump(features, weightsTrain);
        //committee.add(bestDS);
        adaboostPasses++;

        DenseMatrix prediction = new DenseMatrix(1, N);
        predictLabel(round, N, 0, prediction, true);

        // cwise product = mul
        //DenseMatrix agree = labelsTrain[round].mul(prediction.t());
        DenseMatrix agree = labelsTrain.mul(prediction.t());

        DenseMatrix weightUpdate = DenseMatrix.ones(1, N);

        boolean werror = false;

        for (int index = 0; index < N; index++) {
            if (agree.get(0, index) < 0) {
                // TODO : uncomment when the decisionStump is computed
                //weightUpdate[index] = 1 / bestDS.getError() - 1;
                werror = true;
            }
        }

        if (werror) {
            // Update weights
            weightsTrain = weightsTrain.mul(weightUpdate);
            DecisionStump.positiveTotalWeights = 0;

            for (int i = 0; i < weightsTrain.cols; i++) {
                weightsTrain.set(0, i, weightsTrain.get(0, i) / weightsTrain.s());

                // Update pos weight
                if (i < countTrainPos)
                    DecisionStump.positiveTotalWeights += weightsTrain.get(0, i);
            }

            // Update positiveTotalWeights
            // minWeight ?
        }

        System.out.println("Adaboost passes : " + adaboostPasses);
        return committee;
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

        DenseMatrix agree = labels.mul(verdicts.t());
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

                float tmp[] = calcEmpiricalError(round, trainN, countTrainPos, labelsTrain);
                ctrlFalsePositive = tmp[0];
                ctrlDetectionRate = tmp[1];

                /*
                // FIXME : train without using the test values... maybe less detection rate doing that ?
                tmp = calcEmpiricalError(round, testN, countTestPos, labelsTest);
                falsePositive = tmp[0];
                detectionRate = tmp[1];


                float worstFalsePositive = Math.max(falsePositive, ctrlFalsePositive);
                float worstDetectionRate = Math.min(detectionRate, ctrlDetectionRate);
                */

                float worstFalsePositive = ctrlFalsePositive;
                float worstDetectionRate = ctrlDetectionRate;

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

    /**
     * Pour chaque feature:
     *      vector<pair<valeur-de-la-feature, l'index de l'exemple (image)>> ascendingFeatures;
     *      Pour chaque exemple:
     *          ascendingFeatures.add(<valeur-de-cette-feature-pour-cet-example, index-de-l'exemple>)
     *          trier ascendingFeatures en fonction de pair.first
     *          Write sur disque:
     *              * OrganizedFeatures (à l'index de la feature actuelle le ascendingFeatures.first en entier) tmp/training
     *              * OrganizedSample (à l'index de la feature actuelle le ascendingFeatures.second en entier)
     */
    private void organizeFeatures() {
        System.out.println("Organizing features...");

        if (Files.exists(Paths.get(Conf.ORGANIZED_FEATURES)))
            try {
                Files.delete(Paths.get(Conf.ORGANIZED_FEATURES));
            } catch (IOException e) {
                e.printStackTrace();
                exit();
            }
        if (Files.exists(Paths.get(Conf.ORGANIZED_SAMPLE)))
            try {
                Files.delete(Paths.get(Conf.ORGANIZED_SAMPLE));
            } catch (IOException e) {
                e.printStackTrace();
                exit();
            }


        final Comparator<Pair<Integer, Integer>> c = comparing(Pair::getValue);

        ArrayList<String> examples = new ArrayList<>(trainN);
        examples.addAll(train_faces);
        examples.addAll(train_nonfaces);

        long presumableFreeMemory = Runtime.getRuntime().maxMemory() - (Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory());
        boolean allInMemory = presumableFreeMemory > (readArrayFromDisk(examples.get(0)).size() * Integer.BYTES * trainN);

        ArrayList<ArrayList<Integer>> allImagesFeatures = null;
        if (allInMemory) {
            allImagesFeatures = new ArrayList<>();
            for (String e : examples)
                allImagesFeatures.add(readArrayFromDisk(e + Conf.FEATURE_EXTENSION));
        }
        for (long featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            // <exampleIndex, value>
            ArrayList<Pair<Integer, Integer>> ascendingFeatures = new ArrayList<>();

            if (allInMemory) {
                for (int exampleIndex = 0; exampleIndex < trainN; exampleIndex++) {
                    ascendingFeatures.add(new Pair<>(exampleIndex, allImagesFeatures.get(exampleIndex).get((int)featureIndex)));
                }
            }
            else {
                for (int exampleIndex = 0; exampleIndex < trainN; exampleIndex++) {
                    ascendingFeatures.add(new Pair<>(exampleIndex, readIntFromDisk(examples.get(exampleIndex) + Conf.FEATURE_EXTENSION, featureIndex)));
                }
            }
            ascendingFeatures.stream().sorted(c);

            ArrayList<Integer> permutedSamples = new ArrayList<>(trainN);
            ArrayList<Integer> permutedFeatures = new ArrayList<>(trainN);

            for (int k = 0; k < trainN; k++) {
                permutedSamples.add(ascendingFeatures.get(k).getKey());
                permutedFeatures.add(ascendingFeatures.get(k).getValue());
            }

            appendArrayToDisk(Conf.ORGANIZED_SAMPLE, permutedSamples);
            appendArrayToDisk(Conf.ORGANIZED_FEATURES, permutedFeatures);
        }
    }

    private void computeFeaturesTimed(String path) {
        System.out.println("Computing features for:");
        System.out.println("  - " + path);
        int count = 0;
        long startTime = System.currentTimeMillis();
        count += computeSetFeatures(path + "/faces", path + "/non-faces", true);
        if (count > 0) {
            long elapsedTimeMS = (new Date()).getTime() - startTime;
            System.out.println("  Statistics:");
            System.out.println("    - Elapsed time: " + elapsedTimeMS / 1000 + "s");
            System.out.println("    - Images computed: " + count);
            System.out.println("    - image/seconds: " + count / (elapsedTimeMS / 1000));
        }
        else
            System.out.println("  - All features already computed!");
    }

    public void train(String dir, float overallTargetDetectionRate, float overallTargetFalsePositiveRate,
                      float targetDetectionRate, float targetFalsePositiveRate) {
        /**
         * In order to avoid excessive memory usage, this training temporary stores metadata on disk.         *
         */

        if (computed) {
            System.out.println("Training already done!");
            return;
        }

        train_dir = dir;
        countTrainPos = countFiles(train_dir + "/faces", Conf.IMAGES_EXTENSION);
        countTrainNeg = countFiles(train_dir + "/non-faces", Conf.IMAGES_EXTENSION);
        trainN = countTrainPos + countTrainNeg;
        System.out.println("Total number of training images: " + trainN +
                " (pos: " + countTrainPos + ", neg: " + countTrainNeg + ")");

        train_faces = new ArrayList<>(countTestPos);
        train_nonfaces = new ArrayList<>(countTestNeg);

        train_faces = listFiles(train_dir + "/faces", Conf.IMAGES_EXTENSION);
        train_faces = listFiles(train_dir + "/non-faces", Conf.IMAGES_EXTENSION);


        // FIXME: What is that? - Used later
        layerMemory = new ArrayList<>();

        // Compute all features for train & test set
        computeFeaturesTimed(train_dir);

        // Now organize all those features, so that it is easier to make requests on it
        organizeFeatures();

        // Estimated number of rounds needed
        int boostingRounds = (int) (Math.ceil(Math.log(overallTargetFalsePositiveRate) / Math.log(targetFalsePositiveRate)) + 20);
        System.out.println("Boosting rounds : " + boostingRounds);

        // Initialization
        tweaks = new ArrayList<>(boostingRounds);
        cascade = new ArrayList[boostingRounds];
        for (int i = 0; i < boostingRounds; i++)
            cascade[i] = new ArrayList<>();

        // Init weights & labels
        // FIXME : already init later in the next for

        double accumulatedFalsePositive = 1;

        // Training: run Cascade until we arrive to a certain wanted rate of success
        for (int round = 0; round < boostingRounds && accumulatedFalsePositive > GOAL; round++) {

            DecisionStump.positiveTotalWeights = 0.5;

            // TODO : for now only for train
            double posAverageWeight = DecisionStump.positiveTotalWeights / countTrainPos;
            double negAverageWeight = (1 - DecisionStump.positiveTotalWeights) / countTrainNeg;

            labelsTrain = new DenseMatrix(1, trainN);
            weightsTrain = new DenseMatrix(1, trainN);

            // FIXME : sould it be sorted in ascending order ?
            // IMPORTANT : Images should be set as follow : first the countTrainPos first positive images, and then the negative images
            for (int i = 0; i < trainN; i++) {
                labelsTrain.set(0, i, i < countTrainPos ? 1 : -1);
                weightsTrain.set(0, i, i < countTrainPos ? posAverageWeight : negAverageWeight);
            }

            int committeeSizeGuide = Math.min(20 + round * 10, 200);
            System.out.println("CommitteeSizeGuide = " + committeeSizeGuide);

            attentionalCascade(round, committeeSizeGuide, overallTargetDetectionRate, overallTargetFalsePositiveRate);


            // -- Display results for this round --

            System.out.println("Layer " + round + 1 + " done!");

            //layerMemory.add(trainSet.committee.size());
            layerMemory.add(cascade[round].size());
            System.out.println("The committee size is " + cascade[round].size());

            float detectionRate, falsePositive;
            float[] tmp = calcEmpiricalError(round, trainN, countTrainPos, labelsTrain);
            falsePositive = tmp[0];
            detectionRate = tmp[1];
            System.out.println("The current tweak " + tweaks.get(round) + " has falsePositive " + falsePositive + " and detectionRate " + detectionRate + " on the training examples.");

            /*
            tmp = calcEmpiricalError(round, testN, countTestPos, labelsTest);
            falsePositive = tmp[0];
            detectionRate = tmp[1];
            System.out.println("The current tweak " + tweaks.get(round) + " has falsePositive " + falsePositive + " and detectionRate " + detectionRate + " on the validation examples.");
            */
            accumulatedFalsePositive *= falsePositive;
            System.out.println("Accumulated False Positive Rate is around " + accumulatedFalsePositive);

            // TODO : blackList ??

            //record the boosted rule into a target file
            recordRule(cascade[round], round == 0, round == boostingRounds - 1 || accumulatedFalsePositive <= GOAL);

        }
        recordLayerMemory();


        // Serialize training
        computed = true;
    }

    private void recordRule(ArrayList<DecisionStump> committee, boolean firstRound, boolean lastRound) {

        try {
            int memberCount = committee.size();

            PrintWriter writer = new PrintWriter(new FileWriter(Conf.TRAIN_FEATURES, true));

            if (firstRound)
                writer.println(System.lineSeparator() + "double stumps[][4]={");

            for (int i = 0; i < memberCount; i++) {
                DecisionStump decisionStump = committee.get(i);
                writer.print("{" + decisionStump.getFeatureIndex() + "," + decisionStump.getError() + ","
                        + decisionStump.getThreshold() + "," + decisionStump.getToggle() + "}");

                if (i == memberCount - 1 && lastRound)
                    writer.println(System.lineSeparator() + "};");
                else
                    writer.println(",");
            }

            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void recordLayerMemory() {

        try {
            int layerCount = this.layerMemory.size();
            PrintWriter writer = new PrintWriter(new FileWriter(Conf.TRAIN_FEATURES, true));

            writer.println("int layerCount=" + layerCount + ";");
            writer.print("int layerCommitteeSize[]={");

            for (int i = 0; i < layerCount; i++) {
                writer.print(this.layerMemory.get(i));
                if (i < layerCount - 1)
                    writer.print(",");
                else
                    writer.println("};");
            }

            writer.print("float tweaks[]={");
            for (int i = 0; i < layerCount; i++) {
                writer.print(this.tweaks.get(i));
                if (i < layerCount - 1)
                    writer.print(",");
                else
                    writer.println("};");
            }

            writer.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public float test(String dir) {
        if (!computed) {
            System.err.println("Train the classifier before testing it!");
            exit();
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

