package process.training;

import GUI.ImageHandler;
import jeigen.DenseMatrix;
import process.Conf;
import process.DecisionStump;
import process.features.FeatureExtractor;
import process.features.FeaturesSerializer;

import java.util.*;

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
     *
     * CAUTION: the training only works on same-sized images!
     *
     * This Classifier uses what is know as Strong & Weak classifiers
     *      A Weak classifier is just a simple basic classifier, it could be Binary, Naive Bayes or anything else.
     *      The only need for a weak classifier is to return results with a success rate > 0.5, which is actually better than random.
     *      The combination of all these Weak classifier create of very good classifier, which is called the Strong classifier.
     *
     */

    private DenseMatrix predictLabel(ArrayList<DecisionStump> committee, int n) {
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
            memberWeight.set(member, log((1.0/committee.get(member).getError()) - 1));
            UUID featureId = committee.get(member).getFeatureId();
            for (int i = 0; i < n; i++) {
                // TODO
//                int exampleIndex = getExampleIndex(featureId, i);
//                memberVerdict.set(member, exampleIndex, (getExampleFeature(featureId, i) > committee.get(member).getThreshold() ? 1 : -1) * committee.get(member).getToggle());
            }
        }

        DenseMatrix prediction = new DenseMatrix(1, n);
        DenseMatrix finalVerdict = memberWeight.mmul(memberVerdict);
        for(int exampleIndex = 0; exampleIndex < n; exampleIndex++)
            prediction.set(1, exampleIndex, finalVerdict.get(1, exampleIndex) > 0 ? 1 : -1);

        return prediction;
    }

    private ArrayList<DecisionStump> adaboost(int n, int T) {
        ArrayList<DecisionStump> committee = new ArrayList<>();
        ArrayList<Long> weights = new ArrayList<>();

        for (int t = 0; t < T; t++) {
            // TODO
//            DecisionStump bestDS = DecisionStump.bestStump();
//            committee.add(bestDS);


            DenseMatrix prediction = predictLabel(committee, n);


            boolean werror = false;


            if (werror) {
                // Update weights

            } else {
                // Training ends, just return

            }

        }
        return committee;
    }

    public static void train(String faces_dir, String nonfaces_dir, int width, int height) {
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

        double averageWeightPos = 0.5/countPos;
        double averageWeightNeg = 0.5/countNeg;

        DenseMatrix weights = new DenseMatrix(N, 1); // weight vector for all examples involved
        DenseMatrix labels = new DenseMatrix(N, 1); // -1 = negative | 1 = positive example

        // Init weights & labels
        for(int i = 0; i < N; i++){
            labels.set(i, 0, i < countPos ? 1 : -1); // labels = [positives then negatives] = [1 1 ..., -1 -1 ...]
            weights.set(i, 0, i < countPos ? averageWeightPos : averageWeightNeg);
        }

        long featuresCount = FeatureExtractor.countAllFeatures(width, height);

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
}