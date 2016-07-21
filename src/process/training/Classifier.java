package process.training;

import GUI.ImageHandler;
import jeigen.DenseMatrix;
import process.DecisionStump;

import java.util.*;

import static java.lang.Math.log;
import static javafx.application.Platform.exit;
import static utils.Utils.countFiles;
import static utils.Utils.streamImageHandler;

public class Classifier {
    /**
     * A Classifier maps an observation to a label valued in a finite set.
     * f(HaarFeaturesOfTheImage) = -1 or 1
     *
     * Caution: the training only works on same-sized images!
      */

    private static final int ROUNDS = 20;

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

    public void train(String faces_dir, String nonfaces_dir) {
        Iterable<ImageHandler> faces = streamImageHandler(faces_dir);
        Iterable<ImageHandler> nonFaces = streamImageHandler(nonfaces_dir);

        int countFaces = countFiles(faces_dir);
        int countNonFaces = countFiles(nonfaces_dir);

        int n = countFaces + countNonFaces;

        adaboost(n, ROUNDS);
    }
}
