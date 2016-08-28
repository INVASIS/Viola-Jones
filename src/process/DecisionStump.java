package process;

import javafx.util.Pair;

import java.util.ArrayList;
import java.util.Objects;

public class DecisionStump { // == stumpRule

    public static double positiveTotalWeights = 0.5;

    // Values that will be used to find the best DecisionStump
    private int featureIndex;
    private double error;
    private double threshold;
    private double margin;
    private boolean toggle; // = polarity {-1; 1}


    private double rPos;
    private double rNeg;
    private double lPos;
    private double lNeg;

    private ArrayList<Pair<Integer, Boolean>> features;
    private ArrayList<Double> w;


    // Initialisation
    public DecisionStump(ArrayList<Pair<Integer, Boolean>> features, ArrayList<Double> w, int featureIndex) {

        this.featureIndex = featureIndex;
        this.error = 2;
        this.threshold = this.features.get(0).getKey() - 1; // FIXME with organized features
        this.margin = -1; // Like that in the paper implem...
        this.toggle = false;

        this.rPos = positiveTotalWeights;
        this.rNeg = 1 - positiveTotalWeights;

        this.lPos = 0;
        this.lNeg = 0;

        // Should be arranged in ascending order
        this.features = (ArrayList<Pair<Integer, Boolean>>) features.clone();
        this.w = (ArrayList<Double>) w.clone();

    }

    public void compute() {

        int num_features = this.features.size() - 1;
        int iter = -1;


        double tmp_threshold = this.threshold;
        double tmp_margin = this.margin;
        double small_error = 0;
        boolean tmp_togle = false;

        while (true) {
            double errorPlus = lPos + rNeg;
            double errorMin = rPos + lNeg;

            if (errorPlus < errorMin) {
                small_error = errorPlus;
                tmp_togle = true;
            } else {
                small_error = errorMin;
                tmp_togle = false;
            }

            if (small_error < this.error || small_error == this.error && tmp_margin > this.margin) {
                this.error = small_error;
                this.threshold = tmp_threshold;
                this.margin = tmp_margin;
                this.toggle = tmp_togle;
            }

            iter++;

            // Pas sur si c'est toutes les features ou pas... mais ça doit etre bon...
            if (iter == num_features)
                break;

            while (true) {

                if (!this.features.get(iter).getValue()) {
                    this.lNeg += this.w.get(iter);
                    this.rNeg -= this.w.get(iter);
                } else {
                    this.lPos += this.w.get(iter);
                    this.rPos -= this.w.get(iter);
                }

                if (iter == num_features || Objects.equals(this.features.get(iter).getKey(), this.features.get(iter + 1).getKey()))
                    break;

                iter++;
            }

            if (iter < num_features - 1) {
                tmp_threshold = (this.features.get(iter).getKey() + this.features.get(iter + 1).getKey()) / 2;
                tmp_margin = this.features.get(iter + 1).getKey() - this.features.get(iter).getKey();
            } else {
                tmp_threshold = this.features.get(iter).getKey() + 1;
                tmp_margin = 0;
            }
        }
    }

    /**
     * <Integer, Boolean> indique pour chaque feature (qui normalement doivent etre triées), si c'est un face ou non-face
     */
    public static DecisionStump bestStump(ArrayList<ArrayList<Pair<Integer, Boolean>>> features, ArrayList<Double> w) {
        /**
         * VERIF OK
         * Not exactly like in the implem, but improved a bit...
         */

        DecisionStump best = new DecisionStump(features.get(0), w, 0);
        best.compute();

        for (int i = 1; i < features.size(); i++) {
            DecisionStump decisionStump = new DecisionStump(features.get(i), w, i);
            decisionStump.compute();

            if (decisionStump.error < best.error || decisionStump.error == best.error && decisionStump.margin > best.margin) {
                best = decisionStump;
            }
        }

        if (best.error >= 0.5)
            System.err.println("Failed best stump, error : " + best.error + " >= 0.5 ! (not good but we still continue)");

        return best;
    }

    public double getMargin() {
        return margin;
    }

    public double getThreshold() {
        return threshold;
    }

    public boolean getToggle() {
        return toggle;
    }

    public double getError() {
        return error;
    }

    public int getFeatureIndex() {
        return featureIndex;
    }
}
