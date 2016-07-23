package process;

import javafx.util.Pair;

import java.util.ArrayList;
import java.util.Objects;
import java.util.UUID;

public class DecisionStump {

    // Values that will be used to find the best DecisionStump
    private double threshold;
    private boolean toggle;
    private double error;
    private double margin;
    private UUID featureId;


    private double W1plus;
    private double W0plus;
    private double W1min;
    private double W0min;

    private ArrayList<Pair<Integer, Boolean>> features;
    private ArrayList<Double> w;


    // Initialisation
    public DecisionStump(ArrayList<Pair<Integer, Boolean>> features, ArrayList<Double> w, UUID featureId) {
        this.margin = -1; // Like that in the paper implem...
        this.error = 2;
        this.featureId = featureId;

        this.W1plus = 0;
        this.W1min = 0;
        this.W1min = 0;
        this.W0min = 0;

        // Should be arranged in ascending order
        this.features = (ArrayList<Pair<Integer, Boolean>>) features.clone();
        this.w = (ArrayList<Double>) w.clone();

        this.threshold = this.features.get(0).getKey();
        for (int i = 0; i < this.features.size(); i++) {
            if (this.features.get(i).getKey() < this.threshold)
                this.threshold = this.features.get(i).getKey();

            // Pas sur...
            if (this.w.get(i) == 1)
                this.W1plus += this.w.get(i);
            else
                this.W0plus += this.w.get(i);
        }
        this.threshold--;


    }

    public void compute() {

        int n = this.features.size() - 1;
        int j = 0;
        double tmp_threshold = this.threshold;
        double tmp_margin = this.margin;
        double tmp_error = 0;
        boolean tmp_togle = false;

        while (true) {
            double errorPlus = W1plus + W0plus;
            double errorMin = W1min + W0min;

            if (errorPlus < errorMin) {
                tmp_error = errorPlus;
                tmp_togle = true;
            } else {
                tmp_error = errorMin;
                tmp_togle = false;
            }

            if (tmp_error < this.error || tmp_error == this.error && tmp_margin > this.margin) {
                this.error = tmp_error;
                this.threshold = tmp_threshold;
                this.margin = tmp_margin;
                this.toggle = tmp_togle;
            }

            if (j == n)
                break;

            j++;

            while (true) {
                if (!this.features.get(j).getValue()) {
                    this.W0min += this.w.get(j);
                    this.W0plus -= this.w.get(j);
                } else {
                    this.W1min += this.w.get(j);
                    this.W1plus -= this.w.get(j);
                }

                if (j == n || Objects.equals(this.features.get(j).getKey(), this.features.get(j + 1).getKey()))
                    break;

                j++;
            }

            if (j == n) {
                tmp_threshold = this.features.get(j).getKey();
                tmp_margin = 0;
            } else {
                tmp_threshold = (this.features.get(j).getKey() + this.features.get(j + 1).getKey()) / 2;
                tmp_margin = this.features.get(j + 1).getKey() + this.features.get(j).getKey();
            }
        }
    }

    public static DecisionStump bestStump(ArrayList<ArrayList<Pair<Integer, Boolean>>> features, ArrayList<Double> w) {
        // FIXME
//        DecisionStump best = new DecisionStump(features.get(0), w);
//        best.compute();
//
//        for (int i = 1; i < features.size(); i++) {
//            DecisionStump decisionStump = new DecisionStump(features.get(i), w);
//            decisionStump.compute();
//
//            if (decisionStump.error < best.error || decisionStump.error == best.error && decisionStump.margin > best.margin) {
//                best = decisionStump;
//            }
//        }
//
//        return best;
        return new DecisionStump(features.get(0), w, UUID.fromString("temporary"));
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

    public UUID getFeatureId() {
        return featureId;
    }
}
