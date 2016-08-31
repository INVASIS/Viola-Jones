package process;

import jeigen.DenseMatrix;

import java.util.ArrayList;
import java.util.Objects;

import static process.DecisionStump.compare;
import static process.features.FeatureExtractor.*;

public class ThreadManager extends Thread {

    private DenseMatrix labels;
    private DenseMatrix weights;
    private long featureIndex;
    private int N;
    private double totalWeightPos;
    private double totalWeightNeg;
    private double minWeight;

    private DecisionStump best;

    public ThreadManager(DenseMatrix labels, DenseMatrix weights, long featureCount, int N, double totalWeightPos, double totalWeightNeg, double minWeight) {
        this.labels = labels;
        this.weights = weights;
        this.featureIndex = featureCount;
        this.N = N;
        this.totalWeightPos = totalWeightPos;
        this.totalWeightNeg = totalWeightNeg;
        this.minWeight = minWeight;
    }

    public DecisionStump getBest() {
        return best;
    }

    @Override
    public void run() {

        DecisionStump best = new DecisionStump(featureIndex, 2, getExampleFeature(featureIndex, 0, N) - 1, -1, 0);
        DecisionStump current = deepCopy(best); // copy of best

        // Left & Right hand of the stump
        double leftWeightPos = 0;
        double leftWeightNeg = 0;
        double rightWeightPos = totalWeightPos;
        double rightWeightNeg = totalWeightNeg;

        // Go through all these observations one after another
        int iterator = -1;

        // To build a decision stump, you need a toggle and an admissible threshold
        // which doesn't coincide with any of the observations

        ArrayList<Integer> featureExampleIndexes = getFeatureExamplesIndexes(featureIndex, N);
        ArrayList<Integer> featureValues = getFeatureValues(featureIndex, N);
        assert featureExampleIndexes.size() == N;
        assert featureValues.size() == N;
        assert getExampleIndex(featureIndex, 0, N) == featureExampleIndexes.get(0);
        assert getExampleFeature(featureIndex, 0, N) == featureValues.get(0);

        while (true) {
            double errorPlus = leftWeightPos + rightWeightNeg;
            double errorMinus = rightWeightPos + leftWeightNeg;

            double Epsilon_hat;
            if (errorPlus < errorMinus) {
                Epsilon_hat = errorPlus;
                current.toggle = 1;
            } else {
                Epsilon_hat = errorMinus;
                current.toggle = -1;
            }

            current.error = Epsilon_hat < minWeight * 0.9 ? 0 : Epsilon_hat;

            if (compare(current, best))
                best = deepCopy(current);

            iterator++;


            // We don't actually need to look at the sample with the largest feature
            // because its rule is exactly equivalent to those produced
            // by the sample with the smallest feature on training observations
            // but it won't do any harm anyway
            if (iterator == N)
                break;

            while (true) {
                int exampleIndex = featureExampleIndexes.get(iterator);
                double label = (int) labels.get(0, exampleIndex);
                double weight = weights.get(0, exampleIndex);

                if (label < 0) {
                    leftWeightNeg += weight;
                    rightWeightNeg -= weight;
                } else {
                    leftWeightPos += weight;
                    rightWeightPos -= weight;
                }

                // if a new threshold can be found, break
                // two cases are possible:
                //   - Either it is the last observation:
                if ((iterator == N - 1) || (!Objects.equals(featureValues.get(iterator), featureValues.get(iterator + 1))))
                    break;

                iterator++;
            }

            if (iterator < N - 1) {
                current.threshold = ((double) featureValues.get(iterator) + (double) featureValues.get(iterator + 1)) / 2.0d;
                current.margin = featureValues.get(iterator + 1) - featureValues.get(iterator);
            } else {
                current.threshold = featureValues.get(iterator) + 1;
                current.margin = 0;
            }
        }

        this.best = best;
    }

    private static DecisionStump deepCopy(DecisionStump other) {
        return new DecisionStump(
                other.featureIndex,
                other.error,
                other.threshold,
                other.margin,
                other.toggle
        );
    }


}
