package process;

import jeigen.DenseMatrix;

import java.util.ArrayList;
import java.util.Objects;

import static process.features.FeatureExtractor.*;

public class DecisionStump extends Thread {
    // X are examples
    // Y are labels
    // V are feature values
    // (Xi, Yi, Vi) is a tuple of an image, its label (either -1 or 1), and the value the feature for image Xi.

    private DenseMatrix Y;
    private DenseMatrix weights;
    private long featureIndex;
    private int N;
    private double totalWeightPos;
    private double totalWeightNeg;
    private double minWeight;

    private StumpRule best;

    public DecisionStump(DenseMatrix labels, DenseMatrix weights, long featureCount, int N, double totalWeightPos, double totalWeightNeg, double minWeight) {
        this.Y = labels;
        this.weights = weights;
        this.featureIndex = featureCount;
        this.N = N;
        this.totalWeightPos = totalWeightPos;
        this.totalWeightNeg = totalWeightNeg;
        this.minWeight = minWeight;
    }

    public StumpRule getBest() {
        return best;
    }

    @Override
    public void run() {
        // To build a decision stump, we need a toggle (polarity) and an admissible threshold
        // which minimizes the weighted classification error
        //
        // For N training example, there is exactly N+1 possible StumpRule.


        // Get needed values
        ArrayList<Integer> X = getFeatureExamplesIndexes(featureIndex, N);
        ArrayList<Integer> V = getFeatureValues(featureIndex, N);

        // Left & Right hand of the stump
        double leftWeightPos = 0;
        double leftWeightNeg = 0;
        double rightWeightPos = totalWeightPos;
        double rightWeightNeg = totalWeightNeg;

        assert X.size() == N;
        assert V.size() == N;
        assert getExampleIndex(featureIndex, 0, N) == X.get(0);
        assert getExampleFeature(featureIndex, 0, N) == V.get(0);

        // First compute all threshold and margins between all values of V, but also before (-1) and after (+1) V values!
        // We will then find the best threshold in this list based on the error it gives
        ArrayList<Double> thresholds;
        ArrayList<Double> margins;
        {
            thresholds = new ArrayList<>(N + 1);
            margins = new ArrayList<>(N + 1);

            thresholds.add((double) (V.get(0) - 1));
            margins.add((double) -1);
            for (int i = 0; i < N-1; i++) {
                thresholds.add(((double) (V.get(i) + V.get(i + 1))) / 2.0d);
                margins.add((double) V.get(i + 1) - V.get(i));
            }
            thresholds.add((double) (V.get(N-1) + 1));
            margins.add((double) 0);
        }

        StumpRule best = new StumpRule(featureIndex, 2, thresholds.get(0),margins.get(0), 0);
        StumpRule current = deepCopy(best);
        for (int i = 0; i < N; i++) {

        }

        // Go through all these observations one after another
        int iterator = -1;
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

            if (current.compare(best))
                best = deepCopy(current);

            iterator++;


            // We don't actually need to look at the sample with the largest feature
            // because its rule is exactly equivalent to those produced
            // by the sample with the smallest feature on training observations
            // but it won't do any harm anyway
            if (iterator == N)
                break;

            while (true) {
                int exampleIndex = X.get(iterator);
                double label = (int) Y.get(0, exampleIndex); // FIXME: why casting to int?
                double weight = weights.get(0, exampleIndex);

                if (label < 0) {
                    leftWeightNeg = leftWeightNeg + weight; // leftWeightNeg += weight
                    rightWeightNeg = rightWeightNeg - weight; // rightWeightNeg -= weight
                } else {
                    leftWeightPos = leftWeightPos + weight; // leftWeightPos += weight
                    rightWeightPos = rightWeightPos - weight; // rightWeightPos -= weight
                }

                // if a new threshold can be found, break
                // two cases are possible:
                //   - Either it is the last observation:
                if ((iterator == N - 1) || (!Objects.equals(V.get(iterator), V.get(iterator + 1))))
                    break;

                iterator++;
            }

            current.threshold = thresholds.get(iterator+1);
            current.margin = margins.get(iterator+1);
        }
        this.best = best;
    }

    private static StumpRule deepCopy(StumpRule other) {
        return new StumpRule(
                other.featureIndex,
                other.error,
                other.threshold,
                other.margin,
                other.toggle
        );
    }
}
