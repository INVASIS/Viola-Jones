package process;

import jeigen.DenseMatrix;

import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.Callable;

import static process.features.FeatureExtractor.*;

public class DecisionStump implements Callable<StumpRule> {
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
    private boolean[] removed;

    public DecisionStump(DenseMatrix labels, DenseMatrix weights, long featureIndex, int N, double totalWeightPos, double totalWeightNeg, double minWeight, boolean[] removed) {
        this.Y = labels;
        this.weights = weights;
        this.featureIndex = featureIndex;
        this.N = N;
        this.totalWeightPos = totalWeightPos;
        this.totalWeightNeg = totalWeightNeg;
        this.minWeight = minWeight;
        this.removed = removed;
    }

    @Override
    public StumpRule call() throws Exception {
        // The best StumpRule is the one with the lower error. This is equivalent to find the one which best
        // separates positive and negative examples.
        //
        // For N training example, there is exactly N+1 possible StumpRule.
        //
        // To compute the error, we use totalWeightPos & totalWeightNeg, as a result,
        // the best StumpRule returned is (almost?) always different.

        // Get needed values
        int[] X = getFeatureExamplesIndexes(featureIndex, N);
        int[] V = getFeatureValues(featureIndex, N); // V is already sorted in ascending order.

        // Left & Right hand of the stump
        double leftWeightPos = 0;
        double leftWeightNeg = 0;
        double rightWeightPos = totalWeightPos;
        double rightWeightNeg = totalWeightNeg;

        // First compute all threshold and margins between all values of V, but also before (-1) and after (+1) V values!
        // We will then find the best threshold in this list based on the error it gives
        ArrayList<Double> thresholds = new ArrayList<>(N + 1);
        ArrayList<Double> margins = new ArrayList<>(N + 1);
        {
            thresholds.add((double) (V[0] - 1));
            margins.add((double) -1);
            for (int i = 0; i < N-1; i++) {
                thresholds.add(((double) (V[i] + V[i + 1])) / 2.0d);
                margins.add((double) V[i + 1] - V[i]);
            }
            thresholds.add((double) (V[N-1] + 1));
            margins.add((double) 0);
        }
        StumpRule best = new StumpRule(featureIndex, 2, thresholds.get(0),margins.get(0), 0);
        StumpRule current = StumpRule.deepCopy(best);

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
                best = StumpRule.deepCopy(current);

            iterator++;

            // We don't actually need to look at the sample with the largest feature
            // because its rule is exactly equivalent to those produced
            // by the sample with the smallest feature on training observations
            // but it won't do any harm anyway
            if (iterator == N)
                break;

            while (true) {
                int exampleIndex = X[iterator];
                if (!removed[exampleIndex]) {
                    double label = (int) Y.get(0, exampleIndex); // FIXME: why casting to int?
                    double weight = weights.get(0, exampleIndex);

                    if (label < 0) {
                        leftWeightNeg += weight; // leftWeightNeg += weight
                        rightWeightNeg -= weight; // rightWeightNeg -= weight
                    } else {
                        leftWeightPos += weight; // leftWeightPos += weight
                        rightWeightPos -= weight; // rightWeightPos -= weight
                    }
                }
                // It is possible to have the same feature values from different examples
                // if a new threshold can be found, break
                // two cases are possible:
                //   - Either it is the last observation:
                if ((iterator == N - 1) || (!Objects.equals(V[iterator], V[iterator + 1])))
                    break;

                iterator++;
            }

            current.threshold = thresholds.get(iterator+1);
            current.margin = margins.get(iterator+1);
        }
        return best;
    }
}
