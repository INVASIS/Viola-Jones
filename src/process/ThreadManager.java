package process;

import jeigen.DenseMatrix;
import utils.DoubleDouble;

import java.util.ArrayList;
import java.util.Objects;

import static process.StumpRule.compare;
import static process.features.FeatureExtractor.*;

public class ThreadManager extends Thread {

    private DenseMatrix labels;
    private ArrayList<DoubleDouble> weights;
    private long featureIndex;
    private int N;
    private DoubleDouble totalWeightPos;
    private DoubleDouble totalWeightNeg;
    private DoubleDouble minWeight;

    private StumpRule best;

    public ThreadManager(DenseMatrix labels, ArrayList<DoubleDouble> weights, long featureCount, int N, DoubleDouble totalWeightPos, DoubleDouble totalWeightNeg, DoubleDouble minWeight) {
        this.labels = labels;
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

        StumpRule best = new StumpRule(featureIndex, 2, getExampleFeature(featureIndex, 0, N) - 1, -1, 0);
        StumpRule current = deepCopy(best); // copy of best

        // Left & Right hand of the stump
        DoubleDouble leftWeightPos = DoubleDouble.ZERO;
        DoubleDouble leftWeightNeg = DoubleDouble.ZERO;
        DoubleDouble rightWeightPos = totalWeightPos;
        DoubleDouble rightWeightNeg = totalWeightNeg;

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
            DoubleDouble errorPlus = leftWeightPos.add(rightWeightNeg);
            DoubleDouble errorMinus = rightWeightPos.add(leftWeightNeg);

            DoubleDouble Epsilon_hat;
            if (errorPlus.lt(errorMinus)) {
                Epsilon_hat = errorPlus;
                current.toggle = 1;
            } else {
                Epsilon_hat = errorMinus;
                current.toggle = -1;
            }

            current.error = Epsilon_hat.lt(minWeight.multiplyBy(0.9)) ? new DoubleDouble(0) : Epsilon_hat;

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
                double label = (int) labels.get(0, exampleIndex); // FIXME: why casting to int?
                double weight = weights.get(exampleIndex).doubleValue();

                if (label < 0) {
                    leftWeightNeg = leftWeightNeg.add(weight); // leftWeightNeg += weight
                    rightWeightNeg = rightWeightNeg.subtract(weight); // rightWeightNeg -= weight
                } else {
                    leftWeightPos = leftWeightPos.add(weight); // leftWeightPos += weight
                    rightWeightPos = rightWeightPos.subtract(weight); // rightWeightPos -= weight
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
