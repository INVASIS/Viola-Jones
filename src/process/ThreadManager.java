package process;

import jeigen.DenseMatrix;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Objects;

import static process.DecisionStump.compare;
import static process.features.FeatureExtractor.*;

public class ThreadManager extends Thread {

    private DenseMatrix labels;
    private ArrayList<BigDecimal> weights;
    private long featureIndex;
    private int N;
    private BigDecimal totalWeightPos;
    private BigDecimal totalWeightNeg;
    private BigDecimal minWeight;

    private DecisionStump best;

    public ThreadManager(DenseMatrix labels, ArrayList<BigDecimal> weights, long featureCount, int N, BigDecimal totalWeightPos, BigDecimal totalWeightNeg, BigDecimal minWeight) {
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

        DecisionStump best = new DecisionStump(featureIndex, new BigDecimal(2), getExampleFeature(featureIndex, 0, N) - 1, -1, 0);
        DecisionStump current = deepCopy(best); // copy of best

        // Left & Right hand of the stump
        BigDecimal leftWeightPos = new BigDecimal(0);
        BigDecimal leftWeightNeg = new BigDecimal(0);
        BigDecimal rightWeightPos = totalWeightPos;
        BigDecimal rightWeightNeg = totalWeightNeg;

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
            BigDecimal errorPlus = leftWeightPos.add(rightWeightNeg);
            BigDecimal errorMinus = rightWeightPos.add(leftWeightNeg);

            BigDecimal Epsilon_hat;
            if (errorPlus.compareTo(errorMinus) == -1) { // <=> if (errorPlus < errorMinus)
                Epsilon_hat = errorPlus;
                current.toggle = 1;
            } else {
                Epsilon_hat = errorMinus;
                current.toggle = -1;
            }

            current.error = Epsilon_hat.compareTo(minWeight.multiply(new BigDecimal(0.9))) == -1 ? new BigDecimal(0) : Epsilon_hat; // <=> Epsilon_hat < minWeight * 0.9

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
                double label = (int) labels.get(0, exampleIndex); // FIXME: why wasting to int?
                double weight = weights.get(exampleIndex).doubleValue();

                if (label < 0) {
                    leftWeightNeg = leftWeightNeg.add(new BigDecimal(weight)); // leftWeightNeg += weight
                    rightWeightNeg = rightWeightNeg.subtract(new BigDecimal(weight)); // rightWeightNeg -= weight
                } else {
                    leftWeightPos = leftWeightPos.add(new BigDecimal(weight)); // leftWeightPos += weight
                    rightWeightPos = rightWeightPos.subtract(new BigDecimal(weight)); // ightWeightPos -= weight
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
