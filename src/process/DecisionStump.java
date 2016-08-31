package process;

import jeigen.DenseMatrix;

import static javafx.application.Platform.exit;
import static process.features.FeatureExtractor.getExampleFeature;
import static process.features.FeatureExtractor.getExampleIndex;

public class DecisionStump { // == stumpRule

    // Values that will be used to find the best DecisionStump
    public long featureIndex;
    public double error;
    public double threshold;
    public double margin;
    public int toggle; // = polarity {-1; 1}


    // Initialisation
    public DecisionStump(long featureIndex, double error, double threshold, double margin, int toggle) {
        this.featureIndex = featureIndex;
        this.error = error;
        this.threshold = threshold;
        this.margin = margin;
        this.toggle = toggle;
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

    private static boolean compare(DecisionStump first, DecisionStump second) {
        return (first.error < second.error ||
                (first.error == second.error && first.margin > second.margin));
    }

    public static DecisionStump compute(DenseMatrix labels, DenseMatrix weights, long featureIndex, int N, double totalWeightPos, double totalWeightNeg, double minWeight) {
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
                int exampleIndex = getExampleIndex(featureIndex, iterator, N);
                double label = (int) labels.get(0, exampleIndex);
                double weight = weights.get(0, exampleIndex);

                if (label < 0) {
                    leftWeightNeg += weight;
                    rightWeightNeg -= weight;
                }
                else {
                    leftWeightPos += weight;
                    rightWeightPos -= weight;
                }

                // if a new threshold can be found, break
                // two cases are possible:
                //   - Either it is the last observation:
                if (iterator == N - 1)
                    break;
                //   - Or no duplicate. If there is a duplicate, repeat:
                if (getExampleFeature(featureIndex, iterator, N) != getExampleFeature(featureIndex, iterator + 1, N)) {
                    double test = getExampleFeature(featureIndex, iterator, N) + getExampleFeature(featureIndex, iterator + 1, N);
                    test /= 2;

                    if (getExampleFeature(featureIndex, iterator, N) < test && test < getExampleFeature(featureIndex, iterator + 1, N))
                        break;
                    else {
                        System.err.println("FATAL: Numerical precision breached: problem feature values " +
                                getExampleFeature(featureIndex, iterator, N) + " : " +
                                getExampleFeature(featureIndex, iterator + 1, N) + ". Problem feature " +
                                featureIndex + " and problem example " + getExampleIndex(featureIndex, iterator, N) + " :" +
                                getExampleIndex(featureIndex, iterator + 1, N));
                        exit();
                    }
                }

                iterator++;
            }

            if (iterator < N - 1) {
                current.threshold = ((double)getExampleFeature(featureIndex, iterator, N) + (double)getExampleFeature(featureIndex, iterator + 1, N)) / 2.0d ;
                current.margin = getExampleFeature(featureIndex, iterator + 1, N) - getExampleFeature(featureIndex, iterator, N);
            } else {
                current.threshold = getExampleFeature(featureIndex, iterator, N) + 1;
                current.margin = 0;
            }
        }

        return best;
    }

    /**
     * Algorithm 5 from the original paper
     *
     * Return the most discriminative feature and its rule
     * We compute each DecisionStump, and find the one with:
     *   - the lower weighted error first
     *   - the wider margin
     *
     *
     * Pair<Integer i, Boolean b> indicates whether feature i is a face (b=true) or not (b=false)
     */
    public static DecisionStump bestStump(DenseMatrix labels, DenseMatrix weights, long featureCount, int N, double totalWeightPos, double totalWeightNeg, double minWeight) {

        // Compare each DecisionStump and find the best by following this algorithm:
        //   if (current.weightedError < best.weightedError) -> best = current
        //   else if (current.weightedError == best.weightedError && current.margin > best.margin) -> best = current

        DecisionStump best = compute(labels, weights, 0, N, totalWeightPos, totalWeightNeg, minWeight);
        for (long i = 1; i < featureCount; i++) {
            DecisionStump current = compute(labels, weights, i, N, totalWeightPos, totalWeightNeg, minWeight);
            if (compare(current, best))
                best = current;
        }

        if (best.error >= 0.5) {
            System.out.println("Failed best stump, error : " + best.error + " >= 0.5 !");
            exit();
        }

        return best;
    }
}

