package process;

import jeigen.DenseMatrix;

import java.math.BigDecimal;
import java.util.ArrayList;

public class DecisionStump { // == stumpRule

    // Values that will be used to find the best DecisionStump
    public long featureIndex;
    public BigDecimal error;
    public double threshold;
    public double margin;
    public int toggle; // = polarity {-1; 1}


    // Initialisation
    public DecisionStump(long featureIndex, BigDecimal error, double threshold, double margin, int toggle) {
        this.featureIndex = featureIndex;
        this.error = error;
        this.threshold = threshold;
        this.margin = margin;
        this.toggle = toggle;
    }

    public static boolean compare(DecisionStump first, DecisionStump second) {
        return (first.error.compareTo(second.error) == -1 || // <=> first.error < second.error
                (first.error.compareTo(second.error) == 0 && first.margin > second.margin)); // <=> first.error == second.error
    }

    /**
     * Algorithm 5 from the original paper
     * <p>
     * Return the most discriminative feature and its rule
     * We compute each DecisionStump, and find the one with:
     * - the lower weighted error first
     * - the wider margin
     * <p>
     * <p>
     * Pair<Integer i, Boolean b> indicates whether feature i is a face (b=true) or not (b=false)
     */
    public static DecisionStump bestStump(DenseMatrix labels, ArrayList<BigDecimal> weights, long featureCount, int N, BigDecimal totalWeightPos, BigDecimal totalWeightNeg, BigDecimal minWeight) {

        // Compare each DecisionStump and find the best by following this algorithm:
        //   if (current.weightedError < best.weightedError) -> best = current
        //   else if (current.weightedError == best.weightedError && current.margin > best.margin) -> best = current

        System.out.println("[BestStump] Calling bestStump with : ");
        System.out.println("[BestStump] featureCount : " + featureCount + " N : " + N + " totalWeightsPos : " + totalWeightPos + " totalWeightNeg : " + totalWeightNeg + " minWeight : " + minWeight);
        int nb_threads = Runtime.getRuntime().availableProcessors();
        ThreadManager managerFor0 = new ThreadManager(labels, weights, 0, N, totalWeightPos, totalWeightNeg, minWeight);
        managerFor0.run();
        DecisionStump best = managerFor0.getBest();
        for (long i = 1; i < featureCount; i++) {

            ArrayList<ThreadManager> listThreads = new ArrayList<>(nb_threads);
            long j = 0;
            for (j = 0; j < nb_threads && j + i < featureCount; j++) {
                ThreadManager threadManager = new ThreadManager(labels, weights, i + j, N, totalWeightPos, totalWeightNeg, minWeight);
                listThreads.add(threadManager);
                threadManager.start();
            }
            i += (j - 1);
            for (int k = 0; k < j; k++) {
                try {
                    listThreads.get(k).join();
                } catch (InterruptedException e) {
                    System.err.println("Error in thread while computing bestStump - i = " + i + " k = " + k + " j = " + j);
                    e.printStackTrace();
                }
            }

            for (int k = 0; k < j; k++) {
                if (compare(listThreads.get(k).getBest(), best))
                    best = listThreads.get(k).getBest();
            }
        }

        System.out.println("[BestStump] BestStump : ");
        System.out.println("[BestStump] FeatureIndex : " + best.featureIndex + " error : " + best.error + " Threshold : "
                + best.threshold + " margin : " + best.margin + " toggle : " + best.toggle);
        if (best.error.compareTo(new BigDecimal(0.5)) >= 0) { // if (best.error >= 0.5)
            System.out.println("Failed best stump, error : " + best.error + " >= 0.5 !");
            System.exit(1);
        }

        System.out.println("      - Found best stump: (featureIdx: " + best.featureIndex + ", threshold: " + best.threshold + ", margin:" + best.margin + ", toggle:" + best.toggle + ", error:" + best.error + ")");

        return best;
    }
}

