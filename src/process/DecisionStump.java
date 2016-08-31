package process;

import jeigen.DenseMatrix;

import java.util.ArrayList;

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

    public static boolean compare(DecisionStump first, DecisionStump second) {
        return (first.error < second.error ||
                (first.error == second.error && first.margin > second.margin));
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
    public static DecisionStump bestStump(DenseMatrix labels, DenseMatrix weights, long featureCount, int N, double totalWeightPos, double totalWeightNeg, double minWeight) {

        // Compare each DecisionStump and find the best by following this algorithm:
        //   if (current.weightedError < best.weightedError) -> best = current
        //   else if (current.weightedError == best.weightedError && current.margin > best.margin) -> best = current

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
            i += j;
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

        if (best.error >= 0.5) {
            System.out.println("Failed best stump, error : " + best.error + " >= 0.5 !");
            System.exit(1);
        }

        return best;
    }
}

