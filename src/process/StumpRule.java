package process;

import jeigen.DenseMatrix;

import java.math.BigDecimal;
import java.util.ArrayList;

public class StumpRule { // == stumpRule

    // Values that will be used to find the best StumpRule
    public long featureIndex;
    public BigDecimal error;
    public double threshold;
    public double margin;
    public int toggle; // = polarity {-1; 1}


    // Initialisation
    public StumpRule(long featureIndex, BigDecimal error, double threshold, double margin, int toggle) {
        this.featureIndex = featureIndex;
        this.error = error;
        this.threshold = threshold;
        this.margin = margin;
        this.toggle = toggle;
    }

    public StumpRule(long featureIndex, double error, double threshold, double margin, int toggle) {
        this.featureIndex = featureIndex;
        this.error = new BigDecimal(error);
        this.threshold = threshold;
        this.margin = margin;
        this.toggle = toggle;
    }

    public static boolean compare(StumpRule first, StumpRule second) {
        return (first.error.compareTo(second.error) == -1 || // <=> first.error < second.error
                (first.error.compareTo(second.error) == 0 && first.margin > second.margin)); // <=> first.error == second.error
    }
}

