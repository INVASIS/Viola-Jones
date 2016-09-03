package process;

import utils.DoubleDouble;

public class StumpRule { // == stumpRule

    // Values that will be used to find the best StumpRule
    public long featureIndex;
    public DoubleDouble error;
    public double threshold;
    public double margin;
    public int toggle; // = polarity {-1; 1}


    // Initialisation
    public StumpRule(long featureIndex, DoubleDouble error, double threshold, double margin, int toggle) {
        this.featureIndex = featureIndex;
        this.error = error;
        this.threshold = threshold;
        this.margin = margin;
        this.toggle = toggle;
    }

    public StumpRule(long featureIndex, double error, double threshold, double margin, int toggle) {
        this.featureIndex = featureIndex;
        this.error = new DoubleDouble(error);
        this.threshold = threshold;
        this.margin = margin;
        this.toggle = toggle;
    }

    public static boolean compare(StumpRule first, StumpRule second) {
        return (first.error.lt(second.error) || (first.error.eq(second.error) && first.margin > second.margin));
    }
}

