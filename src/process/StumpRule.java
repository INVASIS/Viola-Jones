package process;

import utils.DoubleDouble;

public class StumpRule {

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

    public boolean compare(StumpRule other) {
        return (this.error.lt(other.error) || (this.error.eq(other.error) && this.margin > other.margin));
    }
}

