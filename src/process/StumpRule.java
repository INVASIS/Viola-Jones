package process;

public class StumpRule {

    // Values that will be used to find the best StumpRule
    public long featureIndex;
    public double error;
    public double threshold;
    public double margin;
    public int toggle; // = polarity {-1; 1}


    // Initialisation
    public StumpRule(long featureIndex, double error, double threshold, double margin, int toggle) {
        this.featureIndex = featureIndex;
        this.error = error;
        this.threshold = threshold;
        this.margin = margin;
        this.toggle = toggle;
    }


    public boolean compare(StumpRule other) {
        return (this.error < other.error || (this.error == other.error && this.margin > other.margin));
    }
}

