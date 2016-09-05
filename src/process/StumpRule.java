package process;


public class StumpRule { // == Weak classifier
    //    (Your feature value > this.threshold?)
    //         | NO                   YES |
    //    (not a face)                 (a face)

    // Values that will be used to find the best StumpRule
    public long featureIndex;
    public double error;
    public double threshold;
    public double margin; // = The margin between the feature values of the two examples
    public int toggle; // = polarity {-1; 1} = if (Your feature value > this.threshold ? toggle : -toggle)


    // Initialisation
    public StumpRule(long featureIndex, double error, double threshold, double margin, int toggle) {
        this.featureIndex = featureIndex;
        this.error = error;
        this.threshold = threshold;
        this.margin = margin;
        this.toggle = toggle;
    }

    public boolean compare(double otherError, double otherMargin) {
        // We want the threshold which best separate positive & negative examples, this is equivalent to find the
        // StumpRule with the lowest error. When error is the same, a larger margin means a larger separation between
        // positive and negative examples, so it's a better choice.
        return (this.error < otherError || (this.error == otherError && this.margin > otherMargin));
    }

    public boolean compare(StumpRule other) {
        return this.compare(other.error, other.margin);
    }

    public static StumpRule deepCopy(StumpRule other) {
        return new StumpRule(
                other.featureIndex,
                other.error,
                other.threshold,
                other.margin,
                other.toggle
        );
    }
}