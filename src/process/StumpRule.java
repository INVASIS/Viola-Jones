package process;


import org.w3c.dom.Document;
import org.w3c.dom.Element;

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

    public Element toXML(Document document) {
        Element stump = document.createElement("StumpRule");

        Element fi = document.createElement("FeatureIndex");
        fi.appendChild(document.createTextNode(String.valueOf(this.featureIndex)));
        stump.appendChild(fi);

        Element er = document.createElement("Error");
        er.appendChild(document.createTextNode(String.valueOf(this.error)));
        stump.appendChild(er);

        Element th = document.createElement("Threshold");
        th.appendChild(document.createTextNode(String.valueOf(this.threshold)));
        stump.appendChild(th);

        Element ma = document.createElement("Margin");
        ma.appendChild(document.createTextNode(String.valueOf(this.margin)));
        stump.appendChild(ma);

        Element to = document.createElement("Toggle");
        to.appendChild(document.createTextNode(String.valueOf(this.toggle)));
        stump.appendChild(to);

        return stump;
    }

    public static StumpRule fromXML(Element stump) {

        final long fi = Long.valueOf(stump.getElementsByTagName("FeatureIndex").item(0).getTextContent());
        final double er = Double.valueOf(stump.getElementsByTagName("Error").item(0).getTextContent());
        final double th = Double.valueOf(stump.getElementsByTagName("Threshold").item(0).getTextContent());
        final double ma = Double.valueOf(stump.getElementsByTagName("Margin").item(0).getTextContent());
        final int to = Integer.valueOf(stump.getElementsByTagName("Toggle").item(0).getTextContent());

        return new StumpRule(fi, er, th, ma, to);
    }
}