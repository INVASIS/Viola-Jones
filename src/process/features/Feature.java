package process.features;

import static process.features.FeatureExtractor.*;

/**
 * Created by Dubrzr on 13/07/2016.
 */
public class Feature {
    private final Rectangle r;
    private final int type;
    private int computeValue = -1;
    private final int[][] integralImage;
    private final int threshold;
    private final int polarity;

    public Feature(Rectangle r, int type, int[][] integralImage, int threshold, int polarity) {
        this.r = r;
        this.type = type;
        this.integralImage = integralImage;
        this.threshold = threshold;
        this.polarity = polarity;
    }

    public int getValue() {
        if (this.computeValue == -1) {
            if (this.type == typeA)
                this.computeValue = computeTypeA(this.integralImage, this.r);
            else if (this.type == typeB)
                this.computeValue = computeTypeB(this.integralImage, this.r);
            else if (this.type == typeC)
                this.computeValue = computeTypeC(this.integralImage, this.r);
            else if (this.type == typeD)
                this.computeValue = computeTypeD(this.integralImage, this.r);
            else if (this.type == typeE)
                this.computeValue = computeTypeE(this.integralImage, this.r);
        }
        return this.computeValue;
    }

    public Rectangle getRectangle() {
        return r;
    }

    public int getType() {
        return type;
    }

    public int getPolarity() {
        return polarity;
    }

    public int getThreshold() {
        return threshold;
    }
}
